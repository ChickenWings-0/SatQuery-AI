"""Geometry between two rasters: CRS reconciliation, overlap, GSD and pixel offset.

Everything here is pure geometry and numpy — no policy. The thresholds that turn
these measurements into PASS/WARN/FAIL live in :mod:`satquery.ingest.compatibility`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.warp import Resampling as RioResampling
from rasterio.warp import reproject, transform_bounds
from skimage.registration import phase_cross_correlation

from satquery.ingest.reader import RasterInfo
from satquery.schemas.enums import Resampling

Bounds = tuple[float, float, float, float]

UPSAMPLE_FACTOR: Final[int] = 20
"""Phase-correlation upsampling. 20 resolves shifts to 0.05 px, well inside the
+/-0.5 px recovery the plan requires."""

REGISTRATION_MAX_PX: Final[int] = 512
"""Longest edge of the grid the two images are resampled onto for correlation."""


@dataclass(frozen=True)
class GridSpec:
    """A concrete raster grid: CRS, affine, size and the kernel used to reach it."""

    crs: str
    transform: Affine
    width: int
    height: int
    gsd_m: float
    resampling: Resampling

    @property
    def transform_list(self) -> list[float]:
        """The affine as the contract's 6-tuple."""
        a = self.transform
        return [a.a, a.b, a.c, a.d, a.e, a.f]

    @property
    def bounds(self) -> Bounds:
        """(minx, miny, maxx, maxy) of the grid in its own CRS."""
        minx = self.transform.c
        maxy = self.transform.f
        maxx = minx + self.width * self.transform.a
        miny = maxy + self.height * self.transform.e
        return (min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))


@dataclass(frozen=True)
class OffsetEstimate:
    """Sub-pixel translation recovered by phase cross-correlation."""

    dy_px: float
    dx_px: float
    magnitude_px: float
    error: float
    reliable: bool
    detail: str


def bounds_iou(a: Bounds, b: Bounds) -> float:
    """Intersection-over-union of two axis-aligned boxes in the same CRS.

    Returns 0.0 for disjoint boxes and for degenerate (zero-area) input.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter_w, inter_h = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def intersection_bounds(a: Bounds, b: Bounds) -> Bounds | None:
    """Return the overlapping box, or None when the two are disjoint."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return (ix0, iy0, ix1, iy1)


def gsd_ratio(a: float | None, b: float | None) -> float | None:
    """Coarser GSD over finer, so the result is always >= 1.0. None if either is unknown."""
    if not a or not b or a <= 0 or b <= 0:
        return None
    return float(max(a, b) / min(a, b))


def size_ratio(a: RasterInfo, b: RasterInfo) -> float:
    """Ratio of the two images' longest edges, always >= 1.0."""
    longest_a = float(max(a.width, a.height))
    longest_b = float(max(b.width, b.height))
    return float(max(longest_a, longest_b) / max(1.0, min(longest_a, longest_b)))


def crs_equivalent(a: RasterInfo, b: RasterInfo) -> bool:
    """True when both images already share a coordinate reference system."""
    ca, cb = a.crs_obj, b.crs_obj
    if ca is None or cb is None:
        return False
    return bool(ca.equals(cb))


def reproject_bounds(bounds: Bounds, src: CRS, dst: CRS) -> Bounds:
    """Reproject a bounding box between CRSs, densifying the edges."""
    return tuple(  # type: ignore[return-value]
        float(v) for v in transform_bounds(src, dst, *bounds, densify_pts=21)
    )


def overlap_iou_wgs84(a: RasterInfo, b: RasterInfo) -> float | None:
    """IoU of the two footprints in WGS84, the CRS-neutral comparison space.

    Comparing in lon/lat means a CRS mismatch does not, by itself, look like a
    footprint mismatch — the two failures stay independently diagnosable.
    """
    if a.bounds_wgs84 is None or b.bounds_wgs84 is None:
        return None
    return bounds_iou(a.bounds_wgs84, b.bounds_wgs84)


def common_grid(
    reference: RasterInfo,
    other: RasterInfo,
    resampling: Resampling = Resampling.BILINEAR,
) -> GridSpec | None:
    """Compute the grid both images can be resampled onto.

    The reference image's CRS is kept (no third projection is invented), the
    coarser of the two GSDs is used so neither image is upsampled beyond its real
    resolution, and the extent is the intersection of the two footprints.

    Returns None when either image is not georeferenced or the footprints are
    disjoint.
    """
    if not reference.is_georeferenced or not other.is_georeferenced:
        return None
    ref_crs, other_crs = reference.crs_obj, other.crs_obj
    if ref_crs is None or other_crs is None or reference.bounds_native is None:
        return None
    if other.bounds_native is None:
        return None

    other_bounds = (
        other.bounds_native
        if ref_crs.equals(other_crs)
        else reproject_bounds(other.bounds_native, other_crs, ref_crs)
    )
    overlap = intersection_bounds(reference.bounds_native, other_bounds)
    if overlap is None:
        return None

    gsd = max(g for g in (reference.gsd_m, other.gsd_m) if g)
    # A geographic reference CRS stores degrees, so the grid must step in degrees.
    pixel = abs(reference.transform.a) if ref_crs.is_geographic else gsd

    minx, miny, maxx, maxy = overlap
    width = max(1, int(math.floor((maxx - minx) / pixel)))
    height = max(1, int(math.floor((maxy - miny) / pixel)))
    return GridSpec(
        crs=reference.crs or str(ref_crs.name),
        transform=Affine(pixel, 0.0, minx, 0.0, -pixel, maxy),
        width=width,
        height=height,
        gsd_m=float(gsd),
        resampling=resampling,
    )


def _decimate(grid: GridSpec, max_px: int) -> GridSpec:
    """Return *grid* scaled down so its longest edge is at most *max_px*."""
    longest = max(grid.width, grid.height)
    if longest <= max_px:
        return grid
    scale = longest / max_px
    return GridSpec(
        crs=grid.crs,
        transform=grid.transform * Affine.scale(scale, scale),
        width=max(1, int(grid.width / scale)),
        height=max(1, int(grid.height / scale)),
        gsd_m=grid.gsd_m * scale,
        resampling=grid.resampling,
    )


def resample_to_grid(info: RasterInfo, grid: GridSpec, band: int = 1) -> npt.NDArray[np.float32]:
    """Warp one band of *info* onto *grid*, returning a float32 array with NaN nodata."""
    destination = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    with rasterio.open(info.path) as dataset:
        source = dataset.read(band, masked=True).astype(np.float32).filled(np.nan)
        reproject(
            source=source,
            destination=destination,
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            dst_transform=grid.transform,
            dst_crs=CRS.from_user_input(grid.crs),
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=RioResampling[grid.resampling.value],
        )
    return destination


def _prepare_for_correlation(array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Fill nodata, remove the mean and apply a Hann window.

    Windowing suppresses the wrap-around edge energy that otherwise dominates the
    cross-power spectrum and pulls the estimate toward zero shift.
    """
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array)
    filled = np.where(finite, array, float(np.nanmean(array)))
    centred = filled - float(filled.mean())
    rows = np.hanning(centred.shape[0]).astype(np.float32)
    cols = np.hanning(centred.shape[1]).astype(np.float32)
    return (centred * np.outer(rows, cols)).astype(np.float32)


def estimate_offset(
    reference: RasterInfo,
    moving: RasterInfo,
    grid: GridSpec | None = None,
) -> OffsetEstimate | None:
    """Recover the translational offset of *moving* relative to *reference*.

    Both images are warped onto a shared grid first, so the measurement is a true
    residual mis-registration rather than a restatement of their differing
    geotransforms. The result is reported in pixels of that shared grid.

    Returns None when the pair cannot be placed on a common grid at all.
    """
    grid = grid or common_grid(reference, moving)
    if grid is None:
        return None
    work = _decimate(grid, REGISTRATION_MAX_PX)
    if work.width < 16 or work.height < 16:
        return None

    ref_array = resample_to_grid(reference, work)
    mov_array = resample_to_grid(moving, work)
    valid = np.isfinite(ref_array) & np.isfinite(mov_array)
    coverage = float(valid.mean())
    if coverage < 0.10:
        return OffsetEstimate(
            dy_px=0.0,
            dx_px=0.0,
            magnitude_px=0.0,
            error=1.0,
            reliable=False,
            detail="Too little valid overlap to estimate a registration offset.",
        )

    # scikit-image ships py.typed but leaves this function unannotated.
    shift, error, _phasediff = phase_cross_correlation(  # type: ignore[no-untyped-call]
        _prepare_for_correlation(ref_array),
        _prepare_for_correlation(mov_array),
        upsample_factor=UPSAMPLE_FACTOR,
        normalization=None,
    )
    dy, dx = (float(shift[0]), float(shift[1]))
    # Report in the caller's full-resolution grid, undoing any decimation.
    scale = grid.width / work.width
    dy, dx = dy * scale, dx * scale
    magnitude = math.hypot(dy, dx)
    error_value = float(error) if np.isfinite(error) else 1.0
    return OffsetEstimate(
        dy_px=round(dy, 4),
        dx_px=round(dx, 4),
        magnitude_px=round(magnitude, 4),
        error=round(error_value, 4),
        reliable=True,
        detail=(
            f"Phase correlation recovered a shift of {magnitude:.2f} px "
            f"(dx {dx:+.2f}, dy {dy:+.2f}) on a {work.gsd_m:.2f} m grid."
        ),
    )
