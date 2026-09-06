"""Raster reading: open a file with rasterio and extract everything downstream needs.

This module is deliberately the only place that touches ``rasterio.open``. It
produces a :class:`RasterInfo`, a plain dataclass that the manifest builder, the
modality classifier and the co-registration checks all consume, so none of them
need to reopen the dataset or agree on rasterio's conventions.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.errors import RasterioError

from satquery.ingest.errors import InvalidRasterError, UnsupportedFormatError

ALLOWED_DRIVERS: Final[frozenset[str]] = frozenset({"GTiff", "COG", "PNG", "JPEG"})
"""Drivers we accept. Anything else is a 415 (API_CONTRACT §6)."""

STATS_MAX_PX: Final[int] = 512
"""Longest edge of the decimated read used for statistics and correlation."""

WGS84: Final[str] = "EPSG:4326"

_DATETIME_TAGS: Final[tuple[str, ...]] = (
    "TIFFTAG_DATETIME",
    "ACQUISITION_DATETIME",
    "ACQUISITION_DATE",
    "DATETIME",
    "DATE_ACQUIRED",
    "PRODUCT_SCENE_RASTER_START_TIME",
)

_DATETIME_FORMATS: Final[tuple[str, ...]] = (
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y%m%dT%H%M%S",
)

_METRES_PER_DEGREE: Final[float] = 111_320.0


@dataclass(frozen=True)
class BandStats:
    """Summary statistics of one band, computed over valid pixels of a decimated read."""

    index: int
    name: str | None
    minimum: float
    maximum: float
    mean: float
    std: float
    p2: float
    p98: float

    @property
    def dynamic_range(self) -> float:
        """Spread between the 2nd and 98th percentile — robust to outliers."""
        return self.p98 - self.p2


@dataclass
class RasterInfo:
    """Everything read off one raster file, before any interpretation.

    ``sample`` is a decimated float32 read (at most :data:`STATS_MAX_PX` on the
    long edge) shaped ``(bands, rows, cols)``, with nodata replaced by NaN. It
    backs both the modality heuristics and phase correlation, so no consumer
    needs to re-read the file.
    """

    path: Path
    filename: str
    size_bytes: int
    sha256: str
    driver: str
    crs: str | None
    transform: Affine
    bounds_native: tuple[float, float, float, float] | None
    bounds_wgs84: tuple[float, float, float, float] | None
    width: int
    height: int
    band_count: int
    dtype: str
    band_names: list[str] | None
    nodata: float | None
    gsd_m: float | None
    nodata_pct: float
    acquisition_time: datetime | None
    is_georeferenced: bool
    stats: list[BandStats]
    sample: npt.NDArray[np.float32]
    warnings: list[str] = field(default_factory=list)

    @property
    def transform_list(self) -> list[float] | None:
        """The affine as the contract's 6-tuple, or None when not georeferenced."""
        if not self.is_georeferenced:
            return None
        a = self.transform
        return [a.a, a.b, a.c, a.d, a.e, a.f]

    @property
    def crs_obj(self) -> CRS | None:
        """The pyproj CRS, or None for a non-georeferenced input."""
        return CRS.from_user_input(self.crs) if self.crs else None

    def structure_band(self) -> npt.NDArray[np.float32]:
        """Return a single 2-D image suitable for translational registration.

        Averaging the first few bands is more robust than picking one: it keeps
        edges (which drive phase correlation) while suppressing per-band noise.
        NaNs are replaced by the finite mean so the FFT stays well defined.
        """
        stack = self.sample[: min(3, self.band_count)]
        merged: npt.NDArray[np.float32] = np.nanmean(stack, axis=0).astype(np.float32)
        if np.all(np.isnan(merged)):
            return np.zeros_like(merged)
        filled: npt.NDArray[np.float32] = np.nan_to_num(merged, nan=float(np.nanmean(merged)))
        return filled


def sha256_of(path: Path) -> str:
    """Return the lowercase hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_acquisition_time(tags: dict[str, str]) -> datetime | None:
    """Pull an acquisition timestamp out of GDAL tags, if one is recognisable."""
    for key in _DATETIME_TAGS:
        raw = tags.get(key)
        if not raw:
            continue
        text = raw.strip().rstrip("Z")
        for fmt in _DATETIME_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
            except ValueError:
                continue
            return parsed.replace(tzinfo=UTC)
        try:
            return datetime.fromisoformat(text).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _band_names(dataset: Any) -> list[str] | None:
    """Read band descriptions, falling back to per-band GDAL tags."""
    descriptions = [d for d in dataset.descriptions if d]
    if len(descriptions) == dataset.count:
        return [str(d) for d in descriptions]

    from_tags: list[str] = []
    for index in range(1, dataset.count + 1):
        tags = dataset.tags(index)
        name = tags.get("BAND_NAME") or tags.get("band_name") or tags.get("POLARISATION")
        if not name:
            return None
        from_tags.append(str(name))
    return from_tags or None


def _decimated_shape(width: int, height: int) -> tuple[int, int]:
    """Return the (rows, cols) of a read decimated to at most STATS_MAX_PX per edge."""
    longest = max(width, height)
    if longest <= STATS_MAX_PX:
        return height, width
    scale = STATS_MAX_PX / longest
    return max(1, round(height * scale)), max(1, round(width * scale))


def _gsd_from_transform(transform: Affine, crs: CRS | None, lat: float | None) -> float | None:
    """Derive the ground sample distance in metres from the pixel size.

    A geographic CRS stores degrees, so the value is converted at the scene's
    centre latitude; the caller records a warning when that approximation is used.
    """
    pixel = (abs(transform.a) + abs(transform.e)) / 2.0
    if pixel <= 0:
        return None
    if crs is None:
        return None
    if crs.is_geographic:
        cos_lat = math.cos(math.radians(lat)) if lat is not None else 1.0
        return float(pixel * _METRES_PER_DEGREE * max(cos_lat, 0.01))
    factor = 1.0
    axis = crs.axis_info[0] if crs.axis_info else None
    if axis is not None and axis.unit_conversion_factor:
        factor = float(axis.unit_conversion_factor)
    return float(pixel * factor)


def _bounds_wgs84(
    bounds: tuple[float, float, float, float], crs: CRS
) -> tuple[float, float, float, float] | None:
    """Reproject native bounds to lon/lat, densifying the edges.

    ``always_xy=True`` normalises axis order, so the result is lon-first as the
    contract requires regardless of the source CRS's declared axis order.
    """
    transformer = Transformer.from_crs(crs, CRS.from_user_input(WGS84), always_xy=True)
    minx, miny, maxx, maxy = bounds
    steps = 8
    xs = [minx + (maxx - minx) * i / steps for i in range(steps + 1)]
    ys = [miny + (maxy - miny) * i / steps for i in range(steps + 1)]
    edge_x = xs + xs + [minx] * (steps + 1) + [maxx] * (steps + 1)
    edge_y = [miny] * (steps + 1) + [maxy] * (steps + 1) + ys + ys
    lons, lats = transformer.transform(edge_x, edge_y)
    finite = [(lon, lat) for lon, lat in zip(lons, lats, strict=True) if math.isfinite(lon + lat)]
    if not finite:
        return None
    lon_values = [lon for lon, _ in finite]
    lat_values = [lat for _, lat in finite]
    return (min(lon_values), min(lat_values), max(lon_values), max(lat_values))


def _band_stats(sample: npt.NDArray[np.float32], names: list[str] | None) -> list[BandStats]:
    """Compute robust per-band statistics over the valid (non-NaN) pixels."""
    stats: list[BandStats] = []
    for index in range(sample.shape[0]):
        band = sample[index]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            stats.append(
                BandStats(
                    index=index + 1,
                    name=names[index] if names else None,
                    minimum=0.0,
                    maximum=0.0,
                    mean=0.0,
                    std=0.0,
                    p2=0.0,
                    p98=0.0,
                )
            )
            continue
        p2, p98 = (float(v) for v in np.percentile(valid, [2, 98]))
        stats.append(
            BandStats(
                index=index + 1,
                name=names[index] if names else None,
                minimum=float(valid.min()),
                maximum=float(valid.max()),
                mean=float(valid.mean()),
                std=float(valid.std()),
                p2=p2,
                p98=p98,
            )
        )
    return stats


def open_raster(path: str | Path, ref: str | None = None) -> RasterInfo:
    """Open *path* and read everything the pipeline needs from it.

    Args:
        path: File to open. Must already be on local disk.
        ref: Image reference (``img_0``, …) used in error payloads.

    Returns:
        A fully populated :class:`RasterInfo`.

    Raises:
        InvalidRasterError: The file cannot be opened or carries no bands.
        UnsupportedFormatError: The driver is outside :data:`ALLOWED_DRIVERS`.
    """
    path = Path(path)
    warnings: list[str] = []

    try:
        dataset_ctx = rasterio.open(path)
    except (RasterioError, OSError, ValueError) as exc:
        raise InvalidRasterError(ref=ref, detail=_open_failure_message(exc)) from exc

    with dataset_ctx as dataset:
        driver = str(dataset.driver)
        if driver not in ALLOWED_DRIVERS:
            raise UnsupportedFormatError(driver=driver, ref=ref)
        if dataset.count == 0:
            raise InvalidRasterError(ref=ref, detail="The file contains no image bands.")

        width = int(dataset.width)
        height = int(dataset.height)
        band_count = int(dataset.count)
        dtype = str(dataset.dtypes[0])
        nodata = float(dataset.nodata) if dataset.nodata is not None else None
        crs_obj = CRS.from_user_input(dataset.crs.to_wkt()) if dataset.crs else None
        transform = dataset.transform

        is_georeferenced = crs_obj is not None and not _is_identity(transform)
        if crs_obj is not None and _is_identity(transform):
            is_georeferenced = False
            warnings.append("A CRS is present but the affine transform is the identity.")

        bounds_native: tuple[float, float, float, float] | None = None
        bounds_wgs84: tuple[float, float, float, float] | None = None
        if is_georeferenced and crs_obj is not None:
            b = dataset.bounds
            bounds_native = (float(b.left), float(b.bottom), float(b.right), float(b.top))
            bounds_wgs84 = _bounds_wgs84(bounds_native, crs_obj)
            if bounds_wgs84 is None:
                warnings.append("Native bounds could not be reprojected to WGS84.")

        rows, cols = _decimated_shape(width, height)
        raw = dataset.read(out_shape=(band_count, rows, cols), masked=True)
        sample = np.ma.filled(raw.astype(np.float32), np.nan)
        valid_mask = np.any(np.isfinite(sample), axis=0)

        nodata_pct = 100.0 * float(1.0 - valid_mask.mean())
        if nodata is None and nodata_pct == 0.0:
            zero_pixels = np.all(np.nan_to_num(sample, nan=0.0) == 0.0, axis=0)
            inferred = 100.0 * float(zero_pixels.mean())
            if inferred > 0.0:
                nodata_pct = inferred
                warnings.append(
                    "No nodata value is declared; all-zero pixels were counted as nodata."
                )

        names = _band_names(dataset)
        stats = _band_stats(sample, names)
        tags = {str(k): str(v) for k, v in dataset.tags().items()}
        acquisition_time = _parse_acquisition_time(tags)

        centre_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2.0 if bounds_wgs84 else None
        gsd_m = _gsd_from_transform(transform, crs_obj, centre_lat) if is_georeferenced else None
        if gsd_m is not None and crs_obj is not None and crs_obj.is_geographic:
            warnings.append(
                "Ground sample distance was approximated from a geographic CRS at the "
                "scene centre latitude."
            )

    if not is_georeferenced:
        warnings.append("The image is not georeferenced; geometric checks were skipped.")

    return RasterInfo(
        path=path,
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=sha256_of(path),
        driver=driver,
        crs=_crs_identifier(crs_obj) if is_georeferenced else None,
        transform=transform,
        bounds_native=bounds_native,
        bounds_wgs84=bounds_wgs84,
        width=width,
        height=height,
        band_count=band_count,
        dtype=dtype,
        band_names=names,
        nodata=nodata,
        gsd_m=gsd_m,
        nodata_pct=round(nodata_pct, 4),
        acquisition_time=acquisition_time,
        is_georeferenced=is_georeferenced,
        stats=stats,
        sample=sample,
        warnings=warnings,
    )


def _crs_identifier(crs: CRS | None) -> str | None:
    """Return ``EPSG:xxxxx`` when the CRS has an authority code, else its WKT-lite name."""
    if crs is None:
        return None
    code = crs.to_epsg()
    return f"EPSG:{code}" if code else str(crs.name)


def _is_identity(transform: Affine) -> bool:
    """True when the transform is GDAL's default placeholder for 'ungeoreferenced'."""
    return bool(transform == Affine.identity())


def _open_failure_message(exc: Exception) -> str:
    """Turn a rasterio exception into a user-safe sentence, never leaking a path."""
    kind = type(exc).__name__
    if "NotGeoreferenced" in kind:
        return "The file opened but carries no usable image data."
    return "The file could not be opened as a raster image."
