"""Windowed reads and view geometry (Master.md §8 Phase 2).

A 10 000 x 10 000 x 12-band uint16 raster is 2.2 GB if you read it whole. Views
are only 448 px on a side, so the whole raster never needs to be resident: this
module walks the *destination* grid in tiles and asks rasterio for each
corresponding source window at the output resolution. Peak memory is therefore
set by the tile size, not the file size.

It also owns view geometry — the aspect-preserving fit into a square canvas, the
zero padding, and the affine transform of the resulting grid, which the
georeferenced artifact companions depend on being exactly right.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window

VIEW_SIZE_PX: Final[int] = 448
"""Default edge of the square canvas a view is rendered onto."""

TILE_PX: Final[int] = 256
"""Destination-side tile edge. 256 keeps a 12-band float32 tile under 3 MB."""


@dataclass(frozen=True)
class PadExtent:
    """Zero padding added to fit a non-square scene into a square canvas."""

    top: int
    left: int
    bottom: int
    right: int

    @property
    def is_empty(self) -> bool:
        """True when no padding was needed."""
        return not (self.top or self.left or self.bottom or self.right)

    def as_dict(self) -> dict[str, int]:
        """Serialisable form, recorded on the artifact's stats."""
        return {"top": self.top, "left": self.left, "bottom": self.bottom, "right": self.right}


@dataclass(frozen=True)
class ViewGeometry:
    """The grid a view was rendered onto, and how it relates to the source raster."""

    width: int
    height: int
    content_width: int
    content_height: int
    pad: PadExtent
    scale: float
    transform: Affine | None
    crs: str | None

    @property
    def transform_list(self) -> list[float] | None:
        """The affine as the contract's 6-tuple, or None when not georeferenced."""
        if self.transform is None:
            return None
        a = self.transform
        return [a.a, a.b, a.c, a.d, a.e, a.f]


def fit_dimensions(width: int, height: int, size: int) -> tuple[int, int]:
    """Scale (width, height) so the longest edge is *size*, preserving aspect ratio."""
    longest = max(width, height)
    if longest == 0:
        raise ValueError("cannot fit a zero-sized raster")
    scale = size / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def iter_windows(width: int, height: int, tile: int = TILE_PX) -> Iterator[Window]:
    """Yield tiles covering a grid exactly once, in row-major order."""
    if tile <= 0:
        raise ValueError("tile size must be positive")
    for row in range(0, height, tile):
        rows = min(tile, height - row)
        for col in range(0, width, tile):
            cols = min(tile, width - col)
            yield Window(col_off=col, row_off=row, width=cols, height=rows)


def read_downsampled(
    dataset: Any,
    indexes: list[int],
    out_width: int,
    out_height: int,
    resampling: Resampling = Resampling.bilinear,
    tile: int = TILE_PX,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Read *indexes* at reduced resolution without materialising the full raster.

    The destination grid is walked in tiles; each tile pulls only the source
    window that maps onto it, decimated by GDAL during the read.

    Args:
        dataset: An open rasterio dataset.
        indexes: 1-indexed band numbers, in output order.
        out_width: Destination width in pixels.
        out_height: Destination height in pixels.
        resampling: Kernel for reflectance (bilinear) or masks (nearest).
        tile: Destination tile edge.

    Returns:
        ``(bands, valid)`` where bands is ``(n, out_height, out_width)`` float32
        with NaN at nodata, and valid is the ``(out_height, out_width)`` mask.
        A pixel is invalid when *any* requested band is nodata there, so a view
        never mixes real and missing data in one colour.
    """
    scale_x = dataset.width / out_width
    scale_y = dataset.height / out_height
    bands = np.full((len(indexes), out_height, out_width), np.nan, dtype=np.float32)

    for window in iter_windows(out_width, out_height, tile):
        row_off = int(window.row_off)
        col_off = int(window.col_off)
        rows = int(window.height)
        cols = int(window.width)

        src_row0 = int(math.floor(row_off * scale_y))
        src_col0 = int(math.floor(col_off * scale_x))
        src_row1 = min(dataset.height, int(math.ceil((row_off + rows) * scale_y)))
        src_col1 = min(dataset.width, int(math.ceil((col_off + cols) * scale_x)))
        if src_row1 <= src_row0 or src_col1 <= src_col0:
            continue

        source = Window(
            col_off=src_col0,
            row_off=src_row0,
            width=src_col1 - src_col0,
            height=src_row1 - src_row0,
        )
        chunk = dataset.read(
            indexes,
            window=source,
            out_shape=(len(indexes), rows, cols),
            resampling=resampling,
            masked=True,
        )
        bands[:, row_off : row_off + rows, col_off : col_off + cols] = np.ma.filled(
            chunk.astype(np.float32), np.nan
        )

    valid = np.all(np.isfinite(bands), axis=0)
    return bands, valid


def pad_to_square(
    bands: npt.NDArray[np.float32],
    valid: npt.NDArray[np.bool_],
    size: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], PadExtent]:
    """Centre a rendered scene on a square canvas, padding with nodata.

    Padding is invalid rather than zero-valued so it renders pure black and stays
    out of every statistic — a black bar must never be read as "reflectance 0".
    """
    _, height, width = bands.shape
    top = (size - height) // 2
    left = (size - width) // 2
    pad = PadExtent(top=top, left=left, bottom=size - height - top, right=size - width - left)
    if pad.is_empty:
        return bands, valid, pad

    canvas = np.full((bands.shape[0], size, size), np.nan, dtype=np.float32)
    canvas_valid = np.zeros((size, size), dtype=bool)
    canvas[:, top : top + height, left : left + width] = bands
    canvas_valid[top : top + height, left : left + width] = valid
    return canvas, canvas_valid, pad


def view_geometry(
    source_width: int,
    source_height: int,
    source_transform: Affine | None,
    crs: str | None,
    size: int = VIEW_SIZE_PX,
) -> ViewGeometry:
    """Compute the geometry of the view a source raster will be rendered onto.

    The returned transform describes the *padded canvas*, so a GeoTIFF companion
    written with it reopens at the right place on the ground.
    """
    content_width, content_height = fit_dimensions(source_width, source_height, size)
    top = (size - content_height) // 2
    left = (size - content_width) // 2
    pad = PadExtent(
        top=top, left=left, bottom=size - content_height - top, right=size - content_width - left
    )
    scale = source_width / content_width

    transform: Affine | None = None
    if source_transform is not None:
        # Scale to the view resolution, then step back over the padding so the
        # canvas origin, not the content origin, anchors the transform.
        scaled = source_transform * Affine.scale(
            source_width / content_width, source_height / content_height
        )
        transform = scaled * Affine.translation(-left, -top)

    return ViewGeometry(
        width=size,
        height=size,
        content_width=content_width,
        content_height=content_height,
        pad=pad,
        scale=scale,
        transform=transform,
        crs=crs,
    )


def render_source(
    path: Any,
    indexes: list[int],
    size: int = VIEW_SIZE_PX,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], ViewGeometry]:
    """Read the requested bands of a raster onto a padded square view canvas.

    This is the single entry point every renderer uses, so all views of one image
    share pixel-for-pixel geometry and can be overlaid on each other.
    """
    with rasterio.open(path) as dataset:
        crs = dataset.crs.to_string() if dataset.crs else None
        geometry = view_geometry(
            dataset.width, dataset.height, dataset.transform if dataset.crs else None, crs, size
        )
        bands, valid = read_downsampled(
            dataset,
            indexes,
            out_width=geometry.content_width,
            out_height=geometry.content_height,
            resampling=resampling,
        )
    padded, padded_valid, _ = pad_to_square(bands, valid, size)
    return padded, padded_valid, geometry
