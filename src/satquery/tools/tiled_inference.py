"""Tiled inference with overlap blending, for rasters that will not fit at once.

A 2048x2048 bi-temporal pair through a Siamese ResNet at bf16 is a few gigabytes
of activations, and a Cartosat-2S scene is far larger than that. Tiling is not an
optimisation here; it is what makes the tool work at all on a card that also has
to hold an 8B VLM (Master.md §8 Phase 5).

Naive tiling produces a mask with visible grid lines. The seams are real: a tile
boundary cuts the receptive field, so a pixel one column from the edge is
predicted from half the context a pixel in the tile's centre gets. Overlapping the
tiles and blending them with a taper that goes to (nearly) zero at the tile edge
makes every output pixel dominated by the tile that saw the most context around
it, and the artefact disappears.

Two properties are worth stating because the tests pin them:

* **Blending is a partition of unity.** Weights are accumulated alongside the
  predictions and divided out, so for any predictor that is itself tile-invariant
  the tiled result is *exactly* the whole-image result. Blending cannot introduce
  a bias of its own; it can only suppress edge effects that the model introduces.
* **Tapering is neighbour-aware.** A tile touching the raster border is not
  tapered on that side. There is no neighbour to blend with there, and tapering
  anyway would down-weight the outermost rows against nothing, leaving a faint
  frame around every scene.

Nothing here imports torch. The predictor is an ordinary callable, which is what
lets the tiling be tested for correctness and for memory at 2048x2048 on a machine
with no GPU and no checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float32]
Predictor = Callable[[Array, Array], Array]
"""Maps a co-registered ``(C, h, w)`` tile pair to an ``(h, w)`` field."""

DEFAULT_TILE_PX: Final[int] = 512
DEFAULT_OVERLAP_PX: Final[int] = 64
"""64 px at stride 32 is two receptive-field steps of the deepest ResNet stage —
enough for the blend region to cover the context a boundary pixel is missing,
without paying for a third of the scene twice."""

TAPER_FLOOR: Final[float] = 1e-3
"""The taper never reaches zero. A pixel covered by exactly one tile whose taper
was zero there would divide 0 by 0; the floor makes coverage sufficient on its
own and keeps the partition of unity exact."""


class TilingError(ValueError):
    """The tiling was configured in a way that cannot cover the raster."""


@dataclass(frozen=True)
class TileWindow:
    """One tile's placement on the raster, with its neighbour situation.

    The four ``touches_*`` flags are what make the taper neighbour-aware, and they
    describe the *raster* border rather than the tile's own edges.
    """

    row: int
    col: int
    height: int
    width: int
    touches_top: bool
    touches_left: bool
    touches_bottom: bool
    touches_right: bool

    @property
    def rows(self) -> slice:
        """Row slice this tile writes into."""
        return slice(self.row, self.row + self.height)

    @property
    def cols(self) -> slice:
        """Column slice this tile writes into."""
        return slice(self.col, self.col + self.width)


@dataclass(frozen=True)
class TilePlan:
    """How a raster is cut up, and how the pieces are put back together."""

    tile: int = DEFAULT_TILE_PX
    overlap: int = DEFAULT_OVERLAP_PX

    def __post_init__(self) -> None:
        """Reject a plan whose tiles cannot advance.

        Raises:
            TilingError: The tile is non-positive, or the overlap is at least the
                tile size — which would make the stride zero and never terminate.
        """
        if self.tile <= 0:
            raise TilingError(f"tile size must be positive, got {self.tile}")
        if self.overlap < 0:
            raise TilingError(f"overlap must not be negative, got {self.overlap}")
        if self.overlap >= self.tile:
            raise TilingError(
                f"overlap {self.overlap} must be smaller than the tile {self.tile}; "
                f"otherwise the window never advances"
            )

    @property
    def stride(self) -> int:
        """Distance between consecutive tile origins."""
        return self.tile - self.overlap

    def count(self, height: int, width: int) -> int:
        """How many tiles this plan produces for a raster of that size."""
        return _steps(height, self.tile, self.stride) * _steps(width, self.tile, self.stride)

    def windows(self, height: int, width: int) -> Iterator[TileWindow]:
        """Yield tiles covering the raster exactly, in row-major order.

        The final tile of each row and column is *shifted back* to end flush with
        the raster rather than padded. Padding the edge with zeros would feed the
        model a hard synthetic boundary, and a change detector reads a hard
        boundary as change.

        Raises:
            TilingError: The raster has no extent.
        """
        if height <= 0 or width <= 0:
            raise TilingError(f"cannot tile a {height}x{width} raster")

        for row in _origins(height, self.tile, self.stride):
            tile_h = min(self.tile, height)
            for col in _origins(width, self.tile, self.stride):
                tile_w = min(self.tile, width)
                yield TileWindow(
                    row=row,
                    col=col,
                    height=tile_h,
                    width=tile_w,
                    touches_top=row == 0,
                    touches_left=col == 0,
                    touches_bottom=row + tile_h >= height,
                    touches_right=col + tile_w >= width,
                )


def _steps(extent: int, tile: int, stride: int) -> int:
    """Number of tile origins needed to cover *extent*."""
    if extent <= tile:
        return 1
    return int(math.ceil((extent - tile) / stride)) + 1


def _origins(extent: int, tile: int, stride: int) -> list[int]:
    """Tile origins covering *extent*, with the last one flush to the far edge."""
    if extent <= tile:
        return [0]
    origins = list(range(0, extent - tile + 1, stride))
    if origins[-1] != extent - tile:
        origins.append(extent - tile)
    return origins


def _ramp(length: int, overlap: int, taper_start: bool, taper_end: bool) -> Array:
    """A 1-D taper: raised cosine over *overlap*, flat 1.0 through the middle."""
    weights = np.ones(length, dtype=np.float32)
    span = min(overlap, length // 2)
    if span <= 0:
        return weights
    # Half a cosine period, excluding the endpoints so the ramp starts above the
    # floor rather than at it.
    ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(span) + 0.5) / span)).astype(np.float32)
    if taper_start:
        weights[:span] = ramp
    if taper_end:
        weights[length - span :] = ramp[::-1]
    return weights


def taper(window: TileWindow, overlap: int) -> Array:
    """Blending weights for one tile, tapered only where a neighbour exists."""
    rows = _ramp(
        window.height, overlap, not window.touches_top, not window.touches_bottom
    )
    cols = _ramp(
        window.width, overlap, not window.touches_left, not window.touches_right
    )
    return np.maximum(np.outer(rows, cols), TAPER_FLOOR).astype(np.float32)


def predict_tiled(
    pre: Array,
    post: Array,
    predict: Predictor,
    plan: TilePlan | None = None,
) -> Array:
    """Run *predict* over a bi-temporal pair tile by tile and blend the results.

    Args:
        pre: ``(C, H, W)`` pre-change stack.
        post: ``(C, H, W)`` post-change stack, on the same grid.
        predict: The model, as a callable over one tile pair.
        plan: Tiling; defaults to 512 px tiles with 64 px overlap.

    Returns:
        An ``(H, W)`` float32 field on the input grid.

    Raises:
        TilingError: The two stacks are not on a common grid, or a predictor
            returned a tile of the wrong shape.
    """
    if pre.shape != post.shape:
        raise TilingError(
            f"the two epochs must share a grid, got {pre.shape} and {post.shape}"
        )
    if pre.ndim != 3:
        raise TilingError(f"expected a (C, H, W) stack, got {pre.shape}")

    _, height, width = pre.shape
    resolved = plan or TilePlan()

    # Two (H, W) float32 planes, whatever the raster size and however many tiles
    # it takes: no tile output is retained after it has been accumulated, which
    # is what keeps a 2048² pair — or a far larger one — bounded.
    accumulator = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    for window in resolved.windows(height, width):
        rows, cols = window.rows, window.cols
        predicted = predict(pre[:, rows, cols], post[:, rows, cols])
        if predicted.shape != (window.height, window.width):
            raise TilingError(
                f"predictor returned {predicted.shape} for a "
                f"{window.height}x{window.width} tile"
            )
        blend = taper(window, resolved.overlap)
        accumulator[rows, cols] += predicted.astype(np.float32) * blend
        weights[rows, cols] += blend

    # The floor guarantees every covered pixel has positive weight, and the
    # windows cover the raster exactly, so this division is always defined.
    return (accumulator / np.maximum(weights, TAPER_FLOOR)).astype(np.float32)


def peak_tile_bytes(plan: TilePlan, channels: int, bytes_per_element: int = 4) -> int:
    """Bytes one tile pair occupies — what the tiling actually bounds.

    Reported into the trace so a run that was tiled can say what it was tiled
    *to*, rather than leaving the reader to infer it from the parameters.
    """
    return 2 * channels * plan.tile * plan.tile * bytes_per_element


__all__ = [
    "DEFAULT_OVERLAP_PX",
    "DEFAULT_TILE_PX",
    "TAPER_FLOOR",
    "Predictor",
    "TilePlan",
    "TileWindow",
    "TilingError",
    "peak_tile_bytes",
    "predict_tiled",
    "taper",
]
