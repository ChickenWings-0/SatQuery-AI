"""Colormaps for index and backscatter views (DATA_ADAPTATION_PLAN §2.1).

The three diverging maps are the ColorBrewer 11-class ramps that matplotlib
publishes as ``RdYlGn``, ``BrBG`` and ``RdBu``. Their anchors are embedded here
and interpolated exactly the way ``LinearSegmentedColormap.from_list`` does —
piecewise-linear between equally spaced stops, sampled at ``linspace(0, 1, 256)``
— so the rendered colours match matplotlib without taking a plotting library as a
runtime dependency.

Colour is load-bearing here, not decoration: an index view is rendered on a fixed
value domain, so a given colour means the same physical value in every image the
model has ever seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

LUT_SIZE: Final[int] = 256

BLACK: Final[tuple[int, int, int]] = (0, 0, 0)
"""Nodata renders as pure black in every view (DATA_ADAPTATION_PLAN §2.3)."""

# ColorBrewer 11-class anchors, 0-255 per channel, in matplotlib's declaration order.
_RD_YL_GN: Final[tuple[tuple[int, int, int], ...]] = (
    (165, 0, 38),
    (215, 48, 39),
    (244, 109, 67),
    (253, 174, 97),
    (254, 224, 139),
    (255, 255, 191),
    (217, 239, 139),
    (166, 217, 106),
    (102, 189, 99),
    (26, 152, 80),
    (0, 104, 55),
)

_BR_BG: Final[tuple[tuple[int, int, int], ...]] = (
    (84, 48, 5),
    (140, 81, 10),
    (191, 129, 45),
    (223, 194, 125),
    (246, 232, 195),
    (245, 245, 245),
    (199, 234, 229),
    (128, 205, 193),
    (53, 151, 143),
    (1, 102, 94),
    (0, 60, 48),
)

_RD_BU: Final[tuple[tuple[int, int, int], ...]] = (
    (103, 0, 31),
    (178, 24, 43),
    (214, 96, 77),
    (244, 165, 130),
    (253, 219, 199),
    (247, 247, 247),
    (209, 229, 240),
    (146, 197, 222),
    (67, 147, 195),
    (33, 102, 172),
    (5, 48, 97),
)


def _ramp(anchors: tuple[tuple[int, int, int], ...]) -> npt.NDArray[np.uint8]:
    """Interpolate equally spaced anchors into a 256-entry lookup table."""
    stops = np.linspace(0.0, 1.0, len(anchors))
    samples = np.linspace(0.0, 1.0, LUT_SIZE)
    channels = [
        np.interp(samples, stops, [anchor[channel] for anchor in anchors]) for channel in range(3)
    ]
    return np.round(np.stack(channels, axis=1)).astype(np.uint8)


def _grayscale() -> npt.NDArray[np.uint8]:
    """Matplotlib's ``gray``: a linear black-to-white ramp."""
    ramp = np.round(np.linspace(0.0, 255.0, LUT_SIZE)).astype(np.uint8)
    return np.stack([ramp, ramp, ramp], axis=1)


@dataclass(frozen=True)
class Colormap:
    """A named 256-entry RGB lookup table."""

    name: str
    lut: npt.NDArray[np.uint8]

    def __post_init__(self) -> None:
        """Reject a table that is not 256 RGB entries."""
        if self.lut.shape != (LUT_SIZE, 3):
            raise ValueError(f"colormap {self.name} must be {LUT_SIZE}x3, got {self.lut.shape}")


COLORMAPS: Final[dict[str, Colormap]] = {
    "RdYlGn": Colormap("RdYlGn", _ramp(_RD_YL_GN)),
    "RdYlGn_r": Colormap("RdYlGn_r", _ramp(_RD_YL_GN[::-1])),
    "BrBG": Colormap("BrBG", _ramp(_BR_BG)),
    "BrBG_r": Colormap("BrBG_r", _ramp(_BR_BG[::-1])),
    "RdBu": Colormap("RdBu", _ramp(_RD_BU)),
    "RdBu_r": Colormap("RdBu_r", _ramp(_RD_BU[::-1])),
    "gray": Colormap("gray", _grayscale()),
}


def get_colormap(name: str) -> Colormap:
    """Look up a colormap by name.

    Raises:
        KeyError: The name is not one of the frozen view catalogue's colormaps.
    """
    if name not in COLORMAPS:
        raise KeyError(f"unknown colormap {name!r}; available: {sorted(COLORMAPS)}")
    return COLORMAPS[name]


def normalise_fixed(
    values: npt.NDArray[np.float32], vmin: float, vmax: float
) -> npt.NDArray[np.float32]:
    """Map a **fixed** value domain onto [0, 1], preserving NaN.

    This is the anti-per-image-stretch primitive. Clipping to a fixed domain is
    what makes "NDVI 0.72" render identically in a low-contrast scene and a
    high-contrast one.
    """
    if vmax <= vmin:
        raise ValueError(f"empty domain [{vmin}, {vmax}]")
    scaled = (values - vmin) / (vmax - vmin)
    return np.clip(scaled, 0.0, 1.0, out=scaled).astype(np.float32)


def apply_colormap(
    normalised: npt.NDArray[np.float32],
    colormap: str | Colormap,
    valid: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.uint8]:
    """Colour a [0, 1] field, rendering NaN and invalid pixels as pure black.

    Args:
        normalised: Values already mapped to [0, 1] (see :func:`normalise_fixed`).
        colormap: A colormap or its name.
        valid: Optional validity mask; False renders black.

    Returns:
        An ``(H, W, 3)`` uint8 RGB image.
    """
    cmap = get_colormap(colormap) if isinstance(colormap, str) else colormap
    finite = np.isfinite(normalised)
    # matplotlib indexes with int(x * N) clipped to N-1; match it exactly.
    indices = np.zeros(normalised.shape, dtype=np.uint16)
    np.multiply(np.nan_to_num(normalised, nan=0.0), LUT_SIZE, out=indices, casting="unsafe")
    rgb = cmap.lut[np.clip(indices, 0, LUT_SIZE - 1)]

    blank = ~finite if valid is None else (~finite | ~valid)
    rgb[blank] = BLACK
    return rgb.astype(np.uint8)
