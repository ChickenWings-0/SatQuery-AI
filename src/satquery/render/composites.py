"""Pixel-to-RGB renderers for every view in the catalogue (DATA_ADAPTATION_PLAN §2).

One rule governs this module: **reflectance composites stretch per image, index
and SAR views never do.** A composite is a picture, so a p2-p98 stretch over its
own valid pixels is the right way to make it legible. An index view is a
measurement, so it is mapped through a fixed domain (from :mod:`indices`) and a
fixed colormap (from :mod:`colormaps`), and a given colour means the same value
in every image the model has ever seen.

Nodata renders as pure black ``(0, 0, 0)`` in every view and is excluded from
every statistic.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from satquery.render.colormaps import BLACK, apply_colormap, normalise_fixed
from satquery.render.indices import (
    RATIO_DB_CLIP,
    SARDB_DOMAIN,
    VH_DB_CLIP,
    VV_DB_CLIP,
    vv_vh_ratio_db,
)

Array = npt.NDArray[np.float32]
Mask = npt.NDArray[np.bool_]
Rgb = npt.NDArray[np.uint8]

STRETCH_PERCENTILES: Final[tuple[float, float]] = (2.0, 98.0)
"""The frozen reflectance stretch window (DATA_ADAPTATION_PLAN §2.3)."""

_FLAT_BAND_VALUE: Final[int] = 128
"""A constant band has no contrast to stretch; render it mid-grey rather than
black, which would be indistinguishable from nodata."""


def percentile_stretch(
    band: Array,
    valid: Mask | None = None,
    percentiles: tuple[float, float] = STRETCH_PERCENTILES,
) -> Rgb:
    """Stretch one band to uint8 using percentiles of its **valid** pixels.

    Args:
        band: Single-band float32 values, NaN at nodata.
        valid: Optional validity mask; NaN is treated as invalid regardless.
        percentiles: Low and high cut, defaulting to the frozen p2-p98.

    Returns:
        A ``(H, W)`` uint8 band, 0 wherever the input is invalid.
    """
    low_p, high_p = percentiles
    finite = np.isfinite(band)
    usable = finite if valid is None else (finite & valid)
    out = np.zeros(band.shape, dtype=np.uint8)
    if not usable.any():
        return out

    low, high = (float(v) for v in np.percentile(band[usable], [low_p, high_p]))
    if high <= low:
        out[usable] = _FLAT_BAND_VALUE
        return out

    scaled = (band - low) / (high - low)
    np.clip(scaled, 0.0, 1.0, out=scaled)
    out[usable] = np.round(scaled[usable] * 255.0).astype(np.uint8)
    return out


def stretch_rgb(channels: list[Array], valid: Mask | None = None) -> Rgb:
    """Stretch three bands independently and stack them into an RGB image.

    Per-band stretching is what makes a true-colour composite look natural: a
    single shared window would let one bright band wash the other two out.
    """
    if len(channels) != 3:
        raise ValueError(f"an RGB composite needs exactly 3 channels, got {len(channels)}")
    stacked = np.stack([percentile_stretch(band, valid) for band in channels], axis=-1)
    if valid is not None:
        stacked[~valid] = BLACK
    return stacked


def _clip_to_uint8(values: Array, low: float, high: float, valid: Mask | None) -> Rgb:
    """Map a **fixed** clip window linearly onto 0-255, black where invalid."""
    normalised = normalise_fixed(values, low, high)
    out = np.zeros(values.shape, dtype=np.uint8)
    usable = np.isfinite(values) if valid is None else (np.isfinite(values) & valid)
    out[usable] = np.round(normalised[usable] * 255.0).astype(np.uint8)
    return out


# ------------------------------------------------------------- reflectance composites


def true_colour(red: Array, green: Array, blue: Array, valid: Mask | None = None) -> Rgb:
    """``TC`` — true colour, per-image p2-p98 stretch."""
    return stretch_rgb([red, green, blue], valid)


def false_colour_ir(nir: Array, red: Array, green: Array, valid: Mask | None = None) -> Rgb:
    """``FCIR`` — false-colour infrared: vegetation reads red."""
    return stretch_rgb([nir, red, green], valid)


def swir_composite(swir2: Array, nir: Array, red: Array, valid: Mask | None = None) -> Rgb:
    """``SWIR`` — short-wave infrared composite: burn scars and built-up separate."""
    return stretch_rgb([swir2, nir, red], valid)


def panchromatic(pan: Array, valid: Mask | None = None) -> Rgb:
    """``PAN`` — a single band, stretched and replicated across three channels."""
    stretched = percentile_stretch(pan, valid)
    rgb = np.stack([stretched] * 3, axis=-1)
    if valid is not None:
        rgb[~valid] = BLACK
    return rgb


# ------------------------------------------------------------------------ SAR views


def sar_false_colour(vv_db: Array, vh_db: Array, valid: Mask | None = None) -> Rgb:
    """``SARFC`` — VV, VH and their ratio on **fixed** dB clips.

    The clips (VV [-25, 0], VH [-30, -5], ratio [0, 15]) are image-independent,
    so a bright red pixel means the same backscatter everywhere.
    """
    ratio_db = vv_vh_ratio_db(vv_db, vh_db)
    channels = [
        _clip_to_uint8(vv_db, *VV_DB_CLIP, valid),
        _clip_to_uint8(vh_db, *VH_DB_CLIP, valid),
        _clip_to_uint8(ratio_db, *RATIO_DB_CLIP, valid),
    ]
    rgb = np.stack(channels, axis=-1)
    finite = np.isfinite(vv_db) & np.isfinite(vh_db)
    blank = ~finite if valid is None else (~finite | ~valid)
    rgb[blank] = BLACK
    return rgb


def sar_backscatter(vv_db: Array, valid: Mask | None = None) -> Rgb:
    """``SARDB`` — one polarisation on the **fixed** [-25, 0] dB domain, grey ramp."""
    return apply_colormap(normalise_fixed(vv_db, *SARDB_DOMAIN), "gray", valid)


# ---------------------------------------------------------------------- index views


def index_view(
    values: Array,
    colormap: str,
    domain: tuple[float, float] = (-1.0, 1.0),
    valid: Mask | None = None,
) -> Rgb:
    """Colour a normalised-difference index on its **fixed** domain.

    The domain defaults to [-1, +1] and must not be replaced by per-image
    percentiles: that substitution is the one change that would silently destroy
    the domain adaptation while every test still passed.
    """
    return apply_colormap(normalise_fixed(values, *domain), colormap, valid)
