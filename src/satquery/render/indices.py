"""Spectral indices and SAR backscatter maths (DATA_ADAPTATION_PLAN §2.1, §2.3).

Every index is a normalised difference in ``[-1, +1]`` and every view built from
one is rendered on that **fixed** domain. Per-image normalisation is forbidden
here: it would make NDVI 0.7 in a dull scene render like NDVI 0.2 in a vivid one,
and the language model could never learn absolute physical semantics.

SAR clips are fixed for the same reason.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float32]

EPS: Final[float] = 1e-6
"""Floor for the log in the dB conversion (DATA_ADAPTATION_PLAN §2.3)."""

INDEX_DOMAIN: Final[tuple[float, float]] = (-1.0, 1.0)
"""The fixed, image-independent domain every normalised-difference index renders on."""

VV_DB_CLIP: Final[tuple[float, float]] = (-25.0, 0.0)
VH_DB_CLIP: Final[tuple[float, float]] = (-30.0, -5.0)
RATIO_DB_CLIP: Final[tuple[float, float]] = (0.0, 15.0)
SARDB_DOMAIN: Final[tuple[float, float]] = (-25.0, 0.0)
"""Fixed SAR domains. SARDB uses the VV window; SARFC clips each channel separately."""

_DENOMINATOR_FLOOR: Final[float] = 1e-9


def normalised_difference(a: Array, b: Array) -> Array:
    """Return ``(a - b) / (a + b)``, NaN where the denominator vanishes.

    A vanishing denominator means both bands read zero — a nodata or shadow
    pixel — where the index is undefined. Returning NaN keeps it out of every
    statistic and renders it black, rather than inventing a 0.0 that would read
    as "bare soil".
    """
    numerator = a.astype(np.float32) - b.astype(np.float32)
    denominator = a.astype(np.float32) + b.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(
            np.abs(denominator) < _DENOMINATOR_FLOOR,
            np.float32(np.nan),
            numerator / denominator,
        )
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def ndvi(nir: Array, red: Array) -> Array:
    """Normalised Difference Vegetation Index: ``(nir - red) / (nir + red)``."""
    return normalised_difference(nir, red)


def ndwi(green: Array, nir: Array) -> Array:
    """Normalised Difference Water Index: ``(green - nir) / (green + nir)``."""
    return normalised_difference(green, nir)


def ndbi(swir1: Array, nir: Array) -> Array:
    """Normalised Difference Built-up Index: ``(swir1 - nir) / (swir1 + nir)``."""
    return normalised_difference(swir1, nir)


def nbr(nir: Array, swir2: Array) -> Array:
    """Normalised Burn Ratio: ``(nir - swir2) / (nir + swir2)``."""
    return normalised_difference(nir, swir2)


def looks_like_db(values: Array) -> bool:
    """True when a backscatter band is already expressed in decibels.

    Linear-power sigma0 is non-negative; a dB band is mostly negative. Converting
    a dB band a second time would produce nonsense, so this guard runs before
    every conversion.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(np.nanmin(finite) < 0.0)


def to_db(values: Array, already_db: bool | None = None) -> Array:
    """Convert linear-power sigma0 to decibels: ``10 * log10(max(DN, eps))``.

    Args:
        values: Backscatter, linear power or already in dB.
        already_db: Skip the conversion. Auto-detected when omitted.

    Returns:
        Backscatter in dB, with NaN preserved.
    """
    array = values.astype(np.float32)
    if already_db if already_db is not None else looks_like_db(array):
        return array
    with np.errstate(divide="ignore", invalid="ignore"):
        return (10.0 * np.log10(np.maximum(array, EPS))).astype(np.float32)


def vv_vh_ratio_db(vv_db: Array, vh_db: Array) -> Array:
    """The VV/VH ratio in dB, which is a difference in the log domain."""
    return (vv_db - vh_db).astype(np.float32)


def valid_fraction(values: Array) -> float:
    """Fraction of finite pixels — used to decide whether a statistic is meaningful."""
    if values.size == 0:
        return 0.0
    return float(np.isfinite(values).mean())


def index_statistics(values: Array, prefix: str) -> dict[str, float]:
    """Summarise an index over its valid pixels only.

    Nodata is excluded from every statistic (DATA_ADAPTATION_PLAN §2.3), so an
    empty scene yields an empty dict rather than a misleading 0.0.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_std": float(finite.std()),
        f"{prefix}_min": float(finite.min()),
        f"{prefix}_max": float(finite.max()),
        f"{prefix}_valid_pct": round(100.0 * finite.size / values.size, 4),
    }
