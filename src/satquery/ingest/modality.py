"""Modality and sensor classification from band count, dtype and dynamic range.

The fingerprints below are the detection half of the band alias table in
``DOCS/DATA_ADAPTATION_PLAN.md`` §2.2. They are deliberately ordered most-specific
first: a 2-band VV/VH stack is Sentinel-1 before it is "some 2-band raster".

Nothing here reads a file — it works entirely off :class:`~satquery.ingest.reader.RasterInfo`,
so it is trivially testable and never surprises the caller with I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from satquery.ingest.reader import BandStats, RasterInfo
from satquery.schemas.enums import Modality

_SAR_POLARISATIONS: Final[frozenset[str]] = frozenset({"vv", "vh", "hh", "hv"})

_DB_RANGE: Final[tuple[float, float]] = (-50.0, 10.0)
"""A backscatter band expressed in decibels lives roughly in this window."""


@dataclass(frozen=True)
class ModalityResult:
    """The classifier's verdict, with the evidence that produced it."""

    modality: Modality
    confidence: float
    sensor_guess: str | None
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SensorFingerprint:
    """One row of the sensor detection table."""

    sensor: str
    modality: Modality
    confidence: float
    matches: Callable[[RasterInfo], bool]
    reason: str


def normalised_band_names(info: RasterInfo) -> list[str]:
    """Return lowercase, stripped band names (empty list when the file declares none)."""
    return [name.strip().lower() for name in (info.band_names or [])]


def is_db_like(stats: BandStats) -> bool:
    """True when a band's distribution looks like backscatter in decibels.

    Decibel imagery is signed, sits inside roughly [-50, 10] and has a modest
    spread. Reflectance counts are unsigned and typically span thousands of DN.
    """
    low, high = _DB_RANGE
    return bool(stats.minimum < 0.0 and low <= stats.p2 and stats.p98 <= high)


def is_linear_power_like(stats: BandStats) -> bool:
    """True when a band looks like linear-power sigma0 (small, positive, skewed)."""
    return bool(stats.minimum >= 0.0 and stats.p98 <= 5.0 and stats.mean < 1.5 and stats.std > 0.0)


def _has_sar_polarisation_names(info: RasterInfo) -> bool:
    names = normalised_band_names(info)
    return bool(names) and all(name in _SAR_POLARISATIONS for name in names)


def _is_float(info: RasterInfo) -> bool:
    return info.dtype.startswith("float")


def _sar_signal(info: RasterInfo) -> bool:
    """True when the pixel values themselves look like radar rather than reflectance."""
    if not info.stats:
        return False
    return all(is_db_like(band) for band in info.stats) or (
        _is_float(info) and all(is_linear_power_like(band) for band in info.stats)
    )


FINGERPRINTS: Final[tuple[SensorFingerprint, ...]] = (
    SensorFingerprint(
        sensor="Sentinel-1 GRD",
        modality=Modality.SAR,
        confidence=0.97,
        matches=lambda i: i.band_count == 2 and set(normalised_band_names(i)) == {"vv", "vh"},
        reason="two bands named VV and VH",
    ),
    SensorFingerprint(
        sensor="Sentinel-2 (BigEarthNet-v2, 12-band)",
        modality=Modality.OPTICAL,
        confidence=0.96,
        matches=lambda i: i.band_count == 12 and i.dtype == "uint16",
        reason="12 uint16 bands",
    ),
    SensorFingerprint(
        sensor="Sentinel-2 L2A",
        modality=Modality.OPTICAL,
        confidence=0.96,
        matches=lambda i: i.band_count == 13 and i.dtype == "uint16",
        reason="13 uint16 bands",
    ),
    SensorFingerprint(
        sensor="Cartosat-2S MX",
        modality=Modality.OPTICAL,
        confidence=0.85,
        matches=lambda i: (
            i.band_count == 4 and i.gsd_m is not None and i.gsd_m < 3.0 and not _sar_signal(i)
        ),
        reason="4 bands at sub-3 m ground sample distance",
    ),
    SensorFingerprint(
        sensor="RISAT-1",
        modality=Modality.SAR,
        confidence=0.82,
        matches=lambda i: i.band_count == 1 and _sar_signal(i),
        reason="a single band with a decibel-like distribution",
    ),
    SensorFingerprint(
        sensor="Cartosat-2S PAN",
        modality=Modality.PANCHROMATIC,
        confidence=0.85,
        matches=lambda i: (
            i.band_count == 1 and i.gsd_m is not None and i.gsd_m < 1.0 and not _sar_signal(i)
        ),
        reason="a single band at sub-metre ground sample distance",
    ),
    SensorFingerprint(
        sensor="Generic RGB",
        modality=Modality.OPTICAL,
        confidence=0.70,
        matches=lambda i: i.band_count == 3 and i.dtype == "uint8",
        reason="three uint8 bands",
    ),
)


def _fallback(info: RasterInfo) -> ModalityResult:
    """Classify a raster that matched no fingerprint, from its shape alone."""
    if _has_sar_polarisation_names(info) or _sar_signal(info):
        return ModalityResult(
            modality=Modality.SAR,
            confidence=0.65,
            sensor_guess=None,
            reasons=["band names or pixel distribution look like radar backscatter"],
        )
    if info.band_count >= 4:
        return ModalityResult(
            modality=Modality.OPTICAL,
            confidence=0.75,
            sensor_guess=None,
            reasons=[f"{info.band_count} bands is characteristic of a multispectral sensor"],
        )
    if info.band_count == 3:
        return ModalityResult(
            modality=Modality.OPTICAL,
            confidence=0.70,
            sensor_guess=None,
            reasons=["three bands read as a colour composite"],
        )
    if info.band_count == 1:
        return ModalityResult(
            modality=Modality.PANCHROMATIC,
            confidence=0.55,
            sensor_guess=None,
            reasons=["a single non-radar band reads as panchromatic"],
        )
    return ModalityResult(
        modality=Modality.UNKNOWN,
        confidence=0.30,
        sensor_guess=None,
        reasons=[f"no rule matched {info.band_count} bands of {info.dtype}"],
    )


def classify(info: RasterInfo) -> ModalityResult:
    """Classify the modality of *info* and guess the sensor that produced it.

    Fingerprints are tried in declaration order (most specific first); anything
    unmatched falls through to shape-based heuristics with a lower confidence.
    """
    for fingerprint in FINGERPRINTS:
        if fingerprint.matches(info):
            return ModalityResult(
                modality=fingerprint.modality,
                confidence=fingerprint.confidence,
                sensor_guess=fingerprint.sensor,
                reasons=[f"matched {fingerprint.sensor}: {fingerprint.reason}"],
            )
    return _fallback(info)
