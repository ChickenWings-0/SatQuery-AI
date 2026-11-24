"""Correct the imagery provenance a generated answer claims.

A large share of the fine-tuning captions (VRSBench) open with "The image,
sourced from GoogleEarth, shows a high-resolution view of …", and the adapter
learned the sentence along with the task. On a Sentinel-2 or Sentinel-1 scene
that sentence is a factual error about the input, stated before the answer
starts. The model cannot know where an image came from; ingestion does. So the
claim is rewritten from the manifest, never left to the model:

* "sourced from GoogleEarth" becomes the sensor ingestion identified, or is
  dropped when the sensor is unknown — no source is better than a wrong one.
* "high-resolution" is dropped when the measured ground sample distance is
  3 m or coarser: 10 m Sentinel-2 is medium resolution by any convention.

Only these phrases are touched; the rest of the answer, its numbers and its
citation markers pass through unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from satquery.schemas.manifest import InputManifest

HIGH_RES_MAX_GSD_M: Final[float] = 3.0

_GOOGLE: Final[str] = r"google\s?earth"
_VERB: Final[str] = r"(?:sourced|taken|captured|obtained|acquired|derived)"

_CLAUSE: Final[re.Pattern[str]] = re.compile(rf",\s*{_VERB}\s+from\s+{_GOOGLE}\s*,", re.IGNORECASE)
_FROM: Final[re.Pattern[str]] = re.compile(rf"\s+(?:{_VERB}\s+)?from\s+{_GOOGLE}\b", re.IGNORECASE)
_BARE: Final[re.Pattern[str]] = re.compile(rf"\b{_GOOGLE}\b", re.IGNORECASE)
_HIGH_RES: Final[re.Pattern[str]] = re.compile(
    r"\b(?:very\s+)?high[- ]resolution\s+", re.IGNORECASE
)


def short_sensor(manifest: InputManifest | None) -> str | None:
    """``"Sentinel-2 (BigEarthNet-v2, 12-band)"`` → ``"Sentinel-2"``; None when unknown."""
    if manifest is None or not manifest.sensor_guess:
        return None
    name = manifest.sensor_guess.split(" (", 1)[0].strip()
    # "Generic RGB" is ingestion saying it does not know the platform; naming it
    # as a source would be as wrong as the claim it replaces.
    if not name or name.lower().startswith(("generic", "unknown")):
        return None
    return name


def correct_provenance(text: str, manifests: Sequence[InputManifest]) -> str:
    """Rewrite source and resolution claims in *text* to match *manifests*."""
    if not text:
        return text
    first = manifests[0] if manifests else None
    sensor = short_sensor(first)
    out = text

    if re.search(_GOOGLE, out, re.IGNORECASE):
        out = _CLAUSE.sub(f" from {sensor}" if sensor else "", out)
        out = _FROM.sub(f" from {sensor}" if sensor else "", out)
        out = _BARE.sub(sensor or "the satellite", out)

    gsd = first.gsd_m if first is not None else None
    if gsd is not None and gsd >= HIGH_RES_MAX_GSD_M:
        out = _HIGH_RES.sub("", out)

    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


__all__ = ["HIGH_RES_MAX_GSD_M", "correct_provenance", "short_sensor"]
