"""The view label builder — DATA_ADAPTATION_PLAN §2.5.

**This is the single source of truth for view labels.** Training corpus
construction and the runtime prompt builder both import it, because the whole
domain-adaptation recipe rests on the language model binding a named view to its
physical meaning. If corpus labels and runtime labels ever diverge, the
adaptation is silently destroyed while every test still passes. Duplicating any
string in this module elsewhere is a defect.

Grammar::

    Image {slot} ({modality} {view_name}{, temporal_role}{, sensor}{, date}{, scale_note})

Every optional component is omitted, comma and all, when it is not known.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

from satquery.schemas.enums import ImageRole, Modality

VIEW_NAMES: Final[dict[str, str]] = {
    "TC": "true colour",
    "FCIR": "false colour infrared",
    "SWIR": "SWIR composite",
    "NDVI": "NDVI heatmap",
    "NDWI": "NDWI heatmap",
    "NDBI": "NDBI heatmap",
    "SARFC": "false colour VV/VH/ratio",
    "SARDB": "backscatter",
    "PAN": "panchromatic",
    "CHANGE": "change overlay",
}
"""The name of each view as it appears inside a label."""

SCALE_NOTES: Final[dict[str, str]] = {
    "NDVI": "fixed scale -1 to +1",
    "NDWI": "fixed scale -1 to +1",
    "NDBI": "fixed scale -1 to +1",
    "SARFC": "fixed scale",
    "SARDB": "fixed scale -25 to 0 dB",
}
"""Views rendered on a fixed domain say so in their label: the note is how the
model learns that the colour is absolute rather than relative."""

MODALITY_WORDS: Final[dict[Modality, str]] = {
    Modality.OPTICAL: "optical",
    Modality.SAR: "SAR",
    Modality.PANCHROMATIC: "panchromatic",
    Modality.UNKNOWN: "image",
}

TEMPORAL_WORDS: Final[dict[ImageRole, str]] = {
    ImageRole.PRE: "pre-change",
    ImageRole.POST: "post-change",
}

_SENSOR_SHORT_NAMES: Final[dict[str, str | None]] = {
    "Sentinel-2 (BigEarthNet-v2, 12-band)": "Sentinel-2",
    "Sentinel-2 L2A": "Sentinel-2",
    "Sentinel-1 GRD": "Sentinel-1",
    "Cartosat-2S MX": "Cartosat-2S",
    "Cartosat-2S PAN": "Cartosat-2S",
    "RISAT-1": "RISAT-1",
    # A generic RGB file names no real sensor, so the label claims none.
    "Generic RGB": None,
}


def short_sensor_name(sensor: str | None) -> str | None:
    """Reduce a sensor fingerprint to the platform name a label should carry.

    ``"Sentinel-2 (BigEarthNet-v2, 12-band)"`` becomes ``"Sentinel-2"``: the
    label names the platform, not the product or the band count.
    """
    if not sensor:
        return None
    if sensor in _SENSOR_SHORT_NAMES:
        return _SENSOR_SHORT_NAMES[sensor]
    return sensor.split(" (")[0].strip() or None


def format_date(moment: datetime | None) -> str | None:
    """Render an acquisition time as the ISO date a label uses."""
    return moment.date().isoformat() if moment else None


def build_view_label(
    slot: int,
    modality: Modality | str,
    view_name: str,
    temporal_role: str | None = None,
    sensor: str | None = None,
    date: str | None = None,
    scale_note: str | None = None,
) -> str:
    """Assemble one view label from its components.

    Args:
        slot: 1-indexed position of this view in the prompt.
        modality: The imaging modality, or a word to use verbatim.
        view_name: The view's name, e.g. ``"true colour"``.
        temporal_role: ``"pre-change"`` / ``"post-change"``, when applicable.
        sensor: Short platform name, e.g. ``"Sentinel-2"``.
        date: ISO acquisition date.
        scale_note: Fixed-domain note for measurement views.

    Returns:
        The label, e.g. ``"Image 1 (optical true colour, Sentinel-2)"``.
    """
    if slot < 1:
        raise ValueError(f"slot is 1-indexed, got {slot}")
    if isinstance(modality, Modality):
        word = MODALITY_WORDS.get(modality, str(modality))
    else:
        word = modality
    parts = [f"{word} {view_name}".strip()]
    parts.extend(part for part in (temporal_role, sensor, date, scale_note) if part)
    return f"Image {slot} ({', '.join(parts)})"


def label_for_view(
    view_id: str,
    slot: int,
    modality: Modality,
    sensor: str | None = None,
    role: ImageRole | None = None,
    acquisition_time: datetime | None = None,
    include_date: bool = True,
) -> str:
    """Build the label for a catalogue view, filling every component it can.

    Args:
        view_id: A key of :data:`VIEW_NAMES`, e.g. ``"NDVI"``.
        slot: 1-indexed prompt position.
        modality: Modality of the source image.
        sensor: Sensor fingerprint; shortened for display.
        role: Temporal or cross-modal role of the source image.
        acquisition_time: Used only when the view carries a temporal role, where
            the date is what distinguishes the two otherwise identical views.
        include_date: Set False to suppress the date entirely.

    Returns:
        The label string.

    Raises:
        KeyError: The view id is not in the catalogue.
    """
    if view_id not in VIEW_NAMES:
        raise KeyError(f"unknown view {view_id!r}; available: {sorted(VIEW_NAMES)}")
    temporal = TEMPORAL_WORDS.get(role) if role else None
    date = format_date(acquisition_time) if (include_date and temporal) else None
    return build_view_label(
        slot=slot,
        modality=modality,
        view_name=VIEW_NAMES[view_id],
        temporal_role=temporal,
        sensor=short_sensor_name(sensor),
        date=date,
        scale_note=SCALE_NOTES.get(view_id),
    )
