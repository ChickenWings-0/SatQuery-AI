"""What every change detector produces, so the two produce the same thing.

``image_diff_change`` and ``siamese_change_detector`` are interchangeable by
design — the second declares the first as its ``fallback``, and the executor may
substitute one for the other mid-plan. That only works if their *outputs* are
interchangeable too: the same artifact types, the same labels, the same basemap
under the same overlay, and the same :class:`MaskPayload` for
``change_statistics`` to measure.

Duplicating that in two files would let them drift, and the drift would be
invisible — a degraded run would silently render its overlay over a different
basemap than the nominal run, and the evidence gallery would show two different
pictures for what the trace calls the same measurement.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from satquery.render.composites import panchromatic, sar_backscatter, true_colour
from satquery.render.indices import to_db
from satquery.render.overlays import CHANGE_ALPHA, change_overlay
from satquery.schemas.enums import ArtifactType, Modality
from satquery.tools.base import (
    Array,
    ArtifactDraft,
    Bands,
    Draft,
    ImageBundle,
    Mask,
    MaskPayload,
    Rgb,
)

MASK_KEY: Final[str] = "change_mask"
OVERLAY_KEY: Final[str] = "change_overlay"

MASK_LABEL: Final[str] = "Change mask (binary)"
OVERLAY_LABEL: Final[str] = "Change overlaid on the post-change view"


def basemap_for(image: ImageBundle, stack: Bands) -> Rgb:
    """Render the post-change view a change overlay is composited onto.

    True colour whenever the image can produce it, because that is the view a
    human reads a change map against. SAR falls back to fixed-domain backscatter
    and anything else to a single-band grey — both still absolute, so the picture
    under the mask means the same thing scene to scene.
    """
    if image.modality is Modality.SAR and "vv" in stack.order:
        return sar_backscatter(to_db(stack.band("vv")), stack.valid)
    if image.has("red", "green", "blue") and {"red", "green", "blue"} <= set(stack.order):
        return true_colour(stack.band("red"), stack.band("green"), stack.band("blue"), stack.valid)
    first: Array = stack.stack[0]
    return panchromatic(first, stack.valid)


def mask_rgb(mask: Mask) -> Rgb:
    """The binary mask as a black-and-white image, for the CHANGE_MASK artifact."""
    channel = (mask.astype(np.uint8) * 255)[:, :, np.newaxis]
    return np.repeat(channel, 3, axis=2)


def change_drafts(
    mask: Mask,
    post: ImageBundle,
    post_stack: Bands,
    stats: dict[str, object],
) -> tuple[list[Draft], MaskPayload]:
    """Build the artifacts and the in-memory payload a change detector owes.

    Produces, in order:

    1. ``CHANGE_MASK`` — the binary mask as a PNG, with a georeferenced GeoTIFF
       companion written from the *rendered canvas* transform, so it reopens in
       QGIS exactly over the imagery it was computed from.
    2. ``OVERLAY_PNG`` — the ``CHANGE`` view of DATA_ADAPTATION_PLAN §2.1: the
       mask in red at 45 % alpha over the post-change true-colour basemap.

    Args:
        mask: The boolean change mask, on the canvas.
        post: The post-change image, for the basemap and provenance.
        post_stack: The bands the mask was computed over — the same grid.
        stats: Detector-specific facts recorded on the mask artifact.

    Returns:
        ``(drafts, payload)``. The payload is what ``change_statistics`` measures;
        a PNG URL is not pixels.
    """
    overlay = change_overlay(basemap_for(post, post_stack), mask)
    drafts: list[Draft] = [
        ArtifactDraft(
            key=MASK_KEY,
            type=ArtifactType.CHANGE_MASK,
            label=MASK_LABEL,
            image=mask_rgb(mask),
            raster=mask.astype(np.uint8),
            geometry=post_stack.geometry,
            stats=dict(stats),
        ),
        ArtifactDraft(
            key=OVERLAY_KEY,
            type=ArtifactType.OVERLAY_PNG,
            label=OVERLAY_LABEL,
            image=overlay,
            geometry=post_stack.geometry,
            stats={"alpha": CHANGE_ALPHA, "source_image": post.id},
        ),
    ]
    payload = MaskPayload(
        mask=mask,
        pixel_area_m2=post_stack.pixel_area_m2,
        geometry=post_stack.geometry,
    )
    return drafts, payload


__all__ = [
    "MASK_KEY",
    "MASK_LABEL",
    "OVERLAY_KEY",
    "OVERLAY_LABEL",
    "basemap_for",
    "change_drafts",
    "mask_rgb",
]
