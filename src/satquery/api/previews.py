"""Small quicklook previews of uploaded rasters, for the pre-flight file card.

A browser cannot decode a GeoTIFF, and a 12-band uint16 Sentinel-2 stack would
not look like anything if it could. The server already knows each image's
sensor and band layout after ingestion, so it renders the same composite the
evidence views use — true colour for optical, VV backscatter for SAR, the single
band otherwise — at thumbnail size, and returns it inline as a JPEG data URL.
Inline, because a preview is needed before any trace exists to address an
artifact under, and because it must pass the access code with no extra request.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence

import numpy as np

from satquery.ingest.bands import resolve_bands
from satquery.ingest.pipeline import IngestResult, SourceImage
from satquery.render.composites import panchromatic, sar_backscatter, true_colour
from satquery.render.indices import to_db
from satquery.render.tiling import render_source
from satquery.schemas.enums import Modality
from satquery.schemas.manifest import InputManifest

PREVIEW_PX = 256
JPEG_QUALITY = 80


def _rgb(manifest: InputManifest, path: str) -> np.ndarray | None:
    bands = resolve_bands(
        sensor=manifest.sensor_guess,
        band_count=manifest.band_count,
        band_names=manifest.band_names,
    )
    if manifest.modality is Modality.SAR and "vv" in bands:
        stack, valid, _ = render_source(path, [bands["vv"]], size=PREVIEW_PX)
        return sar_backscatter(to_db(stack[0]), valid)
    if all(name in bands for name in ("red", "green", "blue")):
        indexes = [bands["red"], bands["green"], bands["blue"]]
        stack, valid, _ = render_source(path, indexes, size=PREVIEW_PX)
        return true_colour(stack[0], stack[1], stack[2], valid)
    stack, valid, _ = render_source(path, [1], size=PREVIEW_PX)
    return panchromatic(stack[0], valid)


def _data_url(rgb: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(
        buffer, "JPEG", quality=JPEG_QUALITY
    )
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def previews_for(result: IngestResult, sources: Sequence[SourceImage]) -> dict[str, str]:
    """``img_* -> data URL`` for every input a quicklook could be rendered for.

    Best effort: a preview is a convenience, so an image it fails on is simply
    left out rather than failing the pre-flight it rides along with.
    """
    out: dict[str, str] = {}
    for manifest, source in zip(result.inputs, sources, strict=False):
        try:
            rgb = _rgb(manifest, str(source.path))
        except Exception:  # noqa: BLE001 - a preview never fails pre-flight
            continue
        if rgb is not None:
            out[manifest.id] = _data_url(rgb)
    return out


__all__ = ["PREVIEW_PX", "previews_for"]
