"""``image_diff_change`` — the deterministic change baseline (CVA / log-ratio).

This is the declared fallback for ``siamese_change_detector`` and, until Phase 5
trains that checkpoint, the tool that actually answers every change question.
It has no weights, which is exactly why it can be trusted as a floor: if the
learned detector is unavailable or fails, the system still measures change, it
just says so by recording ``DEGRADED``.

Two classical estimators, picked by modality:

* **Optical — change vector analysis.** Bands are standardised against the
  *pooled* pre+post distribution, so a radiometric offset between two
  acquisitions shifts both equally and cancels instead of registering as change.
* **SAR — log-ratio.** Speckle is multiplicative, so a difference of dB values
  (a ratio of powers) is the statistically correct comparison; a linear
  difference would scale with brightness.

The decision threshold is Otsu over the magnitude field: data-derived, closed
form, and identical on every rerun.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from skimage.filters import threshold_otsu

from satquery.render.indices import to_db
from satquery.schemas.enums import Device, Modality, ToolStatus
from satquery.tools.base import (
    Array,
    Bands,
    ImageBundle,
    Mask,
    MissingInputError,
    ToolContext,
    ToolResult,
)
from satquery.tools.change_common import change_drafts

NAME: Final[str] = "image_diff_change"

_OPTICAL_BANDS: Final[tuple[str, ...]] = ("blue", "green", "red", "nir", "swir1", "swir2")
_EPS: Final[float] = 1e-6


def _shared_bands(pre: ImageBundle, post: ImageBundle, candidates: Sequence[str]) -> list[str]:
    """The candidate bands both images carry, in canonical order."""
    return [band for band in candidates if pre.has(band) and post.has(band)]


def _robust_standardise(pre: Array, post: Array) -> tuple[Array, Array]:
    """Centre and scale both epochs by the *pooled* median and IQR.

    Pooling is the point: it removes the shared radiometric offset between two
    acquisitions without removing the difference between them, which is what a
    per-epoch normalisation would do.
    """
    pooled = np.concatenate([pre[np.isfinite(pre)], post[np.isfinite(post)]])
    if pooled.size == 0:
        return pre, post
    centre = float(np.median(pooled))
    q1, q3 = np.percentile(pooled, [25.0, 75.0])
    scale = float(q3 - q1) or float(pooled.std()) or 1.0
    return (pre - centre) / scale, (post - centre) / scale


def _cva_magnitude(pre: Bands, post: Bands, bands: Sequence[str]) -> Array:
    """Root-mean-square change vector magnitude across the shared bands."""
    squares = np.zeros_like(pre.band(bands[0]), dtype=np.float32)
    for band in bands:
        a, b = _robust_standardise(pre.band(band), post.band(band))
        squares += np.square(b - a, dtype=np.float32)
    magnitude: Array = np.sqrt(squares / float(len(bands))).astype(np.float32)
    return magnitude


def _log_ratio_magnitude(pre: Bands, post: Bands, channel: str) -> Array:
    """Absolute dB difference — the speckle-correct SAR change statistic."""
    magnitude: Array = np.abs(to_db(post.band(channel)) - to_db(pre.band(channel))).astype(
        np.float32
    )
    return magnitude


def _otsu(magnitude: Array, valid: Mask) -> float:
    """Otsu's threshold over the finite magnitudes, or 0 for a degenerate field."""
    finite = magnitude[valid & np.isfinite(magnitude)]
    if finite.size < 2 or float(finite.max() - finite.min()) < _EPS:
        return 0.0
    return float(threshold_otsu(finite))  # type: ignore[no-untyped-call]


class ImageDiffChange:
    """Classical change detection over a co-registered bi-temporal pair."""

    name = NAME

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Measure per-pixel change between the pre and post acquisitions.

        Raises:
            MissingInputError: The pair shares no band the estimator can use.
        """
        pre, post = ctx.pair()
        sensitivity = float(params.get("sensitivity", 1.0))
        min_threshold = float(params.get("min_threshold", 0.0))

        sar = pre.modality is Modality.SAR and post.modality is Modality.SAR
        if sar:
            if not (pre.has("vv") and post.has("vv")):
                raise MissingInputError("a SAR pair needs a VV channel on both epochs")
            bands = ["vv"]
            method = "log_ratio"
        else:
            bands = _shared_bands(pre, post, _OPTICAL_BANDS)
            if not bands:
                raise MissingInputError("the pair shares no comparable optical band")
            method = "cva"

        pre_stack = ctx.pixels.read(pre, bands)
        post_stack = ctx.pixels.read(post, bands)
        if pre_stack.stack.shape != post_stack.stack.shape:
            raise MissingInputError("the two epochs did not render onto a common grid")

        magnitude = (
            _log_ratio_magnitude(pre_stack, post_stack, "vv")
            if sar
            else _cva_magnitude(pre_stack, post_stack, bands)
        )
        valid = pre_stack.valid & post_stack.valid
        threshold = max(_otsu(magnitude, valid) * sensitivity, min_threshold)
        mask: Mask = valid & np.isfinite(magnitude) & (magnitude > threshold)

        finite = magnitude[valid & np.isfinite(magnitude)]
        valid_px = int(valid.sum())
        changed_px = int(mask.sum())
        changed_pct = round(100.0 * changed_px / valid_px, 4) if valid_px else 0.0

        drafts, payload = change_drafts(
            mask=mask,
            post=post,
            post_stack=post_stack,
            stats={
                "method": method,
                "threshold": round(threshold, 6),
                "changed_pixel_count": changed_px,
                "valid_pixel_count": valid_px,
            },
        )

        return ToolResult(
            status=ToolStatus.OK,
            scalars={
                "changed_area_pct": changed_pct,
                "changed_pixel_count": changed_px,
                "threshold": round(threshold, 6),
                "method": method,
                "magnitude_mean": round(float(finite.mean()), 6) if finite.size else 0.0,
                "magnitude_std": round(float(finite.std()), 6) if finite.size else 0.0,
            },
            artifacts=drafts,
            data={"change_mask": payload},
            params={
                "method": method,
                "bands": bands,
                "sensitivity": sensitivity,
                "min_threshold": min_threshold,
                "threshold_rule": "otsu",
            },
            # A classical estimator, honestly scored: it is a defensible floor,
            # not a trained detector, and the confidence must say so.
            confidence=0.72,
            device=Device.CPU,
        )
