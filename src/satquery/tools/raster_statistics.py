"""``raster_statistics`` — per-band reflectance statistics and scene brightness.

The cheapest evidence in the system, and the one every generic plan falls back
on. It says nothing clever; it says something *true*, which is what the answer
needs when no specialist tool applies.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from satquery.schemas.enums import ArtifactType, Device, ToolStatus
from satquery.tools.base import (
    ArtifactDraft,
    ToolContext,
    ToolResult,
    suffix_for,
)

NAME: Final[str] = "raster_statistics"

_BAND_ORDER: Final[tuple[str, ...]] = (
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "pan",
    "vv",
    "vh",
)
"""Canonical reading order, so the band list a run reports is stable."""


class RasterStatistics:
    """Band statistics over the rendered canvas."""

    name = NAME

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Summarise every resolved band of every input image."""
        suffixes = [str(s) for s in params.get("suffixes", ["_pre", "_post"])]
        percentiles = [float(p) for p in params.get("percentiles", [2.0, 98.0])]
        effective: dict[str, Any] = {"percentiles": percentiles}
        if len(ctx.images) > 1:
            effective["suffixes"] = suffixes

        scalars: dict[str, float | str] = {}
        inline: dict[str, Any] = {}
        for index, image in enumerate(ctx.images):
            suffix = suffix_for(ctx.images, index, suffixes)
            bands = [name for name in _BAND_ORDER if image.has(name)]
            if not bands:
                continue
            stack = ctx.pixels.read(image, bands)
            finite = stack.stack[np.isfinite(stack.stack)]
            if finite.size == 0:
                continue
            low, high = np.percentile(finite, percentiles)
            scalars[f"band_count{suffix}"] = len(bands)
            scalars[f"brightness_mean{suffix}"] = round(float(finite.mean()), 6)
            scalars[f"brightness_std{suffix}"] = round(float(finite.std()), 6)
            scalars[f"dynamic_range{suffix}"] = round(float(high - low), 6)
            scalars[f"valid_pixel_pct{suffix}"] = round(
                100.0 * float(stack.valid.mean()), 4
            )
            inline[image.id] = {
                "bands": bands,
                "per_band_mean": [
                    round(float(np.nanmean(stack.stack[i])), 6) for i in range(len(bands))
                ],
                "per_band_std": [
                    round(float(np.nanstd(stack.stack[i])), 6) for i in range(len(bands))
                ],
            }

        return ToolResult(
            status=ToolStatus.OK,
            scalars=scalars,
            artifacts=[
                ArtifactDraft(
                    key="stats",
                    type=ArtifactType.SCALARS,
                    label="Band statistics",
                    inline=inline,
                )
            ],
            params=effective,
            confidence=0.99,
            device=Device.CPU,
        )
