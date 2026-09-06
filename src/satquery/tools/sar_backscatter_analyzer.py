"""``sar_backscatter_analyzer`` — sigma-nought statistics, VV/VH ratio and speckle.

SAR measures roughness and geometry, not colour, so the useful statements are
about backscatter level rather than reflectance. Everything here is computed in
dB, because the linear amplitude distribution is heavily skewed and its mean is
dominated by a handful of bright scatterers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from satquery.render.composites import sar_backscatter
from satquery.render.indices import index_statistics, to_db, vv_vh_ratio_db
from satquery.schemas.enums import ArtifactType, Device, ToolStatus
from satquery.tools.base import (
    ArtifactDraft,
    Draft,
    MissingInputError,
    ToolContext,
    ToolResult,
    suffix_for,
)

NAME: Final[str] = "sar_backscatter_analyzer"

WATER_THRESHOLD_DB: Final[float] = -18.0
"""Below this, a surface is specular enough to read as open water. Frozen so the
number means the same thing on Sentinel-1 and RISAT-1."""


class SarBackscatterAnalyzer:
    """Backscatter statistics over one or two SAR images."""

    name = NAME

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Characterise the backscatter of every SAR input image.

        Raises:
            MissingInputError: No input carries a VV channel.
        """
        water_db = float(params.get("water_threshold_db", WATER_THRESHOLD_DB))
        suffixes = [str(s) for s in params.get("suffixes", ["_pre", "_post"])]
        effective: dict[str, Any] = {"water_threshold_db": water_db}
        if len(ctx.images) > 1:
            effective["suffixes"] = suffixes

        scalars: dict[str, float | str] = {}
        drafts: list[Draft] = []
        polarisations: list[str] = []

        for index, image in enumerate(ctx.images):
            if not image.has("vv"):
                continue
            suffix = suffix_for(ctx.images, index, suffixes)
            channels = ["vv"] + (["vh"] if image.has("vh") else [])
            stack = ctx.pixels.read(image, channels)

            vv_db = to_db(stack.band("vv"))
            for key, value in index_statistics(vv_db, "sigma0_vv_db").items():
                scalars[f"{key}{suffix}"] = round(value, 6)

            finite = vv_db[np.isfinite(vv_db)]
            if finite.size:
                scalars[f"low_backscatter_fraction_pct{suffix}"] = round(
                    100.0 * float((finite < water_db).mean()), 4
                )
                # Coefficient of variation in linear power: the standard
                # single-look speckle descriptor. Computed in linear space
                # because a CV over dB values is not a speckle measure.
                linear = np.power(10.0, finite / 10.0)
                mean = float(linear.mean())
                scalars[f"speckle_cv{suffix}"] = (
                    round(float(linear.std()) / mean, 6) if mean > 0 else 0.0
                )

            if "vh" in channels:
                vh_db = to_db(stack.band("vh"))
                for key, value in index_statistics(vh_db, "sigma0_vh_db").items():
                    scalars[f"{key}{suffix}"] = round(value, 6)
                ratio = vv_vh_ratio_db(vv_db, vh_db)
                for key, value in index_statistics(ratio, "vv_vh_ratio_db").items():
                    scalars[f"{key}{suffix}"] = round(value, 6)

            polarisations.extend(channels)
            drafts.append(
                ArtifactDraft(
                    key=f"sigma0{suffix}",
                    type=ArtifactType.HEATMAP,
                    label=f"Sigma-nought VV in dB ({image.id})",
                    image=sar_backscatter(vv_db, stack.valid),
                    raster=vv_db,
                    geometry=stack.geometry,
                    stats={
                        "source_image": image.id,
                        "fixed_domain": [-25.0, 0.0],
                        "water_threshold_db": water_db,
                    },
                )
            )

        if not drafts:
            raise MissingInputError("no input image carries a VV channel")
        scalars["polarisations"] = ",".join(dict.fromkeys(polarisations))

        return ToolResult(
            status=ToolStatus.OK,
            scalars=scalars,
            artifacts=drafts,
            params=effective,
            confidence=0.93,
            device=Device.CPU,
        )
