"""``spectral_index_analyzer`` — NDVI / NDWI / NDBI statistics and class fractions.

Indices are the workhorse of optical reasoning here, and the reason is
domain-adaptation rather than convenience: a normalised difference is a *ratio*,
so it survives the radiometric differences between Sentinel-2 and Cartosat-2S
that a raw reflectance threshold would not. That is what lets the same fixed
thresholds mean the same thing on a sensor the system has never seen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from satquery.render.composites import index_view
from satquery.render.indices import index_statistics, ndbi, ndvi, ndwi
from satquery.schemas.enums import ArtifactType, Device, ToolStatus
from satquery.tools.base import (
    Array,
    ArtifactDraft,
    Draft,
    MissingInputError,
    ToolContext,
    ToolResult,
    suffix_for,
)

NAME: Final[str] = "spectral_index_analyzer"

INDEX_BANDS: Final[dict[str, tuple[str, ...]]] = {
    "ndvi": ("nir", "red"),
    "ndwi": ("green", "nir"),
    "ndbi": ("swir1", "nir"),
}
"""Which logical bands each index needs. An index whose bands a sensor does not
carry is *omitted*, never approximated from a neighbouring band."""

INDEX_COLORMAPS: Final[dict[str, str]] = {"ndvi": "RdYlGn", "ndwi": "BrBG_r", "ndbi": "RdBu_r"}

FRACTION_THRESHOLDS: Final[dict[str, tuple[str, float]]] = {
    "vegetation_fraction_pct": ("ndvi", 0.30),
    "water_fraction_pct": ("ndwi", 0.00),
    "built_up_fraction_pct": ("ndbi", 0.00),
}
"""Fixed decision thresholds (DATA_ADAPTATION_PLAN §2.3). Fixed, not
percentile-derived: a per-image threshold would make "20 % built-up" mean
something different in every scene, and the FactSheet is a citation source."""

_DOMAIN: Final[tuple[float, float]] = (-1.0, 1.0)


def _compute(values: Array, name: str) -> tuple[Array, dict[str, float]]:
    """Return the index field and its statistics over valid pixels only."""
    return values, index_statistics(values, name)


class SpectralIndexAnalyzer:
    """Normalised-difference indices over one or two optical images."""

    name = NAME

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Compute the requested indices for every optical input image.

        Raises:
            MissingInputError: No requested index was computable on any input.
        """
        requested: Sequence[str] = [str(i).lower() for i in params.get("indices", ["ndvi"])]
        suffixes = [str(s) for s in params.get("suffixes", ["_pre", "_post"])]
        thresholds = {
            key: float(params.get(key, default))
            for key, (_, default) in FRACTION_THRESHOLDS.items()
        }
        effective: dict[str, Any] = {"indices": list(requested), **thresholds}
        if len(ctx.images) > 1:
            effective["suffixes"] = suffixes

        scalars: dict[str, float | str] = {}
        drafts: list[Draft] = []
        computed_any: list[str] = []

        for index, image in enumerate(ctx.images):
            suffix = suffix_for(ctx.images, index, suffixes)
            fields: dict[str, Array] = {}
            available: list[str] = []
            for name in requested:
                bands = INDEX_BANDS.get(name)
                if bands is None or not image.has(*bands):
                    continue
                stack = ctx.pixels.read(image, bands)
                if name == "ndvi":
                    values = ndvi(stack.band("nir"), stack.band("red"))
                elif name == "ndwi":
                    values = ndwi(stack.band("green"), stack.band("nir"))
                else:
                    values = ndbi(stack.band("swir1"), stack.band("nir"))
                field_values, statistics = _compute(values, name)
                fields[name] = field_values
                available.append(name)
                for key, value in statistics.items():
                    scalars[f"{key}{suffix}"] = round(value, 6)

                colours = index_view(
                    field_values,
                    INDEX_COLORMAPS[name],
                    _DOMAIN,
                    np.isfinite(field_values),
                )
                drafts.append(
                    ArtifactDraft(
                        key=f"{name}{suffix}",
                        type=ArtifactType.HEATMAP,
                        label=f"{name.upper()} ({image.id}{suffix or ''}), domain -1 to 1",
                        image=colours,
                        raster=field_values,
                        geometry=stack.geometry,
                        stats={
                            "fixed_domain": list(_DOMAIN),
                            "source_image": image.id,
                            **statistics,
                        },
                    )
                )

            for fraction, (source, _) in FRACTION_THRESHOLDS.items():
                source_values = fields.get(source)
                if source_values is None:
                    continue
                finite = source_values[np.isfinite(source_values)]
                if finite.size == 0:
                    continue
                above = float((finite > thresholds[fraction]).mean())
                scalars[f"{fraction}{suffix}"] = round(100.0 * above, 4)

            computed_any.extend(available)

        if not computed_any:
            raise MissingInputError(
                "none of the requested indices are computable from the available bands"
            )
        scalars["indices_computed"] = ",".join(dict.fromkeys(computed_any))

        return ToolResult(
            status=ToolStatus.OK,
            scalars=scalars,
            artifacts=drafts,
            params=effective,
            confidence=0.95,
            device=Device.CPU,
        )


