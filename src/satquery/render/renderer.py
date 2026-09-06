"""The renderer: one substrate, two consumers (Master.md §8 Phase 2).

Given the images an analysis is working on, this produces the named views that
go into the VLM prompt *and* the ``ArtifactRef`` evidence the frontend renders —
from the same pixels, with the same labels. That identity is the point: the
evidence gallery shows literally the images the model saw.

Each source image is read exactly once, for the union of the bands its views
need, and every view of that image shares one geometry, so overlays composite
without resampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from satquery.render.artifact_store import ArtifactStore
from satquery.render.composites import (
    false_colour_ir,
    index_view,
    panchromatic,
    sar_backscatter,
    sar_false_colour,
    swir_composite,
    true_colour,
)
from satquery.render.indices import index_statistics, ndbi, ndvi, ndwi, to_db
from satquery.render.tiling import VIEW_SIZE_PX, ViewGeometry, render_source
from satquery.render.view_labels import label_for_view
from satquery.render.views import (
    CATALOGUE,
    ImageViewContext,
    ViewId,
    ViewRequest,
    ViewSpec,
    select_views,
)
from satquery.schemas.enums import ArtifactType, ImageRole, Modality, PairType
from satquery.schemas.trace import ArtifactGeo, ArtifactRef

Rgb = npt.NDArray[np.uint8]
Array = npt.NDArray[np.float32]
Mask = npt.NDArray[np.bool_]

_GEOTIFF_COMPANION_TYPES: Final[frozenset[ArtifactType]] = frozenset(
    {ArtifactType.HEATMAP, ArtifactType.CHANGE_MASK, ArtifactType.SEGMENTATION}
)
"""Artifact types the contract emits as PNG *and* GeoTIFF (API_CONTRACT §2.5)."""


@dataclass(frozen=True)
class RenderSource:
    """One input image, with everything the renderer needs to name and read it."""

    image_id: str
    path: Path
    modality: Modality
    resolved_bands: dict[str, int]
    sensor: str | None = None
    role: ImageRole | None = None
    acquisition_time: datetime | None = None


@dataclass
class RenderedView:
    """A rendered view: the pixels the model sees and the artifact the UI shows."""

    view: ViewSpec
    slot: int
    label: str
    image_id: str
    rgb: Rgb
    geometry: ViewGeometry
    artifact: ArtifactRef
    scalars: dict[str, float] = field(default_factory=dict)

    @property
    def view_id(self) -> ViewId:
        """The catalogue key of this view."""
        return self.view.id


@dataclass
class _ImagePixels:
    """The band stack read once for one source image."""

    order: list[str]
    bands: Array
    valid: Mask
    geometry: ViewGeometry

    def band(self, name: str) -> Array:
        """Return one logical band from the cached stack."""
        selected: Array = self.bands[self.order.index(name)]
        return selected


def _read_image(source: RenderSource, requests: list[ViewRequest], size: int) -> _ImagePixels:
    """Read the union of the bands this image's views need, in one pass."""
    needed: list[str] = []
    for request in requests:
        for band in request.view.required_bands:
            if band not in needed:
                needed.append(band)
    indexes = [source.resolved_bands[name] for name in needed]
    bands, valid, geometry = render_source(source.path, indexes, size=size)
    return _ImagePixels(order=needed, bands=bands, valid=valid, geometry=geometry)


def _render_one(view: ViewSpec, pixels: _ImagePixels) -> tuple[Rgb, Array | None, str | None]:
    """Render one view, returning its RGB, the underlying field, and a stat prefix.

    The field is returned for measurement views so the caller can both compute
    statistics over it and write a georeferenced companion — the same numbers the
    picture shows.
    """
    valid = pixels.valid
    band = pixels.band
    domain = view.domain or (-1.0, 1.0)

    if view.id is ViewId.TC:
        return true_colour(band("red"), band("green"), band("blue"), valid), None, None
    if view.id is ViewId.FCIR:
        return false_colour_ir(band("nir"), band("red"), band("green"), valid), None, None
    if view.id is ViewId.SWIR:
        return swir_composite(band("swir2"), band("nir"), band("red"), valid), None, None
    if view.id is ViewId.PAN:
        return panchromatic(band("pan"), valid), None, None
    if view.id is ViewId.SARFC:
        rgb = sar_false_colour(to_db(band("vv")), to_db(band("vh")), valid)
        return rgb, None, None
    if view.id is ViewId.SARDB:
        vv_db = to_db(band("vv"))
        return sar_backscatter(vv_db, valid), vv_db, "sigma0_vv_db"

    if view.id is ViewId.NDVI:
        values, colormap, prefix = ndvi(band("nir"), band("red")), "RdYlGn", "ndvi"
    elif view.id is ViewId.NDWI:
        values, colormap, prefix = ndwi(band("green"), band("nir")), "BrBG_r", "ndwi"
    elif view.id is ViewId.NDBI:
        values, colormap, prefix = ndbi(band("swir1"), band("nir")), "RdBu_r", "ndbi"
    else:
        raise ValueError(f"{view.id} is not renderable from source bands")
    return index_view(values, colormap, domain, valid), values, prefix


def render_views(
    sources: list[RenderSource],
    pair_type: PairType,
    trace_id: str,
    store: ArtifactStore,
    produced_by_step: int = 1,
    size: int = VIEW_SIZE_PX,
    first_artifact_index: int = 0,
) -> list[RenderedView]:
    """Render every view the policy selects, store it, and describe it.

    Args:
        sources: The input images, in upload order.
        pair_type: Drives view selection (DATA_ADAPTATION_PLAN §2.4).
        trace_id: The trace these artifacts belong to.
        store: Where encoded artifacts are written.
        produced_by_step: Plan step index recorded on each artifact.
        size: View edge in pixels.
        first_artifact_index: Starting number for ``art_{n}`` ids.

    Returns:
        The rendered views in slot order. Slots are 1-indexed and contiguous:
        a view dropped for missing bands renumbers everything after it, because
        the label carries the meaning rather than the slot index.
    """
    contexts = [
        ImageViewContext(
            modality=source.modality, resolved_bands=source.resolved_bands, sensor=source.sensor
        )
        for source in sources
    ]
    requests = select_views(pair_type, contexts)
    if not requests:
        return []

    pixels_by_image = {
        index: _read_image(sources[index], [r for r in requests if r.image_index == index], size)
        for index in sorted({request.image_index for request in requests})
    }

    rendered: list[RenderedView] = []
    for slot, request in enumerate(requests, start=1):
        source = sources[request.image_index]
        pixels = pixels_by_image[request.image_index]
        rgb, field_values, prefix = _render_one(request.view, pixels)

        artifact_id = f"art_{first_artifact_index + len(rendered)}"
        blob = store.put_image(trace_id, artifact_id, rgb, request.view.image_format)

        geotiff_url: str | None = None
        if (
            request.view.artifact_type in _GEOTIFF_COMPANION_TYPES
            and field_values is not None
            and pixels.geometry.transform is not None
        ):
            companion = store.put_geotiff(
                trace_id,
                artifact_id,
                field_values.astype(np.float32),
                pixels.geometry.transform,
                pixels.geometry.crs,
                nodata=float("nan"),
            )
            geotiff_url = companion.url

        label = label_for_view(
            view_id=request.view.id.value,
            slot=slot,
            modality=source.modality,
            sensor=source.sensor,
            role=source.role,
            acquisition_time=source.acquisition_time,
        )
        scalars = (
            index_statistics(field_values, prefix) if (field_values is not None and prefix) else {}
        )

        geometry = pixels.geometry
        rendered.append(
            RenderedView(
                view=request.view,
                slot=slot,
                label=label,
                image_id=source.image_id,
                rgb=rgb,
                geometry=geometry,
                scalars=scalars,
                artifact=ArtifactRef(
                    id=artifact_id,
                    type=request.view.artifact_type,
                    mime=blob.mime,
                    label=label,
                    url=blob.url,
                    geotiff_url=geotiff_url,
                    geo=(
                        ArtifactGeo(crs=geometry.crs, transform=geometry.transform_list)
                        if geometry.crs and geometry.transform_list
                        else None
                    ),
                    width=geometry.width,
                    height=geometry.height,
                    stats={
                        "view_id": request.view.id.value,
                        "source_image": source.image_id,
                        "pad": geometry.pad.as_dict(),
                        "fixed_domain": list(request.view.domain) if request.view.domain else None,
                        **scalars,
                    },
                    produced_by_step=produced_by_step,
                ),
            )
        )
    return rendered


def unavailable_views(sources: list[RenderSource], pair_type: PairType) -> dict[str, list[str]]:
    """Report which catalogue views each image cannot produce, and why.

    Unavailability is explicit and never substituted: Cartosat-2S has no SWIR, so
    NDBI is reported missing rather than faked from another band.
    """
    report: dict[str, list[str]] = {}
    for source in sources:
        gaps = [
            f"{view_id.value}: missing {', '.join(spec.missing_bands(source.resolved_bands))}"
            for view_id, spec in CATALOGUE.items()
            if source.modality in spec.modalities
            and view_id is not ViewId.CHANGE
            and not spec.is_available_for(source.resolved_bands)
        ]
        if gaps:
            report[source.image_id] = gaps
    return report
