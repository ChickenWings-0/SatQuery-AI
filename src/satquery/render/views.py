"""The frozen view catalogue and selection policy (DATA_ADAPTATION_PLAN §2.1, §2.4).

Views are the interface between a raster and the language model: instead of
teaching the model to read 12-band imagery, we render a small set of *named,
interpretable 3-channel views* and name them in the prompt. This module declares
what those views are, which logical bands each needs, and which of them a given
pair type and modality should produce.

Availability is data-driven: a view whose required bands cannot be resolved is
dropped and later slots renumber. That is what makes Cartosat-2S — which has no
SWIR — degrade to four views instead of silently substituting a band and
producing a confidently wrong built-up answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from satquery.render.indices import INDEX_DOMAIN, SARDB_DOMAIN
from satquery.render.view_labels import SCALE_NOTES, VIEW_NAMES
from satquery.schemas.enums import ArtifactType, Modality, PairType

MAX_VIEWS: Final[int] = 6
"""Hard cap per sample — the dominant VRAM knob (Master.md §6.2)."""


class ViewId(StrEnum):
    """The frozen view catalogue keys."""

    TC = "TC"
    FCIR = "FCIR"
    SWIR = "SWIR"
    NDVI = "NDVI"
    NDWI = "NDWI"
    NDBI = "NDBI"
    SARFC = "SARFC"
    SARDB = "SARDB"
    PAN = "PAN"
    CHANGE = "CHANGE"


class ImageFormat(StrEnum):
    """Encoding for a rendered view."""

    PNG = "png"
    JPEG = "jpeg"

    @property
    def mime(self) -> str:
        """The MIME type this format is served as."""
        return "image/png" if self is ImageFormat.PNG else "image/jpeg"

    @property
    def extension(self) -> str:
        """The file extension used in artifact URLs."""
        return "png" if self is ImageFormat.PNG else "jpg"


@dataclass(frozen=True)
class ViewSpec:
    """One row of the view catalogue."""

    id: ViewId
    required_bands: tuple[str, ...]
    image_format: ImageFormat
    artifact_type: ArtifactType
    modalities: frozenset[Modality]
    colormap: str | None = None
    domain: tuple[float, float] | None = None

    @property
    def name(self) -> str:
        """The view's display name, from the single label source of truth."""
        return VIEW_NAMES[self.id.value]

    @property
    def scale_note(self) -> str | None:
        """The fixed-domain note, when this view has one."""
        return SCALE_NOTES.get(self.id.value)

    @property
    def is_fixed_domain(self) -> bool:
        """True when this view must never be stretched per image."""
        return self.domain is not None

    def is_available_for(self, resolved_bands: dict[str, int]) -> bool:
        """True when every band this view needs is present in the image."""
        return all(band in resolved_bands for band in self.required_bands)

    def missing_bands(self, resolved_bands: dict[str, int]) -> list[str]:
        """The required bands this image cannot supply."""
        return [band for band in self.required_bands if band not in resolved_bands]


_OPTICAL: Final[frozenset[Modality]] = frozenset({Modality.OPTICAL})
_SAR: Final[frozenset[Modality]] = frozenset({Modality.SAR})
_PAN: Final[frozenset[Modality]] = frozenset({Modality.PANCHROMATIC})

CATALOGUE: Final[dict[ViewId, ViewSpec]] = {
    ViewId.TC: ViewSpec(
        id=ViewId.TC,
        required_bands=("red", "green", "blue"),
        image_format=ImageFormat.JPEG,
        artifact_type=ArtifactType.RENDERED_VIEW,
        modalities=_OPTICAL,
    ),
    ViewId.FCIR: ViewSpec(
        id=ViewId.FCIR,
        required_bands=("nir", "red", "green"),
        image_format=ImageFormat.JPEG,
        artifact_type=ArtifactType.RENDERED_VIEW,
        modalities=_OPTICAL,
    ),
    ViewId.SWIR: ViewSpec(
        id=ViewId.SWIR,
        required_bands=("swir2", "nir", "red"),
        image_format=ImageFormat.JPEG,
        artifact_type=ArtifactType.RENDERED_VIEW,
        modalities=_OPTICAL,
    ),
    ViewId.NDVI: ViewSpec(
        id=ViewId.NDVI,
        required_bands=("nir", "red"),
        image_format=ImageFormat.PNG,
        artifact_type=ArtifactType.HEATMAP,
        modalities=_OPTICAL,
        colormap="RdYlGn",
        domain=INDEX_DOMAIN,
    ),
    ViewId.NDWI: ViewSpec(
        id=ViewId.NDWI,
        required_bands=("green", "nir"),
        image_format=ImageFormat.PNG,
        artifact_type=ArtifactType.HEATMAP,
        modalities=_OPTICAL,
        colormap="BrBG_r",
        domain=INDEX_DOMAIN,
    ),
    ViewId.NDBI: ViewSpec(
        id=ViewId.NDBI,
        required_bands=("swir1", "nir"),
        image_format=ImageFormat.PNG,
        artifact_type=ArtifactType.HEATMAP,
        modalities=_OPTICAL,
        colormap="RdBu_r",
        domain=INDEX_DOMAIN,
    ),
    ViewId.SARFC: ViewSpec(
        id=ViewId.SARFC,
        required_bands=("vv", "vh"),
        image_format=ImageFormat.JPEG,
        artifact_type=ArtifactType.RENDERED_VIEW,
        modalities=_SAR,
    ),
    ViewId.SARDB: ViewSpec(
        id=ViewId.SARDB,
        required_bands=("vv",),
        image_format=ImageFormat.PNG,
        artifact_type=ArtifactType.HEATMAP,
        modalities=_SAR,
        colormap="gray",
        domain=SARDB_DOMAIN,
    ),
    ViewId.PAN: ViewSpec(
        id=ViewId.PAN,
        required_bands=("pan",),
        image_format=ImageFormat.JPEG,
        artifact_type=ArtifactType.RENDERED_VIEW,
        modalities=_PAN,
    ),
    ViewId.CHANGE: ViewSpec(
        id=ViewId.CHANGE,
        required_bands=(),
        image_format=ImageFormat.PNG,
        artifact_type=ArtifactType.OVERLAY_PNG,
        modalities=frozenset(Modality),
    ),
}

# Slot order per modality, longest-first; unavailable views drop out (§2.4).
_OPTICAL_ORDER: Final[tuple[ViewId, ...]] = (
    ViewId.TC,
    ViewId.FCIR,
    ViewId.SWIR,
    ViewId.NDVI,
    ViewId.NDBI,
    ViewId.NDWI,
)
_SAR_ORDER: Final[tuple[ViewId, ...]] = (ViewId.SARFC, ViewId.SARDB)
_PAN_ORDER: Final[tuple[ViewId, ...]] = (ViewId.PAN,)

_CROSS_MODAL_OPTICAL: Final[tuple[ViewId, ...]] = (
    ViewId.TC,
    ViewId.FCIR,
    ViewId.NDVI,
    ViewId.NDBI,
)
_BI_TEMPORAL_OPTICAL: Final[tuple[ViewId, ...]] = (ViewId.TC, ViewId.FCIR, ViewId.NDBI)
_BI_TEMPORAL_SAR: Final[tuple[ViewId, ...]] = (ViewId.SARFC, ViewId.SARDB)


@dataclass(frozen=True)
class ViewRequest:
    """One view to render, bound to the image it comes from."""

    view: ViewSpec
    image_index: int

    @property
    def view_id(self) -> ViewId:
        """The catalogue key of the requested view."""
        return self.view.id


@dataclass(frozen=True)
class ImageViewContext:
    """What the selector needs to know about one input image."""

    modality: Modality
    resolved_bands: dict[str, int]
    sensor: str | None = None


def _order_for(modality: Modality) -> tuple[ViewId, ...]:
    """The default slot order for a modality."""
    if modality is Modality.SAR:
        return _SAR_ORDER
    if modality is Modality.PANCHROMATIC:
        return _PAN_ORDER
    return _OPTICAL_ORDER


def _available(
    order: tuple[ViewId, ...], context: ImageViewContext, index: int
) -> list[ViewRequest]:
    """Filter a slot order down to the views this image can actually produce."""
    return [
        ViewRequest(view=CATALOGUE[view_id], image_index=index)
        for view_id in order
        if CATALOGUE[view_id].is_available_for(context.resolved_bands)
    ]


def select_views(
    pair_type: PairType,
    contexts: list[ImageViewContext],
    max_views: int = MAX_VIEWS,
) -> list[ViewRequest]:
    """Choose which views to render, in the frozen slot order (§2.4).

    Args:
        pair_type: How the inputs relate to one another.
        contexts: One entry per input image, in upload order.
        max_views: Hard cap, defaulting to the frozen 6.

    Returns:
        Requests in slot order. Slot numbers are assigned later by the renderer,
        because a dropped view renumbers everything after it — the *label*
        carries the meaning, not the slot index.
    """
    if not contexts:
        return []

    if pair_type is PairType.SINGLE or len(contexts) == 1:
        requests = _available(_order_for(contexts[0].modality), contexts[0], 0)

    elif pair_type is PairType.CROSS_MODAL:
        requests = []
        for index, context in enumerate(contexts):
            order = _SAR_ORDER if context.modality is Modality.SAR else _CROSS_MODAL_OPTICAL
            requests.extend(_available(order, context, index))

    else:  # BI_TEMPORAL and INCOMPATIBLE both render both images side by side.
        sar = all(context.modality is Modality.SAR for context in contexts)
        order = _BI_TEMPORAL_SAR if sar else _BI_TEMPORAL_OPTICAL
        requests = []
        for view_id in order:
            # Interleave pre/post so the two halves of a comparison stay adjacent.
            for index, context in enumerate(contexts):
                if CATALOGUE[view_id].is_available_for(context.resolved_bands):
                    requests.append(ViewRequest(view=CATALOGUE[view_id], image_index=index))

    return requests[:max_views]
