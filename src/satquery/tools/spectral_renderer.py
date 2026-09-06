"""``spectral_renderer`` — the Phase 2 renderer behind the tool protocol.

Step 1 of every plan, and the only step marked non-optional: without rendered
views there is no VLM input and no evidence gallery, so its failure is a 500
rather than a degraded answer (AGENT_POLICY_DAG §6.3).

The render itself is deferred to materialisation time. ``render_views`` numbers
its own artifacts, and the executor is the only thing that knows what the next
free ``art_{n}`` is — so the tool returns the request and the executor performs
it. That is what keeps artifact ids identical across reruns even though the
steps around them ran concurrently.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from satquery.render.renderer import RenderSource, unavailable_views
from satquery.render.tiling import VIEW_SIZE_PX
from satquery.schemas.enums import Device, ToolStatus
from satquery.tools.base import (
    ImageBundle,
    MissingInputError,
    RenderDraft,
    ToolContext,
    ToolResult,
)

NAME: Final[str] = "spectral_renderer"


def render_source_for(image: ImageBundle) -> RenderSource:
    """Adapt an ingested image to the renderer's input record."""
    return RenderSource(
        image_id=image.id,
        path=image.path,
        modality=image.modality,
        resolved_bands=image.resolved_bands,
        sensor=image.sensor,
        role=image.role,
        acquisition_time=image.acquisition_time,
    )


class SpectralRenderer:
    """Renders the named views both the VLM and the frontend consume."""

    name = NAME

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Describe the render this step will perform, and what it cannot produce.

        Raises:
            MissingInputError: The step resolved to no input images.
        """
        if not ctx.images:
            raise MissingInputError("spectral_renderer needs at least one input image")

        size = int(params.get("size_px", VIEW_SIZE_PX))
        sources = [render_source_for(image) for image in ctx.images]
        gaps = unavailable_views(sources, ctx.pair_type)
        notes = [
            f"{image_id}: {reason}" for image_id, reasons in gaps.items() for reason in reasons
        ]

        return ToolResult(
            status=ToolStatus.OK,
            # views_rendered is filled in by the executor once the render has
            # actually run; a count promised before the fact would be a guess.
            scalars={"views_unavailable": len(notes)},
            artifacts=[
                RenderDraft(
                    key="views",
                    sources=tuple(sources),
                    pair_type=ctx.pair_type,
                    size=size,
                )
            ],
            params={"size_px": size, "pair_type": ctx.pair_type.value},
            confidence=0.99,
            device=Device.CPU,
            notes=notes,
        )
