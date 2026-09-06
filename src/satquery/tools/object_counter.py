"""``object_counter`` — counting discrete objects from a mask or a box set.

Registered with no fallback because it cannot meaningfully fail: connected
components on a binary mask is closed-form arithmetic. What it *can* do is be
skipped, when the grounding or segmentation step that was supposed to feed it
did not produce anything — and a skipped count is far better than a guessed one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from skimage.measure import label, regionprops

from satquery.schemas.enums import ArtifactType, Device, ToolStatus
from satquery.tools.base import (
    ArtifactDraft,
    ClassMap,
    Mask,
    MaskPayload,
    MissingInputError,
    ToolContext,
    ToolResult,
)

NAME: Final[str] = "object_counter"

MIN_COMPONENT_PX: Final[int] = 16
_CONNECTIVITY: Final[int] = 2


class ObjectCounter:
    """Counts objects in an upstream BBOX_SET or SEGMENTATION artifact."""

    name = NAME

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Count the discrete objects the upstream detector delimited.

        Raises:
            MissingInputError: Neither a box set nor a mask reached this step.
        """
        min_component_px = int(params.get("min_component_px", MIN_COMPONENT_PX))
        effective = {"min_component_px": min_component_px, "connectivity": _CONNECTIVITY}

        boxes = self._boxes(ctx)
        if boxes is not None:
            return ToolResult(
                scalars={"count": len(boxes)},
                artifacts=[
                    ArtifactDraft(
                        key="count",
                        type=ArtifactType.SCALARS,
                        label="Object count",
                        inline={"count": len(boxes), "source": "bbox_set"},
                    )
                ],
                params={**effective, "source": "bbox_set"},
                confidence=0.96,
                device=Device.CPU,
                status=ToolStatus.OK,
            )

        mask, pixel_area, source, per_class = self._raster(ctx, min_component_px)
        labelled = label(mask, connectivity=_CONNECTIVITY)  # type: ignore[no-untyped-call]
        regions = [r for r in regionprops(labelled) if r.area >= min_component_px]  # type: ignore[no-untyped-call]

        scalars: dict[str, float | str] = {"count": len(regions), **per_class}
        if pixel_area is not None and regions:
            areas = [float(r.area) * pixel_area for r in regions]
            scalars["mean_object_area_m2"] = round(sum(areas) / len(areas), 2)
            scalars["largest_object_area_m2"] = round(max(areas), 2)

        return ToolResult(
            scalars=scalars,
            artifacts=[
                ArtifactDraft(
                    key="count",
                    type=ArtifactType.SCALARS,
                    label="Object count",
                    inline={"count": len(regions), "source": source},
                )
            ],
            params={**effective, "source": source},
            confidence=0.94,
            device=Device.CPU,
            status=ToolStatus.OK,
        )

    def _boxes(self, ctx: ToolContext) -> list[Any] | None:
        """Return the upstream box list, or None when the step got a mask instead."""
        for artifact in ctx.artifacts:
            if artifact.type is ArtifactType.BBOX_SET:
                payload = ctx.data.get(artifact.id)
                if isinstance(payload, list):
                    return payload
                inline = artifact.inline or {}
                if isinstance(inline.get("boxes"), list):
                    boxes: list[Any] = inline["boxes"]
                    return boxes
        return None

    def _raster(
        self, ctx: ToolContext, min_component_px: int
    ) -> tuple[Mask, float | None, str, dict[str, float | str]]:
        """Return the raster to count over, and any per-class counts beside it.

        A change mask is one binary field and counts as one population. A
        segmentation is several: counting its components as a single mask would
        merge a building touching a road into one object, so each class is
        labelled separately and the per-class counts are reported alongside the
        total.

        Raises:
            MissingInputError: Neither a box set, a mask nor a class map reached
                this step.
        """
        for artifact in ctx.artifacts:
            payload = ctx.data.get(artifact.id)
            if isinstance(payload, MaskPayload):
                return payload.mask, payload.pixel_area_m2, "mask", {}
            if isinstance(payload, ClassMap):
                return self._from_classes(payload, min_component_px)
        raise MissingInputError(
            "object_counter needs a box set, a mask or a segmentation to count"
        )

    def _from_classes(
        self, class_map: ClassMap, min_component_px: int
    ) -> tuple[Mask, float | None, str, dict[str, float | str]]:
        """Per-class component counts, plus the union mask for the total."""
        per_class: dict[str, float | str] = {}
        union = class_map.labels >= 0
        for name, index in class_map.named():
            mask = class_map.mask_for(index)
            if not mask.any():
                continue
            labelled = label(mask, connectivity=_CONNECTIVITY)  # type: ignore[no-untyped-call]
            counted = len(
                [
                    region
                    for region in regionprops(labelled)  # type: ignore[no-untyped-call]
                    if region.area >= min_component_px
                ]
            )
            per_class[f"{name}_count"] = counted
        return union, class_map.pixel_area_m2, "segmentation", per_class
