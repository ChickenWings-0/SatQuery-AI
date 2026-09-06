"""``change_statistics`` — turning a change mask into ground truth you can cite.

A mask is a picture; an area in square metres is evidence. This tool does the
conversion, and it does it through the *rendered canvas* transform rather than
the source raster's, because the mask was produced on the canvas — measuring it
against a different grid would silently misreport the area by the downsampling
factor.

Connected-component filtering exists for the same reason: isolated single-pixel
differences are speckle and co-registration residual, not change, and letting
them into a headline percentage makes the number dishonest.

**Per-class change is produced when, and only when, a segmentation is present.**
"7.4 % of the scene changed" is a fact; "5.1 % of the scene became built-up" is
the fact an analyst actually wanted, and the difference between them is a label
map. When an upstream ``SEGMENTATION`` step supplies one (Phase 6's
``semantic_segmenter``), every changed pixel is attributed to the class it
occupies in the *post* epoch and the breakdown joins the FactSheet. When none is
supplied the breakdown is simply absent — never estimated from the change mask
alone, which would be inventing a land-cover claim out of a binary raster.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from pyproj import Transformer
from skimage.measure import label, regionprops

from satquery.schemas.enums import ArtifactType, Device, ToolStatus
from satquery.tools.base import (
    ArtifactDraft,
    ClassMap,
    Draft,
    Mask,
    MaskPayload,
    MissingInputError,
    ToolContext,
    ToolResult,
)

NAME: Final[str] = "change_statistics"

MIN_COMPONENT_PX: Final[int] = 25
"""Components smaller than this are dropped as noise (Master.md §8 Phase 3)."""

_CONNECTIVITY: Final[int] = 2
"""8-connectivity: a diagonally touching pair of pixels is one feature."""

MIN_CLASS_FRACTION_PCT: Final[float] = 0.05
"""A class below this share of the changed area is folded away rather than
reported. Attributing 0.01 % of a scene to "water" off a handful of pixels at a
segmentation boundary reads as a finding and is an artefact."""


class ChangeStatistics:
    """Connected-component statistics and ground areas over a change mask."""

    name = NAME

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Measure the change mask supplied by an upstream detector.

        Raises:
            MissingInputError: No CHANGE_MASK payload reached this step.
        """
        payload = ctx.payload(ArtifactType.CHANGE_MASK)
        if not isinstance(payload, MaskPayload):
            raise TypeError(f"expected a MaskPayload, got {type(payload).__name__}")

        min_component_px = int(params.get("min_component_px", MIN_COMPONENT_PX))
        emit_geojson = bool(params.get("emit_geojson", False))

        labelled = label(payload.mask, connectivity=_CONNECTIVITY)  # type: ignore[no-untyped-call]
        regions = [r for r in regionprops(labelled) if r.area >= min_component_px]  # type: ignore[no-untyped-call]
        # Sorted by area so the "largest component" is well defined and the
        # GeoJSON feature order is identical across reruns.
        regions.sort(key=lambda r: (-int(r.area), int(r.bbox[0]), int(r.bbox[1])))

        kept_px = int(sum(int(r.area) for r in regions))
        total_px = payload.total_px
        pixel_area = payload.pixel_area_m2
        area_m2 = kept_px * pixel_area if pixel_area is not None else None

        scalars: dict[str, float | str] = {
            "changed_pixel_count": kept_px,
            "changed_area_pct": round(100.0 * kept_px / total_px, 4) if total_px else 0.0,
            "component_count": len(regions),
            "min_component_px": min_component_px,
        }
        if area_m2 is not None and pixel_area is not None:
            scalars["changed_area_m2"] = round(area_m2, 2)
            scalars["changed_area_km2"] = round(area_m2 / 1e6, 6)
            scalars["largest_component_m2"] = (
                round(float(regions[0].area) * pixel_area, 2) if regions else 0.0
            )
            scalars["mean_component_m2"] = (
                round(area_m2 / len(regions), 2) if regions else 0.0
            )

        drafts: list[Draft] = [
            ArtifactDraft(
                key="change_stats",
                type=ArtifactType.SCALARS,
                label="Change statistics",
                inline={
                    "components": [
                        {
                            "bbox_px": [int(v) for v in region.bbox],
                            "area_px": int(region.area),
                            "area_m2": (
                                round(float(region.area) * pixel_area, 2)
                                if pixel_area is not None
                                else None
                            ),
                            "centroid_px": [round(float(v), 2) for v in region.centroid],
                        }
                        # Ten is what a UI can list; the totals above cover the rest.
                        for region in regions[:10]
                    ],
                    "component_count": len(regions),
                    "min_component_px": min_component_px,
                },
            )
        ]
        if emit_geojson:
            drafts.append(
                ArtifactDraft(
                    key="change_polygons",
                    type=ArtifactType.GEOJSON,
                    label="Changed regions (bounding boxes)",
                    geojson=_feature_collection(regions, payload),
                )
            )

        filtered_out = int(payload.mask.sum()) - kept_px
        notes = (
            [f"dropped {filtered_out} px in components smaller than {min_component_px} px"]
            if filtered_out
            else []
        )

        # Only the components that survived filtering are attributed, so the
        # per-class shares add up to the headline changed area rather than to the
        # raw mask.
        kept_mask: Mask = _kept_labels(labelled, regions)
        classes = _class_map(ctx)
        if classes is not None:
            class_scalars, class_notes = per_class_change(
                kept_mask, classes, pixel_area, kept_px
            )
            scalars.update(class_scalars)
            notes.extend(class_notes)

        return ToolResult(
            status=ToolStatus.OK,
            scalars=scalars,
            artifacts=drafts,
            params={
                "min_component_px": min_component_px,
                "connectivity": _CONNECTIVITY,
                "emit_geojson": emit_geojson,
                "per_class": classes is not None,
                "classes": list(classes.classes) if classes is not None else [],
            },
            confidence=0.97,
            device=Device.CPU,
            notes=notes,
        )


def _kept_labels(labelled: npt.NDArray[np.int_], regions: list[Any]) -> Mask:
    """Boolean selection of the component labels that survived size filtering."""
    keep = np.zeros(int(labelled.max()) + 1, dtype=bool)
    for region in regions:
        keep[int(region.label)] = True
    selected: Mask = keep[labelled]
    return selected


def _class_map(ctx: ToolContext) -> ClassMap | None:
    """Return the segmentation this step was given, if it was given one."""
    try:
        payload = ctx.payload(ArtifactType.SEGMENTATION)
    except MissingInputError:
        return None
    if not isinstance(payload, ClassMap):
        return None
    return payload


def per_class_change(
    mask: Mask, classes: ClassMap, pixel_area_m2: float | None, changed_px: int
) -> tuple[dict[str, float | str], list[str]]:
    """Attribute changed pixels to the land-cover class they became.

    The *post*-epoch label is the one that matters: an analyst asking what changed
    wants to know what the ground is now, not what it stopped being. A future
    from/to transition matrix is a different measurement and would need both
    epochs segmented.

    Args:
        mask: The filtered change mask.
        classes: The post-epoch label map, on the same canvas.
        pixel_area_m2: Ground area of one canvas pixel, when georeferenced.
        changed_px: Total changed pixels, for the share calculation.

    Returns:
        ``(scalars, notes)``. Scalars are named ``changed_{class}_pct`` — a share
        of the *changed area*, not of the scene — and ``changed_{class}_m2``.

    Raises:
        MissingInputError: The label map is not on the mask's grid, which would
            attribute every pixel to the wrong place.
    """
    if classes.labels.shape != mask.shape:
        raise MissingInputError(
            f"the segmentation is {classes.labels.shape} but the change mask is "
            f"{mask.shape}; they were not produced on a common canvas"
        )

    scalars: dict[str, float | str] = {}
    notes: list[str] = []
    if changed_px == 0:
        return scalars, notes

    attributed = 0
    small: list[str] = []
    for name, index in classes.named():
        pixels = int((mask & classes.mask_for(index)).sum())
        if pixels == 0:
            continue
        attributed += pixels
        share = round(100.0 * pixels / changed_px, 4)
        if share < MIN_CLASS_FRACTION_PCT:
            small.append(name)
            continue
        scalars[f"changed_{name}_pct"] = share
        if pixel_area_m2 is not None:
            scalars[f"changed_{name}_m2"] = round(pixels * pixel_area_m2, 2)

    unlabelled = changed_px - attributed
    if unlabelled:
        notes.append(
            f"{unlabelled} changed px ({100.0 * unlabelled / changed_px:.2f}%) fall "
            f"outside every segmented class and are not attributed"
        )
    if small:
        notes.append(
            f"classes below {MIN_CLASS_FRACTION_PCT}% of the changed area were not "
            f"reported: {', '.join(sorted(small))}"
        )
    return scalars, notes


def _feature_collection(regions: list[Any], payload: MaskPayload) -> dict[str, Any]:
    """Build a GeoJSON FeatureCollection of component bounding boxes in EPSG:4326.

    Geometry is emitted only when the mask is georeferenced; an unreferenced
    mask yields an empty collection rather than pixel coordinates masquerading
    as ground coordinates. Coordinates are reprojected to WGS84 because the
    contract says GeoJSON is EPSG:4326 (API_CONTRACT §2.5), and a file that
    claims to be lon/lat while carrying UTM metres is worse than no file.
    """
    transform = payload.geometry.transform
    crs = payload.geometry.crs
    features: list[dict[str, Any]] = []
    if transform is None or not crs:
        return {"type": "FeatureCollection", "features": features}

    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    for index, region in enumerate(regions):
        min_row, min_col, max_row, max_col = (int(v) for v in region.bbox)
        corners = [
            transform * (min_col, min_row),
            transform * (max_col, min_row),
            transform * (max_col, max_row),
            transform * (min_col, max_row),
        ]
        ring = [
            [round(float(lon), 6), round(float(lat), 6)]
            for lon, lat in (to_wgs84.transform(x, y) for x, y in corners)
        ]
        features.append(
            {
                "type": "Feature",
                "id": f"component_{index}",
                "geometry": {"type": "Polygon", "coordinates": [[*ring, ring[0]]]},
                "properties": {
                    "area_px": int(region.area),
                    "area_m2": (
                        round(float(region.area) * payload.pixel_area_m2, 2)
                        if payload.pixel_area_m2 is not None
                        else None
                    ),
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "source_crs": crs,
        "features": features,
    }
