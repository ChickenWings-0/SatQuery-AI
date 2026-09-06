"""``text_grounding`` — referring expressions to boxes, in pixels and on the ground.

The grounding half of mandatory requirement 2 (Master.md §8 Phase 6). The model
is asked where something is; this tool decides what "where" means to everything
downstream of it, and that is a coordinate problem rather than a language one:

* **The model answers on the canvas, not on the raster.** Qwen3-VL sees the 448 px
  rendered view, so it returns fractions of that frame. The canvas is padded to a
  square and downsampled from a scene that may be 10,000 px across, and
  :class:`~satquery.render.tiling.ViewGeometry` carries the affine for exactly
  that padded grid — which is why boxes are converted through the canvas
  transform and never through the source raster's.
* **Ground coordinates are the deliverable.** A box in canvas pixels is unusable
  outside this response; a box in WGS84 opens in QGIS over the analyst's own
  basemap. Both are emitted, because the frontend draws the first and the
  operations team files the second.
* **An unreferenced image gets pixels and no GeoJSON.** The benchmark PNG path
  has no CRS, and inventing lon/lat for it would be worse than omitting it.

The parse contract is not this module's to define: it belongs to
:mod:`satquery.models.prompts.box_format`, which the Phase 7 corpus builder also
imports. A format drift between the two produces a model that grounds perfectly
and a tool that returns nothing (DATA_ADAPTATION_PLAN §4.5).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from pyproj import Transformer

from satquery.models.loader import VlmBackend
from satquery.models.prompts import ViewInput, build_prompt
from satquery.models.prompts.box_format import BOX_SCALE, NormalisedBox, parse
from satquery.render.tiling import ViewGeometry
from satquery.schemas.enums import ArtifactType, Device, PairType, TaskType, ToolStatus
from satquery.tools.base import (
    ArtifactDraft,
    Bands,
    ImageBundle,
    MissingInputError,
    ToolContext,
    ToolResult,
)
from satquery.tools.vlm_runtime import SynthesisParams, collect_views, generate

NAME: Final[str] = "text_grounding"

MAX_BOXES: Final[int] = 20
SCORE_THRESHOLD: Final[float] = 0.30
MIN_BOX_AREA_FRACTION: Final[float] = 1e-5
"""A box under a hundred-thousandth of the frame is a decoding artefact, not a
detection: at 448 px that is under two pixels of area."""

DEFAULT_SCORE: Final[float] = 0.5
"""What an unscored box is worth. Qwen's native format carries no confidence, so
a box that arrives without one is neither promoted nor filtered out by the score
threshold — the threshold exists for checkpoints that do emit scores."""

BBOX_LABEL: Final[str] = "Grounded boxes"
GEOJSON_LABEL: Final[str] = "Grounded boxes (EPSG:4326)"

_VIEW_BANDS: Final[tuple[str, ...]] = ("red", "green", "blue")


# ------------------------------------------------------------------ coordinates


def pixel_to_wgs84(
    box: Sequence[int], geometry: ViewGeometry, transformer: Transformer | None = None
) -> list[list[float]] | None:
    """Convert a canvas pixel box into a closed WGS84 ring, or None if unreferenced.

    Args:
        box: ``(x_min, y_min, x_max, y_max)`` on the rendered canvas.
        geometry: The canvas geometry, carrying its affine and CRS.
        transformer: A reusable ``pyproj`` transformer; built per call if absent.

    Returns:
        Five ``[lon, lat]`` pairs — four corners plus the repeated first, as
        GeoJSON requires — or None when the image carries no georeference.
    """
    transform = geometry.transform
    if transform is None or not geometry.crs:
        return None
    to_wgs84 = transformer or Transformer.from_crs(geometry.crs, "EPSG:4326", always_xy=True)
    x_min, y_min, x_max, y_max = (int(v) for v in box)
    corners = [
        transform * (x_min, y_min),
        transform * (x_max, y_min),
        transform * (x_max, y_max),
        transform * (x_min, y_max),
    ]
    ring = [
        [round(float(lon), 7), round(float(lat), 7)]
        for lon, lat in (to_wgs84.transform(x, y) for x, y in corners)
    ]
    return [*ring, ring[0]]


def wgs84_to_pixel(
    ring: Sequence[Sequence[float]],
    geometry: ViewGeometry,
    transformer: Transformer | None = None,
) -> tuple[int, int, int, int]:
    """Invert :func:`pixel_to_wgs84`, recovering the canvas pixel box.

    The inverse exists so the round trip is *testable* rather than asserted:
    Master.md §8 Phase 6 requires boxes to survive pixel -> WGS84 -> pixel within
    one pixel, and a conversion with no inverse cannot be held to that.

    Raises:
        MissingInputError: The geometry carries no transform to invert.
    """
    transform = geometry.transform
    if transform is None or not geometry.crs:
        raise MissingInputError("cannot invert a box against an unreferenced canvas")
    to_native = transformer or Transformer.from_crs(
        "EPSG:4326", geometry.crs, always_xy=True
    )
    inverse = ~transform
    columns: list[float] = []
    rows: list[float] = []
    for lon, lat in ring:
        x, y = to_native.transform(float(lon), float(lat))
        col, row = inverse * (x, y)
        columns.append(col)
        rows.append(row)
    return (
        int(round(min(columns))),
        int(round(min(rows))),
        int(round(max(columns))),
        int(round(max(rows))),
    )


# ------------------------------------------------------------------- the boxes


def _select(
    boxes: Sequence[NormalisedBox], max_boxes: int, score_threshold: float
) -> list[NormalisedBox]:
    """Drop artefacts and low-confidence detections, then cap the count.

    The *cap* is applied by confidence and then area, so truncating keeps the
    detections the model was most sure of rather than the ones it happened to
    emit first. The *result* is then restored to emission order, because that is
    the order ``box_format.parse`` promises and the order the box ids are
    assigned in — a caller reading ``box_3`` off the map and ``box_3`` out of the
    GeoJSON has to get the same object, and reordering here would silently
    renumber every box whenever a cap took effect.
    """
    kept = [
        box
        for box in boxes
        if box.area / (BOX_SCALE * BOX_SCALE) >= MIN_BOX_AREA_FRACTION
        and (box.score if box.score is not None else DEFAULT_SCORE) >= score_threshold
    ]
    if len(kept) <= max_boxes:
        return kept
    ranked = sorted(
        range(len(kept)),
        key=lambda i: (
            kept[i].score if kept[i].score is not None else DEFAULT_SCORE,
            kept[i].area,
        ),
        reverse=True,
    )
    return [kept[index] for index in sorted(ranked[:max_boxes])]


def build_box_records(
    boxes: Sequence[NormalisedBox], geometry: ViewGeometry
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Turn normalised boxes into the inline BBOX_SET and the GeoJSON collection.

    Both come from one pass so a box can never appear in one and not the other:
    the frontend draws the inline record and the analyst downloads the feature,
    and the two disagreeing about how many objects were found is a bug the user
    would see before we would.
    """
    referenced = geometry.transform is not None and bool(geometry.crs)
    transformer = (
        Transformer.from_crs(geometry.crs, "EPSG:4326", always_xy=True)
        if referenced and geometry.crs
        else None
    )

    records: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        pixels = box.to_pixels(geometry.width, geometry.height)
        ring = pixel_to_wgs84(pixels, geometry, transformer)
        identifier = f"box_{index}"
        score = box.score if box.score is not None else DEFAULT_SCORE
        records.append(
            {
                "id": identifier,
                "label": box.label,
                "score": round(float(score), 4),
                "bbox_px": list(pixels),
                "bbox_normalised": [box.x_min, box.y_min, box.x_max, box.y_max],
                "bbox_wgs84": _envelope(ring),
            }
        )
        if ring is not None:
            features.append(
                {
                    "type": "Feature",
                    "id": identifier,
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "label": box.label,
                        "score": round(float(score), 4),
                        "bbox_px": list(pixels),
                    },
                }
            )

    collection: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if referenced:
        collection["source_crs"] = geometry.crs
    return records, collection


def _envelope(ring: list[list[float]] | None) -> list[float] | None:
    """``[min_lon, min_lat, max_lon, max_lat]`` for a ring — lon first, per §1."""
    if ring is None:
        return None
    lons = [point[0] for point in ring]
    lats = [point[1] for point in ring]
    return [min(lons), min(lats), max(lons), max(lats)]


# -------------------------------------------------------------------- the tool


class TextGrounding:
    """Locates a referring expression and reports it in pixels and in WGS84."""

    name = NAME

    def __init__(self, backend: VlmBackend | None = None) -> None:
        """Optionally pin a backend; production resolves the process-wide one."""
        self.backend = backend

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Ground the analyst's query and emit the boxes it resolved to.

        Raises:
            MissingInputError: The step did not resolve to one usable image.
            ToolError: No VLM backend could be reached, which the executor turns
                into the declared ``semantic_segmenter`` fallback.
        """
        image = self._image(ctx)
        bands = ctx.pixels.read(image, list(_VIEW_BANDS))
        max_boxes = int(params.get("max_boxes", MAX_BOXES))
        score_threshold = float(params.get("score_threshold", SCORE_THRESHOLD))

        text = self._generate(ctx, params, bands)
        parsed = parse(text, width=bands.geometry.width, height=bands.geometry.height)
        selected = _select(parsed, max_boxes, score_threshold)
        records, collection = build_box_records(selected, bands.geometry)

        scalars: dict[str, float | str] = {"n_boxes": len(records)}
        if records:
            scalars["mean_box_score"] = round(
                sum(float(r["score"]) for r in records) / len(records), 4
            )
            if bands.pixel_area_m2 is not None:
                areas = [
                    (r["bbox_px"][2] - r["bbox_px"][0])
                    * (r["bbox_px"][3] - r["bbox_px"][1])
                    * bands.pixel_area_m2
                    for r in records
                ]
                scalars["mean_box_area_m2"] = round(float(np.mean(areas)), 2)

        notes: list[str] = []
        if len(parsed) > len(selected):
            notes.append(
                f"{len(parsed) - len(selected)} of {len(parsed)} boxes were dropped "
                f"below the score threshold, the area floor, or the {max_boxes}-box cap"
            )
        georeferenced = bool(collection["features"]) or (
            bands.geometry.transform is not None and bool(bands.geometry.crs)
        )
        if not georeferenced and records:
            notes.append(
                "the image carries no CRS, so boxes are reported in canvas pixels "
                "only and no GeoJSON was written"
            )

        artifacts: list[Any] = [
            ArtifactDraft(
                key="boxes",
                type=ArtifactType.BBOX_SET,
                label=BBOX_LABEL,
                inline={
                    "boxes": records,
                    "frame": {
                        "width": bands.geometry.width,
                        "height": bands.geometry.height,
                    },
                    "source_image": image.id,
                },
                geometry=bands.geometry,
            )
        ]
        if georeferenced:
            artifacts.append(
                ArtifactDraft(
                    key="boxes_geojson",
                    type=ArtifactType.GEOJSON,
                    label=GEOJSON_LABEL,
                    geojson=collection,
                    geometry=bands.geometry,
                )
            )

        return ToolResult(
            status=ToolStatus.OK,
            scalars=scalars,
            artifacts=artifacts,
            data={"boxes": records},
            params={
                "max_boxes": max_boxes,
                "score_threshold": score_threshold,
                "boxes_parsed": len(parsed),
                "georeferenced": georeferenced,
            },
            confidence=0.88 if records else 0.60,
            device=Device.ROCM_0,
            notes=notes,
        )

    def _image(self, ctx: ToolContext) -> ImageBundle:
        """The image to ground on.

        A cross-modal plan grounds on the optical scene (policy entry 20): a
        referring expression like "the blue-roofed warehouse" is a statement
        about reflectance, and asking for it against a backscatter view is asking
        the model to invent.

        Raises:
            MissingInputError: No input carries the three bands a view needs.
        """
        candidates = ctx.images
        if ctx.pair_type is PairType.CROSS_MODAL:
            optical = [image for image in ctx.images if image.has(*_VIEW_BANDS)]
            candidates = optical or ctx.images
        for image in candidates:
            if image.has(*_VIEW_BANDS):
                return image
        raise MissingInputError(
            f"{NAME} needs an image resolving {', '.join(_VIEW_BANDS)} to ground on"
        )

    def _generate(
        self, ctx: ToolContext, params: Mapping[str, Any], bands: Bands
    ) -> str:
        """Ask the model where the object is, and return its raw answer.

        The generation is greedy and the raw text is kept: the boxes downstream
        are a *parse* of a specific string, and a trace that cannot show the
        string cannot explain a box that came out wrong.
        """
        synthesis = SynthesisParams.read(params, default_mode="grounding")
        views = collect_views(ctx, limit=synthesis.max_views) or [
            ViewInput(label="True colour", rgb=_as_rgb(bands))
        ]
        prompt = build_prompt(
            task=TaskType.GROUNDING,
            pair_type=ctx.pair_type,
            sheet=ctx.facts,
            views=views,
            question=ctx.question,
            slots=ctx.slots,
            prompt_version=synthesis.prompt_version,
            max_views=synthesis.max_views,
        )
        return generate(prompt, synthesis, ctx, self.backend).text


def _as_rgb(bands: Bands) -> np.ndarray:
    """A last-resort true-colour view, for a step the renderer did not feed.

    Percentile-stretched per channel rather than min-max, so one bright roof
    cannot flatten the rest of the scene into the bottom of the range.
    """
    stack = bands.stack[:3]
    out = np.zeros((*stack.shape[1:], 3), dtype=np.uint8)
    for channel in range(min(3, stack.shape[0])):
        plane = stack[channel]
        finite = plane[np.isfinite(plane) & bands.valid]
        if finite.size == 0:
            continue
        low, high = (float(v) for v in np.percentile(finite, (2.0, 98.0)))
        if high - low < 1e-9:
            continue
        scaled = np.clip((plane - low) / (high - low), 0.0, 1.0)
        out[..., channel] = (np.nan_to_num(scaled) * 255.0).astype(np.uint8)
    return out


__all__ = [
    "BBOX_LABEL",
    "GEOJSON_LABEL",
    "NAME",
    "TextGrounding",
    "build_box_records",
    "pixel_to_wgs84",
    "wgs84_to_pixel",
]
