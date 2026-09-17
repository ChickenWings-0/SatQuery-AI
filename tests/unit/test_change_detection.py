"""Phase 5: tiled inference, physical areas, and the change path end to end.

Nothing here trains anything and nothing downloads a dataset. What is under test
is the code we wrote around the model, and that splits into two halves with very
different dependencies:

* **The parts that must work everywhere** — tiling, blending, pixel-to-m²
  conversion, the overlay composite, availability gating, the FactSheet join.
  These are numpy and are tested unconditionally, which is why
  :mod:`satquery.tools.tiled_inference` takes a plain callable rather than a
  ``nn.Module``: the 2048x2048 memory assertion runs on a machine with no GPU,
  no checkpoint and no torch.
* **The model itself** — shapes, swap symmetry, checkpoint round-trip. These skip
  when torch is absent.
"""

from __future__ import annotations

import asyncio
import tracemalloc
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from affine import Affine

from satquery.agent.executor import ExecutionCache
from satquery.agent.pipeline import AnalysisRequest, analyze
from satquery.ingest.pipeline import SourceImage, ingest
from satquery.registry.registry import default_registry
from satquery.render.artifact_store import ArtifactStore
from satquery.render.overlays import CHANGE_ALPHA, CHANGE_COLOUR
from satquery.render.tiling import PadExtent, ViewGeometry
from satquery.schemas.enums import ArtifactType, PairType, ToolStatus
from satquery.tools.base import Bands, ClassMap, MaskPayload, ToolContext, ToolResult
from satquery.tools.catalog import VLM_TOOLS
from satquery.tools.change_common import basemap_for, change_drafts
from satquery.tools.change_statistics import ChangeStatistics, per_class_change
from satquery.tools.tiled_inference import (
    TilePlan,
    TilingError,
    peak_tile_bytes,
    predict_tiled,
    taper,
)

UTM43N = "EPSG:32643"


# ------------------------------------------------------------------- helpers


def _geometry(gsd_m: float, size: int, crs: str | None = UTM43N) -> ViewGeometry:
    """A canvas geometry with a known ground sample distance."""
    transform = Affine(gsd_m, 0.0, 712_340.0, 0.0, -gsd_m, 3_161_780.0) if crs else None
    return ViewGeometry(
        width=size,
        height=size,
        content_width=size,
        content_height=size,
        pad=PadExtent(0, 0, 0, 0),
        scale=1.0,
        transform=transform,
        crs=crs,
    )


def _payload(mask: np.ndarray, gsd_m: float = 10.0, crs: str | None = UTM43N) -> MaskPayload:
    geometry = _geometry(gsd_m, mask.shape[0], crs)
    pixel_area = abs(gsd_m * gsd_m) if crs else None
    return MaskPayload(mask=mask.astype(bool), pixel_area_m2=pixel_area, geometry=geometry)


def _context(
    artifacts: list[Any] | None = None, data: dict[str, Any] | None = None
) -> ToolContext:
    return ToolContext(
        trace_id="tr_cd",
        step=3,
        pair_type=PairType.BI_TEMPORAL,
        images=[],
        artifacts=artifacts or [],
        data=data or {},
    )


def _mask_input(payload: MaskPayload) -> tuple[list[Any], dict[str, Any]]:
    """A CHANGE_MASK artifact reference and its in-memory payload."""
    from satquery.schemas.trace import ArtifactRef

    reference = ArtifactRef(
        id="art_1",
        type=ArtifactType.CHANGE_MASK,
        mime="image/png",
        label="Change mask (binary)",
        produced_by_step=2,
    )
    return [reference], {"art_1": payload}


def _absolute_difference(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """A tile-invariant predictor: its whole-image result is exactly reproducible."""
    return np.abs(post - pre).mean(axis=0).astype(np.float32)


# ==================================================================== tiling


def test_a_2048_pair_is_processed_in_bounded_memory() -> None:
    """The whole point of tiling: peak memory must not scale with the raster.

    Master.md §8 Phase 5 asks for a 2048² pair. The bound asserted here is on the
    *tiling machinery*, which is what we wrote — the model's activations are
    bounded separately, by the tile size the plan hands it.
    """
    rng = np.random.default_rng(5)
    pre = rng.random((3, 2048, 2048), dtype=np.float32)
    post = rng.random((3, 2048, 2048), dtype=np.float32)
    plan = TilePlan(tile=512, overlap=64)
    assert plan.count(2048, 2048) == 25

    tracemalloc.start()
    try:
        result = predict_tiled(pre, post, _absolute_difference, plan)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.shape == (2048, 2048)
    # Two (H, W) float32 accumulators are 32 MB; everything else is one tile at a
    # time. A hundredth of a gigabyte of headroom over that, and nowhere near the
    # ~200 MB that holding every tile's output would cost.
    assert peak < 120 * 1024**2, f"tiling peaked at {peak / 1024**2:.0f} MB"


def test_blending_is_a_partition_of_unity() -> None:
    """Overlap blending must suppress seams without biasing the result.

    For a predictor that is itself tile-invariant, the tiled output has to equal
    the whole-image output exactly. Anything else means the weights do not sum to
    one somewhere, and the mask would carry a faint grid of its own.
    """
    rng = np.random.default_rng(11)
    pre = rng.random((4, 700, 900), dtype=np.float32)
    post = rng.random((4, 700, 900), dtype=np.float32)

    tiled = predict_tiled(pre, post, _absolute_difference, TilePlan(tile=256, overlap=64))
    whole = _absolute_difference(pre, post)
    assert np.allclose(tiled, whole, atol=1e-5)


def test_a_tile_edge_artefact_is_blended_away() -> None:
    """A model that misfires near its own tile edge must not leave a grid.

    This is the artefact tiling actually introduces: a pixel near a tile boundary
    is predicted from half the context, so the model over-fires there. The band is
    made narrower than the overlap on purpose — at exactly the overlap width both
    neighbours would misfire on the same columns, and no weighting can rescue
    that. The raster border is a different case and is asserted separately: there
    is no neighbour out there to blend with, so the band survives, which is the
    behaviour :func:`taper` is written to have.
    """
    plan = TilePlan(tile=128, overlap=32)
    band = 8
    pre = np.zeros((1, 128, 512), dtype=np.float32)

    def edgy(pre_tile: np.ndarray, post_tile: np.ndarray) -> np.ndarray:
        """Correct (0.0) in its interior, wrong (1.0) within *band* of its edge."""
        out = np.zeros(pre_tile.shape[1:], dtype=np.float32)
        out[:, :band] = 1.0
        out[:, -band:] = 1.0
        return out

    blended = predict_tiled(pre, pre, edgy, plan)
    row = blended[64]

    interior = row[plan.overlap : -plan.overlap]
    assert interior.max() < 0.25, "a tile boundary left a visible artefact"
    # ...and that span really does contain tile edges, so the naive stack would
    # have shown a grid across it.
    origins = [w.col for w in plan.windows(128, 512)][:5]
    assert [o for o in origins if plan.overlap < o < 512 - plan.overlap]

    # ...and the outermost columns, which border nothing, keep the model's answer.
    assert row[0] == pytest.approx(1.0)
    assert row[-1] == pytest.approx(1.0)


def test_the_last_tile_is_shifted_flush_rather_than_zero_padded() -> None:
    """Padding the edge feeds the model a synthetic boundary it reads as change."""
    plan = TilePlan(tile=256, overlap=32)
    windows = list(plan.windows(600, 600))
    assert all(w.height == 256 and w.width == 256 for w in windows)
    assert max(w.row + w.height for w in windows) == 600
    assert max(w.col + w.width for w in windows) == 600


def test_tiles_smaller_than_the_raster_still_cover_it_exactly() -> None:
    """Every pixel must be written by at least one tile, at every odd size."""
    for height, width in ((100, 100), (513, 129), (1, 1000), (2048, 7)):
        plan = TilePlan(tile=256, overlap=64)
        coverage = np.zeros((height, width), dtype=np.int32)
        for window in plan.windows(height, width):
            coverage[window.rows, window.cols] += 1
        assert coverage.min() >= 1, f"{height}x{width} left a gap"


def test_the_raster_border_is_not_tapered_against_a_neighbour_it_does_not_have() -> None:
    """Tapering the outermost rows would leave a faint frame around every scene."""
    plan = TilePlan(tile=128, overlap=32)
    corner = next(plan.windows(400, 400))
    assert corner.touches_top and corner.touches_left
    weights = taper(corner, plan.overlap)
    assert weights[0, 0] == pytest.approx(1.0), "the raster corner was tapered"
    assert weights[-1, -1] < 0.2, "the interior edge was not tapered"


def test_an_overlap_at_least_the_tile_size_is_refused() -> None:
    """It would make the stride zero, and the window would never advance."""
    with pytest.raises(TilingError):
        TilePlan(tile=64, overlap=64)
    with pytest.raises(TilingError):
        TilePlan(tile=0, overlap=0)


def test_a_predictor_returning_the_wrong_shape_is_caught_at_the_tile() -> None:
    """Silently broadcasting a wrong-shaped tile would corrupt the whole mask."""
    pre = np.zeros((1, 64, 64), dtype=np.float32)
    with pytest.raises(TilingError):
        predict_tiled(pre, pre, lambda a, b: np.zeros((8, 8), np.float32), TilePlan(32, 8))


def test_mismatched_epochs_are_refused_rather_than_broadcast() -> None:
    with pytest.raises(TilingError):
        predict_tiled(
            np.zeros((3, 64, 64), np.float32),
            np.zeros((3, 32, 32), np.float32),
            _absolute_difference,
        )


def test_the_bounded_tile_footprint_is_reported() -> None:
    """A tiled run should be able to say what it was tiled to."""
    assert peak_tile_bytes(TilePlan(512, 64), channels=3) == 2 * 3 * 512 * 512 * 4


# ========================================================= physical areas


def test_pixel_area_becomes_square_metres_through_the_affine_transform() -> None:
    """The headline number is an area on the ground, not a pixel count.

    A 10 m grid means each pixel is exactly 100 m². Four hundred changed pixels
    are 40,000 m² and 0.04 km², and both have to appear in the FactSheet at a
    precision the CitationValidator can resolve.
    """
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True  # 400 px
    artifacts, data = _mask_input(_payload(mask, gsd_m=10.0))

    result = ChangeStatistics().run(_context(artifacts, data), {"min_component_px": 25})

    assert result.scalars["changed_pixel_count"] == 400
    assert result.scalars["changed_area_m2"] == pytest.approx(40_000.0)
    assert result.scalars["changed_area_km2"] == pytest.approx(0.04)
    assert result.scalars["changed_area_pct"] == pytest.approx(4.0)
    assert result.scalars["largest_component_m2"] == pytest.approx(40_000.0)


def test_a_half_metre_grid_gives_a_quarter_square_metre_per_pixel() -> None:
    """The conversion must follow the transform, not a hard-coded resolution."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[:8, :8] = True  # 64 px at 0.5 m -> 0.25 m2 each
    artifacts, data = _mask_input(_payload(mask, gsd_m=0.5))

    result = ChangeStatistics().run(_context(artifacts, data), {"min_component_px": 25})
    assert result.scalars["changed_area_m2"] == pytest.approx(16.0)


def test_an_unreferenced_mask_reports_no_area_rather_than_a_wrong_one() -> None:
    """Pixels are not metres, and a PNG with no CRS cannot be told how big it is."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[:8, :8] = True
    artifacts, data = _mask_input(_payload(mask, crs=None))

    result = ChangeStatistics().run(_context(artifacts, data), {})
    assert result.scalars["changed_pixel_count"] == 64
    assert "changed_area_m2" not in result.scalars


def test_speckle_below_the_component_floor_is_excluded_from_the_area() -> None:
    """Single-pixel differences are co-registration residual, not change."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    rng = np.random.default_rng(3)
    rows, cols = rng.integers(60, 99, size=(2, 40))
    mask[rows, cols] = True

    artifacts, data = _mask_input(_payload(mask, gsd_m=10.0))
    result = ChangeStatistics().run(_context(artifacts, data), {"min_component_px": 25})

    assert result.scalars["changed_pixel_count"] == 400
    assert result.scalars["component_count"] == 1
    assert any("dropped" in note for note in result.notes)


# ==================================================== per-class attribution


def _class_map(labels: np.ndarray, classes: tuple[str, ...], gsd_m: float = 10.0) -> ClassMap:
    return ClassMap(
        labels=labels.astype(np.int16),
        classes=classes,
        geometry=_geometry(gsd_m, labels.shape[0]),
        pixel_area_m2=gsd_m * gsd_m,
    )


def test_changed_pixels_are_attributed_to_the_class_they_became() -> None:
    """"7.4 % changed" is a fact; "became built-up" is the fact anyone wanted."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[:20, :] = True  # 800 px
    labels = np.full((40, 40), -1, dtype=np.int16)
    labels[:10, :] = 0  # 400 px built_up
    labels[10:20, :] = 1  # 400 px water

    scalars, notes = per_class_change(
        mask, _class_map(labels, ("built_up", "water")), pixel_area_m2=100.0, changed_px=800
    )
    assert scalars["changed_built_up_pct"] == pytest.approx(50.0)
    assert scalars["changed_water_pct"] == pytest.approx(50.0)
    assert scalars["changed_built_up_m2"] == pytest.approx(40_000.0)
    assert notes == []


def test_changed_pixels_outside_every_class_are_reported_not_absorbed() -> None:
    """Attributing unlabelled ground to class 0 would invent a land-cover claim."""
    mask = np.ones((20, 20), dtype=bool)
    labels = np.full((20, 20), -1, dtype=np.int16)
    labels[:10, :] = 0

    scalars, notes = per_class_change(
        mask, _class_map(labels, ("built_up",)), pixel_area_m2=100.0, changed_px=400
    )
    assert scalars["changed_built_up_pct"] == pytest.approx(50.0)
    assert any("not attributed" in note for note in notes)


def test_a_segmentation_on_a_different_canvas_is_refused() -> None:
    """Joining two grids by index would attribute every pixel to the wrong place."""
    from satquery.tools.base import MissingInputError

    mask = np.ones((20, 20), dtype=bool)
    labels = np.zeros((10, 10), dtype=np.int16)
    with pytest.raises(MissingInputError):
        per_class_change(mask, _class_map(labels, ("built_up",)), 100.0, 400)


def test_without_a_segmentation_no_per_class_number_is_produced() -> None:
    """Per-class change is a join with a label map, never a guess from the mask."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    artifacts, data = _mask_input(_payload(mask))

    result = ChangeStatistics().run(_context(artifacts, data), {})
    assert result.params["per_class"] is False
    assert not [key for key in result.scalars if str(key).startswith("changed_built")]


def test_every_per_class_scalar_is_accepted_by_the_registry_schema() -> None:
    """A scalar the registry does not declare is dropped, never cited."""
    from satquery.evidence.fact_sheet import validate_scalar

    schema = default_registry()["change_statistics"].scalars_schema
    for name in ("changed_built_up_pct", "changed_water_m2", "changed_bare_soil_pct"):
        assert validate_scalar(name, 12.5, schema) is None
    assert validate_scalar("changed_built_up_pct", "lots", schema) is not None


# ============================================================ the overlay


def _bands(size: int = 32, gsd_m: float = 10.0) -> Bands:
    stack = np.full((3, size, size), 0.5, dtype=np.float32)
    return Bands(
        order=("red", "green", "blue"),
        stack=stack,
        valid=np.ones((size, size), dtype=bool),
        geometry=_geometry(gsd_m, size),
    )


class _StubImage:
    """The bits of an ImageBundle the artifact builder actually reads."""

    id = "img_1"
    modality = None

    def has(self, *bands: str) -> bool:
        return set(bands) <= {"red", "green", "blue"}


def test_the_change_overlay_is_red_at_45_percent_over_the_true_colour_post_view() -> None:
    """DATA_ADAPTATION_PLAN §2.1 freezes the CHANGE view's colour and alpha."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:16, 8:16] = True
    bands = _bands()

    drafts, payload = change_drafts(mask, _StubImage(), bands, {"method": "siamese"})
    mask_draft, overlay_draft = drafts

    assert mask_draft.type is ArtifactType.CHANGE_MASK
    assert overlay_draft.type is ArtifactType.OVERLAY_PNG
    assert overlay_draft.stats is not None
    assert overlay_draft.stats["alpha"] == CHANGE_ALPHA == 0.45

    base = basemap_for(_StubImage(), bands)
    overlay = overlay_draft.image
    assert overlay is not None
    unchanged = ~mask
    assert np.array_equal(overlay[unchanged], base[unchanged]), "untouched pixels moved"

    expected = base[mask].astype(np.float32) * (1 - CHANGE_ALPHA) + np.array(
        CHANGE_COLOUR, dtype=np.float32
    ) * CHANGE_ALPHA
    assert np.allclose(overlay[mask], expected.round(), atol=1.0)
    assert overlay[mask][:, 0].min() > base[mask][:, 0].max(), "the tint is not red"


def test_the_mask_artifact_carries_a_georeferenced_companion() -> None:
    """A mask that does not reopen over its own imagery in QGIS is not evidence."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[:4, :4] = True
    drafts, payload = change_drafts(mask, _StubImage(), _bands(), {})

    assert drafts[0].raster is not None
    assert drafts[0].geometry is not None
    assert drafts[0].geometry.crs == UTM43N
    assert payload.pixel_area_m2 == pytest.approx(100.0)


def test_both_detectors_produce_the_same_artifact_shape() -> None:
    """The executor may substitute one for the other; the gallery must not change."""
    from satquery.tools import change_common

    assert change_common.MASK_LABEL == "Change mask (binary)"
    assert change_common.OVERLAY_LABEL == "Change overlaid on the post-change view"


# ========================================================== availability


def test_the_detector_is_advertised_only_where_a_checkpoint_exists() -> None:
    """A capabilities panel promising an untrained model is worse than a gap."""
    from satquery.tools.catalog import change_detector_servable, runnable_tools

    trained = change_detector_servable()
    assert ("siamese_change_detector" in runnable_tools()) is trained

    spec = default_registry()["siamese_change_detector"]
    assert spec.available is trained
    if not trained:
        assert spec.unavailable_reason
        assert "image_diff_change" in spec.unavailable_reason
        assert spec.fallback == "image_diff_change"


def test_an_untrained_machine_still_answers_change_questions(
    scene_paths: dict[str, Path], tmp_path: Path
) -> None:
    """Capability matching substitutes the classical baseline, visibly."""
    sources = [
        SourceImage(path=scene_paths[name], filename=scene_paths[name].name)
        for name in ("s2_pre", "s2_post")
    ]
    result = ingest(sources)
    analysis = asyncio.run(
        analyze(
            AnalysisRequest(
                query="what changed between these two images",
                sources=sources,
                ingest=result,
            ),
            store=ArtifactStore(tmp_path / "artifacts"),
            registry=default_registry().with_availability(
                {"siamese_change_detector": False}
            ),
            cache=ExecutionCache(),
        )
    )
    trace = analysis.trace
    detector = next(e for e in trace.executions if e.step == 2)
    assert detector.tool == "image_diff_change"
    assert detector.fallback_of == "siamese_change_detector"
    assert detector.status is ToolStatus.DEGRADED
    assert trace.fact_sheet["change_statistics.changed_area_m2"] > 0


def test_a_trained_detector_feeds_real_areas_into_the_fact_sheet(
    scene_paths: dict[str, Path], tmp_path: Path
) -> None:
    """The whole Phase 5 chain: detector -> mask -> components -> m2 -> answer."""

    class StubDetector:
        """Stands in for the checkpoint, exercising every step around it."""

        name = "siamese_change_detector"

        def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
            pre, post = ctx.pair()
            stack = ctx.pixels.read(post, ["red", "green", "blue"])
            mask = np.zeros(stack.valid.shape, dtype=bool)
            mask[: stack.valid.shape[0] // 4, :] = True
            mask &= stack.valid
            drafts, payload = change_drafts(
                mask, post, stack, {"method": "siamese", "threshold": 0.5}
            )
            return ToolResult(
                scalars={
                    "changed_area_pct": round(
                        100.0 * float(mask.sum()) / float(stack.valid.sum()), 4
                    ),
                    "changed_pixel_count": int(mask.sum()),
                    "threshold": 0.5,
                    "mean_change_logit": 0.42,
                },
                artifacts=drafts,
                data={"change_mask": payload},
                confidence=0.9,
            )

    sources = [
        SourceImage(path=scene_paths[name], filename=scene_paths[name].name)
        for name in ("s2_pre", "s2_post")
    ]
    result = ingest(sources)
    analysis = asyncio.run(
        analyze(
            AnalysisRequest(
                query="what changed between these two images",
                sources=sources,
                ingest=result,
            ),
            store=ArtifactStore(tmp_path / "artifacts"),
            # The detector is forced on because this test supplies a stub for it;
            # the VLM tools are forced off for the same reason the contract tests
            # pin them off — with the Qwen weights on the machine, the answer
            # below becomes a live generation, and the "every number is bound to
            # a measurement" assertion at the end starts describing the model's
            # prose rather than our templating.
            registry=default_registry().with_availability(
                {"siamese_change_detector": True, **dict.fromkeys(VLM_TOOLS, False)}
            ),
            tools={"siamese_change_detector": StubDetector()},
            cache=ExecutionCache(),
        )
    )
    trace = analysis.trace

    detector = next(e for e in trace.executions if e.tool == "siamese_change_detector")
    assert detector.status is ToolStatus.OK
    assert trace.fact_sheet["siamese_change_detector.changed_area_pct"] > 0
    assert trace.fact_sheet["change_statistics.changed_area_m2"] > 0
    assert trace.fact_sheet["change_statistics.changed_area_km2"] > 0

    # The mask and its overlay both reached the evidence gallery.
    types = {a.type for a in trace.artifacts}
    assert ArtifactType.CHANGE_MASK in types
    assert ArtifactType.OVERLAY_PNG in types

    # Every number in the answer is still bound to a measurement.
    assert trace.answer.uncited_numeric_spans == []
    assert trace.answer.citations


# =============================================================== the model


torch = pytest.importorskip("torch", reason="the model needs torch; `uv sync --extra cd`")


def test_the_detector_is_symmetric_in_its_two_epochs() -> None:
    """Absolute differencing means swapping the epochs cannot change the answer.

    Concatenation would let the model learn acquisition order as a shortcut, which
    on 637 training tiles it reliably does.

    Run in float64, on the CPU, from a forked RNG seeded here.

    The symmetry is exact — in float64 the two orderings agree to the last bit —
    so this asserts the architectural property rather than a tolerance. In
    float32 the same comparison is only approximate, and *how* approximate
    depends on which weights were drawn: most initialisations agree exactly,
    while some (seed 0, for one) differ by ~1.2e-3. An unseeded test therefore
    passed or failed on the luck of the draw, which says nothing about whether
    the epochs are handled symmetrically.

    The RNG is forked and the default device forced, because both are global: by
    the time this runs, another test may have loaded a model and left the
    default device or the generator somewhere else, and the initialisation this
    test believes it pinned would quietly be a different one.
    """
    from satquery.training.cd.model import build_detector

    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(42)
        model = build_detector(pretrained=False).double().eval()
        pre = torch.rand(2, 3, 128, 128, dtype=torch.float64)
        post = torch.rand(2, 3, 128, 128, dtype=torch.float64)
        with torch.no_grad():
            forward, reversed_ = model(pre, post), model(post, pre)

    assert torch.allclose(forward, reversed_, atol=1e-9)


def test_the_logits_come_back_on_the_grid_they_went_in_on() -> None:
    """Including sizes that are not a multiple of the 32-pixel deepest stride."""
    from satquery.training.cd.model import build_detector

    model = build_detector(pretrained=False).eval()
    for height, width in ((128, 128), (150, 201), (64, 320)):
        with torch.no_grad():
            logits = model(torch.rand(1, 3, height, width), torch.rand(1, 3, height, width))
        assert logits.shape == (1, 1, height, width)


def test_mismatched_epochs_are_refused_by_the_model_too() -> None:
    from satquery.training.cd.model import build_detector

    model = build_detector(pretrained=False).eval()
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 32, 32))


def test_the_stem_keeps_each_pretrained_filter_on_its_own_wavelength() -> None:
    """Averaging all 13 filters into each channel destroys spectral selectivity."""
    from satquery.training.cd.encoder import SENTINEL2_BAND_INDEX, adapt_stem

    # Each pretrained input channel is filled with its own band index, so an
    # adapted filter's value says which wavelength it came from.
    weight = torch.zeros(8, 13, 3, 3)
    for index in SENTINEL2_BAND_INDEX.values():
        weight[:, index] = float(index)

    adapted = adapt_stem(weight, 13, ("red", "green", "blue"))
    assert adapted.shape == (8, 3, 3, 3)
    scale = 13 / 3
    for channel, band in enumerate(("red", "green", "blue")):
        expected = SENTINEL2_BAND_INDEX[band] * scale
        assert torch.allclose(adapted[:, channel], torch.full((8, 3, 3), expected))


def test_a_band_the_pretraining_never_saw_falls_back_only_for_itself() -> None:
    """A panchromatic channel gets the mean filter; the others keep their own."""
    from satquery.training.cd.encoder import adapt_stem

    weight = torch.arange(13, dtype=torch.float32).reshape(1, 13, 1, 1).repeat(4, 1, 3, 3)
    adapted = adapt_stem(weight, 13, ("red", "pan"))
    scale = 13 / 2
    assert torch.allclose(adapted[:, 0], torch.full((4, 3, 3), 3.0 * scale))
    assert torch.allclose(adapted[:, 1], torch.full((4, 3, 3), 6.0 * scale))


def test_the_loss_is_not_satisfied_by_predicting_no_change_anywhere() -> None:
    """Plain BCE is; that solution scores 95 % accuracy and an F1 of zero."""
    from satquery.training.cd.module import dice_loss

    target = torch.zeros(1, 1, 32, 32)
    target[:, :, :4, :4] = 1.0  # 1.5 % changed
    all_negative = torch.full((1, 1, 32, 32), -10.0)
    perfect = torch.where(target > 0, 10.0, -10.0)

    assert float(dice_loss(all_negative, target)) > 0.9
    assert float(dice_loss(perfect, target)) < 0.05


def test_the_threshold_is_calibrated_rather_than_left_at_one_half() -> None:
    """On class-imbalanced change data the F1 optimum is not where the loss is."""
    from satquery.training.cd.module import calibrate_threshold

    target = torch.zeros(1, 1, 64, 64)
    target[:, :, :8, :] = 1.0
    probabilities = torch.where(target > 0, 0.30, 0.05)

    threshold, counts = calibrate_threshold(probabilities, target)
    assert threshold < 0.5
    assert counts.f1 == pytest.approx(1.0)


def test_a_checkpoint_round_trips_with_everything_needed_to_serve_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weights alone are not servable: the normalisation and threshold travel too."""
    from satquery.training.cd.checkpoint import (
        BUNDLE_SUFFIX,
        CHECKPOINT_ENV,
        CheckpointBundle,
        Normalisation,
        find_checkpoint,
        load_bundle,
        save_bundle,
    )

    # `find_checkpoint` honours the env override first; a developer `.env` that
    # an earlier `create_app()` copied into os.environ must not win over tmp_path.
    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    from satquery.training.cd.model import SiameseConfig, build_detector

    model = build_detector(pretrained=False)
    bundle = CheckpointBundle(
        config=SiameseConfig(pretrained=False).to_dict(),
        state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        normalisation=Normalisation(mean=(0.2, 0.3, 0.4), std=(0.1, 0.1, 0.1)),
        threshold=0.31,
        dataset="levircd",
        gsd_m=0.5,
        metrics={"f1": 0.9},
        encoder_source="ssl4eo-s12",
    )
    path = save_bundle(bundle, tmp_path / f"levircd{BUNDLE_SUFFIX}")

    restored = load_bundle(path)
    assert restored.threshold == pytest.approx(0.31)
    assert restored.normalisation.mean == (0.2, 0.3, 0.4)
    assert restored.gsd_m == 0.5
    assert restored.version == "1.0.0+ckpt:levircd@0.5m"
    assert path.with_suffix(".json").is_file(), "the provenance sidecar was not written"

    rebuilt = build_detector(pretrained=False)
    rebuilt.load_state_dict(restored.state_dict, strict=True)
    assert find_checkpoint(tmp_path) == path


def test_a_zero_standard_deviation_is_refused_rather_than_dividing_by_zero() -> None:
    from satquery.training.cd.checkpoint import CheckpointError, Normalisation

    with pytest.raises(CheckpointError):
        Normalisation(mean=(0.0, 0.0), std=(1.0, 0.0))
    with pytest.raises(CheckpointError):
        Normalisation(mean=(0.0, 0.0), std=(1.0,))


def test_a_file_that_is_not_a_bundle_is_refused_with_a_readable_reason(
    tmp_path: Path,
) -> None:
    from satquery.training.cd.checkpoint import CheckpointError, load_bundle

    path = tmp_path / "not-a-bundle.ckpt.pt"
    torch.save({"weights": 1}, path)
    with pytest.raises(CheckpointError, match="not a SatQuery"):
        load_bundle(path)
    with pytest.raises(CheckpointError, match="no checkpoint"):
        load_bundle(tmp_path / "absent.ckpt.pt")


def test_the_torchgeo_batch_layouts_both_reduce_to_one_contract() -> None:
    """Torchgeo has shipped separate epochs and a stacked tensor; both must work."""
    from satquery.training.cd.data import DataError, normalise_batch

    separate = normalise_batch(
        {
            "image1": torch.rand(2, 3, 16, 16),
            "image2": torch.rand(2, 3, 16, 16),
            "mask": torch.zeros(2, 16, 16),
        }
    )
    stacked = normalise_batch(
        {"image": torch.rand(2, 2, 3, 16, 16), "mask": torch.zeros(2, 16, 16)}
    )
    for batch in (separate, stacked):
        assert batch["image1"].shape == (2, 3, 16, 16)
        assert batch["mask"].shape == (2, 1, 16, 16)

    with pytest.raises(DataError):
        normalise_batch({"mask": torch.zeros(2, 16, 16)})


def test_constructing_a_datamodule_downloads_nothing(tmp_path: Path) -> None:
    """A dataset download is an explicit act, never a side effect of a test."""
    from satquery.training.cd.data import ChangeDataModule

    module = ChangeDataModule(dataset="levircd", root=tmp_path)
    assert module.download is False
    assert module.profile.gsd_m == 0.5
    assert not any(tmp_path.iterdir())

    coarse = ChangeDataModule(dataset="oscd", root=tmp_path)
    assert coarse.profile.gsd_m == 10.0
    assert coarse.profile.bands == module.profile.bands, (
        "the two ablation rows must differ in resolution and nothing else"
    )


# --------------------------------------------------- known-answer load check


def _bundle_with_probe(fraction: float = 0.05) -> Any:
    """A minimal bundle carrying a self-test, for the checker alone."""
    from satquery.training.cd.checkpoint import CheckpointBundle, Normalisation

    rng = np.random.default_rng(0)
    pre = rng.integers(60, 90, (3, 64, 64)).astype(np.uint8)
    post = pre.copy()
    post[:, 16:48, 16:48] = 235
    return CheckpointBundle(
        config={},
        state_dict={},
        normalisation=Normalisation.identity(3),
        threshold=0.9,
        self_test={
            "pre": pre,
            "post": post,
            "changed_fraction": fraction,
            "tolerance": 0.02,
        },
    )


class _FakeLoaded:
    """A LoadedModel stand-in whose predictions the test dictates."""

    def __init__(self, bundle: Any, output: Any) -> None:
        self.bundle = bundle
        self.device = "cpu"
        self._output = output

    def predict(self, pre: Any, post: Any) -> Any:
        if callable(self._output):
            return self._output(pre, post)
        return np.full(pre.shape[1:], self._output, dtype=np.float32)


def test_a_healthy_load_reproduces_its_own_known_answer() -> None:
    """The check must pass the model it was measured from."""
    from satquery.tools.change_detect import _self_check

    bundle = _bundle_with_probe(fraction=1.0)
    _self_check(_FakeLoaded(bundle, 0.99))  # every pixel changed, as recorded


def test_a_corrupt_load_is_refused_rather_than_served() -> None:
    """The failure that motivated this: weights that survive the load broken.

    A corrupt detector marked a third of every scene changed for the life of the
    process, and produced a mask, a percentage and a fluent sentence quoting it.
    Nothing downstream could tell the number was invented.
    """
    from satquery.tools.change_detect import DetectorUnavailableError, _self_check

    bundle = _bundle_with_probe(fraction=0.05)
    with pytest.raises(DetectorUnavailableError, match="did not survive the load"):
        _self_check(_FakeLoaded(bundle, 0.99))


def test_a_dead_load_is_refused_too() -> None:
    """The other observed failure: confident, uniform, wrong no-change."""
    from satquery.tools.change_detect import DetectorUnavailableError, _self_check

    bundle = _bundle_with_probe(fraction=0.50)
    with pytest.raises(DetectorUnavailableError):
        _self_check(_FakeLoaded(bundle, 0.0))


def test_non_finite_probabilities_are_refused() -> None:
    """NaN output appeared in roughly one load in ten on this card."""
    from satquery.tools.change_detect import DetectorUnavailableError, _self_check

    bundle = _bundle_with_probe()
    with pytest.raises(DetectorUnavailableError, match="non-finite"):
        _self_check(_FakeLoaded(bundle, np.nan))


def test_a_checkpoint_without_a_probe_is_served_unverified() -> None:
    """A bundle predating the probe must still load; refusing it would be worse."""
    from satquery.tools.change_detect import _self_check
    from satquery.training.cd.checkpoint import CheckpointBundle, Normalisation

    legacy = CheckpointBundle(
        config={}, state_dict={}, normalisation=Normalisation.identity(3), threshold=0.9
    )
    _self_check(_FakeLoaded(legacy, 0.99))


def test_the_cache_retries_a_corrupt_load_and_serves_the_good_one() -> None:
    """Corruption is per-load, so rebuilding is a fix rather than a hope.

    Detection alone would degrade one demo run in eight; the retry is what makes
    the fault invisible instead of merely honest.
    """
    from satquery.tools.change_detect import ModelCache

    bundle = _bundle_with_probe(fraction=1.0)
    attempts: list[int] = []

    cache = ModelCache()
    def flaky(path: Any, device: str) -> Any:
        attempts.append(1)
        # First two loads come up corrupt, the third is clean.
        return _FakeLoaded(bundle, 0.0 if len(attempts) < 3 else 0.99)

    cache._load = flaky  # type: ignore[method-assign]
    served = cache.get(Path("unused.ckpt.pt"), "cpu")

    assert len(attempts) == 3
    assert served.predict(np.zeros((3, 8, 8), np.float32), None).max() > 0.9


def test_a_permanently_corrupt_detector_is_reported_not_served() -> None:
    """When every attempt fails, the step degrades visibly instead of lying."""
    from satquery.tools.change_detect import (
        SELF_CHECK_ATTEMPTS,
        DetectorUnavailableError,
        ModelCache,
    )

    bundle = _bundle_with_probe(fraction=1.0)
    attempts: list[int] = []
    cache = ModelCache()
    def always_bad(path: Any, device: str) -> Any:
        attempts.append(1)
        return _FakeLoaded(bundle, 0.0)

    cache._load = always_bad  # type: ignore[method-assign]
    with pytest.raises(DetectorUnavailableError, match="all 4 load attempts"):
        cache.get(Path("unused.ckpt.pt"), "cpu")
    assert len(attempts) == SELF_CHECK_ATTEMPTS
