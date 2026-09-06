"""Phase 6: cross-modal optical/SAR reasoning, and grounding that lands on the ground.

Two mandatory requirements meet here, and each has one claim that has to be true
for the rest of the phase to mean anything:

* **Requirement 4** is answered by the physics layer, and its claim is that radar
  resolves a land-cover confusion *optical alone gets wrong*. The bare-soil /
  built-up case is not an illustration of that — it is the whole demonstration,
  so the tests below run the real ``spectral_index_analyzer`` over the same
  pixels and assert it reaches the wrong answer before asserting the fusion
  layer reaches the right one. A test that only checked the fusion output would
  pass just as happily if the confusion did not exist.
* **Requirement 2's grounding option** is answered by boxes, and its claim is
  that a box means a place on the Earth rather than a place in a PNG. Master.md
  §8 Phase 6 fixes the tolerance at one pixel for the pixel -> WGS84 -> pixel
  round trip, so that is asserted against a real UTM transform and a real pyproj
  reprojection, not a mocked one.

No weights are loaded anywhere in this file. DOFA and SegFormer reach their tools
through injected encoders, and the VLM through a scripted backend — which is
possible because all three tools take that seam as a constructor argument, and is
why the cross-modal and grounding paths are testable on a machine with no GPU.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from affine import Affine

from satquery.agent.executor import ExecutionCache
from satquery.agent.pipeline import AnalysisRequest, analyze
from satquery.agent.planner import PLANNER_VERSION, default_table, resolve_key
from satquery.ingest.pipeline import SourceImage, ingest
from satquery.models.loader import (
    BackendConfig,
    BackendKind,
    GenerationRequest,
    GenerationResult,
    LazyBackend,
)
from satquery.models.prompts.box_format import (
    BOX_SCALE,
    BoxFormatError,
    NormalisedBox,
    from_pixels,
    parse,
    serialise,
)
from satquery.registry.registry import default_registry
from satquery.render.artifact_store import ArtifactStore
from satquery.render.tiling import PadExtent, ViewGeometry
from satquery.schemas.enums import (
    ArtifactType,
    ImageRole,
    Modality,
    PairType,
    TaskType,
    ToolStatus,
)
from satquery.schemas.manifest import InputManifest
from satquery.tools.base import (
    Bands,
    ClassMap,
    ImageBundle,
    MissingInputError,
    PixelReader,
    ToolContext,
    ToolError,
)
from satquery.tools.crossmodal_dofa import (
    WAVELENGTH_UM,
    CrossModalConsistency,
    cosine_agreement,
    upsample,
)
from satquery.tools.object_counter import ObjectCounter
from satquery.tools.physics_agreement import (
    NDVI_VEGETATED,
    SIGMA0_BUILT_UP_DB,
    PhysicsAgreement,
    resolve_confusion,
)
from satquery.tools.semantic_segmenter import (
    CANONICAL_CLASSES,
    LOVEDA_CLASSES,
    SegmenterUnavailableError,
    SemanticSegmenter,
    class_boxes,
    to_canonical,
)
from satquery.tools.spectral_index_analyzer import SpectralIndexAnalyzer
from satquery.tools.text_grounding import TextGrounding, pixel_to_wgs84, wgs84_to_pixel

UTM43N = "EPSG:32643"
CANVAS_PX = 64


# ============================================================ surface signatures
#
# Reflectances and backscatter for four surfaces, chosen so each one's indices
# land unambiguously on the right side of the tool's frozen thresholds. The two
# middle rows are the point: they are nearly identical in every optical band and
# eleven decibels apart in radar.

BARE_SOIL = {"red": 0.30, "green": 0.28, "nir": 0.34, "swir1": 0.42, "vv": -14.0, "vh": -21.0}
BUILT_UP = {"red": 0.28, "green": 0.27, "nir": 0.30, "swir1": 0.38, "vv": -3.0, "vh": -11.0}
VEGETATION = {"red": 0.05, "green": 0.09, "nir": 0.45, "swir1": 0.20, "vv": -9.0, "vh": -12.5}
WATER = {"red": 0.03, "green": 0.08, "nir": 0.02, "swir1": 0.01, "vv": -22.0, "vh": -28.0}

OPTICAL_BANDS = ("red", "green", "nir", "swir1")
SAR_BANDS = ("vv", "vh")


# ------------------------------------------------------------------- test doubles


def _geometry(size: int = CANVAS_PX, gsd_m: float = 10.0, crs: str | None = UTM43N) -> ViewGeometry:
    """A canvas geometry with a known transform, north-up on a UTM grid."""
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


def _manifest(
    identifier: str, modality: Modality, role: ImageRole, bands: Sequence[str]
) -> InputManifest:
    """A manifest carrying just what the tools under test read off it."""
    return InputManifest(
        id=identifier,
        role=role,
        filename=f"{identifier}.tif",
        sha256="0" * 64,
        size_bytes=1024,
        driver="GTiff",
        modality=modality,
        modality_confidence=0.99,
        sensor_guess="Sentinel-2 L2A" if modality is Modality.OPTICAL else "Sentinel-1 GRD",
        crs=UTM43N,
        transform=[10.0, 0.0, 712_340.0, 0.0, -10.0, 3_161_780.0],
        bounds_native=[712_340.0, 3_161_140.0, 712_980.0, 3_161_780.0],
        bounds_wgs84=[77.0, 28.5, 77.01, 28.51],
        gsd_m=10.0,
        width=CANVAS_PX,
        height=CANVAS_PX,
        band_count=len(bands),
        dtype="float32",
        band_names=list(bands),
        nodata_pct=0.0,
        acquisition_time=datetime(2024, 3, 1, tzinfo=UTC),
        is_georeferenced=True,
    )


def _bundle(
    identifier: str, modality: Modality, role: ImageRole, bands: Sequence[str]
) -> ImageBundle:
    """An image bundle resolving exactly *bands*."""
    return ImageBundle(
        manifest=_manifest(identifier, modality, role, bands),
        path=Path(f"/nonexistent/{identifier}.tif"),
        resolved_bands={name: index for index, name in enumerate(bands, start=1)},
    )


class _ScriptedReader(PixelReader):
    """A pixel reader serving arrays a test designed, rather than a raster.

    The physics rules are per-pixel thresholds on indices, so the only way to
    assert what they do at a decision boundary is to place pixels exactly where
    the test intends. Reading a synthetic GeoTIFF back through rasterio would put
    resampling and dtype quantisation between the intent and the assertion.
    """

    def __init__(self, plans: Mapping[str, np.ndarray], size: int = CANVAS_PX) -> None:
        """Bind each image id to a ``(size, size)`` map of surface indices."""
        super().__init__(size=size)
        self.plans = plans
        self.size = size

    def read(
        self, image: ImageBundle, bands: Sequence[str], size: int | None = None
    ) -> Bands:
        """Compose the requested bands from the surface map for this image."""
        missing = image.missing(bands)
        if missing:
            raise MissingInputError(f"{image.id} lacks {missing}")
        order = tuple(dict.fromkeys(bands))
        plan = self.plans[image.id]
        stack = np.stack([_paint(plan, name) for name in order]).astype(np.float32)
        return Bands(
            order=order,
            stack=stack,
            valid=np.ones(plan.shape, dtype=bool),
            geometry=_geometry(self.size),
        )


SURFACES = (BARE_SOIL, BUILT_UP, VEGETATION, WATER)


def _paint(plan: npt.NDArray[np.int_], band: str) -> npt.NDArray[np.float32]:
    """Turn a map of surface indices into one band's values."""
    out = np.zeros(plan.shape, dtype=np.float32)
    for index, surface in enumerate(SURFACES):
        out[plan == index] = surface[band]
    return out


def _uniform(surface: dict[str, float], size: int = CANVAS_PX) -> npt.NDArray[np.int_]:
    """A scene of one surface throughout."""
    return np.full((size, size), SURFACES.index(surface), dtype=int)


def _context(
    plans: Mapping[str, np.ndarray],
    *,
    sar_bands: Sequence[str] = SAR_BANDS,
    optical_bands: Sequence[str] = OPTICAL_BANDS,
    question: str = "do the sensors agree about the built-up area?",
) -> ToolContext:
    """A CROSS_MODAL step over one optical and one SAR image."""
    return ToolContext(
        trace_id="tr_cm",
        step=4,
        pair_type=PairType.CROSS_MODAL,
        images=[
            _bundle("img_0", Modality.OPTICAL, ImageRole.OPTICAL, optical_bands),
            _bundle("img_1", Modality.SAR, ImageRole.SAR, sar_bands),
        ],
        pixels=_ScriptedReader(plans),
        question=question,
    )


class _ScriptedBackend(LazyBackend):
    """A VLM that returns a fixed generation through the real backend interface."""

    kind = BackendKind.HF

    def __init__(self, answer: str) -> None:
        """Script one answer, and record the requests that asked for it."""
        super().__init__(BackendConfig(model_id="test/qwen3-vl-stub", device="cpu"))
        self.answer = answer
        self.requests: list[GenerationRequest] = []

    def _load(self) -> None:
        return None

    def _unload(self) -> None:
        return None

    def _generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(
            text=self.answer, backend=self.kind, model_id=self.model_id, device="cpu"
        )


# ==================================================== the bare-soil / built-up case


def test_optical_alone_calls_bare_soil_built_up() -> None:
    """The confusion this phase exists to resolve, established before resolving it.

    NDBI is the normalised difference of SWIR and NIR, and dry soil is SWIR-bright
    for the same reason concrete is. The index analyser is not wrong to report
    what it measures — it simply has no band that separates the two, so it reports
    a quarry as built-up and cannot do better.
    """
    plans = {"img_0": _uniform(BARE_SOIL), "img_1": _uniform(BARE_SOIL)}
    ctx = _context(plans)
    ctx.images = [ctx.images[0]]

    result = SpectralIndexAnalyzer().run(ctx, {"indices": ["ndvi", "ndwi", "ndbi"]})

    assert result.scalars["built_up_fraction_pct"] == pytest.approx(100.0)
    assert result.scalars["vegetation_fraction_pct"] == pytest.approx(0.0)


def test_radar_reassigns_the_whole_confused_area_to_bare_soil() -> None:
    """The demo moment: same pixels, and the physics layer gets it right.

    Bare soil is a rough single surface. Its backscatter is a surface return —
    far too weak to be the double-bounce of a wall over a ground plane — so every
    pixel optical flagged as built-up is reassigned.
    """
    plans = {"img_0": _uniform(BARE_SOIL), "img_1": _uniform(BARE_SOIL)}
    result = PhysicsAgreement().run(_context(plans), {})

    assert result.scalars["ndbi_ambiguous_pct"] == pytest.approx(100.0)
    assert result.scalars["bare_soil_reassigned_pct"] == pytest.approx(100.0)
    assert result.scalars["built_up_confirmed_pct"] == pytest.approx(0.0)

    statements = result.data["statements"]
    assert any("bare soil" in line for line in statements)
    assert any("double-bounce" in line for line in statements)


def test_radar_confirms_genuine_built_up_rather_than_reassigning_everything() -> None:
    """The rule has to be able to say yes, or it is not separating anything.

    Built-up reflectance here is within two hundredths of bare soil's in every
    optical band. Only sigma-nought distinguishes them, and it must.
    """
    plans = {"img_0": _uniform(BUILT_UP), "img_1": _uniform(BUILT_UP)}
    result = PhysicsAgreement().run(_context(plans), {})

    assert result.scalars["ndbi_ambiguous_pct"] == pytest.approx(100.0)
    assert result.scalars["built_up_confirmed_pct"] == pytest.approx(100.0)
    assert result.scalars["bare_soil_reassigned_pct"] == pytest.approx(0.0)
    assert any("built-up" in line for line in result.data["statements"])


def test_a_mixed_scene_is_split_in_the_proportion_radar_sees() -> None:
    """Half quarry, half housing — optically one class, and radar splits it evenly."""
    plan = _uniform(BARE_SOIL)
    plan[:32, :] = SURFACES.index(BUILT_UP)
    plans = {"img_0": plan, "img_1": plan}

    result = PhysicsAgreement().run(_context(plans), {})

    assert result.scalars["ndbi_ambiguous_pct"] == pytest.approx(100.0)
    assert result.scalars["built_up_confirmed_pct"] == pytest.approx(50.0)
    assert result.scalars["bare_soil_reassigned_pct"] == pytest.approx(50.0)


def test_vegetation_is_never_a_candidate_for_the_built_up_split() -> None:
    """A canopy is not SWIR-bright, and must not enter the ambiguous set at all."""
    plans = {"img_0": _uniform(VEGETATION), "img_1": _uniform(VEGETATION)}
    result = PhysicsAgreement().run(_context(plans), {})

    assert result.scalars["ndbi_ambiguous_pct"] == pytest.approx(0.0)
    assert result.scalars["vegetated_agree_pct"] == pytest.approx(100.0)
    assert float(result.scalars["ndvi_mean"]) > NDVI_VEGETATED


def test_a_single_polarisation_acquisition_still_resolves_the_confusion() -> None:
    """RISAT-1 arrives single-pol more often than not, and must not lose the split.

    Sigma-nought VV alone carries the double-bounce signal; the VV/VH ratio only
    corroborates it. Losing VH must cost the corroboration, not the answer.
    """
    plans = {"img_0": _uniform(BUILT_UP), "img_1": _uniform(BUILT_UP)}
    result = PhysicsAgreement().run(_context(plans, sar_bands=("vv",)), {})

    assert result.scalars["built_up_confirmed_pct"] == pytest.approx(100.0)
    assert "vv_vh_ratio_mean_db" not in result.scalars
    assert result.params["polarisations"] == "vv"
    assert any("single-polarisation" in note for note in result.notes)
    assert any("single-polarisation" in line for line in result.data["statements"])


def test_a_bright_canopy_is_not_counted_as_double_bounce() -> None:
    """A depolarising return is a volume, whatever its brightness.

    Directly at the rule, because the surface table has no scene that produces
    this combination — which is the point: it is the failure mode a
    sigma-nought-only rule would have, and dual-pol data is what forecloses it.
    """
    valid = np.ones((4, 4), dtype=bool)
    bright = np.full((4, 4), SIGMA0_BUILT_UP_DB + 3.0, dtype=np.float32)

    co_polarised = resolve_confusion(
        built_like=valid,
        vegetation_index=np.zeros((4, 4), dtype=np.float32),
        sigma0=bright,
        ratio=np.full((4, 4), 8.0, dtype=np.float32),
        valid=valid,
        vegetated=NDVI_VEGETATED,
        built_db=SIGMA0_BUILT_UP_DB,
    )
    depolarised = resolve_confusion(
        built_like=valid,
        vegetation_index=np.zeros((4, 4), dtype=np.float32),
        sigma0=bright,
        ratio=np.full((4, 4), 2.0, dtype=np.float32),
        valid=valid,
        vegetated=NDVI_VEGETATED,
        built_db=SIGMA0_BUILT_UP_DB,
    )
    assert co_polarised.built_up.all()
    assert not depolarised.built_up.any()
    assert depolarised.bare_soil.all()


def test_water_is_where_the_two_sensors_agree_most_strongly() -> None:
    """The most reliable rule in the set: dark in NIR and specular in microwave."""
    plans = {"img_0": _uniform(WATER), "img_1": _uniform(WATER)}
    result = PhysicsAgreement().run(_context(plans), {})

    assert result.scalars["water_agree_pct"] == pytest.approx(100.0)
    assert float(result.scalars["sigma0_vv_mean_db"]) == pytest.approx(-22.0)


def test_without_a_swir_band_the_confusion_is_declared_unresolved() -> None:
    """No NDBI, no ambiguous set — and the trace must say so rather than imply zero."""
    plans = {"img_0": _uniform(BARE_SOIL), "img_1": _uniform(BARE_SOIL)}
    result = PhysicsAgreement().run(
        _context(plans, optical_bands=("red", "green", "nir")), {}
    )

    assert "ndbi_ambiguous_pct" not in result.scalars
    assert "built_up_confirmed_pct" not in result.scalars
    assert any("SWIR" in note for note in result.notes)


def test_the_statements_quote_only_numbers_that_reached_the_fact_sheet() -> None:
    """A sentence citing a scalar the tool did not emit would be stripped as uncited."""
    plans = {"img_0": _uniform(BARE_SOIL), "img_1": _uniform(BARE_SOIL)}
    result = PhysicsAgreement().run(_context(plans), {})

    for key in ("ndvi_mean", "ndbi_mean", "sigma0_vv_mean_db", "vv_vh_ratio_mean_db"):
        assert key in result.scalars

    schema = default_registry()["physics_agreement"].scalars_schema
    from satquery.evidence.fact_sheet import validate_scalar

    for name, value in result.scalars.items():
        assert validate_scalar(str(name), value, schema) is None, name


def test_the_rules_need_one_image_of_each_modality() -> None:
    """Two optical scenes are a bi-temporal pair, not a cross-modal one."""
    ctx = _context({"img_0": _uniform(WATER), "img_1": _uniform(WATER)})
    ctx.images = [ctx.images[0]]
    with pytest.raises(MissingInputError):
        PhysicsAgreement().run(ctx, {})


# ================================================================ grounding maths


@pytest.mark.parametrize(
    "box",
    [
        (0, 0, 64, 64),
        (10, 20, 30, 44),
        (1, 1, 2, 2),
        (100, 3, 447, 448),
        (200, 200, 448, 448),
    ],
)
def test_a_box_round_trips_pixel_to_wgs84_to_pixel_within_one_pixel(
    box: tuple[int, int, int, int]
) -> None:
    """Master.md §8 Phase 6's acceptance criterion, at its stated tolerance.

    The trip is a real one: the canvas affine into UTM 43N metres, then a pyproj
    reprojection onto the WGS84 ellipsoid and back. What is being asserted is
    that the two conversions in ``text_grounding`` are actual inverses, because a
    box that does not come back is a box that was never on the ground.
    """
    geometry = _geometry(size=448, gsd_m=0.65)

    ring = pixel_to_wgs84(box, geometry)
    assert ring is not None
    recovered = wgs84_to_pixel(ring, geometry)

    for original, restored in zip(box, recovered, strict=True):
        assert abs(original - restored) <= 1, f"{box} came back as {recovered}"


def test_the_ring_is_closed_and_lon_first() -> None:
    """GeoJSON needs the first point repeated; the contract needs lon before lat."""
    ring = pixel_to_wgs84((10, 10, 40, 40), _geometry(size=64))
    assert ring is not None
    assert len(ring) == 5
    assert ring[0] == ring[-1]
    # 712 km easting in UTM 43N is eastern India: longitude near 77, latitude near 28.
    for lon, lat in ring:
        assert 70.0 < lon < 85.0, "latitude appears to be in the longitude slot"
        assert 20.0 < lat < 35.0


def test_an_unreferenced_canvas_yields_no_coordinates_rather_than_wrong_ones() -> None:
    """The benchmark PNG path has no CRS, and pixel indices are not lon/lat."""
    unreferenced = _geometry(crs=None)
    assert pixel_to_wgs84((1, 1, 5, 5), unreferenced) is None
    with pytest.raises(MissingInputError):
        wgs84_to_pixel([[77.0, 28.0]], unreferenced)


@pytest.mark.parametrize("frame", [448, 512, 1024, 333])
def test_a_box_round_trips_pixel_to_normalised_to_pixel_within_one_pixel(
    frame: int,
) -> None:
    """The other half of the trip: Qwen's 0-1000 frame quantises, but not by much.

    At 448 px one normalised unit is 0.448 px, so the format itself cannot cost
    more than a pixel. Asserting it keeps the box scale honest if it ever changes.
    """
    for box in ((0, 0, frame, frame), (17, 41, 200, 300), (frame - 3, 0, frame, 9)):
        restored = from_pixels(box, frame, frame).to_pixels(frame, frame)
        for original, value in zip(box, restored, strict=True):
            assert abs(original - value) <= 1, f"{box} at {frame} became {restored}"


def test_the_canonical_format_round_trips_through_the_shared_serialiser() -> None:
    """DATA_ADAPTATION_PLAN §4.5: the corpus writes what the tool parses.

    Byte-identical is the requirement, so the assertion is on the string as well
    as on the boxes recovered from it.
    """
    boxes = [
        NormalisedBox(112, 340, 288, 512, label="aircraft"),
        NormalisedBox(600, 100, 700, 250, label="ship"),
    ]
    text = serialise(boxes)

    assert text == (
        "<|object_ref_start|>aircraft<|object_ref_end|>"
        "<|box_start|>(112,340),(288,512)<|box_end|>"
        "<|object_ref_start|>ship<|object_ref_end|>"
        "<|box_start|>(600,100),(700,250)<|box_end|>"
    )
    assert parse(text) == boxes


def test_the_parser_tolerates_the_json_form_some_checkpoints_emit() -> None:
    """Emission has one format; parsing has to survive the ones in the wild."""
    normalised = parse('[{"bbox": [10, 20, 30, 40], "label": "car", "score": 0.8}]')
    assert normalised == [NormalisedBox(10, 20, 30, 40, label="car", score=0.8)]

    # Absolute pixels are only *detectable* by exceeding the normalised frame, so
    # they are rescaled against the frame they were measured on.
    absolute = parse(
        '[{"bbox_2d": [512, 1024, 1536, 2048], "label": "ship"}]', width=2048, height=2048
    )
    assert absolute == [NormalisedBox(250, 500, 750, 1000, label="ship")]


def test_a_box_that_could_be_either_convention_is_read_as_normalised() -> None:
    """The one case the heuristic cannot decide, pinned so the choice is visible.

    On a 448 px canvas the coordinates (112, 224, 336, 448) are a valid reading
    under both conventions, and nothing in the text says which. Normalised wins
    because that is the format the corpus is built with and the only one this
    system ever emits (DATA_ADAPTATION_PLAN §4.5) — a wrong guess towards the
    format we control is recoverable; one towards the format we do not is not.
    """
    boxes = parse(
        '[{"bbox_2d": [112, 224, 336, 448], "label": "ship"}]', width=448, height=448
    )
    assert boxes == [NormalisedBox(112, 224, 336, 448, label="ship")]


def test_a_degenerate_box_is_dropped_rather_than_failing_the_answer() -> None:
    """One bad box is a mistake in one box, not in the whole generation."""
    text = serialise([NormalisedBox(10, 10, 20, 20)]) + "<|box_start|>(50,50),(50,90)<|box_end|>"
    assert parse(text) == [NormalisedBox(10, 10, 20, 20)]

    with pytest.raises(BoxFormatError):
        NormalisedBox(30, 10, 20, 20)
    with pytest.raises(BoxFormatError):
        NormalisedBox(0, 0, BOX_SCALE + 1, 10)


def test_prose_with_no_boxes_is_an_empty_answer_not_an_error() -> None:
    """"There are no aircraft here" is a correct grounding answer."""
    assert parse("I could not find any aircraft in this scene. NONE") == []


# ============================================================== the grounding tool


def _grounding_context(crs: str | None = UTM43N) -> ToolContext:
    """A GROUNDING step over one optical image."""
    optical = _bundle("img_0", Modality.OPTICAL, ImageRole.SINGLE, ("red", "green", "blue"))

    class _Reader(PixelReader):
        def read(
            self, image: ImageBundle, bands: Sequence[str], size: int | None = None
        ) -> Bands:
            order = tuple(dict.fromkeys(bands))
            return Bands(
                order=order,
                stack=np.full((len(order), CANVAS_PX, CANVAS_PX), 0.4, dtype=np.float32),
                valid=np.ones((CANVAS_PX, CANVAS_PX), dtype=bool),
                geometry=_geometry(crs=crs),
            )

    return ToolContext(
        trace_id="tr_g",
        step=2,
        pair_type=PairType.SINGLE,
        images=[optical],
        pixels=_Reader(),
        question="where are the aircraft?",
    )


def test_grounding_emits_both_a_box_set_and_reprojected_features() -> None:
    """The frontend draws pixels; the operations team opens lon/lat. Both, from one pass."""
    answer = serialise(
        [
            NormalisedBox(100, 100, 300, 300, label="aircraft"),
            NormalisedBox(500, 500, 700, 800, label="aircraft"),
        ]
    )
    result = TextGrounding(backend=_ScriptedBackend(answer)).run(_grounding_context(), {})

    assert result.scalars["n_boxes"] == 2
    types = {draft.type for draft in result.artifacts}
    assert types == {ArtifactType.BBOX_SET, ArtifactType.GEOJSON}

    boxes = next(d for d in result.artifacts if d.type is ArtifactType.BBOX_SET)
    features = next(d for d in result.artifacts if d.type is ArtifactType.GEOJSON)
    assert boxes.inline is not None
    assert len(boxes.inline["boxes"]) == len(features.geojson["features"]) == 2
    assert features.geojson["source_crs"] == UTM43N

    # Reported in emission order, so box_0 on the map is box_0 in the file.
    assert [box["id"] for box in boxes.inline["boxes"]] == ["box_0", "box_1"]
    assert [feature["id"] for feature in features.geojson["features"]] == [
        "box_0",
        "box_1",
    ]

    first = boxes.inline["boxes"][0]
    assert first["label"] == "aircraft"
    assert first["bbox_px"] == [6, 6, 20, 20]
    assert first["bbox_wgs84"] is not None
    # 196 px and 260 px on a 10 m grid.
    assert result.scalars["mean_box_area_m2"] == pytest.approx(22_800.0)


def test_an_unreferenced_image_grounds_in_pixels_and_says_so() -> None:
    """No CRS means no GeoJSON — and a note, because the absence is the finding."""
    answer = serialise([NormalisedBox(100, 100, 300, 300, label="ship")])
    result = TextGrounding(backend=_ScriptedBackend(answer)).run(
        _grounding_context(crs=None), {}
    )

    assert result.scalars["n_boxes"] == 1
    assert {d.type for d in result.artifacts} == {ArtifactType.BBOX_SET}
    assert result.params["georeferenced"] is False
    assert any("no CRS" in note for note in result.notes)


def test_boxes_below_the_score_threshold_are_dropped_and_counted() -> None:
    """A dropped detection is a fact about the run, not something to hide."""
    answer = (
        '[{"bbox": [100, 100, 300, 300], "label": "ship", "score": 0.9},'
        ' {"bbox": [400, 400, 500, 500], "label": "ship", "score": 0.1}]'
    )
    result = TextGrounding(backend=_ScriptedBackend(answer)).run(
        _grounding_context(), {"score_threshold": 0.5}
    )

    assert result.scalars["n_boxes"] == 1
    assert result.params["boxes_parsed"] == 2
    assert any("dropped" in note for note in result.notes)


def test_the_box_cap_keeps_the_most_confident_detections() -> None:
    """A cap that kept whatever the model emitted first would be arbitrary."""
    entries = ", ".join(
        f'{{"bbox": [{10 * i}, 10, {10 * i + 60}, 90], "label": "car", "score": {i / 10:.1f}}}'
        for i in range(1, 9)
    )
    result = TextGrounding(backend=_ScriptedBackend(f"[{entries}]")).run(
        _grounding_context(), {"max_boxes": 3}
    )

    assert result.scalars["n_boxes"] == 3
    assert float(result.scalars["mean_box_score"]) == pytest.approx(0.7, abs=1e-6)


def test_grounding_without_a_backend_asks_for_its_fallback() -> None:
    """No VLM is a ToolError, which the executor turns into semantic_segmenter."""

    class _Dead(_ScriptedBackend):
        def _generate(self, request: GenerationRequest) -> GenerationResult:
            raise RuntimeError("no weights here")

    with pytest.raises((ToolError, RuntimeError)):
        TextGrounding(backend=_Dead("")).run(_grounding_context(), {})


def test_the_grounding_prompt_states_the_format_the_parser_expects() -> None:
    """The training/inference contract, asserted where it would actually drift."""
    backend = _ScriptedBackend(serialise([NormalisedBox(10, 10, 90, 90)]))
    TextGrounding(backend=backend).run(_grounding_context(), {})

    system = backend.requests[0].system
    assert "<|box_start|>(x1,y1),(x2,y2)<|box_end|>" in system
    assert "0 to 1000" in system


# ==================================================================== DOFA fusion


class _StubEmbedder:
    """A wavelength-conditioned encoder that records what it was asked to encode."""

    def __init__(self, grids: Sequence[np.ndarray]) -> None:
        """Return each grid in turn; optical first, then SAR."""
        self.grids = list(grids)
        self.calls: list[tuple[tuple[int, ...], list[float]]] = []

    def embed(self, stack: np.ndarray, wavelengths: Sequence[float]) -> np.ndarray:
        self.calls.append((stack.shape, list(wavelengths)))
        return self.grids[len(self.calls) - 1]


def _grid(vector: Sequence[float], side: int = 14) -> np.ndarray:
    """A patch grid whose every token is the same embedding."""
    return np.tile(np.asarray(vector, dtype=np.float32), (side, side, 1))


def test_identical_embeddings_agree_completely_and_opposite_ones_not_at_all() -> None:
    """Cosine spans [-1, 1]; the tool reports agreement on [0, 1], once, here."""
    same = _grid([1.0, 0.0, 0.0])
    assert cosine_agreement(same, same.copy()) == pytest.approx(1.0)
    assert cosine_agreement(same, _grid([-1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert cosine_agreement(same, _grid([0.0, 1.0, 0.0])) == pytest.approx(0.5)


def test_embedding_magnitude_does_not_change_the_agreement() -> None:
    """DOFA normalises direction, not length; a brighter scene is not a different one."""
    left = _grid([0.3, 0.4, 0.0])
    right = _grid([3.0, 4.0, 0.0])
    assert cosine_agreement(left, right) == pytest.approx(1.0)


def test_two_different_patch_grids_are_refused_rather_than_broadcast() -> None:
    with pytest.raises(MissingInputError):
        cosine_agreement(_grid([1.0, 0.0], side=14), _grid([1.0, 0.0], side=7))


def test_upsampling_a_patch_field_invents_no_intermediate_verdicts() -> None:
    """Each value is one patch's verdict; interpolating would fabricate others."""
    field = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    enlarged = upsample(field, 8, 8)

    assert enlarged.shape == (8, 8)
    assert set(np.unique(enlarged).tolist()) == {0.0, 1.0}
    assert enlarged[0, 0] == 0.0
    assert enlarged[0, 7] == 1.0


def test_the_encoder_is_told_each_channel_s_physical_wavelength() -> None:
    """The whole reason DOFA can compare two sensors is that it is told the physics.

    Optical wavelengths are sub-micrometre; Sentinel-1's C-band carrier is 5.5 cm,
    four orders of magnitude away. Handing the encoder the wrong units would make
    the hypernetwork generate optical filters for a radar channel.
    """
    embedder = _StubEmbedder([_grid([1.0, 0.0, 0.0]), _grid([1.0, 0.0, 0.0])])
    plans = {"img_0": _uniform(VEGETATION), "img_1": _uniform(VEGETATION)}
    CrossModalConsistency(embedder=embedder).run(
        _context(plans, optical_bands=("red", "green", "nir", "swir1")), {}
    )

    (_, optical_wavelengths), (_, sar_wavelengths) = embedder.calls
    assert optical_wavelengths == [
        WAVELENGTH_UM["green"],
        WAVELENGTH_UM["red"],
        WAVELENGTH_UM["nir"],
        WAVELENGTH_UM["swir1"],
    ]
    assert sar_wavelengths == [WAVELENGTH_UM["vv"], WAVELENGTH_UM["vh"]]
    assert all(value < 3.0 for value in optical_wavelengths)
    assert all(value > 10_000.0 for value in sar_wavelengths)


def test_a_checkpoint_s_own_wavelength_convention_can_be_supplied() -> None:
    """A pretrained encoder trained against different units must not silently mismatch."""
    embedder = _StubEmbedder([_grid([1.0, 0.0]), _grid([1.0, 0.0])])
    plans = {"img_0": _uniform(WATER), "img_1": _uniform(WATER)}
    result = CrossModalConsistency(embedder=embedder).run(
        _context(plans), {"wavelengths_um": {"vv": 375_000.0, "vh": 375_000.0}}
    )

    assert embedder.calls[1][1] == [375_000.0, 375_000.0]
    assert result.params["wavelengths_um"]["vv"] == 375_000.0


def test_the_agreement_map_reaches_the_gallery_on_the_canvas_grid() -> None:
    """A 14x14 patch verdict has to be viewable over the imagery it describes."""
    embedder = _StubEmbedder([_grid([1.0, 0.0, 0.0]), _grid([0.0, 1.0, 0.0])])
    plans = {"img_0": _uniform(BUILT_UP), "img_1": _uniform(BUILT_UP)}

    result = CrossModalConsistency(embedder=embedder).run(_context(plans), {})

    assert result.scalars["agreement_mean"] == pytest.approx(0.5)
    assert result.scalars["patch_grid"] == "14x14"
    assert result.scalars["embedding_dim"] == 3
    assert result.scalars["divergent_area_pct"] == pytest.approx(0.0)

    heatmap = result.artifacts[0]
    assert heatmap.type is ArtifactType.HEATMAP
    assert heatmap.raster is not None
    assert heatmap.raster.shape == (CANVAS_PX, CANVAS_PX)
    assert heatmap.image is not None
    assert heatmap.image.shape == (CANVAS_PX, CANVAS_PX, 3)
    assert heatmap.geometry is not None
    assert heatmap.geometry.crs == UTM43N


def test_divergence_is_reported_because_it_is_the_interesting_half() -> None:
    """Where the sensors encode different surfaces is what SAR adds to optical."""
    optical = _grid([1.0, 0.0, 0.0])
    radar = _grid([1.0, 0.0, 0.0])
    radar[:7] = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    embedder = _StubEmbedder([optical, radar])
    plans = {"img_0": _uniform(BUILT_UP), "img_1": _uniform(BUILT_UP)}

    result = CrossModalConsistency(embedder=embedder).run(_context(plans), {})

    assert result.scalars["agreement_min"] == pytest.approx(0.0)
    assert result.scalars["agreement_max"] == pytest.approx(1.0)
    assert result.scalars["divergent_area_pct"] == pytest.approx(50.0, abs=1.0)


def test_dofa_needs_one_image_of_each_modality() -> None:
    ctx = _context({"img_0": _uniform(WATER), "img_1": _uniform(WATER)})
    ctx.images = [ctx.images[1], ctx.images[1]]
    with pytest.raises(MissingInputError):
        CrossModalConsistency(embedder=_StubEmbedder([])).run(ctx, {})


def test_every_dofa_scalar_is_accepted_by_the_registry_schema() -> None:
    """A scalar the registry does not declare is dropped, never cited."""
    from satquery.evidence.fact_sheet import validate_scalar

    embedder = _StubEmbedder([_grid([1.0, 0.0]), _grid([0.0, 1.0])])
    plans = {"img_0": _uniform(BUILT_UP), "img_1": _uniform(BUILT_UP)}
    result = CrossModalConsistency(embedder=embedder).run(_context(plans), {})

    schema = default_registry()["crossmodal_consistency"].scalars_schema
    for name, value in result.scalars.items():
        assert validate_scalar(str(name), value, schema) is None, name


# ================================================================== segmentation


class _StubSegmenter:
    """A segmenter returning a scripted LoveDA label map."""

    classes = LOVEDA_CLASSES

    def __init__(self, labels: npt.NDArray[np.int16]) -> None:
        self.labels = labels

    def segment(self, rgb: np.ndarray) -> npt.NDArray[np.int16]:
        return self.labels


def test_loveda_labels_are_translated_onto_the_shared_vocabulary() -> None:
    """The join with change_statistics is by class *name*, so the names must match.

    ``configs/class_vocabulary.yaml`` is the authority. A segmenter emitting
    "agriculture" where the planner and the FactSheet say "vegetation" produces
    per-class change numbers that silently never match anything.
    """
    labels = np.array([[0, 1, 2], [3, 4, 5], [6, 6, 0]], dtype=np.int16)
    canonical = to_canonical(labels, LOVEDA_CLASSES)

    names = [
        CANONICAL_CLASSES[value] if value >= 0 else None
        for value in canonical.ravel().tolist()
    ]
    assert names == [
        None, "built_up", "road",
        "water", "bare_soil", "vegetation",
        "vegetation", "vegetation", None,
    ]


def test_background_becomes_unlabelled_rather_than_a_land_cover_claim() -> None:
    """LoveDA's background is "not annotated", which is not a class."""
    canonical = to_canonical(np.zeros((4, 4), dtype=np.int16), LOVEDA_CLASSES)
    assert (canonical == -1).all()


def test_the_segmenter_reports_fractions_over_the_valid_extent() -> None:
    """Padding a non-square raster into a square canvas must not dilute a class."""
    labels = np.zeros((CANVAS_PX, CANVAS_PX), dtype=np.int16)
    labels[:16, :] = LOVEDA_CLASSES.index("building")
    labels[16:32, :] = LOVEDA_CLASSES.index("water")
    labels[32:, :] = LOVEDA_CLASSES.index("forest")

    result = SemanticSegmenter(segmenter=_StubSegmenter(labels)).run(
        _grounding_context(), {}
    )

    assert result.scalars["built_up_fraction_pct"] == pytest.approx(25.0)
    assert result.scalars["water_fraction_pct"] == pytest.approx(25.0)
    assert result.scalars["vegetation_fraction_pct"] == pytest.approx(50.0)
    assert result.scalars["class_count"] == 3
    assert result.scalars["unlabelled_pct"] == pytest.approx(0.0)

    class_map = result.data["segmentation"]
    assert isinstance(class_map, ClassMap)
    assert class_map.classes == CANONICAL_CLASSES
    assert class_map.pixel_area_m2 == pytest.approx(100.0)
    assert result.scalars["water_area_m2"] == pytest.approx(102_400.0)


def test_the_segmenter_also_emits_boxes_because_grounding_falls_back_to_it() -> None:
    """text_grounding -> semantic_segmenter must not change the artifact shape."""
    labels = np.zeros((CANVAS_PX, CANVAS_PX), dtype=np.int16)
    labels[4:12, 4:12] = LOVEDA_CLASSES.index("building")
    labels[40:56, 40:56] = LOVEDA_CLASSES.index("water")

    result = SemanticSegmenter(segmenter=_StubSegmenter(labels)).run(
        _grounding_context(), {}
    )

    types = {draft.type for draft in result.artifacts}
    assert types == {ArtifactType.SEGMENTATION, ArtifactType.BBOX_SET}
    boxes = next(d for d in result.artifacts if d.type is ArtifactType.BBOX_SET)
    assert boxes.inline is not None
    assert {box["label"] for box in boxes.inline["boxes"]} == {"built_up", "water"}
    assert boxes.inline["boxes"][0]["bbox_px"] == [4, 4, 12, 12]


def test_regions_below_the_floor_are_not_reported_as_objects() -> None:
    """A four-pixel blob is a segmentation artefact, not a building."""
    labels = np.zeros((32, 32), dtype=np.int16)
    labels[0:2, 0:2] = 1  # 4 px
    labels[10:20, 10:20] = 1  # 100 px
    canonical = to_canonical(labels, LOVEDA_CLASSES)

    assert len(class_boxes(canonical, min_region_px=32)) == 1
    assert len(class_boxes(canonical, min_region_px=2)) == 2


def test_a_segmenter_with_no_checkpoint_asks_for_its_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No weights is a ToolError, which becomes spectral_index_analyzer."""
    monkeypatch.delenv("SATQUERY_SEG_CHECKPOINT", raising=False)
    with pytest.raises(SegmenterUnavailableError, match="no segmentation checkpoint"):
        SemanticSegmenter().run(_grounding_context(), {})


def test_the_counter_reports_a_component_count_per_class() -> None:
    """Counting a class map as one mask would merge a building touching a road."""
    labels = np.full((32, 32), -1, dtype=np.int16)
    labels[2:6, 2:6] = CANONICAL_CLASSES.index("built_up")
    labels[2:6, 10:14] = CANONICAL_CLASSES.index("built_up")
    labels[20:28, 20:28] = CANONICAL_CLASSES.index("water")

    from satquery.schemas.trace import ArtifactRef

    reference = ArtifactRef(
        id="art_1",
        type=ArtifactType.SEGMENTATION,
        mime="image/png",
        label="Land-cover segmentation",
        produced_by_step=2,
    )
    ctx = ToolContext(
        trace_id="tr_c",
        step=3,
        pair_type=PairType.SINGLE,
        images=[],
        artifacts=[reference],
        data={
            "art_1": ClassMap(
                labels=labels,
                classes=CANONICAL_CLASSES,
                geometry=_geometry(size=32),
                pixel_area_m2=100.0,
            )
        },
    )

    result = ObjectCounter().run(ctx, {"min_component_px": 4})

    assert result.scalars["built_up_count"] == 2
    assert result.scalars["water_count"] == 1
    assert result.scalars["count"] == 3
    assert result.params["source"] == "segmentation"


# ==================================================== registry and policy wiring


@pytest.mark.parametrize(
    ("tool", "fallback"),
    [
        ("crossmodal_consistency", "physics_agreement"),
        ("text_grounding", "semantic_segmenter"),
        ("semantic_segmenter", "spectral_index_analyzer"),
        ("physics_agreement", None),
        ("object_counter", None),
    ],
)
def test_the_declared_fallback_chains_are_the_frozen_ones(
    tool: str, fallback: str | None
) -> None:
    """AGENT_POLICY_DAG §5's table, asserted against the registry that implements it."""
    assert default_registry()[tool].fallback == fallback


def test_requirement_four_is_answerable_with_no_weights_at_all() -> None:
    """The reason the physics layer is the fallback and not the other way round.

    DOFA needs torchgeo and a 350 MB download; the rule layer needs neither. A
    judging machine with no accelerator still answers the cross-modal question,
    visibly DEGRADED, which is the whole point of declaring that fallback.
    """
    registry = default_registry()
    assert registry["physics_agreement"].available is True
    assert registry["physics_agreement"].fallback is None

    dofa = registry["crossmodal_consistency"]
    if not dofa.available:
        assert dofa.unavailable_reason
        assert "physics_agreement" in dofa.unavailable_reason


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (TaskType.CROSS_MODAL_VQA, "CROSS_MODAL_VQA|CROSS_MODAL|mixed"),
        (TaskType.CROSS_MODAL_COMPARE, "CROSS_MODAL_COMPARE|CROSS_MODAL|mixed"),
        (TaskType.VQA, "VQA|CROSS_MODAL|mixed"),
        (TaskType.CAPTION, "CAPTION|CROSS_MODAL|mixed"),
        (TaskType.GROUNDING, "GROUNDING|CROSS_MODAL|mixed"),
        (TaskType.SCENE_CLASSIFY, "SCENE_CLASSIFY|CROSS_MODAL|mixed"),
    ],
)
def test_every_cross_modal_policy_entry_resolves_to_itself(
    task: TaskType, expected: str
) -> None:
    """AGENT_POLICY_DAG §4.2 entries 16-21, each reachable rather than shadowed.

    Resolution falls back key -> ``TASK|PAIR|*`` -> generic, so an entry that is
    present but never selected looks identical to one that works until a judge
    asks the cross-modal question.
    """
    table = default_table()
    assert expected in table.entries

    resolved, entry, planner = resolve_key(table, task, PairType.CROSS_MODAL, "mixed")
    assert resolved == expected
    assert entry is table.entries[expected]
    assert planner == PLANNER_VERSION, "the entry exists but the router did not reach it"


def test_the_cross_modal_plans_run_fusion_before_the_synthesiser() -> None:
    """A plan whose VLM step does not depend on the fusion would not cite it."""
    table = default_table()
    for key in ("CROSS_MODAL_VQA|CROSS_MODAL|mixed", "VQA|CROSS_MODAL|mixed"):
        steps = table.entries[key].steps
        tools = [step.tool for step in steps]
        assert tools.index("crossmodal_consistency") < tools.index("vlm_vqa")
        assert tools.index("physics_agreement") < tools.index("vlm_vqa")

        terminal = steps[-1]
        fusion_steps = {
            index
            for index, step in enumerate(steps, start=1)
            if step.tool in {"crossmodal_consistency", "physics_agreement"}
        }
        assert fusion_steps <= set(terminal.depends_on)


def test_the_grounding_plans_feed_the_boxes_to_the_synthesiser() -> None:
    """A grounding answer that did not see its own boxes would describe the view."""
    table = default_table()
    for key in ("GROUNDING|SINGLE|*", "GROUNDING|CROSS_MODAL|mixed"):
        steps = table.entries[key].steps
        terminal = steps[-1]
        assert any("BBOX_SET" in str(token) for token in terminal.inputs)


def test_the_count_plan_counts_what_grounding_found() -> None:
    """Entry 9: the count is measured, never generated."""
    steps = default_table().entries["COUNT|SINGLE|*"].steps
    tools = [step.tool for step in steps]
    assert tools == ["spectral_renderer", "text_grounding", "object_counter", "vlm_vqa"]


def test_every_policy_tool_exists_in_the_registry() -> None:
    """A table naming a tool the registry does not declare fails at request time."""
    from satquery.agent.planner import lint_against_registry

    assert lint_against_registry(default_table(), list(default_registry().tools)) == []


# ============================================================ end to end, no weights


def test_a_cross_modal_question_is_answered_from_both_sensors(
    scene_paths: dict[str, Path], tmp_path: Path
) -> None:
    """API_CONTRACT §7.3, on real rasters through the real DAG controller.

    Neither DOFA nor a VLM is available here, so this exercises what a judging
    machine actually runs: capability matching drops the learned encoder,
    substitutes the deterministic rule layer, and the answer still carries a
    backscatter statistic and a spectral index — which is the mandatory
    requirement 4 acceptance criterion.
    """
    sources = [
        SourceImage(path=scene_paths[name], filename=scene_paths[name].name)
        for name in ("s2_pre", "s1_vvvh")
    ]
    result = ingest(sources)
    analysis = asyncio.run(
        analyze(
            AnalysisRequest(
                query="what does the SAR reveal that the optical does not?",
                sources=sources,
                ingest=result,
            ),
            store=ArtifactStore(tmp_path / "artifacts"),
            registry=default_registry().with_availability(
                {"crossmodal_consistency": False}
            ),
            cache=ExecutionCache(),
        )
    )
    trace = analysis.trace

    assert trace.compatibility.pair_type is PairType.CROSS_MODAL
    assert any(
        check.name.value == "modality_distinct" and check.status.value == "PASS"
        for check in trace.compatibility.checks
    )

    fusion = next(e for e in trace.executions if e.tool == "physics_agreement")
    assert fusion.status in {ToolStatus.OK, ToolStatus.DEGRADED}

    # Both sensors reached the evidence, which is what "cross-modal" has to mean.
    assert any(key.startswith("spectral_index_analyzer.") for key in trace.fact_sheet)
    assert any(key.startswith("sar_backscatter_analyzer.") for key in trace.fact_sheet)
    assert "physics_agreement.agreement_pct" in trace.fact_sheet
    assert ArtifactType.HEATMAP in {a.type for a in trace.artifacts}


def test_the_learned_encoder_is_substituted_visibly_when_it_cannot_run(
    scene_paths: dict[str, Path], tmp_path: Path
) -> None:
    """A DEGRADED cross-modal answer must say which tool actually produced it."""
    sources = [
        SourceImage(path=scene_paths[name], filename=scene_paths[name].name)
        for name in ("s2_pre", "s1_vvvh")
    ]
    result = ingest(sources)
    analysis = asyncio.run(
        analyze(
            AnalysisRequest(
                query="do the optical and radar signals agree about built-up area?",
                sources=sources,
                ingest=result,
            ),
            store=ArtifactStore(tmp_path / "artifacts"),
            registry=default_registry().with_availability(
                {"crossmodal_consistency": False}
            ),
            cache=ExecutionCache(),
        )
    )

    substituted = [
        e for e in analysis.trace.executions if e.fallback_of == "crossmodal_consistency"
    ]
    assert substituted, "the DOFA step was skipped rather than degraded"
    assert substituted[0].tool == "physics_agreement"
    assert substituted[0].status is ToolStatus.DEGRADED
