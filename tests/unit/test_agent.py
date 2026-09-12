"""The 20 Phase 3 golden tests (AGENT_POLICY_DAG.md §9).

Four families, each guarding a different failure mode:

* **Routing** — a query and an input shape must select one specific policy key.
  These are the tests that catch a regex reordering silently changing what the
  system does.
* **Structure** — static lints over the committed tables. They belong in CI
  rather than at request time: a policy row naming a tool nobody registered
  should fail the build.
* **Execution** — determinism, fallbacks and caching. The reproducibility test
  asserts exactly the guaranteed field set of §8, no more: asserting
  ``duration_ms`` would make the suite flaky and asserting nothing would make it
  worthless.
* **Evidence** — the citation validator has to catch a number the FactSheet
  cannot support, and the confidence has to be clamped when it does.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from satquery.agent import planner, task_classifier
from satquery.agent.executor import ExecutionCache, ToolFailedNoFallbackError
from satquery.agent.pipeline import AnalysisRequest, analyze
from satquery.evidence import confidence as confidence_module
from satquery.evidence import fact_sheet as fact_sheet_module
from satquery.evidence.citation_validator import CitationPolicy, validate
from satquery.ingest.errors import IncompatibleInputsError
from satquery.ingest.pipeline import IngestResult, SourceImage, ingest
from satquery.registry.capability_match import MatchStatus, match
from satquery.registry.registry import ToolRegistry, default_registry
from satquery.render.artifact_store import ArtifactStore
from satquery.schemas.enums import (
    ArtifactType,
    Overall,
    PairType,
    TaskType,
    ToolStatus,
)
from satquery.tools.base import ToolContext, ToolError, ToolResult
from satquery.trace.store import TraceStore

# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def registry() -> Any:
    """The committed registry, checked against the built-in implementations."""
    return default_registry()


@pytest.fixture(scope="module")
def table() -> planner.PolicyTable:
    """The committed policy table."""
    return planner.default_table()


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    """An artifact store scoped to one test."""
    return ArtifactStore(tmp_path / "artifacts")


def _sources(scene_paths: dict[str, Path], *names: str) -> list[SourceImage]:
    return [SourceImage(path=scene_paths[name], filename=scene_paths[name].name) for name in names]


def _ingest(scene_paths: dict[str, Path], *names: str) -> tuple[list[SourceImage], IngestResult]:
    sources = _sources(scene_paths, *names)
    return sources, ingest(sources)


def _run(
    scene_paths: dict[str, Path],
    store: ArtifactStore,
    query: str,
    *names: str,
    tools: Mapping[str, Any] | None = None,
    cache: ExecutionCache | None = None,
    registry: Any = None,
    traces: TraceStore | None = None,
) -> Any:
    """Run the full pipeline over named fixtures and return the trace."""
    sources, result = _ingest(scene_paths, *names)
    analysis = asyncio.run(
        analyze(
            AnalysisRequest(query=query, sources=sources, ingest=result),
            store=store,
            registry=registry,
            tools=tools,
            cache=cache if cache is not None else ExecutionCache(),
            traces=traces,
        )
    )
    return analysis.trace


def _policy_key(
    query: str, pair_type: PairType, scene_paths: dict[str, Path], *names: str
) -> tuple[str, task_classifier.Classification]:
    """Classify and plan without running anything, for the routing tests."""
    _, result = _ingest(scene_paths, *names)
    classification = task_classifier.classify(query, result.compatibility.pair_type)
    plan = planner.plan_for(
        classification.primary, result.compatibility.pair_type, result.inputs
    )
    assert result.compatibility.pair_type is pair_type
    return plan.plan.policy_key, classification


# ---------------------------------------------------------------------- routing


def test_01_change_vqa_routes_to_bi_temporal_optical(scene_paths: dict[str, Path]) -> None:
    key, resolved = _policy_key(
        "what changed between these two images", PairType.BI_TEMPORAL, scene_paths,
        "s2_pre", "s2_post",
    )
    assert key == "CHANGE_VQA|BI_TEMPORAL|optical"
    assert resolved.primary is TaskType.CHANGE_VQA
    assert resolved.confidence >= 0.90


def test_02_change_map_beats_change_vqa(scene_paths: dict[str, Path]) -> None:
    key, resolved = _policy_key(
        "show me a map of the changed areas", PairType.BI_TEMPORAL, scene_paths,
        "s2_pre", "s2_post",
    )
    assert resolved.primary is TaskType.CHANGE_MAP
    assert key == "CHANGE_MAP|BI_TEMPORAL|*"


def test_03_change_caption_beats_change_vqa(scene_paths: dict[str, Path]) -> None:
    key, resolved = _policy_key(
        "describe what happened between the two dates", PairType.BI_TEMPORAL, scene_paths,
        "s2_pre", "s2_post",
    )
    assert resolved.primary is TaskType.CHANGE_CAPTION
    assert key == "CHANGE_CAPTION|BI_TEMPORAL|optical"


def test_04_count_fills_the_object_class_slot(scene_paths: dict[str, Path]) -> None:
    key, resolved = _policy_key(
        "how many buildings are there", PairType.SINGLE, scene_paths, "s2_pre"
    )
    assert resolved.primary is TaskType.COUNT
    assert key == "COUNT|SINGLE|*"
    assert resolved.slots["object_class"] == "built_up"
    # COUNT's DAG produces boxes on the way to the count, so grounding is free.
    assert resolved.secondary == [TaskType.GROUNDING]


def test_05_grounding(scene_paths: dict[str, Path]) -> None:
    key, resolved = _policy_key(
        "where is the airport", PairType.SINGLE, scene_paths, "s2_pre"
    )
    assert resolved.primary is TaskType.GROUNDING
    assert key == "GROUNDING|SINGLE|*"
    assert resolved.slots["referring_expression"] == "the airport"


def test_06_caption_on_sar_selects_the_sar_entry(scene_paths: dict[str, Path]) -> None:
    key, resolved = _policy_key(
        "describe this scene", PairType.SINGLE, scene_paths, "s1_vvvh"
    )
    assert resolved.primary is TaskType.CAPTION
    assert key == "CAPTION|SINGLE|sar"


def test_07_cross_modal_compare(scene_paths: dict[str, Path]) -> None:
    key, resolved = _policy_key(
        "what does the SAR show that the optical misses", PairType.CROSS_MODAL, scene_paths,
        "s2_pre", "s1_vvvh",
    )
    assert resolved.primary is TaskType.CROSS_MODAL_COMPARE
    assert key == "CROSS_MODAL_COMPARE|CROSS_MODAL|mixed"
    assert resolved.secondary == [TaskType.CROSS_MODAL_VQA]


def test_08_segmentation(scene_paths: dict[str, Path]) -> None:
    key, resolved = _policy_key(
        "outline the water bodies", PairType.SINGLE, scene_paths, "s2_pre"
    )
    assert resolved.primary is TaskType.SEGMENTATION
    assert key == "SEGMENTATION|SINGLE|optical"
    assert resolved.slots["target_class"] == "water"
    assert resolved.slots["output_format"] == "text+mask"


def test_09_scene_classify(scene_paths: dict[str, Path]) -> None:
    key, resolved = _policy_key(
        "what land cover types are present", PairType.SINGLE, scene_paths, "s2_pre"
    )
    assert resolved.primary is TaskType.SCENE_CLASSIFY
    assert key == "SCENE_CLASSIFY|SINGLE|optical"


def test_10_change_word_outranks_count_on_a_pair(scene_paths: dict[str, Path]) -> None:
    """Priority ordering is load-bearing: "how many changed regions" is a change query."""
    key, resolved = _policy_key(
        "how many changed regions", PairType.BI_TEMPORAL, scene_paths, "s2_pre", "s2_post"
    )
    assert resolved.primary is TaskType.CHANGE_VQA
    assert key == "CHANGE_VQA|BI_TEMPORAL|optical"
    assert resolved.secondary == [TaskType.CHANGE_MAP]


def test_11_pair_type_gate_forbids_change_on_a_single_image(
    scene_paths: dict[str, Path],
) -> None:
    """Stage A is structural: one image cannot have changed, whatever the words say."""
    key, resolved = _policy_key("what changed", PairType.SINGLE, scene_paths, "s2_pre")
    assert resolved.primary is TaskType.VQA
    assert key == "VQA|SINGLE|optical"
    assert TaskType.CHANGE_VQA in resolved.gated_out
    assert any(code == "TASK_GATED_BY_PAIR_TYPE" for code, _ in resolved.warnings)


def test_12_gibberish_falls_through_to_the_flagged_generic_plan(
    scene_paths: dict[str, Path],
) -> None:
    key, resolved = _policy_key("zxqwv plkjh mnbvc", PairType.SINGLE, scene_paths, "s2_pre")
    assert resolved.primary is TaskType.UNSUPPORTED
    assert resolved.confidence < task_classifier.MIN_CONFIDENCE
    assert key == planner.GENERIC_KEY
    assert any(code == "LOW_CLASSIFICATION_CONFIDENCE" for code, _ in resolved.warnings)

    _, result = _ingest(scene_paths, "s2_pre")
    plan = planner.plan_for(resolved.primary, result.compatibility.pair_type, result.inputs)
    assert plan.plan.planner == planner.GENERIC_PLANNER
    assert any(code == "GENERIC_PLAN_USED" for code, _ in plan.warnings)


# -------------------------------------------------------------------- structure


def test_13_every_entry_is_a_dag_in_topological_order(table: planner.PolicyTable) -> None:
    for entry in table.entries.values():
        numbers = [step.step for step in entry.steps]
        assert numbers == list(range(1, len(numbers) + 1)), entry.key
        for step in entry.steps:
            for dependency in step.depends_on:
                assert 1 <= dependency < step.step, f"{entry.key} step {step.step}"
            # Reachability, not a direct edge: the frozen change entry has its
            # terminal step read @2:CHANGE_MASK while depending on [1, 3, 4],
            # and step 3 already orders it after step 2.
            ordered = planner.reachable_from(entry, step)
            for selector in step.artifact_selectors:
                assert selector.step in ordered, (
                    f"{entry.key} step {step.step} reads {selector} without being ordered "
                    f"after step {selector.step}"
                )


def test_14_every_policy_tool_exists_in_the_registry(
    table: planner.PolicyTable, registry: Any
) -> None:
    assert planner.lint_against_registry(table, registry.names) == []


def test_15_every_required_band_resolves_for_some_sensor(registry: Any) -> None:
    from satquery.ingest.bands import load_table as load_band_table

    aliases = load_band_table()
    known = {band for bands in aliases.sensors.values() for band in bands}
    for spec in registry.tools.values():
        for band in spec.accepts.required_bands:
            assert band in known, f"{spec.name} requires {band!r}, which no sensor provides"


# -------------------------------------------------------------------- execution


def test_16_identical_inputs_produce_identical_plans_params_and_facts(
    scene_paths: dict[str, Path], tmp_path: Path
) -> None:
    """The guaranteed set of §8, field by field — and nothing that is not guaranteed."""
    query = "what changed between these two images"
    first = _run(scene_paths, ArtifactStore(tmp_path / "a"), query, "s2_pre", "s2_post")
    second = _run(scene_paths, ArtifactStore(tmp_path / "b"), query, "s2_pre", "s2_post")

    assert first.plan.model_dump() == second.plan.model_dump()
    assert first.fact_sheet == second.fact_sheet
    assert [e.params for e in first.executions] == [e.params for e in second.executions]
    assert [e.scalars for e in first.executions] == [e.scalars for e in second.executions]
    assert [e.status for e in first.executions] == [e.status for e in second.executions]
    assert [c.model_dump() for c in first.compatibility.checks] == [
        c.model_dump() for c in second.compatibility.checks
    ]
    assert first.confidence.components == second.confidence.components
    assert first.answer.text == second.answer.text
    # Explicitly *not* guaranteed, and not asserted: duration_ms, created_at,
    # trace_id, cache_hit and the trace-scoped artifact URLs.
    assert first.trace_id != second.trace_id


def test_17_a_raising_tool_degrades_to_its_fallback(
    scene_paths: dict[str, Path], store: ArtifactStore, registry: Any
) -> None:
    """A tool failure is data, not a 500."""

    class ExplodingDetector:
        name = "siamese_change_detector"

        def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
            raise ToolError("checkpoint refused to load")

    available = registry.with_availability({"siamese_change_detector": True})
    trace = _run(
        scene_paths,
        store,
        "what changed between these two images",
        "s2_pre",
        "s2_post",
        registry=available,
        tools={"siamese_change_detector": ExplodingDetector()},
    )

    detector = next(e for e in trace.executions if e.step == 2)
    assert detector.tool == "image_diff_change"
    assert detector.status is ToolStatus.DEGRADED
    assert detector.fallback_of == "siamese_change_detector"
    assert detector.scalars["changed_area_pct"] > 0
    assert any(error.code == "TOOL_FAILED" for error in trace.errors)
    # The analysis still answered, and the degradation is priced in.
    assert "degraded_execution" in trace.confidence.caps_applied


def test_18_unavailable_tool_is_substituted_by_capability_matching(
    scene_paths: dict[str, Path], store: ArtifactStore, registry: Any
) -> None:
    _, result = _ingest(scene_paths, "s2_pre")
    from satquery.agent.pipeline import bundles_for

    request = AnalysisRequest(
        query="outline the water bodies",
        sources=_sources(scene_paths, "s2_pre"),
        ingest=result,
    )
    decision = match(
        tool="semantic_segmenter",
        registry=registry.with_availability({"semantic_segmenter": False}),
        pair_type=PairType.SINGLE,
        images=bundles_for(request),
    )
    assert decision.status is MatchStatus.SUBSTITUTED
    assert decision.tool == "spectral_index_analyzer"

    # Availability is pinned rather than inherited from the machine: with a
    # segmentation checkpoint actually installed the substitution correctly does
    # not happen, and the test would be asserting a property of the developer's
    # filesystem instead of one of the capability matcher.
    trace = _run(
        scene_paths,
        store,
        "outline the water bodies",
        "s2_pre",
        registry=registry.with_availability({"semantic_segmenter": False}),
    )
    segmenter = next(e for e in trace.executions if e.step == 3)
    assert segmenter.tool == "spectral_index_analyzer"
    assert segmenter.fallback_of == "semantic_segmenter"
    assert segmenter.status is ToolStatus.DEGRADED


def test_19_a_second_identical_request_hits_the_cache(
    scene_paths: dict[str, Path], tmp_path: Path
) -> None:
    cache = ExecutionCache()
    query = "what changed between these two images"
    cold = _run(
        scene_paths, ArtifactStore(tmp_path / "a"), query, "s2_pre", "s2_post", cache=cache
    )
    warm = _run(
        scene_paths, ArtifactStore(tmp_path / "b"), query, "s2_pre", "s2_post", cache=cache
    )

    deterministic = [e for e in warm.executions if e.status is not ToolStatus.SKIPPED]
    assert deterministic, "the plan produced no runnable step"
    assert all(execution.cache_hit for execution in deterministic)

    cold_ms = sum(e.duration_ms for e in cold.executions)
    warm_ms = sum(e.duration_ms for e in warm.executions)
    assert cold_ms > 10 * max(warm_ms, 1)
    # A replay must reproduce the evidence, not merely be fast.
    assert cold.fact_sheet == warm.fact_sheet
    assert [len(e.output_refs) for e in cold.executions] == [
        len(e.output_refs) for e in warm.executions
    ]


def test_19b_a_failing_non_optional_step_is_the_one_permitted_500(
    scene_paths: dict[str, Path], store: ArtifactStore, registry: Any
) -> None:
    """``spectral_renderer`` is the only step whose failure is fatal (§6.3)."""

    class BrokenRenderer:
        name = "spectral_renderer"

        def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
            raise ToolError("no views could be rendered")

    with pytest.raises(ToolFailedNoFallbackError):
        _run(
            scene_paths,
            store,
            "describe this scene",
            "s2_pre",
            tools={"spectral_renderer": BrokenRenderer()},
        )


def test_19c_incompatible_inputs_are_refused_rather_than_analysed(
    scene_paths: dict[str, Path], store: ArtifactStore
) -> None:
    """A FAIL compatibility verdict is a clean 422, not a confident answer."""
    sources, result = _ingest(scene_paths, "s2_pre", "s2_disjoint")
    assert result.compatibility.overall is Overall.FAIL

    with pytest.raises(IncompatibleInputsError) as raised:
        asyncio.run(
            analyze(
                AnalysisRequest(query="what changed", sources=sources, ingest=result),
                store=store,
            )
        )
    error = raised.value.to_api_error()
    assert error.http_status == 422
    assert error.code == "INSUFFICIENT_OVERLAP"
    assert error.hint


# ------------------------------------------------------------- capability edges


def test_gsd_out_of_range_warns_but_still_runs(
    scene_paths: dict[str, Path], registry: Any
) -> None:
    """The one soft check (§5, resolved ambiguity 9).

    A tool trained at one resolution applied at another is extrapolating, not
    being misused. Refusing would turn a degraded answer into no answer, which
    on a hidden evaluation set is the worse failure by a wide margin.
    """
    from satquery.agent.pipeline import bundles_for

    sources, result = _ingest(scene_paths, "cartosat_pan")
    images = bundles_for(
        AnalysisRequest(query="describe this scene", sources=sources, ingest=result)
    )
    assert images[0].manifest.gsd_m is not None

    narrow = registry["raster_statistics"].model_copy(
        update={"accepts": registry["raster_statistics"].accepts.model_copy(
            update={"gsd_range_m": [5.0, 30.0]}
        )}
    )
    decision = match(
        tool="raster_statistics",
        registry=ToolRegistry(
            version=registry.version, tools={**registry.tools, "raster_statistics": narrow}
        ),
        pair_type=PairType.SINGLE,
        images=images,
    )

    assert decision.status is MatchStatus.OK, "a GSD mismatch must never block a step"
    assert decision.gsd_out_of_range
    assert [check.name for check in decision.warnings] == ["gsd"]

    # It is not free, though: it costs input_quality, and therefore confidence.
    penalised = confidence_module.input_quality(
        result.inputs, Overall.PASS, gsd_out_of_range=True
    )
    clean = confidence_module.input_quality(result.inputs, Overall.PASS)
    assert penalised < clean


def test_a_scalar_the_registry_does_not_declare_is_dropped(registry: Any) -> None:
    """The FactSheet is the citation ground truth, so it admits nothing undeclared."""
    from satquery.schemas.enums import Device
    from satquery.schemas.tool import Execution

    execution = Execution(
        step=2,
        tool="change_statistics",
        version="1.0.0",
        status=ToolStatus.OK,
        device_used=Device.CPU,
        duration_ms=1,
        params={},
        scalars={"changed_area_pct": 7.4, "invented_metric": 99.0},
        confidence=0.9,
    )
    sheet = fact_sheet_module.build([execution], registry)

    assert "change_statistics.changed_area_pct" in sheet
    assert "change_statistics.invented_metric" not in sheet
    assert any("invented_metric" in violation for violation in sheet.violations)


def test_stage_c_recovers_a_query_the_rules_only_default_on() -> None:
    """Stage C exists to rescue the 0.70 default, and only runs when it is reached."""
    outcome = task_classifier.stage_c(
        "count the water bodies", {TaskType.COUNT, TaskType.VQA, TaskType.CAPTION}
    )
    assert outcome.task is TaskType.COUNT
    assert outcome.confidence > 0


# --------------------------------------------------------------------- evidence


def test_20_an_uncited_number_is_flagged_and_clamps_confidence(
    scene_paths: dict[str, Path], store: ArtifactStore, registry: Any
) -> None:
    """The honesty signal, end to end: flag the span, then cap the score."""
    trace = _run(scene_paths, store, "what changed between these two images",
                 "s2_pre", "s2_post")
    sheet = fact_sheet_module.build(trace.executions, registry)

    injected = f"{trace.answer.text} About 42% of the area is affected."
    result = validate(injected, sheet, policy=CitationPolicy.FLAG)

    assert "42%" in result.uncited_numeric_spans
    assert result.citations, "the genuine numbers must still resolve"
    assert all(citation.claim != "42%" for citation in result.citations)

    conditions = confidence_module.Conditions(
        overall_compat=Overall.PASS,
        task_proposed_by_llm=False,
        template_fallback=False,
        has_uncited_claims=True,
        generic_plan=False,
    )
    score = confidence_module.score(
        task_confidence=0.92,
        manifests=trace.inputs,
        executions=trace.executions,
        registry=registry,
        sheet=sheet,
        conditions=conditions,
    )
    assert "uncited_claims" in score.caps_applied
    assert score.overall <= 0.60


def test_20b_strip_policy_removes_the_offending_sentence(
    scene_paths: dict[str, Path], store: ArtifactStore, registry: Any
) -> None:
    trace = _run(scene_paths, store, "what changed between these two images",
                 "s2_pre", "s2_post")
    sheet = fact_sheet_module.build(trace.executions, registry)
    injected = f"{trace.answer.text} About 42% of the area is affected."

    stripped = validate(injected, sheet, policy=CitationPolicy.STRIP)
    assert stripped.uncited_numeric_spans == []
    assert "42%" not in stripped.text
    assert stripped.citations


# ------------------------------------------------- silent bi-temporal failures


def test_two_uploads_sharing_a_filename_stay_two_images(
    scene_paths: dict[str, Path],
) -> None:
    """A pair must join to its files by position, never by name.

    Change datasets are organised by epoch directory, so both halves of a pair
    routinely arrive under one basename — LEVIR-CD ships ``A/test_100.png`` and
    ``B/test_100.png``. Joining on the filename collapsed them onto whichever
    file was written last, and the detector then compared the post-change image
    against itself: a black mask and 0% changed area, for every scene, with no
    error anywhere in the trace to say so.
    """
    from satquery.agent.pipeline import bundles_for

    paths = [scene_paths["s2_pre"], scene_paths["s2_post"]]
    shared = paths[0].name
    sources = [SourceImage(path=path, filename=shared) for path in paths]

    images = bundles_for(
        AnalysisRequest(query="what changed?", sources=sources, ingest=ingest(sources))
    )

    assert [image.path for image in images] == paths, "the two epochs collapsed onto one file"
    assert images[0].path != images[1].path


def test_a_skipped_optional_step_does_not_cascade(
    scene_paths: dict[str, Path], store: ArtifactStore, registry: Any
) -> None:
    """An enrichment step that cannot run must not take the synthesiser with it.

    ``spectral_index_analyzer`` needs NIR, which RGB imagery does not carry, so
    it skips — and the change entry lists it among ``vlm_change_vqa``'s
    dependencies only for ordering, since the synthesiser reads ``@1`` and
    ``@2:CHANGE_MASK``. Cascading the skip answered a change question from a
    template while a measured mask sat unused in the evidence gallery.
    """

    class NoIndices:
        name = "spectral_index_analyzer"

        def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
            raise ToolError("no NIR band on either epoch")

    trace = _run(
        scene_paths,
        store,
        "what changed between these two images",
        "s2_pre",
        "s2_post",
        tools={"spectral_index_analyzer": NoIndices()},
    )

    indices = next(e for e in trace.executions if e.tool == "spectral_index_analyzer")
    assert indices.status in {ToolStatus.SKIPPED, ToolStatus.FAILED}

    # Whether the synthesiser then runs depends on a VLM backend being servable,
    # which a unit run has no business requiring. What is asserted is the thing
    # under test: it was never turned away at the dependency gate.
    synthesiser = next(e for e in trace.executions if e.tool == "vlm_change_vqa")
    assert "did not produce output" not in (synthesiser.error or ""), (
        "an optional dependency's skip must not skip the synthesiser"
    )
    assert not any(
        error.code == "DEPENDENCY_SKIPPED" and error.step == synthesiser.step
        for error in trace.errors
    )
    # The reduced evidence is recorded rather than passed over in silence.
    assert any(warning.code == "DEPENDENCY_INCOMPLETE" for warning in trace.warnings)


def test_a_coverage_question_routes_to_segmentation() -> None:
    """"What percentage is covered by X" is the segmentation question.

    It named no tool and matched no rule, so it fell through to the nearest seed
    neighbour and came back VQA — a plan with no segmenter in it. The FactSheet
    then carried no class fractions and the VLM answered the proportion question
    from the pixels alone, with "Forest: 100%".
    """
    coverage = task_classifier.classify(
        "What percentage of this area is covered by forest versus barren land?",
        PairType.SINGLE,
    )
    assert coverage.primary is TaskType.SEGMENTATION

    # The neighbouring intents must not be swallowed by it.
    assert (
        task_classifier.classify("How many buildings are there?", PairType.SINGLE).primary
        is TaskType.COUNT
    )
    assert (
        task_classifier.classify(
            "What percentage of the scene changed between the two dates?",
            PairType.BI_TEMPORAL,
        ).primary
        is TaskType.CHANGE_VQA
    )


def test_the_cache_key_moves_when_the_checkpoint_does(tmp_path: Path) -> None:
    """Weights are part of a step's identity, and the registry version is not.

    ``tool_version`` is a constant in the registry YAML, so it does not move when
    a checkpoint is retrained or repointed. Without the weights in the key, every
    pair analysed before a retrain keeps returning the old model's mask.
    """
    from satquery.tools.catalog import checkpoint_fingerprint

    bundle = tmp_path / "detector.ckpt.pt"
    bundle.write_bytes(b"weights")
    monkey = pytest.MonkeyPatch()
    monkey.setenv("SATQUERY_CD_CHECKPOINT", str(bundle))
    try:
        before = checkpoint_fingerprint("siamese_change_detector")
        bundle.write_bytes(b"retrained weights, a different size")
        after = checkpoint_fingerprint("siamese_change_detector")
    finally:
        monkey.undo()

    assert before != after, "a rewritten checkpoint must invalidate the cache"
    assert checkpoint_fingerprint("raster_statistics") == "", (
        "a tool with no weights must not pay for a stat on every step"
    )


# ---------------------------------------------- process-wide gates + cancel


def test_two_executors_share_one_gpu_permit(
    scene_paths: dict[str, Path], tmp_path: Path, registry: Any
) -> None:
    """One ``DeviceGates`` across two requests: their GPU steps never overlap."""
    import threading
    import time

    from satquery.agent.concurrency import DeviceGates

    peak = {"current": 0, "max": 0}
    lock = threading.Lock()

    class CountingDetector:
        """A ROCm-declared tool that records how many of it run at once."""

        name = "siamese_change_detector"

        def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
            with lock:
                peak["current"] += 1
                peak["max"] = max(peak["max"], peak["current"])
            time.sleep(0.2)
            with lock:
                peak["current"] -= 1
            raise ToolError("counted; let the fallback answer")

    available = registry.with_availability({"siamese_change_detector": True})
    gates = DeviceGates(max_parallel_gpu=1)
    sources, result = _ingest(scene_paths, "s2_pre", "s2_post")

    async def both() -> None:
        def one(name: str) -> Any:
            return analyze(
                AnalysisRequest(
                    query="what changed between these two images", sources=sources, ingest=result
                ),
                store=ArtifactStore(tmp_path / name),
                registry=available,
                tools={"siamese_change_detector": CountingDetector()},
                cache=ExecutionCache(),
                gates=gates,
            )

        await asyncio.gather(one("a"), one("b"))

    asyncio.run(both())
    assert peak["max"] == 1, "two requests each held their own GPU permit"


def test_a_cancelled_run_skips_what_has_not_started(
    scene_paths: dict[str, Path], store: ArtifactStore
) -> None:
    """The executor's half of DELETE: stop at the next boundary, record the rest SKIPPED."""
    cancelled = asyncio.Event()

    class CancelAfterRender:
        """Wraps the renderer; sets the flag once the first wave is done."""

        name = "spectral_renderer"

        def __init__(self) -> None:
            from satquery.tools.catalog import BUILTIN_TOOLS

            self.inner = BUILTIN_TOOLS["spectral_renderer"]

        def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
            out = self.inner.run(ctx, params)
            cancelled.set()
            return out

    sources, result = _ingest(scene_paths, "s2_pre", "s2_post")
    trace = asyncio.run(
        analyze(
            AnalysisRequest(
                query="what changed between these two images",
                sources=sources,
                ingest=result,
                cancelled=cancelled,
            ),
            store=store,
            tools={"spectral_renderer": CancelAfterRender()},
            cache=ExecutionCache(),
        )
    ).trace

    statuses = {e.step: e.status for e in trace.executions}
    assert statuses[1] is ToolStatus.OK, "the wave in flight finishes"
    assert all(s is ToolStatus.SKIPPED for step, s in statuses.items() if step > 1)
    assert all("cancelled" in (e.error or "") for e in trace.executions if e.step > 1)
    assert any(w.code == "RUN_CANCELLED" for w in trace.warnings)


def test_the_execution_cache_is_bounded_by_bytes_not_only_entries() -> None:
    from satquery.agent.executor import CachedStep, CapturedArtifact
    from satquery.schemas.trace import ArtifactRef

    def entry(size: int) -> CachedStep:
        ref = ArtifactRef(
            id="a",
            type=ArtifactType.RENDERED_VIEW,
            label="x",
            url=None,
            mime="image/png",
            produced_by_step=1,
        )
        return CachedStep(
            result=ToolResult(), artifacts=(CapturedArtifact(ref, {"png": b"x" * size}),)
        )

    cache = ExecutionCache(max_entries=100, max_bytes=1_000)
    cache.put("one", entry(400))
    cache.put("two", entry(400))
    assert len(cache) == 2 and cache.resident_bytes == 800
    cache.put("three", entry(400))  # evicts "one"
    assert cache.get("one") is None and cache.get("three") is not None
    assert cache.resident_bytes == 800
    cache.put("huge", entry(5_000))  # larger than the whole budget: not cached
    assert cache.get("huge") is None
    cache.clear()
    assert cache.resident_bytes == 0
