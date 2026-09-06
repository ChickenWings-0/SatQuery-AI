"""Phase 4: the VLM serving stack, the prompt library and the two synthesisers.

The 8B weights are never read here. A real generation would make the suite depend
on a 16 GB download and a working ROCm stack, and would test Qwen rather than
testing us — so every test below drives a :class:`ScriptedBackend` that returns a
fixed string through the *real* backend interface. What is under test is the
thing we actually wrote: that the FactSheet reaches the prompt intact, that a
number without a measurement behind it is caught, and that a machine with no
weights degrades to the templated answer instead of failing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from satquery.agent.executor import ExecutionCache, cache_key
from satquery.agent.pipeline import AnalysisRequest, analyze
from satquery.evidence.fact_sheet import Fact, FactSheet
from satquery.ingest.pipeline import SourceImage, ingest
from satquery.models import loader
from satquery.models.hf_backend import build_messages as hf_messages
from satquery.models.llamacpp_client import (
    LlamaCppBackend,
    encode_data_uri,
)
from satquery.models.llamacpp_client import build_messages as gguf_messages
from satquery.models.loader import (
    BackendConfig,
    BackendKind,
    GenerationRequest,
    GenerationResult,
    LazyBackend,
    PromptImage,
    VlmBackend,
    VramGuardError,
    estimate_weight_bytes,
    guard_vram,
    resolve_torch_device,
)
from satquery.models.prompts import builder, templates
from satquery.registry.registry import default_registry
from satquery.render.artifact_store import ArtifactStore
from satquery.render.view_labels import label_for_view
from satquery.schemas.enums import (
    ArtifactType,
    ImageRole,
    Modality,
    PairType,
    TaskType,
    ToolStatus,
)
from satquery.tools.base import ToolContext, ToolError
from satquery.tools.vlm_caption import VlmCaption
from satquery.tools.vlm_vqa import VlmVqa

VLM_TOOLS = ("vlm_vqa", "vlm_caption")


# --------------------------------------------------------------------- doubles


@dataclass
class ScriptedBackend(LazyBackend):
    """A backend that returns a fixed answer through the real interface.

    Subclassing :class:`LazyBackend` rather than faking the protocol wholesale is
    deliberate: the lazy-load, unload and locking behaviour the production
    backends inherit is exercised by every test that uses this.
    """

    kind = BackendKind.HF

    def __init__(self, answer: str = "", truncated: bool = False) -> None:
        """Script a fixed answer, and count how often the weights are brought up."""
        super().__init__(BackendConfig(model_id="test/qwen3-vl-stub", device="cpu"))
        self.answer = answer
        self.truncated = truncated
        self.requests: list[GenerationRequest] = []
        self.loads = 0
        self.unloads = 0

    def _load(self) -> None:
        self.loads += 1

    def _unload(self) -> None:
        self.unloads += 1

    def _generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(
            text=self.answer,
            backend=self.kind,
            model_id=self.model_id,
            device="cpu",
            prompt_tokens=128,
            completion_tokens=32,
            truncated=self.truncated,
        )


def _sheet(**values: float | str) -> FactSheet:
    sheet = FactSheet()
    for index, (key, value) in enumerate(values.items(), start=1):
        tool, _, scalar = key.rpartition("__")
        namespaced = f"{tool}.{scalar}"
        sheet.facts[namespaced] = Fact(
            key=namespaced, tool=tool, scalar=scalar, step=index, value=value
        )
    return sheet


def _view(label: str, size: int = 8) -> builder.ViewInput:
    rgb = np.full((size, size, 3), 128, dtype=np.uint8)
    return builder.ViewInput(label=label, rgb=rgb)


def _context(
    sheet: FactSheet,
    question: str,
    artifacts: list[Any] | None = None,
    data: dict[str, Any] | None = None,
) -> ToolContext:
    return ToolContext(
        trace_id="tr_test",
        step=4,
        pair_type=PairType.SINGLE,
        images=[],
        artifacts=artifacts or [],
        data=data or {},
        question=question,
        facts=sheet,
    )


# ---------------------------------------------------------------------- device


def test_rocm_device_strings_map_onto_the_cuda_api_torch_exposes() -> None:
    """ROCm builds of torch address HIP devices as ``cuda:N``."""
    assert resolve_torch_device("rocm:0") == "cuda:0"
    assert resolve_torch_device("rocm:1") == "cuda:1"
    assert resolve_torch_device("cpu") == "cpu"


def test_the_vram_guard_refuses_a_load_that_would_pass_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 16 GB load onto a card already holding 10 GB must not be attempted."""
    monkeypatch.setattr(
        loader,
        "vram_snapshot",
        lambda device="cuda:0": loader.VramSnapshot(
            device=device, total=24 * 1024**3, free=14 * 1024**3
        ),
    )
    with pytest.raises(VramGuardError):
        guard_vram(estimate_weight_bytes(), device="cuda:0")


def test_the_vram_guard_admits_the_load_it_was_sized_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """8B in bf16 plus activations fits on an idle 24 GB card under a 22 GB cap."""
    monkeypatch.setattr(
        loader,
        "vram_snapshot",
        lambda device="cuda:0": loader.VramSnapshot(
            device=device, total=24 * 1024**3, free=24 * 1024**3
        ),
    )
    assert estimate_weight_bytes() < loader.VRAM_BUDGET_BYTES
    guard_vram(estimate_weight_bytes(), device="cuda:0")


def test_a_machine_with_no_device_is_slow_rather_than_refused() -> None:
    """Without a GPU there is no VRAM to exhaust; a CPU load is merely slow."""
    guard_vram(64 * 1024**3, device="cpu")


# --------------------------------------------------------------------- loading


def test_the_weights_are_not_read_until_something_generates() -> None:
    """Constructing a backend must cost nothing — the registry constructs one."""
    backend = ScriptedBackend(answer="hello")
    assert not backend.is_loaded
    assert backend.loads == 0

    backend.generate(GenerationRequest(system="s", user="u"))
    assert backend.is_loaded
    assert backend.loads == 1

    backend.generate(GenerationRequest(system="s", user="u"))
    assert backend.loads == 1, "a second generation must not reload the weights"


def test_an_idle_backend_gives_the_card_back() -> None:
    """Phase 5's CV models want the VRAM an idle VLM is sitting on."""
    backend = ScriptedBackend(answer="hello")
    backend.generate(GenerationRequest(system="s", user="u"))

    assert backend.unload_if_idle(idle_s=3600.0) is False
    assert backend.unload_if_idle(idle_s=0.0) is True
    assert not backend.is_loaded and backend.unloads == 1
    assert backend.unload_if_idle(idle_s=0.0) is False, "unloading twice is not an unload"


def test_a_machine_with_no_weights_reports_no_backend(tmp_path: Path) -> None:
    """The offline probe never contacts the hub and never reads a weight file."""
    config = BackendConfig(model_path=tmp_path / "absent", gguf_path=tmp_path / "absent.gguf")
    assert loader.available_backend(config) is BackendKind.NONE


def test_a_configured_llamacpp_server_is_a_backend() -> None:
    """The offline demo path is available on its address alone, unprobed."""
    config = BackendConfig(kind=BackendKind.LLAMACPP, server_url="http://127.0.0.1:8080")
    assert loader.available_backend(config) is BackendKind.LLAMACPP


def test_an_explicitly_disabled_vlm_stays_disabled() -> None:
    """``SATQUERY_VLM_BACKEND=none`` is the deterministic-only baseline."""
    config = BackendConfig(kind=BackendKind.NONE, server_url="http://127.0.0.1:8080")
    assert loader.available_backend(config) is BackendKind.NONE


def test_both_backends_satisfy_the_one_generation_interface() -> None:
    """Interchangeability is the whole reason the offline path is affordable."""
    assert isinstance(ScriptedBackend(), VlmBackend)
    assert isinstance(
        LlamaCppBackend(BackendConfig(server_url="http://127.0.0.1:8080")), VlmBackend
    )


# ---------------------------------------------------------------------- prompt


def test_the_prompt_hands_over_every_measured_number_and_its_key() -> None:
    """The model cannot cite what it was not shown, and must not state anything else."""
    sheet = _sheet(
        change_statistics__changed_area_pct=7.4213,
        spectral_index_analyzer__ndvi_mean=0.62,
    )
    system = builder.build_system_prompt(
        TaskType.CHANGE_VQA, PairType.BI_TEMPORAL, sheet, ["Image 1 (optical true colour)"]
    )
    assert "change_statistics.changed_area_pct = 7.42" in system
    assert "spectral_index_analyzer.ndvi_mean = 0.62" in system
    assert "(percent)" in system, "the model must be told what a number is"
    assert "Nothing outside this list is a measurement." in system


def test_a_number_copied_out_of_the_prompt_resolves_back_to_its_own_fact() -> None:
    """The sheet is printed with the formatter the validator's tolerance assumes.

    If these two ever disagree, a model that obeyed every rule still fails the
    citation check — silently, and only on the values that round badly.
    """
    from satquery.evidence.citation_validator import validate

    sheet = _sheet(
        change_statistics__changed_area_pct=7.4213,
        change_statistics__changed_area_km2=1.0749,
        sar_backscatter_analyzer__sigma0_vv_db_mean=-8.4391,
    )
    system = builder.build_system_prompt(TaskType.VQA, PairType.SINGLE, sheet, [])
    for fact in sheet.facts.values():
        printed = builder.format_number(float(fact.value))
        assert f"= {printed}" in system
        unit = {"pct": "%", "km2": " km2", "db": " dB"}[fact.scalar.split("_")[-1]
            if fact.scalar.split("_")[-1] in {"pct", "km2"} else "db"]
        result = validate(f"the value is {printed}{unit}.", sheet)
        assert result.citations, f"{fact.key} printed as {printed} resolved to nothing"


def test_the_prompt_names_views_with_the_renderer_s_own_labels() -> None:
    """DATA_ADAPTATION_PLAN §2.5: corpus and runtime labels are the same string."""
    label = label_for_view("TC", slot=1, modality=Modality.OPTICAL, sensor="Sentinel-2 L2A")
    assert label == "Image 1 (optical true colour, Sentinel-2)"

    prompt = builder.build_prompt(
        task=TaskType.VQA,
        pair_type=PairType.SINGLE,
        sheet=FactSheet(),
        views=[_view(label)],
        question="what is here?",
    )
    assert label in prompt.system
    assert prompt.view_labels == (label,)


def test_a_fixed_scale_view_is_labelled_as_absolute_and_explained() -> None:
    """A heatmap read as a relative stretch produces confidently wrong physics."""
    label = label_for_view(
        "NDVI", slot=2, modality=Modality.OPTICAL, role=ImageRole.PRE
    )
    assert "fixed scale -1 to +1" in label
    system = builder.build_system_prompt(TaskType.VQA, PairType.SINGLE, FactSheet(), [label])
    assert "the colours are absolute" in system


def test_an_empty_fact_sheet_forbids_every_number_rather_than_inviting_one() -> None:
    """No measurement is a reason to say nothing numeric, not a blank to fill in."""
    system = builder.build_system_prompt(TaskType.CAPTION, PairType.SINGLE, FactSheet(), [])
    assert "FACT SHEET — empty" in system
    assert "you may not state any number at all" in system


def test_unfilled_slots_are_omitted_rather_than_printed_as_none() -> None:
    """``target_class: None`` invites the model to reason about the absence."""
    system = builder.build_system_prompt(
        TaskType.VQA,
        PairType.SINGLE,
        FactSheet(),
        [],
        slots={"target_class": None, "output_format": "text"},
    )
    assert "target_class" not in system
    assert "output_format: text" in system


def test_a_caption_gets_a_user_turn_even_though_nobody_asked_a_question() -> None:
    """An empty user turn makes an instruct model answer the system prompt."""
    assert builder.build_user_prompt(TaskType.CAPTION, "") == "Describe this scene."
    assert builder.build_user_prompt(TaskType.VQA, "  how much water?  ") == "how much water?"


def test_prompt_versions_are_frozen_rather_than_edited() -> None:
    """Phase 7 trains against a specific version; silently moving it is the bug."""
    assert templates.get_template("grounded_v1") is templates.GROUNDED_V1
    with pytest.raises(KeyError):
        templates.get_template("grounded_v99")


def test_citation_markers_leave_the_prose_and_stay_in_the_trace() -> None:
    """The reader gets a sentence; the audit gets the keys."""
    sheet = _sheet(change_statistics__changed_area_pct=7.42)
    marked = builder.strip_citation_markers(
        "About 7.42% [change_statistics.changed_area_pct] of the scene changed.", sheet
    )
    assert marked.text == "About 7.42% of the scene changed."
    assert marked.cited_keys == ["change_statistics.changed_area_pct"]
    assert marked.unknown_keys == []


def test_an_invented_citation_key_is_reported_even_when_its_number_is_plausible() -> None:
    """An invented key means an invented measurement, whatever value sits beside it."""
    sheet = _sheet(change_statistics__changed_area_pct=7.42)
    marked = builder.strip_citation_markers(
        "About 7.42% [flood_model.inundated_pct] of the scene flooded.", sheet
    )
    assert marked.unknown_keys == ["flood_model.inundated_pct"]


# -------------------------------------------------------------------- backends


def test_every_view_is_labelled_immediately_before_its_own_pixels() -> None:
    """Each view must be named immediately before its own pixels.

    Naming views only in the system prompt makes the model align six tensors
    against a list, which it does unreliably and which breaks when a slot drops.
    """
    request = GenerationRequest(
        system="rules",
        user="what is here?",
        images=(
            PromptImage("Image 1 (optical true colour, Sentinel-2)", np.zeros((4, 4, 3), np.uint8)),
            PromptImage("Image 2 (optical NDVI heatmap, fixed scale -1 to +1)",
                        np.zeros((4, 4, 3), np.uint8)),
        ),
    )
    for messages in (hf_messages(request), gguf_messages(request)):
        parts = messages[1]["content"]
        assert parts[0]["text"] == request.images[0].label
        assert parts[1]["type"] in {"image", "image_url"}
        assert parts[2]["text"] == request.images[1].label
        assert parts[-1]["text"] == "what is here?"


def test_the_offline_path_sends_images_inline_so_it_needs_no_file_server() -> None:
    """A demo with the network unplugged cannot serve a URL to itself."""
    uri = encode_data_uri(PromptImage("Image 1", np.zeros((4, 4, 3), np.uint8)))
    assert uri.startswith("data:image/jpeg;base64,")


def test_a_view_that_is_not_an_rgb_image_is_rejected_at_the_boundary() -> None:
    with pytest.raises(ValueError):
        PromptImage("Image 1", np.zeros((4, 4), np.uint8))


def test_greedy_decoding_is_what_the_policy_table_asks_for() -> None:
    """§8 can only promise a reproducible answer.text if decoding is greedy."""
    assert GenerationRequest(system="s", user="u", temperature=0.0).greedy
    assert not GenerationRequest(system="s", user="u", temperature=0.7).greedy


# ----------------------------------------------------------------------- tools


def test_the_synthesiser_produces_prose_and_never_evidence() -> None:
    """Its registry entry declares no scalars: a VLM step may not add a number."""
    sheet = _sheet(spectral_index_analyzer__ndvi_mean=0.62)
    backend = ScriptedBackend(
        answer="Mean NDVI is 0.62 [spectral_index_analyzer.ndvi_mean], so the scene is vegetated."
    )
    result = VlmVqa(backend=backend).run(
        _context(sheet, "how green is this?"), {"mode": "vqa", "max_new_tokens": 128}
    )

    assert result.status is ToolStatus.OK
    assert result.scalars == {}
    (artifact,) = result.artifacts
    assert artifact.type is ArtifactType.TEXT
    assert artifact.inline is not None
    assert artifact.inline["text"] == "Mean NDVI is 0.62, so the scene is vegetated."
    assert artifact.inline["cited_keys"] == ["spectral_index_analyzer.ndvi_mean"]
    assert result.params["prompt_version"] == templates.DEFAULT_VERSION
    assert result.params["facts_offered"] == 1


def test_a_number_the_plan_never_measured_degrades_the_step_that_said_it() -> None:
    """This is the one behaviour the whole prompt exists to prevent."""
    sheet = _sheet(spectral_index_analyzer__ndvi_mean=0.62)
    backend = ScriptedBackend(answer="Roughly 41.7% of the scene is under water.")
    result = VlmVqa(backend=backend).run(_context(sheet, "how much water?"), {})

    assert result.status is ToolStatus.DEGRADED
    assert result.confidence < 0.85
    (artifact,) = result.artifacts
    assert artifact.inline is not None
    assert artifact.inline["uncited_numeric_spans"] == ["41.7%"]
    assert any("no measurement behind them" in note for note in result.notes)


def test_the_question_reaches_the_model_and_the_fact_sheet_reaches_the_prompt() -> None:
    sheet = _sheet(change_statistics__changed_area_pct=7.42)
    backend = ScriptedBackend(answer="Nothing numeric here.")
    VlmVqa(backend=backend).run(_context(sheet, "what changed and by how much?"), {})

    (request,) = backend.requests
    assert request.user == "what changed and by how much?"
    assert "change_statistics.changed_area_pct = 7.42" in request.system
    assert request.temperature == 0.0


def test_a_caption_and_a_question_get_different_task_instructions() -> None:
    """One prompt library, two tasks: the constraint is shared, the ask is not."""
    sheet = _sheet(spectral_index_analyzer__ndvi_mean=0.62)
    caption_backend, vqa_backend = ScriptedBackend("A scene."), ScriptedBackend("An answer.")
    VlmCaption(backend=caption_backend).run(_context(sheet, ""), {"mode": "caption"})
    VlmVqa(backend=vqa_backend).run(_context(sheet, "what is here?"), {"mode": "vqa"})

    assert "Describe this scene in two to four sentences" in caption_backend.requests[0].system
    assert "Answer the analyst's question directly" in vqa_backend.requests[0].system


def test_a_truncated_generation_is_recorded_rather_than_presented_as_finished() -> None:
    """A cut-off answer may have lost the citation of its last number."""
    backend = ScriptedBackend(answer="The scene is largely", truncated=True)
    result = VlmVqa(backend=backend).run(_context(FactSheet(), "describe"), {})
    assert result.confidence < 0.85
    assert any("token budget" in note for note in result.notes)


def test_an_unreachable_backend_is_a_tool_failure_not_a_crashed_request() -> None:
    """No weights must degrade the answer, never the service."""

    class Unavailable(ScriptedBackend):
        def _generate(self, request: GenerationRequest) -> GenerationResult:
            raise loader.ModelLoadError("the card is busy")

    with pytest.raises(ToolError):
        VlmVqa(backend=Unavailable()).run(_context(FactSheet(), "what is here?"), {})


def test_the_views_the_model_sees_are_the_ones_the_gallery_shows() -> None:
    """One renderer, two consumers — read from memory, not from a re-decoded blob."""
    from satquery.schemas.trace import ArtifactRef

    rgb = np.full((8, 8, 3), 200, dtype=np.uint8)
    rendered = type("Rendered", (), {"rgb": rgb, "label": "Image 1 (optical true colour)"})()
    reference = ArtifactRef(
        id="art_0",
        type=ArtifactType.RENDERED_VIEW,
        mime="image/jpeg",
        label="True colour",
        produced_by_step=1,
    )
    backend = ScriptedBackend(answer="A description.")
    VlmVqa(backend=backend).run(
        _context(FactSheet(), "describe", [reference], {"art_0": rendered}), {}
    )

    (request,) = backend.requests
    assert request.images[0].label == "Image 1 (optical true colour)"
    assert np.array_equal(request.images[0].rgb, rgb)


def test_two_questions_over_one_scene_do_not_share_a_cached_answer() -> None:
    """A synthesiser's identity includes the question it was asked.

    The standard cache key excludes the query, which is right for every tool but
    this one: a synthesiser's output *is* a function of what was asked.
    """
    registry = default_registry()
    spec = registry["vlm_vqa"]
    keys = {
        cache_key(
            spec.name,
            spec.version,
            {"mode": "vqa"},
            ["sha-of-the-same-image"],
            "single",
            salt=f'{{"question":"{question}"}}',
        )
        for question in ("how much water?", "how much vegetation?")
    }
    assert len(keys) == 2


# -------------------------------------------------------------------- pipeline


def _run_with(
    scene_paths: dict[str, Path], store: ArtifactStore, query: str, backend: VlmBackend, *names: str
) -> Any:
    sources = [
        SourceImage(path=scene_paths[name], filename=scene_paths[name].name) for name in names
    ]
    result = ingest(sources)
    registry = default_registry().with_availability(dict.fromkeys(VLM_TOOLS, True))
    analysis = asyncio.run(
        analyze(
            AnalysisRequest(query=query, sources=sources, ingest=result),
            store=store,
            registry=registry,
            tools={"vlm_vqa": VlmVqa(backend=backend), "vlm_caption": VlmCaption(backend=backend)},
            cache=ExecutionCache(),
        )
    )
    return analysis.trace


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


def test_a_served_model_replaces_the_templated_answer_end_to_end(
    scene_paths: dict[str, Path], store: ArtifactStore
) -> None:
    """The synthesiser is the author of record once there is one."""
    backend = ScriptedBackend(
        answer="The scene is predominantly vegetated, with a river crossing it from north to west."
    )
    trace = _run_with(scene_paths, store, "describe this scene", backend, "s2_pre")

    assert trace.answer.template_fallback is False
    assert trace.answer.text.startswith("The scene is predominantly vegetated")
    assert trace.answer.generator.startswith("hf:")
    assert trace.answer.uncited_numeric_spans == []

    synthesiser = next(e for e in trace.executions if e.tool.startswith("vlm_"))
    assert synthesiser.status is ToolStatus.OK
    assert synthesiser.scalars == {}, "a synthesiser contributes no evidence"


def test_the_model_is_handed_the_evidence_the_policy_table_made_it_wait_for(
    scene_paths: dict[str, Path], store: ArtifactStore
) -> None:
    """A synthesiser sees exactly its upstream steps' measurements, and the views."""
    backend = ScriptedBackend(answer="A description with no numbers in it.")
    _run_with(scene_paths, store, "describe this scene", backend, "s2_pre")

    (request,) = backend.requests
    assert request.images, "the renderer's views must reach the model"
    assert all("Image " in image.label for image in request.images)
    assert "spectral_index_analyzer.ndvi_mean" in request.system


def test_an_invented_number_survives_into_the_trace_as_a_flagged_span(
    scene_paths: dict[str, Path], store: ArtifactStore
) -> None:
    """``flag`` keeps the sentence and shows the judge where the model overreached."""
    backend = ScriptedBackend(answer="Exactly 41.7% of this scene is open water.")
    trace = _run_with(scene_paths, store, "how much water is here?", backend, "s2_pre")

    assert trace.answer.uncited_numeric_spans == ["41.7%"]
    assert "uncited_claims" in trace.confidence.caps_applied
    assert any(w.code == "UNCITED_NUMERIC_SPANS" for w in trace.warnings)


def test_a_machine_with_no_weights_still_answers_from_the_measurements(
    scene_paths: dict[str, Path], store: ArtifactStore
) -> None:
    """Phase 3's behaviour is preserved as the honest degraded mode, not removed."""
    sources = [SourceImage(path=scene_paths["s2_pre"], filename=scene_paths["s2_pre"].name)]
    result = ingest(sources)
    analysis = asyncio.run(
        analyze(
            AnalysisRequest(query="describe this scene", sources=sources, ingest=result),
            store=store,
            registry=default_registry().with_availability(dict.fromkeys(VLM_TOOLS, False)),
            cache=ExecutionCache(),
        )
    )
    trace = analysis.trace
    assert trace.answer.template_fallback is True
    assert trace.answer.uncited_numeric_spans == []
    assert any(e.tool.startswith("vlm_") and e.status is ToolStatus.SKIPPED
               for e in trace.executions)


def test_the_registry_never_advertises_a_model_this_machine_cannot_serve() -> None:
    """A capabilities panel that promises an unloadable tool is worse than a gap."""
    from satquery.tools.catalog import runnable_tools, vlm_servable

    runnable = set(runnable_tools())
    for name in VLM_TOOLS:
        assert (name in runnable) is vlm_servable()
        spec = default_registry()[name]
        assert spec.available is vlm_servable()
        if not spec.available:
            assert spec.unavailable_reason
