"""Shared machinery behind ``vlm_vqa`` and ``vlm_caption``.

Both tools do the same five things — collect the rendered views, build the
FactSheet-constrained prompt, generate, split the citation markers off the prose,
and check the result against the sheet — and differ only in which task the prompt
is written for. Keeping that in one place is what stops the two prompts drifting
apart, which is the failure mode that would silently halve the value of the
adaptation in Phase 7.

The synthesiser is *not* allowed to add evidence. Its registry entry declares an
empty ``scalars_schema``, so any scalar it emitted would be dropped as a
``SCALAR_SCHEMA_VIOLATION`` — which is the design, stated as a schema: a VLM step
consumes measurements and produces prose. Everything it learned about its own
output, including its self-check against the FactSheet, travels in the ``TEXT``
artifact and in the step's notes, where it is auditable but cannot be cited.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from satquery.evidence.citation_validator import validate
from satquery.models.loader import (
    BackendUnavailableError,
    GenerationResult,
    ModelLoadError,
    VlmBackend,
    get_backend,
)
from satquery.models.prompts.box_format import strip_boxes
from satquery.models.prompts.builder import (
    MAX_VIEWS,
    BuiltPrompt,
    MarkedAnswer,
    ViewInput,
    build_prompt,
    strip_citation_markers,
)
from satquery.schemas.enums import ArtifactType, Device, PairType, TaskType, ToolStatus
from satquery.tools.base import (
    ArtifactDraft,
    ToolContext,
    ToolError,
    ToolResult,
)

ANSWER_KEY: Final[str] = "answer"
"""Draft key of the ``TEXT`` artifact every synthesiser produces."""

BASE_CONFIDENCE: Final[float] = 0.85
"""A zero-shot 8B model that obeyed every constraint. Not 1.0: it is still a
language model, and the confidence score is not the place to pretend otherwise.
Phase 7's adapted model raises this; Phase 8's measurement decides by how much."""

UNCITED_PENALTY: Final[float] = 0.25
"""Charged once when the answer states a number it did not cite. This is a strong
signal — the one behaviour the entire prompt exists to prevent."""

UNKNOWN_KEY_PENALTY: Final[float] = 0.20
"""Charged once when the model quoted a FactSheet key that does not exist. An
invented key means an invented measurement, even when the number beside it
happens to land within tolerance of a real one."""

TRUNCATION_PENALTY: Final[float] = 0.15
"""A generation cut off at the token budget may have lost its final citation."""

MIN_CONFIDENCE: Final[float] = 0.10

_BI_TEMPORAL_TASKS: Final[dict[TaskType, TaskType]] = {
    TaskType.VQA: TaskType.CHANGE_VQA,
    TaskType.CAPTION: TaskType.CHANGE_CAPTION,
}
"""The bi-temporal counterpart of each single-scene task."""


@dataclass(frozen=True)
class SynthesisParams:
    """The effective parameters of one synthesis step."""

    mode: str
    max_new_tokens: int
    temperature: float
    max_views: int
    prompt_version: str | None

    @classmethod
    def read(cls, params: Mapping[str, Any], default_mode: str) -> SynthesisParams:
        """Read the policy table's params, filling in this tool's defaults."""
        return cls(
            mode=str(params.get("mode", default_mode)),
            max_new_tokens=int(params.get("max_new_tokens", 384)),
            temperature=float(params.get("temperature", 0.0)),
            max_views=int(params.get("max_views", MAX_VIEWS)),
            prompt_version=(
                str(params["prompt_version"]) if params.get("prompt_version") else None
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """The record the trace shows under ``executions[].params``."""
        recorded: dict[str, Any] = {
            "mode": self.mode,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "max_views": self.max_views,
        }
        if self.prompt_version:
            recorded["prompt_version"] = self.prompt_version
        return recorded


def collect_views(ctx: ToolContext, limit: int = MAX_VIEWS) -> list[ViewInput]:
    """Gather the rendered views this step was given, in artifact order.

    Reads the in-memory payload the renderer left on the context rather than the
    stored PNG: the artifact ids are the same pixels either way, but decoding the
    blob back would let a JPEG round trip sit between the evidence gallery and
    what the model saw, and the whole point of the shared renderer is that those
    are the same image.
    """
    views: list[ViewInput] = []
    for artifact in ctx.artifacts:
        if artifact.type is not ArtifactType.RENDERED_VIEW:
            continue
        payload = ctx.data.get(artifact.id)
        rgb = getattr(payload, "rgb", None)
        if rgb is None:
            continue
        label = getattr(payload, "label", None) or artifact.label
        views.append(ViewInput(label=str(label), rgb=rgb))
        if len(views) >= limit:
            break
    return views


def resolve_task(ctx: ToolContext, mode: str, fallback: TaskType) -> TaskType:
    """Decide which task instruction this step's prompt should carry.

    The policy table's ``mode`` is the authority — it is what the table author
    chose for this entry — and the tool's own default covers the modes the table
    spells differently from a :class:`TaskType`. The pair type has the last word
    over a *generic* mode: ``mode: caption`` on a bi-temporal plan asks for a
    description of the change, not of one of the two scenes.
    """
    candidate = mode.strip().upper()
    if candidate in set(TaskType):
        return TaskType(candidate)
    if ctx.pair_type is PairType.BI_TEMPORAL:
        return _BI_TEMPORAL_TASKS.get(fallback, fallback)
    return fallback


def acquire_backend(backend: VlmBackend | None = None) -> VlmBackend:
    """Return the backend to generate with.

    Raises:
        ToolError: No backend is reachable. Raised as a tool failure so the
            executor degrades the step and the aggregator writes the templated
            answer, rather than the request failing.
    """
    if backend is not None:
        return backend
    try:
        return get_backend()
    except BackendUnavailableError as error:
        raise ToolError(str(error)) from error


def generate(prompt: BuiltPrompt, params: SynthesisParams, ctx: ToolContext,
             backend: VlmBackend | None = None) -> GenerationResult:
    """Run one generation, turning any model-level failure into a tool failure.

    Raises:
        ToolError: The backend could not produce an answer.
    """
    resolved = acquire_backend(backend)
    request = prompt.to_request(
        max_new_tokens=params.max_new_tokens,
        temperature=params.temperature,
        seed=ctx.seed,
    )
    try:
        return resolved.generate(request)
    except ModelLoadError as error:
        raise ToolError(f"{resolved.kind.value} backend failed: {error}") from error


def score(marked: MarkedAnswer, uncited: Sequence[str], truncated: bool) -> float:
    """How much of the answer's own constraint the model actually honoured."""
    confidence = BASE_CONFIDENCE
    if uncited:
        confidence -= UNCITED_PENALTY
    if marked.unknown_keys:
        confidence -= UNKNOWN_KEY_PENALTY
    if truncated:
        confidence -= TRUNCATION_PENALTY
    return round(max(MIN_CONFIDENCE, confidence), 4)


def synthesise(
    ctx: ToolContext,
    params: Mapping[str, Any],
    tool_name: str,
    default_mode: str,
    default_task: TaskType,
    label: str,
    backend: VlmBackend | None = None,
) -> ToolResult:
    """Run one synthesiser end to end.

    Args:
        ctx: The step's resolved inputs, the question and the FactSheet.
        params: The policy table's parameters for this step.
        tool_name: Used only in messages.
        default_mode: This tool's mode when the table does not set one.
        default_task: The task instruction to fall back on.
        label: Display label of the ``TEXT`` artifact.
        backend: Injected in tests; production resolves the process-wide one.

    Returns:
        A :class:`ToolResult` carrying the answer as a ``TEXT`` artifact, with no
        scalars — a synthesiser produces prose, never evidence.

    Raises:
        ToolError: No backend was reachable, or generation failed.
    """
    settings = SynthesisParams.read(params, default_mode)
    views = collect_views(ctx, settings.max_views)
    task = resolve_task(ctx, settings.mode, default_task)

    prompt = build_prompt(
        task=task,
        pair_type=ctx.pair_type,
        sheet=ctx.facts,
        views=views,
        question=ctx.question,
        slots=ctx.slots,
        prompt_version=settings.prompt_version,
        max_views=settings.max_views,
        mode=settings.mode,
    )
    generated = generate(prompt, settings, ctx, backend)
    marked = strip_citation_markers(generated.text, ctx.facts)

    # The tool checks its own output against the same validator the pipeline will
    # run over the final answer. Doing it here is what lets a badly grounded
    # generation lower this step's confidence, instead of only being noticed once
    # the answer is already written.
    # A grounding answer is boxes by contract, so its coordinates are validated
    # as what they are — positions, not measurements. See box_format.strip_boxes.
    claims = strip_boxes(marked.text) if task is TaskType.GROUNDING else marked.text
    # Key-aware: a number the model attached a [key] to must resolve to *that*
    # key. Grounding answers are boxes, whose removal shifts every offset, so
    # they keep the value search (they carry no citations by contract anyway).
    checked = validate(
        claims, ctx.facts, markers=None if task is TaskType.GROUNDING else marked.markers
    )

    notes: list[str] = []
    if not views:
        notes.append("no rendered views reached the synthesiser; the answer is text-only")
    if not ctx.facts.facts:
        notes.append("the FactSheet was empty, so the answer may state no measurement")
    if checked.uncited_numeric_spans:
        notes.append(
            "the answer states numbers with no measurement behind them: "
            + ", ".join(
                f"{span} ({reason})"
                for span, reason in zip(
                    checked.uncited_numeric_spans, checked.uncited_reasons, strict=False
                )
            )
        )
    if marked.unknown_keys:
        notes.append(
            "the answer cited FactSheet keys that do not exist: "
            + ", ".join(sorted(set(marked.unknown_keys)))
        )
    if generated.truncated:
        notes.append(f"generation stopped at the {settings.max_new_tokens}-token budget")

    ungrounded = bool(checked.uncited_numeric_spans) or not marked.text
    status = ToolStatus.DEGRADED if ungrounded else ToolStatus.OK
    if not marked.text:
        notes.append(f"{tool_name} produced an empty answer")

    return ToolResult(
        status=status,
        scalars={},
        artifacts=[
            ArtifactDraft(
                key=ANSWER_KEY,
                type=ArtifactType.TEXT,
                label=label,
                inline={
                    "text": marked.text,
                    "generator": f"{generated.backend.value}:{generated.model_id}",
                    "prompt_version": prompt.prompt_version,
                    "task": task.value,
                    "view_labels": list(prompt.view_labels),
                    "fact_keys": list(prompt.fact_keys),
                    "cited_keys": marked.cited_keys,
                    "unknown_keys": marked.unknown_keys,
                    # Carried so the aggregator's pass over the final answer
                    # can be key-aware too; the two validations then agree.
                    "citation_markers": [[offset, key] for offset, key in marked.markers],
                    "uncited_numeric_spans": checked.uncited_numeric_spans,
                    "uncited_reasons": checked.uncited_reasons,
                    "truncated": generated.truncated,
                },
            )
        ],
        data={ANSWER_KEY: marked.text},
        params={
            **settings.as_dict(),
            "prompt_version": prompt.prompt_version,
            "backend": generated.backend.value,
            "model_id": generated.model_id,
            "views_attached": len(prompt.view_labels),
            "facts_offered": len(prompt.fact_keys),
        },
        confidence=score(marked, checked.uncited_numeric_spans, generated.truncated),
        device=Device.ROCM_0 if generated.device.startswith("cuda") else Device.CPU,
        notes=notes,
    )


__all__ = [
    "ANSWER_KEY",
    "BASE_CONFIDENCE",
    "SynthesisParams",
    "acquire_backend",
    "collect_views",
    "generate",
    "resolve_task",
    "score",
    "synthesise",
]
