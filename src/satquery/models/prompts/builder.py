"""Assemble one prompt out of a FactSheet, the rendered views and a question.

The builder owns three responsibilities and delegates everything else:

* **It injects the FactSheet verbatim.** Values are printed with
  :func:`~satquery.evidence.citation_validator.format_number`, the same formatter
  the templated answers use, so a number the model copies straight out of the
  sheet is guaranteed to resolve back to its own fact under the validator's
  tolerance. Printing at some other precision here would let a model that obeyed
  every rule still fail the citation check.
* **It never writes a view label.** Labels come from
  :mod:`satquery.render.view_labels` through the renderer, because the corpus
  builder uses that module too and DATA_ADAPTATION_PLAN §2.5 makes a divergence
  between the two a defect rather than a bug.
* **It reads the citation markers back off the generated text.** The model is
  asked to write ``7.42% [change_statistics.changed_area_pct]``; the tools show
  the reader ``7.42%`` and keep the key as the signal that the model quoted the
  measurement it meant to.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from satquery.evidence.citation_validator import format_number
from satquery.evidence.fact_sheet import Fact, FactSheet
from satquery.models.loader import CHAT_STOP_STRINGS, GenerationRequest, PromptImage
from satquery.models.prompts.templates import PromptTemplate, get_template
from satquery.schemas.enums import PairType, TaskType

CITATION_MARKER: Final[re.Pattern[str]] = re.compile(
    r"\s*\[\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_]+)\s*\]"
)
"""``[tool.scalar]``, with the whitespace in front of it, so removing a marker
does not leave a double space behind."""

MAX_VIEWS: Final[int] = 6
"""Views per prompt. A bi-temporal optical plan renders six, which is the most
the 22 GB budget accommodates at 448 px alongside an 8B model in bf16."""

_UNIT_HINTS: Final[dict[str, str]] = {
    "pct": "percent",
    "km2": "square kilometres",
    "m2": "square metres",
    "db": "decibels",
    "px": "pixels",
    "count": "a count",
}
"""Read off the scalar's own name segments. The model is told what a number *is*
so it does not narrate a percentage as an area — a mistake that produces a
sentence which passes citation validation and is nonetheless wrong."""


@dataclass(frozen=True)
class ViewInput:
    """One rendered view on its way into the prompt."""

    label: str
    rgb: object
    """The ``(H, W, 3)`` uint8 array the renderer produced. Typed loosely here to
    keep the prompt layer free of a numpy dependency in its signature; it is
    validated when :class:`~satquery.models.loader.PromptImage` wraps it."""


@dataclass(frozen=True)
class BuiltPrompt:
    """A prompt, and everything about it a trace should be able to show."""

    system: str
    user: str
    images: tuple[PromptImage, ...]
    prompt_version: str
    fact_keys: tuple[str, ...]
    view_labels: tuple[str, ...]

    def to_request(
        self,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        seed: int = 0,
        stop: Sequence[str] = CHAT_STOP_STRINGS,
    ) -> GenerationRequest:
        """Turn the prompt into a backend-agnostic generation request.

        The stop set is passed explicitly rather than left to the dataclass
        default: this is the only place in the system that builds a generation
        request, so it is the only place where forgetting it goes unnoticed.
        """
        return GenerationRequest(
            system=self.system,
            user=self.user,
            images=self.images,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            stop=tuple(stop),
        )


@dataclass
class MarkedAnswer:
    """A generated answer split into prose and the citation keys it carried."""

    text: str
    """The answer with the ``[key]`` markers removed — what a human reads."""

    cited_keys: list[str] = field(default_factory=list)
    """Every key the model quoted, in order, including repeats."""

    unknown_keys: list[str] = field(default_factory=list)
    """Keys the model wrote that are not in the FactSheet. A model that invents a
    key has invented the measurement behind it, which is worth surfacing even
    when the number itself happens to land within tolerance of a real fact."""


def unit_hint(scalar: str) -> str | None:
    """Name the unit a scalar carries, read off its own name segments."""
    for segment in scalar.lower().split("_"):
        hint = _UNIT_HINTS.get(segment)
        if hint is not None:
            return hint
    return None


def render_fact(fact: Fact) -> str:
    """One line of the FACT SHEET block."""
    if fact.is_numeric:
        value = format_number(float(fact.value))
        hint = unit_hint(fact.scalar)
        suffix = f"  ({hint})" if hint else ""
    else:
        value = str(fact.value)
        suffix = ""
    return f"  {fact.key} = {value}{suffix}"


def render_fact_sheet(sheet: FactSheet, template: PromptTemplate) -> str:
    """Render the whole sheet as the block injected into the system prompt."""
    if not sheet.facts:
        return template.empty_facts
    lines = [template.facts_header]
    lines.extend(render_fact(fact) for fact in sheet.facts.values())
    return "\n".join(lines)


def render_view_index(labels: Sequence[str]) -> str:
    """The block naming each attached view, in the order it is attached."""
    if not labels:
        return "IMAGES — none were rendered for this scene."
    lines = ["IMAGES — attached in this order, and referred to by these exact labels:"]
    lines.extend(f"  {label}" for label in labels)
    return "\n".join(lines)


def render_context(pair_type: PairType, slots: Mapping[str, object]) -> str:
    """State what the analyst asked *about*, where the slots say so.

    Only slots with a value are printed. A line reading ``target_class: None``
    invites the model to reason about the absence, which is never what an unfilled
    optional slot means.
    """
    filled = [
        f"  {name}: {value}"
        for name, value in sorted(slots.items())
        if value not in (None, "", [], {})
    ]
    lines = [f"CONTEXT — pair type: {pair_type.value}."]
    if filled:
        lines.append("The query parser resolved these constraints:")
        lines.extend(filled)
    return "\n".join(lines)


def build_system_prompt(
    task: TaskType,
    pair_type: PairType,
    sheet: FactSheet,
    view_labels: Sequence[str],
    slots: Mapping[str, object] | None = None,
    template: PromptTemplate | None = None,
    mode: str | None = None,
) -> str:
    """Build the FactSheet-constrained system prompt.

    Ordering is deliberate: persona, then the evidence, then the rules, then the
    task. The rules sit *after* the sheet so the sentence "nothing outside this
    list is a measurement" has the list immediately above it, and the task sits
    last because it is the instruction the model acts on.
    """
    resolved = template or get_template()
    return "\n\n".join(
        (
            resolved.persona,
            render_view_index(view_labels),
            render_fact_sheet(sheet, resolved),
            render_context(pair_type, slots or {}),
            resolved.rules,
            resolved.instruction_for(task, mode),
        )
    )


def build_user_prompt(task: TaskType, question: str) -> str:
    """The analyst's turn.

    A caption task carries no question, so one is supplied rather than sending an
    empty user turn — an instruction-tuned model given an empty turn answers the
    system prompt, which produces a description of the rules instead of the scene.
    """
    text = question.strip()
    if text:
        return text
    if task in {TaskType.CAPTION, TaskType.CHANGE_CAPTION}:
        return "Describe this scene."
    return "Report what these views and the measurements above establish."


def build_prompt(
    task: TaskType,
    pair_type: PairType,
    sheet: FactSheet,
    views: Iterable[ViewInput],
    question: str = "",
    slots: Mapping[str, object] | None = None,
    prompt_version: str | None = None,
    max_views: int = MAX_VIEWS,
    mode: str | None = None,
) -> BuiltPrompt:
    """Assemble the complete prompt for one ``vlm_*`` step.

    Args:
        task: The resolved primary task; selects the task instruction.
        pair_type: Recorded in the context block.
        sheet: Every measurement the plan produced so far — the only numbers the
            model is permitted to state.
        views: The rendered views, in slot order, with their canonical labels.
        question: The analyst's raw query. Empty for a caption.
        slots: Resolved query slots, printed when filled.
        prompt_version: Template version; defaults to the current one.
        max_views: Cap on attached images, for the VRAM budget.
        mode: The policy table's ``mode`` for this step. Selects a
            synthesiser-specific instruction where the template defines one;
            the grounding *tool* passes none and keeps the frozen box format.

    Returns:
        The assembled :class:`BuiltPrompt`.
    """
    template = get_template(prompt_version)
    selected = list(views)[:max_views]
    images = tuple(PromptImage(label=view.label, rgb=view.rgb) for view in selected)  # type: ignore[arg-type]
    labels = tuple(image.label for image in images)
    return BuiltPrompt(
        system=build_system_prompt(
            task, pair_type, sheet, labels, slots, template, mode
        ),
        user=build_user_prompt(task, question),
        images=images,
        prompt_version=template.version,
        fact_keys=tuple(sheet.facts),
        view_labels=labels,
    )


def strip_citation_markers(text: str, sheet: FactSheet | None = None) -> MarkedAnswer:
    """Split ``[key]`` markers off the generated text.

    The markers are how the prompt makes citation mechanically checkable, but
    they are not prose. The reader gets the sentence; the trace gets the keys.

    Args:
        text: The raw generation.
        sheet: When given, keys absent from it are reported as unknown.

    Returns:
        A :class:`MarkedAnswer` with the cleaned text and the keys it carried.
    """
    cited: list[str] = []
    unknown: list[str] = []

    def _take(match: re.Match[str]) -> str:
        key = match.group("key")
        cited.append(key)
        if sheet is not None and key not in sheet:
            unknown.append(key)
        return ""

    cleaned = CITATION_MARKER.sub(_take, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return MarkedAnswer(text=cleaned, cited_keys=cited, unknown_keys=unknown)


__all__ = [
    "CITATION_MARKER",
    "MAX_VIEWS",
    "BuiltPrompt",
    "MarkedAnswer",
    "ViewInput",
    "build_prompt",
    "build_system_prompt",
    "build_user_prompt",
    "render_context",
    "render_fact",
    "render_fact_sheet",
    "render_view_index",
    "strip_citation_markers",
    "unit_hint",
]
