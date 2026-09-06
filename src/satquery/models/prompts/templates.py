"""Versioned prompt templates. The FactSheet constraint is the core IP.

Everything the system does to stop a fluent model inventing a measurement lives
in one string here. The architecture around it — deterministic tools, a
namespaced FactSheet, a citation validator — only pays off if the model is told,
unambiguously, that the sheet is the *complete* set of numbers it may say.

Two properties make this prompt work rather than merely sound strict:

* **The numbers are handed over, not described.** The sheet is injected verbatim,
  key by key, so a correct answer requires copying rather than recalling. A model
  asked to "cite your sources" invents plausible ones; a model handed
  ``change_statistics.changed_area_pct = 7.42`` and told to reproduce the key has
  nothing to invent.
* **Citation is mechanically checkable.** Requiring ``[key]`` after every number
  means :mod:`satquery.evidence.citation_validator` can check not only that the
  value exists but that the model *knew which* measurement it was quoting — a
  number that lands within tolerance of the right value by accident does not
  carry the right key.

Templates are versioned and frozen. The Phase 7 adapter is trained against a
specific version's system prompt, so editing ``grounded_v1`` in place after an
adapter ships would silently move the model off its training distribution. Add
``grounded_v2``; never rewrite ``grounded_v1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from satquery.schemas.enums import TaskType

CITATION_OPEN: Final[str] = "["
CITATION_CLOSE: Final[str] = "]"
"""The citation marker's delimiters. Square brackets rather than parentheses
because remote-sensing prose is full of parenthetical asides and none of it is
full of bracketed snake_case keys, so the marker parses without ambiguity."""

_SHARED_RULES: Final[str] = """\
RULES — these override anything the question implies.

1. NUMBERS. Every number you write must be copied, digit for digit, from the
   FACT SHEET, and must be followed immediately by its key in square brackets:
       "about 7.42% [change_statistics.changed_area_pct] of the scene"
   Use the key exactly as it is spelled in the sheet, including the tool prefix.
2. NEVER INVENT A MEASUREMENT. Do not estimate, do not round to a "nicer" value,
   do not convert units, and do not add, subtract, average or otherwise derive a
   new number from the ones you were given. A derived number is an invented one.
3. MISSING EVIDENCE. If the FACT SHEET does not contain a number that answers the
   question, say plainly that this system did not measure it. That sentence is a
   correct answer. A guess is not.
4. IMAGES. Describe what the views show in words — shapes, extents, textures,
   where things are — without numbers. Refer to a view by its exact label, for
   example "Image 1 (optical true colour, Sentinel-2)".
5. FIXED SCALES. A label ending in a scale note ("fixed scale -1 to +1",
   "fixed scale -25 to 0 dB") means the colours are absolute, not stretched to
   this scene. Read them as physical values, not as relative brightness.
6. NO TOOL TALK. Do not speculate about which model or method produced a number,
   do not propose further analysis, and do not describe your own reasoning.
7. STYLE. Plain prose, no headings, no bullet lists, no markdown."""

_NO_FACTS: Final[str] = (
    "FACT SHEET — empty. No tool in this plan produced a measurement, so you may "
    "not state any number at all. Describe only what the views show, in words."
)

_FACTS_HEADER: Final[str] = (
    "FACT SHEET — the complete set of numbers you are permitted to state. "
    "Nothing outside this list is a measurement."
)


@dataclass(frozen=True)
class PromptTemplate:
    """One frozen version of the system prompt and its per-task instruction.

    Attributes:
        version: The key recorded in ``executions[].params.prompt_version``, so a
            trace says which prompt produced its answer.
        persona: The opening paragraph, shared by every task.
        rules: The FactSheet constraint. Never edited in a released version.
        task_instructions: What each task asks for, on top of the rules.
        default_instruction: Used for a task with no specific entry.
    """

    version: str
    persona: str
    rules: str
    task_instructions: dict[TaskType, str]
    default_instruction: str
    facts_header: str = _FACTS_HEADER
    empty_facts: str = _NO_FACTS

    def instruction_for(self, task: TaskType) -> str:
        """The task-specific paragraph, falling back to the generic one."""
        return self.task_instructions.get(task, self.default_instruction)


GROUNDED_V1: Final[PromptTemplate] = PromptTemplate(
    version="grounded_v1",
    persona=(
        "You are SatQuery AI, a remote-sensing analyst reporting to an operations "
        "team. You did not measure anything yourself. Deterministic tools have "
        "already measured this scene, and every measurement they produced is listed "
        "in the FACT SHEET below. Your job is to answer using those measurements and "
        "what the attached views show — nothing else."
    ),
    rules=_SHARED_RULES,
    task_instructions={
        TaskType.VQA: (
            "TASK. Answer the analyst's question directly in one to four sentences. "
            "Lead with the answer, then the evidence for it."
        ),
        TaskType.CAPTION: (
            "TASK. Describe this scene in two to four sentences: the dominant land "
            "cover, how it is arranged, and anything unusual. Quote a measurement "
            "only where it sharpens the description."
        ),
        TaskType.SCENE_CLASSIFY: (
            "TASK. Name the land-cover classes present in this scene, most extensive "
            "first, and give the measured evidence for each you can support."
        ),
        TaskType.COUNT: (
            "TASK. State the count that was measured, and nothing more. If no count "
            "was measured, say so; do not count objects yourself from the views."
        ),
        TaskType.CHANGE_VQA: (
            "TASK. Answer what changed between the two acquisitions, where, and by "
            "how much. Compare the pre- and post- views by their labels."
        ),
        TaskType.CHANGE_CAPTION: (
            "TASK. Describe the difference between the two acquisitions in two to "
            "four sentences: what changed, where in the scene, and how much."
        ),
        TaskType.CROSS_MODAL_VQA: (
            "TASK. Answer using both sensors. Say what the optical views support, "
            "what the radar views support, and whether the two agree."
        ),
        TaskType.CROSS_MODAL_COMPARE: (
            "TASK. Compare what the optical and radar acquisitions each reveal about "
            "this scene, and state explicitly where they agree and where they do not."
        ),
        TaskType.SEGMENTATION: (
            "TASK. Describe how the scene divides into land-cover regions and how "
            "much of the scene each covers, using the measured fractions."
        ),
        # Grounding is the one task whose answer is not prose. The format named
        # here is emitted by satquery.models.prompts.box_format.serialise and
        # parsed back by tools/text_grounding.py, and the Phase 7 corpus is built
        # with the same function — DATA_ADAPTATION_PLAN §4.5 requires all three
        # to be byte-identical, so this wording is frozen alongside them.
        TaskType.GROUNDING: (
            "TASK. Locate every instance of what the analyst asked for, and reply "
            "with nothing but boxes in this exact format, one per instance:\n"
            "<|object_ref_start|>NAME<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)"
            "<|box_end|>\n"
            "Coordinates are integers from 0 to 1000, measured as thousandths of "
            "the view's width and height from its top-left corner, with (x1,y1) "
            "the top-left corner of the box and (x2,y2) the bottom-right. If the "
            "scene contains none, reply with the single word NONE."
        ),
    },
    default_instruction=(
        "TASK. Answer the analyst's request in one to four sentences, grounded in "
        "the measurements above."
    ),
)
"""The prompt the Phase 4 zero-shot baseline and the Phase 7 adapter both use.
Frozen: an adapter trained against this text is served against this text."""


TEMPLATES: Final[dict[str, PromptTemplate]] = {
    GROUNDED_V1.version: GROUNDED_V1,
}

DEFAULT_VERSION: Final[str] = GROUNDED_V1.version


def get_template(version: str | None = None) -> PromptTemplate:
    """Return one frozen template by version.

    Raises:
        KeyError: No template by that version exists. Deliberately fatal rather
            than silently falling back — an answer produced by a different prompt
            than the trace claims is worse than no answer.
    """
    key = version or DEFAULT_VERSION
    try:
        return TEMPLATES[key]
    except KeyError:
        raise KeyError(
            f"unknown prompt version {key!r}; available: {sorted(TEMPLATES)}"
        ) from None


__all__ = [
    "CITATION_CLOSE",
    "CITATION_OPEN",
    "DEFAULT_VERSION",
    "GROUNDED_V1",
    "TEMPLATES",
    "PromptTemplate",
    "get_template",
]
