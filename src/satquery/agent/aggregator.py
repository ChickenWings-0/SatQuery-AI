"""Answer synthesis, grounded in the FactSheet.

From Phase 4 the answer is normally the VLM's: :func:`vlm_answer` picks the
``TEXT`` artifact a ``vlm_*`` step produced and :func:`aggregate` validates it
against the same sheet the model was constrained by. Whenever no such step ran —
no servable weights, a skipped or failed synthesiser, or a task the table
deliberately terminates deterministically — the aggregator writes the answer
itself from measurements only and marks it ``template_fallback: true``.

Every number a template emits is read straight out of the FactSheet and printed
at a precision the CitationValidator can resolve back to it, so a templated
answer is fully cited by construction. That is not a workaround: for
``CHANGE_MAP`` it is the *designed* terminal behaviour (policy entry 15), because
a change map is completely answerable from deterministic outputs and paying VLM
latency and hallucination risk for it would be strictly worse.

The templates are deliberately plain. They are evidence read aloud, not prose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from satquery.evidence.citation_validator import (
    CitationPolicy,
    ValidationResult,
    format_number,
    validate,
)
from satquery.evidence.fact_sheet import FactSheet
from satquery.models.prompts.box_format import strip_boxes
from satquery.schemas.enums import ArtifactType, PairType, TaskType, ToolStatus
from satquery.schemas.tool import Execution
from satquery.schemas.trace import Answer, ArtifactRef

TEMPLATE_GENERATOR: Final[str] = "template_answer_v1"

_CHANGE_TASKS: Final[frozenset[TaskType]] = frozenset(
    {TaskType.CHANGE_VQA, TaskType.CHANGE_CAPTION, TaskType.CHANGE_MAP}
)
_CROSS_MODAL_TASKS: Final[frozenset[TaskType]] = frozenset(
    {TaskType.CROSS_MODAL_VQA, TaskType.CROSS_MODAL_COMPARE}
)

INDEX_SHIFT_THRESHOLD: Final[float] = 0.02
"""Below this an index has not meaningfully moved, and the answer says so."""

NO_EVIDENCE: Final[str] = (
    "No tool in this plan produced a measurement, so there is nothing this "
    "system can state about the scene without guessing."
)


@dataclass
class Aggregation:
    """The synthesised answer and the citation check behind it."""

    answer: Answer
    validation: ValidationResult

    @property
    def has_uncited(self) -> bool:
        """True when some number in the answer resolved to no measurement."""
        return self.validation.has_uncited


class _Writer:
    """Accumulates sentences, skipping any whose measurement is absent."""

    def __init__(self, sheet: FactSheet) -> None:
        self.sheet = sheet
        self.sentences: list[str] = []

    def has(self, *keys: str) -> bool:
        """True when every named fact was measured."""
        return all(self.sheet.number(key) is not None for key in keys)

    def num(self, key: str) -> str:
        """The formatted value of one numeric fact."""
        value = self.sheet.number(key)
        return format_number(value) if value is not None else ""

    def say(self, sentence: str) -> None:
        """Append a sentence."""
        self.sentences.append(sentence)

    def index_shift(self, prefix: str, label: str, surface: str) -> None:
        """Report how a bi-temporal index moved, without overstating a flat one.

        A tenth of an index unit is the smallest shift worth attributing to a
        surface change; below that the honest sentence is "essentially
        unchanged", and claiming new impervious surface from a 0.001 move would
        be exactly the kind of confident nonsense the FactSheet exists to stop.
        """
        before_key, after_key = f"{prefix}_mean_pre", f"{prefix}_mean_post"
        if not self.has(before_key, after_key):
            return
        before = self.sheet.number(before_key)
        after = self.sheet.number(after_key)
        if before is None or after is None:
            return
        if abs(after - before) < INDEX_SHIFT_THRESHOLD:
            self.say(
                f"Mean {label} is essentially unchanged across the pair "
                f"({self.num(before_key)} then {self.num(after_key)})."
            )
            return
        direction = "rose" if after > before else "fell"
        self.say(
            f"Mean {label} {direction} from {self.num(before_key)} to {self.num(after_key)} "
            f"over the same period, consistent with a change in {surface} rather than a "
            f"purely seasonal effect."
        )

    def text(self) -> str:
        """The assembled answer."""
        return " ".join(self.sentences).strip()


def _change_answer(writer: _Writer, detector: str) -> None:
    """Describe measured change, preferring the areal statistics over raw pixels."""
    stats = "change_statistics"
    if writer.has(f"{stats}.changed_area_pct"):
        area = writer.num(f"{stats}.changed_area_pct")
        if writer.has(f"{stats}.changed_area_km2"):
            writer.say(
                f"Approximately {area}% of the analysed scene "
                f"({writer.num(f'{stats}.changed_area_km2')} km2) changed between the two "
                f"acquisitions."
            )
        else:
            writer.say(f"Approximately {area}% of the analysed scene changed between the two "
                       f"acquisitions.")
    elif writer.has(f"{detector}.changed_area_pct"):
        writer.say(
            f"Approximately {writer.num(f'{detector}.changed_area_pct')}% of the analysed "
            f"scene changed between the two acquisitions."
        )

    if writer.has(f"{stats}.component_count"):
        count = writer.num(f"{stats}.component_count")
        noun = "region" if writer.sheet.number(f"{stats}.component_count") == 1 else "regions"
        if writer.has(f"{stats}.largest_component_m2"):
            writer.say(
                f"The change is distributed across {count} distinct {noun}, the largest "
                f"covering {writer.num(f'{stats}.largest_component_m2')} m2."
            )
        else:
            writer.say(f"The change is distributed across {count} distinct {noun}.")

    index = "spectral_index_analyzer"
    writer.index_shift(f"{index}.ndbi", "NDBI", "impervious surface")
    writer.index_shift(f"{index}.ndvi", "NDVI", "vegetation cover")


def _single_optical_answer(writer: _Writer) -> None:
    """Describe a single optical scene from its index fractions."""
    index = "spectral_index_analyzer"
    parts: list[str] = []
    for fraction, label in (
        ("vegetation_fraction_pct", "vegetated"),
        ("water_fraction_pct", "open water"),
        ("built_up_fraction_pct", "built-up"),
    ):
        key = f"{index}.{fraction}"
        if writer.has(key):
            parts.append(f"{writer.num(key)}% {label}")
    if parts:
        writer.say(f"By fixed-threshold index classification the scene is {', '.join(parts)}.")
    if writer.has(f"{index}.ndvi_mean"):
        writer.say(f"Mean NDVI over the valid pixels is {writer.num(f'{index}.ndvi_mean')}.")


def _sar_answer(writer: _Writer) -> None:
    """Describe a SAR scene from its backscatter statistics."""
    sar = "sar_backscatter_analyzer"
    if writer.has(f"{sar}.sigma0_vv_db_mean"):
        writer.say(
            f"Mean VV backscatter is {writer.num(f'{sar}.sigma0_vv_db_mean')} dB across the "
            f"valid pixels."
        )
    if writer.has(f"{sar}.low_backscatter_fraction_pct"):
        writer.say(
            f"{writer.num(f'{sar}.low_backscatter_fraction_pct')}% of the scene falls below "
            f"the open-water backscatter threshold."
        )


def _cross_modal_answer(writer: _Writer) -> None:
    """Report how far the two sensors agree."""
    physics = "physics_agreement"
    if writer.has(f"{physics}.agreement_pct"):
        writer.say(
            f"The optical and SAR observations agree over {writer.num(f'{physics}.agreement_pct')}"
            f"% of the pixels either sensor makes a claim about."
        )
    for rule, label in (
        ("water_agree_pct", "open water"),
        ("vegetated_agree_pct", "vegetation"),
        ("built_up_agree_pct", "built-up surface"),
    ):
        key = f"{physics}.{rule}"
        if writer.has(key):
            writer.say(f"For {label} the two sensors concur on {writer.num(key)}% of pixels.")


def _count_answer(writer: _Writer) -> None:
    """State the count, and only the count."""
    if writer.has("object_counter.count"):
        writer.say(f"The detector delimited {writer.num('object_counter.count')} objects.")


def _generic_answer(writer: _Writer) -> None:
    """Fall back to raster statistics when nothing more specific was measured."""
    stats = "raster_statistics"
    if writer.has(f"{stats}.valid_pixel_pct"):
        writer.say(
            f"{writer.num(f'{stats}.valid_pixel_pct')}% of the scene carries valid data."
        )
    if writer.has(f"{stats}.brightness_mean"):
        writer.say(
            f"Mean band value across the scene is {writer.num(f'{stats}.brightness_mean')}."
        )


def compose(task: TaskType, pair_type: PairType, sheet: FactSheet) -> str:
    """Write the answer text for one task from the measurements alone."""
    writer = _Writer(sheet)

    if task in _CHANGE_TASKS or pair_type is PairType.BI_TEMPORAL:
        detector = (
            "image_diff_change"
            if sheet.number("image_diff_change.changed_area_pct") is not None
            else "siamese_change_detector"
        )
        _change_answer(writer, detector)
    elif task in _CROSS_MODAL_TASKS or pair_type is PairType.CROSS_MODAL:
        _cross_modal_answer(writer)
        _single_optical_answer(writer)
        _sar_answer(writer)
    elif task is TaskType.COUNT:
        _count_answer(writer)
    else:
        _single_optical_answer(writer)
        _sar_answer(writer)

    if not writer.sentences:
        _generic_answer(writer)
    if not writer.sentences:
        writer.say(NO_EVIDENCE)
    return writer.text()


@dataclass(frozen=True)
class VlmAnswer:
    """The prose a synthesiser produced, and which step produced it."""

    text: str
    generator: str
    step: int
    tool: str
    status: ToolStatus

    @property
    def degraded(self) -> bool:
        """True when the synthesiser itself flagged a problem with its output."""
        return self.status is not ToolStatus.OK


def vlm_answer(
    executions: Sequence[Execution], artifacts: Sequence[ArtifactRef]
) -> VlmAnswer | None:
    """Find the answer a ``vlm_*`` step wrote, if one did.

    The *last* synthesiser wins. A plan may run more than one — a caption
    followed by a grounded answer, say — and the table lists the terminal step
    last, so reading in step order and keeping the final one needs no extra
    declaration in the policy table.

    Returns:
        The generated answer, or None when no synthesiser produced usable text.
    """
    by_id = {artifact.id: artifact for artifact in artifacts}
    found: VlmAnswer | None = None
    for execution in sorted(executions, key=lambda e: e.step):
        if not execution.tool.startswith("vlm_"):
            continue
        if execution.status not in {ToolStatus.OK, ToolStatus.DEGRADED}:
            continue
        for reference in execution.output_refs:
            artifact = by_id.get(reference)
            if artifact is None or artifact.type is not ArtifactType.TEXT:
                continue
            inline = artifact.inline or {}
            text = str(inline.get("text") or "").strip()
            if not text:
                continue
            found = VlmAnswer(
                text=text,
                generator=str(inline.get("generator") or execution.tool),
                step=execution.step,
                tool=execution.tool,
                status=execution.status,
            )
    return found


def vlm_unavailable(executions: Sequence[Execution]) -> list[str]:
    """The ``vlm_*`` steps that did not run, so the answer can say why it is templated."""
    return [
        execution.tool
        for execution in executions
        if execution.tool.startswith("vlm_")
        and execution.status in {ToolStatus.SKIPPED, ToolStatus.FAILED}
    ]


def aggregate(
    task: TaskType,
    pair_type: PairType,
    sheet: FactSheet,
    executions: Sequence[Execution],
    text: str | None = None,
    generator: str | None = None,
    policy: CitationPolicy = CitationPolicy.FLAG,
) -> Aggregation:
    """Produce the final :class:`Answer`, cited against the FactSheet.

    Args:
        task: The resolved primary task.
        pair_type: The analysis pair type.
        sheet: The measurements the plan produced.
        executions: Used to explain why a templated answer was written.
        text: A VLM-generated answer to validate instead of writing one. When
            None — every Phase 3 path — the template writes it.
        generator: Overrides the recorded generator string.
        policy: Citation policy; ``flag`` by default.

    Returns:
        An :class:`Aggregation` carrying the answer and the citation outcome.
    """
    templated = text is None
    body = text if text is not None else compose(task, pair_type, sheet)

    if templated:
        skipped = vlm_unavailable(executions)
        if skipped:
            body = (
                f"{body} This answer was written directly from the measured evidence because "
                f"{', '.join(sorted(set(skipped)))} did not run."
            )

    # Boxes are removed before the claim check, never from the answer itself:
    # the coordinates are the grounding answer, and they are not citable claims.
    claims = strip_boxes(body) if task is TaskType.GROUNDING else body
    result = validate(claims, sheet, policy=policy)
    answer = Answer(
        text=body if task is TaskType.GROUNDING else result.text,
        citations=result.citations,
        uncited_numeric_spans=result.uncited_numeric_spans,
        generator=generator or (TEMPLATE_GENERATOR if templated else "vlm"),
        template_fallback=templated,
    )
    return Aggregation(answer=answer, validation=result)
