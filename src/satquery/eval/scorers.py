"""Pure scoring functions, one family per task type.

Everything here is ``(reference, prediction[, fact sheet]) -> numbers`` with no
I/O and no model, so the whole module is unit-tested against hand-written
answers (``tests/unit/test_eval_scorers.py``) and a scoring bug is a failing
test, not a wrong slide.

Three decisions worth knowing before reading the numbers:

* **Answers are compared with their citation markers removed.** ``0.82
  [spectral_index_analyzer.ndvi_mean]`` and ``0.82`` are the same claim; the
  marker is scored separately, as a citation.
* **Grounding is scored with the production parser.** A box the model emits in
  a format :func:`satquery.models.prompts.box_format.parse` cannot read is a
  miss, because it would be a miss at serving time too; ``format_ok`` reports
  how often that happens so a low recall can be told apart from a formatting
  regression.
* **Citation precision is the validator's verdict, not a regex count.** A
  marker counts as correct only when
  :func:`satquery.evidence.citation_validator.validate` binds the number in
  front of it to that key with the value the sheet holds — the same test every
  served answer passes through.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Final

from satquery.evidence.citation_validator import (
    CitationPolicy,
    extract_spans,
    validate,
)
from satquery.evidence.fact_sheet import FactSheet
from satquery.models.prompts.box_format import (
    NONE_ANSWER,
    NormalisedBox,
    is_canonical_answer,
    parse,
)
from satquery.models.prompts.builder import strip_citation_markers
from satquery.schemas.enums import TaskType
from satquery.training.corpus_builder import fact_sheet_from

__all__ = [
    "SampleScore",
    "bleu4",
    "citation_metrics",
    "count_of",
    "exact_match",
    "fact_recall",
    "iou",
    "labels_of",
    "match_boxes",
    "normalise",
    "rouge_l",
    "score",
    "set_f1",
]

FREE_TEXT_TASKS: Final[frozenset[TaskType]] = frozenset(
    {TaskType.CAPTION, TaskType.CHANGE_CAPTION, TaskType.CROSS_MODAL_COMPARE}
)
"""Tasks whose reference is a paragraph: overlap metrics, never exact match."""

IOU_THRESHOLD: Final[float] = 0.5
"""A predicted box counts as a hit at this IoU, the COCO/DIOR convention."""

_WS: Final[re.Pattern[str]] = re.compile(r"\s+")
_TRAILING_PUNCT: Final[re.Pattern[str]] = re.compile(r"[.!]+$")
_INTEGER: Final[re.Pattern[str]] = re.compile(r"-?\d[\d,]*")
_TOKEN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")
_SCENE_PREFIX: Final[re.Pattern[str]] = re.compile(r"^the scene contains\s*", re.IGNORECASE)


# ------------------------------------------------------------------ text


def strip_markers(text: str) -> str:
    """The answer as a reader sees it: ``[tool.scalar]`` markers removed."""
    return strip_citation_markers(text).text


def normalise(text: str) -> str:
    """Lower-case, marker-free, single-spaced, without terminal punctuation."""
    cleaned = _WS.sub(" ", strip_markers(text)).strip().lower()
    return _TRAILING_PUNCT.sub("", cleaned).strip()


def exact_match(reference: str, prediction: str) -> bool:
    """True when the two answers are the same after :func:`normalise`."""
    return normalise(reference) == normalise(prediction)


def labels_of(text: str) -> frozenset[str]:
    """The class labels a ``SCENE_CLASSIFY`` answer names.

    ``"The scene contains Coniferous forest, Pastures and Inland waters."`` →
    ``{"coniferous forest", "pastures", "inland waters"}``. Commas inside a
    CLC label ("Land principally occupied by agriculture, with significant
    areas of natural vegetation") split it into two labels on both sides of
    the comparison, so the F1 is unaffected in practice.
    """
    body = _SCENE_PREFIX.sub("", normalise(text))
    parts = re.split(r",\s*|\s+and\s+", body)
    return frozenset(part.strip() for part in parts if part.strip())


def set_f1(reference: frozenset[str], prediction: frozenset[str]) -> float:
    """F1 between two label sets; 1.0 when both are empty."""
    if not reference and not prediction:
        return 1.0
    hits = len(reference & prediction)
    if hits == 0:
        return 0.0
    precision = hits / len(prediction)
    recall = hits / len(reference)
    return 2 * precision * recall / (precision + recall)


def count_of(text: str) -> int | None:
    """The first integer in a ``COUNT`` answer, or None when it states none."""
    match = _INTEGER.search(strip_markers(text))
    return int(match.group(0).replace(",", "")) if match else None


# ------------------------------------------------------------- grounding


def iou(a: NormalisedBox, b: NormalisedBox) -> float:
    """Intersection over union of two normalised boxes."""
    width = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
    height = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    return intersection / (a.area + b.area - intersection)


def match_boxes(
    reference: Sequence[NormalisedBox],
    prediction: Sequence[NormalisedBox],
    threshold: float = IOU_THRESHOLD,
) -> tuple[float, float, list[tuple[int, int, float]]]:
    """Greedy one-to-one matching by descending IoU.

    Returns ``(recall_at_threshold, mean_iou, pairs)`` where ``mean_iou`` is
    averaged over the *reference* boxes — an unmatched reference contributes
    0 — so a model that emits one perfect box for a two-box reference scores
    0.5, not 1.0. Greedy rather than Hungarian because the references carry
    at most a handful of boxes and the two disagree only on ties.

    An empty reference (a ``NONE`` answer) with an empty prediction is a
    perfect score; with any prediction it is a miss.
    """
    if not reference:
        return (1.0, 1.0, []) if not prediction else (0.0, 0.0, [])
    candidates = sorted(
        (
            (iou(ref, pred), r, p)
            for r, ref in enumerate(reference)
            for p, pred in enumerate(prediction)
        ),
        key=lambda item: -item[0],
    )
    used_r: set[int] = set()
    used_p: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for value, r, p in candidates:
        if value <= 0.0:
            break
        if r in used_r or p in used_p:
            continue
        used_r.add(r)
        used_p.add(p)
        pairs.append((r, p, value))
    hits = sum(1 for _, _, value in pairs if value >= threshold)
    total_iou = sum(value for _, _, value in pairs)
    return hits / len(reference), total_iou / len(reference), pairs


def boxes_of(text: str) -> list[NormalisedBox]:
    """Every box the answer carries in a format the serving parser accepts."""
    return parse(text)


def grounding_format_ok(text: str) -> bool:
    """True when the answer is exactly what the prompt allows: boxes or NONE."""
    stripped = text.strip()
    return stripped == NONE_ANSWER or is_canonical_answer(stripped)


# ------------------------------------------------------------- citations


@dataclass(frozen=True)
class CitationMetrics:
    """What the validator made of one answer against its sheet."""

    markers: int
    """``[tool.scalar]`` markers the answer carried."""
    bound: int
    """Markers whose number the validator bound to that key and value."""
    spans: int
    """Numeric spans in the marker-free text."""
    uncited: int
    """Spans that resolved to no measurement."""
    cited_keys: frozenset[str]
    """Keys that were cited *correctly* (bound)."""

    @property
    def precision(self) -> float | None:
        """Bound markers over all markers; None when nothing was cited."""
        return self.bound / self.markers if self.markers else None

    @property
    def uncited_rate(self) -> float | None:
        """Uncited spans over all numeric spans; None when there are no numbers."""
        return self.uncited / self.spans if self.spans else None


def citation_metrics(text: str, sheet: FactSheet) -> CitationMetrics:
    """Run the production validator over *text* and count what it decided."""
    marked = strip_citation_markers(text, sheet)
    result = validate(marked.text, sheet, CitationPolicy.FLAG, markers=marked.markers)
    spans = extract_spans(marked.text)
    # A marker is bound when the span in front of it produced a citation whose
    # source is that key's fact. Unknown keys and key/value mismatches are the
    # two ways a marker fails; both are listed in uncited_reasons.
    failed_markers = sum(
        1 for reason in result.uncited_reasons if reason in {"UNKNOWN_KEY", "KEY_VALUE_MISMATCH"}
    )
    bound = max(0, len(marked.markers) - failed_markers)
    by_source = {fact.source: fact.key for fact in sheet.numeric_facts()}
    cited_keys = frozenset(
        by_source[citation.source] for citation in result.citations if citation.source in by_source
    )
    return CitationMetrics(
        markers=len(marked.markers),
        bound=bound,
        spans=len(spans),
        uncited=len(result.uncited_numeric_spans),
        cited_keys=cited_keys,
    )


def fact_recall(reference: CitationMetrics, prediction: CitationMetrics) -> float | None:
    """Of the facts the reference cites, the share the prediction also cites correctly.

    None when the reference cites nothing — the metric is about fact-checking
    answers, and an uncited reference has no facts to check against.
    """
    if not reference.cited_keys:
        return None
    return len(reference.cited_keys & prediction.cited_keys) / len(reference.cited_keys)


# -------------------------------------------------------------- captions


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalise(text))


def bleu4(reference: str, prediction: str) -> float:
    """Sentence BLEU-4 with add-one smoothing on the higher orders.

    Self-contained so the benchmark does not pull ``nltk`` for one number;
    the smoothing is Chen & Cherry's method 2, which is what ``sacrebleu``
    calls ``add-k`` with k=1.
    """
    ref, hyp = _tokens(reference), _tokens(prediction)
    if not hyp or not ref:
        return 0.0
    log_precision = 0.0
    for order in range(1, 5):
        ref_grams = Counter(tuple(ref[i : i + order]) for i in range(len(ref) - order + 1))
        hyp_grams = Counter(tuple(hyp[i : i + order]) for i in range(len(hyp) - order + 1))
        overlap = sum(min(count, ref_grams[gram]) for gram, count in hyp_grams.items())
        total = max(sum(hyp_grams.values()), 0)
        if order == 1:
            if overlap == 0:
                return 0.0
            precision = overlap / total
        else:
            precision = (overlap + 1) / (total + 1)
        log_precision += math.log(precision) / 4
    brevity = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
    return brevity * math.exp(log_precision)


def rouge_l(reference: str, prediction: str) -> float | None:
    """ROUGE-L F-measure via ``rouge-score``; None when the package is absent.

    ``rouge-score`` is a dev-group dependency (``pyproject.toml``), not a
    runtime one: the serving image does not need it, and a benchmark run
    without it still produces every other column.
    """
    scorer = _rouge()
    if scorer is None:
        return None
    return float(scorer.score(normalise(reference), normalise(prediction))["rougeL"].fmeasure)


_ROUGE: list[Any] = []
"""One-slot cache: the scorer logs "Using default tokenizer." on every
construction, and 500 of those would drown the progress line."""


def _rouge() -> Any | None:
    if not _ROUGE:
        try:
            from rouge_score import rouge_scorer
        except ImportError:  # pragma: no cover - environment-dependent
            _ROUGE.append(None)
        else:
            _ROUGE.append(rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True))
    return _ROUGE[0]


# ------------------------------------------------------------- per-sample


@dataclass
class SampleScore:
    """Every metric one sample can contribute; None where it does not apply."""

    exact_match: float | None = None
    set_f1: float | None = None
    count_abs_error: float | None = None
    grounding_recall: float | None = None
    grounding_iou: float | None = None
    grounding_format_ok: float | None = None
    citation_precision: float | None = None
    uncited_rate: float | None = None
    fact_recall: float | None = None
    bleu4: float | None = None
    rouge_l: float | None = None
    markers: int = 0
    spans: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Plain mapping for JSON and CSV."""
        return asdict(self)


def score(
    task: TaskType,
    reference: str,
    prediction: str,
    fact_sheet: Mapping[str, float | str],
) -> SampleScore:
    """Score one prediction against its reference under the task's rules.

    Args:
        task: The corpus task type; selects the metric family.
        reference: The supervised answer from the corpus.
        prediction: What the model generated (markers included).
        fact_sheet: The sample's ``{tool.scalar: value}`` measurements.
    """
    result = SampleScore()
    sheet = fact_sheet_from(fact_sheet)

    if task is TaskType.GROUNDING:
        ref_boxes = boxes_of(reference)
        pred_boxes = boxes_of(prediction)
        recall, mean_iou, _ = match_boxes(ref_boxes, pred_boxes)
        result.grounding_recall = recall
        result.grounding_iou = mean_iou
        result.grounding_format_ok = 1.0 if grounding_format_ok(prediction) else 0.0
        if not result.grounding_format_ok:
            result.notes.append("grounding answer is not in the canonical box format")
        return result

    metrics = citation_metrics(prediction, sheet)
    result.markers = metrics.markers
    result.spans = metrics.spans
    result.citation_precision = metrics.precision
    result.uncited_rate = metrics.uncited_rate

    if task in FREE_TEXT_TASKS:
        # A paragraph is never an exact match; it is scored on overlap and, for
        # a cross-modal summary, on whether it cites the facts the reference did.
        result.bleu4 = bleu4(reference, prediction)
        result.rouge_l = rouge_l(reference, prediction)
        if task is not TaskType.CAPTION:
            result.fact_recall = fact_recall(citation_metrics(reference, sheet), metrics)
        return result

    result.exact_match = 1.0 if exact_match(reference, prediction) else 0.0

    if task is TaskType.SCENE_CLASSIFY:
        result.set_f1 = set_f1(labels_of(reference), labels_of(prediction))
    elif task is TaskType.COUNT:
        expected, got = count_of(reference), count_of(prediction)
        if expected is not None:
            result.count_abs_error = (
                float(abs(expected - got)) if got is not None else float(expected)
            )
            result.exact_match = 1.0 if got == expected else 0.0
    if task in {TaskType.CROSS_MODAL_VQA, TaskType.VQA}:
        reference_metrics = citation_metrics(reference, sheet)
        result.fact_recall = fact_recall(reference_metrics, metrics)
    return result
