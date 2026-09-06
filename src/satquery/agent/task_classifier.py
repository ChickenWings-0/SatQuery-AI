"""Task classification: a four-stage cascade (AGENT_POLICY_DAG.md §2).

The cascade is ordered cheapest-and-most-certain first, and every stage can
terminate it. What it is *not* is an LLM deciding what the system should do:
Stage D fills slots, and may only propose a task when the deterministic stages
have already admitted they do not know.

    A. PairType gate     structural — a SINGLE image cannot have changed
    B. Rule layer        priority-ordered regex, first match wins
    C. Embedding kNN     only when B is unsure
    D. Slot fill         always runs; proposes a task only when B+C failed

Stage A is the highest-value stage in the file and it costs nothing: it removes
whole families of misclassification by construction rather than by confidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from satquery.agent.query_parser import fill_slots, normalise
from satquery.ingest.capabilities import REQUIREMENTS
from satquery.schemas.enums import Modality, PairType, TaskType

SEED_SET_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "configs" / "query_seed_set.jsonl"
)

RULES_VERSION: Final[str] = "rules_v1"
EMBED_VERSION: Final[str] = "embed_v1"
SLOTFILL_VERSION: Final[str] = "llm_slotfill_v1"

STAGE_C_TRIGGER: Final[float] = 0.75
"""Stage B confidence below this is not trusted on its own."""

STAGE_C_ACCEPT: Final[float] = 0.70
"""Stage C similarity at or above this terminates the cascade."""

STAGE_D_PROPOSAL_TRIGGER: Final[float] = 0.55
"""Only below this may Stage D propose a task rather than merely fill slots."""

MIN_CONFIDENCE: Final[float] = 0.40
"""Below this the query is UNSUPPORTED and the generic plan runs, flagged."""

DEFAULT_CONFIDENCE: Final[float] = 0.70
"""What the priority-99 default rule is worth: a real answer, not a certain one."""

CHANGE_WORD: Final[str] = (
    r"\b(chang(?:e|ed|es|ing)|differ\w*|appear\w*|disappear\w*|new|removed|lost|gained|"
    r"grew|expand\w*|shrink|shrank|shrunk|before and after|between the two)\b"
)

SECONDARY_TASKS: Final[dict[TaskType, list[TaskType]]] = {
    TaskType.CHANGE_VQA: [TaskType.CHANGE_MAP],
    TaskType.CHANGE_CAPTION: [TaskType.CHANGE_MAP],
    TaskType.COUNT: [TaskType.GROUNDING],
    TaskType.CROSS_MODAL_COMPARE: [TaskType.CROSS_MODAL_VQA],
}
"""Derived, never classified: these are the tasks whose artifacts the primary
task's DAG produces as a by-product. Declaring them lets the UI offer the change
map it already computed instead of asking the user to re-run."""

DEFAULT_TASK: Final[dict[PairType, TaskType]] = {
    PairType.SINGLE: TaskType.VQA,
    PairType.BI_TEMPORAL: TaskType.CHANGE_VQA,
    PairType.CROSS_MODAL: TaskType.CROSS_MODAL_VQA,
}


@dataclass(frozen=True)
class Rule:
    """One priority-ordered classification rule."""

    priority: int
    task: TaskType
    confidence: float
    pattern: re.Pattern[str]
    also: re.Pattern[str] | None = None
    pair_types: frozenset[PairType] | None = None

    def matches(self, text: str, pair_type: PairType) -> bool:
        """True when the query, and the pair it is asked about, fit this rule."""
        if self.pair_types is not None and pair_type not in self.pair_types:
            return False
        if not self.pattern.search(text):
            return False
        return self.also is None or bool(self.also.search(text))


def _rule(
    priority: int,
    task: TaskType,
    confidence: float,
    pattern: str,
    also: str | None = None,
    pair_types: frozenset[PairType] | None = None,
) -> Rule:
    return Rule(
        priority=priority,
        task=task,
        confidence=confidence,
        pattern=re.compile(pattern, re.IGNORECASE),
        also=re.compile(also, re.IGNORECASE) if also else None,
        pair_types=pair_types,
    )


RULES: Final[tuple[Rule, ...]] = (
    # CHANGE_MAP and CHANGE_CAPTION precede CHANGE_VQA because their patterns are
    # strict subsets of the change space: checked the other way round, the
    # general rule would swallow both.
    _rule(10, TaskType.CHANGE_MAP, 0.90,
          r"\b(map|mask|highlight|where exactly|show the area|footprint)\b", also=CHANGE_WORD),
    _rule(11, TaskType.CHANGE_CAPTION, 0.90,
          r"\b(describe|summari[sz]e|caption|what happened|narrate|explain|tell me the story)\b",
          also=CHANGE_WORD),
    _rule(12, TaskType.CHANGE_VQA, 0.92, CHANGE_WORD),
    # COUNT precedes GROUNDING: "how many buildings are there" contains a
    # grounding-shaped noun phrase and would otherwise be routed to it.
    _rule(20, TaskType.COUNT, 0.93, r"\b(how many|count|number of|total of)\b"),
    _rule(21, TaskType.SEGMENTATION, 0.88,
          r"\b(segment\w*|delineate|outline|extent of|boundary of)\b"),
    _rule(22, TaskType.GROUNDING, 0.88,
          r"\b(where|locate|find|point to|point out|bounding box|highlight)\b"),
    _rule(30, TaskType.CROSS_MODAL_COMPARE, 0.91,
          r"\b(sar|radar|backscatter|microwave)\b",
          also=r"\b(optical|spectral|visible|multispectral)\b"),
    _rule(31, TaskType.CROSS_MODAL_VQA, 0.86,
          r"\b(complement\w*|combin\w*|together|both sensors|fuse\w*)\b",
          pair_types=frozenset({PairType.CROSS_MODAL})),
    _rule(40, TaskType.CAPTION, 0.90,
          r"\b(describe|caption|summari[sz]e|overview|what does this (?:image )?show)\b"),
    _rule(50, TaskType.SCENE_CLASSIFY, 0.85,
          r"\b(classif\w*|land ?cover|land ?use|what (?:type|kind) of "
          r"(?:land|terrain|area|region))\b"),
)
"""Priority ordering is load-bearing; see the two comments above."""

_TOKEN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")

_FUNCTION_WORDS: Final[frozenset[str]] = frozenset({
    "a", "an", "and", "any", "are", "area", "is", "it", "its", "of", "on", "or", "the",
    "this", "that", "these", "those", "there", "they", "to", "what", "where", "when",
    "which", "who", "why", "how", "much", "many", "does", "do", "did", "can", "could",
    "would", "should", "show", "me", "tell", "give", "in", "at", "by", "for", "from",
    "with", "between", "over", "under", "near", "along", "about", "into", "image", "images",
    "picture", "scene", "tile", "raster", "look", "looks", "see", "visible", "present",
})
"""Enough English to tell a question from a keyboard mash. A query containing
none of these and no domain term is not a query the classifier should guess at:
guessing there produces a confident answer to a question nobody asked."""


@dataclass(frozen=True)
class Seed:
    """One labelled example from ``configs/query_seed_set.jsonl``."""

    query: str
    task: TaskType
    tokens: frozenset[str]


@lru_cache(maxsize=1)
def load_seed_set(path: Path | None = None) -> tuple[Seed, ...]:
    """Load the labelled seed queries backing Stage C."""
    source = path or SEED_SET_PATH
    seeds: list[Seed] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        record: dict[str, Any] = json.loads(stripped)
        query = normalise(str(record["query"]))
        seeds.append(
            Seed(
                query=query,
                task=TaskType(record["task"]),
                tokens=frozenset(_TOKEN.findall(query)),
            )
        )
    return tuple(seeds)


@lru_cache(maxsize=1)
def _domain_terms() -> frozenset[str]:
    """Every token the seed set and the class vocabulary know about."""
    from satquery.agent.query_parser import load_vocabulary

    tokens = {token for seed in load_seed_set() for token in seed.tokens}
    for phrase in load_vocabulary().index:
        tokens.update(_TOKEN.findall(phrase))
    return frozenset(tokens)


def candidates(pair_type: PairType) -> list[TaskType]:
    """Stage A — the tasks this input shape can possibly support.

    A ``SINGLE`` input can never produce ``CHANGE_*``; a ``BI_TEMPORAL`` input can
    never produce ``CROSS_MODAL_*``. Enforcing that structurally removes most
    misclassification risk before a single regex runs.
    """
    return [
        task
        for task, requirement in REQUIREMENTS.items()
        if pair_type in requirement.pair_types
    ]


def is_intelligible(text: str) -> bool:
    """True when the query is a question about imagery at all.

    The priority-99 default is a *fallback*, not a catch-all: applying it to a
    keyboard mash would produce a plausible-looking VQA answer to nothing, which
    is worse than admitting the query is unsupported.
    """
    tokens = set(_TOKEN.findall(text))
    return bool(tokens & _FUNCTION_WORDS) or bool(tokens & _domain_terms())


@dataclass
class StageOutcome:
    """What one stage concluded."""

    task: TaskType | None = None
    confidence: float = 0.0
    detail: str = ""


@dataclass
class Classification:
    """The resolved task, its slots, and how the cascade got there."""

    primary: TaskType
    secondary: list[TaskType]
    slots: dict[str, Any]
    confidence: float
    classifier: str
    stages: dict[str, StageOutcome] = field(default_factory=dict)
    warnings: list[tuple[str, str]] = field(default_factory=list)
    gated_out: list[TaskType] = field(default_factory=list)
    task_proposed_by_llm: bool = False


def stage_b(
    text: str, pair_type: PairType, allowed: set[TaskType]
) -> tuple[StageOutcome, list[TaskType]]:
    """Stage B — priority-ordered rules over the gated candidate set.

    Returns:
        The outcome plus any task a rule matched that Stage A had already ruled
        out. Reporting those is what lets "what changed" over a single image say
        *why* it became a VQA rather than silently becoming one.
    """
    gated: list[TaskType] = []
    for rule in RULES:
        if not rule.matches(text, pair_type):
            continue
        if rule.task not in allowed:
            gated.append(rule.task)
            continue
        return (
            StageOutcome(rule.task, rule.confidence, f"rule {rule.priority}"),
            gated,
        )

    default = DEFAULT_TASK.get(pair_type)
    if default in allowed and is_intelligible(text):
        return StageOutcome(default, DEFAULT_CONFIDENCE, "rule 99 (default)"), gated
    return StageOutcome(None, 0.0, "no rule matched"), gated


def stage_c(text: str, allowed: set[TaskType], k: int = 5) -> StageOutcome:
    """Stage C — kNN over the labelled seed set.

    Stands in for the sentence-embedding backend with a token-Jaccard similarity:
    same k, same "mean similarity of the winning class" confidence, same
    termination rule, no model to download. Swapping in real embeddings changes
    :func:`_similarity` and nothing else.
    """
    tokens = frozenset(_TOKEN.findall(text))
    if not tokens:
        return StageOutcome(None, 0.0, "empty query")

    scored = [
        (_similarity(tokens, seed.tokens), seed)
        for seed in load_seed_set()
        if seed.task in allowed
    ]
    if not scored:
        return StageOutcome(None, 0.0, "no seed in the candidate set")

    # Sorted by (-similarity, task, query) so ties break identically every run.
    scored.sort(key=lambda pair: (-pair[0], pair[1].task.value, pair[1].query))
    neighbours = scored[:k]

    by_task: dict[TaskType, list[float]] = {}
    for score, seed in neighbours:
        by_task.setdefault(seed.task, []).append(score)
    winner, scores = max(
        by_task.items(), key=lambda item: (len(item[1]), sum(item[1]), item[0].value)
    )
    confidence = sum(scores) / len(scores)
    return StageOutcome(winner, round(confidence, 4), f"knn k={k} over {len(scored)} seeds")


def _similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard overlap — the placeholder for cosine over sentence embeddings."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def classify(
    query: str,
    pair_type: PairType,
    modalities: list[Modality] | None = None,
    allow_generic_fallback: bool = True,
) -> Classification:
    """Run the full cascade and resolve the task the plan will be built for.

    Args:
        query: The raw user question.
        pair_type: The compatibility report's verdict on the inputs.
        modalities: Present for signature stability; the gate is pair-shaped.
        allow_generic_fallback: When False, an unclassifiable query is an error
            for the caller to turn into ``422 QUERY_UNCLASSIFIABLE`` rather than
            a generic plan.

    Returns:
        A :class:`Classification`. ``primary`` is ``UNSUPPORTED`` when the
        cascade never reached :data:`MIN_CONFIDENCE`.
    """
    text = normalise(query)
    allowed = set(candidates(pair_type))
    warnings: list[tuple[str, str]] = []
    stages: dict[str, StageOutcome] = {}

    rules, gated = stage_b(text, pair_type, allowed)
    stages[RULES_VERSION] = rules
    used = [RULES_VERSION]
    best = rules

    if rules.confidence < STAGE_C_TRIGGER:
        knn = stage_c(text, allowed)
        stages[EMBED_VERSION] = knn
        used.append(EMBED_VERSION)
        accepted = knn.confidence >= STAGE_C_ACCEPT
        if knn.task is not None and (accepted or knn.confidence > best.confidence):
            best = knn

    # Stage D always runs: it is the only source of slot values.
    used.append(SLOTFILL_VERSION)
    proposed = best.confidence < STAGE_D_PROPOSAL_TRIGGER
    classifier = "+".join(used)

    if gated:
        warnings.append(
            (
                "TASK_GATED_BY_PAIR_TYPE",
                f"The query reads as {gated[0].value}, which {pair_type.value} inputs cannot "
                f"support; it was routed to the nearest supported task instead.",
            )
        )

    if best.task is None or best.confidence < MIN_CONFIDENCE:
        if not allow_generic_fallback:
            raise QueryUnclassifiableError(query=query, confidence=best.confidence)
        warnings.append(
            (
                "LOW_CLASSIFICATION_CONFIDENCE",
                f"The query could not be classified (confidence {best.confidence:.2f} < "
                f"{MIN_CONFIDENCE:.2f}); a generic plan was used.",
            )
        )
        slots, unmapped = fill_slots(TaskType.VQA, text)
        return Classification(
            primary=TaskType.UNSUPPORTED,
            secondary=[],
            slots=slots,
            confidence=round(best.confidence, 4),
            classifier=classifier,
            stages=stages,
            warnings=warnings,
            gated_out=gated,
        )

    task = best.task
    slots, unmapped = fill_slots(task, text)
    if unmapped:
        warnings.append(
            (
                "UNMAPPED_CLASS_TERM",
                f"{', '.join(unmapped)} is not in the controlled vocabulary; it was passed to "
                f"the language model but never used for a threshold.",
            )
        )

    if proposed:
        classifier = f"{classifier}(task_proposed)"

    return Classification(
        primary=task,
        secondary=list(SECONDARY_TASKS.get(task, [])),
        slots=slots,
        confidence=round(best.confidence, 4),
        classifier=classifier,
        stages=stages,
        warnings=warnings,
        gated_out=gated,
        task_proposed_by_llm=proposed,
    )


class QueryUnclassifiableError(ValueError):
    """The cascade could not classify the query and generic fallback is off."""

    def __init__(self, query: str, confidence: float) -> None:
        """Record the query and the confidence that fell short."""
        super().__init__(f"could not classify {query!r} (confidence {confidence:.2f})")
        self.query = query
        self.confidence = confidence
