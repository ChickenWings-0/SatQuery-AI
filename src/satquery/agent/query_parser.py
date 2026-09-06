"""Query normalisation and slot extraction (AGENT_POLICY_DAG.md §3).

Slots are the only thing the language layer is allowed to decide. They never
choose a tool and never order a step — they parameterise steps the policy table
already fixed. That boundary is the whole reason the trace is reproducible.

Class normalisation is mandatory and happens here: "buildings", "built up",
"urban area" and "settlement" all become ``built_up`` before the planner sees
them, so a threshold decision never depends on which synonym the user typed. A
term the vocabulary cannot map is kept verbatim, marked unmapped, and passed to
the VLM only — never used to pick a threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml

from satquery.schemas.enums import TaskType

CONFIG_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "configs" / "class_vocabulary.yaml"

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")

OUTPUT_FORMATS: Final[dict[str, str]] = {
    "text": "text",
    "mask": "text+mask",
    "boxes": "text+boxes",
}

_MASK_TASKS: Final[frozenset[TaskType]] = frozenset(
    {TaskType.SEGMENTATION, TaskType.CHANGE_MAP}
)
_BOX_TASKS: Final[frozenset[TaskType]] = frozenset({TaskType.GROUNDING, TaskType.COUNT})

_SPATIAL: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b((?:north|south|east|west|north[- ]?east|north[- ]?west|south[- ]?east|"
        r"south[- ]?west|central|centre|center)(?:ern)?\s+(?:quadrant|corner|half|"
        r"part|section|edge|side|region|area))\b",
        r"\b(along the [a-z ]{3,20}?)(?:\s*[,.]|$)",
        r"\b(near the [a-z ]{3,20}?)(?:\s*[,.]|$)",
        r"\b(in the (?:top|bottom|left|right)(?:[- ](?:left|right))?)\b",
    )
)
"""Spatial constraints are extracted verbatim, not normalised: "along the river"
is meaningful to the VLM and meaningless to a threshold, so it is passed through
rather than forced into a vocabulary."""

_DIRECTION_INCREASE: Final[re.Pattern[str]] = re.compile(
    r"\b(increas\w*|grew|grow\w*|expand\w*|gained|appear\w*|new|added|more)\b"
)
_DIRECTION_DECREASE: Final[re.Pattern[str]] = re.compile(
    r"\b(decreas\w*|shrank|shrink\w*|shrunk|lost|losing|removed|disappear\w*|less|reduced)\b"
)
_METRIC_AREA: Final[re.Pattern[str]] = re.compile(
    r"\b(how much|how many|area|extent|hectare|km2|square|percentage|percent|proportion)\b"
)
_METRIC_PRESENCE: Final[re.Pattern[str]] = re.compile(
    r"\b(is there|are there|any|did anything|has anything|was anything)\b"
)

_FOCUS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"\b(complement\w*|combin\w*|together|both sensors|fuse\w*)\b"), "complementarity"),
    (re.compile(r"\b(agree\w*|consistent\w*|match\w*|disagree\w*)\b"), "agreement"),
    (re.compile(r"\b(sar|radar|microwave|backscatter) only\b"), "sar_only"),
    (re.compile(r"\boptical only\b"), "optical_only"),
)

_TAXONOMY: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"\bbigearthnet[- ]?19\b"), "bigearthnet19"),
    (re.compile(r"\bbigearthnet[- ]?6\b"), "bigearthnet6"),
    (re.compile(r"\bbigearthnet\b"), "bigearthnet19"),
)

MAX_BOXES: Final[int] = 20


@dataclass(frozen=True)
class ClassVocabulary:
    """The controlled land-cover / object vocabulary, as a synonym index."""

    version: str
    canonical: dict[str, list[str]]
    index: dict[str, str]

    def normalise(self, term: str | None) -> tuple[str | None, bool]:
        """Map *term* onto the vocabulary.

        Returns:
            ``(value, mapped)``. An unmappable term comes back verbatim with
            ``mapped=False``, so the caller can pass it to the VLM while keeping
            it out of every threshold decision.
        """
        if not term:
            return None, False
        key = _WHITESPACE.sub(" ", term.strip().lower())
        if key in self.canonical:
            return key, True
        mapped = self.index.get(key)
        if mapped is not None:
            return mapped, True
        return key, False

    def find(self, text: str) -> str | None:
        """Return the canonical class named in *text*, longest synonym first."""
        for phrase in self._ordered_phrases:
            if re.search(rf"\b{re.escape(phrase)}\b", text):
                return self.index[phrase]
        return None

    @property
    def _ordered_phrases(self) -> list[str]:
        """Synonyms longest-first, so "water body" beats "water"."""
        return sorted(self.index, key=lambda phrase: (-len(phrase), phrase))


@lru_cache(maxsize=1)
def load_vocabulary(path: Path | None = None) -> ClassVocabulary:
    """Load and cache ``configs/class_vocabulary.yaml``."""
    raw: dict[str, Any] = yaml.safe_load((path or CONFIG_PATH).read_text(encoding="utf-8"))
    canonical = {
        str(name): [str(s).lower() for s in (synonyms or [])]
        for name, synonyms in (raw.get("classes") or {}).items()
    }
    index: dict[str, str] = {}
    for name, synonyms in canonical.items():
        index[name.replace("_", " ")] = name
        index[name] = name
        for synonym in synonyms:
            index[synonym] = name
    return ClassVocabulary(
        version=str(raw.get("version", "unknown")), canonical=canonical, index=index
    )


@dataclass(frozen=True)
class ParsedQuery:
    """The user's question, normalised and stripped down to slot material."""

    raw: str
    normalized: str
    language: str = "en"
    unmapped_terms: list[str] = field(default_factory=list)


def normalise(query: str) -> str:
    """Lowercase and collapse whitespace. Nothing else — punctuation carries meaning."""
    return _WHITESPACE.sub(" ", query.strip().lower())


def parse(query: str) -> ParsedQuery:
    """Normalise a raw query for both the classifier and the trace."""
    return ParsedQuery(raw=query, normalized=normalise(query))


def _spatial_constraint(text: str) -> str | None:
    for pattern in _SPATIAL:
        found = pattern.search(text)
        if found:
            return found.group(1).strip()
    return None


def _output_format(task: TaskType, text: str) -> str:
    if task in _MASK_TASKS or re.search(r"\b(mask|segment\w*|outline|delineate)\b", text):
        return OUTPUT_FORMATS["mask"]
    if task in _BOX_TASKS or re.search(r"\b(bounding box|boxes|bbox)\b", text):
        return OUTPUT_FORMATS["boxes"]
    return OUTPUT_FORMATS["text"]


def _direction(text: str) -> str:
    if _DIRECTION_INCREASE.search(text):
        return "increase"
    if _DIRECTION_DECREASE.search(text):
        return "decrease"
    return "any"


def _metric(text: str) -> str | None:
    if _METRIC_AREA.search(text):
        return "area_delta"
    if _METRIC_PRESENCE.search(text):
        return "presence"
    if _DIRECTION_INCREASE.search(text) or _DIRECTION_DECREASE.search(text):
        return "direction"
    return None


def _referring_expression(text: str) -> str | None:
    """Take everything after the locating verb as the referring expression."""
    found = re.search(
        r"\b(?:where is|where are|locate|find|point to|point out|show me)\s+(.+?)"
        r"(?:\s*\?|\s*$)",
        text,
    )
    return found.group(1).strip() if found else None


def fill_slots(task: TaskType, text: str) -> tuple[dict[str, Any], list[str]]:
    """Derive the slot dictionary for *task* from the normalised query.

    This stands in for the Stage D constrained-decoding call (``llm_slotfill_v1``)
    until Phase 4 lands the VLM. It is deterministic by construction, which the
    real thing will approximate with ``temperature: 0.0`` and a fixed seed.

    Returns:
        ``(slots, unmapped_terms)``. Every optional slot is present and ``None``
        rather than absent, because the contract says so and because a missing
        key and a null key mean different things to a client.
    """
    vocabulary = load_vocabulary()
    found = vocabulary.find(text)
    target_class, mapped = vocabulary.normalise(found)
    unmapped = [] if (mapped or target_class is None) else [target_class]

    slots: dict[str, Any] = {
        "target_class": target_class,
        "spatial_constraint": _spatial_constraint(text),
        "output_format": _output_format(task, text),
    }

    if task is TaskType.COUNT:
        slots["object_class"] = target_class
    elif task is TaskType.GROUNDING:
        slots["referring_expression"] = _referring_expression(text)
        slots["max_boxes"] = MAX_BOXES
    elif task in {TaskType.CHANGE_VQA, TaskType.CHANGE_CAPTION, TaskType.CHANGE_MAP}:
        slots["metric"] = _metric(text)
        slots["direction"] = _direction(text)
    elif task in {TaskType.CROSS_MODAL_VQA, TaskType.CROSS_MODAL_COMPARE}:
        slots["focus"] = next(
            (value for pattern, value in _FOCUS if pattern.search(text)), "complementarity"
        )
    elif task is TaskType.SCENE_CLASSIFY:
        slots["taxonomy"] = next(
            (value for pattern, value in _TAXONOMY if pattern.search(text)), "free"
        )

    return slots, unmapped
