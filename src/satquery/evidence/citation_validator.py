"""The CitationValidator (AGENT_POLICY_DAG.md §7.1).

Every number in the answer must resolve to a measurement, or be listed as one
that did not. That list is the honesty signal the demo is built around: a
frontend that renders ``uncited_numeric_spans`` shows a judge exactly where the
model went beyond its evidence, which is worth considerably more than a tidy
paragraph that quietly contains an invented figure.

Two decisions are deliberate and worth defending:

* **Units must match, not just magnitude.** ``7.4 %`` may only resolve against a
  ``*_pct`` scalar. Letting ``7.4`` match ``changed_area_km2`` because the digits
  agree would manufacture a citation for a claim nobody measured.
* **The default policy is ``flag``, not ``strip``.** Deleting the sentence around
  an unsupported number produces incoherent text and hides the problem;
  ``strip`` exists for benchmark runs where an unsupported number scores as
  wrong, and nowhere else.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from satquery.evidence.fact_sheet import Fact, FactSheet
from satquery.schemas.trace import Citation

NUMERIC_SPAN: Final[re.Pattern[str]] = re.compile(
    r"(?<![\d.,])(?P<number>[-+]?\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?:\s*(?P<unit>%|km²|km2|m²|m2|dB|px|m)(?!\w))?",
    re.IGNORECASE,
)
"""``m2`` and ``km2`` precede the bare ``m`` in the alternation, or "1.07 km2"
would parse as 1.07 metres followed by a stray "2". The unit's ``(?!\\w)`` stops
"5 meters" from reading as "5 m" — the unit has to end its token, not begin one —
and the leading lookbehind stops "2019" from being read as the three digits "019",
which would sneak a year past the year exclusion."""

RELATIVE_TOLERANCE: Final[float] = 0.01
ABSOLUTE_TOLERANCE: Final[float] = 0.05
TOLERANCE_PIVOT: Final[float] = 10.0

UNIT_TOKENS: Final[dict[str, frozenset[str]]] = {
    "%": frozenset({"pct", "percent"}),
    "km2": frozenset({"km2"}),
    "km²": frozenset({"km2"}),
    "m2": frozenset({"m2"}),
    "m²": frozenset({"m2"}),
    "m": frozenset({"m", "metres", "meters"}),
    "db": frozenset({"db"}),
    "px": frozenset({"px", "pixel"}),
}
"""Which name segment a unit is allowed to resolve against.

Matching is on ``_``-delimited *segments* rather than on the suffix, because the
unit sits in the middle of most scalar names: ``sigma0_vv_db_mean`` is a mean of
decibels, and a suffix rule would refuse to let any dB claim cite it."""

_ALL_UNIT_TOKENS: Final[frozenset[str]] = frozenset(
    token for tokens in UNIT_TOKENS.values() for token in tokens
)

_YEAR: Final[re.Pattern[str]] = re.compile(r"^\d{4}$")
_IMAGE_INDEX: Final[re.Pattern[str]] = re.compile(r"\bimage\s*$", re.IGNORECASE)
_LIST_MARKER_BEFORE: Final[re.Pattern[str]] = re.compile(r"(?:^|\n)\s*$")
_LIST_MARKER_AFTER: Final[re.Pattern[str]] = re.compile(r"^[.)]\s")
_QUOTED_SOURCE: Final[re.Pattern[str]] = re.compile(r"(step:\d+/scalars\.[\w.]+)")
_SENTENCE: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+")


class CitationPolicy(StrEnum):
    """What to do with a number that resolves to nothing."""

    FLAG = "flag"
    """Keep the sentence and list the span. The default."""

    STRIP = "strip"
    """Remove the containing sentence. For benchmark runs only."""


@dataclass(frozen=True)
class NumericSpan:
    """One number found in the answer text."""

    text: str
    value: float
    unit: str | None
    start: int
    end: int


@dataclass
class ValidationResult:
    """The outcome of checking one answer against the FactSheet."""

    text: str
    citations: list[Citation]
    uncited_numeric_spans: list[str]

    @property
    def has_uncited(self) -> bool:
        """True when at least one number resolved to no measurement."""
        return bool(self.uncited_numeric_spans)


def format_number(value: float) -> str:
    """Print a measurement so :func:`resolve` binds it back to its own fact.

    Two decimals with thousands separators, trailing zeros trimmed: inside the
    relative-1 % tolerance above 10 and the absolute-0.05 tolerance below it for
    every scalar the registry declares. It lives here rather than beside its
    callers because the guarantee it makes is a property of the tolerance rule
    directly above, and two formatters would eventually disagree with it.
    """
    text = f"{value:,.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _is_excluded(text: str, span: NumericSpan) -> bool:
    """Numbers that are not claims: years, image indices, list markers, sources."""
    raw = span.text.strip()
    if span.unit is None and _YEAR.match(raw) and 1900 <= int(raw) <= 2100:
        return True
    if _IMAGE_INDEX.search(text[max(0, span.start - 8) : span.start]):
        return True
    # A list marker is a bare integer that *opens a line and is punctuated as a
    # marker* ("3) ..."). Excluding every line-initial integer would also excuse
    # "25 distinct regions changed", which is exactly the kind of claim that has
    # to be cited.
    if (
        raw.isdigit()
        and span.unit is None
        and _LIST_MARKER_BEFORE.search(text[max(0, span.start - 4) : span.start])
        and _LIST_MARKER_AFTER.match(text[span.end : span.end + 2])
    ):
        return True
    return any(
        found.start() <= span.start < found.end() for found in _QUOTED_SOURCE.finditer(text)
    )


def extract_spans(text: str) -> list[NumericSpan]:
    """Find every numeric span worth validating, in reading order."""
    spans: list[NumericSpan] = []
    for found in NUMERIC_SPAN.finditer(text):
        raw = found.group("number")
        unit = found.group("unit")
        try:
            value = float(raw.replace(",", ""))
        except ValueError:  # pragma: no cover - the pattern guarantees a number
            continue
        span = NumericSpan(
            text=found.group(0).strip(),
            value=value,
            unit=unit.lower() if unit else None,
            start=found.start(),
            end=found.end(),
        )
        if _is_excluded(text, span):
            continue
        spans.append(span)
    return spans


def unit_compatible(scalar: str, unit: str | None) -> bool:
    """True when a scalar name is allowed to back a claim carrying *unit*.

    A unitless claim may only resolve against a unitless scalar. That asymmetry
    is intentional: "the mean NDBI rose to 0.03" must not silently cite
    ``changed_area_pct`` because 0.03 happens to be within tolerance of it.
    """
    segments = set(scalar.lower().split("_"))
    if unit is None:
        return not (segments & _ALL_UNIT_TOKENS)
    allowed = UNIT_TOKENS.get(unit)
    if allowed is None:
        return False
    return bool(segments & allowed)


def within_tolerance(claimed: float, measured: float) -> bool:
    """Relative 1 % at or above 10, absolute 0.05 below it."""
    if abs(measured) >= TOLERANCE_PIVOT:
        return abs(claimed - measured) <= abs(measured) * RELATIVE_TOLERANCE
    return abs(claimed - measured) <= ABSOLUTE_TOLERANCE


def resolve(span: NumericSpan, facts: Iterable[Fact]) -> Fact | None:
    """Find the measurement a numeric span cites, if any.

    Ties are broken by the closest value and then by key, so the citation a
    given answer produces is identical on every run.
    """
    matches = [
        fact
        for fact in facts
        if fact.is_numeric
        and unit_compatible(fact.scalar, span.unit)
        and within_tolerance(span.value, float(fact.value))
    ]
    if not matches:
        return None
    return min(matches, key=lambda fact: (abs(float(fact.value) - span.value), fact.key))


def validate(
    text: str, sheet: FactSheet, policy: CitationPolicy = CitationPolicy.FLAG
) -> ValidationResult:
    """Bind every numeric claim in *text* to a measurement, or flag it.

    Args:
        text: The synthesised answer.
        sheet: The measurements the plan actually produced.
        policy: ``FLAG`` keeps the sentence and lists the span; ``STRIP`` removes
            the containing sentence.

    Returns:
        A :class:`ValidationResult` carrying the (possibly rewritten) text, the
        citations that resolved, and the spans that did not.
    """
    facts = sheet.numeric_facts()
    citations: list[Citation] = []
    uncited: list[NumericSpan] = []
    seen: set[tuple[str, str]] = set()

    for span in extract_spans(text):
        fact = resolve(span, facts)
        if fact is None:
            uncited.append(span)
            continue
        identity = (span.text, fact.source)
        if identity in seen:
            continue
        seen.add(identity)
        citations.append(Citation(claim=span.text, source=fact.source, value=fact.value))

    if policy is CitationPolicy.STRIP and uncited:
        text = _strip_sentences(text, uncited)
        return ValidationResult(text=text, citations=citations, uncited_numeric_spans=[])

    return ValidationResult(
        text=text,
        citations=citations,
        uncited_numeric_spans=[span.text for span in uncited],
    )


def _strip_sentences(text: str, uncited: list[NumericSpan]) -> str:
    """Drop every sentence containing an unresolved number."""
    sentences = _SENTENCE.split(text)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for sentence in sentences:
        start = text.find(sentence, cursor)
        offsets.append((start, start + len(sentence)))
        cursor = start + len(sentence)

    kept = [
        sentence
        for sentence, (start, end) in zip(sentences, offsets, strict=True)
        if not any(start <= span.start < end for span in uncited)
    ]
    return " ".join(kept).strip()
