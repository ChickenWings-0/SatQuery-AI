"""Evidence: what was measured, what was claimed, and how far the two agree."""

from satquery.evidence.citation_validator import (
    CitationPolicy,
    NumericSpan,
    ValidationResult,
    extract_spans,
    format_number,
    resolve,
    unit_compatible,
    validate,
    within_tolerance,
)
from satquery.evidence.confidence import (
    CAPS,
    METHOD,
    WEIGHTS,
    Conditions,
    cross_tool_agreement,
    input_quality,
    load_agreement_pairs,
    score,
    tool_mean,
)
from satquery.evidence.fact_sheet import Fact, FactSheet, build, validate_scalar

__all__ = [
    "CAPS",
    "METHOD",
    "WEIGHTS",
    "CitationPolicy",
    "Conditions",
    "Fact",
    "FactSheet",
    "NumericSpan",
    "ValidationResult",
    "build",
    "cross_tool_agreement",
    "extract_spans",
    "format_number",
    "input_quality",
    "load_agreement_pairs",
    "resolve",
    "score",
    "tool_mean",
    "unit_compatible",
    "validate",
    "validate_scalar",
    "within_tolerance",
]
