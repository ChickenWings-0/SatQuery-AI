"""The declarative tool registry and the capability matcher in front of it."""

from satquery.registry.capability_match import (
    CHECK_ORDER,
    SOFT_CHECKS,
    CapabilityCheck,
    MatchResult,
    MatchStatus,
    evaluate,
    match,
)
from satquery.registry.registry import (
    RegistryError,
    ToolRegistry,
    default_registry,
    load_registry,
    parse_registry,
)

__all__ = [
    "CHECK_ORDER",
    "SOFT_CHECKS",
    "CapabilityCheck",
    "MatchResult",
    "MatchStatus",
    "RegistryError",
    "ToolRegistry",
    "default_registry",
    "evaluate",
    "load_registry",
    "match",
    "parse_registry",
]
