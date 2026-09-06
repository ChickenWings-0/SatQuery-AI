"""The agentic controller: deterministic classification, planning and execution."""

from satquery.agent.aggregator import Aggregation, aggregate, compose
from satquery.agent.executor import (
    DagExecutor,
    ExecutionCache,
    ExecutionReport,
    ToolFailedNoFallbackError,
    cache_key,
)
from satquery.agent.pipeline import AnalysisRequest, AnalysisResult, analyze
from satquery.agent.planner import (
    PlanResult,
    PolicyEntry,
    PolicyTable,
    default_table,
    lint_against_registry,
    modality_key,
    plan_for,
    resolve_key,
)
from satquery.agent.query_parser import ParsedQuery, fill_slots, load_vocabulary, normalise, parse
from satquery.agent.task_classifier import (
    Classification,
    QueryUnclassifiableError,
    candidates,
    classify,
)

__all__ = [
    "Aggregation",
    "AnalysisRequest",
    "AnalysisResult",
    "Classification",
    "DagExecutor",
    "ExecutionCache",
    "ExecutionReport",
    "ParsedQuery",
    "PlanResult",
    "PolicyEntry",
    "PolicyTable",
    "QueryUnclassifiableError",
    "ToolFailedNoFallbackError",
    "aggregate",
    "analyze",
    "cache_key",
    "candidates",
    "classify",
    "compose",
    "default_table",
    "fill_slots",
    "lint_against_registry",
    "load_vocabulary",
    "modality_key",
    "normalise",
    "parse",
    "plan_for",
    "resolve_key",
]
