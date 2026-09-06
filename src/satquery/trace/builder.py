"""Assembling the :class:`AuditTrace` (API_CONTRACT §3.11).

The trace is the requirement-5 deliverable and the thing the rubric actually
reads, so this module's only job is to put every part in its frozen place
without editorialising: what was asked, what was uploaded, what the checks said,
what the planner chose, what each tool did with which parameters, what was
measured, what was claimed, and how confident the system is in the result.

Nothing is computed here that is not already known. If a value had to be derived
to fill a field, that derivation belongs in the module that owns it.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from datetime import UTC, datetime

from satquery.agent.planner import PlanResult
from satquery.evidence.fact_sheet import FactSheet
from satquery.schemas.compatibility import CompatibilityReport
from satquery.schemas.enums import TaskType
from satquery.schemas.manifest import InputManifest
from satquery.schemas.tool import Execution
from satquery.schemas.trace import (
    Answer,
    ArtifactRef,
    AuditTrace,
    Confidence,
    ErrorItem,
    QuerySpec,
    ResolvedTask,
    WarningItem,
)


def new_trace_id() -> str:
    """Return a fresh 32-character lowercase hex trace id."""
    return secrets.token_hex(16)


def build(
    trace_id: str,
    query: QuerySpec,
    resolved_task: ResolvedTask,
    inputs: Sequence[InputManifest],
    compatibility: CompatibilityReport,
    plan: PlanResult,
    executions: Sequence[Execution],
    artifacts: Sequence[ArtifactRef],
    fact_sheet: FactSheet,
    answer: Answer,
    confidence: Confidence,
    duration_ms: int,
    warnings: Sequence[WarningItem] = (),
    errors: Sequence[ErrorItem] = (),
    created_at: datetime | None = None,
) -> AuditTrace:
    """Assemble one replayable record of an analysis."""
    return AuditTrace(
        trace_id=trace_id,
        created_at=created_at or datetime.now(UTC),
        duration_ms=duration_ms,
        query=query,
        resolved_task=resolved_task,
        inputs=list(inputs),
        compatibility=compatibility,
        plan=plan.plan,
        executions=sorted(executions, key=lambda execution: execution.step),
        artifacts=list(artifacts),
        fact_sheet=fact_sheet.as_dict(),
        answer=answer,
        confidence=confidence,
        warnings=list(warnings),
        errors=list(errors),
    )


def resolved_task(
    primary: TaskType,
    secondary: Sequence[TaskType],
    slots: dict[str, object],
    confidence: float,
    classifier: str,
) -> ResolvedTask:
    """Shape the classifier's verdict for the trace."""
    return ResolvedTask(
        primary=primary,
        secondary=list(secondary),
        slots=dict(slots),
        confidence=confidence,
        classifier=classifier,
    )
