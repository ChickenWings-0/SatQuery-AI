"""The end-to-end analysis pipeline: images plus a query in, an AuditTrace out.

This is the only place that knows the order of the whole thing, which is exactly
where that knowledge should live — no router has to, and no tool is allowed to.

    ingest -> classify -> plan -> capability match -> execute
           -> FactSheet -> answer -> cite -> confidence -> trace

Every stage is deterministic apart from wall-clock timings and the trace id.
Master.md §3.1 makes that a hard requirement rather than a nicety: the observable
trace is what gets scored, so two identical runs have to produce two identical
traces.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

from satquery.agent import aggregator, planner, task_classifier
from satquery.agent.executor import DagExecutor, ExecutionCache, ToolFailedNoFallbackError
from satquery.agent.query_parser import parse
from satquery.evidence import confidence as confidence_module
from satquery.evidence import fact_sheet as fact_sheet_module
from satquery.evidence.citation_validator import CitationPolicy
from satquery.ingest.bands import resolve_bands
from satquery.ingest.errors import IncompatibleInputsError
from satquery.ingest.pipeline import IngestResult, SourceImage
from satquery.registry.registry import ToolRegistry, default_registry
from satquery.render.artifact_store import ArtifactStore
from satquery.schemas.compatibility import CompatibilityReport
from satquery.schemas.enums import CheckStatus, Overall, PairType, TaskType
from satquery.schemas.trace import AuditTrace, QuerySpec, WarningItem
from satquery.tools.base import ImageBundle, Tool
from satquery.tools.catalog import implementations
from satquery.trace import builder
from satquery.trace.store import TraceStore

DEFAULT_SEED: Final[int] = 0


@dataclass
class AnalysisRequest:
    """Everything one analysis needs beyond the ingested images."""

    query: str
    sources: Sequence[SourceImage]
    ingest: IngestResult
    trace_id: str = ""
    seed: int = DEFAULT_SEED
    citation_policy: CitationPolicy = CitationPolicy.FLAG
    allow_generic_fallback: bool = True
    enable_tools: Sequence[str] | None = None
    disable_tools: Sequence[str] = ()


@dataclass
class AnalysisResult:
    """The finished analysis, ready to be shaped into any response."""

    trace: AuditTrace
    warnings: list[WarningItem] = field(default_factory=list)


def bundles_for(request: AnalysisRequest) -> list[ImageBundle]:
    """Join each manifest to its file and its resolved band map.

    Band resolution happens once, here, so no tool ever has to know how a given
    sensor names its channels.
    """
    by_filename = {source.filename: source for source in request.sources}
    bundles: list[ImageBundle] = []
    for index, manifest in enumerate(request.ingest.inputs):
        source = by_filename.get(manifest.filename)
        if source is None:
            source = request.sources[index]
        bundles.append(
            ImageBundle(
                manifest=manifest,
                path=source.path,
                resolved_bands=resolve_bands(
                    sensor=manifest.sensor_guess,
                    band_count=manifest.band_count,
                    band_names=manifest.band_names,
                ),
            )
        )
    return bundles


def refuse_if_incompatible(report: CompatibilityReport) -> None:
    """Turn a ``FAIL`` compatibility verdict into the right 422.

    Pre-flight explains; analysis refuses. Planning over images the checks say
    cannot be compared would put a confident sentence in front of a judge about
    a relationship the pixels do not support, and no confidence cap makes that
    acceptable.

    Raises:
        IncompatibleInputsError: The images cannot be analysed together.
    """
    if report.overall is not Overall.FAIL and report.pair_type is not PairType.INCOMPATIBLE:
        return
    failed = next((c for c in report.checks if c.status is CheckStatus.FAIL), None)
    raise IncompatibleInputsError(
        check=failed.name.value if failed else "",
        detail=(
            failed.detail
            if failed
            else "The uploaded images did not pass the compatibility checks."
        ),
    )


def _availability_overrides(
    registry: ToolRegistry,
    enable: Sequence[str] | None,
    disable: Sequence[str],
) -> dict[str, bool]:
    """Turn ``options.enable_tools`` / ``disable_tools`` into registry overrides."""
    overrides: dict[str, bool] = {}
    if enable is not None:
        allowed = set(enable)
        overrides.update({name: name in allowed for name in registry.names})
    for name in disable:
        overrides[name] = False
    return overrides


async def analyze(
    request: AnalysisRequest,
    store: ArtifactStore,
    registry: ToolRegistry | None = None,
    tools: Mapping[str, Tool] | None = None,
    cache: ExecutionCache | None = None,
    traces: TraceStore | None = None,
) -> AnalysisResult:
    """Run the whole pipeline and return the trace it produced.

    Args:
        request: The query, the uploaded files and the ingestion result.
        store: Where evidence artifacts are written.
        registry: Override for tests; defaults to the committed registry.
        tools: Implementation overrides layered over the built-ins.
        cache: Execution cache; defaults to the process-wide one.
        traces: When given, the finished trace is persisted to it.

    Returns:
        An :class:`AnalysisResult` carrying the full :class:`AuditTrace`.

    Raises:
        IncompatibleInputsError: The compatibility battery returned ``FAIL``.
        ToolFailedNoFallbackError: A non-optional step could not be recovered.
        QueryUnclassifiableError: The query could not be classified and generic
            fallback was disabled.
    """
    started = time.perf_counter()
    created_at = datetime.now(UTC)
    trace_id = request.trace_id or builder.new_trace_id()

    base_registry = registry or default_registry()
    overrides = _availability_overrides(
        base_registry, request.enable_tools, request.disable_tools
    )
    active_registry = base_registry.with_availability(overrides) if overrides else base_registry

    ingest = request.ingest
    refuse_if_incompatible(ingest.compatibility)
    parsed = parse(request.query)
    warnings: list[WarningItem] = list(ingest.warnings)

    classification = task_classifier.classify(
        query=request.query,
        pair_type=ingest.compatibility.pair_type,
        allow_generic_fallback=request.allow_generic_fallback,
    )
    warnings += [
        WarningItem(code=code, message=message) for code, message in classification.warnings
    ]

    # UNSUPPORTED never reaches the policy table as a key: it routes to the
    # generic entry by construction, which is what flags it in the trace.
    task = classification.primary
    plan = planner.plan_for(
        task=task if task is not TaskType.UNSUPPORTED else TaskType.UNSUPPORTED,
        pair_type=ingest.compatibility.pair_type,
        manifests=ingest.inputs,
    )
    warnings += [WarningItem(code=code, message=message) for code, message in plan.warnings]

    executor = DagExecutor(
        registry=active_registry,
        tools=implementations(tools),
        store=store,
        cache=cache,
    )
    report = await executor.run(
        plan=plan,
        images=bundles_for(request),
        trace_id=trace_id,
        common_grid=ingest.compatibility.common_grid,
        slots=classification.slots,
        seed=request.seed,
        question=parsed.raw,
    )
    warnings += report.warnings

    sheet = fact_sheet_module.build(report.executions, active_registry)
    warnings += [
        WarningItem(code="SCALAR_SCHEMA_VIOLATION", message=violation)
        for violation in sheet.violations
    ]

    # The VLM is the synthesiser, not the author of record: its text is only
    # accepted after the same CitationValidator pass a templated answer gets, and
    # when no synthesiser ran the template writes the answer exactly as it did in
    # Phase 3. That is why an unservable model degrades the answer's confidence
    # rather than the request.
    generated = aggregator.vlm_answer(report.executions, report.artifacts)
    aggregation = aggregator.aggregate(
        task=task,
        pair_type=plan.pair_type,
        sheet=sheet,
        executions=report.executions,
        text=generated.text if generated else None,
        generator=generated.generator if generated else None,
        policy=request.citation_policy,
    )
    if aggregation.has_uncited:
        warnings.append(
            WarningItem(
                code="UNCITED_NUMERIC_SPANS",
                message=(
                    "The answer contains numbers that resolve to no measurement: "
                    f"{', '.join(aggregation.answer.uncited_numeric_spans)}."
                ),
            )
        )

    conditions = confidence_module.Conditions(
        overall_compat=ingest.compatibility.overall,
        task_proposed_by_llm=classification.task_proposed_by_llm,
        template_fallback=aggregation.answer.template_fallback,
        has_uncited_claims=aggregation.has_uncited,
        generic_plan=plan.is_generic,
        caps_waived=frozenset(plan.caps_waived),
    )
    score = confidence_module.score(
        task_confidence=classification.confidence,
        manifests=ingest.inputs,
        executions=report.executions,
        registry=active_registry,
        sheet=sheet,
        conditions=conditions,
        gsd_out_of_range=report.gsd_out_of_range,
    )

    trace = builder.build(
        trace_id=trace_id,
        query=QuerySpec(raw=parsed.raw, normalized=parsed.normalized, language=parsed.language),
        resolved_task=builder.resolved_task(
            primary=classification.primary,
            secondary=classification.secondary,
            slots=classification.slots,
            confidence=classification.confidence,
            classifier=classification.classifier,
        ),
        inputs=ingest.inputs,
        compatibility=ingest.compatibility,
        plan=plan,
        executions=report.executions,
        artifacts=report.artifacts,
        fact_sheet=sheet,
        answer=aggregation.answer,
        confidence=score,
        duration_ms=int((time.perf_counter() - started) * 1000),
        warnings=warnings,
        errors=report.errors,
        created_at=created_at,
    )
    if traces is not None:
        traces.put(trace)
    return AnalysisResult(trace=trace, warnings=warnings)


__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "IncompatibleInputsError",
    "ToolFailedNoFallbackError",
    "analyze",
    "bundles_for",
    "refuse_if_incompatible",
]
