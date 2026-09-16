"""The async DAG executor (AGENT_POLICY_DAG.md §6).

Scheduling is deterministic: dependencies first, ties broken by ascending step
number, and artifacts numbered serially in step order after each wave even
though the wave itself ran concurrently. That last detail is what lets two runs
of the same analysis produce identical ``art_*`` ids without giving up
parallelism.

Three rules carry most of the behaviour:

* **A tool failure is never a 5xx.** It becomes a fallback, or a ``FAILED``
  execution and a reduced confidence. The only exception is a step the policy
  entry marks non-optional — currently just the renderer, without which there is
  no evidence at all.
* **Status propagates downhill.** A step consuming a ``DEGRADED`` step's output
  is at most ``DEGRADED`` however well it ran; a step consuming a ``FAILED`` or
  ``SKIPPED`` step is ``SKIPPED``. Anything else would let the trace claim a
  measurement was sound when its input was not.
* **The cache key excludes wall-clock.** ``trace_id``, timestamps, the query
  text and upload filenames are all deliberately out of it, so the same pixels
  and the same parameters hit the cache across traces.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from satquery.agent.concurrency import (
    MAX_PARALLEL_GPU_TOOLS,
    MAX_PARALLEL_TOOLS,
    DeviceGates,
    ToolTimeoutError,
)
from satquery.agent.events import Emit, emit_to
from satquery.agent.planner import ArtifactSelector, PlannedStep, PlanResult
from satquery.evidence import fact_sheet as fact_sheet_module
from satquery.registry.capability_match import CapabilityCheck, MatchStatus, match
from satquery.registry.registry import ToolRegistry
from satquery.render.artifact_store import ArtifactStore
from satquery.render.renderer import render_views
from satquery.schemas.compatibility import CommonGrid
from satquery.schemas.enums import ArtifactType, Device, ToolCategory, ToolStatus
from satquery.schemas.tool import Execution, ToolSpec
from satquery.schemas.trace import ArtifactGeo, ArtifactRef, ErrorItem, WarningItem
from satquery.tools.base import (
    ArtifactDraft,
    Draft,
    ImageBundle,
    PixelReader,
    RenderDraft,
    Tool,
    ToolContext,
    ToolResult,
)
from satquery.tools.catalog import CHECKPOINT_TOOLS, checkpoint_fingerprint

MIN_TIMEOUT_MS: Final[int] = 5_000
TIMEOUT_FACTOR: Final[int] = 3

COLD_START_TIMEOUT_MS: Final[int] = 90_000
"""The budget a tool gets on its first call in this process.

``est_ms`` in the registry describes steady-state work: a model already resident,
an import already done. The first call pays neither of those. The detector reads a
checkpoint and builds a ROCm context — tens of seconds on a cold card, against a
``3 x 1800 ms`` budget — and it is not only the weight-loading tools that suffer,
because a cold load holds the GIL. ``spectral_index_analyzer`` normally finishes
in 270 ms; running beside a cold detector it overran 5 s and failed.

So the allowance is per-tool and unconditional on the first call. The timeout is a
backstop against a wedged tool, not a latency target, and it is paid once per tool
per process. What it buys is the thing the first request after a restart could not
otherwise have: the same answer the second request gets."""

_WARMED: Final[set[str]] = set()
"""Tools that have completed a run in this process.

Process-wide because the costs it tracks are process-wide — module imports,
``ModelCache`` and the VLM backend all live for the life of the interpreter, so
the second request pays none of them no matter which executor instance serves
it."""

_MIME_FOR_TYPE: Final[dict[ArtifactType, str]] = {
    ArtifactType.SCALARS: "application/json",
    ArtifactType.TEXT: "application/json",
    ArtifactType.BBOX_SET: "application/json",
    ArtifactType.GEOJSON: "application/geo+json",
}


class ToolFailedNoFallbackError(RuntimeError):
    """A step the policy entry declared non-optional failed with no fallback.

    This is the one tool-level condition that is allowed to become a 500: the
    renderer failing means there is no evidence, no VLM input and nothing
    honest left to say.
    """

    def __init__(self, step: int, tool: str, detail: str) -> None:
        """Record which step could not be recovered."""
        super().__init__(f"step {step} ({tool}) failed with no fallback: {detail}")
        self.step = step
        self.tool = tool
        self.detail = detail


def canonical_json(payload: Any) -> str:
    """Stable JSON: sorted keys, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(
    tool_name: str,
    tool_version: str,
    params: Mapping[str, Any],
    input_hashes: Sequence[str],
    common_grid_hash: str,
    salt: str = "",
) -> str:
    """The content-addressed identity of one tool invocation (§6.2).

    Deliberately excluded: ``trace_id``, ``created_at``, the raw query text,
    upload filenames and every wall-clock value. Two different questions that
    happen to need the same measurement share the answer, which is the whole
    point of caching a deterministic tool.

    ``salt`` is the exception, and it exists for exactly one class of tool. A
    synthesiser's output *is* a function of the question and of the evidence it
    was handed, neither of which appears anywhere else in this key — so caching a
    ``vlm_*`` step without one would serve the answer to the previous question
    (see :meth:`DagExecutor._cache_salt`).
    """
    digest = hashlib.sha256()
    digest.update(tool_name.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(tool_version.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(canonical_json(dict(sorted(params.items()))).encode("utf-8"))
    digest.update(b"\x00")
    digest.update("".join(sorted(input_hashes)).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(common_grid_hash.encode("utf-8"))
    if salt:
        digest.update(b"\x00")
        digest.update(salt.encode("utf-8"))
    return digest.hexdigest()


def grid_hash(grid: CommonGrid | None) -> str:
    """Hash the common grid, or name its absence for a single-image analysis."""
    if grid is None:
        return "single"
    return hashlib.sha256(canonical_json(grid.model_dump(mode="json")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CapturedArtifact:
    """One materialised artifact, kept whole so a replay need not recompute it.

    The ``art_*`` id inside :attr:`template` belongs to the trace that first
    produced it and is discarded on replay — only the encoded bytes and the
    descriptive fields carry over.
    """

    template: ArtifactRef
    blobs: dict[str, bytes]
    payload: Any | None = None


@dataclass(frozen=True)
class CachedStep:
    """Everything one invocation produced, ready to be replayed into a new trace."""

    result: ToolResult
    artifacts: tuple[CapturedArtifact, ...]


DEFAULT_CACHE_BYTES: Final[int] = 512 << 20
"""The default byte budget for cached artifact blobs.

The cache used to be bounded by entry count alone. Each entry holds every
*encoded* artifact of one step — a rendered view set is several PNGs and a
GeoTIFF — so 256 entries could be gigabytes of resident bytes on a long demo.
Overridable with ``SATQUERY_CACHE_MB``."""


def _cache_budget_bytes() -> int:
    """Read ``SATQUERY_CACHE_MB``, falling back to the default on anything odd."""
    raw = os.environ.get("SATQUERY_CACHE_MB", "").strip()
    return int(raw) << 20 if raw.isdigit() else DEFAULT_CACHE_BYTES


def _entry_bytes(entry: CachedStep) -> int:
    """The bytes an entry keeps resident: its encoded artifact blobs."""
    return sum(len(blob) for artifact in entry.artifacts for blob in artifact.blobs.values())


class ExecutionCache:
    """An in-process cache of deterministic tool results.

    Entries hold both the scalars and the *encoded* artifact bytes. Keeping the
    bytes is what makes a cache hit actually cheap: without them a replay would
    re-read the rasters and re-encode every PNG, and the only thing saved would
    be the arithmetic, which was never the expensive part.

    Bounded twice: by entry count and by resident bytes. Eviction is oldest
    first on either limit.
    """

    def __init__(self, max_entries: int = 256, max_bytes: int | None = None) -> None:
        """Create a cache holding at most *max_entries* results and *max_bytes* of blobs."""
        self.max_entries = max_entries
        self.max_bytes = max_bytes if max_bytes is not None else _cache_budget_bytes()
        self._entries: dict[str, CachedStep] = {}
        self._sizes: dict[str, int] = {}
        self.resident_bytes = 0

    def get(self, key: str) -> CachedStep | None:
        """Return a cached step, or None."""
        return self._entries.get(key)

    def put(self, key: str, entry: CachedStep) -> None:
        """Store a step, evicting the oldest entries until both limits hold.

        An entry larger than the whole budget is not cached at all: caching it
        would evict everything else to hold one result.
        """
        if key in self._entries:
            return
        size = _entry_bytes(entry)
        if size > self.max_bytes:
            return
        while self._entries and (
            len(self._entries) >= self.max_entries or self.resident_bytes + size > self.max_bytes
        ):
            self._evict_oldest()
        self._entries[key] = entry
        self._sizes[key] = size
        self.resident_bytes += size

    def _evict_oldest(self) -> None:
        oldest = next(iter(self._entries))
        del self._entries[oldest]
        self.resident_bytes -= self._sizes.pop(oldest)

    def clear(self) -> None:
        """Drop every entry."""
        self._entries.clear()
        self._sizes.clear()
        self.resident_bytes = 0

    def __len__(self) -> int:
        """Number of cached results."""
        return len(self._entries)


DEFAULT_CACHE: Final[ExecutionCache] = ExecutionCache()


@dataclass
class ExecutionReport:
    """Everything one run of the DAG produced."""

    executions: list[Execution] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    checks: dict[int, list[CapabilityCheck]] = field(default_factory=dict)
    warnings: list[WarningItem] = field(default_factory=list)
    errors: list[ErrorItem] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    gsd_out_of_range: bool = False
    cancelled: bool = False

    def by_step(self, step: int) -> Execution | None:
        """Return the execution record for one step."""
        return next((e for e in self.executions if e.step == step), None)


@dataclass
class _StepState:
    """Bookkeeping for one step while the DAG runs."""

    status: ToolStatus
    artifacts: list[ArtifactRef] = field(default_factory=list)


class DagExecutor:
    """Runs a :class:`PlanResult` and produces the executions the trace records."""

    def __init__(
        self,
        registry: ToolRegistry,
        tools: Mapping[str, Tool],
        store: ArtifactStore,
        cache: ExecutionCache | None = None,
        max_parallel: int = MAX_PARALLEL_TOOLS,
        max_parallel_gpu: int = MAX_PARALLEL_GPU_TOOLS,
        emit: Emit | None = None,
        gates: DeviceGates | None = None,
    ) -> None:
        """Wire the executor to a registry, an implementation table and a store.

        *emit*, when given, receives the ``step_started`` / ``artifact`` /
        ``step_completed`` events of API_CONTRACT §5 as they happen. ``None`` —
        the default, and what ``/v1/analyze`` passes — makes every emission a
        single ``is None`` check.

        *gates* is the process-wide :class:`DeviceGates` the application owns.
        Every executor serving a request must share one, or two requests each
        hold their own GPU permit. Left ``None`` — tests, scripts — the executor
        gets a private set sized from *max_parallel* / *max_parallel_gpu*.
        """
        self.registry = registry
        self.tools = dict(tools)
        self.store = store
        self.cache = cache if cache is not None else DEFAULT_CACHE
        self.emit = emit
        self.gates = gates if gates is not None else DeviceGates(max_parallel, max_parallel_gpu)

    async def run(
        self,
        plan: PlanResult,
        images: Sequence[ImageBundle],
        trace_id: str,
        common_grid: CommonGrid | None = None,
        slots: Mapping[str, Any] | None = None,
        seed: int = 0,
        question: str = "",
        cancelled: asyncio.Event | None = None,
    ) -> ExecutionReport:
        """Execute every step of the plan in dependency order.

        *cancelled*, when set between waves, stops the DAG at the next step
        boundary: the wave in flight finishes (a worker thread cannot be killed),
        every step not yet started is recorded ``SKIPPED`` with the reason, and
        the partial report is returned. This is the deterministic half of
        ``DELETE /v1/jobs/{id}``; the request-level half is the task cancel.

        Raises:
            ToolFailedNoFallbackError: A non-optional step could not be recovered.
        """
        report = ExecutionReport()
        pixels = PixelReader()
        by_id = {image.id: image for image in images}
        state: dict[int, _StepState] = {}
        artifact_hashes: dict[str, str] = {}
        grid = grid_hash(common_grid)

        pending = {step.step: step for step in plan.steps}
        while pending:
            if cancelled is not None and cancelled.is_set():
                self._skip_all(pending, state, report, "cancelled by the client")
                report.cancelled = True
                break
            ready = sorted(
                (
                    step
                    for step in pending.values()
                    if all(dependency in state for dependency in step.depends_on)
                ),
                key=lambda step: step.step,
            )
            if not ready:
                # Unreachable for a linted table, but a cycle must not hang the
                # request: skip what is left and say so.
                self._skip_all(pending, state, report, "the step's dependencies form a cycle")
                break

            for step in ready:
                spec = self.registry.get(step.tool)
                emit_to(
                    self.emit,
                    "step_started",
                    {
                        "step": step.step,
                        "tool": step.tool,
                        "est_ms": spec.est_ms if spec is not None else 0,
                    },
                )

            # The cold-start allowance is a property of the *wave*, not of the
            # cold tool alone: a first-time checkpoint load holds the GIL, and a
            # warmed CPU tool running beside it inherits the stall. Budgeting it
            # at steady state failed spectral_index_analyzer (270 ms warm) next
            # to a cold detector, and that ran a good change scene into a
            # templated answer — on whether some earlier request had happened
            # to warm the detector first.
            wave_cold = any(step.tool not in _WARMED for step in ready)
            outcomes = await asyncio.gather(
                *(
                    self._run_step(
                        step=step,
                        plan=plan,
                        by_id=by_id,
                        state=state,
                        pixels=pixels,
                        trace_id=trace_id,
                        grid=grid,
                        artifact_hashes=artifact_hashes,
                        report=report,
                        slots=dict(slots or {}),
                        seed=seed,
                        question=question,
                        wave_cold=wave_cold,
                    )
                    for step in ready
                )
            )

            # Materialisation is serial and in step order, so artifact ids do
            # not depend on which member of the wave finished first.
            for step, (execution, result, cached) in zip(ready, outcomes, strict=True):
                artifacts: list[ArtifactRef] = []
                key = str(execution.params.pop("__cache_key__", ""))
                stored_at = time.perf_counter()

                if cached is not None:
                    artifacts = self._replay(
                        cached.artifacts,
                        trace_id=trace_id,
                        step=step.step,
                        first_index=len(report.artifacts),
                        report=report,
                    )
                elif result is not None and result.artifacts:
                    artifacts, captured = self._materialize(
                        drafts=result.artifacts,
                        data=result.data,
                        trace_id=trace_id,
                        step=step.step,
                        first_index=len(report.artifacts),
                        report=report,
                    )
                    if key:
                        self.cache.put(key, CachedStep(result=result, artifacts=captured))

                # Encoding and writing the blobs is the renderer's real cost, and
                # it happens here rather than inside run(); charging it to the
                # step keeps duration_ms honest.
                execution.duration_ms += int((time.perf_counter() - stored_at) * 1000)
                report.artifacts.extend(artifacts)

                for offset, ref in enumerate(artifacts):
                    artifact_hashes[ref.id] = hashlib.sha256(f"{key}:{offset}".encode()).hexdigest()

                execution.output_refs = [artifact.id for artifact in artifacts]
                if execution.tool == "spectral_renderer" and artifacts:
                    # Assigned after validation, so it has to carry the field's
                    # declared float type rather than relying on coercion.
                    execution.scalars["views_rendered"] = float(len(artifacts))
                report.executions.append(execution)

                # Evidence streams: each artifact is announced the moment it is
                # addressable, before the step that produced it is summarised.
                for ref in artifacts:
                    emit_to(self.emit, "artifact", ref.model_dump(mode="json"))
                emit_to(
                    self.emit,
                    "step_completed",
                    {
                        "step": execution.step,
                        "status": execution.status.value,
                        "duration_ms": execution.duration_ms,
                        "confidence": execution.confidence,
                        "output_refs": list(execution.output_refs),
                    },
                )
                state[step.step] = _StepState(execution.status, artifacts)
                del pending[step.step]

        report.executions.sort(key=lambda execution: execution.step)
        return report

    # ----------------------------------------------------------------- one step

    async def _run_step(
        self,
        step: PlannedStep,
        plan: PlanResult,
        by_id: Mapping[str, ImageBundle],
        state: Mapping[int, _StepState],
        pixels: PixelReader,
        trace_id: str,
        grid: str,
        artifact_hashes: Mapping[str, str],
        report: ExecutionReport,
        slots: dict[str, Any],
        seed: int,
        question: str = "",
        wave_cold: bool = False,
    ) -> tuple[Execution, ToolResult | None, CachedStep | None]:
        """Resolve, match, run and record one step."""
        # A dependency that produced nothing only blocks this step when this step
        # actually needs it, and there are exactly two ways it can: the policy
        # entry declared that dependency non-optional — load-bearing for
        # everything downstream — or this step selects an artifact out of it.
        #
        # Anything else is an ordering edge, and cascading it is what turned a
        # good change run into a templated answer: on RGB-only imagery
        # ``spectral_index_analyzer`` has no NIR to compute NDVI from and skips,
        # which is the case ``optional`` exists to describe. The synthesiser
        # depends on it only so the indices reach the FactSheet before the
        # prompt is built; it reads ``@1`` and ``@2:CHANGE_MASK``, never ``@4``.
        # Skipping it left a measured mask and a full statistics step unused.
        upstream_steps = {candidate.step: candidate for candidate in plan.steps}
        consumed = {selector.step for selector in step.selectors}
        unproductive = [
            dependency
            for dependency in step.depends_on
            if state[dependency].status in {ToolStatus.FAILED, ToolStatus.SKIPPED}
        ]
        blocked = [
            dependency
            for dependency in unproductive
            if dependency in consumed or not upstream_steps[dependency].optional
        ]
        if blocked:
            names = ", ".join(f"step {d}" for d in blocked)
            report.errors.append(
                ErrorItem(
                    code="DEPENDENCY_SKIPPED",
                    message=f"{step.tool} was skipped because {names} did not produce output.",
                    step=step.step,
                )
            )
            return self._skipped(step, f"{names} did not produce output"), None, None

        for dependency in unproductive:
            # Recorded rather than silent: the step ran with less evidence than
            # the entry describes, and the trace has to be able to say so.
            report.warnings.append(
                WarningItem(
                    code="DEPENDENCY_INCOMPLETE",
                    message=(
                        f"{step.tool} ran without step {dependency} "
                        f"({upstream_steps[dependency].tool}), which produced no output."
                    ),
                )
            )

        images = [by_id[ref] for ref in step.image_refs if ref in by_id]
        upstream = self._resolve_artifacts(step.selectors, state)
        decision = match(
            tool=step.tool,
            registry=self.registry,
            pair_type=plan.pair_type,
            images=images,
            artifact_count=len(upstream),
        )
        report.checks[step.step] = decision.checks
        if decision.gsd_out_of_range:
            report.gsd_out_of_range = True
            for warning in decision.warnings:
                report.warnings.append(
                    WarningItem(code=f"CHECK_{warning.name.upper()}", message=warning.detail)
                )

        if decision.status is MatchStatus.SKIPPED or decision.spec is None:
            if not step.optional:
                raise ToolFailedNoFallbackError(step.step, step.tool, decision.reason)
            report.warnings.append(
                WarningItem(code="TOOL_SKIPPED", message=f"{step.tool}: {decision.reason}")
            )
            return self._skipped(step, decision.reason), None, None

        if decision.status is MatchStatus.SUBSTITUTED:
            report.warnings.append(
                WarningItem(
                    code="TOOL_SUBSTITUTED",
                    message=f"{step.tool} was replaced by {decision.tool}. {decision.reason}",
                )
            )

        context = ToolContext(
            trace_id=trace_id,
            step=step.step,
            pair_type=plan.pair_type,
            images=images,
            artifacts=upstream,
            data=report.data,
            pixels=pixels,
            slots=slots,
            question=question,
            # Every dependency of this step is in an earlier wave and is therefore
            # already recorded, so the sheet a synthesiser reads holds exactly the
            # evidence the policy table said it should wait for — no more, and
            # never a measurement from a step that has not finished.
            facts=fact_sheet_module.build(report.executions, self.registry),
            seed=seed,
            store=self.store,
        )
        input_hashes = [image.manifest.sha256 for image in images]
        input_hashes += [artifact_hashes.get(a.id, a.id) for a in upstream]
        input_refs = [image.id for image in images] + [a.id for a in upstream]

        execution, result, cached = await self._invoke(
            step=step,
            spec=decision.spec,
            context=context,
            input_hashes=input_hashes,
            input_refs=input_refs,
            grid=grid,
            fallback_of=decision.fallback_of,
            report=report,
            wave_cold=wave_cold,
        )

        degraded = any(state[d].status is ToolStatus.DEGRADED for d in step.depends_on)
        if degraded and execution.status is ToolStatus.OK:
            execution.status = ToolStatus.DEGRADED
            execution.error = execution.error or "an upstream step ran degraded"
        return execution, result, cached

    async def _invoke(
        self,
        step: PlannedStep,
        spec: ToolSpec,
        context: ToolContext,
        input_hashes: list[str],
        input_refs: list[str],
        grid: str,
        fallback_of: str | None,
        report: ExecutionReport,
        wave_cold: bool = False,
    ) -> tuple[Execution, ToolResult | None, CachedStep | None]:
        """Run one tool, falling back once on failure."""
        attempts: list[tuple[ToolSpec, str | None]] = [(spec, fallback_of)]
        if spec.fallback is not None and fallback_of is None:
            # One level only: this is the runtime fallback for the declared tool,
            # and it has to clear capability matching in its own right — a
            # fallback that cannot accept these inputs is not a fallback.
            recovery = match(
                tool=spec.fallback,
                registry=self.registry,
                pair_type=context.pair_type,
                images=context.images,
                artifact_count=len(context.artifacts),
            )
            if recovery.status is MatchStatus.OK and recovery.spec is not None:
                attempts.append((recovery.spec, spec.name))

        last_error = "no implementation is registered"
        for index, (candidate, replaced) in enumerate(attempts):
            implementation = self.tools.get(candidate.name)
            if implementation is None:
                last_error = f"{candidate.name} has no registered implementation"
                continue

            key = cache_key(
                candidate.name,
                candidate.version,
                step.params,
                input_hashes,
                grid,
                salt=self._cache_salt(candidate, context),
            )
            cached = self.cache.get(key)
            if cached is not None:
                execution = self._execution(
                    step,
                    candidate,
                    cached.result,
                    replaced,
                    input_refs,
                    duration_ms=0,
                    cache_hit=True,
                )
                execution.params["__cache_key__"] = key
                # A replay must read like the run it replays: the same
                # substitution note and the same tool notes, or the second
                # submission of one query produces a different trace from the
                # first and the "byte-identical reruns" promise is only true
                # with a cold cache (scripts/e2e_parity.py caught exactly this).
                self._annotate(execution, candidate, cached.result, replaced, index, report)
                return execution, cached.result, cached

            started = time.perf_counter()
            try:
                result = await self._guarded(
                    implementation, candidate, context, step, report, wave_cold=wave_cold
                )
            except Exception as error:  # noqa: BLE001 - a tool failure is data, not a crash
                last_error = f"{type(error).__name__}: {error}"
                report.errors.append(
                    ErrorItem(
                        code="TOOL_FAILED",
                        message=f"{candidate.name} failed: {last_error}",
                        step=step.step,
                    )
                )
                continue

            duration_ms = int((time.perf_counter() - started) * 1000)
            execution = self._execution(
                step, candidate, result, replaced, input_refs, duration_ms, cache_hit=False
            )
            execution.params["__cache_key__"] = key
            self._annotate(execution, candidate, result, replaced, index, report)
            return execution, result, None

        if not step.optional:
            raise ToolFailedNoFallbackError(step.step, step.tool, last_error)

        failed = Execution(
            step=step.step,
            tool=spec.name,
            version=spec.version,
            status=ToolStatus.FAILED,
            device_used=spec.device,
            duration_ms=0,
            params=dict(step.params),
            input_refs=input_refs,
            output_refs=[],
            scalars={},
            confidence=0.0,
            cache_hit=False,
            fallback_of=fallback_of,
            error=last_error,
        )
        return failed, None, None

    @staticmethod
    def _annotate(
        execution: Execution,
        candidate: ToolSpec,
        result: ToolResult,
        replaced: str | None,
        attempt: int,
        report: ExecutionReport,
    ) -> None:
        """Record what the trace must say about this invocation, fresh or replayed."""
        if attempt or replaced:
            # Running something other than what the plan named is a real
            # deviation, whether capability matching or a raised exception
            # caused it, and the trace must show it as one.
            execution.status = ToolStatus.DEGRADED
            execution.error = execution.error or f"substituted for {replaced}"
        for note in result.notes:
            report.warnings.append(
                WarningItem(code="TOOL_NOTE", message=f"{candidate.name}: {note}")
            )

    def _cache_salt(self, spec: ToolSpec, context: ToolContext) -> str:
        """Everything outside the standard key that this tool's output depends on.

        Empty for every deterministic tool: their inputs are the pixels and the
        params, which the key already covers. A VLM step additionally consumes the
        question and the FactSheet, so both go into its identity — otherwise
        "what changed?" and "how much water is there?" over the same pair would
        collide on one cached answer.
        """
        if spec.name in CHECKPOINT_TOOLS:
            # The weights are not in the key: the registry's version string is a
            # constant in the YAML and does not move when a checkpoint is
            # retrained or repointed. Without this, a rewritten bundle keeps
            # serving the mask the old weights produced.
            return checkpoint_fingerprint(spec.name)
        if spec.category is not ToolCategory.VLM:
            return ""
        return canonical_json({"question": context.question, "facts": context.facts.as_dict()})

    async def _guarded(
        self,
        implementation: Tool,
        spec: ToolSpec,
        context: ToolContext,
        step: PlannedStep,
        report: ExecutionReport,
        *,
        wave_cold: bool = False,
    ) -> ToolResult:
        """Run a tool under the right device gate and its per-step timeout.

        The timeout unblocks the request; it cannot kill the worker thread. The
        gate keeps the device permit with the *thread* until it ends (see
        :mod:`satquery.agent.concurrency`), and a timeout that leaves a GPU
        thread running is recorded on the report so the trace says the box is
        carrying work nobody is waiting for. Treating a timeout exactly like a
        failure is what the contract asks for.
        """
        default_ms = max(TIMEOUT_FACTOR * spec.est_ms, MIN_TIMEOUT_MS)
        if wave_cold or spec.name not in _WARMED:
            default_ms = max(default_ms, COLD_START_TIMEOUT_MS)
        # An explicit timeout_ms in the policy table still wins: the allowance is
        # a better default, not an override of a deliberate decision.
        timeout_ms = int(step.params.get("timeout_ms", default_ms))
        try:
            result = await self.gates.run(
                spec.device,
                spec.name,
                lambda: asyncio.to_thread(implementation.run, context, step.params),
                timeout_ms=timeout_ms,
            )
        except ToolTimeoutError as error:
            if error.leaked and spec.device is Device.ROCM_0:
                report.warnings.append(
                    WarningItem(
                        code="TOOL_TIMEOUT_LEAKED_PERMIT",
                        message=(
                            f"{spec.name} exceeded its {timeout_ms} ms budget and its "
                            "thread is still holding the GPU; later GPU steps wait for it"
                        ),
                    )
                )
            raise
        # Only on success: a tool that failed may not have got as far as loading,
        # and charging it the steady-state budget next time would hide that.
        _WARMED.add(spec.name)
        return result

    # ---------------------------------------------------------------- recording

    def _execution(
        self,
        step: PlannedStep,
        spec: ToolSpec,
        result: ToolResult,
        fallback_of: str | None,
        input_refs: list[str],
        duration_ms: int,
        cache_hit: bool,
    ) -> Execution:
        """Turn a tool result into the contract's execution record."""
        params = {**step.params, **result.params}
        return Execution(
            step=step.step,
            tool=spec.name,
            version=spec.version,
            status=result.status,
            device_used=result.device,
            duration_ms=duration_ms,
            params=params,
            input_refs=input_refs,
            output_refs=[],
            scalars=dict(result.scalars),
            confidence=result.confidence,
            cache_hit=cache_hit,
            fallback_of=fallback_of,
            error=result.error,
        )

    def _skip_all(
        self,
        pending: Mapping[int, PlannedStep],
        state: dict[int, _StepState],
        report: ExecutionReport,
        reason: str,
    ) -> None:
        """Record every step still pending as ``SKIPPED`` and tell the client.

        Without the ``step_completed`` events the client would leave these
        nodes PENDING for ever, which reads as a hang rather than a skip.
        """
        for step in sorted(pending.values(), key=lambda s: s.step):
            execution = self._skipped(step, reason)
            report.executions.append(execution)
            state[step.step] = _StepState(ToolStatus.SKIPPED)
            emit_to(
                self.emit,
                "step_completed",
                {
                    "step": execution.step,
                    "status": execution.status.value,
                    "duration_ms": execution.duration_ms,
                    "confidence": execution.confidence,
                    "output_refs": [],
                },
            )

    def _skipped(self, step: PlannedStep, reason: str) -> Execution:
        """Record a step that never ran."""
        spec = self.registry.get(step.tool)
        return Execution(
            step=step.step,
            tool=step.tool,
            version=spec.version if spec else "unknown",
            status=ToolStatus.SKIPPED,
            device_used=spec.device if spec else Device.CPU,
            duration_ms=0,
            params=dict(step.params),
            input_refs=list(step.image_refs),
            output_refs=[],
            scalars={},
            confidence=0.0,
            cache_hit=False,
            fallback_of=None,
            error=reason,
        )

    def _resolve_artifacts(
        self, selectors: Sequence[ArtifactSelector], state: Mapping[int, _StepState]
    ) -> list[ArtifactRef]:
        """Resolve ``@N`` / ``@N:TYPE`` selectors to the artifacts they name."""
        resolved: list[ArtifactRef] = []
        for selector in selectors:
            produced = state.get(selector.step)
            if produced is None:
                continue
            for artifact in produced.artifacts:
                if selector.artifact_type is None or artifact.type is selector.artifact_type:
                    resolved.append(artifact)
        return resolved

    # ----------------------------------------------------------- materialisation

    def _materialize(
        self,
        drafts: Sequence[Draft],
        data: Mapping[str, Any],
        trace_id: str,
        step: int,
        first_index: int,
        report: ExecutionReport,
    ) -> tuple[list[ArtifactRef], tuple[CapturedArtifact, ...]]:
        """Store a step's drafts, name them, and capture them for replay."""
        refs: list[ArtifactRef] = []
        payloads: list[Any] = []

        for draft in drafts:
            index = first_index + len(refs)
            if isinstance(draft, RenderDraft):
                # render_views numbers its own artifacts, which is why the render
                # is deferred to here: this is the first point at which the next
                # free index is known.
                views = render_views(
                    sources=list(draft.sources),
                    pair_type=draft.pair_type,
                    trace_id=trace_id,
                    store=self.store,
                    produced_by_step=step,
                    size=draft.size,
                    first_artifact_index=index,
                )
                for view in views:
                    refs.append(view.artifact)
                    payloads.append(view)
                continue

            artifact_id = f"art_{index}"
            refs.append(self._store_draft(draft, trace_id, artifact_id, step))
            payloads.append(data.get(draft.key))

        captured: list[CapturedArtifact] = []
        for ref, payload in zip(refs, payloads, strict=True):
            if payload is not None:
                report.data[ref.id] = payload
            captured.append(
                CapturedArtifact(template=ref, blobs=self._blobs_of(ref, trace_id), payload=payload)
            )
        return refs, tuple(captured)

    def _replay(
        self,
        captured: Sequence[CapturedArtifact],
        trace_id: str,
        step: int,
        first_index: int,
        report: ExecutionReport,
    ) -> list[ArtifactRef]:
        """Re-register a cached step's artifacts under this trace's ids.

        The bytes are already encoded, so this is a hash and an index write per
        blob — and because the store is content-addressed, the file itself is
        usually already on disk and is not rewritten at all.
        """
        refs: list[ArtifactRef] = []
        for offset, entry in enumerate(captured):
            artifact_id = f"art_{first_index + offset}"
            urls: dict[str, str] = {}
            for extension, payload in sorted(entry.blobs.items()):
                blob = self.store.put(trace_id, artifact_id, extension, payload)
                urls[extension] = blob.url

            template = entry.template
            primary = _primary_extension(template)
            ref = template.model_copy(
                update={
                    "id": artifact_id,
                    "url": urls.get(primary) if primary else None,
                    "geotiff_url": urls.get("tif") if template.geotiff_url else None,
                    "geojson_url": urls.get("geojson") if template.geojson_url else None,
                    "produced_by_step": step,
                }
            )
            refs.append(ref)
            if entry.payload is not None:
                report.data[artifact_id] = entry.payload
        return refs

    def _blobs_of(self, ref: ArtifactRef, trace_id: str) -> dict[str, bytes]:
        """Read back the encoded bytes of everything an artifact just wrote."""
        blobs: dict[str, bytes] = {}
        for url in (ref.url, ref.geotiff_url, ref.geojson_url):
            if not url:
                continue
            extension = url.rsplit(".", 1)[-1]
            try:
                stored = self.store.resolve(trace_id, ref.id, extension)
            except LookupError:  # pragma: no cover - the write just succeeded
                continue
            blobs[extension] = stored.path.read_bytes()
        return blobs

    def _store_draft(
        self,
        draft: ArtifactDraft,
        trace_id: str,
        artifact_id: str,
        step: int,
    ) -> ArtifactRef:
        """Write one draft's payloads and describe the result."""
        store = self.store
        url: str | None = None
        geotiff_url: str | None = None
        geojson_url: str | None = None
        mime = _MIME_FOR_TYPE.get(draft.type, "image/png")
        width: int | None = None
        height: int | None = None

        if draft.image is not None:
            blob = store.put_image(trace_id, artifact_id, draft.image, draft.image_format)
            url, mime = blob.url, blob.mime
            height, width = int(draft.image.shape[0]), int(draft.image.shape[1])

        geometry = draft.geometry
        if draft.raster is not None and geometry is not None and geometry.transform is not None:
            companion = store.put_geotiff(
                trace_id,
                artifact_id,
                draft.raster,
                geometry.transform,
                geometry.crs,
                nodata=None,
            )
            geotiff_url = companion.url

        if draft.geojson is not None:
            vector = store.put_geojson(trace_id, artifact_id, draft.geojson)
            geojson_url = vector.url
            if url is None:
                url, mime = vector.url, vector.mime

        geo = (
            ArtifactGeo(crs=geometry.crs, transform=geometry.transform_list)
            if geometry is not None and geometry.crs and geometry.transform_list
            else None
        )
        return ArtifactRef(
            id=artifact_id,
            type=draft.type,
            mime=mime,
            label=draft.label,
            url=url,
            geotiff_url=geotiff_url,
            geojson_url=geojson_url,
            geo=geo,
            width=width,
            height=height,
            stats=draft.stats,
            inline=draft.inline,
            produced_by_step=step,
        )


def _primary_extension(ref: ArtifactRef) -> str | None:
    """The extension of the artifact's own ``url``, as opposed to a companion's."""
    return ref.url.rsplit(".", 1)[-1] if ref.url else None


__all__ = [
    "DEFAULT_CACHE",
    "CachedStep",
    "CapturedArtifact",
    "MAX_PARALLEL_GPU_TOOLS",
    "MAX_PARALLEL_TOOLS",
    "DagExecutor",
    "ExecutionCache",
    "ExecutionReport",
    "ToolFailedNoFallbackError",
    "cache_key",
    "canonical_json",
    "grid_hash",
]
