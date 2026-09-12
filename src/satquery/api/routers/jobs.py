"""``/v1/jobs`` — asynchronous analysis with a live SSE progress stream.

Same multipart body as ``/v1/analyze`` (API_CONTRACT §4.2); the difference is
that the response returns as soon as the work is queued and the client watches
it happen. Bi-temporal change detection plus VLM synthesis pushes well past a
comfortable synchronous wait, and the execution trace filling in step by step is
the thing the frontend is built to show (§8.3, §8.4).

Two behaviours here are contract obligations rather than choices:

* **A tool failure ends in ``done``, never ``error``.** §4.1 guarantees a tool
  failure produces a complete response with a ``DEGRADED``/``FAILED`` execution
  and a reduced confidence. ``error`` is reserved for a job that produced no
  answer at all — unusable inputs, or an unrecoverable non-optional step.
* **The uploaded files outlive the request.** They are spooled with
  :func:`~satquery.api.uploads.persist_uploads` and removed by the background
  task, not by the handler that returned 202.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse

from satquery.agent.concurrency import DeviceGates
from satquery.agent.events import STAGE_PCT
from satquery.agent.pipeline import AnalysisRequest
from satquery.agent.pipeline import analyze as run_analysis
from satquery.api.dependencies import get_artifact_store, get_device_gates, get_trace_store
from satquery.api.jobs import Job, JobEvent, JobStore, TooManyJobsError, get_job_store
from satquery.api.uploads import parse_options, persist_uploads_async
from satquery.core.config import Settings, get_settings
from satquery.core.logging import get_logger
from satquery.evidence.citation_validator import CitationPolicy
from satquery.ingest.errors import IngestError
from satquery.ingest.pipeline import ingest
from satquery.render.artifact_store import ArtifactStore
from satquery.schemas.api import (
    AnalyzeResponse,
    ApiError,
    ApiErrorResponse,
    JobAccepted,
    JobStatusResponse,
)
from satquery.trace.builder import new_trace_id
from satquery.trace.store import TraceStore

router = APIRouter(tags=["jobs"])
log = get_logger(__name__)

HEARTBEAT_SECONDS: Final[float] = 15.0
"""``: ping`` cadence (§5). Keeps proxies from reaping an idle GPU step."""

QUERY_FORM = Form(min_length=1, max_length=1000, description="1-1000 chars, UTF-8.")
"""The contract's stated bounds, enforced (they used to be description only)."""

CANCELLED_ERROR: Final[ApiError] = ApiError(
    code="JOB_CANCELLED",
    http_status=499,
    message="The run was cancelled by the client.",
    hint="Steps that had not started were skipped; the trace records what did run.",
)
"""The terminal event a cancelled job publishes. 499 is nginx's 'client closed
request' — the closest standard reading of 'the caller asked us to stop'."""

_RUNNING: Final[set[asyncio.Task[None]]] = set()
"""Strong references to in-flight jobs, so the loop cannot garbage-collect them."""


# ------------------------------------------------------------------ the worker


@dataclass(frozen=True)
class _JobSpec:
    """Everything the worker needs that the request had and the job does not."""

    directory: Path
    request: AnalysisRequest
    include_trace: bool
    include_rendered_views: bool


async def _execute(
    job: Job,
    jobs: JobStore,
    spec: _JobSpec,
    store: ArtifactStore,
    traces: TraceStore,
    gates: DeviceGates,
) -> None:
    """Run one analysis to completion, narrating it onto the job's event stream.

    Every way out publishes a terminal event. ``CancelledError`` in particular:
    it is not an ``Exception``, so the old handler let a cancelled task end with
    no ``done``/``error`` at all and every subscriber waited for ever.
    """
    try:
        analysis = await run_analysis(
            spec.request,
            store=store,
            traces=traces,
            emit=lambda event, data: jobs.publish(job, event, data),
            gates=gates,
        )
        trace = analysis.trace
        response = AnalyzeResponse(
            trace_id=trace.trace_id,
            answer=trace.answer,
            artifacts=trace.artifacts if spec.include_rendered_views else [],
            confidence=trace.confidence,
            compatibility=trace.compatibility,
            resolved_task=trace.resolved_task,
            trace=trace if spec.include_trace else None,
        )
        job.result = response
        jobs.publish(job, "done", response.model_dump(mode="json"))

    except asyncio.CancelledError:
        api_error = CANCELLED_ERROR.model_copy(update={"trace_id": job.job_id})
        job.error = api_error
        jobs.publish(job, "error", api_error.model_dump(mode="json"))
        raise
    except IngestError as error:
        api_error = error.to_api_error(trace_id=job.job_id)
        job.error = api_error
        jobs.publish(job, "error", api_error.model_dump(mode="json"))
    except Exception as error:  # noqa: BLE001 - a crash must reach the client as an event
        log.exception("jobs.failed", job_id=job.job_id)
        api_error = ApiError(
            code="INTERNAL_ERROR",
            http_status=500,
            message="The analysis failed unexpectedly.",
            hint="Retry the request; if it persists, check the server logs.",
            trace_id=job.job_id,
        )
        job.error = api_error
        jobs.publish(job, "error", api_error.model_dump(mode="json"))
        log.debug("jobs.failure_detail", job_id=job.job_id, detail=str(error))
    finally:
        shutil.rmtree(spec.directory, ignore_errors=True)


# --------------------------------------------------------------------- routes


@router.post(
    "/jobs",
    response_model=JobAccepted,
    status_code=202,
    summary="Queue an analysis and watch it stream",
)
async def create_job(
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[ArtifactStore, Depends(get_artifact_store)],
    traces: Annotated[TraceStore, Depends(get_trace_store)],
    jobs: Annotated[JobStore, Depends(get_job_store)],
    gates: Annotated[DeviceGates, Depends(get_device_gates)],
    images: list[UploadFile] = File(description="1-2 files. GeoTIFF preferred; PNG/JPEG accepted."),
    query: str = QUERY_FORM,
    options: str | None = Form(default=None, description="JSON, see AnalyzeOptions."),
) -> JobAccepted:
    """Accept the same body as ``/v1/analyze`` and return immediately.

    The returned ``job_id`` *is* the eventual ``trace_id``, so the client can
    address the trace before the pipeline has produced it.
    """
    parsed = parse_options(options)

    # Spooling and ingestion both happen here, inside the request, and for the
    # same reason: an input the pipeline cannot use must be reported the way
    # /v1/analyze reports it — a 400/413/415/422 with the §6 error envelope —
    # rather than as an error event on a job the client has not subscribed to
    # yet. Ingestion is the GPU-free pre-flight /v1/validate already runs, so
    # this costs the caller milliseconds and buys exact parity between the two
    # submission paths. Everything after this point can only degrade, not reject.
    directory, sources = await persist_uploads_async(
        images, max_bytes=settings.max_upload_mb << 20
    )
    try:
        # Off the loop: rasterio plus phase correlation over a large pair used to
        # hold every other job's SSE heartbeat for its whole duration.
        ingested = await asyncio.to_thread(
            ingest,
            sources,
            pair_type_hint=parsed.pair_type,
            roles=parsed.roles,
            max_images=settings.max_images,
        )
        job_id = new_trace_id()
        job = jobs.create(job_id)
    except TooManyJobsError as error:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=429, detail=str(error)) from error
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    jobs.publish(job, "queued", {"job_id": job_id})
    # Narrated after the fact, but buffered, so a subscriber that arrives later
    # still replays the full stage sequence rather than opening mid-run.
    jobs.publish(job, "stage", {"stage": "ingesting", "pct": STAGE_PCT["ingesting"]})
    jobs.publish(job, "stage", {"stage": "validating", "pct": STAGE_PCT["validating"]})

    task = asyncio.create_task(
        _execute(
            job=job,
            jobs=jobs,
            spec=_JobSpec(
                directory=directory,
                request=AnalysisRequest(
                    query=query.strip(),
                    sources=sources,
                    ingest=ingested,
                    trace_id=job_id,
                    seed=parsed.seed,
                    citation_policy=CitationPolicy(settings.citation_policy),
                    allow_generic_fallback=settings.allow_generic_fallback,
                    enable_tools=parsed.enable_tools,
                    disable_tools=parsed.disable_tools,
                    cancelled=job.cancel_requested,
                    max_latency_ms=parsed.max_latency_ms,
                    ignored_options=parsed.ignored(),
                ),
                include_trace=parsed.include_trace,
                include_rendered_views=parsed.include_rendered_views,
            ),
            store=store,
            traces=traces,
            gates=gates,
        )
    )
    job.task = task
    _RUNNING.add(task)
    task.add_done_callback(_RUNNING.discard)

    return JobAccepted(
        job_id=job_id,
        status="queued",
        poll_url=f"{settings.api_prefix}/jobs/{job_id}",
        events_url=f"{settings.api_prefix}/jobs/{job_id}/events",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll one job's progress",
)
async def get_job(
    job_id: str,
    jobs: Annotated[JobStore, Depends(get_job_store)],
    traces: Annotated[TraceStore, Depends(get_trace_store)],
) -> JobStatusResponse:
    """Return the job's current stage, progress and — once finished — its result.

    A job the store no longer holds — evicted, or from before a restart — is
    answered from its persisted trace when one exists: the trace store is the
    durable half by design, and a client reconnecting after a server restart
    should get its answer rather than a 404 it cannot tell from "never existed".
    """
    job = jobs.get(job_id)
    if job is not None:
        return JobStatusResponse.model_validate(job.snapshot())
    trace = await asyncio.to_thread(traces.get, job_id)
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=f"No job {job_id}. It may predate a server restart; check /v1/traces/{job_id}.",
        )
    result = AnalyzeResponse(
        trace_id=trace.trace_id,
        answer=trace.answer,
        artifacts=trace.artifacts,
        confidence=trace.confidence,
        compatibility=trace.compatibility,
        resolved_task=trace.resolved_task,
        trace=trace,
    )
    return JobStatusResponse(
        job_id=job_id,
        status="succeeded",
        stage="done",
        step=len(trace.executions),
        total_steps=len(trace.executions),
        pct=100,
        created_at=trace.created_at,
        updated_at=trace.created_at,
        result=result,
        error=None,
    )


@router.delete(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    status_code=202,
    summary="Cancel a running job",
    responses={
        404: {"description": "Unknown or evicted job."},
        409: {"description": "The job has already finished.", "model": ApiErrorResponse},
    },
)
async def cancel_job(
    job_id: str,
    jobs: Annotated[JobStore, Depends(get_job_store)],
) -> JobStatusResponse:
    """Ask a running job to stop. Best-effort, and honest about what that means.

    The tool step in flight finishes — a worker thread cannot be killed — and no
    further step starts. The job then ends with an ``error`` event carrying
    ``JOB_CANCELLED``, so an open events stream closes the way it would for any
    other failure. ``202`` because the stop is requested, not yet done.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job {job_id}.")
    if job.finished:
        body = ApiErrorResponse(
            error=ApiError(
                code="JOB_ALREADY_FINISHED",
                http_status=409,
                message=f"Job {job_id} has already finished as {job.status}.",
                hint="There is nothing to cancel; fetch the result instead.",
                trace_id=job_id,
            )
        )
        raise HTTPException(status_code=409, detail=body.model_dump(mode="json"))
    job.cancel_requested.set()
    if job.task is not None:
        job.task.cancel()
    return JobStatusResponse.model_validate(job.snapshot())


def _sse(message: JobEvent) -> str:
    """Format one event as an SSE message frame, with its sequence number as ``id``."""
    payload = json.dumps(message.data, separators=(",", ":"), default=str)
    return f"id: {message.seq}\nevent: {message.event}\ndata: {payload}\n\n"


def _last_seen(header: str | None, query: str | None) -> int | None:
    """The last event id the client holds, from ``Last-Event-ID`` or ``?after=``.

    A value that is not a non-negative integer is ignored rather than refused: a
    confused client gets a full replay, which is always safe.
    """
    raw = header if header is not None else query
    if raw is None or not raw.strip().isdigit():
        return None
    return int(raw.strip())


async def _stream(
    jobs: JobStore, job: Job, http_request: Request, after: int | None = None
) -> AsyncIterator[str]:
    """Yield SSE frames for *job*, with a heartbeat while nothing is happening."""
    events = jobs.subscribe(job, after=after)
    pending: asyncio.Task[JobEvent] | None = None
    try:
        while True:
            if await http_request.is_disconnected():
                return
            if pending is None:
                pending = asyncio.ensure_future(anext(events))
            finished, _ = await asyncio.wait({pending}, timeout=HEARTBEAT_SECONDS)
            if not finished:
                yield ": ping\n\n"
                continue
            task, pending = pending, None
            try:
                message = task.result()
            except StopAsyncIteration:
                return
            yield _sse(message)
    finally:
        if pending is not None:
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending
        await events.aclose()


@router.get(
    "/jobs/{job_id}/events",
    summary="Server-sent progress events for one job",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": (
                "SSE stream. Event order is guaranteed: queued -> stage -> plan -> "
                "interleaved step_started/step_completed/artifact -> answer_delta* -> "
                "done|error. `done` and `error` are terminal and mutually exclusive."
            ),
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
        404: {"description": "Unknown or evicted job."},
    },
)
async def job_events(
    job_id: str,
    http_request: Request,
    jobs: Annotated[JobStore, Depends(get_job_store)],
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    after: Annotated[
        str | None,
        Query(description="Resume after this event id; the query form of Last-Event-ID."),
    ] = None,
) -> StreamingResponse:
    """Stream the job's events, replaying everything that already happened.

    A subscriber that connects late — the normal case, since the client needs the
    ``job_id`` from the 202 first — receives the buffered history before the live
    tail, so the DAG can always be drawn from the ``plan`` event. One that
    *reconnects* sends the last ``id:`` it saw and receives only the rest.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job {job_id}.")
    headers: dict[str, str] = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _stream(jobs, job, http_request, after=_last_seen(last_event_id, after)),
        media_type="text/event-stream",
        headers=headers,
    )


__all__: list[str] = ["router"]
