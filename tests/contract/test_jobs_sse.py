"""Contract tests for ``/v1/jobs`` and its SSE stream (API_CONTRACT §4.2-4.4, §5).

The ordering invariant of §5 is the load-bearing promise here. The frontend
draws the DAG from the ``plan`` event and fills it in from ``step_*`` events, so
a ``plan`` that arrived after the first ``step_started``, or a ``done`` followed
by anything at all, would break the UI in a way no schema check would catch.

These post real rasters from the synthetic corpus, because the jobs path runs
the same Phase 1 ingestion and Phase 6 executor that ``/v1/analyze`` does.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from satquery.api.app import app
from satquery.api.dependencies import get_artifact_store, get_trace_store
from satquery.api.jobs import JobStore, get_job_store
from satquery.core.config import get_settings
from satquery.registry.registry import default_registry
from satquery.render.artifact_store import ArtifactStore
from satquery.schemas.api import (
    AnalyzeResponse,
    JobAccepted,
    JobStatusResponse,
)
from satquery.schemas.enums import ToolStatus
from satquery.schemas.tool import PlanStep
from satquery.schemas.trace import ArtifactRef, AuditTrace
from satquery.trace.store import TraceStore

TERMINAL = {"done", "error"}


# ------------------------------------------------------------------- fixtures


@pytest.fixture
def deterministic_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin these tests to the deterministic-only baseline.

    Same reasoning as the analyze contract tests: whether the VLM step runs is
    probed from the machine, and a box with the Qwen weights in its cache would
    otherwise turn these assertions into assertions about an 8B generation.
    """
    monkeypatch.setenv("SATQUERY_VLM_DISABLED", "true")
    get_settings.cache_clear()
    default_registry.cache_clear()
    yield
    get_settings.cache_clear()
    default_registry.cache_clear()


@pytest.fixture
def jobs_client(tmp_path: Path, deterministic_only: None) -> Iterator[TestClient]:
    """A client whose artifact, trace and job stores are scoped to one test."""
    artifacts = ArtifactStore(tmp_path / "artifacts")
    traces = TraceStore(tmp_path / "traces.sqlite3")
    store = JobStore()
    app.dependency_overrides[get_artifact_store] = lambda: artifacts
    app.dependency_overrides[get_trace_store] = lambda: traces
    app.dependency_overrides[get_job_store] = lambda: store
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_artifact_store, None)
    app.dependency_overrides.pop(get_trace_store, None)
    app.dependency_overrides.pop(get_job_store, None)


def read_events(client: TestClient, job_id: str) -> list[tuple[str, dict]]:
    """Drain the SSE stream into ``(event, data)`` pairs, dropping heartbeats."""
    events: list[tuple[str, dict]] = []
    with client.stream("GET", f"/v1/jobs/{job_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        name: str | None = None
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith(":"):  # heartbeat comment
                continue
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                assert name is not None, "a data line arrived before its event line"
                events.append((name, json.loads(line.removeprefix("data: "))))
                name = None
    return events


def submit(client: TestClient, upload_files, *names: str, query: str) -> JobAccepted:
    """Queue one job and return the parsed 202 body."""
    response = client.post("/v1/jobs", files=upload_files(*names), data={"query": query})
    assert response.status_code == 202, response.text
    return JobAccepted.model_validate(response.json())


# --------------------------------------------------------------------- POST


def test_post_jobs_returns_202_with_addressable_urls(
    jobs_client: TestClient, upload_files
) -> None:
    accepted = submit(jobs_client, upload_files, "s2_pre", query="Describe this scene.")
    assert accepted.status == "queued"
    assert accepted.poll_url == f"/v1/jobs/{accepted.job_id}"
    assert accepted.events_url == f"/v1/jobs/{accepted.job_id}/events"


def test_job_id_is_the_trace_id(jobs_client: TestClient, upload_files) -> None:
    """§4.2 says the job_id **is** the eventual trace_id."""
    accepted = submit(jobs_client, upload_files, "s2_pre", query="Describe this scene.")
    events = dict(read_events(jobs_client, accepted.job_id))

    assert "done" in events, "the run must have produced a terminal done event"
    result = AnalyzeResponse.model_validate(events["done"])
    assert result.trace_id == accepted.job_id

    # And the trace is addressable under that same id (§4.7).
    stored = jobs_client.get(f"/v1/traces/{accepted.job_id}")
    assert stored.status_code == 200
    assert AuditTrace.model_validate(stored.json()).trace_id == accepted.job_id


def test_unusable_inputs_are_rejected_with_a_status_not_a_phantom_job(
    jobs_client: TestClient, upload_files
) -> None:
    """Ingestion runs in-request, so /v1/jobs rejects exactly as /v1/analyze does.

    The alternative — 202 followed by an ``error`` event — would force the client
    to open an SSE stream just to discover its upload was never viable, and would
    leave a failed job in the store for every malformed request.
    """
    files = upload_files("s2_pre", "s2_post", "s2_wgs84")
    payload = {"query": "How much built-up area appeared?"}

    queued = jobs_client.post("/v1/jobs", files=files, data=payload)
    synchronous = jobs_client.post("/v1/analyze", files=files, data=payload)

    assert queued.status_code == synchronous.status_code == 400
    assert queued.json()["error"]["code"] == "TOO_MANY_IMAGES"
    assert queued.json()["error"]["code"] == synchronous.json()["error"]["code"]


def test_a_rejected_submission_creates_no_job(jobs_client: TestClient, upload_files) -> None:
    store = app.dependency_overrides[get_job_store]()
    jobs_client.post(
        "/v1/jobs",
        files=upload_files("s2_pre", "s2_post", "s2_wgs84"),
        data={"query": "How much built-up area appeared?"},
    )
    assert len(store) == 0


# ---------------------------------------------------------------- SSE ordering


def test_sse_event_order_matches_the_contract(jobs_client: TestClient, upload_files) -> None:
    """The §5 order: queued, stage*, plan, step_*/artifact*, answer_delta*, done|error."""
    accepted = submit(
        jobs_client, upload_files, "s2_pre", "s2_post", query="How much built-up area appeared?"
    )
    events = read_events(jobs_client, accepted.job_id)
    names = [name for name, _ in events]

    assert names[0] == "queued", "the stream must open with queued"
    assert names[-1] in TERMINAL, "the stream must close on a terminal event"
    assert sum(name in TERMINAL for name in names) == 1, "done and error are mutually exclusive"

    plan_at = names.index("plan")
    # Everything before the plan is stage narration; nothing may execute first.
    assert set(names[1:plan_at]) <= {"stage"}
    for streaming in ("step_started", "step_completed", "artifact"):
        if streaming in names:
            assert names.index(streaming) > plan_at, f"{streaming} preceded the plan"

    if "answer_delta" in names:
        last_step = max(i for i, n in enumerate(names) if n.startswith("step_"))
        assert names.index("answer_delta") > last_step


def test_plan_event_is_emitted_once_and_is_schema_valid(
    jobs_client: TestClient, upload_files
) -> None:
    """§5: 'emitted once, lets the UI draw the DAG before execution'."""
    accepted = submit(
        jobs_client, upload_files, "s2_pre", "s2_post", query="How much built-up area appeared?"
    )
    events = read_events(jobs_client, accepted.job_id)

    plans = [data for name, data in events if name == "plan"]
    assert len(plans) == 1

    steps = [PlanStep.model_validate(step) for step in plans[0]["steps"]]
    assert steps, "the plan must not be empty"
    assert [step.step for step in steps] == list(range(1, len(steps) + 1))
    for step in steps:
        assert all(dependency < step.step for dependency in step.depends_on)
        assert step.reason, "every step must carry the policy key that selected it"


def test_every_planned_step_starts_and_completes_exactly_once(
    jobs_client: TestClient, upload_files
) -> None:
    """A node left PENDING for ever reads as a hang; the UI needs both halves."""
    accepted = submit(
        jobs_client, upload_files, "s2_pre", "s2_post", query="How much built-up area appeared?"
    )
    events = read_events(jobs_client, accepted.job_id)
    by_name = {name: [data for n, data in events if n == name] for name in {n for n, _ in events}}

    planned = {step["step"] for step in by_name["plan"][0]["steps"]}
    started = [data["step"] for data in by_name.get("step_started", [])]
    completed = [data["step"] for data in by_name.get("step_completed", [])]

    assert sorted(started) == sorted(planned)
    assert sorted(completed) == sorted(planned)
    assert len(set(started)) == len(started), "a step started twice"
    assert len(set(completed)) == len(completed), "a step completed twice"

    for data in by_name.get("step_completed", []):
        assert data["status"] in {status.value for status in ToolStatus}
        assert data["duration_ms"] >= 0
        assert 0.0 <= data["confidence"] <= 1.0


def test_artifact_events_stream_and_match_the_final_response(
    jobs_client: TestClient, upload_files
) -> None:
    """§5: artifacts are announced as they are written, so evidence streams in."""
    accepted = submit(jobs_client, upload_files, "s2_pre", query="Describe this scene.")
    events = read_events(jobs_client, accepted.job_id)

    streamed = [ArtifactRef.model_validate(data) for name, data in events if name == "artifact"]
    assert streamed, "the renderer must have emitted evidence"

    final = AnalyzeResponse.model_validate(dict(events)["done"])
    assert [ref.id for ref in streamed] == [ref.id for ref in final.artifacts]


def test_a_degraded_run_still_terminates_in_done(jobs_client: TestClient, upload_files) -> None:
    """§4.1: a tool failure is never an error — it is a complete, degraded answer."""
    options = json.dumps({"disable_tools": ["spectral_index_analyzer"]})
    response = jobs_client.post(
        "/v1/jobs",
        files=upload_files("s2_pre"),
        data={"query": "Describe this scene and identify water bodies.", "options": options},
    )
    assert response.status_code == 202
    accepted = JobAccepted.model_validate(response.json())

    events = read_events(jobs_client, accepted.job_id)
    names = [name for name, _ in events]
    assert "error" not in names
    assert names[-1] == "done"

    result = AnalyzeResponse.model_validate(events[-1][1])
    assert result.trace is not None
    statuses = {execution.status for execution in result.trace.executions}
    assert statuses & {ToolStatus.DEGRADED, ToolStatus.SKIPPED, ToolStatus.FAILED}, (
        "disabling a tool should have visibly degraded the run"
    )
    assert result.answer.text, "a degraded run must still produce an answer"


def test_a_late_subscriber_receives_the_whole_run(jobs_client: TestClient, upload_files) -> None:
    """The normal case: the client needs the job_id from the 202 before it can subscribe."""
    accepted = submit(jobs_client, upload_files, "s2_pre", query="Describe this scene.")

    # Drain once to completion, then subscribe again from scratch.
    first = read_events(jobs_client, accepted.job_id)
    second = read_events(jobs_client, accepted.job_id)

    assert [name for name, _ in first] == [name for name, _ in second]
    assert second[0][0] == "queued"
    assert second[-1][0] == "done"


# ---------------------------------------------------------------------- poll


def test_poll_reports_terminal_state_and_carries_the_result(
    jobs_client: TestClient, upload_files
) -> None:
    accepted = submit(jobs_client, upload_files, "s2_pre", query="Describe this scene.")
    read_events(jobs_client, accepted.job_id)  # run it to completion

    response = jobs_client.get(accepted.poll_url)
    assert response.status_code == 200

    status = JobStatusResponse.model_validate(response.json())
    assert status.job_id == accepted.job_id
    assert status.status == "succeeded"
    assert status.stage == "done"
    assert status.pct == 100
    assert status.total_steps > 0
    assert status.step == status.total_steps
    assert status.error is None
    assert status.result is not None
    assert status.result.trace_id == accepted.job_id


def test_unknown_job_is_404_on_both_poll_and_events(jobs_client: TestClient) -> None:
    missing = "0" * 32
    assert jobs_client.get(f"/v1/jobs/{missing}").status_code == 404
    assert jobs_client.get(f"/v1/jobs/{missing}/events").status_code == 404


def test_unknown_trace_is_404(jobs_client: TestClient) -> None:
    assert jobs_client.get(f"/v1/traces/{'0' * 32}").status_code == 404


# ------------------------------------------------ cancellation (DELETE /v1/jobs)


class _SlowTool:
    """Wraps a real tool and holds it for a while, so a job is cancellable mid-run."""

    def __init__(self, inner: object, seconds: float) -> None:
        self.inner = inner
        self.seconds = seconds
        self.name = getattr(inner, "name", "slow")

    def run(self, ctx: object, params: dict) -> object:
        import time

        time.sleep(self.seconds)
        return self.inner.run(ctx, params)  # type: ignore[attr-defined]


@pytest.fixture
def slow_change_statistics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``change_statistics`` take a second, so DELETE can land during the run."""
    from satquery.agent import pipeline
    from satquery.agent.executor import DEFAULT_CACHE
    from satquery.tools.catalog import BUILTIN_TOOLS

    # A cache hit would skip the tool entirely and the job would finish before
    # the DELETE lands; earlier tests in this process have already run this scene.
    DEFAULT_CACHE.clear()
    slowed = {"change_statistics": _SlowTool(BUILTIN_TOOLS["change_statistics"], 1.0)}

    def implementations(overrides=None):  # type: ignore[no-untyped-def]
        return {**BUILTIN_TOOLS, **slowed, **(overrides or {})}

    monkeypatch.setattr(pipeline, "implementations", implementations)


def test_delete_cancels_a_running_job_and_terminates_the_stream(
    jobs_client: TestClient, upload_files, slow_change_statistics: None
) -> None:
    """Cancel means cancel: the DAG stops at the next boundary and the stream ends."""
    accepted = submit(
        jobs_client, upload_files, "s2_pre", "s2_post", query="what changed between these"
    )
    # Let the renderer finish and the slow step start.
    import time

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = JobStatusResponse.model_validate(
            jobs_client.get(f"/v1/jobs/{accepted.job_id}").json()
        )
        if status.stage == "executing" and status.step >= 2:
            break
        time.sleep(0.05)

    response = jobs_client.delete(f"/v1/jobs/{accepted.job_id}")
    assert response.status_code == 202, response.text
    assert JobStatusResponse.model_validate(response.json()).job_id == accepted.job_id

    events = read_events(jobs_client, accepted.job_id)
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "JOB_CANCELLED"
    final = JobStatusResponse.model_validate(jobs_client.get(f"/v1/jobs/{accepted.job_id}").json())
    assert final.status == "failed"
    assert final.error is not None and final.error.code == "JOB_CANCELLED"


def test_delete_on_a_finished_job_is_409(jobs_client: TestClient, upload_files) -> None:
    accepted = submit(jobs_client, upload_files, "s2_pre", query="describe this scene")
    read_events(jobs_client, accepted.job_id)  # drains to the terminal event
    response = jobs_client.delete(f"/v1/jobs/{accepted.job_id}")
    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "JOB_ALREADY_FINISHED"


def test_delete_unknown_job_is_404(jobs_client: TestClient) -> None:
    assert jobs_client.delete(f"/v1/jobs/{'0' * 32}").status_code == 404


# ---------------------------------------------------------- resume by event id


def test_last_event_id_resumes_without_replaying_delivered_events(
    jobs_client: TestClient, upload_files
) -> None:
    """Every frame carries ``id:``; a reconnect with the last one gets only the rest."""
    accepted = submit(jobs_client, upload_files, "s2_pre", query="describe this scene")
    ids: list[int] = []
    with jobs_client.stream("GET", f"/v1/jobs/{accepted.job_id}/events") as response:
        for line in response.iter_lines():
            if line.startswith("id: "):
                ids.append(int(line.removeprefix("id: ")))
    assert ids == list(range(len(ids))), "ids are the dense sequence of the job's history"

    cut = len(ids) // 2
    with jobs_client.stream(
        "GET",
        f"/v1/jobs/{accepted.job_id}/events",
        headers={"Last-Event-ID": str(ids[cut])},
    ) as response:
        resumed = [
            int(line.removeprefix("id: "))
            for line in response.iter_lines()
            if line.startswith("id: ")
        ]
    assert resumed == ids[cut + 1 :]
    # The query form behaves the same, for clients that cannot set the header.
    with jobs_client.stream(
        "GET", f"/v1/jobs/{accepted.job_id}/events", params={"after": str(ids[-2])}
    ) as response:
        tail = [
            int(line.removeprefix("id: "))
            for line in response.iter_lines()
            if line.startswith("id: ")
        ]
    assert tail == [ids[-1]]


# ---------------------------------------------------------- store lifecycle


def test_a_running_job_is_never_evicted() -> None:
    from satquery.api.jobs import TooManyJobsError

    store = JobStore(max_jobs=2)
    first = store.create("a" * 32)
    store.create("b" * 32)
    store.publish(first, "stage", {"stage": "executing", "pct": 50})
    store.publish(store.get("b" * 32), "stage", {"stage": "executing", "pct": 50})  # type: ignore[arg-type]
    with pytest.raises(TooManyJobsError):
        store.create("c" * 32)
    # Once one finishes it is the one that goes.
    store.publish(first, "error", {"code": "X", "http_status": 500, "message": "m"})
    store.create("c" * 32)
    assert store.get("a" * 32) is None
    assert store.get("b" * 32) is not None
    assert store.running == 2, "b is executing and c is queued; neither is terminal"


def test_a_finished_job_is_recoverable_from_its_trace_after_eviction(
    jobs_client: TestClient, upload_files
) -> None:
    """The trace store is the durable half; the poll endpoint reads it after a restart."""
    accepted = submit(jobs_client, upload_files, "s2_pre", query="describe this scene")
    events = read_events(jobs_client, accepted.job_id)
    assert events[-1][0] == "done"

    # Simulate a restart: the in-memory store forgets the job.
    store = app.dependency_overrides[get_job_store]()
    store._jobs.clear()  # noqa: SLF001 - the point is to forget

    response = jobs_client.get(f"/v1/jobs/{accepted.job_id}")
    assert response.status_code == 200, response.text
    status = JobStatusResponse.model_validate(response.json())
    assert status.status == "succeeded"
    assert status.result is not None
    assert status.result.trace_id == accepted.job_id


def test_shutdown_cancels_running_jobs_and_removes_their_uploads(
    jobs_client: TestClient, upload_files, slow_change_statistics: None
) -> None:
    import tempfile
    import time
    from pathlib import Path

    accepted = submit(
        jobs_client, upload_files, "s2_pre", "s2_post", query="what changed between these"
    )
    store = app.dependency_overrides[get_job_store]()
    time.sleep(0.3)
    assert store.running == 1
    uploads_before = {p for p in Path(tempfile.gettempdir()).glob("satquery-upload-*")}
    job = store.get(accepted.job_id)
    assert job is not None and job.task is not None

    async def stop() -> None:
        await store.shutdown()

    jobs_client.portal.call(stop)  # type: ignore[attr-defined]
    assert store.running == 0
    assert job.error is not None and job.error.code == "JOB_CANCELLED"
    uploads_after = {p for p in Path(tempfile.gettempdir()).glob("satquery-upload-*")}
    assert not (uploads_before - uploads_after) or uploads_before >= uploads_after


# ------------------------------------------------------------ request hygiene


def test_malformed_options_are_a_400_not_a_silent_default(
    jobs_client: TestClient, upload_files
) -> None:
    for bad in ("{not json", '{"seed": "abc"}', '{"unknown_field": 1}'):
        response = jobs_client.post(
            "/v1/jobs",
            files=upload_files("s2_pre"),
            data={"query": "describe this scene", "options": bad},
        )
        assert response.status_code == 400, (bad, response.text)
        assert response.json()["error"]["code"] == "INVALID_OPTIONS"


def test_an_invalid_pair_type_hint_is_a_400_with_the_error_envelope(
    jobs_client: TestClient, upload_files
) -> None:
    response = jobs_client.post(
        "/v1/validate",
        files=upload_files("s2_pre"),
        data={"options": '{"pair_type_hint": "SIDEWAYS"}'},
    )
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "INVALID_OPTIONS"
    assert "pair_type_hint" in body["hint"]


def test_an_empty_query_is_rejected_before_ingestion(jobs_client: TestClient, upload_files) -> None:
    for path in ("/v1/jobs", "/v1/analyze"):
        response = jobs_client.post(path, files=upload_files("s2_pre"), data={"query": ""})
        assert response.status_code == 422, (path, response.text)
        response = jobs_client.post(path, files=upload_files("s2_pre"), data={"query": "x" * 1001})
        assert response.status_code == 422, (path, response.text)


def test_ignored_options_are_reported_not_swallowed(jobs_client: TestClient, upload_files) -> None:
    response = jobs_client.post(
        "/v1/analyze",
        files=upload_files("s2_pre"),
        data={"query": "describe this scene", "options": '{"vlm_backend": "llamacpp"}'},
    )
    assert response.status_code == 200, response.text
    trace = AnalyzeResponse.model_validate(response.json()).trace
    assert trace is not None
    assert any(w.code == "OPTION_IGNORED" and "vlm_backend" in w.message for w in trace.warnings)


def test_health_reports_job_and_gate_counters(jobs_client: TestClient) -> None:
    body = jobs_client.get("/v1/health").json()
    assert body["jobs_running"] == 0
    assert body["jobs_retained"] == 0
    assert body["leaked_gpu_permits"] == 0


# --------------------------------------------------------- the event loop


def test_ingestion_does_not_stall_other_requests(
    jobs_client: TestClient, upload_files, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow ingest on one request must not hold the loop for every other one.

    ``ingest`` is rasterio plus phase correlation; run inline in the handler it
    stalled every SSE heartbeat in the process. Here it is made to take half a
    second and ``/v1/health`` is timed from another thread meanwhile.
    """
    import threading
    import time

    from satquery.api.routers import jobs as jobs_router

    real_ingest = jobs_router.ingest

    def slow_ingest(*args, **kwargs):  # type: ignore[no-untyped-def]
        time.sleep(0.6)
        return real_ingest(*args, **kwargs)

    monkeypatch.setattr(jobs_router, "ingest", slow_ingest)
    # The first health call in a process pays torch's ROCm initialisation; that
    # is not what is being measured here.
    assert jobs_client.get("/v1/health").status_code == 200

    def post() -> None:
        submit(jobs_client, upload_files, "s2_pre", query="describe this scene")

    worker = threading.Thread(target=post)
    worker.start()
    time.sleep(0.15)  # the POST is now inside the slow ingest
    started = time.perf_counter()
    assert jobs_client.get("/v1/health").status_code == 200
    elapsed = time.perf_counter() - started
    worker.join()
    assert elapsed < 0.4, f"/v1/health waited {elapsed:.2f}s behind another request's ingest"
