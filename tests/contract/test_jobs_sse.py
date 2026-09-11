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
