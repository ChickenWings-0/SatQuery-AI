"""The fast-path signpost (``fast_base_url``) and short-poll job events.

A Cloudflare quick tunnel buffers a streamed response until it ends, so a
client there reads events with ``?poll=true``: each reply must end as soon as
there is news, carry SSE frames with ids the client can resume from, and stop
at the terminal event.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from satquery.api import fast_path
from satquery.api.app import create_app
from satquery.api.jobs import JobStore
from satquery.api.routers import jobs as jobs_router
from satquery.core.config import Settings, get_settings


@pytest.fixture
def metrics_server() -> Iterator[tuple[str, dict[str, object]]]:
    """A stand-in for cloudflared's metrics server; mutate the dict to change it."""
    state: dict[str, object] = {"hostname": "fast-example.trycloudflare.com"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            body = json.dumps(state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fast_path.clear_cache()
    try:
        yield f"http://127.0.0.1:{server.server_port}/quicktunnel", state
    finally:
        server.shutdown()
        fast_path.clear_cache()


# ------------------------------------------------------------ the signpost


def test_fast_base_url_reads_the_quick_tunnel_hostname(
    metrics_server: tuple[str, dict[str, object]],
) -> None:
    url, _ = metrics_server
    assert fast_path.fast_base_url(url) == "https://fast-example.trycloudflare.com"


def test_no_metrics_url_or_no_tunnel_means_none(
    metrics_server: tuple[str, dict[str, object]],
) -> None:
    url, state = metrics_server
    assert fast_path.fast_base_url(None) is None
    state["hostname"] = ""
    assert fast_path.fast_base_url(url) is None


def test_an_unreachable_metrics_server_means_none() -> None:
    fast_path.clear_cache()
    assert fast_path.fast_base_url("http://127.0.0.1:9/quicktunnel") is None


def _app(settings: Settings) -> TestClient:
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_health_reports_the_fast_base_url(
    metrics_server: tuple[str, dict[str, object]],
) -> None:
    url, _ = metrics_server
    with _app(Settings(fast_path_metrics_url=url)) as client:
        body = client.get("/v1/health").json()
    assert body["fast_base_url"] == "https://fast-example.trycloudflare.com"


def test_health_fast_base_url_is_null_when_unconfigured() -> None:
    with _app(Settings(fast_path_metrics_url=None)) as client:
        assert client.get("/v1/health").json()["fast_base_url"] is None


# ------------------------------------------------------------ short polling


def _frames(body: str) -> list[tuple[str, str]]:
    out = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            out.append((fields["id"], fields["event"]))
    return out


async def test_poll_returns_the_backlog_at_once() -> None:
    store = JobStore()
    job = store.create("j1")
    store.publish(job, "queued", {"job_id": "j1"})
    store.publish(job, "stage", {"stage": "ingesting", "pct": 5})
    body = await asyncio.wait_for(jobs_router._poll_once(store, job, None), timeout=1)
    assert _frames(body) == [("0", "queued"), ("1", "stage")]


async def test_poll_resumes_after_the_last_id() -> None:
    store = JobStore()
    job = store.create("j2")
    for n in range(3):
        store.publish(job, "stage", {"stage": "ingesting", "pct": n})
    body = await jobs_router._poll_once(store, job, 1)
    assert _frames(body) == [("2", "stage")]


async def test_poll_waits_for_the_next_event_then_answers() -> None:
    store = JobStore()
    job = store.create("j3")
    store.publish(job, "queued", {"job_id": "j3"})

    async def later() -> None:
        await asyncio.sleep(0.1)
        store.publish(job, "stage", {"stage": "planning", "pct": 20})

    publisher = asyncio.create_task(later())
    body = await asyncio.wait_for(jobs_router._poll_once(store, job, 0), timeout=2)
    await publisher
    assert _frames(body) == [("1", "stage")]


async def test_poll_answers_a_ping_when_nothing_happens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs_router, "POLL_WAIT_SECONDS", 0.05)
    store = JobStore()
    job = store.create("j4")
    store.publish(job, "queued", {"job_id": "j4"})
    assert await jobs_router._poll_once(store, job, 0) == ": ping\n\n"


async def test_poll_stops_at_the_terminal_event() -> None:
    store = JobStore()
    job = store.create("j5")
    store.publish(job, "queued", {"job_id": "j5"})
    store.publish(job, "error", {"code": "X"})
    body = await jobs_router._poll_once(store, job, None)
    assert [event for _, event in _frames(body)] == ["queued", "error"]


def test_poll_query_is_served_over_http() -> None:
    from satquery.api.jobs import get_job_store

    with TestClient(create_app(Settings())) as client:
        store = get_job_store()
        job = store.create("http-poll")
        store.publish(job, "queued", {"job_id": "http-poll"})
        store.publish(job, "done", {})
        response = client.get("/v1/jobs/http-poll/events?poll=true")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [event for _, event in _frames(response.text)] == ["queued", "done"]
