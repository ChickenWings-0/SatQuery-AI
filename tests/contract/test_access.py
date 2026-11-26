"""The shared access code and GPU admission (``satquery.api.access``).

The access code is the only thing between a public tunnel and the GPU, so the
cases that matter are the ones a browser actually produces: the header on a
fetch, ``?code=`` on an ``<img src>``, a CORS preflight that carries neither,
and a 401 the browser can read because CORS headers are still on it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from satquery.agent.concurrency import DeviceGates
from satquery.api.access import ACCESS_HEADER, AccessGuard, parse_codes
from satquery.api.app import create_app
from satquery.core.config import Settings

ORIGIN = "https://demo.example"


def _client(**overrides: object) -> TestClient:
    settings = Settings(**overrides)  # type: ignore[arg-type]
    return TestClient(create_app(settings))


@pytest.fixture
def guarded() -> Iterator[TestClient]:
    with _client(access_code=SecretStr("alpha, bravo"), access_max_failures=3) as client:
        yield client


# --------------------------------------------------------------- the guard


def test_no_code_configured_leaves_the_api_open() -> None:
    with _client(access_code=None) as client:
        assert client.get("/v1/access").status_code == 200
        assert client.get("/v1/health").json()["access_required"] is False


def test_missing_code_is_refused_with_the_error_envelope(guarded: TestClient) -> None:
    response = guarded.get("/v1/access")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "ACCESS_CODE_REQUIRED"


def test_any_configured_code_is_accepted_in_the_header(guarded: TestClient) -> None:
    for code in ("alpha", "bravo", "  bravo  "):
        assert guarded.get("/v1/access", headers={ACCESS_HEADER: code}).status_code == 200


def test_code_is_accepted_as_a_query_parameter_for_img_urls(guarded: TestClient) -> None:
    assert guarded.get("/v1/access?code=alpha").status_code == 200


def test_wrong_code_is_invalid_not_missing(guarded: TestClient) -> None:
    response = guarded.get("/v1/access", headers={ACCESS_HEADER: "charlie"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "ACCESS_CODE_INVALID"


def test_repeated_wrong_codes_lock_the_client_out(guarded: TestClient) -> None:
    for _ in range(3):
        assert guarded.get("/v1/access", headers={ACCESS_HEADER: "nope"}).status_code == 401
    locked = guarded.get("/v1/access", headers={ACCESS_HEADER: "alpha"})
    assert locked.status_code == 429
    assert locked.json()["error"]["code"] == "ACCESS_LOCKED"


def test_browsing_stays_public_without_a_code(guarded: TestClient) -> None:
    """The site can be looked through freely: status and the tools catalogue."""
    health = guarded.get("/v1/health")
    assert health.status_code == 200
    assert health.json()["access_required"] is True
    assert guarded.get("/v1/registry").status_code == 200


def test_running_anything_needs_the_code(guarded: TestClient) -> None:
    upload = {"images": ("a.png", b"not a png", "image/png")}
    for method, path, kwargs in [
        ("post", "/v1/validate", {"files": upload}),
        ("post", "/v1/jobs", {"files": upload, "data": {"query": "x"}}),
        ("post", "/v1/analyze", {"files": upload, "data": {"query": "x"}}),
        ("post", "/v1/imagery/fetch", {"json": {"items": [], "bbox": [0, 0, 1, 1]}}),
        ("get", "/v1/traces/" + "0" * 32, {}),
        ("get", "/v1/jobs/" + "0" * 32, {}),
    ]:
        response = getattr(guarded, method)(path, **kwargs)
        assert response.status_code == 401, (path, response.status_code)


def test_a_wrong_code_on_a_public_read_is_not_counted(guarded: TestClient) -> None:
    for _ in range(5):
        assert guarded.get("/v1/health", headers={ACCESS_HEADER: "stale"}).status_code == 200
    assert guarded.get("/v1/access", headers={ACCESS_HEADER: "alpha"}).status_code == 200


def test_paths_outside_the_api_prefix_are_not_gated(guarded: TestClient) -> None:
    assert guarded.get("/openapi.json").status_code == 200


def test_cors_preflight_passes_without_a_code(guarded: TestClient) -> None:
    response = guarded.options(
        "/v1/jobs",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": ACCESS_HEADER.lower(),
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] in ("*", ORIGIN)


def test_a_401_still_carries_cors_headers(guarded: TestClient) -> None:
    """Without them the browser hides the status and the gate never shows."""
    response = guarded.get("/v1/access", headers={"Origin": ORIGIN})
    assert response.status_code == 401
    assert "access-control-allow-origin" in response.headers


def test_parse_codes_drops_blanks() -> None:
    assert parse_codes(" a, ,b ,") == ("a", "b")
    assert parse_codes("") == ()
    assert parse_codes(None) == ()


def test_lockout_expires_after_its_window() -> None:
    now = [0.0]
    guard = AccessGuard(codes=("x",), max_failures=2, lockout_s=60.0, clock=lambda: now[0])
    guard.record_failure("ip")
    guard.record_failure("ip")
    assert guard.locked_out("ip")
    now[0] = 61.0
    assert not guard.locked_out("ip")


# ------------------------------------------------------------ GPU admission


def test_admission_refuses_past_the_line_length() -> None:
    gates = DeviceGates(max_active_jobs=2)
    assert gates.admit()
    assert gates.admit()
    assert not gates.admit()
    gates.release_admission()
    assert gates.admit()


async def test_job_slots_run_one_analysis_at_a_time_in_order() -> None:
    gates = DeviceGates(max_concurrent_jobs=1)
    running = 0
    peak = 0
    order: list[int] = []

    async def job(n: int) -> None:
        nonlocal running, peak
        async with gates.job_slot():
            running += 1
            peak = max(peak, running)
            order.append(n)
            await asyncio.sleep(0.01)
            running -= 1

    await asyncio.gather(*(job(n) for n in range(4)))
    assert peak == 1
    assert order == [0, 1, 2, 3]


def test_a_full_line_answers_429_gpu_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(access_code=None) as client:
        gates: DeviceGates = client.app.state.gates  # type: ignore[attr-defined]
        monkeypatch.setattr(gates, "active_jobs", gates.max_active_jobs)
        response = client.post(
            "/v1/jobs",
            files={"images": ("a.png", b"not really a png", "image/png")},
            data={"query": "anything"},
        )
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "GPU_BUSY"
