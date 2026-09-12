"""Process-wide device gates, the leaked-permit rule, and cooperative cancellation.

These are the parts of the executor that only misbehave under concurrency,
which is exactly when nobody is watching a single trace.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from satquery.agent.concurrency import (
    MAX_LEAKED_PERMITS,
    DeviceGates,
    ToolTimeoutError,
)
from satquery.schemas.enums import Device


class _Counter:
    """Counts how many bodies are inside a gate at once, from worker threads."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        with self._lock:
            self.current -= 1


def _blocking(counter: _Counter, seconds: float) -> str:
    counter.enter()
    try:
        time.sleep(seconds)
    finally:
        counter.leave()
    return "ok"


# --------------------------------------------------------------- one permit


async def test_two_users_of_one_gate_share_one_gpu_permit() -> None:
    """The defect: one semaphore per executor meant one permit per request."""
    gates = DeviceGates(max_parallel=3, max_parallel_gpu=1)
    counter = _Counter()

    async def step() -> str:
        return await gates.run(
            Device.ROCM_0,
            "detector",
            lambda: asyncio.to_thread(_blocking, counter, 0.05),
            timeout_ms=5_000,
        )

    results = await asyncio.gather(step(), step(), step())
    assert results == ["ok"] * 3
    assert counter.peak == 1, "GPU steps must never overlap"


async def test_cpu_tools_still_overlap() -> None:
    """The fix is a shared gate, not 'everything serialised'."""
    gates = DeviceGates(max_parallel=3, max_parallel_gpu=1)
    counter = _Counter()

    async def step() -> str:
        return await gates.run(
            Device.CPU,
            "indices",
            lambda: asyncio.to_thread(_blocking, counter, 0.1),
            timeout_ms=5_000,
        )

    await asyncio.gather(step(), step(), step())
    assert counter.peak == 3


# ------------------------------------------------------------ leaked permit


async def test_a_timed_out_gpu_thread_keeps_its_permit_until_it_finishes() -> None:
    """The next GPU step waits for the orphaned thread rather than OOMing beside it."""
    gates = DeviceGates(max_parallel_gpu=1)
    counter = _Counter()
    started = time.perf_counter()

    with pytest.raises(ToolTimeoutError) as caught:
        await gates.run(
            Device.ROCM_0,
            "slow",
            lambda: asyncio.to_thread(_blocking, counter, 0.3),
            timeout_ms=50,
        )
    assert caught.value.leaked is True
    assert gates.leaked_gpu_permits == 1
    assert time.perf_counter() - started < 0.25, "the awaiter was released at the timeout"

    # The second step cannot start until the first thread is done.
    await gates.run(
        Device.ROCM_0,
        "next",
        lambda: asyncio.to_thread(_blocking, counter, 0.01),
        timeout_ms=5_000,
    )
    assert counter.peak == 1
    assert time.perf_counter() - started >= 0.3
    # Give the done-callback a tick to run.
    await asyncio.sleep(0)
    assert gates.leaked_gpu_permits == 0


async def test_enough_leaks_report_the_process_degraded() -> None:
    gates = DeviceGates(max_parallel_gpu=MAX_LEAKED_PERMITS + 1)
    counter = _Counter()
    for _ in range(MAX_LEAKED_PERMITS):
        with pytest.raises(ToolTimeoutError):
            await gates.run(
                Device.ROCM_0,
                "slow",
                lambda: asyncio.to_thread(_blocking, counter, 0.2),
                timeout_ms=20,
            )
    assert gates.degraded is True
    await asyncio.sleep(0.3)
    assert gates.degraded is False


async def test_a_tool_that_raises_releases_its_permit() -> None:
    gates = DeviceGates(max_parallel_gpu=1)

    def boom() -> None:
        raise RuntimeError("checkpoint refused to load")

    with pytest.raises(RuntimeError):
        await gates.run(Device.ROCM_0, "boom", lambda: asyncio.to_thread(boom), timeout_ms=1_000)
    # If the permit were still held this would hang; the timeout proves it is not.
    await asyncio.wait_for(
        gates.run(Device.ROCM_0, "ok", lambda: asyncio.to_thread(lambda: 1), timeout_ms=1_000),
        timeout=1.0,
    )
