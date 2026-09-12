"""Process-wide limits on how many tools may run at once, per device.

The executor used to create its own ``asyncio.Semaphore`` pair in
``DagExecutor.__init__`` — and the pipeline creates one executor per request. So
every request held its own GPU permit, and two concurrent jobs could each load
SegFormer-B5, the change detector and DOFA on top of an 18.7 GiB resident VLM.
The VLM itself was safe behind the backend's ``RLock``; nothing else was.

:class:`DeviceGates` is the one set of semaphores every executor in the process
shares. It is **not** a module-level global, for a reason that only shows up in
tests: since Python 3.10 an ``asyncio.Semaphore`` binds to the running loop on
its first *contended* ``acquire()``, and a global bound to one loop raises
``RuntimeError: ... is bound to a different event loop`` when contended from
another. pytest-asyncio runs each test on a fresh loop. So the application
creates its gates in ``lifespan`` and hands them to each request through a
dependency; a bare ``DagExecutor()`` gets a private set, which is what the
executor tests want anyway.

The second thing the gates own is the **leaked permit**. ``asyncio.wait_for``
cancels the awaiter on timeout, not the worker thread, and the old ``async with
semaphore:`` released the permit the moment the awaiter was cancelled — while
the thread was still holding its model on the card. The next GPU step then
acquired immediately and OOMed against work the executor believed had finished.
:meth:`DeviceGates.run` holds the permit until the *thread* ends, and counts how
many are outstanding so ``/v1/health`` can say the box is degraded.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Final, TypeVar

from satquery.schemas.enums import Device

MAX_PARALLEL_TOOLS: Final[int] = 3
"""CPU-bound tools. Three keeps the box responsive without leaving cores idle."""

MAX_PARALLEL_GPU_TOOLS: Final[int] = 1
"""One GPU step at a time, process-wide. With 24 GB shared between an 8B VLM and
the CV models, concurrent GPU steps are the shortest path to VRAM exhaustion."""

MAX_LEAKED_PERMITS: Final[int] = 2
"""Past this many timed-out GPU threads still running, the process reports
itself degraded: the card is doing work nobody is waiting for."""

T = TypeVar("T")


class ToolTimeoutError(TimeoutError):
    """A tool overran its budget. Its thread may still be running."""

    def __init__(self, tool: str, timeout_ms: int, *, leaked: bool) -> None:
        """Record which tool, how long it was allowed, and whether it leaked a permit."""
        super().__init__(f"{tool} exceeded its {timeout_ms} ms budget")
        self.tool = tool
        self.timeout_ms = timeout_ms
        self.leaked = leaked


@dataclass
class DeviceGates:
    """One semaphore per device class, plus the count of permits leaked to timeouts."""

    max_parallel: int = MAX_PARALLEL_TOOLS
    max_parallel_gpu: int = MAX_PARALLEL_GPU_TOOLS
    leaked_gpu_permits: int = 0
    _cpu: asyncio.Semaphore = field(init=False, repr=False)
    _gpu: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the semaphores. They bind to a loop lazily, on first contention."""
        self._cpu = asyncio.Semaphore(self.max_parallel)
        self._gpu = asyncio.Semaphore(self.max_parallel_gpu)

    def for_device(self, device: Device) -> asyncio.Semaphore:
        """The semaphore a tool declared for *device* must hold while it runs."""
        return self._gpu if device is Device.ROCM_0 else self._cpu

    @property
    def degraded(self) -> bool:
        """True when enough GPU permits are leaked that new GPU work is unsafe."""
        return self.leaked_gpu_permits >= MAX_LEAKED_PERMITS

    async def run(
        self,
        device: Device,
        tool: str,
        work: Callable[[], Awaitable[T]],
        *,
        timeout_ms: int,
    ) -> T:
        """Run *work* under the device's permit, with a timeout that keeps the permit honest.

        *work* is expected to be an ``asyncio.to_thread`` call (or anything whose
        underlying job cannot be cancelled). It is shielded so the timeout
        cancels only our wait; on timeout the permit is **not** released — a
        done-callback on the still-running future releases it when the thread
        actually finishes, and until then it is counted as leaked.

        Raises:
            ToolTimeoutError: *work* did not finish within *timeout_ms*.
        """
        gate = self.for_device(device)
        await gate.acquire()
        future = asyncio.ensure_future(work())
        try:
            result = await asyncio.wait_for(asyncio.shield(future), timeout=timeout_ms / 1000.0)
        except TimeoutError:
            if future.done():
                # Finished in the gap between the timeout firing and us looking.
                gate.release()
                raise ToolTimeoutError(tool, timeout_ms, leaked=False) from None
            self._release_when_done(future, gate, device)
            raise ToolTimeoutError(tool, timeout_ms, leaked=True) from None
        except BaseException:
            # The tool raised (future is done), or *we* were cancelled — the
            # request went away while the thread still runs. Either way the
            # permit follows the thread, never the awaiter.
            if future.done():
                gate.release()
            else:
                self._release_when_done(future, gate, device)
            raise
        gate.release()
        return result

    def _release_when_done(
        self, future: asyncio.Future[T], gate: asyncio.Semaphore, device: Device
    ) -> None:
        """Hand the permit back when the orphaned thread finishes, counting it meanwhile."""
        is_gpu = device is Device.ROCM_0
        if is_gpu:
            self.leaked_gpu_permits += 1

        def _release(done: asyncio.Future[T]) -> None:
            if is_gpu:
                self.leaked_gpu_permits -= 1
            gate.release()
            if not done.cancelled():
                # Nobody is waiting for this result any more; retrieving the
                # exception keeps asyncio from logging "never retrieved".
                done.exception()

        future.add_done_callback(_release)


__all__ = [
    "MAX_LEAKED_PERMITS",
    "MAX_PARALLEL_GPU_TOOLS",
    "MAX_PARALLEL_TOOLS",
    "DeviceGates",
    "ToolTimeoutError",
]
