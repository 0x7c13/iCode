# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared deadline-based polling helpers for tests.

Polling avoids scheduler races seen on loaded Windows CI workers, where a
single ``asyncio.sleep(0)`` may return before a fire-and-forget task runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import reprlib
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

if TYPE_CHECKING:
    from chrys.orchestration.engine.execution import RunTaskDrainOutcome
    from chrys.orchestration.engine.run.turn_state import TurnRuntimeState


class _RunTaskChainDrainer(Protocol):
    @property
    def turn_lifecycle_task(self) -> asyncio.Task[None] | None: ...

    async def wait_for_run_task(self) -> None: ...


DEFAULT_WAIT_TIMEOUT = 5.0
ENGINE_TURN_TIMEOUT = 15.0
ENGINE_TEST_WAIT_TIMEOUT = 45.0
QUIET_WAIT_TIMEOUT = 20.0
_OBSERVATION_REPR = reprlib.Repr()
_OBSERVATION_REPR.maxother = 240
_WAIT_DEADLINE: ContextVar[float | None] = ContextVar("test_wait_deadline", default=None)


def shared_wait_deadline(timeout: float) -> float:
    """Return one monotonic deadline shared by waits in the current context."""
    deadline = _WAIT_DEADLINE.get()
    if deadline is None:
        deadline = asyncio.get_running_loop().time() + timeout
        _WAIT_DEADLINE.set(deadline)
    return deadline


def reset_wait_deadline() -> None:
    """Clear the current context's shared wait deadline."""
    _WAIT_DEADLINE.set(None)


def with_wait_deadline(
    timeout: float,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Limit all shared waiting helpers called by an async test to one budget."""

    def decorate(test: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(test)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            existing = _WAIT_DEADLINE.get()
            token = _WAIT_DEADLINE.set(min(deadline, existing) if existing is not None else deadline)
            try:
                return await test(*args, **kwargs)
            finally:
                _WAIT_DEADLINE.reset(token)

        return wrapped

    return decorate


def _wait_window(
    timeout: float, *, use_shared_deadline: bool = True
) -> tuple[asyncio.AbstractEventLoop, float, float, float, bool]:
    """Return timing bounds and whether the shared deadline shortened them."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    requested_deadline = started + timeout
    shared = _WAIT_DEADLINE.get() if use_shared_deadline else None
    shared_limited = shared is not None and shared < requested_deadline
    deadline = shared if shared_limited else requested_deadline
    return loop, started, deadline, max(0.0, deadline - started), shared_limited


def _timeout_diagnostic(requested: float, effective: float, *, shared_limited: bool) -> str:
    if shared_limited:
        return f"timeout={effective:.3f}s from shared deadline; requested={requested:.3f}s"
    return f"timeout={requested:.3f}s"


async def _wait_until_observed(
    predicate: Callable[[], object],
    *,
    timeout: float,
    interval: float,
    pilot: Any | None,
    use_shared_deadline: bool,
) -> tuple[bool, object, float, float, bool]:
    """Return success, observation, timing, and shared-limit provenance."""
    loop, started, deadline, effective_timeout, shared_limited = _wait_window(
        timeout, use_shared_deadline=use_shared_deadline
    )
    while True:
        observed = predicate()
        if observed:
            return True, observed, loop.time() - started, effective_timeout, shared_limited
        if loop.time() >= deadline:
            return False, observed, loop.time() - started, effective_timeout, shared_limited
        if pilot is None:
            await asyncio.sleep(interval)
        else:
            # Both yield to asyncio; Pilot.pause also waits on Textual queue barriers.
            await pilot.pause(interval)


async def wait_until(
    predicate: Callable[[], object],
    *,
    timeout: float = DEFAULT_WAIT_TIMEOUT,
    interval: float = 0.02,
    pilot: Any | None = None,
) -> bool:
    """Return whether *predicate* became truthy before the deadline."""
    met, _observed, _elapsed, _effective_timeout, _shared_limited = await _wait_until_observed(
        predicate,
        timeout=timeout,
        interval=interval,
        pilot=pilot,
        use_shared_deadline=False,
    )
    return met


async def wait_until_quiet(
    probe: Callable[[], object],
    *,
    description: str,
    pumps: int = 3,
    timeout: float = QUIET_WAIT_TIMEOUT,
    interval: float = 0.02,
    pilot: Any | None = None,
) -> None:
    """Pump until *probe* returns an unchanged value ``pumps`` times in a row.

    This is a bounded stability observation, not proof that queued or future
    work has drained. Isolate test phases using the owning producer's completion
    boundary or an appropriate queue barrier before clearing call recorders.
    """
    loop, started, deadline, effective_timeout, shared_limited = _wait_window(timeout)
    last = probe()
    previous = last
    quiet = 0
    while quiet < pumps:
        if pilot is None:
            await asyncio.sleep(interval)
        else:
            # Both yield to asyncio; Pilot.pause also waits on Textual queue barriers.
            await pilot.pause(interval)
        current = probe()
        if current == last:
            quiet += 1
        else:
            quiet = 0
            previous = last
            last = current
        if quiet < pumps and loop.time() >= deadline:
            elapsed = loop.time() - started
            raise AssertionError(
                f"{description} still changing after {elapsed:.3f}s "
                f"({_timeout_diagnostic(timeout, effective_timeout, shared_limited=shared_limited)}); "
                f"previous={_OBSERVATION_REPR.repr(previous)}; current={_OBSERVATION_REPR.repr(last)}"
            )


async def wait_for(
    predicate: Callable[[], object],
    *,
    timeout: float = DEFAULT_WAIT_TIMEOUT,
    interval: float = 0.02,
    pilot: Any | None = None,
    description: str = "condition",
) -> None:
    """Raise with *description* unless *predicate* becomes truthy in time."""
    met, observed, elapsed, effective_timeout, shared_limited = await _wait_until_observed(
        predicate,
        timeout=timeout,
        interval=interval,
        pilot=pilot,
        use_shared_deadline=True,
    )
    if not met:
        raise AssertionError(
            f"{description} not met after {elapsed:.3f}s "
            f"({_timeout_diagnostic(timeout, effective_timeout, shared_limited=shared_limited)}); "
            f"last observed={_OBSERVATION_REPR.repr(observed)}"
        )


@overload
async def await_run_task_chain(
    engine: _RunTaskChainDrainer,
    *,
    turn_state: TurnRuntimeState,
    timeout: float = ENGINE_TURN_TIMEOUT,
    propagate_inner_cancel: Literal[False] = False,
    expect_installed: bool = False,
) -> RunTaskDrainOutcome: ...


@overload
async def await_run_task_chain(
    engine: _RunTaskChainDrainer,
    *,
    turn_state: TurnRuntimeState,
    timeout: float = ENGINE_TURN_TIMEOUT,
    propagate_inner_cancel: Literal[True],
    expect_installed: bool = False,
) -> None: ...


async def await_run_task_chain(
    engine: _RunTaskChainDrainer,
    *,
    turn_state: TurnRuntimeState,
    timeout: float = ENGINE_TURN_TIMEOUT,
    propagate_inner_cancel: bool = False,
    expect_installed: bool = False,
) -> RunTaskDrainOutcome | None:
    """Wait for execution, unwinding, history mutation, and final save.

    Tests must not inspect history or persisted session state until the entire
    run-task chain finishes. Both observers shield the product run task, so
    cancelling this test waiter on timeout doesn't cancel the late turn it is
    reporting. Set ``propagate_inner_cancel`` only where a cancelled product
    task is itself the expected signal; otherwise the boundary outcome reports
    it. ``expect_installed`` makes tests that require synchronous task
    installation fail at that contract instead of at a later state assertion.
    ``wait_for`` supplies the shared test deadline and diagnostic.
    """
    if expect_installed and engine.turn_lifecycle_task is None:
        raise AssertionError("engine run-task chain was not installed")
    if propagate_inner_cancel:
        drain_task = asyncio.create_task(engine.wait_for_run_task())
    else:
        drain_task = asyncio.create_task(turn_state.drain_for_boundary())
    try:
        await wait_for(
            drain_task.done,
            timeout=timeout,
            description="engine run-task chain completion",
        )
        return await drain_task
    finally:
        if not drain_task.done():
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task
