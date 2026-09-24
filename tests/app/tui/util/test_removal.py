# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A shielded removal runs to its end whatever happens to the task that awaits it."""

from __future__ import annotations

import asyncio

import pytest

from chrys.app.tui.util.removal import finish_shielded
from tests.support.waiting import wait_for, wait_until


class _Operation:
    """An operation that starts, then waits for the test to let it finish."""

    def __init__(self, error: Exception | None = None) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = False
        self._error = error

    async def run(self) -> None:
        self.started.set()
        await self.release.wait()
        self.finished = True
        if self._error is not None:
            raise self._error


async def _caller_inside(operation: _Operation) -> asyncio.Task[None]:
    caller = asyncio.create_task(finish_shielded(operation.run()))
    await wait_for(operation.started.is_set, description="the operation starts")
    return caller


@pytest.mark.asyncio
async def test_a_cancelled_caller_waits_for_the_operation_and_stays_cancelled() -> None:
    operation = _Operation()
    caller = await _caller_inside(operation)

    caller.cancel()
    assert not await wait_until(caller.done, timeout=0.2)
    assert not operation.finished

    operation.release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert operation.finished


@pytest.mark.asyncio
async def test_cancelling_the_caller_again_ends_its_wait_but_not_the_operation() -> None:
    operation = _Operation()
    caller = await _caller_inside(operation)

    caller.cancel()
    # One scheduler turn delivers the first cancellation; the caller is then waiting again.
    await asyncio.sleep(0)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert not operation.finished

    operation.release.set()
    await wait_for(lambda: operation.finished, description="the operation finishes on its own")


@pytest.mark.asyncio
async def test_the_cancellation_outranks_an_error_from_the_operation() -> None:
    operation = _Operation(ValueError("removal failed"))
    caller = await _caller_inside(operation)

    caller.cancel()
    operation.release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert operation.finished


@pytest.mark.asyncio
async def test_a_caller_that_is_not_cancelled_sees_the_operation_error() -> None:
    operation = _Operation(ValueError("removal failed"))
    operation.release.set()

    with pytest.raises(ValueError, match="removal failed"):
        await finish_shielded(operation.run())
