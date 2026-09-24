# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Characterize the kernel pause-save window, including its current late pause."""

from __future__ import annotations

import asyncio
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationCascadeAborted, InvocationPaused
from chrys.orchestration.invoker.contracts import SubAgentStatus
from tests.orchestration.sub_agents._controller_fixtures import _make_controller, _ScriptedAgent
from tests.support.event_capture import capture_event_sequence
from tests.support.waiting import wait_for


async def test_kernel_cascade_during_pause_save_leaves_a_late_decision_until_caller_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = EventBus()
    controller = _make_controller(_ScriptedAgent(outcomes=[RuntimeError("pause me")]), bus)
    saving_pause = asyncio.Event()
    release_save = asyncio.Event()
    original_write = controller.policy._write_log

    async def write(*args: object, **kwargs: object) -> bool:
        # autospec validates the production signature; forward every argument.
        if kwargs["status"] == "paused":
            saving_pause.set()
            await release_save.wait()
        return await original_write(*args, **kwargs)

    monkeypatch.setattr(controller.policy, "_write_log", create_autospec(original_write, side_effect=write))
    async with capture_event_sequence(bus, InvocationCascadeAborted, InvocationPaused) as events:
        task = asyncio.create_task(controller.run())
        try:
            await asyncio.wait_for(saving_pause.wait(), timeout=5)
            assert controller._pending_decision is None
            await controller.cascade_abort()
            assert controller.status == SubAgentStatus.CASCADE_ABORTED
            release_save.set()
            await wait_for(
                lambda: any(
                    (isinstance(event, InvocationPaused) and event.origin.kind == "sub_agent") for event in events
                ),
                description="late pause",
            )
            # Baseline behavior, not the future OperationBinding close contract:
            # a cascade during the save cannot settle a future created later.
            assert [type(event) for event in events] == [InvocationCascadeAborted, InvocationPaused]
            assert controller._pending_decision is not None
            assert not controller._pending_decision.done()
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert controller._pending_decision is None
            assert len(events) == 2
        finally:
            release_save.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
