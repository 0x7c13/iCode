# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Whole-pass close without an operation binding and exact-pass abort fences."""

from __future__ import annotations

import asyncio
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.types import AgentThinking
from chrys.foundation.hosted_tools import HostedToolStatus
from chrys.kernel import AgentResponse, tool
from chrys.orchestration.invoker.contracts import AbortCause, Aborted, Ok, UsageDelta
from chrys.service.llm.mock import MockChatClient, MockResponse


@pytest.mark.parametrize("window", ["hosted", "sleep"])
async def test_abort_cross_await_does_not_cancel_successor(executor, monkeypatch, window):
    backend = executor.backend
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    interrupt_entered, interrupt_release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def attempt(input_messages, run_kwargs, *, stream, service_side):
        nonlocal calls
        i = calls
        calls += 1

        async def model():
            entered[i].set()
            await release[i].wait()
            return AgentResponse(messages=[])

        task = asyncio.create_task(model())
        backend._attempt_handle.task = task
        try:
            return await task
        finally:
            if backend._attempt_handle.task is task:
                backend._attempt_handle.task = None

    async def wait_for_interrupt():
        interrupt_entered.set()
        await interrupt_release.wait()

    monkeypatch.setattr(backend._attempts, "run", create_autospec(backend._attempts.run, side_effect=attempt))
    first = asyncio.create_task(backend.run(executor.inputs.fresh_request(["first"])))
    aborting = second = None
    try:
        await asyncio.wait_for(entered[0].wait(), 5)
        if window == "hosted":
            bridge = executor._hosted_bridge
            original = bridge.attempt_rejected
            first_notice = True

            async def reject(reason="", *, status=HostedToolStatus.FAILED, preserve_provisional=False):
                nonlocal first_notice
                if first_notice:
                    first_notice = False
                    await wait_for_interrupt()
                await original(reason, status=status, preserve_provisional=preserve_provisional)

            monkeypatch.setattr(bridge, "attempt_rejected", create_autospec(original, side_effect=reject))
        else:
            executor._sleep._active["sleep-call"] = asyncio.get_running_loop().create_future()

            async def sleep_writeback(call_ids):
                assert call_ids == {"sleep-call"}
                executor._sleep._active.clear()
                await wait_for_interrupt()

            monkeypatch.setattr(
                executor,
                "_interrupt_active_sleep",
                create_autospec(executor._interrupt_active_sleep, side_effect=sleep_writeback),
            )
        aborting = asyncio.create_task(backend.abort(backend.active_handle, AbortCause.USER_CANCEL))
        await asyncio.wait_for(interrupt_entered.wait(), 5)
        release[0].set()
        assert isinstance(await first, Aborted)
        executor.inputs.begin_invocation()
        second = asyncio.create_task(backend.run(executor.inputs.fresh_request(["second"])))
        await asyncio.wait_for(entered[1].wait(), 5)
        successor = backend._attempt_handle.task
        interrupt_release.set()
        await aborting
        assert successor.cancelling() == 0
        release[1].set()
        assert isinstance(await second, Ok)
    finally:
        executor._sleep._active.clear()
        interrupt_release.set()
        for event in release:
            event.set()
        await asyncio.gather(*(x for x in (first, aborting, second) if x is not None), return_exceptions=True)


async def test_real_interrupt_cancels_replacement_attempt_in_same_pass(executor, monkeypatch):
    backend = executor.backend
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    interrupt_entered, interrupt_release = asyncio.Event(), asyncio.Event()
    attempts = []

    async def attempt(input_messages, run_kwargs, *, stream, service_side):
        for index in range(2):

            async def model(i=index):
                entered[i].set()
                await release[i].wait()
                return AgentResponse(messages=[])

            task = asyncio.create_task(model())
            attempts.append(task)
            backend._attempt_handle.task = task
            try:
                result = await task
            finally:
                if backend._attempt_handle.task is task:
                    backend._attempt_handle.task = None
        return result

    monkeypatch.setattr(backend._attempts, "run", create_autospec(backend._attempts.run, side_effect=attempt))
    running = asyncio.create_task(backend.run(executor.inputs.fresh_request(["work"])))
    interrupting = None
    try:
        await asyncio.wait_for(entered[0].wait(), 5)
        original = executor._hosted_bridge.attempt_rejected

        async def reject(reason="", *, status=HostedToolStatus.FAILED, preserve_provisional=False):
            interrupt_entered.set()
            await interrupt_release.wait()
            await original(reason, status=status, preserve_provisional=preserve_provisional)

        monkeypatch.setattr(executor._hosted_bridge, "attempt_rejected", create_autospec(original, side_effect=reject))
        # Exercise the observer itself so backend.abort's final cancellation
        # cannot mask an over-broad observer fence based on attempt identity.
        interrupting = asyncio.create_task(executor.interrupt())
        await asyncio.wait_for(interrupt_entered.wait(), 5)
        release[0].set()
        await asyncio.wait_for(entered[1].wait(), 5)
        interrupt_release.set()
        await interrupting
        assert attempts[1].cancelling() == 1
        assert isinstance(await running, Aborted)
        assert attempts[0].cancelled() is False
        assert attempts[1].cancelled() is True
    finally:
        interrupt_release.set()
        for event in release:
            event.set()
        await asyncio.gather(*(task for task in (running, interrupting) if task is not None), return_exceptions=True)


@pytest.mark.parametrize("swallow_cancel", [False, True])
async def test_owner_close_before_attempt_does_not_start_provider(executor, swallow_cancel):
    backend = executor.backend
    entered, release = asyncio.Event(), asyncio.Event()

    async def thinking(event):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            if not swallow_cancel:
                raise

    await executor._bus.subscribe(AgentThinking, thinking)
    task = asyncio.create_task(backend.run(executor.inputs.fresh_request(["work"])))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert backend._attempt_handle.task is None
        closing = asyncio.create_task(backend.owner.aclose())
        # The gate stays closed until both pass and owner have converged.
        outcome = await asyncio.wait_for(asyncio.shield(task), 5)
        await asyncio.wait_for(asyncio.shield(closing), 5)
        assert not release.is_set()
        assert isinstance(outcome, Aborted)
        assert outcome.cause is AbortCause.OWNER_CLOSE
        assert outcome.usage == UsageDelta(unreported=1)
        assert executor._agent.client.call_count == 0
        assert task.cancelling() == 0
        assert task.cancelled() is False
        assert outcome.effects.pass_id == outcome.handle.pass_id
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await executor._bus.unsubscribe(AgentThinking, thinking)


@pytest.mark.parametrize("window", ["succeeded", "finished"])
async def test_owner_close_after_attempt_drains_observer_with_pass_usage(executor, monkeypatch, window):
    backend = executor.backend
    entered, release = asyncio.Event(), asyncio.Event()
    executor._agent.client = MockChatClient(
        responses=[
            MockResponse(
                text="done",
                usage_details={
                    "input_token_count": 7,
                    "output_token_count": 3,
                    "total_token_count": 10,
                },
            )
        ]
    )
    original = executor.succeeded if window == "succeeded" else executor.finished

    async def observe(*args):
        try:
            assert backend._attempt_handle.task is None
            entered.set()
            await release.wait()
        finally:
            await original(*args)

    monkeypatch.setattr(executor, window, create_autospec(original, side_effect=observe))
    task = asyncio.create_task(backend.run(executor.inputs.fresh_request(["work"])))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        handle = backend.active_handle
        closing = asyncio.create_task(backend.owner.aclose())
        outcome = await asyncio.wait_for(asyncio.shield(task), 5)
        await asyncio.wait_for(asyncio.shield(closing), 5)
        assert not release.is_set()
        assert isinstance(outcome, Aborted)
        assert outcome.cause is AbortCause.OWNER_CLOSE
        assert outcome.handle is handle
        assert outcome.effects.pass_id == handle.pass_id
        assert outcome.usage.total_tokens == 10
        assert outcome.usage.complete is True
        assert executor._agent.client.call_count == 1
        assert task.cancelling() == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)


async def test_external_cancel_without_cause_propagates_and_retains_count(executor):
    entered, release = asyncio.Event(), asyncio.Event()

    async def thinking(event):
        entered.set()
        await release.wait()

    await executor._bus.subscribe(AgentThinking, thinking)
    task = asyncio.create_task(executor.backend.run(executor.inputs.fresh_request(["work"])))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert task.cancel() is True
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() is True
        assert task.cancelling() == 1
        assert executor.backend.active_handle is None
        assert executor._agent.client.call_count == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await executor._bus.unsubscribe(AgentThinking, thinking)


@pytest.mark.parametrize("mode", ["timeout", "taskgroup"])
async def test_external_cancellation_frameworks(executor, mode):
    entered = asyncio.Event()
    release = asyncio.Event()
    timeouts = []

    async def gate(event):
        entered.set()
        await release.wait()

    await executor._bus.subscribe(AgentThinking, gate)

    async def timed():
        async with asyncio.timeout(None) as timeout:
            timeouts.append(timeout)
            return await executor.backend.run(executor.inputs.fresh_request(["work"]))

    task = None
    try:
        if mode == "timeout":
            task = asyncio.create_task(timed())
            await entered.wait()
            timeouts[0].reschedule(asyncio.get_running_loop().time())
            with pytest.raises(TimeoutError):
                await task
            assert task.cancelling() == 0
        else:
            with pytest.raises(ExceptionGroup) as group:
                async with asyncio.TaskGroup() as tg:
                    task = tg.create_task(executor.backend.run(executor.inputs.fresh_request(["work"])))
                    await entered.wait()
                    raise ValueError("sibling failed")
            assert len(group.value.exceptions) == 1
            assert str(group.value.exceptions[0]) == "sibling failed"
            assert task.cancelled()
            assert task.cancelling() == 1
        assert executor.backend.active_handle is None
        assert executor._agent.client.call_count == 0
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await executor._bus.unsubscribe(AgentThinking, gate)


async def test_owner_close_inside_timeout_and_taskgroup(executor):
    entered, release = asyncio.Event(), asyncio.Event()

    async def gate(event):
        entered.set()
        await release.wait()

    await executor._bus.subscribe(AgentThinking, gate)

    async def run():
        async with asyncio.timeout(None):
            result = await executor.backend.run(executor.inputs.fresh_request(["work"]))
            current = asyncio.current_task()
            assert current is not None
            assert current.cancelling() == 0
            return result

    try:
        async with asyncio.TaskGroup() as tg:
            task = tg.create_task(run())
            await entered.wait()
            await executor.backend.owner.aclose()
        assert isinstance(task.result(), Aborted)
        assert task.result().cause is AbortCause.OWNER_CLOSE
        assert task.cancelling() == 0
    finally:
        release.set()
        await executor._bus.unsubscribe(AgentThinking, gate)


async def test_owner_close_during_real_attempt_keeps_pass_evidence(executor):
    from chrys.kernel import tool
    from chrys.orchestration.invoker.evidence import Completeness, Count
    from chrys.service.llm.mock import MockChatClient

    entered, release = asyncio.Event(), asyncio.Event()

    @tool
    async def blocking_tool() -> str:
        entered.set()
        await release.wait()
        return "done"

    executor._agent.default_options["tools"] = [blocking_tool]
    executor._agent.client = MockChatClient(responses=[MockResponse(tool_calls=[("blocking_tool", "c1", {})])])
    task = asyncio.create_task(executor.backend.run(executor.inputs.fresh_request(["work"])))
    try:
        await entered.wait()
        active = executor.backend.active_handle
        assert active is not None
        assert executor.backend._attempt_handle.task is not None
        await executor.backend.owner.aclose()
        result = await task
        assert isinstance(result, Aborted)
        assert result.cause is AbortCause.OWNER_CLOSE
        assert result.handle is active
        assert result.effects.pass_id == active.pass_id
        assert result.effects.local_answered == Count(executor._loop_recorder.committed_count, Completeness.EXACT)
        assert result.usage.complete is False
        assert result.usage.unreported == 1
        assert task.cancelling() == 0
        assert executor._agent.client.call_count == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_double_cancel_remains_l0_local(executor, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    counts, tasks = [], []

    @tool
    async def blocked() -> str:
        entered.set()
        await release.wait()
        return "done"

    executor._stream = False
    original = executor._agent.run

    async def run(*args, **kwargs):
        task = asyncio.current_task()
        assert task is not None
        tasks.append(task)
        try:
            async with asyncio.timeout(None):
                return await original(*args, **kwargs)
        finally:
            counts.append(task.cancelling())

    monkeypatch.setattr(executor._agent, "run", create_autospec(original, side_effect=run))
    executor._agent.default_options["tools"] = [blocked]
    executor._agent.client = MockChatClient(responses=[MockResponse(tool_calls=[("blocked", "c1", {})])])
    outer = asyncio.create_task(executor.backend.run(executor.inputs.fresh_request(["work"])))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await executor.backend.abort(executor.backend.active_handle, AbortCause.USER_CANCEL)
        result = await asyncio.wait_for(outer, 5)
        assert isinstance(result, Aborted)
        assert counts == [2]
        assert tasks[0].cancelled()
        assert outer.cancelling() == 0
    finally:
        release.set()
        await asyncio.gather(outer, return_exceptions=True)
