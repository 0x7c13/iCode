# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Main shell to L0 task ownership and synchronous rollback failure boundaries."""

from __future__ import annotations

import asyncio
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.types import Error, InvocationRetryAttempt
from chrys.kernel import AgentResponse, Message, ResponseStream
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.orchestration.invoker.attempts import WireRetryPolicyAdapter
from chrys.service.context.compaction import UnifiedContextStrategy
from tests.orchestration.invoker._main_pass import fresh_pass, retry_pass
from tests.support.event_capture import capture_event_sequence


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("store", [False, True])
async def test_shell_routes_storage_and_wire_policy(
    executor: TurnBindings, monkeypatch: pytest.MonkeyPatch, stream: bool, store: bool
) -> None:
    executor._stream = stream
    executor._chat_options = {"store": store}
    run = create_autospec(executor._attempts.run, side_effect=executor._attempts.run)
    monkeypatch.setattr(executor._attempts, "run", run)

    await fresh_pass(executor, ["input"])

    run.assert_awaited_once()
    assert run.await_args.kwargs == {"stream": stream, "service_side": store}
    run_kwargs = run.await_args.args[1]
    client_kwargs = run_kwargs["client_kwargs"]
    if store:
        assert "wire_retry_policy" not in client_kwargs
    else:
        assert isinstance(client_kwargs["wire_retry_policy"], WireRetryPolicyAdapter)
    assert executor.state.run_failed is False


async def test_interrupt_returns_after_sleep_writeback_clears_attempt_task(
    executor: TurnBindings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hold the sleep writeback return until the real runner's finally clears its task."""
    from chrys.foundation.events.types import InvocationMessage, InvocationToolCallResult, UserInterrupt
    from chrys.kernel.middleware import FunctionInvocationContext
    from chrys.service.agent_middleware.events.hook_dispatch import set_call_id
    from chrys.service.tools.builtins.sleep import sleep

    executor._stream = False
    executor._chat_options = {"store": False}
    sleep_entered = asyncio.Event()
    writeback_finished = asyncio.Event()
    release_writeback = asyncio.Event()
    context = FunctionInvocationContext(function=sleep, arguments={"seconds": 30})
    set_call_id(context, "sleep-call")
    subscribe = executor._bus.subscribe
    interrupt_sleep = executor._interrupt_active_sleep

    async def subscribed(event_type, callback):
        await subscribe(event_type, callback)
        if event_type is UserInterrupt:
            sleep_entered.set()

    async def writeback(call_ids: set[str]) -> None:
        await interrupt_sleep(call_ids)
        writeback_finished.set()
        await release_writeback.wait()

    async def underlying() -> None:
        raise AssertionError("sleep middleware must intercept the tool")

    async def sleeping() -> None:
        await executor._sleep.process(context, underlying)

    async def response() -> AgentResponse:
        await executor.tool_events.process(context, sleeping)
        return AgentResponse(messages=[Message("assistant", ["done"])])

    monkeypatch.setattr(executor._bus, "subscribe", create_autospec(subscribe, side_effect=subscribed))
    barrier = create_autospec(interrupt_sleep, side_effect=writeback)
    monkeypatch.setattr(executor, "_interrupt_active_sleep", barrier)
    monkeypatch.setattr(
        executor._agent, "run", create_autospec(executor._agent.run, side_effect=lambda *args, **kwargs: response())
    )
    async with capture_event_sequence(executor._bus, InvocationMessage, Error, InvocationToolCallResult) as events:
        parent = asyncio.create_task(fresh_pass(executor, ["input"]))
        stop = None
        try:
            await asyncio.wait_for(sleep_entered.wait(), timeout=5)
            handle = executor._attempt_handle
            attempt = handle.task
            assert attempt is not None and not attempt.done()
            assert executor._attempts.handle is handle
            assert executor._sleep.active_call_ids == ("sleep-call",)
            stop = asyncio.create_task(executor.interrupt())
            await asyncio.wait_for(writeback_finished.wait(), timeout=5)
            await asyncio.wait_for(parent, timeout=5)
            assert attempt.done() and not attempt.cancelled()
            assert handle.task is None
            assert not stop.done()
            assert executor.state.was_interrupted is True
            assert executor.state.run_failed is False
            assert executor.state.last_error == ""
            assert executor.state.running is False
            assert handle.active is False
            assert executor._interrupt.is_interrupted is False
            assert executor._sleep.active_call_ids == ()
            assert [type(event) for event in events] == [InvocationToolCallResult]
            assert events[0].call_id == "sleep-call"
            assert events[0].metadata["sleep_interrupted"] is True

            release_writeback.set()
            await asyncio.wait_for(stop, timeout=5)
            barrier.assert_awaited_once_with({"sleep-call"})
            assert handle.task is None
            assert executor.state.was_interrupted is True
            assert executor.state.run_failed is False
            assert executor.state.last_error == ""
            assert executor.state.running is False
            assert executor._interrupt.is_interrupted is False
        finally:
            release_writeback.set()
            tasks = [task for task in (parent, stop) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("phase", ["snapshot", "restore"])
@pytest.mark.parametrize("group", ["recorder", "tool_events", "approval", "compaction"])
async def test_caller_rollback_failure_stops_before_notice_sleep_and_next_attempt(
    executor: TurnBindings, monkeypatch: pytest.MonkeyPatch, stream: bool, phase: str, group: str
) -> None:
    """Every group runs inline; a failure follows the existing shell error path."""
    executor._stream = stream
    executor._chat_options = {"store": True}
    strategy = UnifiedContextStrategy()
    executor._compaction_strategy = strategy
    recorder = executor._loop_recorder
    assert recorder is not None
    order: list[str] = []
    groups = [
        ("recorder", recorder, recorder.snapshot, recorder.restore),
        (
            "tool_events",
            executor.tool_events,
            executor.tool_events.snapshot_retry_state,
            executor.tool_events.restore_retry_state,
        ),
        (
            "approval",
            executor.approval,
            executor.approval.snapshot_retry_state,
            executor.approval.restore_retry_state,
        ),
        ("compaction", strategy, strategy.snapshot_retry_state, strategy.restore_retry_state),
    ]
    for name, participant, snapshot, restore in groups:
        for current_phase, method in [("snapshot", snapshot), ("restore", restore)]:

            def observed(*args, _name=name, _phase=current_phase, _method=method):
                order.append(f"{_phase}:{_name}")
                if (_phase, _name) == (phase, group):
                    raise RuntimeError(f"{phase}:{group} failed")
                return _method(*args)

            monkeypatch.setattr(participant, method.__name__, create_autospec(method, side_effect=observed))

    def failed_run(*args, **kwargs):
        async def fail():
            order.append("agent.run")
            raise ConnectionError("retryable transport failure")

        async def updates():
            await fail()
            yield  # pragma: no cover - makes this the stream-shaped failure

        if kwargs["stream"]:
            return ResponseStream(updates(), finalizer=AgentResponse.from_updates)
        return fail()

    monkeypatch.setattr(executor._agent, "run", create_autospec(executor._agent.run, side_effect=failed_run))
    restore_injection = create_autospec(executor._injection.restore_for_retry)
    end_retry = create_autospec(executor._injection.end_retry, side_effect=executor._injection.end_retry)
    sleep = create_autospec(executor._interruptible_sleep, return_value=False)
    monkeypatch.setattr(executor._injection, "restore_for_retry", restore_injection)
    monkeypatch.setattr(executor._injection, "end_retry", end_retry)
    monkeypatch.setattr(executor._attempts, "_interruptible_sleep", sleep)
    async with capture_event_sequence(executor._bus, Error, InvocationRetryAttempt) as events:
        await fresh_pass(executor, ["input"])

    names = [name for name, _, _, _ in groups]
    prefix = [f"snapshot:{name}" for name in names] + ["agent.run"] if phase == "restore" else []
    assert order == prefix + [f"{phase}:{name}" for name in names[: names.index(group) + 1]]
    assert executor.state.run_failed is True
    assert executor.state.last_error == f"{phase}:{group} failed"
    assert executor.state.was_interrupted is False
    assert executor.state.running is False
    assert executor._attempt_handle.task is None
    assert [type(event) for event in events] == [Error]
    restore_injection.assert_not_called()
    sleep.assert_not_called()
    end_retry.assert_called_once_with()
    assert executor._injection._retry_active is False


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("service_side", [False, True])
@pytest.mark.parametrize("cancel", ["interrupt", "task"])
async def test_shell_control_targets_the_owned_attempt_task_and_preserves_flags(
    executor: TurnBindings, monkeypatch: pytest.MonkeyPatch, stream: bool, service_side: bool, cancel: str
) -> None:
    executor._stream = stream
    executor._chat_options = {"store": service_side}
    entered = asyncio.Event()
    blocked = asyncio.Event()
    tasks: list[asyncio.Task | None] = []

    async def block():
        tasks.append(asyncio.current_task())
        entered.set()
        await blocked.wait()
        return AgentResponse(messages=[Message("assistant", ["done"])])

    def run(*args, **kwargs):
        async def updates():
            await block()
            yield  # pragma: no cover - cancelled before output

        return ResponseStream(updates(), finalizer=AgentResponse.from_updates) if kwargs["stream"] else block()

    monkeypatch.setattr(executor._agent, "run", create_autospec(executor._agent.run, side_effect=run))
    parent = asyncio.create_task(fresh_pass(executor, ["input"]))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        handle = executor._attempt_handle
        assert executor._attempts.handle is handle
        assert handle.task is tasks[0]
        assert handle.task is not parent
        assert handle.active is True
        assert executor.state.running is True
        if cancel == "interrupt":
            await executor.interrupt()
        else:
            assert handle.task is not None
            handle.task.cancel()
        await asyncio.wait_for(parent, timeout=5)
        assert executor.state.was_interrupted is True
        assert executor.state.run_failed is False
        assert executor.state.last_error == ""
        assert executor.state.running is False
        assert handle.task is None
        assert handle.active is False
        assert executor._interrupt.is_interrupted is False
    finally:
        if not parent.done():
            parent.cancel()
        await asyncio.gather(parent, return_exceptions=True)


async def test_progressive_text_precedes_explicit_finalize(
    executor: TurnBindings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the shell observer's boundary after iteration, before L0's final await."""
    from chrys.foundation.events.types import InvocationMessage
    from chrys.kernel import AgentResponseUpdate, Content

    executor._stream = True
    order: list[str] = []

    async def updates():
        yield AgentResponseUpdate(contents=[Content.from_text("first\nlast")], role="assistant")
        order.append("iterator exhausted")

    async def final():
        order.append("explicit finalize")
        return AgentResponse(messages=[Message("assistant", ["first\nlast"])])

    async def message(event: InvocationMessage) -> None:
        order.append(f"{'final' if event.is_final else 'progressive'}:{event.text}")

    stream = create_autospec(ResponseStream, instance=True)
    stream.__aiter__.side_effect = updates
    stream.get_final_response.side_effect = final
    monkeypatch.setattr(executor._agent, "run", create_autospec(executor._agent.run, return_value=stream))
    await executor._bus.subscribe(InvocationMessage, message)
    try:
        await fresh_pass(executor, ["input"])
    finally:
        await executor._bus.unsubscribe(InvocationMessage, message)
    assert executor.state.run_failed is False
    assert order == [
        "iterator exhausted",
        "progressive:first\n",
        "progressive:first\nlast",
        "explicit finalize",
        "final:first\nlast",
    ]


async def test_main_stall_keeps_zero_payload_and_closes_stream(
    executor: TurnBindings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chrys.foundation.retry import StreamStall

    stream = create_autospec(ResponseStream, instance=True)
    monkeypatch.setattr(executor._agent, "run", create_autospec(executor._agent.run, return_value=stream))
    executor._stream_attempt_timeout = 0
    executor._bound_emitter = executor._invocation_publishers.bind(executor.inputs.origin)
    with pytest.raises(StreamStall) as raised:
        await executor._attempts._stream_single_attempt([], {}, watchdog=True)
    assert raised.value.args == (0,)
    assert str(raised.value) == "0"
    stream.aclose.assert_awaited_once_with()
    assert executor._attempt_handle.task is None


async def test_pass_hooks_repeat_only_on_manual_resume(executor: TurnBindings, monkeypatch: pytest.MonkeyPatch) -> None:
    executor._chat_options = {"store": True}
    executor._max_retries_override = 1
    executor._BACKOFF_SCHEDULE = ()
    executor.backend.history_state["messages"] = [Message("user", ["input"])]
    order: list[str] = []
    executor.backend._start_hooks = (lambda: order.append("hook"),)
    outcomes = [ConnectionError("one"), ConnectionError("two"), ConnectionError("three"), AgentResponse(messages=[])]

    async def response():
        order.append("call")
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    run = create_autospec(executor._agent.run, side_effect=lambda *args, **kwargs: response())
    monkeypatch.setattr(executor._agent, "run", run)
    await fresh_pass(executor, ["input"])
    assert executor.state.run_failed is True
    await retry_pass(
        executor,
    )
    assert executor.state.run_failed is False
    assert order == ["hook", "call", "call", "hook", "call", "call"]
    assert run.call_count == 4
    assert outcomes == []
