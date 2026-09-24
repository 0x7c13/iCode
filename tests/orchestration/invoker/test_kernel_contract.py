# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real main backend admission, cancellation, task ownership and live tickets."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationMessage
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.kernel import AgentResponse, Message
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.orchestration.invoker.contracts import (
    AbortCause,
    Aborted,
    AbortResult,
    Failed,
    Ok,
    OverlappingRun,
    RunIntent,
    RunRequest,
    StaleContinuation,
)
from chrys.orchestration.invoker.origin import BoundEmitter, current_invocation_origin


async def test_failed_pass_ticket_is_single_use_and_preserves_logical_identity(executor: TurnBindings) -> None:
    backend = executor.backend
    failure = ValueError("pass hook failed")
    backend._attempts.run = create_autospec(backend._attempts.run, side_effect=[failure, AgentResponse(messages=[])])
    request = executor.inputs.fresh_request(["input"])
    first = await backend.run(request)
    assert isinstance(first, Failed)
    assert first.exception is failure
    assert first.continuation is not None
    retry = RunRequest([], RunIntent.RETRY, request.origin, first.continuation)
    success = await backend.run(retry)
    assert isinstance(success, Ok)
    assert success.handle.invocation_id == first.handle.invocation_id
    assert success.handle.pass_id != first.handle.pass_id
    with pytest.raises(StaleContinuation):
        await backend.run(retry)
    assert backend._attempts.run.await_count == 2
    assert await backend.abort(first.handle, AbortCause.USER_CANCEL) is AbortResult.ALREADY_CONVERGED


@pytest.mark.parametrize("invalidate", ["restore", "new_run", "close", "copy", "other_origin"])
async def test_stale_ticket_is_rejected_before_hooks_and_history_writes(
    executor: TurnBindings, invalidate: str
) -> None:
    backend = executor.backend
    backend._attempts.run = create_autospec(backend._attempts.run, side_effect=ValueError("failed"))
    request = executor.inputs.fresh_request(["input"])
    failed = await backend.run(request)
    assert isinstance(failed, Failed) and failed.continuation is not None
    ticket = failed.continuation
    origin = request.origin
    if invalidate == "restore":
        executor.backend.history_state = {"messages": [Message("user", ["restored"])]}
    elif invalidate == "new_run":
        await backend.run(executor.inputs.fresh_request(["new turn"]))
    elif invalidate == "close":
        await backend.owner.aclose()
    elif invalidate == "copy":
        ticket = replace(ticket)
    else:
        origin = replace(origin, invocation_id="other")
    calls = backend._attempts.run.await_count
    history = executor.backend.history_state.copy()
    with pytest.raises(StaleContinuation):
        await backend.run(RunRequest([Message("user", ["must not land"])], RunIntent.RETRY, origin, ticket))
    assert backend._attempts.run.await_count == calls
    assert executor.backend.history_state == history


async def test_overlap_and_late_abort_cannot_target_another_pass(executor: TurnBindings) -> None:
    backend = executor.backend
    entered = asyncio.Event()
    release = asyncio.Event()
    seen_origins = []

    async def attempt(input_messages, run_kwargs, *, stream, service_side):
        seen_origins.append(current_invocation_origin.get())
        entered.set()
        await release.wait()
        return AgentResponse(messages=[])

    backend._attempts.run = create_autospec(backend._attempts.run, side_effect=attempt)
    request = executor.inputs.fresh_request(["first"])
    task = asyncio.create_task(backend.run(request))
    try:
        await entered.wait()
        handle = backend.active_handle
        assert handle is not None
        assert backend._attempt_handle is executor._attempt_handle
        with pytest.raises(OverlappingRun):
            await backend.run(replace(request, messages=[Message("user", ["overlap"])]))
        release.set()
        outcome = await task
        assert isinstance(outcome, Ok)
        entered.clear()
        release.clear()
        executor.inputs.begin_invocation()
        next_task = asyncio.create_task(backend.run(executor.inputs.fresh_request(["second"])))
        try:
            await entered.wait()
            assert await backend.abort(handle, AbortCause.OWNER_CLOSE) is AbortResult.ALREADY_CONVERGED
            assert not next_task.done()
            release.set()
            assert isinstance(await next_task, Ok)
        finally:
            release.set()
            await asyncio.gather(next_task, return_exceptions=True)
        assert seen_origins[0] == request.origin
        assert seen_origins[1] != request.origin
        assert current_invocation_origin.get() is None
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("cause", list(AbortCause))
async def test_each_abort_cause_survives_the_backend(executor: TurnBindings, cause: AbortCause) -> None:
    backend = executor.backend
    entered = asyncio.Event()
    release = asyncio.Event()

    async def attempt(input_messages, run_kwargs, *, stream, service_side):
        entered.set()
        await release.wait()
        return AgentResponse(messages=[])

    backend._attempts.run = create_autospec(backend._attempts.run, side_effect=attempt)
    task = asyncio.create_task(backend.run(executor.inputs.fresh_request(["input"])))
    try:
        await entered.wait()
        handle = backend.active_handle
        assert handle is not None
        assert await backend.abort(handle, cause) is AbortResult.REQUESTED
        release.set()
        outcome = await task
        assert isinstance(outcome, Aborted)
        assert outcome.cause is cause
        assert await backend.abort(handle, cause) is AbortResult.ALREADY_CONVERGED
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_bound_emitter_keeps_captured_origin_and_rejects_missing_live_identity() -> None:
    bus = EventBus()
    origin = InvocationOrigin("turn", "s1", "first", None)
    emitter = BoundEmitter(bus, origin)
    events = []

    async def capture(event):
        events.append(event)

    await bus.subscribe(InvocationMessage, capture)
    token = current_invocation_origin.set(InvocationOrigin("turn", "s2", "second", None))
    try:
        await emitter.publish(InvocationMessage(session_id="s1", text="late callback", origin=origin))
        with pytest.raises(ValueError, match="session"):
            await emitter.publish(
                InvocationMessage(
                    session_id="s2", text="foreign", origin=InvocationOrigin("turn", "s2", "turn-test", None)
                )
            )
        with pytest.raises(ValueError, match="origin"):
            BoundEmitter(bus, None)  # type: ignore[arg-type]
    finally:
        current_invocation_origin.reset(token)
        await bus.unsubscribe(InvocationMessage, capture)
    assert [event.text for event in events] == ["late callback"]


async def test_owner_close_latches_cause_before_cancelling_turn_operation(executor: TurnBindings) -> None:
    from chrys.orchestration.invoker.resources import TurnTaskBinding

    backend = executor.backend
    entered = asyncio.Event()
    release = asyncio.Event()
    outcomes = []

    async def attempt(input_messages, run_kwargs, *, stream, service_side):
        entered.set()
        await release.wait()
        return AgentResponse(messages=[])

    backend._attempts.run = create_autospec(backend._attempts.run, side_effect=attempt)

    async def operation():
        task = asyncio.current_task()
        assert task is not None
        unbind = backend.owner.bind_operation(TurnTaskBinding(task, backend.latch_abort))
        try:
            outcomes.append(await backend.run(executor.inputs.fresh_request(["input"])))
        finally:
            unbind()

    task = asyncio.create_task(operation())
    try:
        await entered.wait()
        await backend.owner.aclose()
        assert task.done()
        assert len(outcomes) == 1
        assert isinstance(outcomes[0], Aborted)
        assert outcomes[0].cause is AbortCause.OWNER_CLOSE
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_kernel_state_restore_invalidates_ticket_and_keeps_checkpoint_identity(executor: TurnBindings) -> None:
    backend = executor.backend
    message = Message("user", ["kept"])
    backend.history_state = {"messages": [message]}
    snapshot = backend.checkpoint()
    backend.history_state["messages"].append(Message("user", ["dropped"]))
    generation = backend.state_generation
    backend.restore(snapshot)
    assert backend.state_generation > generation
    assert backend.history_state["messages"] == [message]
    assert backend.history_state["messages"][0] is message
    assert backend.export_audit()["backend"] == "kernel"


@pytest.mark.parametrize("case", ["live", "copy", "origin", "generation"])
async def test_live_ticket_read_port_matches_admission_without_mutation(executor, case):
    backend = executor.backend
    backend._attempts.run = create_autospec(backend._attempts.run, side_effect=ValueError("failed"))
    request = executor.inputs.fresh_request(["work"])
    failed = await backend.run(request)
    assert isinstance(failed, Failed)
    ticket = failed.continuation
    assert ticket is not None
    origin = request.origin
    if case == "copy":
        ticket = replace(ticket)
    elif case == "origin":
        origin = replace(origin, invocation_id="other")
    elif case == "generation":
        # Isolate the generation fence while keeping the identical ticket.
        backend._generation += 1
    generation = backend.state_generation
    live_ticket = backend._ticket
    history = backend.export_audit()
    live = backend.continuation_is_live(ticket, origin)
    assert live is (case == "live")
    retry = RunRequest([], RunIntent.RETRY, origin, ticket)
    if live:
        backend.validate(retry)
    else:
        with pytest.raises(StaleContinuation):
            backend.validate(retry)
    assert backend.state_generation == generation
    assert backend._ticket is live_ticket
    assert backend.export_audit() == history
    assert backend._attempts.run.await_count == 1
