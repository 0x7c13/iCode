# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Child caller policies over the single L0 attempt owner."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from functools import partial
from typing import Any, Unpack
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    InvocationAnswerWithdrawn,
    InvocationContentDelta,
    InvocationPaused,
    InvocationResumed,
    InvocationRetryAttempt,
)
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.foundation.retry import StreamRetryLoop
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.kernel import (
    RETRY_BOUNDARY_UPDATE_KEY,
    Agent,
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    Content,
    LoopRecorder,
    Message,
    ResponseStream,
)
from chrys.orchestration.invoker.attempts import AgentRunKwargs
from chrys.orchestration.invoker.contracts import SubAgentStatus
from chrys.orchestration.invoker.resources import Conversation
from chrys.orchestration.sub_agents.shell import SubAgentToolShell
from chrys.service.llm.mock import MockChatClient
from tests.orchestration.sub_agents._controller_fixtures import _make_controller, kernel_shell
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support.event_capture import collect_events
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for


def _agent(monkeypatch: pytest.MonkeyPatch, run) -> Agent:
    agent = Agent(client=MockChatClient())
    monkeypatch.setattr(agent, "run", create_autospec(agent.run, side_effect=run))
    return agent


@pytest.mark.parametrize("initial", [None, "invalid", 42])
def test_child_history_reader_ensures_state_before_snapshot(initial: object) -> None:
    session = AgentSession()
    if initial is not None:
        session.state["chrys_history"] = initial
    controller = _make_controller(Agent(client=MockChatClient()), None, session=session)
    snapshot = controller.policy._rollback.snapshot()
    assert session.state["chrys_history"] == {}
    assert snapshot.messages == []
    session.state["chrys_history"] = "invalid again"
    controller.policy._rollback.restore(snapshot)
    assert session.state["chrys_history"] == {"messages": []}


async def test_child_boundary_keeps_event_loop_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = asyncio.Event()
    boundary_observations: list[bool] = []

    async def updates():
        asyncio.get_running_loop().call_soon(observed.set)
        yield AgentResponseUpdate(contents=[], additional_properties={RETRY_BOUNDARY_UPDATE_KEY: True})
        boundary_observations.append(observed.is_set())
        yield AgentResponseUpdate(role="assistant", contents=[Content.from_text("done")])

    def run(_input, *, stream: bool, **_kwargs: Unpack[AgentRunKwargs]):
        assert stream
        return ResponseStream(updates(), finalizer=AgentResponse.from_updates)

    controller = _make_controller(_agent(monkeypatch, run), None, stream=True)
    assert await controller.run() == "done"
    assert boundary_observations == [True]
    assert controller.policy._attempt_handle.task is None


async def test_child_stall_preserves_final_attempt_and_timeout_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = EventBus()
    session = AgentSession()
    original = Message("user", ["original"])
    session.state["chrys_history"] = {"messages": [original]}
    calls: list[list[Message]] = []
    rejected_states: list[list[Message]] = []
    paused: list[InvocationPaused] = []
    restores: list[object] = []

    class Compaction:
        value = "original"
        max_context_tokens = 200_000

        def snapshot_retry_state(self):
            return self.value

        def restore_retry_state(self, value):
            restores.append(value)
            self.value = value

    compaction = Compaction()
    mutated = [Message("assistant", ["attempt one"]), Message("assistant", ["attempt two"])]

    def run(input: list[Message], *, stream: bool, **_kwargs: Unpack[AgentRunKwargs]):
        assert stream, "child stall exhaustion must never call blocking"
        calls.append(input)

        async def updates():
            session.state["chrys_history"]["messages"].append(mutated[len(calls) - 1])
            compaction.value = f"attempt {len(calls)}"
            # Progress with nothing readable: a stall that showed no text may be retried.
            yield AgentResponseUpdate(role="assistant", contents=[])
            await asyncio.Event().wait()

        return ResponseStream(updates(), finalizer=AgentResponse.from_updates)

    controller = _make_controller(
        _agent(monkeypatch, run),
        bus,
        session=session,
        stream=True,
        max_retries=1,
        run_kwargs={"options": {"store": True}, "compaction_strategy": compaction},
    )
    controller.policy._stream_attempt_timeout = 0.01
    reject = controller.policy._reject_hosted_attempt

    async def record_rejection(message: str) -> None:
        rejected_states.append(list(session.state["chrys_history"]["messages"]))
        await reject(message)

    monkeypatch.setattr(
        controller.policy, "_reject_hosted_attempt", create_autospec(reject, side_effect=record_rejection)
    )

    async def abort(event: InvocationPaused) -> None:
        paused.append(event)
        controller.request_abort()

    await bus.subscribe(InvocationPaused, abort)
    try:
        assert (await controller.run()).startswith("Error:")
    finally:
        await bus.unsubscribe(InvocationPaused, abort)
    assert len(calls) == 2
    assert calls[0] is calls[1]
    assert calls[0][0] is calls[1][0]
    assert calls[0][0].contents[0] is calls[1][0].contents[0]
    assert rejected_states == [[original, mutated[1]]]
    assert restores == ["original"]
    assert compaction.value == "attempt 2"
    assert paused[0].last_error == "Stream stalled after 1 retries: no streaming updates received for 0.01s"
    assert controller.policy._attempt_handle.task is None


@pytest.mark.parametrize(
    ("completed", "shows_answer_text"),
    [(True, True), (False, True), (True, False)],
    ids=["shown_segment", "unfinished_segment", "segment_no_frontend_shows"],
)
async def test_a_child_stall_is_retried_as_a_whole_run_only_while_no_answer_segment_showed(
    monkeypatch: pytest.MonkeyPatch, completed: bool, shows_answer_text: bool
) -> None:
    """Sub-agent views show an answer segment once it completes; one still streaming, or nowhere shown, never showed."""
    bus = EventBus()
    calls: list[list[Message]] = []
    paused: list[InvocationPaused] = []

    def run(input: list[Message], *, stream: bool, **_kwargs: Unpack[AgentRunKwargs]):
        assert stream
        calls.append(input)

        async def updates():
            yield AgentResponseUpdate(role="assistant", contents=[Content.from_text("partial")])
            if completed:
                # Reasoning after it completes the answer segment.
                yield AgentResponseUpdate(role="assistant", contents=[Content.from_text_reasoning(text="hmm")])
            await asyncio.Event().wait()

        return ResponseStream(updates(), finalizer=AgentResponse.from_updates)

    controller = _make_controller(
        _agent(monkeypatch, run),
        bus,
        stream=True,
        max_retries=3,
        run_kwargs={"options": {"store": True}},
        shows_answer_text=shows_answer_text,
    )
    controller.policy._stream_attempt_timeout = 0.01

    async def abort(event: InvocationPaused) -> None:
        paused.append(event)
        controller.request_abort()

    await bus.subscribe(InvocationPaused, abort)
    try:
        assert (await controller.run()).startswith("Error:")
    finally:
        await bus.unsubscribe(InvocationPaused, abort)
    if completed and shows_answer_text:
        # The shown text would be shown again: the stalled pass is not run a second time.
        assert len(calls) == 1
        assert paused[0].retry_attempts == 0
        assert paused[0].last_error == "no streaming updates received for 0.01s"
    else:
        # The whole run is retried until its budget runs out.
        assert len(calls) == 4
        assert paused[0].retry_attempts == 3
        assert paused[0].last_error == "Stream stalled after 3 retries: no streaming updates received for 0.01s"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("stored", [False, True])
async def test_child_checks_cascade_before_attempt(monkeypatch: pytest.MonkeyPatch, stream: bool, stored: bool) -> None:
    agent = Agent(client=MockChatClient())
    run = create_autospec(agent.run, side_effect=asyncio.CancelledError)
    monkeypatch.setattr(agent, "run", run)
    controller: SubAgentToolShell

    def latch() -> None:
        controller._cascade_requested = True

    if stored:
        original_run = StreamRetryLoop.run

        async def run_with_latched_attempt(loop: StreamRetryLoop, attempt_fn):
            async def attempt():
                # The real loop has already checked its latch; isolate the attempt's own check.
                latch()
                return await attempt_fn()

            return await original_run(loop, attempt)

        monkeypatch.setattr(StreamRetryLoop, "run", run_with_latched_attempt)

    controller = _make_controller(
        agent,
        None,
        stream=stream,
        run_kwargs={"options": {"store": stored}},
        pass_start_hooks=() if stored else (latch,),
    )
    with pytest.raises(asyncio.CancelledError):
        await controller.run()
    run.assert_not_called()
    assert controller.status is SubAgentStatus.CASCADE_ABORTED
    assert controller.policy._attempt_handle.task is None


async def test_child_blocking_call_timing_and_single_cancel_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    context = ContextVar("child_attempt_test", default="unset")
    call_tasks: list[asyncio.Task[Any] | None] = []
    body_tasks: list[asyncio.Task[Any] | None] = []
    values: list[str] = []

    def run(_input, *, stream: bool, **_kwargs: Unpack[AgentRunKwargs]):
        assert not stream
        call_tasks.append(asyncio.current_task())
        values.append(context.get())

        async def body():
            body_tasks.append(asyncio.current_task())
            values.append(context.get())
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        return body()

    controller = _make_controller(_agent(monkeypatch, run), None)
    token = context.set("caller context")
    task = asyncio.create_task(controller.run())
    context.reset(token)
    try:
        await entered.wait()
        handle = controller.policy._attempt_handle
        assert controller.policy._attempts.handle is handle
        assert call_tasks == [task]
        assert body_tasks == [handle.task]
        assert values == ["caller context", "caller context"]
        await controller.cascade_abort()
        assert cancelled.is_set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert handle.task is None
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("interrupted", [False, True])
async def test_child_service_notice_trace_sleep_order(monkeypatch: pytest.MonkeyPatch, interrupted: bool) -> None:
    bus = EventBus()
    sink = FakeSink()
    order: list[tuple[str, list[str]]] = []
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def run(_input, *, stream: bool, **_kwargs: Unpack[AgentRunKwargs]):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("transient")
        return AgentResponse(messages=[Message("assistant", ["done"])])

    controller = _make_controller(_agent(monkeypatch, run), bus, run_kwargs={"options": {"store": True}})
    controller.policy._trajectory_context = make_context(sink)
    controller.policy._trajectory_boundary_operation_id = new_analytics_id()

    async def notice(_event: InvocationRetryAttempt) -> None:
        order.append(("notice", list(sink.event_types)))

    async def sleep(_seconds: int) -> bool:
        order.append(("sleep entered", list(sink.event_types)))
        entered.set()
        await release.wait()
        order.append(("sleep complete", list(sink.event_types)))
        return interrupted

    monkeypatch.setattr(
        controller.policy,
        "_interruptible_sleep",
        create_autospec(controller.policy._interruptible_sleep, side_effect=sleep),
    )
    await bus.subscribe(InvocationRetryAttempt, notice)
    task = asyncio.create_task(controller.run())
    try:
        await entered.wait()
        assert sink.event_types == [EventType.RETRY_SCHEDULED]
        release.set()
        if interrupted:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert await task == "done"
        assert order == [
            ("notice", []),
            ("sleep entered", [EventType.RETRY_SCHEDULED]),
            ("sleep complete", [EventType.RETRY_SCHEDULED]),
        ]
        assert sink.event_types == [EventType.RETRY_SCHEDULED] + ([] if interrupted else [EventType.RETRY_STARTED])
        assert calls == (1 if interrupted else 2)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await bus.unsubscribe(InvocationRetryAttempt, notice)


async def test_child_reject_backoff_pause_save_event_retry_input_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from chrys.service.agent_middleware.events.sub_agent_events import SubAgentEventMiddleware

    bus = EventBus()
    events = SubAgentEventMiddleware(
        bus, "Explore", "inv-1", session_id="s-1", origin=InvocationOrigin("sub_agent", "s-1", "inv-1", None)
    )
    order: list[str] = []
    pending_seen: list[bool] = []
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def run(_input, *, stream: bool, **_kwargs: Unpack[AgentRunKwargs]):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise ConnectionError(f"failure {calls}")
        order.append("next pass")
        return AgentResponse(messages=[Message("assistant", ["done"])])

    controller = _make_controller(
        _agent(monkeypatch, run),
        bus,
        max_retries=1,
        run_kwargs={"options": {"store": True}},
        tool_event_middleware=events,
    )
    controller.policy._persist_dir = tmp_path
    original_reject = events.reject_hosted_attempt
    original_save = controller.policy._write_persisted
    original_prepare = controller.policy.prepare_retry

    async def reject(message: str) -> None:
        order.append("reject hosted")
        await original_reject(message)

    def save() -> None:
        original_save()
        order.append("pause-save")

    def prepare() -> tuple[str, ...]:
        order.append("retry input")
        return original_prepare()

    async def sleep(_seconds: int) -> bool:
        entered.set()
        await release.wait()
        order.append("sleep complete")
        return False

    async def retry_notice(_event: InvocationRetryAttempt) -> None:
        order.append("notice")

    async def pause(_event: InvocationPaused) -> None:
        order.append("pause event")
        pending_seen.append(controller.policy._persist_path().exists())
        controller.request_retry()

    monkeypatch.setattr(events, "reject_hosted_attempt", create_autospec(original_reject, side_effect=reject))
    monkeypatch.setattr(controller.policy, "_write_persisted", create_autospec(original_save, side_effect=save))
    monkeypatch.setattr(controller.policy, "prepare_retry", create_autospec(original_prepare, side_effect=prepare))
    monkeypatch.setattr(
        controller.policy,
        "_interruptible_sleep",
        create_autospec(controller.policy._interruptible_sleep, side_effect=sleep),
    )
    await bus.subscribe(InvocationRetryAttempt, retry_notice)
    await bus.subscribe(InvocationPaused, pause)
    task = asyncio.create_task(controller.run())
    try:
        await entered.wait()
        assert order == ["reject hosted", "notice"]
        release.set()
        assert await task == "done"
        assert order == [
            "reject hosted",
            "notice",
            "sleep complete",
            "reject hosted",
            "pause-save",
            "pause event",
            "retry input",
            "next pass",
        ]
        assert pending_seen == [True]
        assert calls == 3
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await bus.unsubscribe(InvocationRetryAttempt, retry_notice)
        await bus.unsubscribe(InvocationPaused, pause)


async def test_a_retry_announces_its_resume_then_withdraws_the_answer_it_dropped_before_the_next_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A details view carries the transcript on from the resume, then drops what the retry asks for again."""
    bus = EventBus()
    order: list[object] = []
    calls = 0

    async def run(_input, *, stream: bool, **_kwargs: Unpack[AgentRunKwargs]):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("failure")
        order.append("next pass")
        return AgentResponse(messages=[Message("assistant", ["done"])])

    controller = _make_controller(_agent(monkeypatch, run), bus, max_retries=0)
    controller.policy._persist_dir = tmp_path
    original_prepare = controller.policy.prepare_retry

    def prepare() -> tuple[str, ...]:
        assert original_prepare() == ()
        order.append("answer dropped")
        return ("segment-1",)

    async def pause(_event: InvocationPaused) -> None:
        controller.request_retry()

    async def resumed(_event: InvocationResumed) -> None:
        order.append("resumed")

    async def withdrawn(event: InvocationAnswerWithdrawn) -> None:
        order.append(("withdrawn", event.origin, event.segment_ids))

    monkeypatch.setattr(controller.policy, "prepare_retry", create_autospec(original_prepare, side_effect=prepare))
    await bus.subscribe(InvocationPaused, pause)
    await bus.subscribe(InvocationResumed, resumed)
    await bus.subscribe(InvocationAnswerWithdrawn, withdrawn)
    try:
        assert await asyncio.wait_for(controller.run(), ENGINE_TURN_TIMEOUT) == "done"
    finally:
        await bus.unsubscribe(InvocationPaused, pause)
        await bus.unsubscribe(InvocationResumed, resumed)
        await bus.unsubscribe(InvocationAnswerWithdrawn, withdrawn)

    assert order == ["resumed", "answer dropped", ("withdrawn", controller.origin, ("segment-1",)), "next pass"]


async def test_a_retried_child_withdraws_the_segment_its_kept_answer_was_shown_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stamp joins the text the child's view showed to what its Retry drops from history to ask again."""
    bus = EventBus()
    session = AgentSession()
    session.state["chrys_history"] = {"messages": [Message("user", ["do the thing"])]}
    deltas: list[InvocationContentDelta] = []
    withdrawals: list[InvocationAnswerWithdrawn] = []
    calls = 0

    async def failing():
        yield AgentResponseUpdate(role="assistant", contents=[Content.from_text("Partial answer")])
        # Reasoning after it completes the segment, so it shows and is kept.
        yield AgentResponseUpdate(role="assistant", contents=[Content.from_text_reasoning(text="hmm")])
        raise RuntimeError("response failed")

    async def answering():
        yield AgentResponseUpdate(role="assistant", contents=[Content.from_text("done")])

    def run(_input: Any, *, stream: bool, **_kwargs: Any) -> ResponseStream:
        nonlocal calls
        calls += 1
        return ResponseStream(failing() if calls == 1 else answering(), finalizer=AgentResponse.from_updates)

    controller = _make_controller(_agent(monkeypatch, run), bus, session=session, stream=True)

    async def retry(_event: InvocationPaused) -> None:
        controller.request_retry()

    await bus.subscribe(InvocationPaused, retry)
    await bus.subscribe(InvocationContentDelta, partial(collect_events, deltas))
    await bus.subscribe(InvocationAnswerWithdrawn, partial(collect_events, withdrawals))
    assert await asyncio.wait_for(controller.run(), ENGINE_TURN_TIMEOUT) == "done"

    (shown,) = {delta.segment_id for delta in deltas if delta.channel == "answer" and delta.text == "Partial answer"}
    (withdrawn,) = withdrawals
    assert (withdrawn.origin, withdrawn.segment_ids) == (controller.origin, (shown,))
    history = session.state["chrys_history"]["messages"]
    assert "Partial answer" not in [message.text for message in history]


@pytest.mark.parametrize("forced_stateless", [False, True], ids=["stored-client", "forced-stateless"])
@pytest.mark.parametrize("with_client_kwargs", [False, True], ids=["kwargs-absent", "kwargs-handles"])
@pytest.mark.parametrize("with_options", [False, True], ids=["options-absent", "options-handles"])
@pytest.mark.parametrize("live_token", [False, True], ids=["no-token", "live-token"])
def test_child_service_restore_clears_session_and_preserves_retry_inputs(
    forced_stateless: bool, with_client_kwargs: bool, with_options: bool, live_token: bool
) -> None:
    class RestoreClient(MockChatClient):
        STORES_BY_DEFAULT = True
        FORCES_STATELESS = forced_stateless

    run_kwargs: AgentRunKwargs = {}
    if with_options:
        run_kwargs["options"] = {
            "store": True,
            "conversation_id": "stale-options-conversation",
            "previous_response_id": "stale-options-response",
            "conversation": {"id": "stale-options-object"},
            "background": True,
            "extra_body": {
                "conversation_id": "stale-nested-options-conversation",
                "previous_response_id": "stale-nested-options-response",
                "conversation": {"id": "stale-nested-options-object"},
                "background": True,
                "keep": "options",
            },
        }
    if with_client_kwargs:
        run_kwargs["client_kwargs"] = {
            "conversation_id": "stale-kwargs-conversation",
            "previous_response_id": "stale-kwargs-response",
            "conversation": {"id": "stale-kwargs-object"},
            "background": True,
            "extra_body": {
                "conversation_id": "stale-nested-kwargs-conversation",
                "previous_response_id": "stale-nested-kwargs-response",
                "conversation": {"id": "stale-nested-kwargs-object"},
                "background": True,
                "keep": "kwargs",
            },
        }
    session = AgentSession()
    ctrl = kernel_shell(
        conversation=Conversation(),
        parent_origin=None,
        invocation_id="restore-invocation",
        tool_name="Explore",
        agent_name="Explore",
        agent=Agent(client=RestoreClient()),
        session=session,
        loop_recorder=LoopRecorder(),
        prompt="restore child inputs",
        run_kwargs=run_kwargs,
        event_bus=None,
    )
    assert ctrl.policy._service_storage is not forced_stateless
    assert (ctrl.policy._run_kwargs.get("options") is None) is not with_options
    token = {"response_id": "pending-child-response"}
    if live_token:
        # A live token materializes options through the real child observer,
        # including when the caller supplied no options initially.
        ctrl.policy.observe_continuation_token(token)
    raw_options = ctrl.policy._run_kwargs.get("options")
    session.service_session_id = "stale-child-session"

    ctrl.policy._attempts._restore_service_retry_inputs(ctrl.policy._run_kwargs)

    assert session.service_session_id is None
    expected_options: dict[str, Any] = {}
    if with_options:
        expected_options = {"store": True, "extra_body": {"keep": "options"}}
        if not forced_stateless:
            expected_options["background"] = True
            expected_options["extra_body"]["background"] = True
    if live_token and not forced_stateless:
        expected_options["continuation_token"] = token
    if raw_options is None:
        assert "options" not in ctrl.policy._run_kwargs
    else:
        assert ctrl.policy._run_kwargs["options"] == expected_options
        assert ctrl.policy._run_kwargs["options"] is not raw_options
    expected_client_kwargs: dict[str, Any] = {}
    if with_client_kwargs:
        expected_client_kwargs = {"extra_body": {"keep": "kwargs"}}
        if not forced_stateless:
            expected_client_kwargs["background"] = True
            expected_client_kwargs["extra_body"]["background"] = True
    assert "client_kwargs" in ctrl.policy._run_kwargs
    assert ctrl.policy._run_kwargs["client_kwargs"] == expected_client_kwargs


@pytest.mark.parametrize("recorded", [True, False], ids=["recorded-anchor", "bare-anchor"])
def test_child_retry_seed_replays_the_reminders_its_anchor_was_sent_with(recorded: bool) -> None:
    session = AgentSession()
    ctrl = kernel_shell(
        conversation=Conversation(),
        parent_origin=None,
        invocation_id="seed-invocation",
        tool_name="Explore",
        agent_name="Explore",
        agent=Agent(client=MockChatClient()),
        session=session,
        loop_recorder=LoopRecorder(),
        prompt="explore the tree",
        run_kwargs={},
        event_bus=None,
    )
    anchor = ctrl.policy._seed_input()[0]
    record = [{"kind": "turn", "text": "<system-reminder>\nsent context\n</system-reminder>"}]
    if recorded:
        anchor.additional_properties[HistoryMarkerKind.SYSTEM_REMINDERS_KEY] = record
    session.state["chrys_history"] = {"messages": [anchor]}

    assert ctrl.policy.prepare_retry() == ()

    assert session.state["chrys_history"]["messages"] == []
    [seed] = ctrl.policy._next_run_input
    assert seed is not anchor
    if recorded:
        assert seed.additional_properties[HistoryMarkerKind.SYSTEM_REMINDERS_KEY] == record
        assert seed.additional_properties[HistoryMarkerKind.SYSTEM_REMINDERS_KEY] is not record
    else:
        assert HistoryMarkerKind.SYSTEM_REMINDERS_KEY not in seed.additional_properties


async def test_a_paused_child_cancelled_later_keeps_its_partial_answer_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pause and the cancel both repair the failed pass; the answer it showed is kept by the first."""
    bus = EventBus()
    session = AgentSession()
    session.state["chrys_history"] = {"messages": [Message("user", ["do the thing"])]}
    paused = asyncio.Event()

    async def note_pause(_event: InvocationPaused) -> None:
        paused.set()

    await bus.subscribe(InvocationPaused, note_pause)

    async def updates():
        yield AgentResponseUpdate(role="assistant", contents=[Content.from_text("Partial answer")])
        # Reasoning after it completes the segment, so it shows; the one cut off after never does.
        yield AgentResponseUpdate(role="assistant", contents=[Content.from_text_reasoning(text="hmm")])
        yield AgentResponseUpdate(role="assistant", contents=[Content.from_text("Cut o")])
        raise RuntimeError("response failed")

    def stream_then_fail(_input: Any, *, stream: bool, **_kwargs: Any) -> ResponseStream:
        return ResponseStream(updates(), finalizer=AgentResponse.from_updates)

    controller = _make_controller(_agent(monkeypatch, stream_then_fail), bus, session=session, stream=True)

    def answers() -> list[str]:
        return [message.text for message in session.state["chrys_history"]["messages"] if message.role == "assistant"]

    task = asyncio.create_task(controller.run())
    try:
        await wait_for(lambda: paused.is_set() or task.done(), timeout=ENGINE_TURN_TIMEOUT, description="the pause")
        assert not task.done()
        assert answers() == ["Partial answer"]
        await controller.cascade_abort()
        await asyncio.gather(task, return_exceptions=True)
        assert answers() == ["Partial answer"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
