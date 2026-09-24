# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for persisted per-message creation timestamps."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationMessage, UserMessage
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.trajectory_timing import TRAJECTORY_TIMING_KEY
from chrys.foundation.util.time import parse_created_at, utc_iso
from chrys.kernel import AgentSession, ChatResponse, Content, Message, SessionContext
from chrys.orchestration.engine.assembly import assemble_agent_engine
from chrys.orchestration.engine.state.machine import Trigger
from chrys.service.agent_middleware.injection import ConsumedInjection, InjectionAnchor
from chrys.service.context.providers.history import CompressibleHistoryProvider
from chrys.service.llm.mock import MockResponse
from chrys.service.session.history import SessionHistoryManager
from chrys.service.session.message_metadata import (
    LAST_ASSISTANT_CREATED_AT_STATE_KEY,
    MESSAGE_CREATED_AT_KEY,
    normalize_created_at,
    stamp_message_response_timing,
    try_normalize_created_at,
)
from tests.orchestration.invoker._resume_policy import drive_retry_policy, make_resume_harness
from tests.support.loaded_agents import install_loaded_agent
from tests.support.pipeline_helpers import create_test_engine, wait_for_idle


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
async def test_session_messages_persist_created_at_for_user_and_assistant(tmp_path, stream: bool) -> None:
    """Fresh user timestamps come from the event; assistant timestamps come from history after_run."""
    ctx = await create_test_engine(
        [
            MockResponse(tool_calls=[("echo", "call-1", {"message": "hi"})]),
            MockResponse(text="done"),
        ],
        tmp_path,
        stream=stream,
    )
    try:
        user_ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        await ctx.bus.publish(UserMessage(text="hello", timestamp=user_ts))
        await wait_for_idle(ctx)

        state = await ctx.store.load_session(ctx.session_id)
        assert state is not None
        messages = state["messages"]

        user = next(m for m in messages if m.role == "user")
        assert user.additional_properties[MESSAGE_CREATED_AT_KEY] == utc_iso(user_ts)
        user_timing = user.additional_properties[TRAJECTORY_TIMING_KEY]
        assert user_timing == {
            "started_at": utc_iso(user_ts),
            "finished_at": utc_iso(user_ts),
            "duration_ms": 0,
        }

        assistants = [
            m for m in messages if m.role == "assistant" and not m.additional_properties.get(HistoryMarkerKind.KEY)
        ]
        assert len(assistants) == 2
        assistant_timestamps = {m.additional_properties.get(MESSAGE_CREATED_AT_KEY) for m in assistants}
        assert len(assistant_timestamps) == 1
        assert next(iter(assistant_timestamps))
        for assistant in assistants:
            timing = assistant.additional_properties[TRAJECTORY_TIMING_KEY]
            assert timing["finished_at"] == assistant.additional_properties[MESSAGE_CREATED_AT_KEY]
            assert isinstance(timing["duration_ms"], int)
            assert timing["duration_ms"] >= 0

        function_call = next(
            content for message in assistants for content in message.contents if content.type == "function_call"
        )
        function_result = next(
            content for message in messages for content in message.contents if content.type == "function_result"
        )
        tool_timing = function_call.additional_properties[TRAJECTORY_TIMING_KEY]
        assert tool_timing == function_result.additional_properties[TRAJECTORY_TIMING_KEY]
        assert tool_timing["started_at"] <= tool_timing["finished_at"]
        assert isinstance(tool_timing["duration_ms"], int)
        assert tool_timing["duration_ms"] >= 0
        final_events = [
            e
            for e in ctx.events
            if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final and not e.is_intermediate
        ]
        assert final_events
        assert utc_iso(final_events[-1].timestamp) == next(iter(assistant_timestamps))

        markers = [m for m in messages if m.additional_properties.get(HistoryMarkerKind.KEY)]
        assert markers
        assert all(MESSAGE_CREATED_AT_KEY not in m.additional_properties for m in markers)

        raw_messages = await ctx.store.load_session_raw(ctx.session_id)
        assert raw_messages is not None
        raw_user = next(m for m in raw_messages if m["role"] == "user")
        assert raw_user["additional_properties"][MESSAGE_CREATED_AT_KEY] == utc_iso(user_ts)
        assert raw_user["additional_properties"][TRAJECTORY_TIMING_KEY] == user_timing
        raw_assistants = [
            m
            for m in raw_messages
            if m["role"] == "assistant"
            and m.get("additional_properties", {}).get(HistoryMarkerKind.KEY) is None
            and m.get("contents")
        ]
        assert {m["additional_properties"].get(MESSAGE_CREATED_AT_KEY) for m in raw_assistants} == assistant_timestamps
        raw_function_call = next(
            content
            for message in raw_assistants
            for content in message["contents"]
            if content["type"] == "function_call"
        )
        raw_function_result = next(
            content
            for message in raw_messages
            for content in message["contents"]
            if content["type"] == "function_result"
        )
        assert raw_function_call["additional_properties"][TRAJECTORY_TIMING_KEY] == tool_timing
        assert raw_function_result["additional_properties"][TRAJECTORY_TIMING_KEY] == tool_timing
    finally:
        await ctx.cleanup()


def test_persist_consumed_injection_preserves_created_at() -> None:
    """Persisted injection messages keep the timestamp from the original user event."""
    anchor_message = Message("assistant", [Content.from_text("working")])
    history = SessionHistoryManager()
    history.bind({"messages": [Message("user", [Content.from_text("go")]), anchor_message]})

    created_at = datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)
    history.persist_consumed_injections(
        [
            ConsumedInjection(
                text="mid-run note",
                anchor=InjectionAnchor.from_message(anchor_message),
                created_at=created_at,
            )
        ]
    )

    injected = next(m for m in history.messages if m.additional_properties.get("_injected"))
    assert injected.text == "mid-run note"
    assert injected.additional_properties[MESSAGE_CREATED_AT_KEY] == utc_iso(created_at)
    assert injected.additional_properties[TRAJECTORY_TIMING_KEY] == {
        "started_at": utc_iso(created_at),
        "finished_at": utc_iso(created_at),
        "duration_ms": 0,
    }


def test_created_at_normalization_is_canonical() -> None:
    """Stored timestamps use one byte-stable UTC format across platforms."""
    value = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)

    assert utc_iso(value) == "2026-05-15T12:00:00.000000+00:00"
    assert parse_created_at("2026-05-15T12:00:00Z") == value
    assert normalize_created_at("2026-05-15T12:00:00Z") == "2026-05-15T12:00:00.000000+00:00"
    assert normalize_created_at("2026-05-15T12:00:00.123456+02:00") == "2026-05-15T10:00:00.123456+00:00"
    assert try_normalize_created_at("not-a-time") is None


@pytest.mark.asyncio
async def test_history_provider_keeps_measured_assistant_timing_as_final_event_timestamp() -> None:
    """Pre-stamped wire responses keep their exact completion time through after_run."""
    started_at = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 5, 15, 12, 0, 2, tzinfo=UTC)
    assistant = Message("assistant", [Content.from_text("done")])
    stamp_message_response_timing(
        assistant,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=2_000,
    )
    user = Message("user", [Content.from_text("go")])
    provider = CompressibleHistoryProvider()
    session = AgentSession(session_id="timed-response")
    context = SessionContext(session_id="timed-response", input_messages=[user])
    state: dict[str, Any] = {"messages": [], "compressed_msgs": []}

    await provider.before_run(agent=object(), session=session, context=context, state=state)
    context._response = ChatResponse(messages=[assistant])
    await provider.after_run(agent=object(), session=session, context=context, state=state)

    assert assistant.additional_properties[MESSAGE_CREATED_AT_KEY] == utc_iso(finished_at)
    assert assistant.additional_properties[TRAJECTORY_TIMING_KEY] == {
        "started_at": utc_iso(started_at),
        "finished_at": utc_iso(finished_at),
        "duration_ms": 2_000,
    }
    assert session.state[LAST_ASSISTANT_CREATED_AT_STATE_KEY] == utc_iso(finished_at)


@pytest.mark.asyncio
async def test_history_provider_ignores_invalid_and_structural_assistant_timestamps() -> None:
    """Only the last valid conversation assistant controls the final event timestamp."""
    started_at = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 5, 15, 12, 0, 2, tzinfo=UTC)
    valid = Message("assistant", [Content.from_text("done")])
    stamp_message_response_timing(
        valid,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=2_000,
    )
    later_at = datetime(2026, 5, 15, 12, 0, 3, tzinfo=UTC)
    datetime_stamped = Message(
        "assistant",
        [Content.from_text("later")],
        additional_properties={MESSAGE_CREATED_AT_KEY: later_at},
    )
    invalid = Message(
        "assistant",
        [Content.from_text("invalid")],
        additional_properties={MESSAGE_CREATED_AT_KEY: "not-a-time"},
    )
    marker = Message(
        "assistant",
        [Content.from_text("interrupted")],
        additional_properties={
            HistoryMarkerKind.KEY: "",
            MESSAGE_CREATED_AT_KEY: "2030-01-01T00:00:00+00:00",
        },
    )
    user = Message("user", [Content.from_text("go")])
    provider = CompressibleHistoryProvider()
    session = AgentSession(session_id="timed-response-tail")
    context = SessionContext(session_id="timed-response-tail", input_messages=[user])
    state: dict[str, Any] = {"messages": [], "compressed_msgs": []}

    await provider.before_run(agent=object(), session=session, context=context, state=state)
    context._response = ChatResponse(messages=[valid, datetime_stamped, invalid, marker])
    await provider.after_run(agent=object(), session=session, context=context, state=state)

    assert session.state[LAST_ASSISTANT_CREATED_AT_STATE_KEY] == utc_iso(later_at)
    assert datetime_stamped.additional_properties[TRAJECTORY_TIMING_KEY] == {
        "started_at": utc_iso(later_at),
        "finished_at": utc_iso(later_at),
        "duration_ms": 0,
    }
    assert TRAJECTORY_TIMING_KEY not in invalid.additional_properties
    assert TRAJECTORY_TIMING_KEY not in marker.additional_properties


@pytest.mark.asyncio
async def test_resume_replayed_legacy_user_message_stays_unstamped() -> None:
    """Retrying an old prompt with no timestamp must not invent one from the retry event."""
    messages = [Message("user", [Content.from_text("legacy prompt")])]
    executor = make_resume_harness(messages)

    retry_ts = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
    await drive_retry_policy(executor, created_at=retry_ts)

    replayed = executor.backend.run.call_args[0][0].messages[0]
    assert replayed.text == "legacy prompt"
    assert MESSAGE_CREATED_AT_KEY not in replayed.additional_properties
    assert TRAJECTORY_TIMING_KEY not in replayed.additional_properties


@pytest.mark.asyncio
async def test_resume_replayed_user_message_preserves_existing_created_at() -> None:
    """Retrying a prompt preserves its original timestamp when one exists."""
    original_ts = "2026-03-04T05:06:07Z"
    normalized_original_ts = "2026-03-04T05:06:07.000000+00:00"
    message = Message("user", [Content.from_text("original prompt")])
    message.additional_properties[MESSAGE_CREATED_AT_KEY] = original_ts
    executor = make_resume_harness([message])

    retry_ts = datetime(2026, 9, 10, 11, 12, 13, tzinfo=UTC)
    await drive_retry_policy(executor, created_at=retry_ts)

    replayed = executor.backend.run.call_args[0][0].messages[0]
    assert replayed.text == "original prompt"
    assert replayed.additional_properties[MESSAGE_CREATED_AT_KEY] == normalized_original_ts
    assert replayed.additional_properties[TRAJECTORY_TIMING_KEY] == {
        "started_at": normalized_original_ts,
        "finished_at": normalized_original_ts,
        "duration_ms": 0,
    }


@pytest.mark.asyncio
async def test_resume_replayed_invalid_created_at_stays_unstamped() -> None:
    """Retrying a prompt with invalid legacy metadata must not invent a timestamp."""
    message = Message("user", [Content.from_text("legacy prompt")])
    message.additional_properties[MESSAGE_CREATED_AT_KEY] = "not-a-time"
    executor = make_resume_harness([message])

    await drive_retry_policy(executor, created_at=datetime(2026, 9, 10, 11, 12, 13, tzinfo=UTC))

    replayed = executor.backend.run.call_args[0][0].messages[0]
    assert replayed.text == "legacy prompt"
    assert MESSAGE_CREATED_AT_KEY not in replayed.additional_properties
    assert TRAJECTORY_TIMING_KEY not in replayed.additional_properties


@pytest.mark.asyncio
async def test_post_run_backfills_interrupt_recovery_messages(*, engine_services) -> None:
    """Interrupted tool-loop messages that bypass provider after_run still get timestamps before save."""

    class _Executor:
        run_failed = False
        was_interrupted = True
        last_error = ""

        def drain_batch_records(self) -> list[object]:
            return []

        def drain_decisions(self) -> list[dict[str, str]]:
            return []

        @property
        def backend(self):
            return self

        @property
        def inputs(self):
            return self

        @property
        def state(self):
            return self

        @property
        def approval(self):
            return self

        @property
        def tool_events(self):
            return self

    class _Injection:
        def drain_pending(self) -> list[Any]:
            return []

    class _LoopRecorder:
        initial_count = None
        captured_count = None
        loop_messages = None

    engine = assemble_agent_engine(EventBus(), settings=Settings())
    install_loaded_agent(engine, bindings=_Executor())  # type: ignore[assignment]
    install_loaded_agent(engine, injection=_Injection())  # type: ignore[assignment]
    install_loaded_agent(engine, loop_recorder=_LoopRecorder())  # type: ignore[assignment]
    legacy_turn = Message("assistant", [Content.from_text("")])
    legacy_turn.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    engine_services(engine).history.bind(
        {
            "messages": [
                Message("user", [Content.from_text("legacy user")]),
                Message("assistant", [Content.from_text("legacy assistant")]),
                legacy_turn,
                Message("user", [Content.from_text("run a tool")]),
                Message("assistant", [Content.from_function_call("call-1", "echo", arguments={"message": "hi"})]),
                Message("tool", [Content.from_function_result("call-1", result="ok")]),
            ]
        }
    )
    engine.turns.turn_state.history_start_index = 3
    engine_services(engine).fsm.try_transition(Trigger.START)
    engine_services(engine).fsm.try_transition(Trigger.USER_MESSAGE)

    saved = False

    async def _save_current_session() -> None:
        nonlocal saved
        saved = True

    engine.writer.save_current_session = _save_current_session  # type: ignore[method-assign]

    await engine.turns.finalize_current_run()

    assert saved
    legacy_user, legacy_assistant, legacy_marker, user, assistant, tool, interrupted, turn = engine_services(
        engine
    ).history.messages
    assert MESSAGE_CREATED_AT_KEY not in legacy_user.additional_properties
    assert MESSAGE_CREATED_AT_KEY not in legacy_assistant.additional_properties
    assert MESSAGE_CREATED_AT_KEY not in legacy_marker.additional_properties
    assert user.additional_properties[MESSAGE_CREATED_AT_KEY]
    assert assistant.additional_properties[MESSAGE_CREATED_AT_KEY]
    assert MESSAGE_CREATED_AT_KEY not in tool.additional_properties
    assert MESSAGE_CREATED_AT_KEY not in interrupted.additional_properties
    assert MESSAGE_CREATED_AT_KEY not in turn.additional_properties


def test_backfill_missing_created_at_uses_turn_marker_when_index_is_stale() -> None:
    legacy = Message("assistant", [Content.from_text("legacy")])
    turn = Message("assistant", [Content.from_text("")])
    turn.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    current = Message("assistant", [Content.from_text("current")])
    history = SessionHistoryManager()
    history.bind({"messages": [legacy, turn, current]})

    stamped = history.backfill_missing_created_at(start_index=99)

    assert stamped == 1
    assert MESSAGE_CREATED_AT_KEY not in legacy.additional_properties
    assert MESSAGE_CREATED_AT_KEY not in turn.additional_properties
    assert current.additional_properties[MESSAGE_CREATED_AT_KEY]
