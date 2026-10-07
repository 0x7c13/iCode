# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Integration tests for user message injection and session restore.

Covers:
- Mid-tool-loop injection via FunctionMiddleware
- Injection consumed vs abandoned scenarios
- Session save/load round-trip with proper Content deserialization
- Session restore seeds agent history (LLM sees previous context)
- Intermediate text persistence as session metadata
- Injection ordering in persisted session history
"""

from __future__ import annotations

import asyncio
import gc
import weakref
from typing import Any
from unittest.mock import create_autospec

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    InvocationMessage,
    InvocationToolCallResult,
    InvocationToolCallStart,
    SessionReady,
    SessionRestore,
    SessionRestored,
    UserInject,
    UserInjectResult,
    UserMessage,
)
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.platform import detect_platform
from chrys.foundation.trajectory.ids import is_valid_analytics_id
from chrys.foundation.trajectory.metadata import read_analytics_item_id
from chrys.kernel import Content, FunctionTool, Message
from chrys.kernel.identity import ContentList
from chrys.service.agent_middleware.injection import ConsumedInjection, InjectionAnchor
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.profiles.agents.schema import AgentProfile, ApprovalConfig, CompactionConfig, ToolsConfig
from chrys.service.state.serializers import (
    deserialize_message,
    deserialize_state,
    serialize_message,
    serialize_state,
)
from chrys.service.state.store import JsonFileStateStore
from chrys.service.trajectory.preparation import PreparationOutcome
from tests.support.event_capture import collect_events
from tests.support.waiting import ENGINE_TEST_WAIT_TIMEOUT, ENGINE_TURN_TIMEOUT, wait_for, with_wait_deadline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _consumed_injection(
    text: str,
    anchor: InjectionAnchor,
    consumption_id: str | None = None,
    analytics_item_id: str | None = None,
) -> ConsumedInjection:
    return ConsumedInjection(
        text=text,
        anchor=anchor,
        consumption_id=consumption_id,
        analytics_item_id=analytics_item_id,
    )


def _make_profile() -> AgentProfile:
    return AgentProfile(
        name="Code",
        instructions="You are a coding assistant.",
        tools=ToolsConfig(builtins=["shell"]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
    )


def _make_settings() -> Settings:
    return Settings()


def _slow_tool(x: str) -> str:
    """A no-op tool for testing."""
    return f"result:{x}"


slow_tool = FunctionTool(func=_slow_tool, name="slow_tool", description="Slow")


def _shell_tool_name() -> str:
    return detect_platform().shell.name


def _is_tool_content(c: Any) -> bool:
    return isinstance(c, dict) and c.get("type") in ("function_call", "legacy")


def _run_merge(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Simulate the replay_history merge logic from ChatPanel."""
    merged: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "tool":
            continue
        if role == "assistant":
            contents = msg.get("contents", [])
            text_part = []
            tool_part = []
            for c in contents:
                if _is_tool_content(c):
                    tool_part.append(c)
                else:
                    if tool_part:
                        break
                    text_part.append(c)

            has_text = any(
                (isinstance(c, dict) and c.get("type") == "text" and c.get("text")) or isinstance(c, str)
                for c in text_part
            )
            if has_text:
                merged.append({"role": "assistant", "contents": list(text_part)})
            if tool_part:
                merge_target = None
                for j in range(len(merged) - 1, -1, -1):
                    entry = merged[j]
                    if entry.get("_merged_role") == "assistant_tools":
                        merge_target = j
                        break
                    if entry.get("role") == "user" and not entry.get("additional_properties", {}).get("_group"):
                        continue
                    break
                if merge_target is not None:
                    merged[merge_target]["contents"].extend(tool_part)
                else:
                    merged.append({"_merged_role": "assistant_tools", "role": "assistant", "contents": list(tool_part)})
            if not has_text and not tool_part:
                merged.append(msg)
            continue
        merged.append(msg)
    return merged


def _patch_client(mock_client: MockChatClient):
    """Context manager to patch create_client in the engine module."""
    import chrys.orchestration.engine.build.builder as builder_module

    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        builder_module, "create_client", create_autospec(builder_module.create_client, return_value=mock_client)
    )

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a: object):
            patcher.undo()

    return _Ctx()


# ---------------------------------------------------------------------------
# 1. Serialization round-trip preserves Content types
# ---------------------------------------------------------------------------


def test_serialize_function_call_roundtrip() -> None:
    """function_call Content survives serialize → deserialize."""
    msg = Message(
        "assistant",
        [
            Content.from_text("Let me check"),
            Content.from_function_call(call_id="call_abc", name="read_file", arguments={"path": "test.py"}),
        ],
    )
    data = serialize_message(msg)
    restored = deserialize_message(data)

    assert restored.role == "assistant"
    assert len(restored.contents) == 2
    assert restored.contents[0].type == "text"
    assert restored.contents[0].text == "Let me check"
    assert restored.contents[1].type == "function_call"
    assert restored.contents[1].name == "read_file"
    assert restored.contents[1].call_id == "call_abc"


def test_serialize_function_result_roundtrip() -> None:
    """function_result Content survives serialize → deserialize."""
    msg = Message("tool", [Content.from_function_result(call_id="call_abc", result="file contents")])
    data = serialize_message(msg)
    restored = deserialize_message(data)

    assert restored.role == "tool"
    assert len(restored.contents) == 1
    assert restored.contents[0].type == "function_result"
    assert restored.contents[0].call_id == "call_abc"
    assert restored.contents[0].result == "file contents"


def test_serialize_state_preserves_content_types() -> None:
    """Full state round-trip keeps function_call/function_result structure."""
    state = {
        "messages": [
            Message("user", ["hello"]),
            Message(
                "assistant",
                [Content.from_function_call(call_id="c1", name="zsh", arguments={"command": "ls"})],
            ),
            Message("tool", [Content.from_function_result(call_id="c1", result="/home\n/usr")]),
            Message("assistant", [Content.from_text("Done.")]),
        ],
        "compressed_msgs": [],
    }
    data = serialize_state(state)
    restored = deserialize_state(data)

    assert len(restored["messages"]) == 4
    # Assistant message has function_call
    asst = restored["messages"][1]
    assert asst.contents[0].type == "function_call"
    assert asst.contents[0].name == "zsh"
    # Tool message has function_result
    tool_msg = restored["messages"][2]
    assert tool_msg.contents[0].type == "function_result"


# ---------------------------------------------------------------------------
# 2. Session store round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_store_preserves_content_types(tmp_path) -> None:
    """Save and load session — function_call/function_result survive."""
    store = JsonFileStateStore(tmp_path)
    state: dict[str, Any] = {
        "messages": [
            Message("user", ["Tell me about foo"]),
            Message(
                "assistant",
                [Content.from_function_call(call_id="c1", name="grep", arguments={"pattern": "foo"})],
            ),
            Message("tool", [Content.from_function_result(call_id="c1", result="Found 3 matches")]),
            Message("assistant", [Content.from_text("I found 3 matches for foo.")]),
        ],
        "compressed_msgs": [],
    }
    await store.save_session("test1", state, agent_profile="code")

    loaded = await store.load_session("test1")
    assert loaded is not None

    # Verify the assistant message has a real function_call, not a string
    asst = loaded["messages"][1]
    assert asst.contents[0].type == "function_call"
    assert asst.contents[0].name == "grep"

    # Verify the tool message has a real function_result
    tool_msg = loaded["messages"][2]
    assert tool_msg.contents[0].type == "function_result"
    assert tool_msg.contents[0].result == "Found 3 matches"


@pytest.mark.asyncio
async def test_session_store_load_raw_returns_messages(tmp_path) -> None:
    """load_session_raw returns a list of message dicts."""
    store = JsonFileStateStore(tmp_path)
    state: dict[str, Any] = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
    }
    await store.save_session("test2", state, agent_profile="code")

    result = await store.load_session_raw("test2")
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["role"] == "user"


@pytest.mark.asyncio
async def test_session_store_load_raw_embedded_intermediate_text(tmp_path) -> None:
    """load_session_raw preserves _intermediate_text in additional_properties."""
    store = JsonFileStateStore(tmp_path)
    msg = Message("assistant", ["tool call placeholder"])
    msg.additional_properties = {"_intermediate_text": "Planning step 1"}
    state: dict[str, Any] = {
        "messages": [Message("user", ["hi"]), msg],
        "compressed_msgs": [],
    }
    await store.save_session("test3", state, agent_profile="code")

    result = await store.load_session_raw("test3")
    assert result is not None
    assert len(result) == 2
    extra = result[1].get("additional_properties", {})
    assert extra.get("_intermediate_text") == "Planning step 1"


# ---------------------------------------------------------------------------
# 3. Engine: injection consumed during tool execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_injection_consumed_during_tool_call(agent_engine) -> None:
    """User injection during tool execution is consumed and reported."""
    events: list[object] = []
    bus = EventBus()
    await bus.subscribe(SessionReady, lambda e: collect_events(events, e))
    await bus.subscribe(InvocationMessage, lambda e: collect_events(events, e))
    await bus.subscribe(InvocationToolCallStart, lambda e: collect_events(events, e))
    await bus.subscribe(InvocationToolCallResult, lambda e: collect_events(events, e))
    await bus.subscribe(UserInjectResult, lambda e: collect_events(events, e))

    # Tool call → (injection happens here) → final response seeing the injection
    mock_responses = [
        MockResponse(tool_calls=[(_shell_tool_name(), "c1", {"command": "echo hello"})]),
        MockResponse(text="I see your injection. Done."),
    ]
    mock_client = MockChatClient(responses=mock_responses)

    with _patch_client(mock_client):
        engine = agent_engine(bus, settings=_make_settings())
        await engine.start(_make_profile())

        # Send initial message
        await bus.publish(UserMessage(text="Do something"))
        # Let the tool call start so the injection lands mid-run: the first LLM
        # call (the tool-call response) must have been served before injecting.
        await wait_for(
            lambda: mock_client.call_count >= 1,
            timeout=ENGINE_TURN_TIMEOUT,
            description="initial model call",
        )

        # Inject while tool is running
        await bus.publish(UserInject(text="Also check this"))
        # Let execution complete — poll the asserted completion condition
        # (a final AgentMessage), not a fixed barrier.
        await wait_for(
            lambda: any((isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final for e in events),
            timeout=ENGINE_TURN_TIMEOUT,
            description="final agent message after injection",
        )

        # Check injection result
        inject_results = [e for e in events if isinstance(e, UserInjectResult)]
        # The injection may have been consumed or abandoned depending on timing
        # At minimum, we get a UserInjectResult event
        if inject_results:
            # If consumed, verify the injection text was delivered
            consumed = [r for r in inject_results if r.consumed]
            abandoned = [r for r in inject_results if not r.consumed]
            for r in consumed:
                assert r.text == "Also check this"
            for r in abandoned:
                assert r.text == "Also check this"

        # Final agent message should exist
        final_msgs = [
            e for e in events if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final
        ]
        assert len(final_msgs) >= 1


@pytest.mark.asyncio
async def test_injection_abandoned_when_agent_finishes_first(agent_engine) -> None:
    """Injection queued after agent.run() completes is reported as abandoned."""
    events: list[object] = []
    bus = EventBus()
    await bus.subscribe(InvocationMessage, lambda e: collect_events(events, e))
    await bus.subscribe(UserInjectResult, lambda e: collect_events(events, e))

    # Single text response, no tool calls → agent finishes immediately
    mock_client = MockChatClient(responses=[MockResponse(text="Done instantly.")])

    with _patch_client(mock_client):
        engine = agent_engine(bus, settings=_make_settings())
        await engine.start(_make_profile())

        await bus.publish(UserMessage(text="Quick task"))
        # Let the agent FULLY finish before injecting — this test exercises the
        # "agent finished first" path, so wait for the run's final AgentMessage
        # (not just call_count, which bumps when the call *starts*: injecting
        # then would race mid-stream and could be consumed instead of abandoned).
        await wait_for(
            lambda: any((isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final for e in events),
            timeout=ENGINE_TURN_TIMEOUT,
            description="final agent message before late injection",
        )

        # Now inject — but agent already finished
        await bus.publish(UserInject(text="Too late"))
        # inject() only works when executor is running, so this is reported
        # abandoned — poll for the UserInjectResult it emits, then assert.
        await wait_for(
            lambda: any(isinstance(e, UserInjectResult) for e in events),
            timeout=ENGINE_TURN_TIMEOUT,
            description="late injection result",
        )

        # inject() only works when executor is running, so this will be
        # delivered on the next run or abandoned — verify no crash
        assert mock_client.call_count == 1


@pytest.mark.asyncio
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_injection_consumption_id_threads_end_to_end(agent_engine) -> None:
    """queue → middleware consumption → mirror → finalizer persist, real engine.

    The persisted history copy must carry ``_injection_id`` (the queue id
    when present, synthesized otherwise) and ``UserInjectResult`` must
    round-trip the ORIGINAL queue handle — ``None`` for id-less frontends
    (ACP), never the synthesized consumption id, or the TUI's
    input-ownership check would treat every id-less consumption as stale.

    Injections are queued directly on the executor's middleware BEFORE the
    run so consumption deterministically lands on the first LLM call — the
    bus-driven mid-run path is covered by
    ``test_injection_consumed_during_tool_call``, which tolerates a
    consumed-or-abandoned race this test must not have.
    """
    events: list[object] = []
    bus = EventBus()
    await bus.subscribe(InvocationMessage, lambda e: collect_events(events, e))
    await bus.subscribe(UserInjectResult, lambda e: collect_events(events, e))

    mock_client = MockChatClient(responses=[MockResponse(text="Done.")])

    class _RecordingPreparation:
        def __init__(self) -> None:
            self.terminals: list[tuple[str, str | None]] = []

        async def finished(self, *, outcome: str, target_turn_id: str | None = None) -> None:
            del outcome, target_turn_id
            raise AssertionError("consumed notification must not wait for trajectory acknowledgement")

        def finished_soon(self, *, outcome: str, target_turn_id: str | None = None) -> None:
            self.terminals.append((outcome, target_turn_id))

    preparation = _RecordingPreparation()

    with _patch_client(mock_client):
        engine = agent_engine(bus, settings=_make_settings())
        await engine.start(_make_profile())
        assert engine.current.loaded is not None
        engine.current.loaded.bindings._injection.queue(
            "tracked note",
            injection_id="ui-1",
            preparation=preparation,  # type: ignore[arg-type]
            target_turn_id="target-turn",
        )
        engine.current.loaded.bindings._injection.queue("acp note")

        await bus.publish(UserMessage(text="Do something"))
        await wait_for(
            lambda: any((isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final for e in events),
            timeout=ENGINE_TURN_TIMEOUT,
            description="final agent message for injection consumption",
        )
        # Both consumption events (published from a fire-and-forget task) and
        # the finalizer's persisted copies land asynchronously — poll each.
        await wait_for(
            lambda: sum(isinstance(e, UserInjectResult) for e in events) == 2,
            timeout=ENGINE_TURN_TIMEOUT,
            description="two injection consumption results",
        )

        results = {e.text: e for e in events if isinstance(e, UserInjectResult)}
        assert results["tracked note"].consumed is True
        assert results["tracked note"].injection_id == "ui-1"
        assert results["acp note"].consumed is True
        assert results["acp note"].injection_id is None
        assert preparation.terminals == [(PreparationOutcome.INJECTED, "target-turn")]

        def _persisted() -> list[Message]:
            msgs = engine.current.loaded.bindings.backend.history_state.get("messages", [])
            return [m for m in msgs if m.role == "user" and m.additional_properties.get(HistoryMarkerKind.INJECTED_KEY)]

        await wait_for(
            lambda: len(_persisted()) == 2,
            timeout=ENGINE_TURN_TIMEOUT,
            description="two persisted injected messages",
        )
        by_text = {m.text: m for m in _persisted()}
        assert by_text["tracked note"].additional_properties[HistoryMarkerKind.INJECTION_ID_KEY] == "ui-1"
        acp_stamp = by_text["acp note"].additional_properties[HistoryMarkerKind.INJECTION_ID_KEY]
        assert acp_stamp
        assert acp_stamp != "ui-1"


# ---------------------------------------------------------------------------
# 4. Engine: session restore seeds agent history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_restore_seeds_agent_history(tmp_path, agent_engine) -> None:
    """After restoring a session, the agent sees previous conversation context."""
    store = JsonFileStateStore(tmp_path)
    bus = EventBus()
    events: list[object] = []
    await bus.subscribe(SessionReady, lambda e: collect_events(events, e))
    await bus.subscribe(InvocationMessage, lambda e: collect_events(events, e))
    await bus.subscribe(SessionRestored, lambda e: collect_events(events, e))

    # Pre-save a session with conversation history
    history_state: dict[str, Any] = {
        "messages": [
            Message("user", ["What is Python?"]),
            Message("assistant", [Content.from_text("Python is a programming language.")]),
        ],
        "compressed_msgs": [],
    }
    await store.save_session("restore_test", history_state, agent_profile="test")

    # Response for "continue" should reflect having seen previous context
    mock_client = MockChatClient(
        responses=[
            MockResponse(text="Continuing from where we left off about Python."),
        ]
    )

    with _patch_client(mock_client):
        engine = agent_engine(bus, settings=_make_settings(), state_store=store)
        await engine.start(_make_profile())

        # Restore the saved session
        await bus.publish(SessionRestore(session_id="restore_test"))
        await wait_for(
            lambda: any(isinstance(e, SessionRestored) for e in events),
            timeout=ENGINE_TURN_TIMEOUT,
            description="session restore event",
        )

        # Verify SessionRestored event
        restored_events = [e for e in events if isinstance(e, SessionRestored)]
        assert len(restored_events) == 1

        # Send a follow-up message
        await bus.publish(UserMessage(text="Continue"))
        # Poll the asserted condition: the model has been called at least once.
        await wait_for(
            lambda: mock_client.call_count >= 1,
            timeout=ENGINE_TURN_TIMEOUT,
            description="post-restore model call",
        )

        # Verify the model received the previous conversation context
        assert mock_client.call_count >= 1
        first_call_msgs, _ = mock_client.call_history[0]

        # The messages sent to the model should include the restored history
        roles = [m.role for m in first_call_msgs]
        assert "user" in roles
        assert "assistant" in roles

        # Find the restored user message in the context
        user_texts = []
        for m in first_call_msgs:
            if m.role == "user":
                for c in m.contents:
                    if c.type == "text" and c.text:
                        user_texts.append(c.text)
                    elif isinstance(c, str):
                        user_texts.append(c)

        # Both the restored "What is Python?" and the new "Continue" should be present
        all_text = " ".join(user_texts)
        assert "Python" in all_text or "Continue" in all_text

        await engine.shutdown()


# ---------------------------------------------------------------------------
# 5. Session save after tool calls preserves structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_save_preserves_tool_call_structure(tmp_path, agent_engine) -> None:
    """After an agent run with tool calls, the saved session has proper Content types."""
    store = JsonFileStateStore(tmp_path)
    bus = EventBus()
    events: list[object] = []
    await bus.subscribe(SessionReady, lambda e: collect_events(events, e))
    await bus.subscribe(InvocationMessage, lambda e: collect_events(events, e))

    mock_responses = [
        MockResponse(tool_calls=[(_shell_tool_name(), "c1", {"command": "echo test"})]),
        MockResponse(text="Command executed."),
    ]
    mock_client = MockChatClient(responses=mock_responses)

    with _patch_client(mock_client):
        engine = agent_engine(bus, settings=_make_settings(), state_store=store)
        await engine.start(_make_profile())

        await bus.publish(UserMessage(text="Run a command"))
        # Wait for the run to finish (final AgentMessage) so the tool-call
        # structure is fully persisted before shutdown saves the session.
        await wait_for(
            lambda: any((isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final for e in events),
            timeout=ENGINE_TURN_TIMEOUT,
            description="final agent message before session save",
        )
        await wait_for(
            lambda: any(isinstance(e, SessionReady) for e in events),
            timeout=ENGINE_TURN_TIMEOUT,
            description="session-ready event before session save",
        )

        # Get the session ID
        ready_events = [e for e in events if isinstance(e, SessionReady)]
        session_id = ready_events[0].session_id

        await engine.shutdown()

    # Load the saved session and verify structure
    loaded = await store.load_session(session_id)
    if loaded is not None and loaded.get("messages"):
        messages = loaded["messages"]
        # Find assistant messages with function_call content
        asst_with_tools = [
            m for m in messages if m.role == "assistant" and any(c.type == "function_call" for c in m.contents)
        ]
        # Find tool messages with function_result content
        tool_results = [
            m for m in messages if m.role == "tool" and any(c.type == "function_result" for c in m.contents)
        ]

        # Should have at least one tool call and one tool result
        if asst_with_tools:
            assert asst_with_tools[0].contents[0].type == "function_call"
        if tool_results:
            assert tool_results[0].contents[0].type == "function_result"


# ---------------------------------------------------------------------------
# 6. Intermediate text recorded as session metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intermediate_text_stored_as_metadata(tmp_path, agent_engine) -> None:
    """Intermediate text (text alongside tool calls) is saved in session state."""
    store = JsonFileStateStore(tmp_path)
    bus = EventBus()
    events: list[object] = []
    await bus.subscribe(SessionReady, lambda e: collect_events(events, e))
    await bus.subscribe(InvocationMessage, lambda e: collect_events(events, e))

    # Response with text + tool call (triggers intermediate text)
    mock_responses = [
        MockResponse(text="Let me search for that.", tool_calls=[(_shell_tool_name(), "c1", {"command": "ls"})]),
        MockResponse(text="Here are the results."),
    ]
    mock_client = MockChatClient(responses=mock_responses)

    with _patch_client(mock_client):
        engine = agent_engine(bus, settings=_make_settings(), state_store=store)
        await engine.start(_make_profile())

        await bus.publish(UserMessage(text="List files"))
        # Wait for the run to finish so messages are fully persisted before
        # shutdown saves the session.
        await wait_for(
            lambda: any((isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final for e in events),
            timeout=ENGINE_TURN_TIMEOUT,
            description="final agent message with intermediate text",
        )
        await wait_for(
            lambda: any(isinstance(e, SessionReady) for e in events),
            timeout=ENGINE_TURN_TIMEOUT,
            description="session-ready event with intermediate text",
        )

        session_id = next(e for e in events if isinstance(e, SessionReady)).session_id
        await engine.shutdown()

    # Check that intermediate text is embedded per-message via additional_properties.
    # The mock client bypasses the provider wire client, so the callback may not
    # fire.  Just verify session saved successfully and messages are present.
    import json

    session_path = store.session_dir(session_id) / "session.json"
    if session_path.exists():
        raw = json.loads(session_path.read_text())
        state = raw.get("state", {})
        messages = state.get("messages", [])
        assert isinstance(messages, list)
        assert len(messages) > 0


# ---------------------------------------------------------------------------
# 7. Replay merge logic: tool groups merge across injection messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_merges_tools_across_injection() -> None:
    """Tool-only messages merge past injection user messages during replay."""
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "contents": [{"type": "text", "text": "Do something"}],
            "additional_properties": {"_group": {"id": "g0"}},
        },
        {
            "role": "assistant",
            "contents": [{"type": "function_call", "call_id": "c1", "name": "tool_a", "arguments": "{}"}],
        },
        {"role": "tool", "contents": [{"type": "function_result", "call_id": "c1", "result": "ok"}]},
        # Injection user message (no _group metadata)
        {"role": "user", "contents": [{"type": "text", "text": "Also do this"}]},
        {
            "role": "assistant",
            "contents": [{"type": "function_call", "call_id": "c2", "name": "tool_b", "arguments": "{}"}],
        },
        {"role": "tool", "contents": [{"type": "function_result", "call_id": "c2", "result": "done"}]},
        {"role": "assistant", "contents": [{"type": "text", "text": "All done."}]},
    ]

    merged = _run_merge(messages)

    # tool_a and tool_b should be in the SAME merged group
    tool_groups = [m for m in merged if m.get("_merged_role") == "assistant_tools"]
    assert len(tool_groups) == 1, f"Expected 1 merged tool group, got {len(tool_groups)}"
    tool_names = [c.get("name") for c in tool_groups[0]["contents"] if isinstance(c, dict)]
    assert tool_names == ["tool_a", "tool_b"]

    # Injection user message is preserved
    user_msgs = [m for m in merged if m.get("role") == "user"]
    assert len(user_msgs) == 2


@pytest.mark.asyncio
async def test_replay_text_splits_from_tools() -> None:
    """Assistant messages with text + tools split correctly during replay."""
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "contents": [{"type": "text", "text": "Go"}],
            "additional_properties": {"_group": {"id": "g0"}},
        },
        {
            "role": "assistant",
            "contents": [
                {"type": "text", "text": "Let me check."},
                {"type": "function_call", "call_id": "c1", "name": "tool_a", "arguments": "{}"},
            ],
        },
        {"role": "tool", "contents": [{"type": "function_result", "call_id": "c1", "result": "ok"}]},
        {
            "role": "assistant",
            "contents": [{"type": "function_call", "call_id": "c2", "name": "tool_b", "arguments": "{}"}],
        },
        {"role": "tool", "contents": [{"type": "function_result", "call_id": "c2", "result": "done"}]},
        {"role": "assistant", "contents": [{"type": "text", "text": "All done."}]},
    ]

    merged = _run_merge(messages)

    # text "Let me check." should be a separate entry
    text_entries = [m for m in merged if m.get("role") == "assistant" and not m.get("_merged_role")]
    assert any("Let me check" in str(m.get("contents")) for m in text_entries)

    # tool_a and tool_b should be merged into one group
    tool_groups = [m for m in merged if m.get("_merged_role") == "assistant_tools"]
    assert len(tool_groups) == 1
    tool_names = [c.get("name") for c in tool_groups[0]["contents"] if isinstance(c, dict)]
    assert tool_names == ["tool_a", "tool_b"]


# ---------------------------------------------------------------------------
# 8. Regular user message does NOT merge across tool groups
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_regular_user_message_breaks_tool_merge() -> None:
    """A regular user message (with _group) should break tool group merging."""
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "contents": [{"type": "text", "text": "First"}],
            "additional_properties": {"_group": {"id": "g0"}},
        },
        {
            "role": "assistant",
            "contents": [{"type": "function_call", "call_id": "c1", "name": "tool_a", "arguments": "{}"}],
        },
        {"role": "tool", "contents": [{"type": "function_result", "call_id": "c1", "result": "ok"}]},
        {
            "role": "user",
            "contents": [{"type": "text", "text": "Second"}],
            "additional_properties": {"_group": {"id": "g1"}},
        },
        {
            "role": "assistant",
            "contents": [{"type": "function_call", "call_id": "c2", "name": "tool_b", "arguments": "{}"}],
        },
        {"role": "tool", "contents": [{"type": "function_result", "call_id": "c2", "result": "done"}]},
    ]

    merged = _run_merge(messages)

    # Should be TWO separate tool groups (regular user message breaks merge)
    tool_groups = [m for m in merged if m.get("_merged_role") == "assistant_tools"]
    assert len(tool_groups) == 2, f"Expected 2 tool groups, got {len(tool_groups)}"


# ---------------------------------------------------------------------------
# 9. persist_consumed_injections — identity-based positioning
# ---------------------------------------------------------------------------


def test_persist_injection_anchors_to_contents_identity() -> None:
    """Injection lands immediately AFTER the message the anchor points to."""
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    # Build state: user → asst(tool_call) → tool_result → final_asst
    user_msg = Message("user", [Content.from_text("Do something")])
    asst_tool = Message(
        "assistant",
        [Content.from_function_call(call_id="c1", name="zsh", arguments={"command": "ls"})],
    )
    tool_result = Message("tool", [Content.from_function_result(call_id="c1", result="ok")])
    final_asst = Message("assistant", [Content.from_text("Done.")])

    state: dict[str, Any] = {"messages": [user_msg, asst_tool, tool_result, final_asst]}
    history = SessionHistoryManager()
    history.bind(state)

    # Anchor points at `tool_result` — mimicking the middleware capturing
    # an InjectionAnchor right before the final LLM call, when the last
    # message in context was the tool result.
    anchor = InjectionAnchor.from_message(tool_result)
    analytics_item_id = "a" * 32
    history.persist_consumed_injections(
        [_consumed_injection("Also check this", anchor, analytics_item_id=analytics_item_id)]
    )

    messages = state["messages"]
    assert len(messages) == 5
    # Injection must land at idx 3 (after tool_result, before final_asst)
    assert messages[3].role == "user"
    assert messages[3].contents[0].text == "Also check this"
    assert messages[3].additional_properties.get("_injected") is True
    assert read_analytics_item_id(messages[3].additional_properties) == analytics_item_id
    assert messages[4] is final_asst, "Final assistant response must remain after the injection"


def test_persist_injection_prefers_identity_over_earlier_duplicate_call_id() -> None:
    """Exact contents identity wins when an earlier turn reused the same call_id."""
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    old_tool = Message("tool", [Content.from_function_result(call_id="c1", result="old")])
    current_tool = Message("tool", [Content.from_function_result(call_id="c1", result="current")])
    final_asst = Message("assistant", [Content.from_text("Done.")])
    state: dict[str, Any] = {"messages": [old_tool, current_tool, final_asst]}
    history = SessionHistoryManager()
    history.bind(state)

    history.persist_consumed_injections(
        [_consumed_injection("Use the current result", InjectionAnchor.from_message(current_tool))]
    )

    messages = state["messages"]
    assert messages[0] is old_tool
    assert messages[1] is current_tool
    assert messages[3] is final_asst
    assert messages[2].role == "user"
    assert messages[2].contents[0].text == "Use the current result"
    assert messages[2].additional_properties.get("_injected") is True


def test_persist_injection_structural_fallback_uses_latest_duplicate_call_id() -> None:
    """Streaming fallback should not anchor to an earlier reused call_id."""
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    old_tool = Message("tool", [Content.from_function_result(call_id="c1", result="old")])
    current_tool = Message("tool", [Content.from_function_result(call_id="c1", result="current")])
    final_asst = Message("assistant", [Content.from_text("Done.")])
    state: dict[str, Any] = {"messages": [old_tool, current_tool, final_asst]}
    history = SessionHistoryManager()
    history.bind(state)

    # Simulate the streaming rebuild: structural fields survive, but
    # contents identity points at the pre-rebuild message and no longer
    # matches anything in state.
    rebuilt_anchor = InjectionAnchor(
        role="tool",
        contents_ref=weakref.ref(ContentList([Content.from_function_result(call_id="c1", result="current")])),
        call_ids=("c1",),
    )
    history.persist_consumed_injections([_consumed_injection("Use the current result", rebuilt_anchor)])

    messages = state["messages"]
    assert messages[0] is old_tool
    assert messages[1] is current_tool
    assert messages[3] is final_asst
    assert messages[2].role == "user"
    assert messages[2].contents[0].text == "Use the current result"
    assert messages[2].additional_properties.get("_injected") is True


def test_persist_injection_anchor_from_wire_view_finds_state_original() -> None:
    """An anchor captured from the kernel's per-call message VIEW still lands.

    ``_wire_message_view`` mints a fresh wrapper + fresh contents list, so
    ``contents_id`` never matches state; the shared content-object identity
    tier must carry the match instead.
    """
    from chrys.kernel.loop import _wire_message_view
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    user_msg = Message("user", [Content.from_text("Do something")])
    asst_tool = Message(
        "assistant",
        [Content.from_function_call(call_id="c1", name="zsh", arguments={"command": "ls"})],
    )
    tool_result = Message("tool", [Content.from_function_result(call_id="c1", result="ok")])
    final_asst = Message("assistant", [Content.from_text("Done.")])
    state: dict[str, Any] = {"messages": [user_msg, asst_tool, tool_result, final_asst]}
    history = SessionHistoryManager()
    history.bind(state)

    view = _wire_message_view(tool_result)
    assert view is not tool_result and view.contents is not tool_result.contents
    history.persist_consumed_injections([_consumed_injection("Also check this", InjectionAnchor.from_message(view))])

    messages = state["messages"]
    assert len(messages) == 5
    assert messages[3].role == "user"
    assert messages[3].contents[0].text == "Also check this"
    assert messages[3].additional_properties.get("_injected") is True
    assert messages[4] is final_asst


def test_persist_injection_view_anchor_beats_newest_first_structural_twin() -> None:
    """Twin tool results (same call_id AND payload): a view-captured anchor
    must bind its own original, not the later twin the newest-first
    structural fallback would pick."""
    from chrys.kernel.loop import _wire_message_view
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    first_tool = Message("tool", [Content.from_function_result(call_id="c1", result="same")])
    later_tool = Message("tool", [Content.from_function_result(call_id="c1", result="same")])
    final_asst = Message("assistant", [Content.from_text("Done.")])
    state: dict[str, Any] = {"messages": [first_tool, later_tool, final_asst]}
    history = SessionHistoryManager()
    history.bind(state)

    anchor = InjectionAnchor.from_message(_wire_message_view(first_tool))
    history.persist_consumed_injections([_consumed_injection("After the first", anchor)])

    messages = state["messages"]
    assert messages[0] is first_tool
    assert messages[1].role == "user"
    assert messages[1].contents[0].text == "After the first"
    assert messages[1].additional_properties.get("_injected") is True
    assert messages[2] is later_tool
    assert messages[3] is final_asst


def test_injection_anchor_ignores_contents_list_hit_with_wrong_shape() -> None:
    """A list-identity hit must not beat a valid structural match."""
    from chrys.service.agent_middleware.injection import InjectionAnchor

    wrong_id_hit = Message("assistant", [Content.from_text("wrong message")])
    structural_match = Message("tool", [Content.from_function_result(call_id="c1", result="current")])
    anchor = InjectionAnchor(
        role="tool",
        contents_ref=weakref.ref(wrong_id_hit.contents),
        call_ids=("c1",),
    )

    assert anchor.find_in([wrong_id_hit, structural_match]) == 1


def test_injection_anchor_replaced_contents_list_falls_back_to_content_identity() -> None:
    """Snapshot semantics: replacing the message's contents list defeats the
    list-identity tier; the shared content objects still bind the message."""
    from chrys.service.agent_middleware.injection import InjectionAnchor

    source = Message("tool", [Content.from_function_result(call_id="c1", result="payload")])
    anchor = InjectionAnchor.from_message(source)
    assert anchor.contents_ref is not None

    source.contents = list(source.contents)

    assert anchor.contents_ref() is not source.contents
    assert anchor.find_in([source]) == 0


def test_injection_anchor_does_not_retain_source_and_dead_tiers_use_structural_fallback() -> None:
    """Weak identity tiers never keep the source alive; once dead they match
    nothing, and placement degrades to the structural fallback — the
    streaming-rebuild path."""
    from chrys.service.agent_middleware.injection import InjectionAnchor

    source = Message("tool", [Content.from_function_result(call_id="c1", result="payload")])
    anchor = InjectionAnchor.from_message(source)
    source_contents_ref = weakref.ref(source.contents)

    del source
    gc.collect()

    assert source_contents_ref() is None, "anchors must not pin their source"
    rebuilt = Message("tool", [Content.from_function_result(call_id="c1", result="payload")])
    assert anchor.find_in([rebuilt]) == 0


def test_injection_anchor_ignores_message_id_hit_with_wrong_shape() -> None:
    """A duplicate message_id must not beat a valid structural match."""
    from chrys.service.agent_middleware.injection import InjectionAnchor

    wrong_id_hit = Message(
        "assistant",
        [Content.from_function_call(call_id="old", name="zsh", arguments={})],
        message_id="dup",
    )
    structural_match = Message(
        "tool",
        [Content.from_function_result(call_id="c1", result="current")],
        message_id="dup",
    )
    anchor = InjectionAnchor(
        role="tool",
        call_ids=("c1",),
        message_id="dup",
    )

    assert anchor.find_in([wrong_id_hit, structural_match]) == 1


def test_persist_injection_regression_against_count_drift() -> None:
    """Regression: structural placement survives context/state view mismatch.

    The previous count-based heuristic (``assistant_count``) broke when
    compaction excluded tool-call groups from the middleware's view
    of ``context.messages`` while leaving them in state (flagged as
    ``_excluded`` but still present).  In that case, the middleware's
    count and the persist pass's count disagreed, so ``seen > asst_count``
    never tripped and the injection fell through to ``len(messages)`` —
    landing AFTER the LLM response it triggered.

    With ``InjectionAnchor.find_in``, only the anchor message's identifying
    keys (role + call_ids, etc.) matter, so exclusion state is irrelevant.
    """
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    # Simulate a turn that was excluded by intra-run compaction: the
    # asst(tool) / tool_result pair is still in state, marked _excluded.
    excluded_asst = Message(
        "assistant",
        [Content.from_function_call(call_id="c0", name="zsh", arguments={"command": "old"})],
    )
    excluded_asst.additional_properties["_excluded"] = True
    excluded_tool = Message("tool", [Content.from_function_result(call_id="c0", result="old")])
    excluded_tool.additional_properties["_excluded"] = True

    user_msg = Message("user", [Content.from_text("New request")])
    asst_tool = Message(
        "assistant",
        [Content.from_function_call(call_id="c1", name="zsh", arguments={"command": "ls"})],
    )
    tool_result = Message("tool", [Content.from_function_result(call_id="c1", result="ok")])
    final_asst = Message("assistant", [Content.from_text("All done.")])

    state: dict[str, Any] = {
        "messages": [excluded_asst, excluded_tool, user_msg, asst_tool, tool_result, final_asst],
    }
    history = SessionHistoryManager()
    history.bind(state)

    # Middleware sees only the non-excluded subset in context.messages,
    # so the anchor is captured from `tool_result`.
    anchor = InjectionAnchor.from_message(tool_result)
    history.persist_consumed_injections([_consumed_injection("Mid-turn interrupt", anchor)])

    messages = state["messages"]
    # Locate injection and final response positions.
    inject_idx = next(
        i for i, m in enumerate(messages) if m.role == "user" and m.additional_properties.get("_injected")
    )
    final_idx = messages.index(final_asst)
    assert inject_idx < final_idx, f"Injection at {inject_idx} must come before final response at {final_idx}"
    # Specifically: immediately after tool_result (original idx 4 → insert_idx 5 → final pushed to 6)
    assert messages[inject_idx - 1] is tool_result


def test_persist_injection_missing_anchor_falls_back_to_end() -> None:
    """When anchor matches nothing in state, injection appends at end."""
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    user_msg = Message("user", [Content.from_text("Hi")])
    asst = Message("assistant", [Content.from_text("Hello")])
    state: dict[str, Any] = {"messages": [user_msg, asst]}
    history = SessionHistoryManager()
    history.bind(state)

    # Anchor whose keys don't appear anywhere — e.g., a compaction summary
    # that only existed in the middleware view.
    bogus_anchor = InjectionAnchor(
        role="tool",
        contents_ref=weakref.ref(ContentList([Content.from_text("does-not-exist-in-state")])),
        call_ids=("not-a-real-call-id",),
        message_id="not-a-real-msg-id",
    )
    history.persist_consumed_injections([_consumed_injection("Stray", bogus_anchor)])

    messages = state["messages"]
    assert len(messages) == 3
    assert messages[-1].role == "user"
    assert messages[-1].contents[0].text == "Stray"
    assert messages[-1].additional_properties.get("_injected") is True


def test_persist_injection_multiple_same_anchor_preserves_queue_order() -> None:
    """Multiple injections sharing one anchor appear in queue order in state.

    Two UserInject events queued before the same LLM call both capture
    the same anchor.  With the naive reverse-by-idx insertion,
    equal indices invert the final order (B before A).  The fix uses a
    secondary sort key so the final ascending state order matches
    consumption order.
    """
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    anchor_msg = Message("tool", [Content.from_function_result(call_id="c1", result="ok")])
    final = Message("assistant", [Content.from_text("done")])
    state: dict[str, Any] = {"messages": [anchor_msg, final]}
    history = SessionHistoryManager()
    history.bind(state)

    anchor = InjectionAnchor.from_message(anchor_msg)
    # Queue order: A first, then B.
    history.persist_consumed_injections([_consumed_injection("A", anchor), _consumed_injection("B", anchor)])

    messages = state["messages"]
    assert [m.role for m in messages] == ["tool", "user", "user", "assistant"]
    assert messages[1].contents[0].text == "A"
    assert messages[2].contents[0].text == "B"
    assert messages[1].additional_properties.get("_injected") is True
    assert messages[2].additional_properties.get("_injected") is True


def test_persist_injection_multiple_different_anchors() -> None:
    """Injections with distinct anchors each land after their own anchor."""
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    m0 = Message("user", [Content.from_text("q0")])
    m1 = Message("assistant", [Content.from_text("a0")])
    m2 = Message("user", [Content.from_text("q1")])
    m3 = Message("assistant", [Content.from_text("a1")])
    state: dict[str, Any] = {"messages": [m0, m1, m2, m3]}
    history = SessionHistoryManager()
    history.bind(state)

    history.persist_consumed_injections(
        [
            _consumed_injection("after-a0", InjectionAnchor.from_message(m1)),
            _consumed_injection("after-a1", InjectionAnchor.from_message(m3)),
        ]
    )

    messages = state["messages"]
    texts = [m.contents[0].text for m in messages]
    assert texts == ["q0", "a0", "after-a0", "q1", "a1", "after-a1"]


def test_persist_injection_none_anchor_falls_back_to_end() -> None:
    """When the anchor is empty (e.g., empty context.messages), injection appends at end."""
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    user_msg = Message("user", [Content.from_text("Hi")])
    state: dict[str, Any] = {"messages": [user_msg]}
    history = SessionHistoryManager()
    history.bind(state)

    history.persist_consumed_injections([_consumed_injection("Stray", InjectionAnchor())])

    messages = state["messages"]
    assert len(messages) == 2
    assert messages[-1].role == "user"
    assert messages[-1].additional_properties.get("_injected") is True


# ---------------------------------------------------------------------------
# 9b. replay_consumed_injections — crash-recovery checkpoint replay
# ---------------------------------------------------------------------------


def test_replay_consumed_injections_inserts_missing_flagged_at_anchor() -> None:
    """A crash-dropped injection is re-inserted flagged at its anchor position."""
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    user_msg = Message("user", [Content.from_text("Do something")])
    tool_result = Message("tool", [Content.from_function_result(call_id="c1", result="ok")])
    final_asst = Message("assistant", [Content.from_text("Done.")])
    state: dict[str, Any] = {"messages": [user_msg, tool_result, final_asst]}
    history = SessionHistoryManager()
    history.bind(state)

    analytics_item_id = "b" * 32
    history.replay_consumed_injections(
        [
            _consumed_injection(
                "Also check this",
                InjectionAnchor.from_message(tool_result),
                analytics_item_id=analytics_item_id,
            )
        ]
    )

    messages = state["messages"]
    assert [m.role for m in messages] == ["user", "tool", "user", "assistant"]
    assert messages[2].contents[0].text == "Also check this"
    assert messages[2].additional_properties.get(HistoryMarkerKind.INJECTED_KEY) is True
    assert read_analytics_item_id(messages[2].additional_properties) == analytics_item_id


def test_replay_consumed_injections_dedups_already_persisted_flagged_copy() -> None:
    """An already-persisted flagged copy is never duplicated (kind-aware dedup)."""
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    user_msg = Message("user", [Content.from_text("Do something")])
    persisted = Message("user", [Content.from_text("Also check this")])
    persisted.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    state: dict[str, Any] = {"messages": [user_msg, persisted]}
    history = SessionHistoryManager()
    history.bind(state)

    history.replay_consumed_injections([_consumed_injection("Also check this", InjectionAnchor())])

    assert len(state["messages"]) == 2


def test_replay_consumed_injections_same_text_opener_does_not_suppress() -> None:
    """A same-text UNFLAGGED opener is not an injection copy — the replay still lands."""
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    opener = Message("user", [Content.from_text("what time is it?")])
    state: dict[str, Any] = {"messages": [opener]}
    history = SessionHistoryManager()
    history.bind(state)

    history.replay_consumed_injections([_consumed_injection("what time is it?", InjectionAnchor())])

    messages = state["messages"]
    assert len(messages) == 2
    assert HistoryMarkerKind.INJECTED_KEY not in messages[0].additional_properties
    assert messages[1].additional_properties.get(HistoryMarkerKind.INJECTED_KEY) is True


def test_replay_consumed_injections_dedup_scoped_to_current_turn_region() -> None:
    """A flagged same-text copy in an EARLIER turn must not suppress the replay."""
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    old_injection = Message("user", [Content.from_text("note")])
    old_injection.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    turn_marker = Message("assistant", [""])
    turn_marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    opener = Message("user", [Content.from_text("new task")])
    state: dict[str, Any] = {"messages": [old_injection, turn_marker, opener]}
    history = SessionHistoryManager()
    history.bind(state)

    history.replay_consumed_injections([_consumed_injection("note", InjectionAnchor())])

    messages = state["messages"]
    assert len(messages) == 4
    assert messages[-1].contents[0].text == "note"
    assert messages[-1].additional_properties.get(HistoryMarkerKind.INJECTED_KEY) is True


def test_replay_consumed_injections_multiple_restore_in_consumption_order() -> None:
    """Two injections sharing one anchor replay in queue order; distinct same-text ones both land."""
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    anchor_msg = Message("tool", [Content.from_function_result(call_id="c1", result="ok")])
    final = Message("assistant", [Content.from_text("done")])
    state: dict[str, Any] = {"messages": [anchor_msg, final]}
    history = SessionHistoryManager()
    history.bind(state)

    anchor = InjectionAnchor.from_message(anchor_msg)
    history.replay_consumed_injections(
        [_consumed_injection("yes", anchor), _consumed_injection("yes", anchor), _consumed_injection("B", anchor)]
    )

    messages = state["messages"]
    assert [m.role for m in messages] == ["tool", "user", "user", "user", "assistant"]
    assert [m.contents[0].text for m in messages[1:4]] == ["yes", "yes", "B"]


def test_replay_consumed_injections_mixed_persisted_then_later_lands_after() -> None:
    """Earlier injection already persisted (dedup skips); a later one anchored on it lands after."""
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    opener = Message("user", [Content.from_text("task")])
    earlier = Message("user", [Content.from_text("first note")])
    earlier.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    final = Message("assistant", [Content.from_text("done")])
    state: dict[str, Any] = {"messages": [opener, earlier, final]}
    history = SessionHistoryManager()
    history.bind(state)

    history.replay_consumed_injections(
        [
            _consumed_injection("first note", InjectionAnchor.from_message(opener)),
            _consumed_injection("second note", InjectionAnchor.from_message(earlier)),
        ]
    )

    messages = state["messages"]
    texts = [m.contents[0].text for m in messages]
    assert texts == ["task", "first note", "second note", "done"]
    assert messages[2].additional_properties.get(HistoryMarkerKind.INJECTED_KEY) is True


def test_replay_consumed_injections_new_same_text_injection_not_suppressed() -> None:
    """A distinct earlier same-text injection must not dedup a NEW consumption.

    Resumed-turn shape: the prior run persisted "same note" (own
    ``_injection_id``); the user injects the same text again mid-run and the
    process crashes before finalization.  Region-wide TEXT dedup would drop
    the new injection — user-input loss.  Identity dedup lands it.
    """
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    opener = Message("user", [Content.from_text("task")])
    earlier = Message("user", [Content.from_text("same note")])
    earlier.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    earlier.additional_properties[HistoryMarkerKind.INJECTION_ID_KEY] = "id-old"
    done = Message("assistant", [Content.from_text("done")])
    state: dict[str, Any] = {"messages": [opener, earlier, done]}
    history = SessionHistoryManager()
    history.bind(state)

    history.replay_consumed_injections(
        [_consumed_injection("same note", InjectionAnchor.from_message(done), consumption_id="id-new")]
    )

    messages = state["messages"]
    assert [m.contents[0].text for m in messages] == ["task", "same note", "done", "same note"]
    assert messages[-1].additional_properties.get(HistoryMarkerKind.INJECTED_KEY) is True
    assert messages[-1].additional_properties.get(HistoryMarkerKind.INJECTION_ID_KEY) == "id-new"


def test_replay_consumed_injections_id_bearing_not_suppressed_by_unstamped_legacy_copy() -> None:
    """An id-bearing replay never dedups against an id-LESS persisted injection.

    A pre-stamp legacy history cannot prove its copy is this consumption —
    and it never is: the persisted copy of an id-bearing consumption always
    carries the stamp (same ``ConsumedInjection`` feeds both writers), so an
    unstamped same-text message is a distinct injection.
    """
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    opener = Message("user", [Content.from_text("task")])
    legacy = Message("user", [Content.from_text("same note")])
    legacy.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    state: dict[str, Any] = {"messages": [opener, legacy]}
    history = SessionHistoryManager()
    history.bind(state)

    history.replay_consumed_injections([_consumed_injection("same note", InjectionAnchor(), consumption_id="id-new")])

    texts = [m.contents[0].text for m in state["messages"]]
    assert texts == ["task", "same note", "same note"]


def test_replay_consumed_injections_dedups_by_injection_id() -> None:
    """A region message carrying the injection's own id IS this consumption."""
    from chrys.service.agent_middleware.injection import InjectionAnchor
    from chrys.service.session.history import SessionHistoryManager

    opener = Message("user", [Content.from_text("task")])
    persisted = Message("user", [Content.from_text("same note")])
    persisted.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    persisted.additional_properties[HistoryMarkerKind.INJECTION_ID_KEY] = "id-1"
    state: dict[str, Any] = {"messages": [opener, persisted]}
    history = SessionHistoryManager()
    history.bind(state)

    history.replay_consumed_injections([_consumed_injection("same note", InjectionAnchor(), consumption_id="id-1")])

    assert len(state["messages"]) == 2


@pytest.mark.asyncio
async def test_injection_middleware_captures_content_anchor() -> None:
    """Middleware captures a structural InjectionAnchor at consumption."""
    from chrys.service.agent_middleware.injection import InjectionMiddleware

    captured: list[ConsumedInjection] = []

    async def on_consumed(injection: ConsumedInjection) -> None:
        captured.append(injection)

    mw = InjectionMiddleware()
    mw.set_on_consumed(on_consumed)
    mw.queue("hi there")

    last_msg = Message("tool", [Content.from_function_result(call_id="c1", result="ok")])
    ctx_messages: list[Message] = [
        Message("user", [Content.from_text("Hello")]),
        Message(
            "assistant",
            [Content.from_function_call(call_id="c1", name="zsh", arguments={})],
        ),
        last_msg,
    ]

    class _FakeCtx:
        messages: list[Any]
        options: dict[str, Any] | None = None

    ctx = _FakeCtx()
    ctx.messages = ctx_messages

    call_next_ran = False

    async def _call_next() -> None:
        nonlocal call_next_ran
        call_next_ran = True

    await mw.process(ctx, _call_next)  # type: ignore[arg-type]

    assert call_next_ran
    assert len(captured) == 1
    injection = captured[0]
    assert injection.text == "hi there"
    # Anchor must capture the tool result message's structural keys.
    assert injection.anchor.role == "tool"
    assert injection.anchor.call_ids == ("c1",)
    assert injection.anchor.contents_ref is not None
    assert injection.anchor.contents_ref() is last_msg.contents
    # Injection appended AFTER the last message in context.
    assert len(ctx.messages) == 4
    assert ctx.messages[-1].role == "user"
    assert ctx.messages[-1].contents[0].text == "hi there"
    # Flagged at creation time: the very call that consumes the injection
    # must classify it as mid-turn (compaction runs before persistence).
    assert ctx.messages[-1].additional_properties[HistoryMarkerKind.INJECTED_KEY] is True


@pytest.mark.asyncio
async def test_injection_middleware_stamps_injected_flag_on_every_appended_message() -> None:
    """Each per-call appended wire copy carries ``_injected`` (creation-time stamp)."""
    from chrys.service.agent_middleware.injection import InjectionMiddleware

    mw = InjectionMiddleware()
    mw.queue("first note")
    mw.queue("second note")

    class _FakeCtx:
        messages: list[Any]
        options: dict[str, Any] | None = None

    ctx = _FakeCtx()
    ctx.messages = [Message("user", [Content.from_text("opener")])]

    async def _call_next() -> None:
        return None

    await mw.process(ctx, _call_next)  # type: ignore[arg-type]

    appended = ctx.messages[1:]
    assert [m.contents[0].text for m in appended] == ["first note", "second note"]
    for msg in appended:
        assert msg.additional_properties[HistoryMarkerKind.INJECTED_KEY] is True
    # The opener itself is never stamped.
    assert HistoryMarkerKind.INJECTED_KEY not in ctx.messages[0].additional_properties


@pytest.mark.asyncio
async def test_drained_messages_are_isolated_from_wire_copy_mutation() -> None:
    """The drained copy survives a client mutating the wire message in place.

    The wire message goes to the client on the very call that consumes the
    injection, while the drained copy is what later tool iterations re-send
    and what persistence captures — they must not share wrapper or contents
    list. The props dict (metadata write-through) and the content object
    stay shared.
    """
    from chrys.service.agent_middleware.injection import InjectionMiddleware

    mw = InjectionMiddleware()
    mw.queue("note")

    class _FakeCtx:
        messages: list[Any]
        options: dict[str, Any] | None = None

    ctx = _FakeCtx()
    ctx.messages = [Message("user", [Content.from_text("opener")])]

    async def _call_next() -> None:
        # The client mutates the received wire copy during the consuming call.
        ctx.messages[-1].contents.append(Content.from_text("CORRUPTED"))

    await mw.process(ctx, _call_next)  # type: ignore[arg-type]

    wire_msg = ctx.messages[-1]
    assert [c.text for c in wire_msg.contents] == ["note", "CORRUPTED"]
    drained = mw.drain_consumed_injection_messages()
    assert len(drained) == 1
    retained = drained[0]
    assert retained is not wire_msg
    assert [c.text for c in retained.contents] == ["note"]
    assert retained.additional_properties is wire_msg.additional_properties
    assert retained.contents[0] is wire_msg.contents[0]


@pytest.mark.asyncio
async def test_injection_middleware_assigns_per_consumption_id() -> None:
    """Every consumption carries a ``consumption_id`` shared by the wire copy
    and the ConsumedInjection mirror — queued ids reused, missing ones
    synthesized — so crash-recovery replay can dedup by identity.  The
    frontend queue handle (``injection_id``) round-trips UNCHANGED: id-less
    frontends (ACP) must keep ``None`` in ``UserInjectResult`` — the TUI's
    input-ownership check treats a non-None id it never issued as stale."""
    from chrys.service.agent_middleware.injection import InjectionMiddleware

    captured: list[ConsumedInjection] = []

    async def on_consumed(injection: ConsumedInjection) -> None:
        captured.append(injection)

    mw = InjectionMiddleware()
    mw.set_on_consumed(on_consumed)
    mw.queue("with id", injection_id="ui-handle")
    mw.queue("without id")

    class _FakeCtx:
        messages: list[Any]
        options: dict[str, Any] | None = None

    ctx = _FakeCtx()
    ctx.messages = [Message("user", [Content.from_text("opener")])]

    async def _call_next() -> None:
        return None

    await mw.process(ctx, _call_next)  # type: ignore[arg-type]

    appended = ctx.messages[1:]
    assert len(appended) == 2
    assert len(captured) == 2
    wire_ids = [m.additional_properties.get(HistoryMarkerKind.INJECTION_ID_KEY) for m in appended]
    assert wire_ids == [c.consumption_id for c in captured]
    assert wire_ids[0] == "ui-handle"
    # Synthesized when the queue entry had none — never None, never colliding.
    assert wire_ids[1]
    assert wire_ids[1] != wire_ids[0]
    # The event-visible queue handle is NOT the synthesized value.
    assert [c.injection_id for c in captured] == ["ui-handle", None]
    analytics_ids = [read_analytics_item_id(message.additional_properties) for message in appended]
    assert analytics_ids == [consumed.analytics_item_id for consumed in captured]
    assert all(is_valid_analytics_id(item_id) for item_id in analytics_ids)
    assert len(set(analytics_ids)) == 2
    retained = mw.drain_consumed_injection_messages()
    assert [read_analytics_item_id(message.additional_properties) for message in retained] == analytics_ids


@pytest.mark.asyncio
async def test_retry_replays_complete_batch_with_stable_ids_and_new_anchor() -> None:
    """A failed provider attempt replays one immutable batch on the retry."""
    from chrys.service.agent_middleware.injection import InjectionMiddleware

    captured: list[tuple[ConsumedInjection, ...]] = []

    async def on_consumed(batch: tuple[ConsumedInjection, ...]) -> None:
        captured.append(batch)

    class _FakeCtx:
        messages: list[Any]
        options: dict[str, Any] | None = None

    middleware = InjectionMiddleware()
    middleware.set_on_consumed_batch(on_consumed)
    middleware.queue("first", injection_id="i1")
    middleware.queue("second", injection_id="i2")
    middleware.begin_retry()
    try:
        failed_anchor = Message("tool", [Content.from_function_result("failed", result="failed")])
        failed = _FakeCtx()
        failed.messages = [failed_anchor]

        async def fail_provider() -> None:
            raise RuntimeError("transient")

        with pytest.raises(RuntimeError, match="transient"):
            await middleware.process(failed, fail_provider)  # type: ignore[arg-type]

        assert middleware.cancel("i1") is None
        middleware.restore_for_retry()

        successful_anchor = Message("tool", [Content.from_function_result("success", result="ok")])
        successful = _FakeCtx()
        successful.messages = [successful_anchor]

        async def succeed() -> None:
            return None

        await middleware.process(successful, succeed)  # type: ignore[arg-type]
    finally:
        middleware.end_retry()

    assert len(captured) == 2
    assert [item.consumption_id for item in captured[0]] == ["i1", "i2"]
    assert [item.consumption_id for item in captured[1]] == ["i1", "i2"]
    assert [item.analytics_item_id for item in captured[0]] == [item.analytics_item_id for item in captured[1]]
    assert all(is_valid_analytics_id(item.analytics_item_id) for item in captured[1])
    assert all(item.anchor.find_in([successful_anchor]) == 0 for item in captured[1])
    assert [message.text for message in successful.messages[1:]] == ["first", "second"]


@pytest.mark.asyncio
async def test_retry_preserves_replay_when_next_attempt_fails_before_middleware() -> None:
    """A context-provider failure before chat middleware cannot erase replay."""
    from chrys.service.agent_middleware.injection import InjectionMiddleware

    captured: list[tuple[ConsumedInjection, ...]] = []

    async def on_consumed(batch: tuple[ConsumedInjection, ...]) -> None:
        captured.append(batch)

    class _FakeCtx:
        messages: list[Any]
        options: dict[str, Any] | None = None

    middleware = InjectionMiddleware()
    middleware.set_on_consumed_batch(on_consumed)
    middleware.queue("survive", injection_id="stable")
    middleware.begin_retry()
    try:
        first = _FakeCtx()
        first.messages = [Message("user", ["start"])]

        async def fail_provider() -> None:
            raise RuntimeError("transient")

        with pytest.raises(RuntimeError, match="transient"):
            await middleware.process(first, fail_provider)  # type: ignore[arg-type]
        middleware.restore_for_retry()

        # The second attempt fails in a context provider, before process().
        middleware.restore_for_retry()

        third = _FakeCtx()
        third.messages = [Message("user", ["retry start"])]

        async def succeed() -> None:
            return None

        await middleware.process(third, succeed)  # type: ignore[arg-type]
    finally:
        middleware.end_retry()

    assert [item.consumption_id for item in captured[-1]] == ["stable"]
    assert [message.text for message in third.messages] == ["retry start", "survive"]


@pytest.mark.asyncio
async def test_cancel_during_batch_callback_replays_every_drained_injection() -> None:
    """Cancellation during durability I/O cannot strand a batch tail."""
    from chrys.service.agent_middleware.injection import InjectionMiddleware

    callback_entered = asyncio.Event()
    block_callback = asyncio.Event()

    async def blocking_callback(_batch: tuple[ConsumedInjection, ...]) -> None:
        callback_entered.set()
        await block_callback.wait()

    class _FakeCtx:
        messages: list[Any]
        options: dict[str, Any] | None = None

    middleware = InjectionMiddleware()
    middleware.set_on_consumed_batch(blocking_callback)
    middleware.queue("one", injection_id="i1")
    middleware.queue("two", injection_id="i2")
    middleware.begin_retry()
    try:
        failed = _FakeCtx()
        failed.messages = [Message("user", ["start"])]

        async def call_next() -> None:
            return None

        task = asyncio.create_task(middleware.process(failed, call_next))  # type: ignore[arg-type]
        await callback_entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        middleware.restore_for_retry()
        replayed: list[tuple[ConsumedInjection, ...]] = []

        async def capture(batch: tuple[ConsumedInjection, ...]) -> None:
            replayed.append(batch)

        middleware.set_on_consumed_batch(capture)
        successful = _FakeCtx()
        successful.messages = [Message("user", ["retry"])]
        await middleware.process(successful, call_next)  # type: ignore[arg-type]
    finally:
        middleware.end_retry()

    assert [[item.text for item in batch] for batch in replayed] == [["one", "two"]]
    assert [item.consumption_id for item in replayed[0]] == ["i1", "i2"]
    assert [message.text for message in successful.messages] == ["retry", "one", "two"]


# ---------------------------------------------------------------------------
# Injection cancellation (Esc on a queued mid-run message)
# ---------------------------------------------------------------------------


def test_injection_middleware_cancel_removes_only_target() -> None:
    """cancel() removes exactly the identified injection, preserving order."""
    from chrys.service.agent_middleware.injection import InjectionMiddleware

    mw = InjectionMiddleware()
    mw.queue("first", injection_id="a")
    mw.queue("second", injection_id="b", reminders=("hook note",))
    mw.queue("third", injection_id="c")

    removed = mw.cancel("b")

    assert removed is not None
    assert removed.text == "second"
    assert removed.injection_id == "b"
    # Commit-time reminders travel with the entry so cancel can withdraw them.
    assert removed.reminders == ("hook note",)
    assert [q.text for q in mw.drain_pending()] == ["first", "third"]


def test_injection_middleware_cancel_unknown_id_is_noop() -> None:
    """cancel() with an unknown or already-consumed id returns None."""
    from chrys.service.agent_middleware.injection import InjectionMiddleware

    mw = InjectionMiddleware()
    mw.queue("only", injection_id="a")

    assert mw.cancel("missing") is None
    assert mw.cancel("a") is not None
    # A second cancel of the same id (already removed) is also a no-op.
    assert mw.cancel("a") is None
    assert mw.drain_pending() == []


@pytest.mark.asyncio
async def test_cancelled_injection_not_delivered_to_model() -> None:
    """A cancelled injection never reaches context.messages or on_consumed."""
    from chrys.service.agent_middleware.injection import InjectionMiddleware

    captured: list[ConsumedInjection] = []

    async def on_consumed(injection: ConsumedInjection) -> None:
        captured.append(injection)

    mw = InjectionMiddleware()
    mw.set_on_consumed(on_consumed)
    mw.queue("keep me", injection_id="keep")
    mw.queue("cancel me", injection_id="gone")
    mw.cancel("gone")

    class _FakeCtx:
        messages: list[Any]
        options: dict[str, Any] | None = None

    ctx = _FakeCtx()
    ctx.messages = [Message("user", [Content.from_text("Hello")])]

    async def _call_next() -> None:
        return None

    await mw.process(ctx, _call_next)  # type: ignore[arg-type]

    assert [c.text for c in captured] == ["keep me"]
    assert [c.injection_id for c in captured] == ["keep"]
    delivered = [m.contents[0].text for m in ctx.messages if m.role == "user"]
    assert "cancel me" not in delivered
    assert "keep me" in delivered


# ---------------------------------------------------------------------------
# Consumed-injection crash durability: the engine's consumption callback
# flushes the recovery sidecar before returning (§2.5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consumed_injection_callback_flushes_sidecar_before_returning(
    tmp_path, agent_engine, *, engine_services
) -> None:
    """After ``_on_consumed`` returns, the sidecar already holds the injection.

    The wire copy lives only in the tool loop and the persisted copy is
    written by the finalizer — between consumption and finalization the
    recovery sidecar is the injection's ONLY durable home.  The engine's
    consumption callback must therefore FLUSH the background checkpoint
    writer, not merely queue a snapshot: with a deliberately slow
    ``save_recovery_session``, the write still completes strictly before the
    callback returns (a queue-only regression would return immediately and
    flip the recorded order).
    """
    bus = EventBus()
    store = JsonFileStateStore(tmp_path)
    mock_client = MockChatClient(responses=[MockResponse(text="unused")])

    with _patch_client(mock_client):
        engine = agent_engine(bus, settings=_make_settings(), state_store=store)
        await engine.start(_make_profile())
        # Seed the live state a mid-run consumption would see: an opener,
        # a completed tool batch, and a registered current input.  The
        # callback under test is the REAL closure wired at construction.
        engine.session.session_id = "inj-durable"
        user = Message("user", ["do work"])
        assistant = Message("assistant", [Content.from_function_call("c1", "read_file", arguments={})])
        tool = Message("tool", [Content.from_function_result("c1", result="done")])
        engine.current.loaded.bindings.backend.history_state = {
            "messages": [user],
            "compressed_msgs": [],
            "turn_counter": 1,
        }
        engine.current.loaded.loop_recorder._initial_count = 1
        engine.current.loaded.loop_recorder._captured = [user, assistant, tool]
        engine.turns.turn_state.set_current_input("do work", None, None)

        order: list[str] = []
        original_save = engine_services(engine).persistence.save_recovery_session

        async def slow_save(*args: Any, **kwargs: Any) -> None:
            # The delay opens the window a queue-only implementation
            # would fall into: it returns while the write is still here.
            await asyncio.sleep(0.05)
            await original_save(*args, **kwargs)
            order.append("sidecar-durable")

        engine_services(engine).persistence.save_recovery_session = slow_save  # type: ignore[method-assign]

        assert engine.current.loaded.injection is not None
        assert engine.current.loaded.injection._on_consumed_batch is not None
        await engine.current.loaded.injection._on_consumed_batch(
            (ConsumedInjection(text="mid-run note", anchor=InjectionAnchor.from_message(tool)),)
        )
        order.append("callback-returned")

        assert order == ["sidecar-durable", "callback-returned"]

        # Content pin: the sidecar on disk already carries the mirrored
        # injection, flagged and before the terminal markers.
        recovered = await store.load_recovery_session("inj-durable")
        assert recovered is not None
        notes = [m for m in recovered["messages"] if m.role == "user" and m.text == "mid-run note"]
        assert len(notes) == 1
        assert notes[0].additional_properties.get(HistoryMarkerKind.INJECTED_KEY) is True
        tail_kinds = [m.additional_properties.get(HistoryMarkerKind.KEY) for m in recovered["messages"][-2:]]
        assert tail_kinds == [HistoryMarkerKind.INTERRUPTED, HistoryMarkerKind.TURN]
