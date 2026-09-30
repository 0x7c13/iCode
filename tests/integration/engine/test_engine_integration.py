# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Integration tests — agent.run() with tool loop + compaction, and full Chrys engine."""

from __future__ import annotations

from typing import Annotated
from unittest.mock import create_autospec

import pytest

from chrys.kernel import Agent, FunctionTool
from chrys.service.context.compaction import MixedLanguageTokenizer, UnifiedContextStrategy
from chrys.service.llm.mock import MockChatClient, MockResponse
from tests.support.phase4_stubs import StubLastWordsGenerator, StubReminderMiddleware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LARGE_RESULT = "x" * 4000  # ~1400 tokens with MixedLanguageTokenizer (4000 * 0.35)


def _wire_phase4(strategy: UnifiedContextStrategy) -> None:
    """Attach LAST_WORDS collaborators so Phase 4 can drop current-turn groups."""
    strategy.set_last_words_generator(StubLastWordsGenerator())
    reminder = StubReminderMiddleware()
    strategy.bind_reminder(reminder, reminder.last_words)


def _lookup(query: Annotated[str, "search query"]) -> str:
    """Return a large fake search result."""
    return f"Results for '{query}': {_LARGE_RESULT}"


lookup_tool = FunctionTool(func=_lookup, name="lookup", description="Search")


def _tool_call_response(tool_name: str, call_id: str, args: dict) -> MockResponse:
    """Create a MockResponse that triggers a tool call."""
    return MockResponse(tool_calls=[(tool_name, call_id, args)])


# Deliberately local: this variant raises a file-specific timeout diagnostic.
async def _wait_until(predicate, *, timeout: float = 10.0, interval: float = 0.01) -> None:
    """Poll *predicate* until it returns true or *timeout* seconds elapse.

    Windows CI is too slow for fixed ``asyncio.sleep`` waits on async engine
    work, so tests poll the asserted condition instead of guessing a duration.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    if not predicate():
        raise AssertionError(f"condition not met within {timeout}s")


# ---------------------------------------------------------------------------
# Test 1: agent.run() with tool loop + compaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_run_compaction_reduces_messages():
    """Verify compaction reduces the messages the model sees during a tool loop.

    Setup: 8 rounds of lookup tool calls (each returning ~1400 tokens), then final text.
    With compaction, later model calls receive fewer messages than the raw accumulated count.

    Note: sequential message_id reuse can make incremental
    annotate_message_groups merge tool groups. The test verifies compaction is
    wired and has observable effect while tolerating that historical collision
    shape.
    """
    responses: list[MockResponse] = []
    for i in range(8):
        responses.append(_tool_call_response("lookup", f"call_{i}", {"query": f"topic_{i}"}))
    responses.append(MockResponse(text="Here is my summary based on all the research."))

    client = MockChatClient(responses=responses)
    # BPE compresses repetitive 'x' chars very efficiently (~500 tokens per
    # 4000-char result), so total conversation is ~4500 tokens.  Set
    # max_context small enough that trigger_pct (85%) fires.
    strategy = UnifiedContextStrategy(
        max_context_tokens=3000,
        trigger_pct=0.85,
        target_pct=0.50,
        tokenizer=MixedLanguageTokenizer(),
    )
    _wire_phase4(strategy)

    agent = Agent(client=client, tools=[lookup_tool])
    async with agent:
        result = await agent.run(
            "Research 8 topics",
            compaction_strategy=strategy,
        )

    # Agent should have produced a final text response
    assert result.text == "Here is my summary based on all the research."

    # Client was called 9 times (8 tool calls + 1 final)
    assert client.call_count == 9

    # Without compaction, the last call would receive 17 messages
    # (user + 8 tool groups * 2 msgs each). With compaction, some messages
    # are excluded or replaced with summaries, so the count should be lower.
    last_msg_count = len(client.call_history[-1][0])
    raw_count = 1 + 8 * 2  # user + 8 * (function_call + function_result)
    assert last_msg_count < raw_count, (
        f"Expected compaction to reduce message count: got {last_msg_count}, raw would be {raw_count}"
    )

    # Compaction strategy should have tracked excluded groups
    # (Phase 1 summaries and/or Phase 2/3 removals)
    assert len(strategy._summary_cache) + len(strategy._removed_group_ids) > 0, (
        "Expected compaction strategy to have tracked compacted groups"
    )


@pytest.mark.asyncio
async def test_agent_run_no_compaction_without_strategy():
    """Without compaction_strategy, all tool results remain in full."""
    responses: list[MockResponse] = []
    for i in range(4):
        responses.append(_tool_call_response("lookup", f"call_{i}", {"query": f"topic_{i}"}))
    responses.append(MockResponse(text="Done."))

    client = MockChatClient(responses=responses)

    agent = Agent(client=client, tools=[lookup_tool])
    async with agent:
        result = await agent.run("Research 4 topics")

    assert result.text == "Done."
    assert client.call_count == 5

    # Last call should have ALL messages — no compaction
    last_msg_count = len(client.call_history[-1][0])
    expected = 1 + 4 * 2  # user + 4 * (function_call + function_result)
    assert last_msg_count == expected

    # No summaries anywhere
    any_summaries = any(
        any(m.text and m.text.startswith("[Tool call:") for m in msgs) for msgs, _ in client.call_history
    )
    assert not any_summaries


# ---------------------------------------------------------------------------
# Test 1b: compress_context reduces messages within the same tool loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compress_context_reduces_messages_in_tool_loop(agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Verify compress_context queued via the strategy reduces API messages.

    Setup:
    - 3 turns of text-only conversation (each turn = user + assistant),
      with markers inserted by the history provider.
    - Turn 4: the LLM calls compress_context(marker_2) which folds turns 1-2.
    - After compress_context result, LLM calls lookup (another tool).
    - After lookup result, LLM produces final text.

    The test verifies that the API call AFTER compress_context sees fewer
    messages than the call that triggered compress_context.  This is the
    core bug that the UnifiedContextStrategy refactor fixes.
    """
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ContextCompressed, InvocationMessage, SessionReady, UserMessage
    from chrys.service.profiles.agents.schema import AgentProfile, ApprovalConfig, CompactionConfig, ToolsConfig

    compress_events: list[ContextCompressed] = []

    async def _noop(_e: object) -> None:
        pass

    async def _collect_compress(e: ContextCompressed) -> None:
        compress_events.append(e)

    bus = EventBus()
    await bus.subscribe(SessionReady, _noop)
    await bus.subscribe(InvocationMessage, _noop)
    await bus.subscribe(ContextCompressed, _collect_compress)

    # Compaction disabled (reserved=1.0) so only compress_context affects context.
    profile = AgentProfile(
        name="test",
        instructions="Test.",
        tools=ToolsConfig(builtins=[]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
    )
    settings = Settings()

    mock_responses = [
        # Turns 1-3: simple text responses
        MockResponse(text="Response 1"),
        MockResponse(text="Response 2"),
        MockResponse(text="Response 3"),
        # Turn 4, iteration 1: LLM calls compress_context
        MockResponse(tool_calls=[("compress_context", "cc1", {"marker_id": "turn_2", "summary": "Turns 1-2 done"})]),
        # Turn 4, iteration 2: after compress result, LLM calls lookup
        MockResponse(tool_calls=[("lookup", "lk1", {"query": "test"})]),
        # Turn 4, iteration 3: final text
        MockResponse(text="All done after compression and lookup."),
    ]
    mock_client = MockChatClient(responses=mock_responses)

    import chrys.orchestration.engine.build.builder as builder_module

    monkeypatch.setattr(
        builder_module, "create_client", create_autospec(builder_module.create_client, return_value=mock_client)
    )
    engine = agent_engine(bus, settings=settings)
    await engine.start(profile)

    # 3 text-only turns to build up history with markers
    for i in range(3):
        await bus.publish(UserMessage(text="do work"))
        # Wait for each turn's model call to land AND its run task to drain
        # rather than sleeping a fixed duration (flaky on slow Windows CI).
        # If the next turn's UserMessage arrives while this turn's FSM is
        # still RUNNING the engine injects it into the current turn instead
        # of starting a new one, throwing off the call/marker counts.
        await _wait_until(
            lambda i=i: (
                len(mock_client.call_history) >= i + 1
                and engine.turns.turn_state.lease.run_task is not None
                and engine.turns.turn_state.lease.run_task.done()
            )
        )

    # Turn 4: triggers compress_context → lookup → final text (three model
    # calls). Wait until the tool loop has settled and folded turns 1-2.
    await bus.publish(UserMessage(text="compress and do more"))
    await _wait_until(
        lambda: (
            len(mock_client.call_history) >= 6
            and len(engine.current.loaded.bindings.backend.history_state.get("compressed_msgs", [])) == 1
            and len(compress_events) == 1
        )
    )

    # Verify compression happened
    state = engine.current.loaded.bindings.backend.history_state
    compressed = state.get("compressed_msgs", [])
    assert len(compressed) == 1, f"Expected 1 compressed block, got {len(compressed)}"
    assert compressed[0].marker_id == "turn_2"

    # The ContextCompressed event should have fired
    assert len(compress_events) == 1
    assert compress_events[0].turn_range == (1, 2)

    # Key assertion: the final API call should see fewer messages
    # than the uncompressed count.
    #
    # Without compression, the final call would see all messages:
    #   user1+asst1 + user2+asst2 + user3+asst3 + user4 +
    #   asst(compress_call) + tool(compress_result) +
    #   asst(lookup_call) + tool(lookup_result) = 11
    # With compression, turns 1-2 messages (4) are replaced by 1
    # summary via _excluded flags, so the count drops by 3.
    history = mock_client.call_history
    assert len(history) >= 6, f"Expected >= 6 API calls, got {len(history)}"

    msgs_at_final = len(history[-1][0])
    uncompressed_count = 11
    assert msgs_at_final < uncompressed_count, (
        f"Expected compression to reduce messages: "
        f"final saw {msgs_at_final}, without compression would be {uncompressed_count}"
    )


@pytest.mark.asyncio
async def test_compress_context_with_session_save_restore(
    agent_engine, monkeypatch: pytest.MonkeyPatch, *, engine_services
):
    """End-to-end: multi-turn → compress_context → session save/restore.

    Verifies:
    1. Agent calls compress_context to fold completed turns
    2. Context is reduced for subsequent API calls within the same tool loop
    3. Session is saved with compressed blocks
    4. Session restore preserves compressed blocks and summary messages
    """
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ContextCompressed, InvocationMessage, SessionReady, UserMessage
    from chrys.service.profiles.agents.schema import AgentProfile, ApprovalConfig, CompactionConfig, ToolsConfig

    compress_events: list[ContextCompressed] = []

    async def _collect_compress(e: ContextCompressed) -> None:
        compress_events.append(e)

    async def _noop(_e: object) -> None:
        pass

    bus = EventBus()
    await bus.subscribe(SessionReady, _noop)
    await bus.subscribe(InvocationMessage, _noop)
    await bus.subscribe(ContextCompressed, _collect_compress)

    profile = AgentProfile(
        name="test",
        instructions="Test.",
        tools=ToolsConfig(builtins=[]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
    )
    settings = Settings()

    # 3 text-only turns, then turn 4: compress_context + more tool calls
    mock_responses = [
        MockResponse(text="Response 1"),
        MockResponse(text="Response 2"),
        MockResponse(text="Response 3"),
        MockResponse(tool_calls=[("compress_context", "cc1", {"marker_id": "turn_2", "summary": "Turns 1-2 done"})]),
        MockResponse(
            tool_calls=[
                ("list_compressed_contexts", "lc1", {}),
            ]
        ),
        MockResponse(text="Compressed and verified."),
    ]
    mock_client = MockChatClient(responses=mock_responses)

    import chrys.orchestration.engine.build.builder as builder_module

    monkeypatch.setattr(
        builder_module, "create_client", create_autospec(builder_module.create_client, return_value=mock_client)
    )
    engine = agent_engine(bus, settings=settings)
    await engine.start(profile)

    for i in range(3):
        await bus.publish(UserMessage(text="do work"))
        # Each text-only turn makes one model call; wait for it to land AND
        # the run task to drain rather than guessing a sleep duration (flaky
        # on slow Windows CI).  A next-turn UserMessage that arrives while
        # this turn's FSM is still RUNNING is injected into the current turn
        # instead of starting a new one.
        await _wait_until(
            lambda i=i: (
                len(mock_client.call_history) >= i + 1
                and engine.turns.turn_state.lease.run_task is not None
                and engine.turns.turn_state.lease.run_task.done()
            )
        )

    await bus.publish(UserMessage(text="compress and verify"))
    # Turn 4 makes three model calls (compress_context + result,
    # list_compressed_contexts + result, final text). Wait until the tool
    # loop has fully settled and folded the completed turns.
    await _wait_until(
        lambda: (
            len(mock_client.call_history) >= 6
            and len(engine.current.loaded.bindings.backend.history_state.get("compressed_msgs", [])) == 1
            and len(compress_events) == 1
        )
    )

    state = engine.current.loaded.bindings.backend.history_state

    # ---- Verify compression ----
    compressed = state.get("compressed_msgs", [])
    assert len(compressed) == 1
    assert compressed[0].marker_id == "turn_2"
    assert len(compress_events) == 1

    # ---- Verify API saw reduced messages after compression ----
    history = mock_client.call_history
    msgs_at_final = len(history[-1][0])
    # Without compression: u1+a1+u2+a2+u3+a3+u4 + compress_call+result + list_call+result = 11
    uncompressed_count = 11
    assert msgs_at_final < uncompressed_count, (
        f"Expected compression to reduce messages: "
        f"final saw {msgs_at_final}, without compression would be {uncompressed_count}"
    )

    # ---- Session save + restore ----
    import tempfile
    from pathlib import Path

    from chrys.service.state.store import JsonFileStateStore

    # ignore_cleanup_errors: on Windows the state store's .write.lock file
    # can still be held by the OS at rmtree time even after the lock handle
    # is closed, raising PermissionError during cleanup. Every save/restore
    # assertion runs before cleanup, so tolerating a cleanup failure is safe.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = JsonFileStateStore(Path(tmpdir))
        engine_services(engine).persistence._state_store = store

        session_id = engine.session.session_id
        assert session_id is not None
        await engine.writer.save_current_session()

        loaded = await store.load_session(session_id)
        assert loaded is not None

        loaded_compressed = loaded.get("compressed_msgs", [])
        assert len(loaded_compressed) == 1
        assert loaded_compressed[0].marker_id == "turn_2"
        assert loaded_compressed[0].summary_text == "Turns 1-2 done"
        assert len(loaded_compressed[0].messages) > 0

        from chrys.service.context.providers.history import _is_compressed_summary

        summaries = [m for m in loaded.get("messages", []) if _is_compressed_summary(m)]
        assert len(summaries) == 1


@pytest.mark.asyncio
async def test_compaction_phases_with_parallel_tool_calls():
    """Verify compaction phases fire with parallel tool calls and tight context.

    Setup: 4 rounds of 3 parallel lookup tool calls (each returning ~1400 tokens).
    With max_context_tokens=3000, compaction triggers and progressively
    summarises/removes old tool groups.
    """
    responses: list[MockResponse] = []
    for round_idx in range(4):
        responses.append(
            MockResponse(
                tool_calls=[
                    ("lookup", f"r{round_idx}_a", {"query": f"topic_{round_idx}a"}),
                    ("lookup", f"r{round_idx}_b", {"query": f"topic_{round_idx}b"}),
                    ("lookup", f"r{round_idx}_c", {"query": f"topic_{round_idx}c"}),
                ]
            )
        )
    responses.append(MockResponse(text="Done with all lookups."))

    client = MockChatClient(responses=responses)
    strategy = UnifiedContextStrategy(
        max_context_tokens=3000,
        trigger_pct=0.85,
        target_pct=0.50,
        tokenizer=MixedLanguageTokenizer(),
    )
    _wire_phase4(strategy)

    agent = Agent(client=client, tools=[lookup_tool])
    async with agent:
        result = await agent.run("Research 12 topics in 4 rounds", compaction_strategy=strategy)

    assert result.text == "Done with all lookups."

    # Compaction should have fired (summary_cache or removed_group_ids populated)
    assert len(strategy._summary_cache) + len(strategy._removed_group_ids) > 0, (
        "Expected compaction to have tracked compacted groups"
    )

    # The final API call should see fewer messages than the raw count
    last_msg_count = len(client.call_history[-1][0])
    # Raw: user + 4 rounds * (asst_parallel_call + tool_results) = 1 + 4*2 = 9
    raw_count = 1 + 4 * 2
    assert last_msg_count < raw_count, (
        f"Expected compaction to reduce messages: got {last_msg_count}, raw would be {raw_count}"
    )


@pytest.mark.asyncio
async def test_compress_with_intermediate_text_and_injection(
    agent_engine, monkeypatch: pytest.MonkeyPatch, *, engine_services
):
    """End-to-end: intermediate agent text + user injection + compress_context.

    Verifies:
    1. Agent produces intermediate text alongside tool calls
    2. User injection mid-turn is persisted with _injected marker
    3. compress_context reduces context for subsequent calls
    4. State is saved correctly with all metadata
    """
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import (
        ContextCompressed,
        InvocationMessage,
        SessionReady,
        UserInject,
        UserInjectResult,
        UserMessage,
    )
    from chrys.service.profiles.agents.schema import AgentProfile, ApprovalConfig, CompactionConfig, ToolsConfig

    compress_events: list[ContextCompressed] = []
    inject_results: list[UserInjectResult] = []

    async def _collect_compress(e: ContextCompressed) -> None:
        compress_events.append(e)

    async def _collect_inject(e: UserInjectResult) -> None:
        inject_results.append(e)

    async def _noop(_e: object) -> None:
        pass

    bus = EventBus()
    await bus.subscribe(SessionReady, _noop)
    await bus.subscribe(InvocationMessage, _noop)
    await bus.subscribe(ContextCompressed, _collect_compress)
    await bus.subscribe(UserInjectResult, _collect_inject)

    profile = AgentProfile(
        name="test",
        instructions="Test.",
        tools=ToolsConfig(builtins=[]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
    )
    settings = Settings()

    # Turn 1: text only
    # Turn 2: intermediate text + tool call (list_compressed_contexts),
    #          then user injection mid-turn, then final text
    # Turn 3: compress_context folds turn 1, then final text
    mock_responses = [
        MockResponse(text="Response 1"),
        # Turn 2: intermediate text with tool call
        MockResponse(
            text="Let me check the context first.",
            tool_calls=[("list_compressed_contexts", "lc1", {})],
        ),
        # After injection is consumed + tool result, agent responds
        MockResponse(text="Got it, context checked and injection noted."),
        # Turn 3: compress turn 1
        MockResponse(tool_calls=[("compress_context", "cc1", {"marker_id": "turn_1", "summary": "Turn 1 done"})]),
        MockResponse(text="Compressed turn 1."),
    ]
    mock_client = MockChatClient(responses=mock_responses)

    import chrys.orchestration.engine.build.builder as builder_module

    monkeypatch.setattr(
        builder_module, "create_client", create_autospec(builder_module.create_client, return_value=mock_client)
    )
    engine = agent_engine(bus, settings=settings)
    await engine.start(profile)

    # Turn 1: one model call. Wait until it lands AND the run task drains
    # so turn 2's UserMessage starts a fresh turn rather than being
    # injected into a still-running turn 1 (flaky on slow Windows CI).
    await bus.publish(UserMessage(text="start"))
    await _wait_until(
        lambda: (
            len(mock_client.call_history) >= 1
            and engine.turns.turn_state.lease.run_task is not None
            and engine.turns.turn_state.lease.run_task.done()
        )
    )

    # Turn 2: agent calls list_compressed_contexts (intermediate text +
    # tool). Let the turn's first model call land (call #2) so the inject
    # below truly lands mid-run.
    await bus.publish(UserMessage(text="check context"))
    await _wait_until(lambda: len(mock_client.call_history) >= 2)

    # Inject mid-turn while agent is running. Turn 2 then makes its final
    # model call (call #3) after the injection + tool result are consumed.
    # Wait until turn 2 has FULLY settled — its calls landed and its run
    # task drained — before publishing turn 3.  If turn 3's UserMessage
    # arrives while turn 2's FSM is still RUNNING the engine injects it
    # into turn 2 instead of starting a new turn, so compress_context
    # never fires and the final poll below would time out.
    await bus.publish(UserInject(text="also note this"))
    await _wait_until(
        lambda: (
            len(mock_client.call_history) >= 3
            and engine.turns.turn_state.lease.run_task is not None
            and engine.turns.turn_state.lease.run_task.done()
        )
    )

    # Turn 3: compress_context folds turn 1, then final text (calls #4, #5).
    # Wait until the fold has settled and the compress event has fired —
    # the exact conditions the assertions below check.
    await bus.publish(UserMessage(text="compress"))
    await _wait_until(
        lambda: (
            len(mock_client.call_history) >= 5
            and len(engine.current.loaded.bindings.backend.history_state.get("compressed_msgs", [])) == 1
            and len(compress_events) == 1
        )
    )

    state = engine.current.loaded.bindings.backend.history_state

    # ---- Verify compression ----
    compressed = state.get("compressed_msgs", [])
    assert len(compressed) == 1
    assert compressed[0].marker_id == "turn_1"
    assert len(compress_events) == 1

    # ---- Verify intermediate text response was processed ----
    # The agent produced a response with text + tool_calls in one
    # response (Turn 2). MockChatClient puts both in one assistant
    # message; FunctionInvocationLayer must execute the function_calls.
    history = mock_client.call_history
    assert len(history) >= 4, f"Expected >= 4 API calls, got {len(history)}"

    # ---- Verify API saw compressed context after the fold ----
    # Comparing against the previous call's count is interleaving-dependent:
    # the mid-turn injection may persist as its own history message or ride
    # along ephemerally, shifting both wire views by one. Assert against the
    # worst-case uncompressed total and the fold semantics directly instead.
    msgs_at_final = len(history[-1][0])
    # user1+asst1 + user2+asst2(intermediate)+tool+injected+asst2(final) +
    # user3+asst(compress_call)+tool(compress_result) = 10
    uncompressed_count = 10
    assert msgs_at_final < uncompressed_count, f"Expected compression to reduce context, final saw {msgs_at_final}"
    final_texts = [m.text or "" for m in history[-1][0]]
    assert any("Turn 1 done" in text for text in final_texts), "compression summary missing from final wire view"
    assert not any(text.startswith("start") or text == "Response 1" for text in final_texts), (
        "turn 1 messages still on the wire after compression"
    )

    # ---- Session save ----
    import tempfile
    from pathlib import Path

    from chrys.service.state.store import JsonFileStateStore

    # ignore_cleanup_errors: on Windows the state store's .write.lock file
    # can still be held by the OS at rmtree time even after the lock handle
    # is closed, raising PermissionError during cleanup. Every save/restore
    # assertion runs before cleanup, so tolerating a cleanup failure is safe.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = JsonFileStateStore(Path(tmpdir))
        engine_services(engine).persistence._state_store = store

        session_id = engine.session.session_id
        assert session_id is not None
        await engine.writer.save_current_session()

        loaded = await store.load_session(session_id)
        assert loaded is not None

        # Verify compressed blocks survived
        assert len(loaded.get("compressed_msgs", [])) == 1


# ---------------------------------------------------------------------------
# Test 2: Full Chrys engine → executor → agent pipeline
# ---------------------------------------------------------------------------


async def _collect_event(events_list: list, event: object) -> None:
    """Async event handler that appends to a list."""
    events_list.append(event)


@pytest.mark.asyncio
async def test_chrys_engine_full_pipeline(agent_engine, monkeypatch: pytest.MonkeyPatch):
    """End-to-end: Engine + TurnBindings + Agent + MockChatClient → AgentMessage.

    Verifies:
    1. Engine starts and publishes SessionReady
    2. UserMessage triggers agent execution
    3. Final AgentMessage is published with correct text
    """
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import (
        InvocationMessage,
        SessionReady,
        UserMessage,
    )
    from chrys.service.profiles.agents.schema import AgentProfile, ApprovalConfig, CompactionConfig, ToolsConfig

    events_received: list[object] = []

    bus = EventBus()
    await bus.subscribe(SessionReady, lambda e: _collect_event(events_received, e))
    await bus.subscribe(InvocationMessage, lambda e: _collect_event(events_received, e))

    profile = AgentProfile(
        name="test",
        instructions="You are a test assistant.",
        tools=ToolsConfig(builtins=["shell"]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(),
    )

    settings = Settings()

    mock_responses = [
        MockResponse(text="Hello! I'm the test assistant."),
    ]

    engine = agent_engine(bus, settings=settings)

    import chrys.orchestration.engine.build.builder as builder_module

    mock_client = MockChatClient(responses=mock_responses)

    monkeypatch.setattr(
        builder_module, "create_client", create_autospec(builder_module.create_client, return_value=mock_client)
    )
    await engine.start(profile)

    # Verify SessionReady
    ready_events = [e for e in events_received if isinstance(e, SessionReady)]
    assert len(ready_events) == 1
    assert ready_events[0].agent_profile == "test"

    # Send a user message
    await bus.publish(UserMessage(text="Say hello"))

    # Poll for the final AgentMessage instead of a fixed sleep: on slower CI
    # runners (Windows) the run pipeline can take longer than 0.2s to publish,
    # which made the fixed-sleep assertion flaky.
    await _wait_until(
        lambda: any(
            (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final for e in events_received
        )
    )

    # Verify AgentMessage
    agent_msgs = [e for e in events_received if (isinstance(e, InvocationMessage) and e.origin.kind == "turn")]
    final_msgs = [m for m in agent_msgs if m.is_final]
    assert len(final_msgs) == 1
    assert "Hello" in final_msgs[0].text

    assert mock_client.call_count == 1


@pytest.mark.asyncio
async def test_chrys_engine_with_tool_calls(agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Engine handles mock tool-call responses end-to-end.

    The mock agent makes detected shell-name tool calls, the shell tool executes them,
    and the final text response is published as AgentMessage.
    """
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import (
        InvocationMessage,
        SessionReady,
        UserMessage,
    )
    from chrys.service.profiles.agents.schema import AgentProfile, ApprovalConfig, CompactionConfig, ToolsConfig

    events_received: list[object] = []
    bus = EventBus()
    await bus.subscribe(SessionReady, lambda e: _collect_event(events_received, e))
    await bus.subscribe(InvocationMessage, lambda e: _collect_event(events_received, e))

    profile = AgentProfile(
        name="researcher",
        instructions="You are a research assistant.",
        tools=ToolsConfig(builtins=["shell"]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(),
    )

    settings = Settings()

    # 3 tool calls + final text — tool name is the detected shell name (e.g. "zsh", "bash", "pwsh")
    from chrys.foundation.platform import detect_platform

    shell_name = detect_platform().shell.name

    mock_responses: list[MockResponse] = [
        MockResponse(tool_calls=[(shell_name, "c1", {"command": "echo hello"})]),
        MockResponse(tool_calls=[(shell_name, "c2", {"command": "echo world"})]),
        MockResponse(tool_calls=[(shell_name, "c3", {"command": "echo done"})]),
        MockResponse(text="Executed 3 commands successfully."),
    ]

    mock_client = MockChatClient(responses=mock_responses)

    import chrys.orchestration.engine.build.builder as builder_module

    monkeypatch.setattr(
        builder_module, "create_client", create_autospec(builder_module.create_client, return_value=mock_client)
    )
    engine = agent_engine(bus, settings=settings)
    await engine.start(profile)

    await bus.publish(UserMessage(text="Run some commands"))

    # Poll for the asserted conditions — a final AgentMessage plus all 4
    # model calls (3 tool + 1 final) — rather than a fixed 1s sleep that
    # flakes on slow Windows CI.
    await _wait_until(
        lambda: (
            any((isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final for e in events_received)
            and mock_client.call_count == 4
        )
    )

    # Verify final response
    agent_msgs = [e for e in events_received if (isinstance(e, InvocationMessage) and e.origin.kind == "turn")]
    final_msgs = [m for m in agent_msgs if m.is_final]
    assert len(final_msgs) >= 1
    assert "3 commands" in final_msgs[-1].text

    # Mock client should have been called 4 times (3 tool + 1 final)
    assert mock_client.call_count == 4

    # Verify tool results appear in the messages sent to the model
    # The last call should contain function_result messages from the shell commands
    last_msgs, _ = mock_client.call_history[-1]
    func_results = [m for m in last_msgs if any(c.type == "function_result" for c in m.contents)]
    assert len(func_results) >= 3


# ---------------------------------------------------------------------------
# Sub-agent Phase 4 wiring (regression): SubAgentTools.register must bind
# a LAST_WORDS generator + SystemReminderMiddleware onto the sub-agent's
# compaction strategy.  Without these bindings Phase 4's ``have_note``
# guard is always False and the sub-agent silently blows past the context
# limit when its tool-loop accumulates too much data.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sub_agent_register_wires_phase4_collaborators() -> None:
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.models.session_env import SessionEnvironment
    from chrys.orchestration.sub_agents.tools import SubAgentTools
    from chrys.service.context.compaction.last_words import LastWordsGenerator
    from chrys.service.profiles.agents.schema import (
        AgentProfile,
        ApprovalConfig,
        CompactionConfig,
        SubAgentRef,
        ToolsConfig,
    )

    profile = AgentProfile(
        name="Explore",
        instructions="Search.",
        tools=ToolsConfig(builtins=[]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
    )
    # Use a mock provider for the parent so register() doesn't try to
    # construct a real OpenAI client (which the openai SDK now refuses
    # without credentials, breaking this test on environments without
    # OPENAI_API_KEY set).
    from chrys.service.profiles.models.schema import ModelProfile

    parent_profile = ModelProfile(
        id="parent",
        name="parent",
        provider="mock",
        model_id="mock",
        max_context_tokens=100_000,
    )
    runtime = SessionEnvironment.capture()

    tools = SubAgentTools(max_total_concurrency=2, parent_approval=profile.approval)
    try:
        await tools.register(
            SubAgentRef(profile="Explore", tool_name="explore"),
            profile,
            runtime,
            settings=Settings(),
            fallback_profile=parent_profile,
        )

        strategy = tools._context_managers["explore"].compaction_strategy
        # Phase 4 collaborators must be bound — not None.
        assert strategy._reminder_middleware is not None, (
            "Sub-agent compaction strategy has no reminder_middleware — Phase 4 would silently no-op"
        )
        assert strategy._last_words_generator is not None, "Sub-agent compaction strategy has no last_words_generator"
        assert isinstance(strategy._last_words_generator, LastWordsGenerator)
        # The LAST_WORDS state must be the one the chain's reminder
        # middleware renders, so the strategy's updates surface on the next
        # LLM call.
        assert strategy._last_words_state is not None
        assert strategy._reminder_middleware.renders_last_words(strategy._last_words_state)
    finally:
        await tools.cleanup()


@pytest.mark.asyncio
async def test_sub_agent_phase4_drops_groups_with_generated_note() -> None:
    """End-to-end: with collaborators wired, a sub-agent compaction strategy
    actually drops current-turn groups during Phase 4.

    Mirrors the single-turn sub-agent scenario from test_compaction, but
    drives it through the exact wiring path that SubAgentTools.register
    sets up — catching a regression where the sub-agent path forgets to
    bind one of the collaborators.
    """
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.models.session_env import SessionEnvironment
    from chrys.orchestration.sub_agents.tools import SubAgentTools
    from chrys.service.profiles.agents.schema import (
        AgentProfile,
        ApprovalConfig,
        CompactionConfig,
        SubAgentRef,
        ToolsConfig,
    )

    # Use the default CompactionConfig (compaction enabled) so the
    # sub-agent's ContextManager builds a strategy with compaction_enabled=True.
    profile = AgentProfile(
        name="Explore",
        instructions="Search.",
        tools=ToolsConfig(builtins=[]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(),
    )
    # ModelProfile.max_context_tokens defaults to 200_000 — override with a tiny
    # window via the parent ModelProfile so the strategy's compaction_enabled
    # stays True but usage trivially exceeds the trigger.
    from chrys.service.profiles.models.schema import ModelProfile

    parent_profile = ModelProfile(
        id="parent",
        name="parent",
        provider="mock",
        model_id="mock",
        max_context_tokens=500,
    )
    runtime = SessionEnvironment.capture()

    tools = SubAgentTools(max_total_concurrency=2, parent_approval=profile.approval)
    try:
        await tools.register(
            SubAgentRef(profile="Explore", tool_name="explore"),
            profile,
            runtime,
            settings=Settings(),
            fallback_profile=parent_profile,
        )

        strategy = tools._context_managers["explore"].compaction_strategy

        # Swap the real LAST_WORDS generator for a stub so we don't need an
        # actual LLM.  The reminder middleware is already bound by register().
        class _StubGen:
            async def generate(  # type: ignore[no-untyped-def]
                self,
                _scoped_groups,
                _previous_last_words,
                **_kwargs,
            ):
                return "[stub progress note]"

            async def publish_breaker_trip(self, _failure_reason: str) -> None:
                return None

            async def publish_committed(self) -> None:
                return None

        strategy.set_last_words_generator(_StubGen())

        # Build a pathologically-large single turn so Phase 4 trips.
        from chrys.kernel import Content as _C
        from chrys.kernel import Message as _M

        msgs: list[_M] = [_M(role="user", contents=[_C.from_text("start")])]
        for i in range(6):
            msgs.append(_M(role="assistant", contents=[_C.from_function_call(f"c{i}", f"t_{i}", arguments={})]))
            msgs.append(_M(role="tool", contents=[_C.from_function_result(f"c{i}", result="x" * 2000)]))
        msgs.append(_M(role="assistant", contents=[_C.from_text("done")]))

        from chrys.kernel import EXCLUDE_REASON_KEY, EXCLUDED_KEY

        changed = await strategy(msgs)
        assert changed
        assert any(m.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop" for m in msgs), (
            "Sub-agent Phase 4 failed to drop current-turn tool calls"
        )
        # User message must be preserved.
        assert not msgs[0].additional_properties.get(EXCLUDED_KEY, False)
        # The LAST_WORDS state must hold the generated note.
        assert strategy._last_words_state.get_last_words() == "[stub progress note]"
    finally:
        await tools.cleanup()
