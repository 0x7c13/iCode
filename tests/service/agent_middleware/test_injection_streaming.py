# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Regression: injection persistence under streaming.

The user-reported bug: a message injected mid-tool-call shows up in the
correct position in the live UI, but in the persisted session history the
injected message is moved to the END (after the final assistant response).

``InjectionMiddleware`` must anchor consumed injections to the right message
when ``persist_consumed_injections`` writes session history. An anchor based
only on ``id(context.messages[-1].contents)`` fails if response assembly copies
that list: identity matching falls through to ``len(messages)`` and appends
the injection at the end. The kernel streaming finalizers retain the loop's
assembled messages; injection placement must stay correct across that boundary.

These tests exercise Agent's streaming tool loop and injection persistence
together. The unit tests in ``test_injection.py`` build state by hand and do
not catch this class of regression.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from chrys.kernel import Agent, AgentSession, Content, Message
from chrys.kernel.loop import ConsumedInjectionMessageProbe
from chrys.service.agent_middleware.injection import ConsumedInjection, InjectionMiddleware
from chrys.service.context.providers.history import CompressibleHistoryProvider
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.session.history import SessionHistoryManager


async def _run_injection_case(
    *,
    stream: bool,
    responses: list[MockResponse],
    tool_factory: Callable[[InjectionMiddleware], list[Callable[..., str]]],
    initial_state: dict[str, Any] | None = None,
    prequeue: list[str] | None = None,
) -> tuple[list[Message], list[ConsumedInjection]]:
    captured: list[ConsumedInjection] = []

    async def _on_consumed(injection: ConsumedInjection) -> None:
        captured.append(injection)

    inject = InjectionMiddleware()
    inject.set_on_consumed(_on_consumed)
    for text in prequeue or []:
        inject.queue(text)

    history = CompressibleHistoryProvider()
    state: dict[str, Any] = initial_state if initial_state is not None else {}
    session = AgentSession()
    session.state[history.source_id] = state

    agent = Agent(
        client=MockChatClient(responses=responses),
        instructions="test",
        tools=tool_factory(inject),
        context_providers=[history],
        middleware=[inject],
    )

    async with agent:
        if stream:
            response_stream = agent.run("q", session=session, stream=True)
            async for _ in response_stream:
                pass
            await response_stream.get_final_response()
        else:
            await agent.run("q", session=session)

    mgr = SessionHistoryManager()
    mgr.bind(state)
    mgr.persist_consumed_injections(captured)

    messages = state["messages"]
    assert isinstance(messages, list)
    return messages, captured


def _injected_indices(messages: list[Message]) -> list[int]:
    return [i for i, msg in enumerate(messages) if msg.additional_properties.get("_injected")]


def _assistant_text_indices(messages: list[Message], text: str) -> list[int]:
    return [
        i
        for i, msg in enumerate(messages)
        if msg.role == "assistant" and any(c.type == "text" and c.text == text for c in msg.contents)
    ]


def _tool_indices(messages: list[Message], *call_ids: str) -> list[int]:
    wanted = set(call_ids)
    return [
        i
        for i, msg in enumerate(messages)
        if msg.role == "tool" and any(c.type == "function_result" and c.call_id in wanted for c in msg.contents)
    ]


def _no_tools(_inject: InjectionMiddleware) -> list[Callable[..., str]]:
    return []


def _single_inject_tool(inject: InjectionMiddleware) -> list[Callable[..., str]]:
    def _dummy_tool() -> str:
        inject.queue("mid-tool injection")
        return "ok"

    return [_dummy_tool]


def _double_inject_tool(inject: InjectionMiddleware) -> list[Callable[..., str]]:
    def _dummy_tool() -> str:
        inject.queue("first injection")
        inject.queue("second injection")
        return "ok"

    return [_dummy_tool]


def _multi_tool_inject_first(inject: InjectionMiddleware) -> list[Callable[..., str]]:
    def _first_tool() -> str:
        inject.queue("after both tools")
        return "one"

    def _second_tool() -> str:
        return "two"

    return [_first_tool, _second_tool]


def _two_iteration_inject_tools(inject: InjectionMiddleware) -> list[Callable[..., str]]:
    def _first_tool() -> str:
        inject.queue("after first tool")
        return "first result"

    def _second_tool() -> str:
        inject.queue("after second tool")
        return "second result"

    return [_first_tool, _second_tool]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
async def test_consumed_injection_remains_on_later_tool_requests(stream: bool) -> None:
    """A consumed note remains in every later request in the same tool loop."""
    inject = InjectionMiddleware()

    def _first_tool() -> str:
        inject.queue("KEEP THIS INSTRUCTION")
        return "first"

    def _second_tool() -> str:
        return "second"

    client = MockChatClient(
        responses=[
            MockResponse(tool_calls=[("_first_tool", "c1", {})]),
            MockResponse(tool_calls=[("_second_tool", "c2", {})]),
            MockResponse(text="done"),
        ]
    )
    agent = Agent(client=client, instructions="test", tools=[_first_tool, _second_tool], middleware=[inject])
    client_kwargs = {
        "consumed_injection_message_probe": ConsumedInjectionMessageProbe(inject.drain_consumed_injection_messages)
    }

    async with agent:
        if stream:
            response_stream = agent.run("q", stream=True, client_kwargs=client_kwargs)
            async for _ in response_stream:
                pass
            await response_stream.get_final_response()
        else:
            await agent.run("q", client_kwargs=client_kwargs)

    wire_texts = [
        [message.text for message in messages if message.role == "user"] for messages, _ in client.call_history
    ]
    assert wire_texts == [["q"], ["q", "KEEP THIS INSTRUCTION"], ["q", "KEEP THIS INSTRUCTION"]]


@pytest.mark.asyncio
async def test_injection_persisted_after_anchor_under_streaming() -> None:
    """Mid-tool-call injection lands AFTER its anchor (tool result), not at end.

    Sequence (streaming):
      iter 1: assistant -> tool_call dummy(c1)
              the tool, while running, queues an injection on the middleware
              (this simulates the user typing while a tool is in flight).
              tool returns; result appended to context.messages
      ** InjectionMiddleware.process fires before iter 2; consumes the
         queued text and captures an anchor for the c1 tool result **
      iter 2: assistant -> final text
    Expected ``state['messages']`` order:
      [user("q"), asst(call=c1), tool(result=c1), INJECTED, asst("done")]
    """
    captured: list[ConsumedInjection] = []

    async def _on_consumed(injection: ConsumedInjection) -> None:
        captured.append(injection)

    inject = InjectionMiddleware()
    inject.set_on_consumed(_on_consumed)

    def _dummy_tool() -> str:
        # Mimic "user types while tool runs": queue an injection that the
        # middleware will consume before the next LLM call.
        inject.queue("mid-tool injection")
        return "ok"

    history = CompressibleHistoryProvider()
    state: dict[str, Any] = {}

    mock = MockChatClient(
        responses=[
            MockResponse(tool_calls=[("_dummy_tool", "c1", {})]),
            MockResponse(text="done"),
        ]
    )

    session = AgentSession()
    session.state[history.source_id] = state

    agent = Agent(
        client=mock,
        instructions="test",
        tools=[_dummy_tool],
        context_providers=[history],
        middleware=[inject],
    )

    stream = agent.run("q", session=session, stream=True)
    async for _ in stream:
        pass
    await stream.get_final_response()

    # Replay what AgentEngine._post_run does for injections.
    mgr = SessionHistoryManager()
    mgr.bind(state)
    mgr.persist_consumed_injections(captured)

    messages = state["messages"]

    # The injection MUST appear before the final assistant text response.
    inj_indices = [i for i, m in enumerate(messages) if m.additional_properties.get("_injected")]
    assert len(inj_indices) == 1, f"expected one injection, got {inj_indices}"

    # Find final assistant text (non-empty).
    final_text_idx = None
    for i, m in enumerate(messages):
        if m.role == "assistant" and any(c.type == "text" and (c.text or "").strip() for c in m.contents):
            final_text_idx = i
    assert final_text_idx is not None
    assert inj_indices[0] < final_text_idx, (
        "Injection landed AFTER the final assistant response — streaming "
        "path lost the placement anchor.  See InjectionMiddleware + "
        "persist_consumed_injections."
    )

    # The injection should sit immediately after the tool result for c1.
    tool_idx = next(
        i
        for i, m in enumerate(messages)
        if m.role == "tool" and any(getattr(c, "call_id", None) == "c1" for c in m.contents)
    )
    assert inj_indices[0] == tool_idx + 1, (
        f"Expected injection at idx {tool_idx + 1} (right after tool result), got {inj_indices[0]}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
async def test_no_tool_turn_has_no_consumed_injection(stream: bool) -> None:
    """A one-shot text response has no mid-turn opportunity to consume queued injections."""
    messages, captured = await _run_injection_case(
        stream=stream,
        responses=[MockResponse(text="done")],
        tool_factory=_no_tools,
    )

    assert captured == []
    assert _injected_indices(messages) == []
    assert _assistant_text_indices(messages, "done")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
async def test_single_tool_injection_persists_before_final_response(stream: bool) -> None:
    """One consumed injection lands after its tool result and before final assistant text."""
    messages, captured = await _run_injection_case(
        stream=stream,
        responses=[
            MockResponse(tool_calls=[("_dummy_tool", "c1", {})]),
            MockResponse(text="done"),
        ],
        tool_factory=_single_inject_tool,
    )

    assert [injection.text for injection in captured] == ["mid-tool injection"]
    inj_indices = _injected_indices(messages)
    assert len(inj_indices) == 1
    tool_idx = _tool_indices(messages, "c1")[0]
    final_idx = _assistant_text_indices(messages, "done")[-1]
    assert inj_indices[0] == tool_idx + 1
    assert inj_indices[0] < final_idx


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
async def test_multiple_injections_same_turn_preserve_queue_order(stream: bool) -> None:
    """Multiple queued injections sharing one anchor keep user queue order."""
    messages, captured = await _run_injection_case(
        stream=stream,
        responses=[
            MockResponse(tool_calls=[("_dummy_tool", "c1", {})]),
            MockResponse(text="done"),
        ],
        tool_factory=_double_inject_tool,
    )

    assert [injection.text for injection in captured] == ["first injection", "second injection"]
    inj_indices = _injected_indices(messages)
    assert len(inj_indices) == 2
    tool_idx = _tool_indices(messages, "c1")[0]
    final_idx = _assistant_text_indices(messages, "done")[-1]
    assert inj_indices == [tool_idx + 1, tool_idx + 2]
    assert [messages[i].text for i in inj_indices] == ["first injection", "second injection"]
    assert inj_indices[-1] < final_idx


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
async def test_multiple_tool_results_anchor_after_last_tool_result(stream: bool) -> None:
    """An injection queued by one of several tools lands after the completed tool-result block."""
    messages, captured = await _run_injection_case(
        stream=stream,
        responses=[
            MockResponse(tool_calls=[("_first_tool", "c1", {}), ("_second_tool", "c2", {})]),
            MockResponse(text="done"),
        ],
        tool_factory=_multi_tool_inject_first,
    )

    assert [injection.text for injection in captured] == ["after both tools"]
    inj_idx = _injected_indices(messages)[0]
    last_tool_idx = max(_tool_indices(messages, "c1", "c2"))
    final_idx = _assistant_text_indices(messages, "done")[-1]
    assert inj_idx == last_tool_idx + 1
    assert inj_idx < final_idx


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
async def test_duplicate_call_id_from_previous_turn_anchors_current_result(stream: bool) -> None:
    """A previous turn with the same call_id must not steal the current injection anchor."""
    initial_state: dict[str, Any] = {
        "messages": [
            Message("user", [Content.from_text("old q")]),
            Message("assistant", [Content.from_function_call("dup", "_dummy_tool", arguments={})]),
            Message("tool", [Content.from_function_result("dup", result="old result")]),
            Message("assistant", [Content.from_text("old done")]),
        ]
    }

    messages, captured = await _run_injection_case(
        stream=stream,
        responses=[
            MockResponse(tool_calls=[("_dummy_tool", "dup", {})]),
            MockResponse(text="done"),
        ],
        tool_factory=_single_inject_tool,
        initial_state=initial_state,
    )

    assert [injection.text for injection in captured] == ["mid-tool injection"]
    inj_idx = _injected_indices(messages)[0]
    current_tool_idx = max(_tool_indices(messages, "dup"))
    final_idx = _assistant_text_indices(messages, "done")[-1]
    assert inj_idx == current_tool_idx + 1
    assert inj_idx < final_idx


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
async def test_prequeued_injection_before_first_llm_call_anchors_after_user(stream: bool) -> None:
    """An injection consumed before the first model call anchors after the current user message."""
    messages, captured = await _run_injection_case(
        stream=stream,
        responses=[MockResponse(text="done")],
        tool_factory=_no_tools,
        prequeue=["before first call"],
    )

    assert [injection.text for injection in captured] == ["before first call"]
    inj_idx = _injected_indices(messages)[0]
    final_idx = _assistant_text_indices(messages, "done")[-1]
    assert messages[inj_idx - 1].role == "user"
    assert inj_idx < final_idx


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
async def test_injections_across_multiple_tool_iterations_keep_distinct_anchors(stream: bool) -> None:
    """Injections consumed on different tool-loop iterations persist after their own tool results."""
    messages, captured = await _run_injection_case(
        stream=stream,
        responses=[
            MockResponse(tool_calls=[("_first_tool", "c1", {})]),
            MockResponse(tool_calls=[("_second_tool", "c2", {})]),
            MockResponse(text="done"),
        ],
        tool_factory=_two_iteration_inject_tools,
    )

    assert [injection.text for injection in captured] == ["after first tool", "after second tool"]
    c1_tool_idx = _tool_indices(messages, "c1")[0]
    c2_tool_idx = _tool_indices(messages, "c2")[0]
    inj_indices = _injected_indices(messages)
    final_idx = _assistant_text_indices(messages, "done")[-1]
    assert inj_indices == [c1_tool_idx + 1, c2_tool_idx + 1]
    assert [messages[i].text for i in inj_indices] == ["after first tool", "after second tool"]
    assert inj_indices[-1] < final_idx


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
async def test_reused_call_id_with_distinct_results_in_same_turn_keeps_each_anchor(stream: bool) -> None:
    """If a provider reuses a call_id in one turn, each injection still follows its own result."""
    messages, captured = await _run_injection_case(
        stream=stream,
        responses=[
            MockResponse(tool_calls=[("_first_tool", "dup", {})]),
            MockResponse(tool_calls=[("_second_tool", "dup", {})]),
            MockResponse(text="done"),
        ],
        tool_factory=_two_iteration_inject_tools,
    )

    assert [injection.text for injection in captured] == ["after first tool", "after second tool"]
    tool_indices = _tool_indices(messages, "dup")
    assert len(tool_indices) == 2
    inj_indices = _injected_indices(messages)
    final_idx = _assistant_text_indices(messages, "done")[-1]
    assert inj_indices == [tool_indices[0] + 1, tool_indices[1] + 1]
    assert [messages[i].text for i in inj_indices] == ["after first tool", "after second tool"]
    assert inj_indices[-1] < final_idx
