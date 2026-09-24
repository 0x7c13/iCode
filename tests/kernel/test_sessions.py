# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Acceptance tests for the chrys-owned session/context/history layer.

Pins THE deliberate divergence:
``SessionContext.extend_messages`` appends the caller's ``Message`` objects
as-is — no copy, no fresh ``additional_properties`` dict, no ``_attribution``
stamp. Includes the local-history sentinel compatibility string.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from chrys.kernel.middleware import ChatMiddleware, FunctionMiddleware
from chrys.kernel.sessions import (
    LOCAL_HISTORY_CONVERSATION_ID,
    AgentSession,
    ContextProvider,
    HistoryProvider,
    InMemoryHistoryProvider,
    ServiceFallbackHistoryProvider,
    SessionContext,
    is_local_history_conversation_id,
)
from chrys.kernel.types import AgentResponse, Message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(text: str, role: str = "user") -> Message:
    return Message(role=role, contents=[text])


class _Source:
    """Provider-like object for the ``source: object`` extend_messages form."""

    source_id = "obj_source"


class _ChatMW(ChatMiddleware):
    async def process(self, context: Any, call_next: Any) -> None:
        await call_next()


class _FuncMW(FunctionMiddleware):
    async def process(self, context: Any, call_next: Any) -> None:
        await call_next()


class _ForeignChatMW:
    async def process(self, context: Any, call_next: Any) -> None:  # pragma: no cover - never invoked
        await call_next()


class _Tool:
    def __init__(self, additional_properties: Any = None) -> None:
        if additional_properties is not None:
            self.additional_properties = additional_properties


class _RecordingHistoryProvider(HistoryProvider):
    """HistoryProvider subclass recording the abstract-method traffic."""

    def __init__(self, source_id: str = "hist", *, preload: list[Message] | None = None, **flags: Any) -> None:
        super().__init__(source_id, **flags)
        self._preload = list(preload or [])
        self.get_calls: list[tuple[str | None, dict[str, Any] | None]] = []
        self.save_calls: list[tuple[str | None, list[Message], dict[str, Any] | None]] = []

    async def get_messages(
        self, session_id: str | None, *, state: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Message]:
        self.get_calls.append((session_id, state))
        return list(self._preload)

    async def save_messages(
        self,
        session_id: str | None,
        messages: Any,
        *,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.save_calls.append((session_id, list(messages), state))


def _context_with_response(response_messages: list[Message], **kwargs: Any) -> SessionContext:
    ctx = SessionContext(input_messages=[], **kwargs)
    ctx._response = AgentResponse(messages=response_messages, response_id=None)
    return ctx


# ---------------------------------------------------------------------------
# A. SessionContext
# ---------------------------------------------------------------------------


class TestSessionContext:
    def test_ctor_defaults(self) -> None:
        inputs = [_msg("hi")]
        ctx = SessionContext(input_messages=inputs)

        assert ctx.session_id is None
        assert ctx.service_session_id is None
        assert ctx.input_messages is inputs
        assert ctx.context_messages == {}
        assert ctx.instructions == []
        assert ctx.tools == []
        assert ctx.middleware == {}
        assert ctx.response is None
        assert ctx.options == {}
        assert ctx.metadata == {}

    def test_ctor_explicit_params_hold_caller_containers(self) -> None:
        context_messages = {"a": [_msg("ctx")]}
        instructions = ["be brief"]
        tools = [object()]
        options = {"model": "m"}
        metadata = {"k": "v"}
        ctx = SessionContext(
            session_id="sid",
            service_session_id="svc",
            input_messages=[],
            context_messages=context_messages,
            instructions=instructions,
            tools=tools,
            options=options,
            metadata=metadata,
        )

        assert ctx.session_id == "sid"
        assert ctx.service_session_id == "svc"
        assert ctx.context_messages is context_messages
        assert ctx.instructions is instructions
        assert ctx.tools is tools
        assert ctx.options is options
        assert ctx.metadata is metadata

    def test_ctor_middleware_routes_through_extend(self) -> None:
        chat, func = _ChatMW(), _FuncMW()
        ctx = SessionContext(input_messages=[], middleware={"p1": [chat, func]})

        assert ctx.middleware == {"p1": [chat, func]}
        assert ctx.get_middleware() == [chat, func]

    def test_ctor_middleware_invalid_raises(self) -> None:
        with pytest.raises(TypeError):
            SessionContext(input_messages=[], middleware={"p1": [lambda c, n: None]})  # type: ignore[list-item]

    def test_extend_messages_appends_same_objects(self) -> None:
        original = _msg("hello", role="assistant")
        ctx = SessionContext(input_messages=[])

        ctx.extend_messages("provider", [original])

        (stored,) = ctx.context_messages["provider"]
        assert stored is original

    def test_extend_messages_shares_additional_properties_without_attribution(self) -> None:
        original = _msg("hello", role="assistant")
        ctx = SessionContext(input_messages=[])

        ctx.extend_messages("provider", [original])

        (stored,) = ctx.context_messages["provider"]
        assert stored.additional_properties is original.additional_properties
        assert "_attribution" not in stored.additional_properties

        # The whole point of the no-copy semantics: flags set on the stored
        # view are visible on the caller's original (they are one object).
        stored.additional_properties["_excluded"] = True
        assert original.additional_properties["_excluded"] is True

    def test_extend_messages_object_source_uses_source_id(self) -> None:
        ctx = SessionContext(input_messages=[])
        message = _msg("ctx")

        ctx.extend_messages(_Source(), [message])

        assert ctx.context_messages == {"obj_source": [message]}

    def test_extend_messages_insertion_order_and_same_source_append(self) -> None:
        ctx = SessionContext(input_messages=[])
        a1, b1, a2 = _msg("a1"), _msg("b1"), _msg("a2")

        ctx.extend_messages("a", [a1])
        ctx.extend_messages("b", [b1])
        ctx.extend_messages("a", [a2])

        assert ctx.context_messages == {"a": [a1, a2], "b": [b1]}
        # Flattening preserves provider execution (insertion) order.
        assert ctx.get_messages() == [a1, a2, b1]

    def test_get_messages_filters_sources(self) -> None:
        ctx = SessionContext(input_messages=[])
        a, b = _msg("a"), _msg("b")
        ctx.extend_messages("a", [a])
        ctx.extend_messages("b", [b])

        assert ctx.get_messages(sources={"a"}) == [a]

    def test_get_messages_filters_exclude_sources(self) -> None:
        ctx = SessionContext(input_messages=[])
        a, b = _msg("a"), _msg("b")
        ctx.extend_messages("a", [a])
        ctx.extend_messages("b", [b])

        assert ctx.get_messages(exclude_sources={"a"}) == [b]

    def test_get_messages_include_input(self) -> None:
        user = _msg("question")
        ctx = SessionContext(input_messages=[user])
        a = _msg("ctx")
        ctx.extend_messages("a", [a])

        assert ctx.get_messages(include_input=True) == [a, user]
        assert ctx.get_messages() == [a]

    def test_get_messages_include_response(self) -> None:
        reply = _msg("answer", role="assistant")
        ctx = _context_with_response([reply])
        a = _msg("ctx")
        ctx.extend_messages("a", [a])

        assert ctx.get_messages(include_response=True) == [a, reply]

    def test_get_messages_returns_fresh_list(self) -> None:
        ctx = SessionContext(input_messages=[])
        a = _msg("a")
        ctx.extend_messages("a", [a])

        result = ctx.get_messages()
        result.append(_msg("intruder"))

        assert ctx.context_messages["a"] == [a]

    def test_extend_instructions_str_and_sequence(self) -> None:
        ctx = SessionContext(input_messages=[])

        ctx.extend_instructions("p", "one")
        ctx.extend_instructions("p", ["two", "three"])

        assert ctx.instructions == ["one", "two", "three"]

    def test_extend_tools_stamps_context_source(self) -> None:
        stamped = _Tool(additional_properties={})
        bare = _Tool()
        non_dict = _Tool(additional_properties="not-a-dict")
        ctx = SessionContext(input_messages=[])

        ctx.extend_tools("prov", [stamped, bare, non_dict])

        assert ctx.tools == [stamped, bare, non_dict]
        assert stamped.additional_properties == {"context_source": "prov"}
        assert non_dict.additional_properties == "not-a-dict"

    def test_extend_middleware_single_and_sequence_with_order(self) -> None:
        chat, func = _ChatMW(), _FuncMW()
        ctx = SessionContext(input_messages=[])

        ctx.extend_middleware("p1", chat)
        ctx.extend_middleware("p2", [func])

        assert ctx.middleware == {"p1": [chat], "p2": [func]}
        assert ctx.get_middleware() == [chat, func]

    def test_extend_middleware_rejects_plain_callable_and_foreign_instances(self) -> None:
        ctx = SessionContext(input_messages=[])

        with pytest.raises(TypeError):
            ctx.extend_middleware("p", lambda c, n: None)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            ctx.extend_middleware("p", _ForeignChatMW())  # type: ignore[arg-type]
        assert ctx.middleware == {}

    def test_response_is_read_only_property(self) -> None:
        ctx = SessionContext(input_messages=[])

        with pytest.raises(AttributeError):
            ctx.response = AgentResponse(messages=[], response_id=None)  # type: ignore[misc]

        response = AgentResponse(messages=[_msg("done", role="assistant")], response_id="rid")
        ctx._response = response
        assert ctx.response is response


# ---------------------------------------------------------------------------
# B. ContextProvider / HistoryProvider
# ---------------------------------------------------------------------------


class TestContextProvider:
    @pytest.mark.asyncio
    async def test_source_id_and_noop_hooks(self) -> None:
        provider = ContextProvider("p1")
        ctx = SessionContext(input_messages=[])
        session = AgentSession()

        assert provider.source_id == "p1"
        assert await provider.before_run(agent=object(), session=session, context=ctx, state={}) is None
        assert await provider.after_run(agent=object(), session=session, context=ctx, state={}) is None
        assert ctx.context_messages == {}


class TestHistoryProvider:
    def test_flag_defaults(self) -> None:
        provider = _RecordingHistoryProvider("h")

        assert provider.load_messages is True
        assert provider.store_inputs is True
        assert provider.store_context_messages is False
        assert provider.store_context_from is None
        assert provider.store_outputs is True

    def test_flags_explicit(self) -> None:
        provider = _RecordingHistoryProvider(
            "h",
            load_messages=False,
            store_inputs=False,
            store_context_messages=True,
            store_context_from={"x"},
            store_outputs=False,
        )

        assert provider.load_messages is False
        assert provider.store_inputs is False
        assert provider.store_context_messages is True
        assert provider.store_context_from == {"x"}
        assert provider.store_outputs is False

    def test_instantiable_without_abc_enforcement(self) -> None:
        # HistoryProvider uses @abstractmethod without ABCMeta, so direct
        # instantiation must remain valid despite the method annotations.
        provider = HistoryProvider("bare")
        assert provider.source_id == "bare"

    @pytest.mark.asyncio
    async def test_before_run_loads_history_into_context(self) -> None:
        history = [_msg("old", role="assistant")]
        provider = _RecordingHistoryProvider("hist", preload=history)
        ctx = SessionContext(session_id="sid", input_messages=[])
        state: dict[str, Any] = {}

        await provider.before_run(agent=object(), session=AgentSession(), context=ctx, state=state)

        assert provider.get_calls == [("sid", state)]
        assert ctx.context_messages["hist"] == history
        assert ctx.context_messages["hist"][0] is history[0]

    @pytest.mark.asyncio
    async def test_after_run_store_order_context_inputs_outputs(self) -> None:
        other_ctx_msg = _msg("from-other")
        user = _msg("question")
        reply = _msg("answer", role="assistant")
        provider = _RecordingHistoryProvider("hist", store_context_messages=True)
        ctx = _context_with_response([reply], session_id="sid")
        ctx.input_messages.append(user)
        ctx.extend_messages("other", [other_ctx_msg])
        ctx.extend_messages("hist", [_msg("own-history")])

        await provider.after_run(agent=object(), session=AgentSession(), context=ctx, state={})

        ((session_id, saved, _state),) = provider.save_calls
        assert session_id == "sid"
        # Order: context-to-store (own source excluded) -> inputs -> outputs.
        assert saved == [other_ctx_msg, user, reply]

    @pytest.mark.asyncio
    async def test_after_run_store_flags_matrix(self) -> None:
        user = _msg("question")
        reply = _msg("answer", role="assistant")

        no_inputs = _RecordingHistoryProvider("hist", store_inputs=False)
        ctx = _context_with_response([reply])
        ctx.input_messages.append(user)
        await no_inputs.after_run(agent=object(), session=AgentSession(), context=ctx, state={})
        assert no_inputs.save_calls[0][1] == [reply]

        no_outputs = _RecordingHistoryProvider("hist", store_outputs=False)
        ctx = _context_with_response([reply])
        ctx.input_messages.append(user)
        await no_outputs.after_run(agent=object(), session=AgentSession(), context=ctx, state={})
        assert no_outputs.save_calls[0][1] == [user]

    @pytest.mark.asyncio
    async def test_after_run_skips_save_when_nothing_to_store(self) -> None:
        provider = _RecordingHistoryProvider("hist", store_inputs=False, store_outputs=False)
        ctx = _context_with_response([_msg("answer", role="assistant")])
        ctx.input_messages.append(_msg("question"))

        await provider.after_run(agent=object(), session=AgentSession(), context=ctx, state={})

        assert provider.save_calls == []

    def test_get_context_messages_to_store_disabled(self) -> None:
        provider = _RecordingHistoryProvider("hist")
        ctx = SessionContext(input_messages=[])
        ctx.extend_messages("other", [_msg("x")])

        assert provider._get_context_messages_to_store(ctx) == []

    def test_get_context_messages_to_store_from_filter(self) -> None:
        wanted, unwanted = _msg("wanted"), _msg("unwanted")
        provider = _RecordingHistoryProvider("hist", store_context_messages=True, store_context_from={"a"})
        ctx = SessionContext(input_messages=[])
        ctx.extend_messages("a", [wanted])
        ctx.extend_messages("b", [unwanted])

        assert provider._get_context_messages_to_store(ctx) == [wanted]

    def test_get_context_messages_to_store_excludes_own_source(self) -> None:
        own, other = _msg("own"), _msg("other")
        provider = _RecordingHistoryProvider("hist", store_context_messages=True)
        ctx = SessionContext(input_messages=[])
        ctx.extend_messages("hist", [own])
        ctx.extend_messages("other", [other])

        assert provider._get_context_messages_to_store(ctx) == [other]


# ---------------------------------------------------------------------------
# C. InMemoryHistoryProvider
# ---------------------------------------------------------------------------


class TestInMemoryHistoryProvider:
    @pytest.mark.asyncio
    async def test_get_returns_fresh_list_sharing_objects(self) -> None:
        stored = [_msg("a"), _msg("b")]
        state: dict[str, Any] = {"messages": stored}
        provider = InMemoryHistoryProvider()

        result = await provider.get_messages(None, state=state)

        assert result == stored
        assert result is not stored
        assert result[0] is stored[0]

    @pytest.mark.asyncio
    async def test_get_none_state_returns_empty(self) -> None:
        provider = InMemoryHistoryProvider()
        assert await provider.get_messages("sid", state=None) == []

    @pytest.mark.asyncio
    async def test_save_appends_with_new_list(self) -> None:
        first, second = _msg("first"), _msg("second")
        original_list = [first]
        state: dict[str, Any] = {"messages": original_list}
        provider = InMemoryHistoryProvider()

        await provider.save_messages(None, [second], state=state)

        assert state["messages"] == [first, second]
        assert state["messages"] is not original_list
        assert original_list == [first]

    @pytest.mark.asyncio
    async def test_save_none_state_noop(self) -> None:
        provider = InMemoryHistoryProvider()
        await provider.save_messages("sid", [_msg("x")], state=None)

    def test_default_and_custom_source_id(self) -> None:
        assert InMemoryHistoryProvider().source_id == "in_memory"
        assert InMemoryHistoryProvider.DEFAULT_SOURCE_ID == "in_memory"
        assert InMemoryHistoryProvider("custom").source_id == "custom"

    @pytest.mark.asyncio
    async def test_hook_roundtrip_persists_same_objects(self) -> None:
        provider = InMemoryHistoryProvider()
        session = AgentSession()
        state: dict[str, Any] = {}
        user = _msg("question")
        reply = _msg("answer", role="assistant")

        run1 = _context_with_response([reply], session_id=session.session_id)
        run1.input_messages.append(user)
        await provider.after_run(agent=object(), session=session, context=run1, state=state)

        run2 = SessionContext(session_id=session.session_id, input_messages=[])
        await provider.before_run(agent=object(), session=session, context=run2, state=state)

        loaded = run2.context_messages["in_memory"]
        assert loaded == [user, reply]
        assert loaded[0] is user
        assert loaded[1] is reply
        # No-copy semantics end to end: marking the loaded view marks state.
        loaded[1].additional_properties["_excluded"] = True
        assert state["messages"][1].additional_properties["_excluded"] is True


class TestServiceFallbackHistoryProvider:
    def test_distinct_default_source_id(self) -> None:
        assert ServiceFallbackHistoryProvider.DEFAULT_SOURCE_ID == "service_history_fallback"
        assert ServiceFallbackHistoryProvider().source_id == "service_history_fallback"
        assert ServiceFallbackHistoryProvider().load_messages is True

    @pytest.mark.asyncio
    async def test_before_run_skipped_while_service_handle_live(self) -> None:
        provider = ServiceFallbackHistoryProvider()
        session = AgentSession(service_session_id="conv-live")
        state: dict[str, Any] = {"messages": [_msg("stored")]}
        context = SessionContext(session_id=session.session_id, input_messages=[])

        await provider.before_run(agent=object(), session=session, context=context, state=state)

        assert context.get_messages() == []

    @pytest.mark.asyncio
    async def test_before_run_loads_once_handle_gone(self) -> None:
        provider = ServiceFallbackHistoryProvider()
        session = AgentSession()
        stored = _msg("stored")
        state: dict[str, Any] = {"messages": [stored]}
        context = SessionContext(session_id=session.session_id, input_messages=[])

        await provider.before_run(agent=object(), session=session, context=context, state=state)

        loaded = context.context_messages["service_history_fallback"]
        assert loaded == [stored]
        assert loaded[0] is stored

    @pytest.mark.asyncio
    async def test_before_run_skipped_for_live_request_level_handle(self) -> None:
        # A live handle riding the request under any provider spelling means
        # the service already holds the conversation this run continues;
        # replaying the local shadow would ride alongside it as duplicated
        # history contaminating that remote transcript.
        provider = ServiceFallbackHistoryProvider()
        session = AgentSession()
        state: dict[str, Any] = {"messages": [_msg("stored")]}
        for options in (
            {"conversation_id": "conv-other"},
            {"previous_response_id": "resp-other"},
            {"conversation": {"id": "conv-other"}},
            {"extra_body": {"conversation_id": "conv-other"}},
            {"continuation_token": {"response_id": "resp-other"}},
        ):
            context = SessionContext(session_id=session.session_id, input_messages=[], options=options)
            await provider.before_run(agent=object(), session=session, context=context, state=state)
            assert context.get_messages() == [], f"replay must stay skipped for {options}"

    @pytest.mark.asyncio
    async def test_before_run_loads_when_request_handles_all_invalidated(self) -> None:
        # Invalidated request handles don't block the replay: the wire choke
        # point strips them, so local history must own continuity in their
        # place.
        provider = ServiceFallbackHistoryProvider()
        session = AgentSession()
        session.invalidated_service_session_ids.add("conv-old")
        stored = _msg("stored")
        state: dict[str, Any] = {"messages": [stored]}
        context = SessionContext(
            session_id=session.session_id, input_messages=[], options={"conversation_id": "conv-old"}
        )

        await provider.before_run(agent=object(), session=session, context=context, state=state)

        assert context.context_messages["service_history_fallback"] == [stored]


# ---------------------------------------------------------------------------
# D. AgentSession
# ---------------------------------------------------------------------------


class TestAgentSession:
    def test_uuid_default_unique(self) -> None:
        first, second = AgentSession(), AgentSession()

        uuid.UUID(first.session_id)
        assert first.session_id != second.session_id
        assert first.service_session_id is None

    def test_explicit_ids(self) -> None:
        session = AgentSession(session_id="sid", service_session_id="svc")

        assert session.session_id == "sid"
        assert session.service_session_id == "svc"

    def test_session_id_read_only(self) -> None:
        session = AgentSession()
        with pytest.raises(AttributeError):
            session.session_id = "other"  # type: ignore[misc]

    def test_state_is_mutable_per_instance_dict(self) -> None:
        first, second = AgentSession(), AgentSession()

        first.state["k"] = {"nested": 1}

        assert first.state == {"k": {"nested": 1}}
        assert second.state == {}


# ---------------------------------------------------------------------------
# E. Sentinel drift pin
# ---------------------------------------------------------------------------


class TestLocalHistorySentinel:
    def test_sentinel_value(self) -> None:
        assert LOCAL_HISTORY_CONVERSATION_ID == "chrys_local_history_persistence"

    def test_is_local_history_conversation_id_truth_table(self) -> None:
        assert is_local_history_conversation_id(LOCAL_HISTORY_CONVERSATION_ID) is True
        for value in ("other", "", None):
            assert is_local_history_conversation_id(value) is False


# ---------------------------------------------------------------------------
# F. End-to-end aliasing through the production stack
# ---------------------------------------------------------------------------


class TestEndToEndAliasing:
    @pytest.mark.asyncio
    async def test_wire_history_messages_are_views_with_shared_metadata(self) -> None:
        """The wire holds per-call VIEWS of state objects — flags still land.

        The view shares the state original's ``additional_properties`` dict
        and content objects, so compaction exclusion flags stamped on the
        wire copy reach stored history without any anchor pass, while the
        wrapper itself is never exposed — a client mutating a received
        history message cannot corrupt session state. Streaming-rebuilt
        current-turn messages remain the only path that still needs the
        compaction anchors; that behavior is pinned end to end by
        test_exclusions.py's
        test_current_turn_exclusions_persist_for_streaming_and_non_streaming.
        """
        from chrys.kernel import Agent
        from chrys.service.context.providers.history import CompressibleHistoryProvider
        from chrys.service.llm.mock import MockChatClient, MockResponse

        provider = CompressibleHistoryProvider()
        session = AgentSession()
        seeded = _msg("seeded history")
        session.state[provider.source_id] = {"messages": [seeded]}

        mock = MockChatClient(responses=[MockResponse(text="done")])
        agent = Agent(client=mock, instructions="t", context_providers=[provider])
        async with agent:
            await agent.run("q", session=session)

        wire_messages, _options = mock.call_history[0]
        assert not any(m is seeded for m in wire_messages)
        seeded_on_wire = next(m for m in wire_messages if m.additional_properties is seeded.additional_properties)
        assert all(x is y for x, y in zip(seeded_on_wire.contents, seeded.contents, strict=True))

        # Marking the wire view marks stored history — shared props dict.
        seeded_on_wire.additional_properties["_excluded"] = True
        state = session.state[provider.source_id]
        assert state["messages"][0] is seeded
        assert state["messages"][0].additional_properties["_excluded"] is True
