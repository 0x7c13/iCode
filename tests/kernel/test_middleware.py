# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Acceptance tests for the chrys-owned middleware foundation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from chrys.kernel.middleware import (
    ChatContext,
    ChatMiddleware,
    ChatMiddlewareLayer,
    ChatMiddlewarePipeline,
    FunctionInvocationContext,
    FunctionMiddleware,
    FunctionMiddlewarePipeline,
    MiddlewareTermination,
    split_middleware,
)
from chrys.kernel.types import ChatResponse, ChatResponseUpdate, FunctionTool, Message, ResponseStream, tool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_message(text: str = "hi") -> Message:
    return Message(role="user", contents=[text])


def _text_response(text: str = "done") -> ChatResponse:
    return ChatResponse(messages=[Message(role="assistant", contents=[text])])


def _text_update(text: str) -> ChatResponseUpdate:
    return ChatResponseUpdate(contents=[{"type": "text", "text": text}])


def _chat_context(messages: list[Message] | None = None, *, stream: bool = False) -> ChatContext:
    return ChatContext(
        client=None, messages=messages if messages is not None else [_user_message()], options=None, stream=stream
    )


def _fn_context() -> FunctionInvocationContext:
    return FunctionInvocationContext(function=object(), arguments={})


def _chat_final(response: Any, calls: list[ChatContext] | None = None) -> Callable[[ChatContext], Any]:
    """Build a chat final handler returning an awaitable (the layer's shape)."""

    def _final(context: ChatContext) -> Any:
        async def _resolve() -> Any:
            if calls is not None:
                calls.append(context)
            return response

        return _resolve()

    return _final


def _fn_final(result: Any, calls: list[FunctionInvocationContext] | None = None) -> Callable[..., Awaitable[Any]]:
    async def _final(context: FunctionInvocationContext) -> Any:
        if calls is not None:
            calls.append(context)
        return result

    return _final


class _NoopChat(ChatMiddleware):
    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        await call_next()


class _NoopFunction(FunctionMiddleware):
    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        await call_next()


class _OrderedChat(ChatMiddleware):
    def __init__(self, label: str, order: list[str]) -> None:
        self._label = label
        self._order = order

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        self._order.append(f"{self._label}-pre")
        await call_next()
        self._order.append(f"{self._label}-post")


class _OrderedFunction(FunctionMiddleware):
    def __init__(self, label: str, order: list[str]) -> None:
        self._label = label
        self._order = order

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        self._order.append(f"{self._label}-pre")
        await call_next()
        self._order.append(f"{self._label}-post")


class _ContextCapture(ChatMiddleware):
    """Captures the live context object (not a snapshot) per call."""

    def __init__(self) -> None:
        self.contexts: list[ChatContext] = []

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        self.contexts.append(context)
        await call_next()


class _InnerClient:
    """Scripted inner client conforming to SupportsGetResponse.

    Mirrors the run-contract kernel wire layer: non-streaming returns an
    awaitable, streaming returns a ResponseStream synchronously.
    """

    def __init__(
        self,
        responses: list[ChatResponse] | None = None,
        update_rounds: list[list[ChatResponseUpdate]] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.update_rounds = list(update_rounds or [])
        self.raw_messages: list[Sequence[Message]] = []
        self.wire_messages: list[list[Message]] = []
        self.wire_options: list[Any] = []
        self.wire_function_invocation_kwargs: list[Any] = []
        self.wire_extra: list[dict[str, Any]] = []

    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: Any = None,
        function_invocation_kwargs: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.raw_messages.append(messages)
        self.wire_messages.append(list(messages))
        self.wire_options.append(options)
        self.wire_function_invocation_kwargs.append(function_invocation_kwargs)
        self.wire_extra.append(dict(kwargs))
        if stream:
            updates = self.update_rounds.pop(0)

            async def _gen() -> Any:
                for update in updates:
                    yield update

            return ResponseStream(_gen(), finalizer=ChatResponse.from_updates)
        response = self.responses.pop(0)

        async def _resolve() -> ChatResponse:
            return response

        return _resolve()


# ---------------------------------------------------------------------------
# Context shape parity (upstream drift guard)
# ---------------------------------------------------------------------------


class TestContextCopyAndAliasContracts:
    def test_chat_context_attribute_semantics(self) -> None:
        messages = [_user_message()]
        metadata = {"k": "v"}
        kwargs = {"a": 1}
        hooks = [_text_update]

        context = ChatContext(
            client=None,
            messages=messages,
            options=None,
            metadata=metadata,
            kwargs=kwargs,
            stream_transform_hooks=hooks,
        )

        assert context.messages is messages  # constructor stores what it is given; copying is the layer's job
        assert context.metadata == {"k": "v"}
        assert context.metadata is not metadata
        metadata["leak"] = True
        assert "leak" not in context.metadata
        assert context.kwargs == {"a": 1}
        assert context.kwargs is not kwargs
        assert context.stream_transform_hooks == hooks
        assert context.stream_transform_hooks is not hooks
        assert context.result is None
        assert context.stream is False
        assert context.function_invocation_kwargs == {}
        assert context.stream_result_hooks == []
        assert context.stream_cleanup_hooks == []

    def test_function_context_attribute_semantics(self) -> None:
        function = object()
        arguments = {"x": 1}
        tools: list[Any] = [object()]

        context = FunctionInvocationContext(function, arguments, None, {"m": 1}, "result", {"k": 2}, tools)

        assert context.function is function
        assert context.arguments is arguments
        assert context.session is None
        assert context.metadata == {"m": 1}
        assert context.result == "result"
        assert context.kwargs == {"k": 2}
        assert context.tools is tools  # the live tools list is aliased on purpose

        bare = FunctionInvocationContext(function, arguments)
        assert bare.metadata == {}
        assert bare.kwargs == {}
        assert bare.result is None
        assert bare.tools is None

    def test_function_context_add_tools_mutates_aliased_live_list(self) -> None:
        existing = FunctionTool(func=lambda: "existing", name="existing", description="")
        added = FunctionTool(func=lambda: "added", name="added", description="")
        live_tools: list[Any] = [existing]
        context = FunctionInvocationContext(existing, {}, tools=live_tools)

        context.add_tools(added)

        assert context.tools is live_tools
        assert live_tools == [existing, added]

    def test_function_context_add_tools_same_identity_is_noop(self) -> None:
        existing = FunctionTool(func=lambda: "existing", name="existing", description="")
        live_tools: list[Any] = [existing]
        context = FunctionInvocationContext(existing, {}, tools=live_tools)

        context.add_tools([existing, existing])

        assert live_tools == [existing]

    def test_function_context_add_tools_duplicate_name_is_atomic(self) -> None:
        existing = FunctionTool(func=lambda: "existing", name="existing", description="")
        valid = FunctionTool(func=lambda: "valid", name="valid", description="")
        conflicting = FunctionTool(func=lambda: "conflict", name="existing", description="")
        live_tools: list[Any] = [existing]
        original = list(live_tools)
        context = FunctionInvocationContext(existing, {}, tools=live_tools)

        with pytest.raises(ValueError, match="Duplicate tool name 'existing'"):
            context.add_tools([valid, conflicting])

        assert live_tools == original
        assert live_tools[0] is existing

    def test_function_context_add_tools_duplicate_within_batch_is_atomic(self) -> None:
        existing = FunctionTool(func=lambda: "existing", name="existing", description="")
        first = FunctionTool(func=lambda: "first", name="duplicate", description="")
        second = FunctionTool(func=lambda: "second", name="duplicate", description="")
        live_tools: list[Any] = [existing]
        context = FunctionInvocationContext(existing, {}, tools=live_tools)

        with pytest.raises(ValueError, match="Duplicate tool name 'duplicate'"):
            context.add_tools([first, second])

        assert live_tools == [existing]

    def test_function_context_remove_tools_accepts_names_objects_and_callables(self) -> None:
        def callable_tool() -> str:
            return "callable"

        existing = FunctionTool(func=lambda: "existing", name="existing", description="")
        removable = FunctionTool(func=lambda: "removable", name="removable", description="")
        callable_wrapped = tool(callable_tool)
        assert isinstance(callable_wrapped, FunctionTool)
        kept = FunctionTool(func=lambda: "kept", name="kept", description="")
        live_tools: list[Any] = [existing, removable, callable_wrapped, kept]
        context = FunctionInvocationContext(existing, {}, tools=live_tools)

        context.remove_tools(["existing", removable, callable_tool, "missing"])

        assert context.tools is live_tools
        assert live_tools == [kept]

    def test_function_context_remove_tool_object_preserves_foreign_same_name(self) -> None:
        owned = FunctionTool(func=lambda: "owned", name="shared", description="")
        foreign = FunctionTool(func=lambda: "foreign", name="shared", description="")
        live_tools: list[Any] = [owned, foreign]
        context = FunctionInvocationContext(owned, {}, tools=live_tools)

        context.remove_tools(owned)

        assert live_tools == [foreign]

    @pytest.mark.parametrize("method_name", ["add_tools", "remove_tools"])
    def test_function_context_tool_mutation_requires_live_list(self, method_name: str) -> None:
        function = FunctionTool(func=lambda: "value", name="function", description="")
        context = FunctionInvocationContext(function, {})

        with pytest.raises(RuntimeError, match="not bound to a live agent run"):
            if method_name == "add_tools":
                context.add_tools(function)
            else:
                context.remove_tools(function)


# ---------------------------------------------------------------------------
# Own chat pipeline
# ---------------------------------------------------------------------------


class TestOwnChatPipeline:
    async def test_first_middleware_in_list_is_outermost(self) -> None:
        order: list[str] = []
        context = _chat_context()

        await ChatMiddlewarePipeline(_OrderedChat("a", order), _OrderedChat("b", order)).execute(
            context, _chat_final(_text_response())
        )

        assert order == ["a-pre", "b-pre", "b-post", "a-post"]

    async def test_final_handler_result_lands_in_context_result(self) -> None:
        observed: list[Any] = []

        class _Observe(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                assert context.result is None
                await call_next()
                observed.append(context.result)

        response = _text_response()
        context = _chat_context()

        result = await ChatMiddlewarePipeline(_Observe()).execute(context, _chat_final(response))

        assert result is response
        assert context.result is response
        assert observed == [response]

    async def test_result_override_without_call_next_skips_final_handler(self) -> None:
        override = _text_response("override")
        final_calls: list[ChatContext] = []

        class _Override(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                context.result = override

        result = await ChatMiddlewarePipeline(_Override()).execute(
            _chat_context(), _chat_final(_text_response(), final_calls)
        )

        assert result is override
        assert final_calls == []

    async def test_termination_is_suppressed_and_skips_outer_post_processing(self) -> None:
        order: list[str] = []
        override = _text_response("terminated")
        final_calls: list[ChatContext] = []

        class _Terminate(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                context.result = override
                raise MiddlewareTermination("stop")

        result = await ChatMiddlewarePipeline(_OrderedChat("outer", order), _Terminate()).execute(
            _chat_context(), _chat_final(_text_response(), final_calls)
        )

        assert result is override
        assert final_calls == []
        assert order == ["outer-pre"]  # outer post-processing skipped (framework parity)

    async def test_empty_pipeline_fast_path_resolves_and_stores_result(self) -> None:
        response = _text_response()
        context = _chat_context()

        result = await ChatMiddlewarePipeline().execute(context, lambda _context: response)

        assert result is response
        assert context.result is response

    async def test_empty_pipeline_streaming_requires_response_stream(self) -> None:
        context = _chat_context(stream=True)

        with pytest.raises(ValueError, match="requires a ResponseStream result"):
            await ChatMiddlewarePipeline().execute(context, _chat_final(_text_response()))

    async def test_context_stream_hooks_attach_to_response_stream_result(self) -> None:
        transformed: list[ChatResponseUpdate] = []
        finalized: list[ChatResponse] = []
        cleanups: list[bool] = []

        def _transform(update: ChatResponseUpdate) -> ChatResponseUpdate:
            transformed.append(update)
            return update

        def _on_result(response: ChatResponse) -> ChatResponse:
            finalized.append(response)
            return response

        def _on_cleanup() -> None:
            cleanups.append(True)

        class _Hooker(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                context.stream_transform_hooks.append(_transform)
                context.stream_result_hooks.append(_on_result)
                context.stream_cleanup_hooks.append(_on_cleanup)
                await call_next()

        async def _gen() -> Any:
            yield _text_update("a")
            yield _text_update("b")

        stream = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)
        context = _chat_context(stream=True)

        result = await ChatMiddlewarePipeline(_Hooker()).execute(context, _chat_final(stream))

        assert result is stream
        consumed = [update async for update in result]
        assert len(consumed) == 2
        assert transformed == consumed
        assert len(finalized) == 1
        assert cleanups == [True]

    def test_rejects_non_chat_middleware(self) -> None:
        async def _callable_middleware(context: Any, call_next: Any) -> None:  # pragma: no cover - never invoked
            await call_next()

        with pytest.raises(TypeError, match="plain callables are not supported"):
            ChatMiddlewarePipeline(_callable_middleware)  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            ChatMiddlewarePipeline(_NoopFunction())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Own function pipeline
# ---------------------------------------------------------------------------


class TestOwnFunctionPipeline:
    async def test_termination_propagates_and_context_result_survives(self) -> None:
        class _Terminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                context.result = "partial"
                raise MiddlewareTermination("stop", result="from-exc")

        context = _fn_context()

        with pytest.raises(MiddlewareTermination) as excinfo:
            await FunctionMiddlewarePipeline(_Terminate()).execute(context, _fn_final("unused"))

        # _invoke_function_call reads both after catching MiddlewareTermination.
        assert excinfo.value.result == "from-exc"
        assert context.result == "partial"

    async def test_mid_chain_termination_skips_outer_post_processing(self) -> None:
        order: list[str] = []

        class _Terminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                order.append("terminate")
                raise MiddlewareTermination("stop")

        with pytest.raises(MiddlewareTermination):
            await FunctionMiddlewarePipeline(_OrderedFunction("outer", order), _Terminate()).execute(
                _fn_context(), _fn_final("unused")
            )

        assert order == ["outer-pre", "terminate"]

    async def test_empty_pipeline_returns_result_without_touching_context(self) -> None:
        context = _fn_context()

        result = await FunctionMiddlewarePipeline().execute(context, _fn_final("raw"))

        assert result == "raw"
        # FunctionMiddlewarePipeline deliberately returns the fast-path value
        # without populating context.result; pin that empty-pipeline asymmetry.
        assert context.result is None

    async def test_result_override_without_call_next_skips_execution(self) -> None:
        final_calls: list[FunctionInvocationContext] = []

        class _Override(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                context.result = "override"

        result = await FunctionMiddlewarePipeline(_Override()).execute(_fn_context(), _fn_final("real", final_calls))

        assert result == "override"
        assert final_calls == []

    async def test_chain_runs_in_order_and_final_result_lands_in_context(self) -> None:
        order: list[str] = []
        context = _fn_context()

        result = await FunctionMiddlewarePipeline(_OrderedFunction("a", order), _OrderedFunction("b", order)).execute(
            context, _fn_final("done")
        )

        assert result == "done"
        assert context.result == "done"
        assert order == ["a-pre", "b-pre", "b-post", "a-post"]

    def test_rejects_non_function_middleware(self) -> None:
        with pytest.raises(TypeError, match="plain callables are not supported"):
            FunctionMiddlewarePipeline(object())  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            FunctionMiddlewarePipeline(_NoopChat())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ChatMiddlewareLayer request isolation and lazy execution (run-contract §2/§3);
# sequential calls exercise fresh per-call contexts without a tool loop.
# ---------------------------------------------------------------------------


class TestOwnChatLayer:
    async def test_context_messages_is_a_fresh_list_across_external_calls(self) -> None:
        inner = _InnerClient(responses=[_text_response(), _text_response()])
        capture = _ContextCapture()
        layer = ChatMiddlewareLayer(inner, middleware=[capture])
        caller_messages = [_user_message()]

        await layer.get_response(caller_messages)
        await layer.get_response(caller_messages)

        first, second = capture.contexts
        assert first is not second
        assert first.messages is not caller_messages
        assert second.messages is not caller_messages
        assert first.messages is not second.messages
        assert first.messages[0] is caller_messages[0]
        assert second.messages[0] is caller_messages[0]

    async def test_element_swap_reaches_wire_and_caller_list_is_untouched(self) -> None:
        inner = _InnerClient(responses=[_text_response()])
        original = _user_message("original")
        replacement = _user_message("replaced")

        class _Swap(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                context.messages[0] = replacement
                await call_next()

        layer = ChatMiddlewareLayer(inner, middleware=[_Swap()])
        caller_messages = [original]

        await layer.get_response(caller_messages)

        assert inner.wire_messages[0][0] is replacement
        assert caller_messages[0] is original

    async def test_stream_update_filter_attaches_on_the_no_middleware_fast_path(self) -> None:
        inner = _InnerClient(update_rounds=[[_text_update("raw")]])
        layer = ChatMiddlewareLayer(inner)

        def tag(update: ChatResponseUpdate) -> ChatResponseUpdate:
            return _text_update(f"filtered:{update.text}")

        stream = layer.get_response([_user_message()], stream=True, stream_update_filter=tag)
        yielded = [update async for update in stream]
        final = await stream.get_final_response()

        assert [update.text for update in yielded] == ["filtered:raw"]
        assert final.text == "filtered:raw", "the client stream's own finalizer assembled filtered updates"
        assert all("stream_update_filter" not in extra for extra in inner.wire_extra), (
            "consumed by the layer, never forwarded as a client kwarg"
        )

    async def test_stream_update_filter_reaches_beneath_a_draining_proxy_middleware(self) -> None:
        # The response-validation shape: the middleware drains the inner
        # stream through its own generator and replays updates, so its
        # source is NOT a ResponseStream and ``with_update_filter``
        # push-down cannot cross it. The request-path filter attaches at
        # the final handler, beneath the proxy — the drained updates and
        # the proxy's cached finalization are already filtered.
        drained: list[str] = []

        class _DrainingProxy(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                await call_next()
                inner_stream = context.result
                assert isinstance(inner_stream, ResponseStream)

                async def _replay() -> Any:
                    async for update in inner_stream:
                        drained.append(update.text)
                        yield update

                context.result = ResponseStream(_replay(), finalizer=ChatResponse.from_updates)

        inner = _InnerClient(update_rounds=[[_text_update("raw")]])
        layer = ChatMiddlewareLayer(inner, middleware=[_DrainingProxy()])

        def tag(update: ChatResponseUpdate) -> ChatResponseUpdate:
            return _text_update(f"filtered:{update.text}")

        stream = layer.get_response([_user_message()], stream=True, stream_update_filter=tag)
        yielded = [update async for update in stream]

        assert drained == ["filtered:raw"], "the proxy consumed filtered updates from the client stream"
        assert [update.text for update in yielded] == ["filtered:raw"]

    async def test_request_observer_sees_post_middleware_messages_on_every_inner_call(self) -> None:
        original = _user_message()
        injected = _user_message("wire-only")
        observed: list[list[Message]] = []

        class _AppendAndRetry(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                context.messages = [*context.messages, injected]
                await call_next()
                await call_next()

        inner = _InnerClient(responses=[_text_response("first"), _text_response("second")])
        layer = ChatMiddlewareLayer(inner, middleware=[_AppendAndRetry()])

        await layer.get_response(
            [original],
            request_message_observer=lambda messages: observed.append(list(messages)),
        )

        assert len(observed) == 2
        assert all(messages == [original, injected] for messages in observed)
        assert inner.wire_messages == observed
        assert all("request_message_observer" not in extra for extra in inner.wire_extra)

    async def test_whole_list_rebind_reaches_wire_only_for_that_call(self) -> None:
        inner = _InnerClient(responses=[_text_response(), _text_response()])
        injected = Message(role="system", contents=["injected"])

        class _RebindOnce(ChatMiddleware):
            def __init__(self) -> None:
                self.armed = True

            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                if self.armed:
                    self.armed = False
                    context.messages = [*context.messages, injected]
                await call_next()

        layer = ChatMiddlewareLayer(inner, middleware=[_RebindOnce()])
        caller_messages = [_user_message()]

        await layer.get_response(caller_messages)
        await layer.get_response(caller_messages)

        assert inner.wire_messages[0][-1] is injected
        assert all(message is not injected for message in inner.wire_messages[1])
        assert injected not in caller_messages

    async def test_streaming_returns_synchronously_and_runs_nothing_until_consumed(self) -> None:
        inner = _InnerClient(update_rounds=[[_text_update("a"), _text_update("b")]])
        capture = _ContextCapture()
        layer = ChatMiddlewareLayer(inner, middleware=[capture])

        stream = layer.get_response([_user_message()], stream=True)

        assert isinstance(stream, ResponseStream)
        assert capture.contexts == []
        assert inner.wire_messages == []

        updates = [update async for update in stream]

        assert len(updates) == 2
        assert len(capture.contexts) == 1
        assert len(inner.wire_messages) == 1

    async def test_streaming_termination_without_result_yields_empty_stream(self) -> None:
        inner = _InnerClient()

        class _Terminate(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                raise MiddlewareTermination("stop")

        layer = ChatMiddlewareLayer(inner, middleware=[_Terminate()])

        stream = layer.get_response([_user_message()], stream=True)
        updates = [update async for update in stream]

        assert updates == []
        assert inner.wire_messages == []

    async def test_streaming_with_non_stream_result_raises_on_consumption(self) -> None:
        inner = _InnerClient()

        class _WrongOverride(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                context.result = _text_response()

        layer = ChatMiddlewareLayer(inner, middleware=[_WrongOverride()])
        stream = layer.get_response([_user_message()], stream=True)

        with pytest.raises(ValueError, match="Expected ResponseStream for streaming"):
            async for _update in stream:  # pragma: no branch
                pass

    async def test_non_streaming_override_never_calls_inner_client(self) -> None:
        inner = _InnerClient()
        override = _text_response("override")

        class _Override(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                context.result = override

        layer = ChatMiddlewareLayer(inner, middleware=[_Override()])

        result = await layer.get_response([_user_message()])

        assert result is override
        assert inner.wire_messages == []

    async def test_constructor_middleware_is_outermost_of_per_call_middleware(self) -> None:
        inner = _InnerClient(responses=[_text_response()])
        order: list[str] = []
        layer = ChatMiddlewareLayer(inner, middleware=[_OrderedChat("ctor", order)])

        await layer.get_response([_user_message()], middleware=[_OrderedChat("call", order)])

        assert order == ["ctor-pre", "call-pre", "call-post", "ctor-post"]

    async def test_kwargs_and_options_forwarding_through_pipeline_path(self) -> None:
        inner = _InnerClient(responses=[_text_response()])
        layer = ChatMiddlewareLayer(inner, middleware=[_NoopChat()])

        await layer.get_response(
            [_user_message()],
            function_invocation_kwargs={"f": 1},
            custom="x",
        )

        # options=None becomes {} through _middleware_handler (`options or {}`, parity).
        assert inner.wire_options == [{}]
        assert inner.wire_function_invocation_kwargs == [{"f": 1}]
        assert inner.wire_extra == [{"custom": "x"}]

    async def test_no_middleware_fast_path_forwards_caller_messages_uncopied(self) -> None:
        inner = _InnerClient(responses=[_text_response()])
        layer = ChatMiddlewareLayer(inner)
        caller_messages = [_user_message()]

        await layer.get_response(caller_messages, custom="x")

        # Framework parity: the fast path forwards the caller's sequence as-is
        # (no copy) and options stays None (no `or {}` rewrite).
        assert inner.raw_messages[0] is caller_messages
        assert inner.wire_options == [None]
        assert inner.wire_extra == [{"custom": "x"}]


# ---------------------------------------------------------------------------
# split_middleware
# ---------------------------------------------------------------------------


class TestSplitMiddleware:
    def test_buckets_preserve_relative_order(self) -> None:
        chat_a, chat_b = _NoopChat(), _NoopChat()
        function_a, function_b = _NoopFunction(), _NoopFunction()

        split = split_middleware([chat_a, function_a, chat_b, function_b])

        assert split.chat == [chat_a, chat_b]
        assert split.function == [function_a, function_b]

    def test_sources_flatten_like_framework(self) -> None:
        chat = _NoopChat()
        function = _NoopFunction()

        split = split_middleware(None, chat, [function], ())

        assert split.chat == [chat]
        assert split.function == [function]

    def test_falsy_bare_middleware_is_not_dropped(self) -> None:
        class _FalsyChat(_NoopChat):
            def __bool__(self) -> bool:
                return False

        chat = _FalsyChat()

        split = split_middleware(chat)

        assert split.chat == [chat]

    def test_rejects_plain_callables_and_unknown_objects(self) -> None:
        async def _callable_middleware(context: Any, call_next: Any) -> None:  # pragma: no cover - never invoked
            await call_next()

        with pytest.raises(TypeError, match="plain callables are not supported"):
            split_middleware([_callable_middleware])

        with pytest.raises(TypeError):
            split_middleware([object()])
