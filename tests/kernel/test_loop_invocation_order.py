# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Call provenance stamps, invocation-order ordinals and result-metadata carriage: the loop as sole authority."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

import pytest

from chrys.foundation.tool_execution_stamp import EXECUTION_STAMP_KEY, write_execution_stamp
from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.foundation.tool_result_metadata import (
    TOOL_ERROR_KIND_METADATA_KEY,
)
from chrys.foundation.trajectory.context import TRAJECTORY_CONTEXT_KWARG
from chrys.foundation.trajectory.event_types import EventType as TrajectoryEventType
from chrys.foundation.trajectory.event_types import ToolOutcome
from chrys.foundation.trajectory.metadata import (
    ANALYTICS_ITEM_ID_KEY,
    OPERATION_ID_KEY,
    TOOL_RESULT_ITEM_ID_METADATA_KEY,
)
from chrys.kernel import FunctionTool, tool
from chrys.kernel.middleware import (
    FunctionInvocationContext,
    FunctionMiddleware,
    MiddlewareTermination,
)
from chrys.kernel.types import (
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
)
from tests.kernel._fakes import (
    _call_contents,
    _call_response,
    _call_update,
    _CarriageFunction,
    _final_response,
    _make_tool,
    _ordinals,
    _ProbeFunction,
    _provenance_tool,
    _result_contents,
    _stack,
    _TerminateFunction,
    _text_response,
    _text_update,
    _user,
)
from tests.service.trajectory._fakes import FakeSink, make_context

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class _ExecutionStampFunction(FunctionMiddleware):
    def __init__(self, completion_order: list[str]) -> None:
        self._completion_order = completion_order

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        outcome: Literal["ok", "error"] = "ok"
        error_kind: str | None = None
        try:
            await call_next()
        except Exception as exc:
            outcome = "error"
            error_kind = type(exc).__name__
            raise
        finally:
            value = str(context.arguments["value"])
            self._completion_order.append(value)
            write_execution_stamp(context.metadata, context.arguments, outcome=outcome, error_kind=error_kind)


# ---------------------------------------------------------------------------
# Pre-pipeline call-provenance stamp (kind + static context on the function_call)
# ---------------------------------------------------------------------------


class TestCallProvenanceStamp:
    """The loop stamps kind + static context on the call, and results never carry them."""

    @pytest.mark.asyncio
    async def test_success_path_stamps_call_and_filters_result(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        function_call.additional_properties["provider_marker"] = "kept"
        tool_obj = _provenance_tool(static={"server_name": "probe"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [tool_obj]})

        # Same-object invariant: the loop mutates the caller's Content in place.
        assert function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "probe.kind"
        assert function_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY] == {"server_name": "probe"}
        result = _result_contents(response)[0]
        assert result.additional_properties["provider_marker"] == "kept"
        assert TOOL_CALL_KIND_METADATA_KEY not in result.additional_properties
        assert TOOL_CALL_CONTEXT_METADATA_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_prevalidation_failure_still_stamps_call(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY

        probe = _ProbeFunction()
        function_call = Content.from_function_call("c1", "echo", arguments={"wrong_arg": 1})
        tool_obj = _provenance_tool(static={"server_name": "probe"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [tool_obj]}, middleware=[probe])

        assert probe.contexts == [], "bad arguments must still fail before the pipeline"
        assert function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "probe.kind"
        assert function_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY] == {"server_name": "probe"}
        result = _result_contents(response)[0]
        assert result.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_parsing"
        assert TOOL_CALL_KIND_METADATA_KEY not in result.additional_properties
        assert TOOL_CALL_CONTEXT_METADATA_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_tool_not_found_stays_unclassified(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY

        function_call = Content.from_function_call("c1", "ghost", arguments={})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [_provenance_tool()]})

        assert TOOL_CALL_KIND_METADATA_KEY not in function_call.additional_properties
        result = _result_contents(response)[0]
        assert result.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "tool_not_found"
        assert TOOL_CALL_KIND_METADATA_KEY not in result.additional_properties
        assert TOOL_CALL_CONTEXT_METADATA_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_stamp_is_first_write_wins(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] = "persisted.kind"
        function_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY] = {"server_name": "persisted"}
        tool_obj = _provenance_tool(static={"server_name": "live"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        await layer.get_response([_user()], options={"tools": [tool_obj]})

        assert function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "persisted.kind"
        assert function_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY] == {"server_name": "persisted"}

    @pytest.mark.asyncio
    async def test_tool_exception_result_carries_no_provenance(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY, set_tool_context
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY, set_tool_kind

        @tool(name="boom")
        async def boom(text: str) -> str:
            raise RuntimeError("kaboom")

        set_tool_kind(boom, "probe.kind")
        set_tool_context(boom, {"server_name": "probe"})
        function_call = Content.from_function_call("c1", "boom", arguments={"text": "a"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [boom]})

        assert function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "probe.kind"
        result = _result_contents(response)[0]
        assert "Error: Function failed." in str(result.result)
        assert TOOL_CALL_KIND_METADATA_KEY not in result.additional_properties
        assert TOOL_CALL_CONTEXT_METADATA_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_middleware_termination_result_carries_no_provenance(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        tool_obj = _provenance_tool(static={"server_name": "probe"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])])])

        response = await layer.get_response(
            [_user()],
            options={"tools": [tool_obj]},
            middleware=[_TerminateFunction(result="interrupted")],
        )

        assert function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "probe.kind"
        result = _result_contents(response)[0]
        assert str(result.result) == "interrupted"
        assert TOOL_CALL_KIND_METADATA_KEY not in result.additional_properties
        assert TOOL_CALL_CONTEXT_METADATA_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_result_props_do_not_share_the_call_context_dict(self) -> None:
        """The shallow-copy hazard: the result must never alias the call's nested context."""
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        tool_obj = _provenance_tool(static={"server_name": "probe"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [tool_obj]})

        stamped = function_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY]
        result = _result_contents(response)[0]
        assert all(value is not stamped for value in result.additional_properties.values())


# ---------------------------------------------------------------------------
# Invocation-order stamping + result-metadata carriage (loop authority)
# ---------------------------------------------------------------------------


class TestInvocationOrderStamping:
    """The loop is the single ordinal producer: stamped at response arrival, pre-dispatch."""

    @pytest.mark.asyncio
    async def test_ordinals_accumulate_across_iterations(self) -> None:
        layer, _wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"}), ("c2", "echo", {"text": "b"})),
                _call_response(("c3", "echo", {"text": "c"})),
                _text_response("done"),
            ]
        )

        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})

        assert _ordinals(response) == [0, 1, 2]
        for result in _result_contents(response):
            assert TOOL_INVOCATION_ORDER_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_prevalidation_failure_and_unknown_tool_consume_ordinals(self) -> None:
        """The issue-589 shape: calls that never enter the pipeline still hold their slot."""
        batch = _call_response(
            ("c1", "echo", {"wrong_arg": "x"}),
            ("c2", "missing_tool", {}),
            ("c3", "echo", {"text": "ok"}),
        )
        # A lenient schema would accept the parser's {"raw": [1]} wrapper; the
        # non-object payload is refused before the pipeline all the same.
        lenient = FunctionTool(
            name="lenient",
            func=lambda **_kwargs: "ran",
            input_model={"type": "object", "properties": {"state": {"type": "string"}}},
        )
        batch.messages[0].contents.append(Content.from_function_call("c4", "lenient", arguments="[1]"))
        layer, _wire = _stack([batch, _text_response("done")])

        response = await layer.get_response([_user()], options={"tools": [_make_tool(), lenient]})

        assert _ordinals(response) == [0, 1, 2, 3]
        results = {r.call_id: r for r in _result_contents(response)}
        assert results["c1"].additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_parsing"
        assert results["c2"].additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "tool_not_found"
        assert str(results["c3"].result) == "echo:ok"
        assert results["c4"].additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_parsing"
        for result in results.values():
            assert TOOL_INVOCATION_ORDER_KEY not in result.additional_properties

    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_stale_foreign_stamp_is_overwritten(self, stream: bool) -> None:
        """A pre-stamped content on a landed response is re-numbered — sole authority.

        Bridged or cached clients can replay content that carries another
        run's ordinal; preserving it while the counter advances would let the
        stale value collide with a fresh assignment.
        """
        stale_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        stale_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 7
        fresh_call = Content.from_function_call("c2", "echo", arguments={"text": "b"})
        if stream:
            stale_update = ChatResponseUpdate(contents=[stale_call, fresh_call], role="assistant")
            turns: list[Any] = [[stale_update], [_text_update("done")]]
        else:
            turns = [ChatResponse(messages=[Message("assistant", [stale_call, fresh_call])]), _text_response("done")]
        layer, _wire = _stack(turns)

        response = await _final_response(layer, [_user()], stream=stream, options={"tools": [_make_tool()]})

        assert stale_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert fresh_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 1
        assert _ordinals(response) == [0, 1]

    @pytest.mark.asyncio
    async def test_revisited_content_keeps_ordinal_and_is_not_dispatched_again(self) -> None:
        """A content this run already numbered is an echo: same ordinal, no re-run.

        An echoing client can re-embed a prior iteration's call object in a
        later response. Re-numbering it would corrupt decisions recorded
        against the first value, re-dispatching it would execute the tool
        twice under one ordinal — collapsing two results onto one approval
        slot — and leaving it in the response would persist a dangling
        duplicate call, so landing removes the echo outright.
        """
        events: list[str] = []
        first_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        second_call = Content.from_function_call("c2", "echo", arguments={"text": "b"})
        third_call = Content.from_function_call("c3", "echo", arguments={"text": "c"})
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [first_call, second_call])]),
                ChatResponse(messages=[Message("assistant", [first_call, third_call])]),
                _text_response("done"),
            ]
        )

        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert first_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert second_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 1
        assert third_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 2
        assert events == ["tool:echo:a", "tool:echo:b", "tool:echo:c"], "the echoed call must execute exactly once"
        assert [r.call_id for r in _result_contents(response)].count("c1") == 1
        assert [c.call_id for c in _call_contents(response)] == ["c1", "c2", "c3"], "the echo must leave no duplicate"

    @pytest.mark.asyncio
    async def test_response_of_only_revisited_calls_ends_the_loop(self) -> None:
        """An echo-only response carries no new work and terminates cleanly.

        The echo is removed from the landed response, so the final transcript
        must end call/result-paired — no dangling duplicate function call
        after its result (providers reject such histories on the next turn).
        """
        events: list[str] = []
        first_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [first_call])]),
                ChatResponse(messages=[Message("assistant", [first_call])]),
            ]
        )

        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:a"]
        assert first_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert [r.call_id for r in _result_contents(response)] == ["c1"]
        assert len(_call_contents(response)) == 1, "the echo must not survive as a dangling duplicate call"
        assert response.messages[-1].role == "tool"

    @pytest.mark.asyncio
    async def test_no_tools_response_calls_are_stamped(self) -> None:
        """A call batch returned without a tools option is stamped before the early exit."""
        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])])])

        response = await layer.get_response([_user()], options={})

        assert function_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert _result_contents(response) == []

    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_a_call_with_no_tools_to_run_it_closes_as_filtered(self, stream: bool) -> None:
        """The stamp is handed out at landing; the operation it opens has to end somewhere."""
        if stream:
            layer, _wire = _stack([[_call_update("c1", "echo", {"text": "a"})]])
        else:
            function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
            layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])])])
        sink = FakeSink()

        await _final_response(
            layer, [_user()], stream=stream, options={}, client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)}
        )

        started = sink.only(TrajectoryEventType.TOOL_OPERATION_STARTED)
        finished = sink.only(TrajectoryEventType.TOOL_OPERATION_FINISHED)
        assert finished.operation_id == started.operation_id
        assert finished.payload["outcome"] == ToolOutcome.FILTERED
        assert "result_item_id" not in finished.payload  # nothing ran, so there is no result to name

    @pytest.mark.asyncio
    async def test_informational_calls_are_not_stamped(self) -> None:
        hosted = Content.from_function_call("h1", "web_search", arguments={}, informational_only=True)
        local = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [hosted, local])]), _text_response("done")])

        await layer.get_response([_user()], options={"tools": [_make_tool()]})

        assert TOOL_INVOCATION_ORDER_KEY not in hosted.additional_properties
        assert local.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0

    @pytest.mark.asyncio
    async def test_result_metadata_and_context_carriage_fold_at_result_construction(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        tool_obj = _provenance_tool(static={"server_name": "static"})
        carrier = _CarriageFunction(result_metadata={"shell_exit_code": 1}, tool_context={"built": "filled"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [tool_obj]}, middleware=[carrier])

        # Call side: kind + static context stamped pre-validation, builder
        # subkeys backfilled post-execution without overwriting static keys.
        assert function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "probe.kind"
        assert function_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY] == {
            "server_name": "static",
            "built": "filled",
        }
        assert TOOL_RESULT_METADATA_KEY not in function_call.additional_properties
        # Result side: carried metadata folded at construction, nothing else.
        result = _result_contents(response)[0]
        assert result.additional_properties[TOOL_RESULT_METADATA_KEY] == {"shell_exit_code": 1}
        assert TOOL_CALL_CONTEXT_METADATA_KEY not in result.additional_properties
        assert TOOL_INVOCATION_ORDER_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_middleware_termination_prebuilt_result_metadata_first_write_wins(self) -> None:
        """A middleware-built result Content owns its own metadata; carriage never overwrites."""
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        prebuilt = Content.from_function_result("c1", result="stopped")
        prebuilt.additional_properties[TOOL_RESULT_METADATA_KEY] = {"owned": True}

        class _CarryThenTerminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                context.metadata[TOOL_RESULT_METADATA_KEY] = {"carried": True}
                raise MiddlewareTermination("stop", result=prebuilt)

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])])])

        response = await layer.get_response(
            [_user()], options={"tools": [_make_tool()]}, middleware=[_CarryThenTerminate()]
        )

        result = _result_contents(response)[0]
        assert result is not prebuilt
        assert result.call_id == "c1"
        assert result.additional_properties[TOOL_RESULT_METADATA_KEY] == {"owned": True}
        assert prebuilt.additional_properties[TOOL_RESULT_METADATA_KEY] == {"owned": True}

    @pytest.mark.asyncio
    async def test_middleware_termination_prebuilt_result_carries_this_calls_identity(self) -> None:
        """The trajectory names the result item before it exists; the saved one has to be it."""
        prebuilt = Content.from_function_result("stale", result="stopped")
        prebuilt.additional_properties[ANALYTICS_ITEM_ID_KEY] = "an-id-from-another-call"
        promised: dict[str, str] = {}

        class _Terminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                promised["result_item_id"] = str(context.metadata[TOOL_RESULT_ITEM_ID_METADATA_KEY])
                promised["operation_id"] = str(context.metadata[OPERATION_ID_KEY])
                raise MiddlewareTermination("stop", result=prebuilt)

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])])])

        response = await layer.get_response([_user()], options={"tools": [_make_tool()]}, middleware=[_Terminate()])

        result = _result_contents(response)[0]
        props = result.additional_properties
        assert props[OPERATION_ID_KEY] == promised["operation_id"]
        assert props[ANALYTICS_ITEM_ID_KEY] == promised["result_item_id"]
        # The cached result keeps the identity it came with; this call's went
        # onto the copy.
        assert prebuilt.additional_properties[ANALYTICS_ITEM_ID_KEY] == "an-id-from-another-call"

    @pytest.mark.asyncio
    async def test_middleware_termination_prebuilt_result_without_metadata_gets_carriage(self) -> None:
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        prebuilt = Content.from_function_result("c1", result="stopped")

        class _CarryThenTerminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                context.metadata[TOOL_RESULT_METADATA_KEY] = {"carried": True}
                raise MiddlewareTermination("stop", result=prebuilt)

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])])])

        response = await layer.get_response(
            [_user()], options={"tools": [_make_tool()]}, middleware=[_CarryThenTerminate()]
        )

        result = _result_contents(response)[0]
        assert result is not prebuilt
        assert result.call_id == "c1"
        assert result.additional_properties[TOOL_RESULT_METADATA_KEY] == {"carried": True}
        assert TOOL_RESULT_METADATA_KEY not in prebuilt.additional_properties

    @pytest.mark.asyncio
    async def test_middleware_termination_shared_prebuilt_result_is_owned_per_parallel_call(self) -> None:
        """Concurrent terminations cannot alias one cached result or retain its stale call id."""
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        prebuilt = Content.from_function_result("stale", result="stopped")

        class _CarryThenTerminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                text = str(context.arguments["text"])
                context.metadata[TOOL_RESULT_METADATA_KEY] = {"text": text}
                write_execution_stamp(context.metadata, context.arguments, outcome="ok")
                raise MiddlewareTermination("stop", result=prebuilt)

        layer, _wire = _stack(
            [
                _call_response(
                    ("c1", "echo", {"text": "a"}),
                    ("c2", "echo", {"text": "b"}),
                )
            ]
        )

        response = await layer.get_response(
            [_user()], options={"tools": [_make_tool()]}, middleware=[_CarryThenTerminate()]
        )

        first, second = _result_contents(response)
        assert first is not second
        assert first is not prebuilt
        assert second is not prebuilt
        assert [first.call_id, second.call_id] == ["c1", "c2"]
        assert first.additional_properties[TOOL_RESULT_METADATA_KEY] == {"text": "a"}
        assert second.additional_properties[TOOL_RESULT_METADATA_KEY] == {"text": "b"}
        assert first.additional_properties[EXECUTION_STAMP_KEY]["effective_args"] == '{"text":"a"}'
        assert second.additional_properties[EXECUTION_STAMP_KEY]["effective_args"] == '{"text":"b"}'
        assert prebuilt.call_id == "stale"
        assert prebuilt.additional_properties == {}

    @pytest.mark.asyncio
    async def test_middleware_termination_shared_prebuilt_result_is_owned_per_run(self) -> None:
        """A cached result reused by later runs receives fresh call-scoped metadata."""
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        prebuilt = Content.from_function_result("stale", result="stopped")

        class _CarryThenTerminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                text = str(context.arguments["text"])
                context.metadata[TOOL_RESULT_METADATA_KEY] = {"text": text}
                write_execution_stamp(context.metadata, context.arguments, outcome="ok")
                raise MiddlewareTermination("stop", result=prebuilt)

        middleware = _CarryThenTerminate()
        layer, _wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "first"})),
                _call_response(("c2", "echo", {"text": "second"})),
            ]
        )

        first_response = await layer.get_response([_user()], options={"tools": [_make_tool()]}, middleware=[middleware])
        second_response = await layer.get_response(
            [_user()], options={"tools": [_make_tool()]}, middleware=[middleware]
        )

        first = _result_contents(first_response)[0]
        second = _result_contents(second_response)[0]
        assert first is not second
        assert [first.call_id, second.call_id] == ["c1", "c2"]
        assert first.additional_properties[TOOL_RESULT_METADATA_KEY] == {"text": "first"}
        assert second.additional_properties[TOOL_RESULT_METADATA_KEY] == {"text": "second"}
        assert first.additional_properties[EXECUTION_STAMP_KEY]["effective_args"] == '{"text":"first"}'
        assert second.additional_properties[EXECUTION_STAMP_KEY]["effective_args"] == '{"text":"second"}'
        assert prebuilt.call_id == "stale"
        assert prebuilt.additional_properties == {}

    @pytest.mark.asyncio
    async def test_cancellation_after_call_next_propagates_without_result(self) -> None:
        """A cancelled invocation constructs no result and fabricates no metadata."""

        class _CancelAfter(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                await call_next()
                raise asyncio.CancelledError

        events: list[str] = []
        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        with pytest.raises(asyncio.CancelledError):
            await layer.get_response([_user()], options={"tools": [_make_tool(events)]}, middleware=[_CancelAfter()])

        assert events == ["tool:echo:a"], "cancellation hit after the tool body ran"
        assert len(wire.calls) == 1, "the loop must not continue past a cancelled batch"
        # The stamp landed at response arrival; nothing result-shaped was built.
        assert function_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0

    @pytest.mark.asyncio
    async def test_parallel_mixed_completion_stamps_each_matching_result(self) -> None:
        release_slow = asyncio.Event()
        fast_finished = asyncio.Event()
        completion_order: list[str] = []

        @tool(name="mixed")
        async def mixed(value: str) -> str:
            if value == "slow-ok":
                await release_slow.wait()
                return "slow result"
            fast_finished.set()
            raise RuntimeError("fast failure")

        async def _release_after_fast() -> None:
            await fast_finished.wait()
            release_slow.set()

        releaser = asyncio.create_task(_release_after_fast())
        layer, _wire = _stack(
            [
                _call_response(
                    ("slow-call", "mixed", {"value": "slow-ok"}),
                    ("fast-call", "mixed", {"value": "fast-error"}),
                ),
                _text_response(),
            ]
        )

        response = await layer.get_response(
            [_user()],
            options={"tools": [mixed]},
            middleware=[_ExecutionStampFunction(completion_order)],
        )
        await releaser

        results = _result_contents(response)
        assert completion_order == ["fast-error", "slow-ok"]
        assert [result.call_id for result in results] == ["slow-call", "fast-call"]
        slow_stamp = results[0].additional_properties[EXECUTION_STAMP_KEY]
        fast_stamp = results[1].additional_properties[EXECUTION_STAMP_KEY]
        assert slow_stamp["effective_args"] == '{"value":"slow-ok"}'
        assert slow_stamp["outcome"] == "ok"
        assert fast_stamp["effective_args"] == '{"value":"fast-error"}'
        assert fast_stamp["outcome"] == "error"
        assert fast_stamp["error_kind"] == "RuntimeError"
