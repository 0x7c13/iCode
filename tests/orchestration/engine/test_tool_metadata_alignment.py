# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cross-layer tests for tool-ordinal stamping and result-metadata alignment.

Engine + real middleware + history (no history-helper doubles): pins the
regression shapes where a call that never entered the middleware pipeline
(argument-validation failure, unknown tool) used to shift every later
result's persisted ``_chrys_tool_result_metadata`` one slot earlier — a
successful ``grep`` inheriting the next ``bash`` call's ``shell_exit_code``
and replaying as ERR.  Under the kernel-stamp design there is no positional
re-association at all: ordinals are stamped at response arrival and result
metadata rides the invocation context into the result content at
construction.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

import pytest

from chrys.foundation.models.invocations import InvocationOrigin
from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.foundation.tool_kinds import KIND_SEARCH, KIND_SHELL, set_tool_kind
from chrys.foundation.tool_result_metadata import (
    SHELL_EXIT_CODE_METADATA_KEY,
    TOOL_ERROR_KIND_METADATA_KEY,
    TOOL_RESULT_METADATA_KEY,
)
from chrys.kernel import FunctionTool
from chrys.service.llm.mock import MockResponse
from tests.support.pipeline_helpers import create_test_engine, extract_approval_requests, wait_for_idle


def _fake_bash(
    command: Annotated[str, "Command to run"],
    exit_code: Annotated[int, "Exit code to report"] = 0,
) -> str:
    from chrys.service.tools.builtins.shell import shell_result_metadata

    metadata = shell_result_metadata.get()
    assert metadata is not None, "shell tools must run with the middleware-provided metadata channel"
    metadata[SHELL_EXIT_CODE_METADATA_KEY] = exit_code
    return f"ran: {command}\n[exit_code: {exit_code}]"


def _fake_grep(pattern: Annotated[str, "Pattern to search"]) -> str:
    return f"matches for {pattern}"


def _make_tools() -> list[FunctionTool]:
    bash_tool = FunctionTool(func=_fake_bash, name="fake_bash", description="Fake shell")
    set_tool_kind(bash_tool, KIND_SHELL)
    grep_tool = FunctionTool(func=_fake_grep, name="fake_grep", description="Fake search")
    set_tool_kind(grep_tool, KIND_SEARCH)
    return [bash_tool, grep_tool]


def _session_calls(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All persisted function_call contents, in history order."""
    return [
        c
        for msg in raw_messages
        if msg.get("role") == "assistant"
        for c in msg.get("contents", [])
        if isinstance(c, dict) and c.get("type") == "function_call"
    ]


def _session_results(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All persisted function_result contents, in history order."""
    return [
        c
        for msg in raw_messages
        if msg.get("role") == "tool"
        for c in msg.get("contents", [])
        if isinstance(c, dict) and c.get("type") == "function_result"
    ]


def _props(content: dict[str, Any]) -> dict[str, Any]:
    return content.get("additional_properties") or {}


def _result_metadata(content: dict[str, Any]) -> dict[str, Any]:
    metadata = _props(content).get(TOOL_RESULT_METADATA_KEY)
    return metadata if isinstance(metadata, dict) else {}


class TestInvalidCallDoesNotShiftMetadata:
    @pytest.mark.asyncio
    async def test_invalid_arg_call_then_grep_then_bash(self, tmp_path):
        """The issue's exact shape: exit code lands only on bash, never on grep."""
        ctx = await create_test_engine(
            [
                MockResponse(tool_calls=[("fake_grep", "call_bad", {"wrong": "x"})]),
                MockResponse(tool_calls=[("fake_grep", "call_grep", {"pattern": "todo"})]),
                MockResponse(tool_calls=[("fake_bash", "call_bash", {"command": "false", "exit_code": 1})]),
                MockResponse(text="done"),
            ],
            tmp_path,
            tools=_make_tools(),
        )
        try:
            await ctx.send_message("go")
            raw = await ctx.get_session_messages()
            calls = _session_calls(raw)
            results = {r.get("call_id"): r for r in _session_results(raw)}

            assert [c.get("call_id") for c in calls] == ["call_bad", "call_grep", "call_bash"]
            # Every non-informational call is stamped — including the one that
            # failed pre-pipeline validation.
            assert [_props(c).get(TOOL_INVOCATION_ORDER_KEY) for c in calls] == [0, 1, 2]

            bad_result = results["call_bad"]
            assert _props(bad_result).get(TOOL_ERROR_KIND_METADATA_KEY) == "argument_parsing"

            grep_result = results["call_grep"]
            assert SHELL_EXIT_CODE_METADATA_KEY not in _result_metadata(grep_result)
            assert SHELL_EXIT_CODE_METADATA_KEY not in _props(grep_result)

            bash_result = results["call_bash"]
            assert _result_metadata(bash_result)[SHELL_EXIT_CODE_METADATA_KEY] == 1
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_unknown_tool_call_then_bash(self, tmp_path):
        ctx = await create_test_engine(
            [
                MockResponse(tool_calls=[("ghost_tool", "call_ghost", {})]),
                MockResponse(tool_calls=[("fake_bash", "call_bash", {"command": "false", "exit_code": 2})]),
                MockResponse(text="done"),
            ],
            tmp_path,
            tools=_make_tools(),
        )
        try:
            await ctx.send_message("go")
            raw = await ctx.get_session_messages()
            calls = _session_calls(raw)
            results = {r.get("call_id"): r for r in _session_results(raw)}

            assert [_props(c).get(TOOL_INVOCATION_ORDER_KEY) for c in calls] == [0, 1]
            assert _props(results["call_ghost"]).get(TOOL_ERROR_KIND_METADATA_KEY) == "tool_not_found"
            assert SHELL_EXIT_CODE_METADATA_KEY not in _result_metadata(results["call_ghost"])
            assert _result_metadata(results["call_bash"])[SHELL_EXIT_CODE_METADATA_KEY] == 2
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_parallel_batch_mixed_invalid_and_valid_same_name(self, tmp_path):
        ctx = await create_test_engine(
            [
                MockResponse(
                    tool_calls=[
                        ("fake_bash", "call_invalid", {"nonsense": 1}),
                        ("fake_bash", "call_valid", {"command": "true", "exit_code": 0}),
                    ]
                ),
                MockResponse(text="done"),
            ],
            tmp_path,
            tools=_make_tools(),
        )
        try:
            await ctx.send_message("go")
            raw = await ctx.get_session_messages()
            calls = _session_calls(raw)
            results = {r.get("call_id"): r for r in _session_results(raw)}

            # Ordinals follow model emission order.
            assert [(c.get("call_id"), _props(c).get(TOOL_INVOCATION_ORDER_KEY)) for c in calls] == [
                ("call_invalid", 0),
                ("call_valid", 1),
            ]
            assert _props(results["call_invalid"]).get(TOOL_ERROR_KIND_METADATA_KEY) == "argument_parsing"
            assert SHELL_EXIT_CODE_METADATA_KEY not in _result_metadata(results["call_invalid"])
            assert _result_metadata(results["call_valid"])[SHELL_EXIT_CODE_METADATA_KEY] == 0
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_parallel_batch_multiple_failure_kinds_with_out_of_order_success(self, tmp_path):
        """Several same-batch failures around live calls that finish out of order.

        Completion order is forced to invert emission order for the two live
        calls (the later fast call finishes first), so any slot-positional
        re-association — the defect class this module pins — would misattach
        exit codes or error kinds. Every persisted call and result must keep
        its own metadata, and the folded results stay in emission order.
        """
        release_slow = asyncio.Event()
        completion_order: list[str] = []

        async def _slow_bash(
            command: Annotated[str, "Command to run"],
            exit_code: Annotated[int, "Exit code to report"] = 0,
        ) -> str:
            from chrys.service.tools.builtins.shell import shell_result_metadata

            await asyncio.wait_for(release_slow.wait(), timeout=5)
            completion_order.append("slow")
            metadata = shell_result_metadata.get()
            assert metadata is not None
            metadata[SHELL_EXIT_CODE_METADATA_KEY] = exit_code
            return f"ran: {command}\n[exit_code: {exit_code}]"

        async def _fast_bash(
            command: Annotated[str, "Command to run"],
            exit_code: Annotated[int, "Exit code to report"] = 0,
        ) -> str:
            from chrys.service.tools.builtins.shell import shell_result_metadata

            completion_order.append("fast")
            metadata = shell_result_metadata.get()
            assert metadata is not None
            metadata[SHELL_EXIT_CODE_METADATA_KEY] = exit_code
            release_slow.set()
            return f"ran: {command}\n[exit_code: {exit_code}]"

        slow_tool = FunctionTool(func=_slow_bash, name="slow_bash", description="Slow fake shell")
        set_tool_kind(slow_tool, KIND_SHELL)
        fast_tool = FunctionTool(func=_fast_bash, name="fast_bash", description="Fast fake shell")
        set_tool_kind(fast_tool, KIND_SHELL)

        ctx = await create_test_engine(
            [
                MockResponse(
                    tool_calls=[
                        ("slow_bash", "call_slow", {"command": "first", "exit_code": 0}),
                        ("slow_bash", "call_bad_args", {"nonsense": 1}),
                        ("ghost_tool", "call_ghost", {}),
                        ("fast_bash", "call_fast", {"command": "last", "exit_code": 7}),
                        ("fast_bash", "call_bad_tail", {"nonsense": 2}),
                    ]
                ),
                MockResponse(text="done"),
            ],
            tmp_path,
            tools=[*_make_tools(), slow_tool, fast_tool],
        )
        try:
            await ctx.send_message("go")
            assert completion_order == ["fast", "slow"], "the batch must actually complete out of order"

            raw = await ctx.get_session_messages()
            calls = _session_calls(raw)
            ordered_results = _session_results(raw)
            results = {r.get("call_id"): r for r in ordered_results}

            # Ordinals follow model emission order; failed calls hold their slots.
            assert [(c.get("call_id"), _props(c).get(TOOL_INVOCATION_ORDER_KEY)) for c in calls] == [
                ("call_slow", 0),
                ("call_bad_args", 1),
                ("call_ghost", 2),
                ("call_fast", 3),
                ("call_bad_tail", 4),
            ]
            # Folded results keep emission order even though completion inverted it.
            assert [r.get("call_id") for r in ordered_results] == [
                "call_slow",
                "call_bad_args",
                "call_ghost",
                "call_fast",
                "call_bad_tail",
            ]
            assert _result_metadata(results["call_slow"])[SHELL_EXIT_CODE_METADATA_KEY] == 0
            assert _result_metadata(results["call_fast"])[SHELL_EXIT_CODE_METADATA_KEY] == 7
            for failed_id, kind in (
                ("call_bad_args", "argument_parsing"),
                ("call_ghost", "tool_not_found"),
                ("call_bad_tail", "argument_parsing"),
            ):
                assert _props(results[failed_id]).get(TOOL_ERROR_KIND_METADATA_KEY) == kind
                assert SHELL_EXIT_CODE_METADATA_KEY not in _result_metadata(results[failed_id])
        finally:
            await ctx.cleanup()


class TestPerCallMetadataIntegrity:
    @pytest.mark.asyncio
    async def test_consecutive_bash_exit_codes(self, tmp_path):
        ctx = await create_test_engine(
            [
                MockResponse(tool_calls=[("fake_bash", "call_0", {"command": "a", "exit_code": 0})]),
                MockResponse(tool_calls=[("fake_bash", "call_1", {"command": "b", "exit_code": 1})]),
                MockResponse(tool_calls=[("fake_bash", "call_2", {"command": "c", "exit_code": 2})]),
                MockResponse(text="done"),
            ],
            tmp_path,
            tools=_make_tools(),
        )
        try:
            await ctx.send_message("go")
            raw = await ctx.get_session_messages()
            results = {r.get("call_id"): r for r in _session_results(raw)}
            assert _result_metadata(results["call_0"])[SHELL_EXIT_CODE_METADATA_KEY] == 0
            assert _result_metadata(results["call_1"])[SHELL_EXIT_CODE_METADATA_KEY] == 1
            assert _result_metadata(results["call_2"])[SHELL_EXIT_CODE_METADATA_KEY] == 2
            calls = _session_calls(raw)
            assert [_props(c).get(TOOL_INVOCATION_ORDER_KEY) for c in calls] == [0, 1, 2]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_duplicate_provider_call_ids_across_iterations(self, tmp_path):
        """Sequential-id providers reuse call ids; construction-time folding never cross-matches."""
        ctx = await create_test_engine(
            [
                MockResponse(tool_calls=[("fake_bash", "call_00", {"command": "x", "exit_code": 1})]),
                MockResponse(tool_calls=[("fake_grep", "call_00", {"pattern": "y"})]),
                MockResponse(tool_calls=[("fake_bash", "call_00", {"command": "z", "exit_code": 0})]),
                MockResponse(text="done"),
            ],
            tmp_path,
            tools=_make_tools(),
        )
        try:
            await ctx.send_message("go")
            raw = await ctx.get_session_messages()
            results = _session_results(raw)
            assert [r.get("call_id") for r in results] == ["call_00", "call_00", "call_00"]
            assert _result_metadata(results[0])[SHELL_EXIT_CODE_METADATA_KEY] == 1
            assert SHELL_EXIT_CODE_METADATA_KEY not in _result_metadata(results[1])
            assert _result_metadata(results[2])[SHELL_EXIT_CODE_METADATA_KEY] == 0
            calls = _session_calls(raw)
            assert [_props(c).get(TOOL_INVOCATION_ORDER_KEY) for c in calls] == [0, 1, 2]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_streaming_run_preserves_stamps_and_metadata(self, tmp_path):
        """The full streamed chain: loop assembly → agent finalizer → history → session.json."""
        ctx = await create_test_engine(
            [
                MockResponse(tool_calls=[("fake_bash", "call_bash", {"command": "false", "exit_code": 1})]),
                MockResponse(text="done"),
            ],
            tmp_path,
            stream=True,
            tools=_make_tools(),
        )
        try:
            await ctx.send_message("go")
            raw = await ctx.get_session_messages()
            calls = _session_calls(raw)
            assert [_props(c).get(TOOL_INVOCATION_ORDER_KEY) for c in calls] == [0]
            results = _session_results(raw)
            assert _result_metadata(results[0])[SHELL_EXIT_CODE_METADATA_KEY] == 1
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_second_turn_ordinals_restart(self, tmp_path):
        """Per-run counters: a later turn's calls are stamped from zero again."""
        ctx = await create_test_engine(
            [
                MockResponse(tool_calls=[("fake_bash", "turn1_call", {"command": "a", "exit_code": 1})]),
                MockResponse(text="turn one done"),
                MockResponse(tool_calls=[("fake_bash", "turn2_call", {"command": "b", "exit_code": 2})]),
                MockResponse(text="turn two done"),
            ],
            tmp_path,
            tools=_make_tools(),
        )
        try:
            await ctx.send_message("first")
            await ctx.send_message("second")
            raw = await ctx.get_session_messages()
            calls = {c.get("call_id"): c for c in _session_calls(raw)}
            assert _props(calls["turn1_call"]).get(TOOL_INVOCATION_ORDER_KEY) == 0
            assert _props(calls["turn2_call"]).get(TOOL_INVOCATION_ORDER_KEY) == 0
            results = {r.get("call_id"): r for r in _session_results(raw)}
            assert _result_metadata(results["turn1_call"])[SHELL_EXIT_CODE_METADATA_KEY] == 1
            assert _result_metadata(results["turn2_call"])[SHELL_EXIT_CODE_METADATA_KEY] == 2
        finally:
            await ctx.cleanup()


class TestApprovalAlignmentAfterInvalidCall:
    @pytest.mark.asyncio
    async def test_approval_lands_on_valid_same_name_call(self, tmp_path):
        """A same-name call that failed validation must not absorb the approval decision."""
        ctx = await create_test_engine(
            [
                MockResponse(tool_calls=[("fake_bash", "call_invalid", {"nonsense": 1})]),
                MockResponse(tool_calls=[("fake_bash", "call_valid", {"command": "rm x", "exit_code": 0})]),
                MockResponse(text="done"),
            ],
            tmp_path,
            approval_default="auto",
            approval_overrides={"fake_bash": "require"},
            tools=_make_tools(),
        )
        try:
            await ctx.send_message("go")
            requests = extract_approval_requests(ctx.events)
            assert len(requests) == 1, "the invalid call must never reach approval"
            await ctx.approve(requests[0]["request_id"])
            await wait_for_idle(ctx)

            raw = await ctx.get_session_messages()
            calls = {c.get("call_id"): c for c in _session_calls(raw)}
            assert "_approval" not in _props(calls["call_invalid"])
            approval = _props(calls["call_valid"]).get("_approval")
            assert approval is not None
            assert approval["status"] == "user_approved"
        finally:
            await ctx.cleanup()


class TestSubAgentInnerStack:
    @pytest.mark.asyncio
    async def test_inner_stack_folds_metadata_without_controller_persistence(self, tmp_path):
        """The sub-agent shape: inner middleware carriage reaches the inner transcript natively."""
        from chrys.foundation.events.bus import EventBus
        from chrys.kernel import Message
        from chrys.service.agent_middleware.events.sub_agent_events import SubAgentEventMiddleware
        from chrys.service.llm.mock import MockChatClient

        bus = EventBus()
        sub_events = SubAgentEventMiddleware(
            bus, "Explore", "inv-1", session_id="s-1", origin=InvocationOrigin("sub_agent", "s-1", "inv-1", None)
        )
        client = MockChatClient(
            responses=[
                MockResponse(tool_calls=[("fake_bash", "inner-call", {"command": "false", "exit_code": 1})]),
                MockResponse(text="inner done"),
            ],
            middleware=[sub_events],
        )

        response = await client.get_response(
            [Message("user", ["go"])],
            options={"tools": _make_tools()},
        )

        calls = [c for m in response.messages for c in m.contents if c.type == "function_call"]
        assert calls[0].additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        results = [c for m in response.messages for c in m.contents if c.type == "function_result"]
        assert results[0].additional_properties[TOOL_RESULT_METADATA_KEY][SHELL_EXIT_CODE_METADATA_KEY] == 1
