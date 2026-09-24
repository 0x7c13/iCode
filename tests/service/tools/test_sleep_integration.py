# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Integration coverage for sleep tool behavior through the engine pipeline."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from chrys.foundation.events.types import InvocationToolCallResult, InvocationToolCallStart, SleepSkip, UserMessage
from chrys.service.llm.mock import MockResponse
from chrys.service.session.message_metadata import TOOL_RESULT_METADATA_KEY
from chrys.service.tools.builtins.sleep import sleep
from tests.support.pipeline_helpers import create_test_engine, echo_tool, extract_final_messages, wait_for_idle


@pytest.mark.asyncio
async def test_bool_seconds_is_rejected_before_sleep_middleware(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loop validation must reject bool before the sleep middleware sees coerced seconds."""
    ctx = await create_test_engine(
        [
            MockResponse(tool_calls=[("sleep", "sleep-call", {"seconds": False, "reason": "invalid"})]),
            MockResponse(text="recovered"),
        ],
        tmp_path,
        tools=[sleep],
    )
    executor = ctx.engine.current.loaded.bindings
    assert executor is not None
    middleware_seconds: list[object] = []
    original_process = executor._sleep.process

    async def _record_sleep_middleware(context: Any, call_next: Any) -> None:
        middleware_seconds.append(context.arguments["seconds"])
        await original_process(context, call_next)

    monkeypatch.setattr(executor._sleep, "process", _record_sleep_middleware)
    try:
        await ctx.send_message("wait")

        assert middleware_seconds == []
        assert extract_final_messages(ctx.events) == ["recovered"]
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_engine_sleep_skip_reaches_final_response_and_persists_replay_metadata(tmp_path) -> None:
    """A model-emitted sleep can be skipped and replay metadata is saved on the tool result."""
    echo_call_id = "echo-call"
    sleep_call_id = "sleep-call"
    ctx = await create_test_engine(
        [
            MockResponse(
                tool_calls=[
                    ("echo", echo_call_id, {"message": "before"}),
                    ("sleep", sleep_call_id, {"seconds": 30, "reason": "integration wait"}),
                ]
            ),
            MockResponse(text="done after skip"),
        ],
        tmp_path,
        tools=[echo_tool, sleep],
    )

    async def _skip_sleep_until_result() -> None:
        call_id = ""
        deadline = asyncio.get_running_loop().time() + 5.0
        while asyncio.get_running_loop().time() < deadline:
            for event in ctx.events:
                if (
                    isinstance(event, InvocationToolCallStart) and event.origin.kind == "turn"
                ) and event.tool_name == "sleep":
                    call_id = event.call_id
                    break
            if call_id:
                await ctx.bus.publish(SleepSkip(call_id=call_id))
            if any(
                (isinstance(event, InvocationToolCallResult) and event.origin.kind == "turn")
                and event.tool_name == "sleep"
                for event in ctx.events
            ):
                return
            await asyncio.sleep(0.01)
        raise TimeoutError("sleep result was not published")

    skip_task = asyncio.create_task(_skip_sleep_until_result())
    try:
        await ctx.bus.publish(UserMessage(text="echo, then wait briefly"))
        await skip_task
        await wait_for_idle(ctx)

        sleep_results = [
            event
            for event in ctx.events
            if (isinstance(event, InvocationToolCallResult) and event.origin.kind == "turn")
            and event.tool_name == "sleep"
        ]
        assert len(sleep_results) == 1
        assert sleep_results[0].call_id
        assert sleep_results[0].metadata["sleep_skipped"] is True
        assert sleep_results[0].result.startswith("Sleep skipped by user")
        assert extract_final_messages(ctx.events) == ["done after skip"]

        raw = await ctx.get_session_messages()
        echo_result = _result_content_with_prefix(_function_results_for_call_id(raw, echo_call_id), "echo: before")
        sleep_result = _result_content_with_prefix(
            _function_results_for_call_id(raw, sleep_call_id),
            "Sleep skipped by user",
        )

        assert TOOL_RESULT_METADATA_KEY not in (echo_result.get("additional_properties") or {})
        assert sleep_result["additional_properties"][TOOL_RESULT_METADATA_KEY] == {"sleep_skipped": True}
    finally:
        if not skip_task.done():
            skip_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await skip_task
        await ctx.cleanup()


def _function_results_for_call_id(messages: list[dict[str, Any]], call_id: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        for content in message.get("contents", []):
            if (
                isinstance(content, dict)
                and content.get("type") == "function_result"
                and content.get("call_id") == call_id
            ):
                results.append(content)
    return results


def _result_content_with_prefix(contents: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    for content in contents:
        result = content.get("result")
        if isinstance(result, str) and result.startswith(prefix):
            return content
    raise AssertionError(f"Missing function_result starting with {prefix!r}")
