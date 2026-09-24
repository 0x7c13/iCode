# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ACP ordinal, usage, audit, and wire input across a manual Retry."""

from __future__ import annotations

import asyncio
import json
from inspect import signature
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.types import (
    InvocationPaused,
    InvocationResumed,
    InvocationRetryRequested,
    InvocationStarted,
    InvocationToolCallResult,
    InvocationToolCallStart,
    UserMessage,
)
from chrys.service.session.sub_agent_transcript import load_persisted_sub_agent_transcript
from tests.orchestration.sub_agents.test_acp_engine import _start_acp_engine
from tests.support.engines import AgentEngineFactory
from tests.support.event_capture import EventNormalizer, capture_event_sequence
from tests.support.waiting import ENGINE_TEST_WAIT_TIMEOUT, await_run_task_chain, with_wait_deadline


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_manual_retry_preserves_invocation_and_advances_transport_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory
) -> None:
    wire_trace = tmp_path / "wire.jsonl"
    engine, bus, _client, _captured = await _start_acp_engine(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        agent_engine=agent_engine,
        scenario="retry_reused_tool_usage",
        flag_path=wire_trace,
    )
    tools = engine.current.loaded.sub_agent_tools
    assert tools is not None
    original_usage = tools._on_sub_agent_usage
    assert original_usage is not None
    usage = create_autospec(original_usage, side_effect=original_usage)
    monkeypatch.setattr(tools, "_on_sub_agent_usage", usage)

    paused = asyncio.Event()

    async def on_paused(event):
        paused.set()

    await bus.subscribe(InvocationPaused, on_paused)
    async with capture_event_sequence(
        bus,
        InvocationStarted,
        InvocationToolCallStart,
        InvocationToolCallResult,
        InvocationPaused,
        InvocationResumed,
        InvocationToolCallResult,
    ) as events:

        async def retry_when_paused() -> tuple[InvocationPaused, Path, Path, Path]:
            await paused.wait()
            pause = next(
                event for event in events if (isinstance(event, InvocationPaused) and event.origin.kind == "sub_agent")
            )
            session_dir = engine.session.session_dir
            assert session_dir is not None
            (audit_path,) = (session_dir / "sub_agents" / "sessions").glob("*.json")
            (pending_path,) = (session_dir / "sub_agents" / "pending").glob("*.json")
            snapshot = json.loads(audit_path.read_text(encoding="utf-8"))
            assert snapshot["meta"]["status"] == "paused"
            assert snapshot["meta"]["total_usage_tokens"] == 21
            assert snapshot["acp_state"]["attempts"] == [{"attempt": 1, "acp_session_id": "stub-session"}]
            assert json.loads(pending_path.read_text(encoding="utf-8"))["invocation_id"] == pause.origin.invocation_id
            await bus.publish(InvocationRetryRequested(invocation_id=pause.origin.invocation_id))
            return pause, session_dir, audit_path, pending_path

        # The real transport allows a 20s handshake. Drive the human response
        # concurrently and budget startup, retry, and final save together;
        # a separate 15s pause wait can expire during a healthy handshake.
        async with asyncio.TaskGroup() as group:
            decision = group.create_task(retry_when_paused())
            await bus.publish(UserMessage(text="Delegate this"))
            await await_run_task_chain(engine, turn_state=engine.turns.turn_state, timeout=ENGINE_TEST_WAIT_TIMEOUT)
            assert paused.is_set(), "ACP invocation completed without the expected pause"
        pause, session_dir, audit_path, pending_path = decision.result()

    # The former child-start subscription excluded the parent tool start.
    events = [
        event for event in events if not (isinstance(event, InvocationToolCallStart) and event.origin.kind == "turn")
    ]
    starts = [
        event for event in events if (isinstance(event, InvocationToolCallStart) and event.origin.kind == "sub_agent")
    ]
    results = [
        event for event in events if (isinstance(event, InvocationToolCallResult) and event.origin.kind == "sub_agent")
    ]
    # The translator re-publishes start with ensure=True before its terminal.
    # Preserve that existing presentation behavior; IDs identify occurrences.
    assert [event.call_id for event in starts] == ["a1:reused", "a1:reused", "a2:reused", "a2:reused"]
    assert [event.call_id for event in results] == ["a1:reused", "a2:reused"]
    normalizer = EventNormalizer()
    normalized = [normalizer.event(event) for event in events]
    child_rows = [row for _, row in normalized if row["origin"]["kind"] == "sub_agent"]
    assert len({row["origin"]["invocation_id"] for row in child_rows}) == 1
    assert child_rows[0]["origin"]["invocation_id"] == normalizer.identity(pause.origin.invocation_id)
    assert child_rows[0]["origin"]["invocation_id"] != pause.origin.invocation_id
    assert all(row["origin"] == child_rows[0]["origin"] for row in child_rows)
    assert [type(event).__name__ for event in events] == [
        "InvocationStarted",
        "InvocationToolCallStart",
        "InvocationToolCallStart",
        "InvocationToolCallResult",
        "InvocationPaused",
        "InvocationResumed",
        "InvocationToolCallStart",
        "InvocationToolCallStart",
        "InvocationToolCallResult",
        "InvocationToolCallResult",
    ]
    assert usage.call_count == 2
    usage_arguments = [
        signature(original_usage).bind(*call.args, **call.kwargs).arguments for call in usage.call_args_list
    ]
    assert [args["usage_source_id"] for args in usage_arguments] == [
        f"{pause.origin.invocation_id}:a1",
        f"{pause.origin.invocation_id}:a2",
    ]
    assert [args["authoritative_total"] for args in usage_arguments] == [21, 8]
    assert engine.session.runtime_meta.total_session_tokens == 29
    assert engine.session.runtime_meta.total_session_input_tokens == 11
    assert engine.session.runtime_meta.total_session_output_tokens == 18
    assert not pending_path.exists()
    terminal = json.loads(audit_path.read_text(encoding="utf-8"))
    state = terminal["acp_state"]
    assert [attempt["attempt"] for attempt in state["attempts"]] == [1, 2]
    assert state["successful_attempt"] == 2
    assert {update["attempt"] for update in state["translated_updates"]} == {1, 2}
    assert terminal["meta"]["total_usage_tokens"] == 29
    assert terminal["meta"]["usage_unreported_attempts"] == 0
    transcript = await load_persisted_sub_agent_transcript(session_dir, audit_path.name)
    assert transcript is not None
    assert transcript.represents_result_text("stub response")
    assert not transcript.represents_result_text("failed pass text\n\nstub response")
    wire = [json.loads(line) for line in wire_trace.read_text(encoding="utf-8").splitlines()]
    assert len(wire) == 2
    assert wire[0]["pid"] != wire[1]["pid"]
    assert [item["session_id"] for item in wire] == ["stub-session", "stub-session"]
    assert [item["prompt"] for item in wire] == [["inspect the project"], ["inspect the project"]]


@pytest.mark.parametrize("case", ["happy", "error_prefix", "empty", "refusal", "abort", "retry", "cascade"])
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_acp_shell_preserves_hook_writer_and_parent_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory, case: str
) -> None:
    from chrys.foundation.events.types import InvocationAbortRequested, UserInterrupt
    from chrys.service.hooks.events import HookEvent
    from chrys.service.hooks.manager import HookManager

    scenario = (
        "retry_reused_tool_usage" if case == "retry" else "prompt_cancelled" if case in {"abort", "cascade"} else case
    )
    engine, bus, _client, captured = await _start_acp_engine(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        agent_engine=agent_engine,
        scenario=scenario,
        flag_path=tmp_path / "wire.jsonl",
    )
    tools = engine.current.loaded.sub_agent_tools
    assert tools is not None
    hooks = create_autospec(HookManager, instance=True)
    hooks.has_hooks_for.return_value = True
    tools._hook_manager = hooks
    paused = asyncio.Event()

    async def on_paused(event):
        paused.set()

    await bus.subscribe(InvocationPaused, on_paused)

    async def decide_when_paused() -> None:
        await paused.wait()
        pause = captured[InvocationPaused][0]
        assert not [call for call in hooks.fire.call_args_list if call.args[0] == HookEvent.SUB_AGENT_END]
        (pending,) = (engine.session.session_dir / "sub_agents" / "pending").glob("*.json")
        assert pending.exists()
        (audit,) = (engine.session.session_dir / "sub_agents" / "sessions").glob("*.json")
        assert json.loads(audit.read_text(encoding="utf-8"))["meta"]["status"] == "paused"
        if case == "retry":
            await bus.publish(InvocationRetryRequested(invocation_id=pause.origin.invocation_id))
        elif case == "abort":
            await bus.publish(InvocationAbortRequested(invocation_id=pause.origin.invocation_id))
        else:
            await bus.publish(UserInterrupt())

    try:
        async with asyncio.TaskGroup() as group:
            if case in {"abort", "retry", "cascade"}:
                group.create_task(decide_when_paused())
            await bus.publish(UserMessage(text="Delegate this"))
            await await_run_task_chain(engine, turn_state=engine.turns.turn_state, timeout=ENGINE_TEST_WAIT_TIMEOUT)
            if case in {"abort", "retry", "cascade"}:
                assert paused.is_set(), "ACP invocation completed without the expected pause"
        (audit,) = (engine.session.session_dir / "sub_agents" / "sessions").glob("*.json")
        terminal = json.loads(audit.read_text(encoding="utf-8"))
        ends = [call.args[1] for call in hooks.fire.call_args_list if call.args[0] == HookEvent.SUB_AGENT_END]
        expected_hook = "ok" if case in {"happy", "retry"} else "cancelled" if case == "cascade" else "failed"
        assert len(ends) == 1
        assert ends[0]["status"] == expected_hook
        assert terminal["meta"]["status"] == ("completed" if expected_hook == "ok" else expected_hook)
        assert terminal["acp_state"].get("successful_attempt") == (
            2 if case == "retry" else 1 if case == "happy" else None
        )
        assert not list((engine.session.session_dir / "sub_agents" / "pending").glob("*.json"))
        assert tools._controllers == {}
        assert tools._total_active == 0
        if case != "cascade":
            results = [event for event in captured[InvocationToolCallResult] if event.origin.kind == "turn"]
            assert len(results) == 1
            assert results[0].result == terminal["meta"]["result_preview"] or results[0].result.startswith(
                terminal["meta"]["result_preview"]
            )
            if case == "error_prefix":
                assert results[0].result.startswith("Error:")
                assert results[0].metadata["failed"] is False
    finally:
        await engine.shutdown()
