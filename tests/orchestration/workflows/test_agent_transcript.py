# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real kernel workflow passes persist their tool history across terminal outcomes and retries."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import create_autospec

import pytest

import chrys.orchestration.workflows.agent_node as node_module
from chrys.foundation.events.types import (
    InvocationMessage,
    InvocationToolCallStart,
    WorkflowNodeRetryRequest,
    WorkflowNodeStateChanged,
)
from chrys.orchestration.workflows.agent_archive import AgentNodeArchive, CoalescedCheckpoint
from chrys.service.llm.mock import MockChatClient
from chrys.service.workflows.layout import run_dir
from chrys.service.workflows.store import read_node_value
from chrys.service.workflows.transcript import read_node_transcript, read_node_usage
from tests.orchestration.workflows._hosting import (
    confirm,
    make_host,
    make_profile,
    make_project,
    patch_runtime,
    run,
    write_workflow,
)
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

from ._transcript_support import ArchiveClient, source


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("stream", [False, True])
async def test_attempt_archive_survives_shutdown_with_reused_call_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, stream: bool
) -> None:
    project = make_project(tmp_path)
    target = project / "input.txt"
    target.write_text("Equal content must survive twice.")
    node_client = ArchiveClient(target, outcome)
    patch_runtime(monkeypatch, [MockChatClient(responses=[]), node_client], builtin_tools=True)
    write_workflow(project, "archive", source())
    host = make_host(tmp_path, project=project, profiles=[make_profile(builtins=["filesystem.read"])], stream=stream)
    transcript: list[InvocationMessage | InvocationToolCallStart] = []

    async def record(event: InvocationMessage | InvocationToolCallStart) -> None:
        if event.origin.kind == "workflow_node":
            transcript.append(event)

    await host.event_bus.subscribe(InvocationMessage, record)
    await host.event_bus.subscribe(InvocationToolCallStart, record)
    task = None
    try:
        await confirm(host, "archive")
        task = asyncio.create_task(run(host, "archive", input_text="Read twice"))
        if outcome == "cancelled":
            await wait_for(lambda: node_client.waiting.is_set() or task.done(), timeout=ENGINE_TURN_TIMEOUT)
            if task.done():
                await task
            assert node_client.waiting.is_set()
            session_dir = host.workflow_session_dir
            assert session_dir is not None
            run_id = host.engine.workflows.active_run_id
            assert run_id is not None
            directory = run_dir(session_dir, run_id)

            def checkpoint_has_results() -> bool:
                checkpoint = read_node_transcript(directory, "node@iter#1", 1)
                return (
                    checkpoint is not None
                    and checkpoint.status == "orphaned"
                    and len(
                        [c for m in checkpoint.replay.messages for c in m["contents"] if c["type"] == "function_result"]
                    )
                    == 2
                )

            await wait_for(checkpoint_has_results, timeout=ENGINE_TURN_TIMEOUT)
            await host.cancel_workflow()
        result, _events = await task
        assert result.outcome.value == ("node_failed" if outcome == "failed" else outcome)
        assert [event.text if isinstance(event, InvocationMessage) else event.tool_name for event in transcript] == [
            "First inspection.",
            "read_file",
            "Second inspection.",
            "read_file",
            *(["Done"] if outcome == "completed" else []),
        ]
        prose = [event for event in transcript if isinstance(event, InvocationMessage)]
        assert all(event.is_intermediate and not event.is_final for event in prose[:2])
        if outcome == "completed":
            assert prose[-1].is_final and not prose[-1].is_intermediate
        session_dir = host.workflow_session_dir
        assert session_dir is not None
        directory = run_dir(session_dir, result.run_id)
    finally:
        await host.shutdown()
        await host.event_bus.unsubscribe(InvocationMessage, record)
        await host.event_bus.unsubscribe(InvocationToolCallStart, record)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
    stored = read_node_transcript(directory, "node@iter#1", 1)
    assert stored is not None and stored.status == outcome
    messages = stored.replay.messages
    assert messages[0]["role"] == "user"
    calls = [c for m in messages for c in m["contents"] if c["type"] == "function_call"]
    results = [c for m in messages for c in m["contents"] if c["type"] == "function_result"]
    assert len(calls) == len(results) == 2
    assert [c["call_id"] for c in calls] == ["reused-call", "reused-call"]
    assert all("Equal content" in c["result"] for c in results)
    assert stored.usage.tool_calls == 2
    usage = read_node_usage(directory, "node@iter#1", 1)
    assert (
        usage is not None
        and usage.tool_calls == stored.usage.tool_calls
        and usage.usage_tokens == stored.usage.usage_tokens
    )
    if outcome == "failed":
        assert "Archive failure [literal]" in stored.error
    if outcome == "completed":
        assert messages[-1]["contents"][0]["text"] == "Done"
        # The node's value is the final answer alone, not the prose between its tool calls.
        output = read_node_value(directory, "node@iter#1", 1, "output", node_kind="agent")
        assert output is not None and output["value"] == {"text": "Done", "data": None}
        assert [item.value.text for item in result.outputs] == ["Done"]


async def test_manual_retry_keeps_the_failed_attempt_archive_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_project(tmp_path)
    target = project / "input.txt"
    target.write_text("Retained work")
    client = ArchiveClient(target, "failed")
    patch_runtime(monkeypatch, [MockChatClient(responses=[]), client], builtin_tools=True)
    write_workflow(project, "archive", source())
    host = make_host(
        tmp_path, project=project, profiles=[make_profile(builtins=["filesystem.read"])], allow_user_interaction=True
    )
    paused: list[WorkflowNodeStateChanged] = []

    async def state(event: WorkflowNodeStateChanged) -> None:
        if event.state == "awaiting_retry":
            paused.append(event)

    await host.event_bus.subscribe(WorkflowNodeStateChanged, state)
    task = None
    try:
        await confirm(host, "archive")
        task = asyncio.create_task(run(host, "archive", input_text="Read twice"))
        await wait_for(lambda: bool(paused) or task.done(), timeout=ENGINE_TURN_TIMEOUT)
        if task.done():
            await task
        assert paused
        event = paused[0]
        session_dir = host.workflow_session_dir
        assert session_dir is not None
        directory = run_dir(session_dir, event.run_id)
        first = read_node_transcript(directory, event.activation_id, 1)
        assert first is not None and first.status == "failed"
        client.outcome = "completed"
        await host.event_bus.publish(
            WorkflowNodeRetryRequest(
                run_id=event.run_id,
                node_id=event.node_id,
                activation_id=event.activation_id,
                expected_failed_attempt=1,
                request_id="retry",
            )
        )
        result, _events = await task
        assert result.outcome.value == "completed"
        assert read_node_transcript(directory, event.activation_id, 1) == first
        second = read_node_transcript(directory, event.activation_id, 2)
        assert second is not None and second.status == "completed"
        assert second.replay.messages[-1]["contents"][0]["text"] == "Done"
    finally:
        await host.shutdown()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)


async def test_kernel_tool_rounds_coalesce_and_final_snapshot_keeps_all_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = make_project(tmp_path)
    target = project / "input.txt"
    target.write_text("Retained result")
    client = ArchiveClient(target, "completed", pause_before_final=True)
    patch_runtime(monkeypatch, [MockChatClient(responses=[]), client], builtin_tools=True)
    write_workflow(project, "archive", source())
    host = make_host(tmp_path, project=project, profiles=[make_profile(builtins=["filesystem.read"])])
    checkpoint_factory = create_autospec(
        CoalescedCheckpoint, side_effect=lambda write: CoalescedCheckpoint(write, interval=3600)
    )
    monkeypatch.setattr(node_module, "CoalescedCheckpoint", checkpoint_factory)
    writes = create_autospec(AgentNodeArchive.write, side_effect=AgentNodeArchive.write)
    monkeypatch.setattr(AgentNodeArchive, "write", writes)
    task = None
    try:
        await confirm(host, "archive")
        task = asyncio.create_task(run(host, "archive", input_text="Read twice"))
        await wait_for(lambda: client.waiting.is_set() or task.done(), timeout=ENGINE_TURN_TIMEOUT)
        if task.done():
            await task
        assert client.waiting.is_set()
        assert writes.await_count == 1  # The pass-start snapshot precedes both tool rounds.
        client.release_final.set()
        result, _ = await task
        assert result.outcome.value == "completed"
        assert writes.await_count == 2
        assert writes.await_args.kwargs["status"] == "completed"
        assert host.workflow_session_dir is not None
        stored = read_node_transcript(run_dir(host.workflow_session_dir, result.run_id), "node@iter#1", 1)
        assert stored is not None and stored.status == "completed"
        results = [c for m in stored.replay.messages for c in m["contents"] if c["type"] == "function_result"]
        assert len(results) == 2
    finally:
        client.release_final.set()
        await host.shutdown()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
