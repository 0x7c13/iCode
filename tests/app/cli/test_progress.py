# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Headless progress lines: one stderr line per event, rendered from the bus facts."""

from __future__ import annotations

import io
import os
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from rich.console import Console

from chrys.app.cli.progress import (
    InvocationActivity,
    ProgressWriter,
    RunContext,
    TurnProgress,
    WorkflowProgress,
    display_path,
    format_duration,
    format_tokens,
    guarded,
    progress_console,
)
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AGENT_LOAD_PHASE_MCP,
    AGENT_LOAD_STATUS_RUNNING,
    WORKFLOW_NOTICE_DATA_DROPPED,
    WORKFLOW_OUTPUT_EMIT,
    AgentLoadFinished,
    AgentLoadProgress,
    CompactionFinished,
    CompactionStarted,
    InvocationAborted,
    InvocationCompactionCommitted,
    InvocationCompactionFinished,
    InvocationCompactionStarted,
    InvocationContextPressure,
    InvocationMessage,
    InvocationPaused,
    InvocationPresentationAttemptAccepted,
    InvocationPresentationAttemptRejected,
    InvocationRetryAttempt,
    InvocationStarted,
    InvocationToolCallArgsUpdated,
    InvocationToolCallResult,
    InvocationToolCallStart,
    InvocationToolCallStatusUpdated,
    ProvisionalPresentation,
    TodoListUpdated,
    ToolCompacted,
    Warning,
    WorkflowLoopIteration,
    WorkflowNodeOutput,
    WorkflowNodeStateChanged,
    WorkflowRunAccepted,
    WorkflowRunNotice,
    WorkflowRunStarted,
)
from chrys.foundation.i18n import Localizer
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.foundation.models.todos import TodoItem
from chrys.foundation.models.workflow_session import WorkflowIdentity, WorkflowSessionSelection, WorkspaceSnapshot
from chrys.foundation.tool_kinds import KIND_FILESYSTEM_READ, KIND_MCP, KIND_SHELL, KIND_SUB_AGENT, KIND_TODO
from tests.support.streams import FailingTextStream

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
TURN = InvocationOrigin("turn", "s1", "turn-1", None)
CHILD = InvocationOrigin("sub_agent", "s1", "child-1", TURN)
GRANDCHILD = InvocationOrigin("sub_agent", "s1", "grandchild-1", CHILD)
RENDER = Localizer("en").render


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _writer(reported: tuple[tuple[str, str], ...] = (), *, quiet: bool = False) -> tuple[ProgressWriter, io.StringIO]:
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
        width=400,
        highlight=False,
        markup=False,
        emoji=False,
        soft_wrap=True,
        _environ={},
    )
    return ProgressWriter(console, reported_warnings=reported, quiet=quiet), buffer


def _turn(context: RunContext | None = None) -> tuple[TurnProgress, io.StringIO]:
    writer, buffer = _writer()
    shown = context or RunContext()
    return TurnProgress(writer, render=RENDER, context=lambda: shown), buffer


def _lines(buffer: io.StringIO) -> list[str]:
    return buffer.getvalue().splitlines()


def _start(
    origin: InvocationOrigin, call_id: str, name: str, kind: str, args: dict[str, Any], at: float = 0, **kw: Any
) -> InvocationToolCallStart:
    return InvocationToolCallStart(
        origin=origin, call_id=call_id, tool_name=name, tool_kind=kind, args=args, timestamp=_at(at), **kw
    )


def _result(
    origin: InvocationOrigin, call_id: str, result: str = "ok", at: float = 0, duration_ms: int = 0, **kw: Any
) -> InvocationToolCallResult:
    return InvocationToolCallResult(
        origin=origin, call_id=call_id, result=result, duration_ms=duration_ms, timestamp=_at(at), **kw
    )


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (-1, "0.0s"),
        (0.04, "0.0s"),
        (0.44, "0.4s"),
        (12.34, "12.3s"),
        (59.94, "59.9s"),
        (59.96, "1m 00s"),
        (134, "2m 14s"),
        (3599.4, "59m 59s"),
        (3600, "1h 00m"),
        (3900, "1h 05m"),
    ],
)
def test_format_duration(seconds: float, text: str) -> None:
    assert format_duration(seconds) == text


@pytest.mark.parametrize(
    ("count", "text"),
    [
        (950, "950"),
        (999, "999"),
        (1000, "1k"),
        (41_200, "41k"),
        (999_499, "999k"),
        (999_600, "1.0M"),
        (999_999, "1.0M"),
        (1_234_567, "1.2M"),
    ],
)
def test_format_tokens(count: int, text: str) -> None:
    assert format_tokens(count) == text


def test_display_path_shortens_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert display_path(str(tmp_path / "proj")) == f"~{str(tmp_path / 'proj')[len(str(tmp_path)) :]}"
    assert display_path(f"{tmp_path}-other") == f"{tmp_path}-other"


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def test_a_tool_that_ends_right_after_its_start_gets_a_compact_end_line() -> None:
    progress, buffer = _turn()

    progress.handle(_start(TURN, "c1", "read_file", KIND_FILESYSTEM_READ, {"path": "src/app.py"}))
    progress.handle(_result(TURN, "c1", "contents", duration_ms=120))
    progress.handle(_start(TURN, "c2", "shell", KIND_SHELL, {"command": "false"}))
    progress.handle(_result(TURN, "c2", "", at=2.5, metadata={"shell_exit_code": 1}))

    assert _lines(buffer) == [
        "→ read   src/app.py",
        "  ✓ 0.1s",
        "→ shell  false",
        "  ✗ exit 1 · 2.5s",
    ]


def test_interleaved_tools_repeat_their_description_on_the_end_line() -> None:
    progress, buffer = _turn()

    progress.handle(_start(TURN, "c1", "read_file", KIND_FILESYSTEM_READ, {"path": "a.py"}))
    progress.handle(_start(TURN, "c2", "read_file", KIND_FILESYSTEM_READ, {"path": "b.py"}, at=0.1))
    progress.handle(_result(TURN, "c1", "Error: gone", at=0.5, metadata={"failed": True}))
    progress.handle(_result(TURN, "c2", at=0.6))
    progress.succeeded(duration=1, session_id="session-1")

    assert _lines(buffer) == [
        "→ read   a.py",
        "→ read   b.py",
        "  ✗ read   a.py · gone · 0.5s",
        "  ✓ read   b.py · 0.5s",
        "",
        "✓ Done · 1.0s · 2 tool calls · session session1",
    ]


def test_args_updates_refresh_the_description_the_end_line_repeats() -> None:
    progress, buffer = _turn()

    progress.handle(_start(TURN, "c1", "read_file", KIND_FILESYSTEM_READ, {}))
    progress.handle(
        InvocationToolCallArgsUpdated(origin=TURN, call_id="c1", tool_name="read_file", args={"path": "late.py"})
    )
    progress.handle(_start(TURN, "c2", "read_file", KIND_FILESYSTEM_READ, {"path": "x.py"}))
    progress.handle(_result(TURN, "c1", at=1))

    assert _lines(buffer)[-1] == "  ✓ read   late.py · 1.0s"


def test_call_ids_are_keyed_by_invocation() -> None:
    progress, buffer = _turn()

    progress.handle(_start(TURN, "c1", "read_file", KIND_FILESYSTEM_READ, {"path": "main.py"}))
    progress.handle(_start(CHILD, "c1", "read_file", KIND_FILESYSTEM_READ, {"path": "child.py"}, agent_name="Explore"))
    progress.handle(_result(CHILD, "c1", "Error: nope", at=1, metadata={"failed": True}, agent_name="Explore"))
    progress.handle(_result(TURN, "c1", at=2))

    assert _lines(buffer) == [
        "→ read   main.py",
        "  ↳ Explore → read   child.py",
        "  ↳ Explore ✗ read   child.py · nope · 1.0s",
        "  ✓ read   main.py · 2.0s",
    ]


def test_nested_calls_show_their_start_and_only_a_failed_end() -> None:
    progress, buffer = _turn()

    progress.handle(InvocationStarted(origin=CHILD, agent_name="Explore"))
    progress.handle(_start(CHILD, "c1", "grep", "search", {"pattern": "x"}))
    progress.handle(_result(CHILD, "c1", at=0.2))
    progress.handle(InvocationStarted(origin=GRANDCHILD, agent_name="General"))
    progress.handle(_start(GRANDCHILD, "c2", "read_file", KIND_FILESYSTEM_READ, {"path": "p"}))
    progress.handle(_result(GRANDCHILD, "c2", at=0.2))
    progress.succeeded(duration=1, session_id="")

    assert _lines(buffer) == [
        "  ↳ Explore → grep   x",
        "    ↳ General → read   p",
        "",
        "✓ Done · 1.0s · 0 tool calls",
    ]


def test_a_repeated_start_updates_the_open_call_and_a_start_after_the_result_is_a_new_one() -> None:
    progress, buffer = _turn()

    progress.handle(_start(TURN, "c1", "fetch", "", {}))
    progress.handle(_start(TURN, "c1", "web_fetch", "web_fetch", {"url": "https://a.example"}, at=5))
    progress.handle(_start(TURN, "c2", "shell", KIND_SHELL, {"command": "ls"}, at=6))
    progress.handle(_result(TURN, "c1", at=7))
    progress.handle(_start(TURN, "c1", "shell", KIND_SHELL, {"command": "pwd"}, at=8))
    progress.succeeded(duration=9, session_id="")

    assert _lines(buffer) == [
        "→ fetch",
        "→ shell  ls",
        # The first start's timestamp stands; the newer description replaces the old one.
        "  ✓ fetch  https://a.example · 7.0s",
        "→ shell  pwd",
        "",
        "✓ Done · 9.0s · 3 tool calls",
    ]


def test_a_hosted_call_ends_on_its_terminal_status_and_a_contradicting_result_corrects_it() -> None:
    progress, buffer = _turn()
    hosted = {"hosted_family": "shell", "provider_hosted": True}

    progress.handle(_start(TURN, "h1", "shell", "", {"commands": ["make"]}, **hosted))
    progress.handle(
        InvocationToolCallStatusUpdated(origin=TURN, call_id="h1", status="in_progress", timestamp=_at(1), **hosted)
    )
    progress.handle(
        InvocationToolCallStatusUpdated(origin=TURN, call_id="h1", status="completed", timestamp=_at(2), **hosted)
    )
    progress.handle(_result(TURN, "h1", "", at=3, metadata={"exit_code": 2}, **hosted))
    # A result for an already corrected call adds nothing.
    progress.handle(_start(TURN, "h2", "search", "", {"query": "q"}, hosted_family="search", provider_hosted=True))
    progress.handle(
        InvocationToolCallStatusUpdated(
            origin=TURN, call_id="h2", status="failed", metadata={"result_text": "Error: quota"}, timestamp=_at(1)
        )
    )
    progress.handle(_result(TURN, "h2", "Error: quota", at=2, provider_status="failed", hosted_family="search"))

    assert _lines(buffer) == [
        "→ shell  make",
        "  ✓ 2.0s",
        "  ✗ shell  make · exit 2",
        "→ search  q",
        "  ✗ quota · 1.0s",
    ]


def test_hosted_shell_timeouts_fail_with_their_reason() -> None:
    progress, buffer = _turn()

    progress.handle(_start(TURN, "h1", "shell", "", {"commands": ["sleep 99"]}, hosted_family="shell"))
    progress.handle(_result(TURN, "h1", "", at=4, metadata={"timed_out": True}, hosted_family="shell"))

    assert _lines(buffer) == ["→ shell  sleep 99", "  ✗ timed out · 4.0s"]


def test_the_builtin_todo_tool_is_silent_and_the_list_prints_instead() -> None:
    progress, buffer = _turn()
    items = [
        TodoItem("Read code", "completed"),
        TodoItem("Run tests", "in_progress", active_form="Running tests"),
        TodoItem("Ship", "pending"),
    ]

    progress.handle(_start(TURN, "t1", "todo_write", KIND_TODO, {"todos": []}))
    progress.handle(TodoListUpdated(items=items))
    progress.handle(_result(TURN, "t1"))
    progress.handle(TodoListUpdated(items=[items[0], TodoItem("Ship", "pending")]))
    progress.handle(TodoListUpdated(items=[items[0]]))
    progress.handle(TodoListUpdated(items=[]))
    progress.handle(_start(TURN, "t2", "todo_write", KIND_TODO, {}))
    progress.handle(_result(TURN, "t2", "Error: bad list", metadata={"failed": True}))
    # A remote tool that shares the name is an ordinary call.
    progress.handle(_start(TURN, "m1", "todo_write", KIND_MCP, {}))
    progress.succeeded(duration=1, session_id="")

    assert _lines(buffer) == [
        "Todo 1/3 · → Running tests",
        "Todo 1/2 · next: Ship",
        "Todo 1/1 · all done",
        "Todo list cleared",
        "  ✗ todo · bad list · 0.0s",
        "→ mcp    todo_write",
        "",
        "✓ Done · 1.0s · 1 tool call",
    ]


# ---------------------------------------------------------------------------
# commentary, retries, lifecycle, compaction, pressure
# ---------------------------------------------------------------------------


def test_commentary_prints_intermediate_turn_text_only() -> None:
    progress, buffer = _turn()

    progress.handle(InvocationMessage(origin=TURN, text="Looking at the parser.\n\nThen tests.", is_intermediate=True))
    progress.handle(InvocationMessage(origin=TURN, text="\x1b[31mred\x1b[0m", is_intermediate=True))
    progress.handle(InvocationMessage(origin=TURN, text="   \n", is_intermediate=True))
    progress.handle(InvocationMessage(origin=CHILD, text="child prose", is_intermediate=True))
    progress.handle(InvocationMessage(origin=TURN, text="the final answer", is_final=True))

    lines = _lines(buffer)
    assert lines[:3] == ["Looking at the parser.", "", "Then tests."]
    assert len(lines) == 4
    assert "\x1b" not in lines[3]
    assert "red" in lines[3]


def test_provisional_segments_print_once_accepted_and_never_when_rejected() -> None:
    progress, buffer = _turn()

    def segment(attempt: str, segment_id: str, text: str) -> InvocationMessage:
        return InvocationMessage(
            origin=TURN,
            text=text,
            is_intermediate=True,
            presentation=ProvisionalPresentation(attempt, segment_id),
        )

    progress.handle(segment("a1", "s1", "rejected text"))
    progress.handle(InvocationPresentationAttemptRejected(origin=TURN, attempt_id="a1"))
    progress.handle(segment("a2", "s1", "first"))
    progress.handle(segment("a2", "s2", "second"))
    progress.handle(segment("a2", "s3", "unlisted"))
    assert _lines(buffer) == []
    progress.handle(InvocationPresentationAttemptAccepted(origin=TURN, attempt_id="a2", segment_ids=("s2", "s1")))
    progress.handle(InvocationPresentationAttemptAccepted(origin=TURN, attempt_id="a2", segment_ids=("s2", "s1")))
    progress.handle(segment("a3", "s1", "never accepted"))

    assert _lines(buffer) == ["second", "first"]


def test_retry_pause_and_abort_lines() -> None:
    progress, buffer = _turn()

    progress.handle(
        InvocationRetryAttempt(origin=TURN, message="Rate limited.", attempt=2, max_attempts=18, delay_seconds=7)
    )
    progress.handle(InvocationRetryAttempt(origin=TURN, message="", attempt=1, max_attempts=0, delay_seconds=3))
    progress.handle(InvocationStarted(origin=CHILD, agent_name="Explore"))
    progress.handle(InvocationPaused(origin=CHILD, reason="framework_exc", last_error="boom"))
    progress.handle(InvocationAborted(origin=CHILD, last_error="gave up"))

    assert _lines(buffer) == [
        "↻ Rate limited. Retrying in 7s (attempt 2/18)",
        "↻ Retrying in 3s",
        "  ↳ Explore paused: boom",
        "  ↳ Explore ✗ gave up",
    ]


def test_compaction_lines_for_the_turn_and_a_sub_agent() -> None:
    progress, buffer = _turn()

    progress.handle(CompactionStarted())
    progress.handle(CompactionFinished(outcome="ok"))
    progress.handle(ToolCompacted(phase="phase3", tokens_before=9, tokens_after=1))
    progress.handle(ToolCompacted(phase="phase4", tokens_before=182_000, tokens_after=41_000))
    progress.handle(CompactionFinished(outcome="canceled"))
    progress.handle(CompactionFinished(outcome="failed", failure_reason="model refused"))
    progress.handle(InvocationStarted(origin=CHILD, agent_name="Explore"))
    progress.handle(InvocationCompactionStarted(origin=CHILD))
    progress.handle(InvocationCompactionFinished(origin=CHILD, outcome="ok"))
    progress.handle(InvocationCompactionCommitted(origin=CHILD))
    progress.handle(InvocationCompactionFinished(origin=CHILD, outcome="failed"))

    assert _lines(buffer) == [
        "• Compacting conversation…",
        "• Conversation compacted · 182k → 41k tokens",
        "• Compaction interrupted",
        "Warning: compaction failed (model refused)",
        "  ↳ Explore compacting conversation…",
        "  ↳ Explore conversation compacted",
        "  ↳ Explore Warning: compaction failed",
    ]


def test_context_pressure_names_the_agent_from_an_earlier_event() -> None:
    progress, buffer = _turn()

    progress.handle(_start(CHILD, "c1", "grep", "search", {"pattern": "x"}, agent_name="Explore"))
    progress.handle(InvocationContextPressure(origin=CHILD, reason="round_limit", source="sub_agent"))
    progress.handle(InvocationContextPressure(origin=GRANDCHILD, reason="", source="sub_agent"))

    lines = _lines(buffer)
    assert lines[1].startswith("  ↳ Explore Warning: ")
    assert lines[2].startswith("    ↳ sub-agent Warning: ")


def test_sub_agent_prompts_name_the_agent_on_the_parent_tool_line() -> None:
    progress, buffer = _turn()

    progress.handle(_start(TURN, "c1", "Explore", KIND_SUB_AGENT, {"prompt": "Find the parser"}))

    assert _lines(buffer) == ["→ agent  Explore: Find the parser"]


# ---------------------------------------------------------------------------
# turn startup and warnings
# ---------------------------------------------------------------------------


def test_the_ready_line_names_agent_model_session_and_directory_once() -> None:
    progress, buffer = _turn(RunContext(model="GPT Test", workdir="/work/proj"))

    progress.handle(AgentLoadProgress(phase=AGENT_LOAD_PHASE_MCP, status=AGENT_LOAD_STATUS_RUNNING, subject="github"))
    progress.handle(AgentLoadProgress(phase=AGENT_LOAD_PHASE_MCP, status=AGENT_LOAD_STATUS_RUNNING))
    progress.handle(AgentLoadProgress(phase=AGENT_LOAD_PHASE_MCP, status="failed", subject="github"))
    progress.handle(AgentLoadFinished(agent_profile="Code", display_name="Coder", session_id="abc-123"))
    progress.handle(AgentLoadFinished(agent_profile="Code", session_id="abc-123"))

    assert _lines(buffer) == [
        "• Connecting MCP server github…",
        "• Coder ready · GPT Test · session abc123 · /work/proj",
    ]


def test_session_and_run_ids_are_made_safe_for_the_terminal() -> None:
    """A restored or persisted id is not guaranteed hexadecimal; only its display copy is sanitized."""
    crafted = "\x1b[2Jab-cdef"
    progress, buffer = _turn()
    progress.handle(AgentLoadFinished(agent_profile="Code", session_id=crafted))
    progress.restored(crafted)
    progress.succeeded(duration=1, session_id=crafted)
    writer, workflow_buffer = _writer()
    workflow = WorkflowProgress(writer, render=RENDER)
    selection = WorkflowSessionSelection(
        crafted, WorkflowIdentity("review", "/review.py", "project"), WorkspaceSnapshot("/project")
    )
    workflow.handle(WorkflowRunAccepted(request_id="r", run_id=crafted, selection=selection))
    workflow.handle(WorkflowRunStarted(run_id=crafted, workflow_id="review"))
    workflow.handle(WorkflowLoopIteration(run_id=crafted, loop_id="loop", iteration=1, verdict="\x1b[31mexit"))

    shown = buffer.getvalue() + workflow_buffer.getvalue()
    assert "\x1b" not in shown
    assert shown.count("session ") == 4
    assert "run " in workflow_buffer.getvalue()


def test_warnings_print_once_per_code_and_message() -> None:
    writer, buffer = _writer(reported=(("old", "already said"),))
    progress = TurnProgress(writer, render=RENDER, context=RunContext)

    progress.handle(Warning(code="old", message="already said"))
    progress.handle(Warning(code="mcp.connect_failed", message="github: refused"))
    progress.handle(Warning(code="mcp.connect_failed", message="github: refused"))
    progress.handle(Warning(code="other", message="github: refused"))

    assert _lines(buffer) == ["Warning: github: refused", "Warning: github: refused"]


def test_warnings_notices_emits_and_node_errors_print_whole() -> None:
    """Their producers bound them already; progress only makes them safe for the terminal."""
    writer, buffer = _writer()
    progress = WorkflowProgress(writer, render=RENDER)
    long = "x" * 500

    progress.handle(Warning(code="settings", message=f"bad value {long} in \x1b[2Jsettings.yaml"))
    progress.handle(WorkflowRunNotice(run_id="run-1", code="something", message=long))
    progress.handle(WorkflowNodeOutput(run_id="run-1", node_id="n", kind=WORKFLOW_OUTPUT_EMIT, summary_text=long))
    progress.handle(
        _node("failed", "n", error='Traceback (most recent call last):\n\n  File "x"\x07\nValueError: boom')
    )

    assert _lines(buffer) == [
        f"Warning: bad value {long} in �[2Jsettings.yaml",
        f"Notice: {long}",
        f"  [n] {long}",
        "✗ [n] failed: Traceback (most recent call last):",
        '      File "x"�',
        "    ValueError: boom",
    ]


def test_quiet_turn_progress_prints_only_warnings_once() -> None:
    writer, buffer = _writer(reported=(("pending", "said at startup"),), quiet=True)
    progress = TurnProgress(writer, render=RENDER, context=RunContext)

    progress.restoring("abc123")
    for event in (
        Warning(code="pending", message="said at startup"),
        AgentLoadFinished(agent_profile="Code", display_name="Code", session_id="s1"),
        _start(TURN, "c1", "shell", KIND_SHELL, {"command": "ls"}),
        _result(TURN, "c1"),
        InvocationMessage(origin=TURN, text="Looking.", is_intermediate=True),
        Warning(code="mcp.connect_failed", message="github: refused"),
        Warning(code="mcp.connect_failed", message="github: refused"),
        CompactionStarted(),
        CompactionFinished(outcome="failed", failure_reason="model refused"),
        InvocationStarted(origin=CHILD, agent_name="Explore"),
        InvocationCompactionFinished(origin=CHILD, outcome="failed"),
        InvocationContextPressure(origin=CHILD, reason="round_limit", source="sub_agent"),
    ):
        progress.handle(event)
    progress.warnings([Warning(code="restore", message="restored settings differ")])
    progress.succeeded(duration=3, session_id="s1")

    lines = _lines(buffer)
    assert lines[:4] == [
        "Warning: github: refused",
        "Warning: compaction failed (model refused)",
        "  ↳ Explore Warning: compaction failed",
        lines[3],
    ]
    assert lines[3].startswith("  ↳ Explore Warning: ")
    assert lines[4:] == ["Warning: restored settings differ"]


def test_quiet_workflow_progress_prints_only_warnings_once() -> None:
    writer, buffer = _writer(quiet=True)
    progress = WorkflowProgress(writer, render=RENDER)

    progress.starting("review")
    progress.reported("preview", "said by the command")
    for event in (
        Warning(code="preview", message="said by the command"),
        WorkflowRunStarted(run_id="run-1", workflow_id="review"),
        _node("running", "build", invocation="inv-1"),
        _start(_node_origin("inv-1"), "c1", "shell", KIND_SHELL, {"command": "make"}),
        InvocationContextPressure(origin=_node_origin("inv-1"), reason="round_limit", source="workflow_node"),
        InvocationCompactionFinished(origin=_node_origin("inv-1"), outcome="failed", failure_reason="too big"),
        WorkflowNodeOutput(run_id="run-1", node_id="build", kind=WORKFLOW_OUTPUT_EMIT, summary_text="halfway"),
        _node("failed", "build", at=2, error="boom"),
        WorkflowRunNotice(run_id="run-1", code="something", message="Heads up"),
        Warning(code="w", message="careful"),
        Warning(code="w", message="careful"),
    ):
        progress.handle(event)
    progress.succeeded(duration=5)

    lines = _lines(buffer)
    assert len(lines) == 3
    assert lines[0].startswith("  [build] Warning: ")
    assert lines[1:] == ["  [build] Warning: compaction failed (too big)", "Warning: careful"]


def test_a_closed_progress_stream_ends_progress_and_never_redirects_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    err_read, err_write = os.pipe()
    out_read, out_write = os.pipe()
    try:
        with open(out_write, "w", encoding="utf-8", closefd=False) as stdout:
            stderr = FailingTextStream(err_write)
            monkeypatch.setattr(sys, "stderr", stderr)
            monkeypatch.setattr(sys, "stdout", stdout)
            writer = ProgressWriter(progress_console())

            writer.line("• Headless ready")
            writer.warning("github: refused")

        assert stderr.writes == 1
        # The answer's stream still reaches its reader.
        os.write(out_write, b"answer")
        assert os.read(out_read, 16) == b"answer"
        # The stderr descriptor now writes to devnull, so the exit-time flush cannot fail either.
        os.write(err_write, b"x")
        assert os.read(err_read, 1) == b""
    finally:
        for descriptor in (err_read, err_write, out_read, out_write):
            os.close(descriptor)


def test_a_failing_progress_stream_stops_progress_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    stderr = FailingTextStream(broken_pipe=False)
    monkeypatch.setattr(sys, "stderr", stderr)
    writer = ProgressWriter(progress_console())

    writer.line("first")
    writer.warning("second")

    # Only a broken pipe silences Rich itself; any other failure is ended by the writer.
    assert stderr.writes == 1
    closed, buffer = _writer()
    buffer.close()
    closed.line("third")
    closed.warning("fourth")


def test_a_display_failure_is_logged_and_the_stream_goes_on(caplog: pytest.LogCaptureFixture) -> None:
    seen: list[str] = []

    def handle(event: Any) -> None:
        if isinstance(event, Warning) and event.code == "boom":
            raise UnicodeEncodeError("ascii", "é", 0, 1, "stderr cannot encode it")
        seen.append(event.code)

    on_event = guarded(handle)
    on_event(Warning(code="boom", message="é"))
    on_event(Warning(code="after", message="fine"))

    assert seen == ["after"]
    assert "Headless progress could not show Warning" in caplog.text


async def test_a_restore_shows_its_mcp_connections_and_warnings_then_the_restored_line() -> None:
    bus = EventBus()
    progress, buffer = _turn(RunContext(model="GPT Test"))

    progress.restoring("abc")
    async with progress.observe_restore(bus):
        await bus.publish(
            AgentLoadProgress(phase=AGENT_LOAD_PHASE_MCP, status=AGENT_LOAD_STATUS_RUNNING, subject="github")
        )
        await bus.publish(Warning(code="mcp.connect_failed", message="github: refused"))
        await bus.publish(AgentLoadFinished(agent_profile="Code", display_name="Coder"))
    progress.warnings([Warning(code="mcp.connect_failed", message="github: refused")])
    progress.restored("abc-def")
    await bus.publish(Warning(code="late", message="after the restore"))
    # The turn's own ready event does not repeat the restored line.
    progress.handle(AgentLoadFinished(agent_profile="Code", display_name="Coder"))

    assert _lines(buffer) == [
        "• Restoring session abc…",
        "• Connecting MCP server github…",
        "Warning: github: refused",
        "• Restored session abcdef · Coder · GPT Test",
    ]


# ---------------------------------------------------------------------------
# workflow runs
# ---------------------------------------------------------------------------


def _node(state: str, node_id: str, *, at: float = 0, attempt: int = 1, invocation: str = "", **kw: Any):
    return WorkflowNodeStateChanged(
        run_id="run-1",
        node_id=node_id,
        activation_id=f"{node_id}-act",
        attempt=attempt,
        state=state,
        invocation_id=invocation,
        timestamp=_at(at),
        **kw,
    )


def _node_origin(invocation: str, attempt: int = 1) -> InvocationOrigin:
    return InvocationOrigin("workflow_node", "s1", invocation, None, attempt)


def test_workflow_progress_prints_header_node_states_and_node_activity() -> None:
    writer, buffer = _writer()
    progress = WorkflowProgress(writer, render=RENDER)
    selection = WorkflowSessionSelection(
        "session-1", WorkflowIdentity("review", "/review.py", "project"), WorkspaceSnapshot("/project")
    )

    progress.starting("review")
    progress.handle(WorkflowRunAccepted(request_id="r", run_id="run-1", selection=selection))
    progress.handle(WorkflowRunStarted(run_id="run-1234567890abcdef", workflow_id="review", title="Code review"))
    progress.handle(_node("running", "plan", invocation="inv-1"))
    progress.handle(_start(_node_origin("inv-1"), "c1", "read_file", KIND_FILESYSTEM_READ, {"path": "a.py"}))
    progress.handle(_result(_node_origin("inv-1"), "c1", at=0.5))
    progress.handle(InvocationMessage(origin=_node_origin("inv-1"), text="Planning.", is_intermediate=True))
    progress.handle(
        WorkflowNodeOutput(run_id="run-1", node_id="plan", kind=WORKFLOW_OUTPUT_EMIT, summary_text="halfway")
    )
    progress.handle(_node("completed", "plan", at=3))
    progress.handle(_node("running", "fix", at=3, invocation="inv-2"))
    progress.handle(_node("awaiting_retry", "fix", at=4, error="Rate limited\ndetail"))
    progress.handle(_node("running", "fix", at=10, attempt=2, invocation="inv-3"))
    progress.handle(_start(_node_origin("inv-3", 2), "c1", "shell", KIND_SHELL, {"command": "make"}, at=10))
    # An invocation the run never announced is not this run's to show.
    progress.handle(_start(_node_origin("inv-9"), "c9", "shell", KIND_SHELL, {"command": "rm -rf /"}))
    progress.handle(_node("failed", "fix", at=12, attempt=2, error="ValueError: boom"))
    progress.handle(_node("skipped", "ship"))
    progress.handle(WorkflowLoopIteration(run_id="run-1", loop_id="loop", iteration=2, verdict="continue"))
    progress.handle(WorkflowRunNotice(run_id="run-1", code="something", message="Heads up"))
    progress.handle(WorkflowRunNotice(run_id="run-1", code=WORKFLOW_NOTICE_DATA_DROPPED, message="dropped"))
    progress.handle(Warning(code="w", message="careful"))
    progress.succeeded(duration=75)

    assert _lines(buffer) == [
        "• Starting workflow review…",
        "Workflow Code review (review) · run run123456789 · session session1",
        "▸ [plan] running",
        "  [plan] → read   a.py",
        "  [plan]   ✓ 0.5s",
        "  [plan] Planning.",
        "  [plan] halfway",
        "✓ [plan] completed · 3.0s",
        "▸ [fix] running",
        "↻ [fix] awaiting retry: Rate limited",
        "    detail",
        "▸ [fix] running (attempt 2)",
        "  [fix] → shell  make",
        "✗ [fix] failed: ValueError: boom",
        "· [ship] skipped",
        "↻ [loop] iteration 2: continue",
        "Notice: Heads up",
        "Warning: careful",
        "✓ Workflow completed · 1m 15s",
    ]


def test_workflow_node_call_ids_repeat_across_attempts() -> None:
    writer, buffer = _writer()
    progress = WorkflowProgress(writer, render=RENDER)

    progress.handle(_node("running", "n", invocation="inv-1"))
    progress.handle(_start(_node_origin("inv-1", 1), "c1", "read_file", KIND_FILESYSTEM_READ, {"path": "a"}))
    progress.handle(_node("running", "n", attempt=2, invocation="inv-1"))
    progress.handle(_start(_node_origin("inv-1", 2), "c1", "read_file", KIND_FILESYSTEM_READ, {"path": "b"}))
    progress.handle(_result(_node_origin("inv-1", 2), "c1", at=1))

    assert _lines(buffer) == [
        "▸ [n] running",
        "  [n] → read   a",
        "▸ [n] running (attempt 2)",
        "  [n] → read   b",
        "  [n]   ✓ 1.0s",
    ]


def test_invocation_activity_ignores_origins_without_a_prefix() -> None:
    writer, buffer = _writer()
    activity = InvocationActivity(writer, prefix=lambda _origin: None, render=RENDER)

    activity.handle(_start(TURN, "c1", "read_file", KIND_FILESYSTEM_READ, {"path": "a"}))
    activity.handle(_result(TURN, "c1"))

    assert _lines(buffer) == []
    assert activity.top_level_calls == 0
