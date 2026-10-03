# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared apps, looks, event-log writers and page helpers for the trajectory-dashboard widget tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from rich.console import Console
from rich.text import Text
from textual.app import App, ComposeResult
from textual.pilot import Pilot

from chrys.app.tui.widgets.trajectory import TrajectoryDashboard
from chrys.app.tui.widgets.trajectory.presentation import DashboardLook
from chrys.app.tui.widgets.trajectory.text_view import TrajectoryTextView
from chrys.foundation.trajectory.envelope import Link, LinkRelation
from chrys.foundation.trajectory.metadata import ANALYTICS_ITEM_ID_KEY
from tests.service.analytics._events import EventLog
from tests.support.waiting import wait_for

_NS = 1_000_000_000


class _DashboardApp(App[None]):
    def compose(self) -> ComposeResult:
        yield TrajectoryDashboard()


class _StyledDashboardApp(_DashboardApp):
    CSS = "TrajectoryTextView { background: #123456; color: #abcdef; }"


def plain_look(theme_variables: Mapping[str, str] | None = None) -> DashboardLook:
    """The look page builders draw with outside an App: a plain console, the given
    theme colours (each style's fallback otherwise) and English messages."""
    return DashboardLook(
        console=Console(force_terminal=False, _environ={}),
        theme_variables={} if theme_variables is None else dict(theme_variables),
        localizer=None,
    )


def _write_operations(path: Path, *, second_turn: bool = False, diagnostics: bool = False) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span(
        "preparation",
        "a" * 32,
        0,
        _NS,
        start_payload={"scope": "turn_preamble", "phase": "turn_dispatch"},
    )
    log.span(
        "model.run",
        "b" * 32,
        _NS,
        10 * _NS,
        links=(Link(relation=LinkRelation.CAUSED_BY, target_operation_id="a" * 32),),
    )
    log.span("model.cycle", "c" * 32, _NS, 10 * _NS, parent_operation_id="b" * 32)
    log.span(
        "model.exchange",
        "d" * 32,
        _NS,
        9 * _NS,
        parent_operation_id="c" * 32,
    )
    log.span(
        "preparation",
        "e" * 32,
        2 * _NS,
        3 * _NS,
        parent_operation_id="d" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "f" * 32},
    )
    log.span(
        "tool.operation",
        "f" * 32,
        3 * _NS,
        7 * _NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "Bash",
            "tool_kind": "shell",
            "argument_fingerprint": "0123456789abcdef",
            "parent_model_operation_id": "d" * 32,
        },
        links=(Link(relation=LinkRelation.CAUSED_BY, target_operation_id="e" * 32),),
    )
    log.span(
        "wait",
        "1" * 32,
        4 * _NS,
        5 * _NS,
        parent_operation_id="f" * 32,
        start_payload={"category": "approval"},
        finish_payload={"duration_ms": 900} if diagnostics else None,
    )
    log.span(
        "hook.operation",
        "2" * 32,
        6 * _NS,
        7 * _NS,
        parent_operation_id="f" * 32,
        start_payload={
            "hook_event": "after_tool_call",
            "hook_key": "register-session-to-git-after-turn",
            "execution_mode": "blocking",
            "scope": "turn",
        },
    )
    log.span(
        "sub_agent",
        "3" * 32,
        7 * _NS,
        9 * _NS,
        parent_operation_id="f" * 32,
        start_payload={"agent_profile": "Explore"},
    )
    log.add(
        "wait.started",
        8 * _NS,
        operation_id="4" * 32,
        parent_operation_id="d" * 32,
        payload={"category": "user_input"},
    )
    if diagnostics:
        log.span(
            "wait",
            "6" * 32,
            10 * _NS,
            11 * _NS,
            parent_operation_id="f" * 32,
            start_payload={"category": "new_wait_shape"},
        )
        log.add("trajectory.checkpoint", 11 * _NS, payload={"reason_code": "test"})
        log.add("profile.switched", 11 * _NS, payload={"kind": "agent"})
    log.add("turn.finished", 12 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    if second_turn:
        log.add("turn.started", 13 * _NS, turn_id="9" * 32, payload={"turn_number": 2})
        log.add(
            "turn.finished",
            14 * _NS,
            turn_id="9" * 32,
            payload={"end_reason": "cancelled", "duration_ms": 0},
        )
    log.write(path)
    if diagnostics:
        lines = path.read_bytes().splitlines()
        checkpoint_index = next(
            index for index, line in enumerate(lines) if json.loads(line)["event_type"] == "trajectory.checkpoint"
        )
        del lines[checkpoint_index]
        unsupported_index = next(
            index for index, line in enumerate(lines) if json.loads(line)["event_type"] == "profile.switched"
        )
        unsupported = json.loads(lines[unsupported_index])
        unsupported["event_type"] = "future.event"
        lines[unsupported_index] = json.dumps(unsupported, separators=(",", ":")).encode()
        lines.insert(unsupported_index, b"{corrupt trajectory test line}")
        path.write_bytes(b"\n".join(lines) + b"\n")


def _write_p1_operations(path: Path) -> None:
    turn_one = "4" * 32
    turn_two = "5" * 32
    verify_call = "7" * 32
    log = EventLog()
    log.coverage()
    log.span(
        "preparation",
        "a" * 32,
        0,
        _NS,
        turn_id=None,
        start_payload={"scope": "pre_turn", "phase": "admission"},
        finish_payload={"scope": "pre_turn", "outcome": "fresh_turn"},
    )
    log.add(
        "turn.started",
        2 * _NS,
        turn_id=turn_one,
        payload={"turn_number": 1, "preparation_scope_operation_id": "a" * 32},
    )
    _tool(log, "1" * 32, turn_one, 3, "search", "search", "search-1")
    _tool(log, "2" * 32, turn_one, 5, "read_file", "filesystem.read", "repeated", outcome="errored")
    _tool(log, "3" * 32, turn_one, 7, "write_file", "filesystem.write", "edit-1")
    log.add(
        "turn.finished",
        9 * _NS,
        turn_id=turn_one,
        payload={"end_reason": "cancelled", "duration_ms": 0},
    )
    log.span(
        "preparation",
        "b" * 32,
        10 * _NS,
        11 * _NS,
        turn_id=None,
        start_payload={"scope": "pre_turn", "phase": "admission"},
        finish_payload={"scope": "pre_turn", "outcome": "fresh_turn"},
    )
    log.add(
        "turn.started",
        12 * _NS,
        turn_id=turn_two,
        payload={"turn_number": 2, "preparation_scope_operation_id": "b" * 32},
    )
    _tool(log, "6" * 32, turn_two, 13, "read_file", "filesystem.read", "repeated", outcome="errored")
    _tool(log, "8" * 32, turn_two, 15, "read_file", "filesystem.read", "repeated")
    _tool(log, "9" * 32, turn_two, 17, "Bash", "shell", "verify", call_item_id=verify_call)
    _tool(log, "c" * 32, turn_two, 19, "write_file", "filesystem.write", "edit-2")
    log.span(
        "preparation",
        "d" * 32,
        21 * _NS,
        22 * _NS,
        turn_id=None,
        start_payload={"scope": "pre_turn", "phase": "admission"},
        finish_payload={"scope": "pre_turn", "outcome": "injected", "target_turn_id": turn_two},
    )
    log.span(
        "preparation",
        "e" * 32,
        23 * _NS,
        24 * _NS,
        turn_id=None,
        start_payload={"scope": "pre_turn", "phase": "admission"},
        finish_payload={"scope": "pre_turn", "outcome": "rejected"},
    )
    log.add(
        "turn.finished",
        25 * _NS,
        turn_id=turn_two,
        payload={"end_reason": "cancelled", "duration_ms": 0},
    )
    log.write(path)
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [
                        {
                            "contents": [
                                {
                                    "type": "function_call",
                                    "arguments": json.dumps({"command": "pytest -q"}),
                                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: verify_call},
                                }
                            ]
                        }
                    ],
                    "chrys_mutations": {
                        "turns": [
                            {
                                "turn_id": 1,
                                "detection_truncated": False,
                                "mutations": [
                                    {"path": "verified[bold].py", "before_hash": "a" * 64, "after_hash": "b" * 64}
                                ],
                            },
                            {
                                "turn_id": 2,
                                "detection_truncated": False,
                                "mutations": [
                                    {"path": "after_verify.py", "before_hash": "c" * 64, "after_hash": "d" * 64},
                                    {"path": "net_zero.py", "before_hash": "5a" * 32, "after_hash": "5a" * 32},
                                ],
                            },
                        ]
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _write_p2_operations(path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("model.exchange.started", 0, operation_id="1" * 32)
    log.add(
        "model.exchange.finished",
        _NS,
        operation_id="1" * 32,
        payload={
            "outcome": "success",
            "duration_ms": 1000,
            "usage": {
                "normalized": {
                    "input_total": 1000,
                    "output_total": 200,
                    "reasoning": 50,
                    "cache_read": 750,
                    "cache_creation": 25,
                }
            },
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            **{
                f"/payload/usage/normalized/{bucket}": {"source": "provider", "adapter_version": 1}
                for bucket in ("input_total", "output_total", "reasoning", "cache_read", "cache_creation")
            },
        },
    )
    mcp_id = "2" * 32
    log.add(
        "wait.started",
        _NS,
        operation_id="3" * 32,
        payload={"category": "mcp_connect", "server_name": "figma", "target_operation_id": mcp_id},
    )
    log.add(
        "wait.finished",
        2 * _NS,
        operation_id="3" * 32,
        payload={
            "category": "mcp_connect",
            "server_name": "figma",
            "target_operation_id": mcp_id,
            "duration_ms": 1000,
        },
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.add(
        "tool.operation.started",
        2 * _NS,
        operation_id=mcp_id,
        payload={
            "tool_name": "figma_render",
            "tool_kind": "mcp",
            "tool_context": {"server_name": "figma", "remote_name": "render"},
        },
    )
    log.add(
        "tool.payload.observed",
        3 * _NS,
        operation_id=mcp_id,
        payload={
            "model_visible_bytes": 4096,
            "local_token_estimate": 100,
            "truncated": True,
            "artifact_id": "artifact-1",
        },
    )
    log.add(
        "tool.operation.finished",
        4 * _NS,
        operation_id=mcp_id,
        payload={"outcome": "success", "duration_ms": 2000},
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    load_id = "4" * 32
    log.add(
        "tool.operation.started",
        5 * _NS,
        operation_id=load_id,
        payload={
            "tool_name": "load_skill",
            "tool_kind": "skill",
            "tool_context": {"skill_name": "slides", "skill_revision": "rev-a"},
        },
    )
    log.add(
        "tool.payload.observed",
        6 * _NS,
        operation_id=load_id,
        payload={"model_visible_bytes": 1000, "local_token_estimate": 250, "truncated": False},
    )
    log.add(
        "tool.operation.finished",
        6 * _NS,
        operation_id=load_id,
        payload={"outcome": "success", "duration_ms": 1000},
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.span(
        "tool.operation",
        "5" * 32,
        7 * _NS,
        8 * _NS,
        start_payload={
            "tool_name": "run_skill_script",
            "tool_kind": "skill",
            "tool_context": {
                "skill_name": "slides",
                "skill_revision": "rev-b",
                "script_name": "scripts/render.py",
            },
        },
        finish_payload={"outcome": "failed", "exit_code": 7},
    )
    log.span(
        "tool.operation",
        "6" * 32,
        8 * _NS,
        9 * _NS,
        start_payload={"tool_name": "load_skill", "tool_kind": "future.kind"},
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [],
                    "chrys_mutations": {
                        "turns": [
                            {
                                "turn_id": 1,
                                "detection_truncated": False,
                                "mutations": [
                                    {"path": "src/widget.py", "before_hash": "a" * 64, "after_hash": "b" * 64},
                                    {"path": "tests/test_widget.py", "before_hash": None, "after_hash": "c" * 64},
                                ],
                            }
                        ]
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _tool(
    log: EventLog,
    operation_id: str,
    turn_id: str,
    second: int,
    tool_name: str,
    tool_kind: str,
    fingerprint: str,
    *,
    outcome: str = "success",
    call_item_id: str | None = None,
) -> None:
    log.span(
        "tool.operation",
        operation_id,
        second * _NS,
        (second + 1) * _NS,
        turn_id=turn_id,
        start_payload={
            "tool_name": tool_name,
            "tool_kind": tool_kind,
            "call_item_id": call_item_id or operation_id,
            "argument_fingerprint": fingerprint,
        },
        finish_payload={"outcome": outcome},
    )


async def _wait_loaded(dashboard: TrajectoryDashboard, pilot: Pilot[None]) -> None:
    await wait_for(
        lambda: dashboard._analysis is not None,
        timeout=5,
        pilot=pilot,
        description="trajectory dashboard analysis",
    )


def _box_column_contents(lines: list[Text]) -> list[str]:
    """Per-box interior text with all whitespace removed, wrap- and column-safe.

    Side-by-side boxes interleave on each visual row; splitting on the border
    glyph and accumulating by column keeps every box's prose contiguous even
    when long lines fold inside a half-width box.
    """
    columns: dict[int, list[str]] = {}
    for line in lines:
        parts = line.plain.split("│")
        for index in range(1, len(parts) - 1, 2):
            columns.setdefault(index, []).append(parts[index])
    return ["".join("".join(parts).split()) for parts in columns.values()]


def _in_any_box(lines: list[Text], needle: str) -> bool:
    squashed = "".join(needle.split())
    return any(squashed in column for column in _box_column_contents(lines))


@asynccontextmanager
async def open_dashboard(
    path: Path,
    *,
    size: tuple[int, int],
    session_id: str = "session",
    app: App[None] | None = None,
) -> AsyncIterator[tuple[TrajectoryDashboard, Pilot[None]]]:
    """Mount a dashboard app (*app*, a plain one by default) and wait for its
    analysis and visible text viewport."""
    async with (_DashboardApp() if app is None else app).run_test(size=size) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session(session_id, path)
        await _wait_loaded(dashboard, pilot)
        # Loading commits lines before the hidden text view receives its next
        # layout. Its zero-sized viewport still reports the whole line width
        # as horizontal overflow until that display change reaches the screen.
        view = dashboard.query_one(TrajectoryTextView)
        await wait_for(
            lambda: view.scrollable_content_region.width > 0 and view.scrollable_content_region.height > 0,
            timeout=5,
            pilot=pilot,
            description="trajectory dashboard visible viewport",
        )
        yield dashboard, pilot


def page_text(view: TrajectoryTextView) -> str:
    """The text view's current virtualized lines joined into one plain-text page."""
    return "\n".join(line.plain for line in view._lines)
