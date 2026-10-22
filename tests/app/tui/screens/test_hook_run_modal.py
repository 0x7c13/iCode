# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The hook run details dialog: its Input and Output tabs, polling, log tail and click-to-copy sections (plan section 4)."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import pytest
from rich.text import Text
from textual.app import App
from textual.content import Content
from textual.widgets import Button, Log, Static, TabbedContent

from chrys.app.tui.screens.dialogs.hook_run import HookRunModal, _CopySection
from chrys.foundation.models.hook_runs import HookRunApplied, HookRunDetail, HookRunIO, HookRunRecord
from chrys.service.trajectory.hooks import HookOutcome
from tests.support.tui_helpers import click_when_settled, rich_plain
from tests.support.waiting import wait_for, wait_until

_PAYLOAD = '{"event":"before_tool_call","tool":{"name":"zsh","args":{"command":"rm -rf build"}}}'


def _record(**fields: object) -> HookRunRecord:
    now = time.monotonic()
    base: dict[str, object] = {
        "run_id": "run-1",
        "session_id": "session-abc",
        "hook_id": "guard",
        "hook_event": "before_tool_call",
        "execution_mode": "blocking",
        "detach": False,
        "tool_name": "zsh",
        "sub_agent": "",
        "retry": False,
        "registered_at": now - 3.0,
        "started_at": now - 3.0,
        "ended_at": now - 2.5,
        "registered_wall": time.time() - 3.0,
        "state": "ended",
        "outcome": HookOutcome.SUCCESS,
        "exit_code": 0,
        "effect": None,
        "number": 7,
    }
    base.update(fields)
    return HookRunRecord(**base)  # type: ignore[arg-type]


def _io(**fields: object) -> HookRunIO:
    base: dict[str, object] = {
        "payload": _PAYLOAD,
        "payload_size": len(_PAYLOAD),
        "result_file": "missing",
        "decision": "",
        "stdout": "",
        "stderr": "",
        "stdout_dropped": 0,
        "stderr_dropped": 0,
    }
    base.update(fields)
    return HookRunIO(**base)  # type: ignore[arg-type]


def _detail(record: HookRunRecord | None = None, **fields: object) -> HookRunDetail:
    record = record or _record()
    base: dict[str, object] = {
        "record": record,
        "tool_call_id": "call-42",
        "timeout_seconds": 30.0,
        "on_error": "warn",
        "delivery": "best_effort",
        "source": "global",
        "source_path": "/config/hooks/hooks.yaml",
        "run_type": "command",
        "run_argv": ("python3", "guard.py", "--strict"),
        "run_cwd": "/repo",
        "run_env_names": ("API_TOKEN",),
        "log_path": "",
        "shared_log": False,
        "io": _io(),
        "applied": HookRunApplied() if record.state == "ended" else None,
    }
    base.update(fields)
    return HookRunDetail(**base)  # type: ignore[arg-type]


class _ModalApp(App):
    def __init__(self, detail: HookRunDetail) -> None:
        super().__init__()
        self.current: HookRunDetail | None = detail
        self.asked: list[str] = []
        self.modal = HookRunModal(detail, latest=self._latest)

    def _latest(self, run_id: str) -> HookRunDetail | None:
        self.asked.append(run_id)
        return self.current

    def on_mount(self) -> None:
        self.push_screen(self.modal)


def _plain(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Content):
        return content.plain
    if isinstance(content, Text):
        return content.plain
    return rich_plain(content)  # type: ignore[arg-type]


def _pane_text(modal: HookRunModal, pane: str) -> str:
    parts: list[str] = []
    for widget in modal.query_one(f"#{pane}").query("*"):
        if isinstance(widget, _CopySection):
            parts.append(_plain(widget.border_title))
        elif isinstance(widget, Static):
            parts.append(_plain(widget.content))
        elif isinstance(widget, Log):
            parts.append("\n".join(widget.lines))
    return "\n".join(parts)


def _active(modal: HookRunModal) -> str:
    return modal.query_one(TabbedContent).active


def _subtitle(modal: HookRunModal) -> str:
    return str(modal.query_one("#hook-run-container").border_subtitle or "")


def _title(modal: HookRunModal) -> str:
    return str(modal.query_one("#hook-run-container").border_title or "")


async def _open(pilot, app: _ModalApp) -> HookRunModal:
    await wait_for(lambda: app.screen is app.modal and app.modal.is_mounted, pilot=pilot, description="modal shown")
    return app.modal


# --- Title, tab choice, Input ---------------------------------------------------


async def test_an_ended_run_opens_on_output_with_its_title_and_state() -> None:
    record = _record(outcome=HookOutcome.FAILED, exit_code=3)
    app = _ModalApp(_detail(record, io=_io(stderr="boom"), applied=HookRunApplied(not_applied="failed")))
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _open(pilot, app)
        assert _title(modal) == "Hook #7 · guard"
        assert _active(modal) == "hook-run-output"
        subtitle = _subtitle(modal)
        assert "exit 3" in subtitle
        assert "500ms" in subtitle or "0.5s" in subtitle, subtitle


@pytest.mark.parametrize("state", ["running", "queued"])
async def test_a_live_run_opens_on_input_and_says_its_output_comes_later(state: str) -> None:
    now = time.monotonic()
    started = now - 1.0 if state == "running" else None
    record = _record(state=state, started_at=started, ended_at=None, outcome=None, exit_code=None)
    app = _ModalApp(_detail(record, applied=None))
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _open(pilot, app)
        assert _active(modal) == "hook-run-input"
        assert "Running… output shows when the hook ends" in _pane_text(modal, "hook-run-output")
        assert state in _subtitle(modal)


async def test_the_input_tab_shows_the_configured_run_and_the_payload() -> None:
    record = _record(registered_at=time.monotonic() - 10.0, started_at=time.monotonic() - 4.0)
    record = dataclasses.replace(record, registered_wall=time.mktime((2026, 10, 21, 9, 30, 0, 0, 0, -1)))
    app = _ModalApp(_detail(record, delivery="durable", on_error="block", timeout_seconds=12.0))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _open(pilot, app)
        text = _pane_text(modal, "hook-run-input")
        for expected in (
            "Event",
            "before_tool_call",
            "Mode",
            "blocking",
            "12",
            "block",
            "durable",
            "Session",
            "session-abc",
            "Triggered by",
            "Main agent",
            "zsh",
            "call-42",
            "Started",
            "09:30:06",
            "Queued",
            "Source",
            "global",
            "/config/hooks/hooks.yaml",
            "Run",
            "guard.py",
            "--strict",
            "/repo",
            "API_TOKEN",
            "Payload",
        ):
            assert expected in text, (expected, text)
        assert "Retry of an earlier run" not in text
        # Pretty-printed: one key per line.
        assert '"command": "rm -rf build"' in text
        assert "\n" in text.split("Payload", 1)[1]


async def test_the_input_tab_names_a_sub_agent_and_a_retry_and_hides_start_and_queue_when_unknown() -> None:
    now = time.monotonic()
    record = _record(sub_agent="Explore", retry=True, registered_at=now - 2.0, started_at=None, state="queued")
    record = dataclasses.replace(record, ended_at=None, outcome=None, exit_code=None)
    app = _ModalApp(_detail(record, applied=None))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _open(pilot, app)
        text = _pane_text(modal, "hook-run-input")
        assert "Explore" in text
        assert "Main agent" not in text
        assert "Retry of an earlier run" in text
        assert "Started" not in text, "still queued: no start time"


async def test_a_run_that_never_queued_shows_no_queue_time() -> None:
    record = _record()
    app = _ModalApp(_detail(record))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _open(pilot, app)
        text = _pane_text(modal, "hook-run-input")
        assert "Started" in text
        assert "Queued" not in text


async def test_a_cut_payload_says_so() -> None:
    io = _io(payload=_PAYLOAD[:40], payload_size=200_000)
    app = _ModalApp(_detail(io=io))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _open(pilot, app)
        assert "Payload cut at 64 KB of" in _pane_text(modal, "hook-run-input")


# --- Output ---------------------------------------------------------------------


async def test_the_output_tab_shows_the_decision_what_was_applied_and_both_streams() -> None:
    decision = {"action": "block", "reason": "rm denied"}
    io = _io(
        result_file="ok",
        decision=json.dumps(decision),
        stdout="checked 3 paths",
        stderr="warning: slow disk",
        stdout_dropped=1234,
    )
    record = _record(effect="blocked")
    app = _ModalApp(_detail(record, io=io, applied=HookRunApplied(block_reason="hook 'guard': rm denied")))
    async with app.run_test(size=(140, 60)) as pilot:
        modal = await _open(pilot, app)
        text = _pane_text(modal, "hook-run-output")
        for expected in (
            "Result",
            "blocked",
            "Decision",
            '"reason": "rm denied"',
            "Applied",
            "rm denied",
            "stdout",
            "checked 3 paths",
            "1234 bytes from the middle were not kept",
            "stderr",
            "warning: slow disk",
        ):
            assert expected in text, (expected, text)
        assert "No result file" not in text


@pytest.mark.parametrize(
    ("status", "expected"),
    [("missing", "No result file"), ("invalid", "The result file was invalid and was ignored")],
)
async def test_no_or_an_invalid_result_file_is_named(status: str, expected: str) -> None:
    app = _ModalApp(_detail(io=_io(result_file=status)))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _open(pilot, app)
        assert expected in _pane_text(modal, "hook-run-output")


@pytest.mark.parametrize(
    ("applied", "expected"),
    [
        (HookRunApplied(args='{"path": "/safe", "mode": "r"}'), '"path": "/safe"'),
        (HookRunApplied(reminder="remember the style guide"), "remember the style guide"),
        (HookRunApplied(extra_context="ruff found 2 issues"), "ruff found 2 issues"),
        (HookRunApplied(not_applied="nonblocking"), "Async and fire_and_forget hooks can't change the action"),
        (HookRunApplied(not_applied="unsupported"), "before_tool_call doesn't support this action"),
        (HookRunApplied(not_applied="failed"), "The hook failed, so its result file was ignored"),
        (
            HookRunApplied(not_applied="failed_blocked"),
            "The hook failed and on_error is block, so the action was blocked",
        ),
    ],
    ids=["args", "reminder", "context", "nonblocking", "unsupported", "failed", "failed-blocked"],
)
async def test_applied_names_what_was_taken_or_why_not(applied: HookRunApplied, expected: str) -> None:
    effect = (
        "modified_args"
        if applied.args
        else "added_reminder"
        if applied.reminder
        else "added_context"
        if applied.extra_context
        else "blocked"
        if applied.not_applied == "failed_blocked"
        else None
    )
    outcome = HookOutcome.FAILED if applied.not_applied in ("failed", "failed_blocked") else HookOutcome.SUCCESS
    record = _record(effect=effect, outcome=outcome, exit_code=1 if outcome == HookOutcome.FAILED else 0)
    app = _ModalApp(_detail(record, applied=applied, io=_io(result_file="ok", decision="{}")))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _open(pilot, app)
        assert expected in _pane_text(modal, "hook-run-output")


async def test_a_timed_out_run_shows_its_partial_output() -> None:
    record = _record(outcome=HookOutcome.TIMED_OUT, exit_code=None)
    app = _ModalApp(_detail(record, io=_io(stdout="half done"), applied=HookRunApplied(not_applied="failed")))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _open(pilot, app)
        text = _pane_text(modal, "hook-run-output")
        assert "timed out" in text
        assert "half done" in text


async def test_a_run_whose_details_are_gone_says_so_and_keeps_the_configured_fields() -> None:
    app = _ModalApp(_detail(io=None))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _open(pilot, app)
        gone = "Details are no longer kept for this run"
        assert gone in _pane_text(modal, "hook-run-output")
        text = _pane_text(modal, "hook-run-input")
        assert gone in text
        assert "guard.py" in text and "before_tool_call" in text
        assert "rm -rf build" not in text


# --- Polling --------------------------------------------------------------------


async def test_a_running_run_fills_its_output_once_it_ends_without_switching_tab() -> None:
    now = time.monotonic()
    record = _record(state="running", started_at=now - 1.0, ended_at=None, outcome=None, exit_code=None)
    app = _ModalApp(_detail(record, applied=None))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _open(pilot, app)
        await wait_for(lambda: app.asked, timeout=3.0, pilot=pilot, description="latest() polled")
        assert app.asked[0] == "run-1"
        ended = dataclasses.replace(record, state="ended", ended_at=time.monotonic(), outcome=HookOutcome.SUCCESS)
        app.current = _detail(ended, io=_io(stdout="all clean"), applied=HookRunApplied())
        await wait_for(
            lambda: "all clean" in _pane_text(modal, "hook-run-output"),
            timeout=4.0,
            pilot=pilot,
            description="output filled",
        )
        assert "Running…" not in _pane_text(modal, "hook-run-output")
        assert _active(modal) == "hook-run-input", "the user may be reading the payload"
        polled = len(app.asked)
        assert not await wait_until(lambda: len(app.asked) > polled, timeout=2.5, pilot=pilot), "polling stops"


# --- Detached runs --------------------------------------------------------------


async def test_a_detached_run_tails_its_log_and_notes_a_shared_one(tmp_path: Path) -> None:
    log = tmp_path / "handed-off.20261021T093000Z.log"
    log.write_text("worker started\n", encoding="utf-8")
    record = _record(detach=True, outcome=HookOutcome.DETACHED, exit_code=None, started_at=None)
    app = _ModalApp(_detail(record, log_path=str(log), shared_log=True, io=_io()))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _open(pilot, app)
        await wait_for(
            lambda: "worker started" in _pane_text(modal, "hook-run-output"),
            timeout=4.0,
            pilot=pilot,
            description="log tail",
        )
        text = _pane_text(modal, "hook-run-output")
        assert "Log" in text and str(log) in text
        assert "Another run of this hook started in the same second and writes to this log too" in text


# --- Click to copy --------------------------------------------------------------


@pytest.fixture
def copied(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", seen.append)
    return seen


def _sections(modal: HookRunModal, pane: str) -> list[_CopySection]:
    return list(modal.query_one(f"#{pane}").query(_CopySection))


def _notices(app: App) -> list[str]:
    return [str(notification.message) for notification in app._notifications]


async def _ended_with_output(pilot) -> HookRunModal:
    modal = await _open(pilot, pilot.app)
    tabs = modal.query_one(TabbedContent)
    await wait_for(lambda: tabs.active == "hook-run-output", pilot=pilot, description="opened on output")
    return modal


_OUTPUT_IO = {"result_file": "ok", "decision": '{"action":"allow"}', "stdout": "out-text\nline2", "stderr": "err-text"}


async def test_the_dialog_has_no_copy_buttons() -> None:
    app = _ModalApp(_detail(io=_io(**_OUTPUT_IO)))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _ended_with_output(pilot)
        assert not modal.query(Button)


async def test_a_click_anywhere_in_a_section_copies_that_section_and_says_which(copied: list[str]) -> None:
    app = _ModalApp(_detail(io=_io(**_OUTPUT_IO)))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _ended_with_output(pilot)
        result, decision, _applied, stdout, _stderr = _sections(modal, "hook-run-output")

        # On the body, not the heading: the whole section is the target.
        await click_when_settled(pilot, stdout.children[-1])
        await wait_for(lambda: copied == ["out-text\nline2"], pilot=pilot, description="stdout copied")
        assert "Copied stdout" in _notices(app)
        await click_when_settled(pilot, decision.children[0])
        await wait_for(lambda: copied[1:] == ['{"action":"allow"}'], pilot=pilot, description="decision copied")
        await click_when_settled(pilot, result.children[0])
        await wait_for(lambda: len(copied) == 3, pilot=pilot, description="result copied")
        assert copied[2].startswith("✓ succeeded")

        modal.query_one(TabbedContent).active = "hook-run-input"
        overview, payload = _sections(modal, "hook-run-input")
        await click_when_settled(pilot, payload.children[-1])
        await wait_for(lambda: len(copied) == 4, pilot=pilot, description="payload copied")
        assert json.loads(copied[3]) == json.loads(_PAYLOAD)
        assert "Copied Payload" in _notices(app)
        await click_when_settled(pilot, overview.children[0])
        await wait_for(lambda: len(copied) == 5, pilot=pilot, description="overview copied")
        assert copied[4].splitlines()[0] == "Event: before_tool_call"
        assert "Payload" not in copied[4]


async def test_a_click_that_ends_a_text_selection_copies_nothing_more(
    copied: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _ModalApp(_detail(io=_io(**_OUTPUT_IO)))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _ended_with_output(pilot)
        stdout = _sections(modal, "hook-run-output")[3]
        with monkeypatch.context() as selecting:
            selecting.setattr(type(modal), "get_selected_text", lambda self: "out")
            await click_when_settled(pilot, stdout.children[-1])
            assert not await wait_until(lambda: bool(copied), timeout=0.5, pilot=pilot)
        await click_when_settled(pilot, stdout.children[-1])
        await wait_for(lambda: copied == ["out-text\nline2"], pilot=pilot, description="copied once not selecting")


async def test_the_mouse_over_any_part_of_a_section_lights_the_whole_section() -> None:
    app = _ModalApp(_detail(io=_io(**_OUTPUT_IO)))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _ended_with_output(pilot)
        _, decision, applied, *_ = _sections(modal, "hook-run-output")
        await pilot.hover(decision.children[-1])
        await wait_for(lambda: decision.has_class("-hover"), pilot=pilot, description="lit from its body")
        await pilot.hover(decision.children[0])
        assert decision.has_class("-hover")
        await pilot.hover(applied.children[0])
        await wait_for(
            lambda: applied.has_class("-hover") and not decision.has_class("-hover"),
            pilot=pilot,
            description="the light moves to the next section",
        )


async def test_a_detached_run_copies_its_log_path_and_its_log_apart(copied: list[str], tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    log.write_text("worker started\n", encoding="utf-8")
    record = _record(detach=True, outcome=HookOutcome.DETACHED, exit_code=None, started_at=None)
    app = _ModalApp(_detail(record, log_path=str(log), shared_log=True, io=_io()))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _ended_with_output(pilot)
        await wait_for(
            lambda: "worker started" in _pane_text(modal, "hook-run-output"), pilot=pilot, description="log tail"
        )
        *_, path, tail = _sections(modal, "hook-run-output")
        await click_when_settled(pilot, path.children[0])
        await wait_for(lambda: copied == [str(log)], pilot=pilot, description="log path copied")
        assert "Copied log path" in _notices(app)
        await click_when_settled(pilot, tail.children[0])
        await wait_for(lambda: len(copied) == 2, pilot=pilot, description="log copied")
        assert copied[1] == "worker started", copied
        await pilot.press("c")
        await wait_for(lambda: len(copied) == 3, pilot=pilot, description="whole tab copied")
        assert f"## Log file\n{log}\n\n## Log\nworker started" in copied[2], copied[2]


async def test_c_copies_the_whole_open_tab(copied: list[str]) -> None:
    app = _ModalApp(_detail(io=_io(**_OUTPUT_IO)))
    async with app.run_test(size=(140, 50)) as pilot:
        modal = await _ended_with_output(pilot)
        await pilot.press("c")
        await wait_for(lambda: len(copied) == 1, pilot=pilot, description="output copied")
        output = copied[0]
        assert output.startswith("## Result\n✓ succeeded")
        sections = ["## Decision\n", "## Applied\n", "## stdout\nout-text", "## stderr\nerr-text"]
        positions = [output.find(section) for section in sections]
        assert -1 not in positions and positions == sorted(positions), output
        assert "## Payload" not in output

        modal.query_one(TabbedContent).active = "hook-run-input"
        await wait_for(lambda: _active(modal) == "hook-run-input", pilot=pilot, description="input tab")
        await pilot.press("c")
        await wait_for(lambda: len(copied) == 2, pilot=pilot, description="input copied")
        fields, payload = copied[1].split("\n\n## Payload\n")
        assert fields.splitlines()[:2] == ["## Overview", "Event: before_tool_call"]
        assert json.loads(payload) == json.loads(_PAYLOAD)
