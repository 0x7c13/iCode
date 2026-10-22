# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""HookRunModal — one hook run's input (what it was handed) and output (what it returned).

The run's details come from the engine's run log, which keeps them in memory
only. While the run is queued or running, the modal reads it again once a
second and fills in the output when it ends; the output is collected when the
hook's process exits, so nothing streams before that.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual import on
from textual.containers import ScrollableContainer, VerticalGroup
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Log, Static, TabbedContent, TabPane

from chrys.app.tui.binding_display import CLOSE_BINDING, COPY_BINDING, localized_binding
from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.screens.dialogs.background_task import LogTail
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.screens.dialogs.tool_view import (
    COPIED_INPUT,
    COPIED_INPUT_TOO_LARGE,
    COPIED_INPUT_UNAVAILABLE,
    COPIED_OUTPUT,
    COPIED_OUTPUT_TOO_LARGE,
    COPIED_OUTPUT_UNAVAILABLE,
    copy_and_notify,
)
from chrys.app.tui.util.removal import remove_children_shielded
from chrys.app.tui.widgets.chat.tool_call import fmt_duration
from chrys.app.tui.widgets.chat.tool_view_builders import build_code_view
from chrys.app.tui.widgets.sidebar.hooks import run_duration, run_icon, run_outcome
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import sanitize_legacy_scalar
from chrys.foundation.platform.files import surrogate_safe_text

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.app import ComposeResult
    from textual.events import Click
    from textual.widget import Widget

    from chrys.foundation.models.hook_runs import HookRunDetail

_POLL_SECONDS = 1.0
_MAX_LOG_LINES = 10_000
_QUEUED_SHOWN_AFTER_SECONDS = 0.01
"""A shorter wait for a slot is no queueing worth naming."""

_TITLE = msg("tui.hook_run.title", fallback="Hook #{number} · {hook_id}")
_INPUT = msg("tui.hook_run.tab.input", fallback="Input")
_OUTPUT = msg("tui.hook_run.tab.output", fallback="Output")
_EVENT = msg("tui.hook_run.field.event", fallback="Event")
_MODE = msg("tui.hook_run.field.mode", fallback="Mode")
_SESSION = msg("tui.hook_run.field.session", fallback="Session")
_TRIGGERED_BY = msg("tui.hook_run.field.triggered_by", fallback="Triggered by")
_STARTED = msg("tui.hook_run.field.started", fallback="Started")
_QUEUED = msg("tui.hook_run.field.queued", fallback="Queued")
_SOURCE = msg("tui.hook_run.field.source", fallback="Source")
_RUN = msg("tui.hook_run.field.run", fallback="Run")
_OVERVIEW = msg("tui.hook_run.field.overview", fallback="Overview")
_PAYLOAD = msg("tui.hook_run.field.payload", fallback="Payload")
_RESULT = msg("tui.hook_run.field.result", fallback="Result")
_DECISION = msg("tui.hook_run.field.decision", fallback="Decision")
_APPLIED = msg("tui.hook_run.field.applied", fallback="Applied")
_STDOUT = msg("tui.hook_run.field.stdout", fallback="stdout")
_STDERR = msg("tui.hook_run.field.stderr", fallback="stderr")
_LOG = msg("tui.hook_run.field.log", fallback="Log")
_LOG_FILE = msg("tui.hook_run.field.log_file", fallback="Log file")
_MAIN_AGENT = msg("tui.hook_run.main_agent", fallback="Main agent")
_RETRY = msg("tui.hook_run.retry", fallback="Retry of an earlier run")
_SUCCEEDED = msg("tui.hook_run.succeeded", fallback="succeeded")
_EXIT_CODE = msg("tui.hook_run.exit_code", fallback="exit code {code}")
_RUNNING = msg("tui.hook_run.running", fallback="Running… output shows when the hook ends")
_DETAILS_GONE = msg("tui.hook_run.details_gone", fallback="Details are no longer kept for this run")
_NO_RESULT_FILE = msg("tui.hook_run.no_result_file", fallback="No result file")
_INVALID_RESULT_FILE = msg("tui.hook_run.invalid_result_file", fallback="The result file was invalid and was ignored")
_APPLIED_NOTHING = msg("tui.hook_run.applied.nothing", fallback="Nothing")
_APPLIED_BLOCKED = msg("tui.hook_run.applied.blocked", fallback="Blocked the action: {reason}")
_APPLIED_ARGS = msg("tui.hook_run.applied.args", fallback="Changed the tool arguments to:")
_APPLIED_REMINDER = msg("tui.hook_run.applied.reminder", fallback="Added a reminder for the model:")
_APPLIED_CONTEXT = msg("tui.hook_run.applied.context", fallback="Added to the tool result:")
_NOT_APPLIED_NONBLOCKING = msg(
    "tui.hook_run.not_applied.nonblocking", fallback="Async and fire_and_forget hooks can't change the action"
)
_NOT_APPLIED_UNSUPPORTED = msg("tui.hook_run.not_applied.unsupported", fallback="{event} doesn't support this action")
_NOT_APPLIED_FAILED = msg("tui.hook_run.not_applied.failed", fallback="The hook failed, so its result file was ignored")
_NOT_APPLIED_FAILED_BLOCKED = msg(
    "tui.hook_run.not_applied.failed_blocked",
    fallback="The hook failed and on_error is block, so the action was blocked",
)
_TRUNCATED = msg("tui.hook_run.truncated", fallback="{bytes} bytes from the middle were not kept")
_PAYLOAD_TRUNCATED = msg("tui.hook_run.payload_truncated", fallback="Payload cut at 64 KB of {size}")
_SHARED_LOG = msg(
    "tui.hook_run.shared_log",
    fallback="Another run of this hook started in the same second and writes to this log too",
)
_COPIED_SECTION = msg("tui.hook_run.copy.section", fallback="Copied {section}")
_COPIED_SECTION_TOO_LARGE = msg(
    "tui.hook_run.copy.section_too_large",
    fallback="Copied {section} (terminal clipboard skipped: payload too large)",
)
_COPIED_SECTION_UNAVAILABLE = msg(
    "tui.hook_run.copy.section_unavailable", fallback="Copied {section} (terminal clipboard unavailable)"
)
_COPIED_LOG_PATH = msg("tui.hook_run.copy.log_path", fallback="Copied log path")
_COPIED_LOG_PATH_UNAVAILABLE = msg(
    "tui.hook_run.copy.log_path_unavailable", fallback="Copied log path (terminal clipboard unavailable)"
)


def _wall_time(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _size(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / (1024 * 1024):.1f} MB"


def _pretty_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except ValueError:
        return text


@dataclass(frozen=True, slots=True)
class _Fields:
    rows: tuple[tuple[str, Text], ...]
    id: str | None = None


@dataclass(frozen=True, slots=True)
class _Note:
    text: str
    copied: bool = True
    """False for a remark about the section, left out of what it copies."""


@dataclass(frozen=True, slots=True)
class _Code:
    language: str
    text: str


@dataclass(frozen=True, slots=True)
class _Line:
    text: Text


@dataclass(frozen=True, slots=True)
class _LogPath:
    path: str


@dataclass(frozen=True, slots=True)
class _LogTail:
    pass


_Block = _Fields | _Line | _Note | _Code | _LogPath | _LogTail


@dataclass(frozen=True, slots=True)
class _Section:
    """One part of a tab, copied whole by a click: its widgets and its copied text come from its blocks."""

    name: str
    """Its box's title, and what the copied notice names."""
    blocks: tuple[_Block, ...]


class _CopySection(VerticalGroup):
    """A section of a tab: a click anywhere in it copies the section, unless it ends a text selection."""

    class Clicked(Message):
        def __init__(self, key: str) -> None:
            super().__init__()
            self.key = key

    def __init__(self, *children: Widget, title: str, key: str) -> None:
        super().__init__(*children, classes="hook-run-section")
        self.border_title = Text(title)
        self.key = key

    def on_click(self, event: Click) -> None:
        event.stop()
        if self.screen.get_selected_text():
            return
        self.post_message(self.Clicked(self.key))

    # Textual marks only the innermost widget hovered: the whole section lights up as one.
    def on_enter(self) -> None:
        self.set_class(True, "-hover")

    def on_leave(self) -> None:
        if not (self.screen.is_current and self.region.contains_point(self.app.mouse_position)):
            self.set_class(False, "-hover")


def _fields_plain(fields: _Fields) -> str:
    lines = []
    for label, value in fields.rows:
        first, *rest = value.plain.split("\n")
        lines.append(f"{label}: {first}")
        lines.extend(" " * (len(label) + 2) + line for line in rest)
    return "\n".join(lines)


class HookRunModal(BaseDialog[None]):
    """A hook run's payload and run settings, then its result file, output and what iCode took from it."""

    CSS_PATH = "hook_run.tcss"

    BINDINGS: ClassVar[list] = [
        localized_binding("escape", "close", CLOSE_BINDING, show=False, priority=True),
        localized_binding("q", "close", CLOSE_BINDING, show=False),
        localized_binding("c", "copy", COPY_BINDING, show=False),
    ]

    def __init__(self, detail: HookRunDetail, *, latest: Callable[[str], HookRunDetail | None]) -> None:
        super().__init__()
        self._shown = detail
        self._latest = latest
        """The run as the run log has it now; None once the log no longer keeps it."""
        self._tail = LogTail(detail.log_path) if detail.log_path else None
        self._reading = False
        self._rebuilding = False

    # ── layout ───────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        detail = self._shown
        with VerticalGroup(id="hook-run-container") as container:
            container.border_title = Text(
                self._text(
                    _TITLE.bind(number=str(detail.record.number), hook_id=sanitize_legacy_scalar(detail.record.hook_id))
                )
            )
            with TabbedContent(id="hook-run-tabs"):
                with (
                    TabPane(self._text(_INPUT.bind()), id="hook-run-input"),
                    ScrollableContainer(classes="hook-run-scroll"),
                ):
                    yield from self._widgets("input", self._input_sections())
                with TabPane(self._text(_OUTPUT.bind()), id="hook-run-output"):
                    yield ScrollableContainer(
                        *self._widgets("output", self._output_sections()),
                        id="hook-run-output-body",
                        classes="hook-run-scroll",
                    )

    def on_mount(self) -> None:
        self._show_state()
        # Like ToolDetailModal: switch tabs once they mounted, never through ``initial=``.
        # An ended run opens on what it returned; one still in flight has only its input yet.
        if self._shown.record.state == "ended":
            self.call_after_refresh(self._open_output)
        if self._shown.record.state != "ended" or self._tail is not None:
            self.set_interval(_POLL_SECONDS, self._poll)
        if self._tail is not None:
            self._read_log()

    def _open_output(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one(TabbedContent).active = "hook-run-output"

    def _text(self, reference: MessageRef) -> str:
        return render_str(widget_localizer(self), reference)

    def _fields(self, rows: list[tuple[MessageRef, Text | str]], *, id: str | None = None) -> _Fields:
        return _Fields(
            tuple(
                (self._text(label), value if isinstance(value, Text) else Text(value)) for label, value in rows if value
            ),
            id=id,
        )

    def _note(self, label: MessageRef, *, copied: bool = True) -> _Note:
        return _Note(self._text(label), copied=copied)

    @staticmethod
    def _grid(fields: _Fields) -> Table:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim", no_wrap=True)
        grid.add_column(overflow="fold")
        for label, value in fields.rows:
            grid.add_row(label, value)
        return grid

    def _widgets(self, tab: str, sections: list[_Section]) -> list[Widget]:
        widgets: list[Widget] = []
        for index, section in enumerate(sections):
            children: list[Widget] = []
            for block in section.blocks:
                match block:
                    case _Fields():
                        children.append(Static(self._grid(block), id=block.id))
                    case _Line():
                        children.append(Static(block.text))
                    case _Note():
                        children.append(Static(Text(block.text), classes="hook-run-note"))
                    case _Code():
                        children.append(
                            build_code_view(
                                block.language, block.text, dark=self.app.current_theme.dark, render_message=self._text
                            )
                        )
                    case _LogPath():
                        children.append(Static(Text(surrogate_safe_text(block.path)), classes="hook-run-note"))
                    case _LogTail():
                        children.append(Log(max_lines=_MAX_LOG_LINES, id="hook-run-log"))
            widgets.append(_CopySection(*children, title=section.name, key=f"{tab}:{index}"))
        return widgets

    def _section_text(self, section: _Section) -> str:
        """What the section shows, in plain text, but its title."""
        parts: list[str] = []
        for block in section.blocks:
            match block:
                case _Fields():
                    parts.append(_fields_plain(block))
                case _Line():
                    parts.append(block.text.plain)
                case _Note() if block.copied:
                    parts.append(block.text)
                case _Code():
                    parts.append(block.text)
                case _LogPath():
                    parts.append(surrogate_safe_text(block.path))
                case _LogTail():
                    with contextlib.suppress(NoMatches):
                        # The log keeps the empty line its last newline opens.
                        parts.append("\n".join(self.query_one("#hook-run-log", Log).lines).rstrip("\n"))
        return "\n".join(part for part in parts if part)

    def _plain(self, sections: list[_Section]) -> str:
        """The whole tab in plain text: each section starts ``## title``."""
        texts = []
        for section in sections:
            text = self._section_text(section)
            texts.append(f"## {section.name}\n{text}" if text else f"## {section.name}")
        return "\n\n".join(texts)

    def _input_sections(self) -> list[_Section]:
        return [_Section(self._text(_OVERVIEW.bind()), (self._input_fields(),)), self._payload_section()]

    def _input_fields(self) -> _Fields:
        detail = self._shown
        record = detail.record
        mode = [record.execution_mode]
        if record.detach:
            mode.append("detach")
        mode.append(f"timeout {detail.timeout_seconds:g}s")
        mode.append(f"on_error: {detail.on_error}")
        if detail.delivery == "durable":
            mode.append("delivery: durable")
        triggered = [sanitize_legacy_scalar(record.sub_agent) if record.sub_agent else self._text(_MAIN_AGENT.bind())]
        if record.tool_name:
            tool = sanitize_legacy_scalar(record.tool_name)
            triggered.append(f"{tool} ({sanitize_legacy_scalar(detail.tool_call_id)})" if detail.tool_call_id else tool)
        if record.retry:
            triggered.append(self._text(_RETRY.bind()))
        started = queued = ""
        now = time.monotonic()
        if record.started_at is not None:
            started = _wall_time(record.registered_wall + (record.started_at - record.registered_at))
            waited = record.started_at - record.registered_at
        else:
            waited = (now if record.state == "queued" else record.registered_at) - record.registered_at
        if waited >= _QUEUED_SHOWN_AFTER_SECONDS:
            queued = fmt_duration(int(waited * 1000))
        source = " · ".join(part for part in (detail.source, surrogate_safe_text(detail.source_path)) if part)
        if detail.run_type == "shell":
            command = detail.run_argv[0] if detail.run_argv else ""
        else:
            command = shlex.join(detail.run_argv)
        run_lines = [f"{detail.run_type}: {sanitize_legacy_scalar(command)}"]
        if detail.run_cwd:
            run_lines.append(f"cwd: {surrogate_safe_text(detail.run_cwd)}")
        if detail.run_env_names:
            run_lines.append(f"env: {', '.join(sanitize_legacy_scalar(name) for name in detail.run_env_names)}")
        return self._fields(
            [
                (_EVENT.bind(), record.hook_event),
                (_MODE.bind(), " · ".join(mode)),
                (_SESSION.bind(), record.session_id),
                (_TRIGGERED_BY.bind(), " · ".join(triggered)),
                (_STARTED.bind(), started),
                (_QUEUED.bind(), queued),
                (_SOURCE.bind(), source),
                (_RUN.bind(), "\n".join(run_lines)),
            ],
            id="hook-run-input-fields",
        )

    def _payload_section(self) -> _Section:
        name = self._text(_PAYLOAD.bind())
        io = self._shown.io
        if io is None:
            return _Section(name, (self._note(_DETAILS_GONE.bind()),))
        if io.payload_size > len(io.payload.encode("utf-8")):
            return _Section(
                name,
                (
                    self._note(_PAYLOAD_TRUNCATED.bind(size=_size(io.payload_size)), copied=False),
                    _Code("text", io.payload),
                ),
            )
        return _Section(name, (_Code("json", _pretty_json(io.payload)),))

    # ── output ───────────────────────────────────────────────────────

    def _result_text(self) -> Text:
        record = self._shown.record
        parts = [self._text(part) for part in run_outcome(record) or (_SUCCEEDED.bind(),)]
        if record.exit_code is not None and record.outcome != "failed":
            parts.append(self._text(_EXIT_CODE.bind(code=str(record.exit_code))))
        duration = run_duration(record, time.monotonic())
        if duration:
            parts.append(duration)
        return Text.assemble(run_icon(record, running_style=Style(color="yellow")), " ", " · ".join(parts))

    def _output_sections(self) -> list[_Section]:
        detail = self._shown
        result = self._text(_RESULT.bind())
        if detail.record.state != "ended":
            return [_Section(result, (self._note(_RUNNING.bind()),))]
        io = detail.io
        outcome: tuple[_Block, ...] = (_Line(self._result_text()),)
        if io is None:
            outcome = (*outcome, self._note(_DETAILS_GONE.bind(), copied=False))
        sections = [_Section(result, outcome)]
        if io is not None and io.result_file is not None:
            if io.result_file == "ok":
                decision: _Block = _Code("json", io.decision)
            else:
                decision = self._note((_NO_RESULT_FILE if io.result_file == "missing" else _INVALID_RESULT_FILE).bind())
            sections.append(_Section(self._text(_DECISION.bind()), (decision,)))
        sections.append(_Section(self._text(_APPLIED.bind()), tuple(self._applied_blocks())))
        if io is not None and not detail.record.detach:
            sections.append(self._stream_section(_STDOUT.bind(), io.stdout, io.stdout_dropped))
            sections.append(self._stream_section(_STDERR.bind(), io.stderr, io.stderr_dropped))
        elif io is not None and io.stderr:
            # A detached hand-off that failed says why.
            sections.append(self._stream_section(_STDERR.bind(), io.stderr, 0))
        if detail.log_path:
            path: tuple[_Block, ...] = (_LogPath(detail.log_path),)
            if detail.shared_log:
                path = (*path, self._note(_SHARED_LOG.bind(), copied=False))
            sections.append(_Section(self._text(_LOG_FILE.bind()), path))
            sections.append(_Section(self._text(_LOG.bind()), (_LogTail(),)))
        return sections

    def _applied_blocks(self) -> list[_Block]:
        detail = self._shown
        applied = detail.applied
        effect = detail.record.effect
        if applied is None:
            return [self._note(_APPLIED_NOTHING.bind())]
        widgets: list[_Block] = []
        if effect == "blocked":
            widgets.append(self._note(_APPLIED_BLOCKED.bind(reason=sanitize_legacy_scalar(applied.block_reason))))
        elif effect == "modified_args":
            widgets.append(self._note(_APPLIED_ARGS.bind()))
            widgets.append(_Code("json", applied.args))
        elif effect == "added_reminder":
            widgets.append(self._note(_APPLIED_REMINDER.bind()))
            widgets.append(_Code("text", applied.reminder))
        elif effect == "added_context":
            widgets.append(self._note(_APPLIED_CONTEXT.bind()))
            widgets.append(_Code("text", applied.extra_context))
        reason = {
            "nonblocking": _NOT_APPLIED_NONBLOCKING.bind(),
            "unsupported": _NOT_APPLIED_UNSUPPORTED.bind(event=detail.record.hook_event),
            "failed": _NOT_APPLIED_FAILED.bind(),
            "failed_blocked": _NOT_APPLIED_FAILED_BLOCKED.bind(),
            None: None,
        }[applied.not_applied]
        if reason is not None:
            widgets.append(self._note(reason))
        return widgets or [self._note(_APPLIED_NOTHING.bind())]

    def _stream_section(self, label: MessageRef, text: str, dropped: int) -> _Section:
        blocks: tuple[_Block, ...] = (_Code("text", text),)
        if dropped:
            blocks = (self._note(_TRUNCATED.bind(bytes=str(dropped)), copied=False), *blocks)
        return _Section(self._text(label), blocks)

    # ── updates ──────────────────────────────────────────────────────

    def _show_state(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#hook-run-container").border_subtitle = self._result_text()

    async def _poll(self) -> None:
        if self._tail is not None and not self._reading:
            self._read_log()
        if self._shown.record.state == "ended":
            return
        latest = self._latest(self._shown.record.run_id)
        if latest is None:
            return
        changed = latest.record != self._shown.record
        self._shown = latest
        self._show_state()
        if not changed or self._rebuilding:
            return
        with contextlib.suppress(NoMatches):
            self.query_one("#hook-run-input-fields", Static).update(self._grid(self._input_fields()))
        if latest.record.state != "ended":
            return
        # Ended: its output arrived. The tab stays where the user is.
        self._rebuilding = True
        try:
            body = self.query_one("#hook-run-output-body", ScrollableContainer)
            await remove_children_shielded(body)
            await body.mount(*self._widgets("output", self._output_sections()))
        except NoMatches:
            return
        finally:
            self._rebuilding = False
        if latest.log_path and self._tail is None:
            self._tail = LogTail(latest.log_path)
            self._read_log()

    def _read_log(self) -> None:
        self._reading = True
        self.run_worker(self._read_more(), exit_on_error=False)

    async def _read_more(self) -> None:
        tail = self._tail
        if tail is None:
            self._reading = False
            return
        try:
            text, restart = await asyncio.to_thread(tail.read)
        except OSError:
            return
        finally:
            self._reading = False
        with contextlib.suppress(NoMatches):
            log = self.query_one("#hook-run-log", Log)
            if restart:
                log.clear()
            if text:
                log.write(text, scroll_end=log.is_vertical_scroll_end)

    # ── actions ──────────────────────────────────────────────────────

    def _input_text(self) -> str:
        """Everything the Input tab shows."""
        return self._plain(self._input_sections())

    def _output_text(self) -> str:
        """Everything the Output tab shows, the log read so far included."""
        return self._plain(self._output_sections())

    @on(_CopySection.Clicked)
    def _on_copy_section(self, event: _CopySection.Clicked) -> None:
        event.stop()
        tab, _, index = event.key.partition(":")
        sections = self._input_sections() if tab == "input" else self._output_sections()
        if not index.isdigit() or int(index) >= len(sections):
            return
        section = sections[int(index)]
        if any(isinstance(block, _LogPath) for block in section.blocks):
            copied, too_large, unavailable = (
                _COPIED_LOG_PATH,
                _COPIED_LOG_PATH_UNAVAILABLE,
                _COPIED_LOG_PATH_UNAVAILABLE,
            )
            copy_and_notify(self, self._shown.log_path, copied=copied, too_large=too_large, unavailable=unavailable)
            return
        name = section.name
        copy_and_notify(
            self,
            self._section_text(section),
            copied=_COPIED_SECTION.bind(section=name),
            too_large=_COPIED_SECTION_TOO_LARGE.bind(section=name),
            unavailable=_COPIED_SECTION_UNAVAILABLE.bind(section=name),
        )

    def action_copy(self) -> None:
        """Copy everything the open tab shows."""
        on_output = False
        with contextlib.suppress(NoMatches):
            on_output = self.query_one(TabbedContent).active == "hook-run-output"
        if on_output:
            copy_and_notify(
                self,
                self._output_text(),
                copied=COPIED_OUTPUT,
                too_large=COPIED_OUTPUT_TOO_LARGE,
                unavailable=COPIED_OUTPUT_UNAVAILABLE,
            )
        else:
            copy_and_notify(
                self,
                self._input_text(),
                copied=COPIED_INPUT,
                too_large=COPIED_INPUT_TOO_LARGE,
                unavailable=COPIED_INPUT_UNAVAILABLE,
            )

    def action_close(self) -> None:
        """Close the dialog."""
        self.dismiss(None)
