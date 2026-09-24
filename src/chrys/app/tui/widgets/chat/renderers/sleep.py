# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Renderer for the ``sleep`` tool."""

from __future__ import annotations

import json
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Button, Static

from chrys.app.tui.i18n import render_str, render_text, widget_localizer
from chrys.app.tui.widgets.chat.tool_call import (
    TOOL_CARD_COMPLETED,
    TOOL_CARD_ERRORED,
    TOOL_CARD_INTERRUPTED,
    TOOL_CARD_REJECTED,
    TOOL_CARD_SKIPPED,
    BaseToolCard,
    ToolCardHeader,
    fmt_duration,
    tool_result_render_status,
)
from chrys.foundation.i18n import msg
from chrys.foundation.i18n.formatting import sanitize_legacy_scalar
from chrys.foundation.tool_kinds import KIND_SLEEP
from chrys.service.tools.builtins.sleep import MAX_SLEEP_SECONDS, format_sleep_seconds, normalize_sleep_seconds

if TYPE_CHECKING:
    from textual.app import ComposeResult

_SLEEP_TITLE = msg("tui.tool_card.sleep.title", fallback="Sleep")
_SLEEP_SLEEPING = msg("tui.tool_card.sleep.sleeping", fallback="sleeping")
_SLEEP_INVALID_DURATION = msg(
    "tui.tool_card.sleep.invalid_duration",
    fallback="Invalid sleep duration",
)
_SLEEP_REMAINING = msg("tui.tool_card.sleep.remaining", fallback="Remaining: {duration}")
_SLEEP_SKIP = msg("tui.tool_card.sleep.skip", fallback="Skip")


class SleepSkipClicked(Message):
    """User clicked Skip on a running sleep card."""

    def __init__(self, call_id: str) -> None:
        super().__init__()
        self.call_id = call_id


class SleepToolCall(BaseToolCard):
    """Renderer for short, skippable sleep tool calls."""

    DEFAULT_CSS = """
    SleepToolCall {
        height: auto;
        padding: 0 0 0 2;
        margin: 0 0 1 0;
    }
    SleepToolCall #sleep-label {
        height: auto;
    }
    SleepToolCall > #sleep-panel {
        height: auto;
        margin: 0 0 0 2;
        border: round $tui-border-warning 50%;
        border-title-color: $warning;
        border-title-style: bold;
        border-subtitle-align: right;
        border-subtitle-color: $warning;
        border-subtitle-style: bold;
        padding: 0 1;
    }
    SleepToolCall.-success > #sleep-panel {
        border: round $tui-border-success 30%;
        border-title-color: $success;
        border-title-style: not bold;
        border-subtitle-color: $success;
        border-subtitle-style: not bold;
    }
    SleepToolCall.-error > #sleep-panel {
        border: round $tui-border-error 50%;
        border-title-color: $text-error;
        border-title-style: not bold;
        border-subtitle-color: $text-error;
        border-subtitle-style: not bold;
    }
    SleepToolCall #sleep-reason {
        height: auto;
        color: $text-muted;
    }
    SleepToolCall #sleep-divider {
        height: 1;
        border-top: solid $tui-border-warning 35%;
    }
    SleepToolCall #sleep-countdown {
        height: 1;
        margin: 0 0 0 2;
        color: $warning;
    }
    SleepToolCall #sleep-result {
        height: auto;
        color: $text-muted;
        display: none;
    }
    SleepToolCall #sleep-actions {
        height: 1;
        margin: 0;
    }
    SleepToolCall #sleep-actions Button {
        min-width: 8;
        height: 1;
        border: none;
        background: $warning;
        color: $text;
        text-style: bold;
    }
    SleepToolCall.-done #sleep-countdown,
    SleepToolCall.-done #sleep-divider,
    SleepToolCall.-done #sleep-actions {
        display: none;
    }
    SleepToolCall.-done #sleep-result {
        display: block;
    }
    """

    _SPINNERS: ClassVar[str] = "|/-\\"

    def __init__(
        self,
        call_id: str,
        tool_name: str,
        args_summary: str = "",
        args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(call_id, tool_name, args_summary, args=args or self._args_from_summary(args_summary))
        self._start_time = time.monotonic()
        self._seconds = normalize_sleep_seconds(self.args.get("seconds"))
        self._reason = str(self.args.get("reason") or "")
        self._spin_idx = 0
        self._timer: Timer | None = None

    def _label_text(self, duration_ms: int = 0) -> Text:
        t = Text()
        t.append("- ", style="bold")
        t.append(self.tool_name, style="bold")
        if duration_ms:
            t.append(f" ({fmt_duration(duration_ms)})", style="dim")
        return t

    def _running_label_text(self) -> Text:
        t = Text()
        t.append("- ", style="bold")
        t.append(self.tool_name, style="bold")
        seconds = self._seconds
        if seconds is not None:
            elapsed = int(time.monotonic() - self._start_time)
            t.append(f" ({format_sleep_seconds(min(elapsed, seconds))}/{format_sleep_seconds(seconds)})", style="dim")
        return t

    def compose(self) -> ComposeResult:
        yield ToolCardHeader(self._running_label_text(), id="sleep-label")
        with Vertical(id="sleep-panel") as panel:
            panel.border_title = render_text(widget_localizer(self), _SLEEP_TITLE.bind())
            panel.border_subtitle = self._spinner_title()
            if self._reason:
                yield Static(Text(sanitize_legacy_scalar(self._reason)), id="sleep-reason")
                yield Static("", id="sleep-divider")
            yield Static("", id="sleep-result")
            with Horizontal(id="sleep-actions"):
                yield Button(render_text(widget_localizer(self), _SLEEP_SKIP.bind()), id="sleep-skip-btn", compact=True)
                yield Static(self._countdown_text(), id="sleep-countdown")

    @staticmethod
    def _args_from_summary(args_summary: str) -> dict[str, Any]:
        if not args_summary:
            return {}
        with suppress(json.JSONDecodeError, TypeError, ValueError):
            parsed = json.loads(args_summary)
            if isinstance(parsed, dict):
                return parsed
        return {}

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.2, self._tick)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()

    def _tick(self) -> None:
        if self.status != "running":
            return
        self._spin_idx = (self._spin_idx + 1) % len(self._SPINNERS)
        with suppress(Exception):
            self.query_one("#sleep-panel").border_subtitle = self._spinner_title()
            self.query_one("#sleep-label", Static).update(self._running_label_text())
            self.query_one("#sleep-countdown", Static).update(self._countdown_text())

    def _spinner_title(self) -> Text:
        t = Text()
        t.append(f"{self._SPINNERS[self._spin_idx]} ", style="yellow")
        t.append(render_str(widget_localizer(self), _SLEEP_SLEEPING.bind()), style="yellow")
        return t

    def _countdown_text(self) -> Text:
        seconds = self._seconds
        if seconds is None or seconds > MAX_SLEEP_SECONDS or seconds < 0:
            return Text(render_str(widget_localizer(self), _SLEEP_INVALID_DURATION.bind()), style="red")
        elapsed = time.monotonic() - self._start_time
        remaining = max(0, seconds - int(elapsed))
        return render_text(
            widget_localizer(self),
            _SLEEP_REMAINING.bind(duration=format_sleep_seconds(remaining)),
        )

    def set_complete(self, result: str, duration_ms: int = 0, **kwargs: Any) -> None:
        """Mark the sleep as complete or skipped."""
        self.result_text = result
        self.duration_ms = duration_ms
        self.approval = kwargs.get("approval")
        metadata = kwargs.get("metadata")
        self.metadata = metadata if isinstance(metadata, dict) else {}
        if self._timer is not None:
            self._timer.stop()

        render_status = tool_result_render_status(result, self.approval, self.tool_kind or KIND_SLEEP, self.metadata)
        self.status = render_status
        if render_status == "rejected":
            self.add_class("-rejected")
        if render_status == "error":
            self.add_class("-error")

        with suppress(Exception):
            self.query_one("#sleep-label", Static).update(self._label_text(duration_ms))
            panel = self.query_one("#sleep-panel")
            sleep_skipped = self.metadata.get("sleep_skipped") is True
            sleep_interrupted = self.metadata.get("sleep_interrupted") is True
            if render_status == "rejected":
                panel.border_subtitle = render_text(widget_localizer(self), TOOL_CARD_REJECTED.bind())
            elif render_status == "error":
                panel.border_subtitle = render_text(widget_localizer(self), TOOL_CARD_ERRORED.bind())
            elif sleep_interrupted:
                panel.border_subtitle = render_text(widget_localizer(self), TOOL_CARD_INTERRUPTED.bind())
                self.add_class("-success")
            elif sleep_skipped:
                panel.border_subtitle = render_text(widget_localizer(self), TOOL_CARD_SKIPPED.bind())
                self.add_class("-success")
            else:
                panel.border_subtitle = render_text(widget_localizer(self), TOOL_CARD_COMPLETED.bind())
                self.add_class("-success")
            self.query_one("#sleep-result", Static).update(Text(result))
        self.add_class("-done")
        self._show_tool_copy_button()

    def set_error(self, error: str) -> None:
        """Mark the sleep as failed."""
        self.result_text = error
        self.status = "error"
        if self._timer is not None:
            self._timer.stop()
        self.add_class("-error")
        with suppress(Exception):
            self.query_one("#sleep-panel").border_subtitle = render_text(
                widget_localizer(self),
                TOOL_CARD_ERRORED.bind(),
            )
            self.query_one("#sleep-result", Static).update(Text(error))
        self.add_class("-done")
        self._show_tool_copy_button()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Translate Skip clicks into a bubbled message."""
        if event.button.id == "sleep-skip-btn" and self.status == "running":
            self.post_message(SleepSkipClicked(self.call_id))
            event.stop()
