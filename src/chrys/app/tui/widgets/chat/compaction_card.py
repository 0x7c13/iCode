# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""CompactionCard — live Phase-4 compaction status card for the main agent."""

from __future__ import annotations

import time
from contextlib import suppress
from typing import TYPE_CHECKING

from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from chrys.app.tui.clipboard import OSC52_COPY_MAX_BYTES, copy_text_to_clipboards
from chrys.app.tui.copy_messages import COPIED_TITLE
from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.widgets.chat.tool_call import (
    TOOL_COPY_EXCLUDED_CLASS,
    ToolCopyButton,
    ToolCopyExcludedMixin,
    fmt_duration,
)
from chrys.app.tui.widgets.click_affordance import ClickAffordance
from chrys.app.tui.widgets.loading import ChrysLoadingIndicator
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.foundation.i18n import DisplayBlock, msg

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.timer import Timer


_COMPACTION_SUMMARY_COPIED = msg(
    "tui.compaction.copy.summary",
    fallback="Copied compaction summary",
)
_COMPACTION_SUMMARY_COPIED_UNAVAILABLE = msg(
    "tui.compaction.copy.summary_unavailable",
    fallback="Copied compaction summary (terminal clipboard unavailable)",
)
_COMPACTION_COPY_SUMMARY_TOOLTIP = msg(
    "tui.compaction.copy.summary_tooltip",
    fallback="Copy summary markdown",
)
_COMPACTION_SUMMARY_TITLE = msg("tui.compaction.summary_title", fallback="Summary")
_COMPACTION_LABEL_RUNNING = msg("tui.compaction.label_running", fallback="Compacting conversation...")
_COMPACTION_LABEL_SUMMARIZED = msg("tui.compaction.label_summarized", fallback="Conversation summarized")
_COMPACTION_LABEL_INTERRUPTED = msg("tui.compaction.label_interrupted", fallback="Compaction interrupted")
_COMPACTION_LABEL_FAILED = msg("tui.compaction.label_failed", fallback="Compaction failed")
_COMPACTION_LABEL_FAILED_REASON = msg(
    "tui.compaction.label_failed_reason",
    fallback="Compaction failed ({reason})",
    multiline=True,
)
_COMPACTION_RETRY_NOTICE = msg(
    "tui.compaction.retry_notice",
    fallback="⚠ {message} — retrying in {delay_seconds}s ({attempt}/{max_attempts})",
    multiline=True,
)


class CompactionCardTitle(ToolCopyExcludedMixin, ClickAffordance):
    """Clickable compaction card header — posts Clicked message on click."""

    class Clicked(Message):
        """Posted when the header is clicked."""

    CLICK_MESSAGE = Clicked


class CompactionCard(Vertical):
    """Live status card for one Phase-4 "compact the conversation" pass.

    Mounted inline in the transcript's tool flow when the backend publishes
    ``CompactionStarted``: a loading indicator plus a "Compacting
    conversation..." label with a live elapsed timer.  On
    ``CompactionFinished`` the header flips to "Conversation summarized"
    (or a failed/interrupted variant) and, when the LAST_WORDS note text is
    available, a bordered unlimited-height markdown panel with the note is
    attached below (collapsed by default) — mirroring the sub-agent result
    rendering.  Like a ``ToolGroup``, the header acts as the expand/collapse
    toggle, and a trailing ``copy`` affordance copies the raw note markdown.
    """

    DEFAULT_CSS = """
    CompactionCard {
        height: auto;
        padding: 0 0 0 2;
        margin: 0 0 1 0;
    }
    CompactionCard > #compaction-header {
        height: 1;
    }
    CompactionCard > #compaction-header > ChrysLoadingIndicator {
        width: 6;
        height: 1;
        margin: 0 1 0 0;
        color: $accent;
    }
    CompactionCard > #compaction-header > #compaction-label {
        width: auto;
        height: 1;
        color: $accent;
    }
    CompactionCard.-finished > #compaction-header > #compaction-label {
        color: $text-muted;
    }
    CompactionCard > #compaction-header > ToolCopyButton {
        display: none;
        width: auto;
        height: 1;
        margin: 0 0 0 2;
        color: $text-muted;
        text-style: dim;
        pointer: pointer;
    }
    CompactionCard > #compaction-header > ToolCopyButton:hover {
        color: $accent;
        text-style: underline not dim;
    }
    CompactionCard > #compaction-retry-notice {
        display: none;
        height: auto;
        color: $warning;
        margin: 0 0 0 2;
    }
    CompactionCard.-done > #compaction-header > ToolCopyButton {
        display: block;
    }
    CompactionCard > #compaction-note-panel {
        display: none;
        height: auto;
        margin: 0 0 0 2;
        border: round $tui-border-success 30%;
        border-title-color: $success;
        border-title-style: not bold;
        padding: 0 1;
    }
    CompactionCard.-done > #compaction-note-panel {
        display: block;
    }
    CompactionCard.-done.-collapsed > #compaction-note-panel {
        display: none;
    }
    CompactionCard > #compaction-note-panel > #compaction-note {
        height: auto;
    }
    """

    collapsed: reactive[bool] = reactive(False)

    # LLM side calls fail transiently often but usually recover by the
    # second or third attempt — the first few retry notices stay hidden so
    # a self-healing compaction never alarms the user.
    QUIET_RETRY_NOTICES = 3

    def __init__(self, compaction_id: str) -> None:
        super().__init__()
        self.compaction_id = compaction_id
        self.status = "running"
        self._start_time = time.monotonic()
        self._duration_ms = 0
        self._has_note = False
        self._last_words = ""
        self._failure_reason = ""
        self._retry_notice_count = 0
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        # The whole header row is chrome (spinner glyphs, title, copy
        # affordance) — excluded from drag-selection copy; only the note
        # panel below carries selectable prose.
        header = Horizontal(id="compaction-header")
        header.add_class(TOOL_COPY_EXCLUDED_CLASS)
        with header:
            yield ChrysLoadingIndicator()
            yield CompactionCardTitle(self._running_label_text(), id="compaction-label")
            yield ToolCopyButton(tooltip_message=_COMPACTION_COPY_SUMMARY_TOOLTIP.bind())
        yield Static("", id="compaction-retry-notice")
        with Vertical(id="compaction-note-panel") as panel:
            panel.border_title = render_str(widget_localizer(self), _COMPACTION_SUMMARY_TITLE.bind())
            yield VirtualizedMarkdown("", id="compaction-note")

    def on_mount(self) -> None:
        self._timer = self.set_interval(1.0, self._tick)

    def on_unmount(self) -> None:
        self._stop_timer()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        if self.status != "running":
            return
        with suppress(Exception):
            self.query_one("#compaction-label", Static).update(self._running_label_text())

    def _running_label_text(self) -> Content:
        # Accent title (widget CSS) with a dim elapsed trail — matching the
        # terminal labels below and the StatusBar trail's visual tone.
        # ``$text-muted`` ("auto 60%") resolves against the card surface and
        # renders visibly brighter than the StatusBar's CSS-colored trail.
        # Localized text enters only via ``$``-substitution, which keeps it
        # literal — the markup skeletons stay prose-free.
        label = render_str(widget_localizer(self), _COMPACTION_LABEL_RUNNING.bind())
        elapsed_ms = int((time.monotonic() - self._start_time) * 1000)
        if elapsed_ms >= 1000:
            return Content.from_markup(
                "[b]$label[/b] [dim]($elapsed)[/]",
                label=label,
                elapsed=fmt_duration(elapsed_ms),
            )
        return Content.from_markup("[b]$label[/b]", label=label)

    def _final_label_text(self) -> Content:
        # Terminal tones come from the theme ($warning/$error), not literal
        # colors — Rich styles cannot resolve Textual variables, so the label
        # goes through Content markup like the running label above.
        localizer = widget_localizer(self)
        elapsed = f" [dim]({fmt_duration(self._duration_ms)})[/]" if self._duration_ms else ""
        if self.status == "ok":
            prefix = ("▶ " if self.collapsed else "▼ ") if self._has_note else "• "
            label = render_str(localizer, _COMPACTION_LABEL_SUMMARIZED.bind())
            return Content.from_markup(f"[b]{prefix}$label[/b] [$success]✓[/]{elapsed}", label=label)
        if self.status == "canceled":
            label = render_str(localizer, _COMPACTION_LABEL_INTERRUPTED.bind())
            return Content.from_markup(f"[$warning]✗ $label[/]{elapsed}", label=label)
        if self._failure_reason:
            # A safety-limit trip reason replaces the (meaningless fast-fail)
            # duration — "Compaction failed (100 attempts limit exceeded for
            # current turn)".  DisplayBlock keeps the reason text literal.
            label = render_str(
                localizer,
                _COMPACTION_LABEL_FAILED_REASON.bind(reason=DisplayBlock(self._failure_reason)),
            )
            return Content.from_markup("[$error]✗ $label[/]", label=label)
        label = render_str(localizer, _COMPACTION_LABEL_FAILED.bind())
        return Content.from_markup(f"[$error]✗ $label[/]{elapsed}", label=label)

    def _refresh_final_label(self) -> None:
        with suppress(Exception):
            self.query_one("#compaction-label", Static).update(self._final_label_text())

    def watch_collapsed(self, collapsed: bool) -> None:
        self.set_class(collapsed, "-collapsed")
        if self.status != "running":
            self._refresh_final_label()

    def on_compaction_card_title_clicked(self) -> None:
        """Toggle the summary panel, mirroring the ToolGroup header."""
        if not self._has_note:
            return
        self.collapsed = not self.collapsed

    def on_tool_copy_button_clicked(self, event: ToolCopyButton.Clicked) -> None:
        """Copy the raw LAST_WORDS markdown to the clipboards."""
        event.stop()
        if not self._last_words:
            return
        localizer = widget_localizer(self)
        if copy_text_to_clipboards(self.app, self._last_words, max_terminal_bytes=OSC52_COPY_MAX_BYTES):
            self.notify(
                render_str(localizer, _COMPACTION_SUMMARY_COPIED.bind()),
                title=render_str(localizer, COPIED_TITLE.bind()),
                timeout=2,
                markup=False,
            )
        else:
            self.notify(
                render_str(localizer, _COMPACTION_SUMMARY_COPIED_UNAVAILABLE.bind()),
                title=render_str(localizer, COPIED_TITLE.bind()),
                timeout=3,
                markup=False,
            )

    def show_retry_notice(self, message: str, attempt: int, max_attempts: int, delay_seconds: int) -> None:
        """Show a side-call retry as a quiet warning line inside the card.

        The first ``QUIET_RETRY_NOTICES`` notices are counted but not
        displayed; from the next one on, the latest notice replaces the
        line.  Rendered in the warning tone — the generation is still
        auto-retrying, so this is not an error state.
        """
        if self.status != "running":
            return
        self._retry_notice_count += 1
        if self._retry_notice_count <= self.QUIET_RETRY_NOTICES:
            return
        with suppress(Exception):
            notice = self.query_one("#compaction-retry-notice", Static)
            # Content keeps provider-derived error text literal — square
            # brackets in it must not be parsed as Textual markup.
            notice.update(
                Content(
                    render_str(
                        widget_localizer(self),
                        _COMPACTION_RETRY_NOTICE.bind(
                            message=DisplayBlock(message),
                            delay_seconds=delay_seconds,
                            attempt=attempt,
                            max_attempts=max_attempts,
                        ),
                    )
                )
            )
            notice.display = True

    def set_complete(
        self,
        *,
        outcome: str,
        duration_ms: int = 0,
        last_words: str = "",
        failure_reason: str = "",
    ) -> None:
        """Finalize the card; reveal the note panel when a note exists.

        Idempotent — the first terminal outcome wins, so a dangling-card
        sweep on run end cannot overwrite a real ``CompactionFinished``.
        """
        if self.status != "running":
            return
        self.status = outcome or "failed"
        self._duration_ms = duration_ms
        self._failure_reason = failure_reason if self.status == "failed" else ""
        self._stop_timer()
        # Drop the live accent styling: finished cards return to the muted
        # transcript tone (ok) or their inline yellow/red terminal styles.
        self.add_class("-finished")
        # Any terminal state supersedes a lingering retry notice: success
        # proves recovery, and failure/cancel carry their own label.
        with suppress(Exception):
            self.query_one("#compaction-retry-notice", Static).display = False
        if self.status == "ok" and last_words:
            self._has_note = True
            self._last_words = last_words
            with suppress(Exception):
                self.query_one("#compaction-note", VirtualizedMarkdown).update(last_words)
                self.add_class("-done")
            # Collapsed by default — the summary only unfolds on an explicit
            # header click, so a mid-run compaction doesn't flood the
            # transcript with a page of markdown.
            self.collapsed = True
        with suppress(Exception):
            self.query_one(ChrysLoadingIndicator).display = False
        self._refresh_final_label()
