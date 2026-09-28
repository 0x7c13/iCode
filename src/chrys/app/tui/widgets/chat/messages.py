# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chat message widgets and inline conversation status actions."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.text import Text
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.geometry import Offset, Region
from textual.message import Message
from textual.reactive import reactive
from textual.selection import Selection
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Button, Static

from chrys.app.tui.clipboard import copy_text_to_clipboards
from chrys.app.tui.copy_messages import COPIED_TITLE
from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.widgets.chat.image_preview import ChatImagePreview, ImagePreviewGrid, extract_image_previews
from chrys.app.tui.widgets.click_affordance import ClickAffordance
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.foundation.i18n import DisplayBlock, MessageRef, msg
from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.util.time import parse_created_at

if TYPE_CHECKING:
    from textual.app import ComposeResult


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

INTERRUPTED_ERROR_HEADER_MESSAGE = msg(
    "transcript.interrupted.error_header",
    fallback="✗ Error",
)
INTERRUPTED_WARNING_HEADER_MESSAGE = msg(
    "transcript.interrupted.warning_header",
    fallback="⚠ Interrupted",
)
INTERRUPTED_REASON_PLAIN_MESSAGE = msg(
    "transcript.interrupted.reason_plain",
    fallback="{reason}",
    multiline=True,
)
INTERRUPTED_REASON_BY_USER_MESSAGE = msg(
    "transcript.interrupted.reason_by_user",
    fallback="{reason} by user",
    multiline=True,
)
INTERRUPTED_RETRY_ACTION_MESSAGE = msg(
    "transcript.interrupted.retry_action",
    fallback="Retry",
)
INTERRUPTED_CONTINUE_ACTION_MESSAGE = msg(
    "transcript.interrupted.continue_action",
    fallback="Continue",
)
_AGENT_RESPONSE_COPIED = msg("tui.copy.agent_response", fallback="Copied agent response")
_THINK_PREFIX = msg("tui.chat.think_prefix", fallback="Think: {body}", multiline=True)
_AGENT_RESPONSE_COPY_TOOLTIP = msg(
    "tui.chat.copy_agent_response_tooltip",
    fallback="Copy agent response",
)
_AGENT_RESPONSE_COPY_BUTTON = msg("tui.chat.copy_agent_response_button", fallback="copy")
_AGENT_FALLBACK_LABEL = msg("tui.chat.agent_fallback_label", fallback="Agent")
_RETRY_MESSAGE = msg(
    "tui.chat.retry_message",
    fallback="{message} Retrying in {delay_seconds}s ({attempt}/{max_attempts})...",
    multiline=True,
)


def _format_duration_ms(duration_ms: int) -> str:
    from chrys.app.tui.widgets.chat.tool_call import fmt_duration

    return fmt_duration(duration_ms)


@dataclass(frozen=True)
class InterruptedMessageCopy:
    """Resolved, mount-stable copy for an interruption notice."""

    header: str
    reason: str
    retry_action: str
    continue_action: str


def resolve_interrupted_message_copy(
    reason: str,
    source: str,
    resolver: Callable[[MessageRef], str] = format_message,
) -> InterruptedMessageCopy:
    """Resolve interruption chrome and its closed-set source prose."""
    header_definition = INTERRUPTED_ERROR_HEADER_MESSAGE if source == "error" else INTERRUPTED_WARNING_HEADER_MESSAGE
    reason_definition = INTERRUPTED_REASON_BY_USER_MESSAGE if source == "user" else INTERRUPTED_REASON_PLAIN_MESSAGE
    return InterruptedMessageCopy(
        header=resolver(header_definition.bind()),
        reason=resolver(reason_definition.bind(reason=DisplayBlock(reason))),
        retry_action=resolver(INTERRUPTED_RETRY_ACTION_MESSAGE.bind()),
        continue_action=resolver(INTERRUPTED_CONTINUE_ACTION_MESSAGE.bind()),
    )


def format_message_created_at(value: Any) -> str:
    """Format a persisted message timestamp in the user's local time."""
    dt = parse_created_at(value)
    if dt is None:
        return ""

    if dt.tzinfo is None or dt.utcoffset() is None:
        dt = dt.replace(tzinfo=UTC)
    local = dt.astimezone()
    hour = local.hour % 12 or 12
    suffix = "AM" if local.hour < 12 else "PM"
    return f"- {hour}:{local.minute:02d} {suffix}"


def strip_think(text: str) -> str:
    """Remove ``<think>…</think>`` blocks entirely."""
    return _THINK_RE.sub("", text).strip()


def think_to_italic(
    text: str,
    render_message: Callable[[MessageRef], str] = format_message,
) -> str:
    """Convert ``<think>…</think>`` blocks to italic markdown with a *Think:* prefix."""

    def _repl(m: re.Match[str]) -> str:
        inner = m.group(1).strip()
        if not inner:
            return ""
        # Wrap each paragraph with * for italic.  A single * pair around a
        # multi-paragraph block breaks in CommonMark, so wrap per paragraph.
        parts: list[str] = []
        for para in re.split(r"\n{2,}", inner):
            stripped = para.strip()
            if stripped:
                parts.append(f"*{stripped}*")
        body = "\n\n".join(parts)
        return render_message(_THINK_PREFIX.bind(body=DisplayBlock(body)))

    return _THINK_RE.sub(_repl, text).strip()


def process_think_tags(
    text: str,
    *,
    intermediate: bool = False,
    render_message: Callable[[MessageRef], str] = format_message,
) -> str:
    """Process ``<think>`` tags: italic for intermediate messages, strip for final."""
    if not text:
        return text
    return think_to_italic(text, render_message) if intermediate else strip_think(text)


def format_chat_copy_payload(messages: list[tuple[str, str]]) -> str:
    """Return the transcript payload used by chat copy actions."""
    return "\n\n".join(f"[{name}]\n{text}" for name, text in messages)


def _user_message_text(
    text: str,
    *,
    timestamp: str,
    is_injection: bool,
) -> Text:
    """Build the selectable text portion of a user message."""
    rendered = Text()
    if is_injection:
        rendered.append("\u2514 ", style="dim")
    rendered.append("\u276f You", style="bold cyan")
    if timestamp:
        rendered.append(f" {timestamp}", style="dim")
    rendered.append("\n")
    if is_injection:
        rendered.append("  ")
    rendered.append(text)
    return rendered


def _user_message_selection(text: str, is_injection: bool, selection: Selection) -> tuple[str, str] | None:
    """Return transcript-friendly selected text from the selectable text portion."""
    include_label = selection.start is None or selection.start.y == 0

    # Selection entirely within the header line → copy just the transcript label.
    end_in_header = selection.end is not None and selection.end.y == 0
    if include_label and end_in_header:
        return "[You]", "\n"

    indent = 2 if is_injection else 0

    def to_body(offset: Offset | None) -> Offset | None:
        if offset is None:
            return None
        if offset.y == 0:
            # Any point in the header maps to the start of the body.
            return Offset(0, 0)
        return Offset(max(0, offset.x - indent), offset.y - 1)

    shifted = Selection(to_body(selection.start), to_body(selection.end))
    body = shifted.extract(text)
    if include_label:
        return f"[You]\n{body}" if body else "[You]", "\n"
    return body, "\n"


class _UserMessageText(Static):
    """Selectable text renderer for a user message."""

    DEFAULT_CSS = """
    _UserMessageText {
        height: auto;
        background: transparent;
    }
    """

    def __init__(
        self,
        text: str,
        *,
        timestamp: str,
        is_injection: bool,
    ) -> None:
        self._text = text
        self._ts = timestamp
        self._is_injection = is_injection
        super().__init__()

    def render(self) -> Text:
        return _user_message_text(
            self._text,
            timestamp=self._ts,
            is_injection=self._is_injection,
        )

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        return _user_message_selection(self._text, self._is_injection, selection)


class _UserImagePreview(Static):
    """Non-selectable image-preview renderer for a user message."""

    ALLOW_SELECT = False

    DEFAULT_CSS = """
    _UserImagePreview {
        height: auto;
        background: transparent;
    }
    """

    def __init__(self, image_previews: list[ChatImagePreview], *, is_injection: bool) -> None:
        self._image_previews = image_previews
        self._is_injection = is_injection
        self._image_preview_lines: dict[tuple[int, str], list[Text]] = {}
        super().__init__()

    def render(self) -> Group:
        max_width = self.content_size.width or self.size.width or 80
        indent = "  " if self._is_injection else ""
        return Group(Text(""), *self._render_image_preview_lines(max_width=max_width, indent=indent))

    def _render_image_preview_lines(self, *, max_width: int, indent: str) -> list[Text]:
        """Return cached image-preview render lines for the current message width."""
        key = (max_width, indent)
        lines = self._image_preview_lines.get(key)
        if lines is None:
            grid = ImagePreviewGrid(
                self._image_previews,
                max_width=max_width,
                indent=indent,
            )
            lines = grid.render_lines()
            self._image_preview_lines[key] = lines
        return lines

    def on_resize(self, _event: Any = None) -> None:
        """Refresh image previews when terminal width changes."""
        self.refresh()

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        _ = selection
        return None


class UserMessage(Widget):
    """User message block with cyan accent."""

    DEFAULT_CSS = """
    UserMessage {
        margin: 1 0;
        padding: 0 1 0 1;
        border-left: thick $tui-border-user-message $border-opacity;
        background: $boost;
        height: auto;
    }
    UserMessage.-highlighted {
        background: $accent 25%;
        border-left: thick $tui-border-warning $border-opacity;
    }
    UserMessage.-compressed {
        border-left: thick $tui-border-neutral-160 $border-opacity;
        color: $text-muted;
    }
    UserMessage.-injection {
        margin-left: 2;
        margin-top: 0;
        border-left: none;
    }
    """

    def __init__(
        self,
        text: str,
        timestamp: str = "",
        compressed: bool = False,
        is_injection: bool = False,
        contents: list[Any] | None = None,
    ) -> None:
        self._text = text
        self._ts = timestamp
        self._compressed = compressed
        self._is_injection = is_injection
        self._image_previews = extract_image_previews(contents)
        super().__init__()
        if compressed:
            self.add_class("-compressed")
        if is_injection:
            self.add_class("-injection")

    def compose(self) -> ComposeResult:
        yield _UserMessageText(
            self._text,
            timestamp=self._ts,
            is_injection=self._is_injection,
        )
        if self._image_previews:
            yield _UserImagePreview(self._image_previews, is_injection=self._is_injection)

    @property
    def text(self) -> str:
        """The message as the user wrote it."""
        return self._text

    @property
    def is_injection(self) -> bool:
        """Whether this user message is a mid-turn injection."""
        return self._is_injection

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Return transcript text when Textual records selection on the parent.

        Drag selection usually lands on ``_UserMessageText``, but Textual can
        still record the container itself for select-all and cross-widget spans.
        If both parent and child are selected, the child has the precise range,
        so let it provide the copy text and avoid duplicating the message.
        """
        if self.is_attached:
            with contextlib.suppress(NoMatches):
                if self.query_one(_UserMessageText) in self.screen.selections:
                    return None
        return _user_message_selection(self._text, self._is_injection, selection)


class _AgentHeader(Static):
    """Clickable header for AgentMessage — posts Clicked on click."""

    class Clicked(Message):
        """Posted when the header is clicked."""

    def __init__(self, label: str, content: Text) -> None:
        self._copy_label = label
        super().__init__(content, classes="agent-header")

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Copy the agent header as a transcript label, not decorative chrome."""
        return f"[{self._copy_label}]", "\n"

    def on_click(self) -> None:
        self.post_message(self.Clicked())


class AgentHeaderRow(Horizontal):
    """Header row for agent messages, including finalized-message actions."""

    ALLOW_SELECT = False

    DEFAULT_CSS = """
    AgentHeaderRow {
        width: 100%;
        height: 1;
    }
    AgentHeaderRow > _AgentHeader {
        width: auto;
        height: 1;
        padding: 0 1 0 0;
    }
    AgentHeaderRow > AgentCopyButton {
        display: none;
        width: auto;
        height: 1;
        padding: 0 0 0 1;
        color: $text-muted;
        text-style: dim not bold;
        pointer: pointer;
    }
    AgentHeaderRow > AgentCopyButton:hover {
        color: $accent;
        text-style: underline;
    }
    """

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        _ = selection
        return None


class AgentCopyButton(ClickAffordance):
    """Clickable copy affordance for finalized agent responses."""

    ALLOW_SELECT = False

    class Clicked(Message):
        """Posted when the copy affordance is clicked."""

    CLICK_MESSAGE = Clicked

    def __init__(self, tooltip: str | None = None, text: str | None = None) -> None:
        super().__init__(
            Text(text or format_message(_AGENT_RESPONSE_COPY_BUTTON.bind())),
            classes="agent-copy-btn",
        )
        self.tooltip = tooltip or format_message(_AGENT_RESPONSE_COPY_TOOLTIP.bind())

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        _ = selection
        return None


class AgentMessage(Widget):
    """Agent message with markdown rendering and streaming support.

    Uses VirtualizedMarkdown for rich rendering of agent responses including
    code blocks, tables, lists, and other markdown elements.
    Click the header to collapse/expand the message body.
    """

    DEFAULT_CSS = """
    AgentMessage {
        margin: 1 0;
        padding: 0 0 0 1;
        border-left: thick $tui-border-agent-message $border-opacity;
        height: auto;
    }
    AgentMessage.--intermediate {
        border-left: thick $tui-border-neutral-gray $border-opacity;
        color: $text-muted;
    }
    AgentMessage.--intermediate > VirtualizedMarkdown {
        color: $text-muted;
    }
    AgentMessage.--compressed {
        border-left: thick $tui-border-neutral-100 $border-opacity;
        color: $text-muted;
    }
    AgentMessage.--compressed > VirtualizedMarkdown {
        color: $text-muted;
    }
    AgentMessage.--structured-completion > VirtualizedMarkdown {
        color: $success;
    }
    AgentMessage > VirtualizedMarkdown {
        padding: 0;
        height: auto;
        min-height: 1;
        background: transparent;
    }
    AgentMessage > .agent-cursor {
        height: 1;
        padding: 0 1;
    }
    """

    collapsed: reactive[bool] = reactive(False)

    def __init__(
        self,
        text: str = "",
        is_final: bool = True,
        profile_name: str = "",
        is_intermediate: bool = False,
        is_compressed: bool = False,
        is_structured_completion: bool = False,
        timestamp: str = "",
        duration_ms: int | None = None,
    ) -> None:
        self._source_text = text
        self._processed_text = process_think_tags(text, intermediate=is_intermediate)
        self._processed_text_stale = False
        """``_source_text`` changed since ``_processed_text`` was derived; ``text`` re-derives it."""
        self._body_stale = False
        """The markdown body has not been handed the latest streamed text."""
        self._body_sync_scheduled = False
        """A callback that brings the markdown body up to date is queued on this widget."""
        self._is_final = is_final
        self._profile_name = profile_name
        self._is_intermediate = is_intermediate
        self._is_compressed = is_compressed
        self._is_structured_completion = is_structured_completion
        self._ts = timestamp
        self._duration_ms = duration_ms
        self._md_widget: VirtualizedMarkdown | None = None
        self._background_strip_cache_key: tuple[int, int, int, str, str, int] | None = None
        self._background_strip_cache: Strip | None = None
        cls = ""
        if is_intermediate:
            cls = "--intermediate"
        elif is_compressed:
            cls = "--compressed"
        elif is_structured_completion:
            cls = "--structured-completion"
        super().__init__(classes=cls)

    @property
    def text(self) -> str:
        """The response text with think blocks processed, as of the latest ``stream_update()``."""
        if self._processed_text_stale:
            self._processed_text = process_think_tags(self._source_text)
            self._processed_text_stale = False
        return self._processed_text

    def _header_text(self) -> Text:
        label = self._copy_label()
        arrow = "\u25b6" if self.collapsed else "\u25c7"
        t = Text.assemble((f"{arrow} {label}", "bold green"))
        if self._ts:
            t.append(f" {self._ts}", style="dim")
        if self._duration_ms is not None:
            t.append(f" ({_format_duration_ms(self._duration_ms)})", style="dim")
        return t

    def _copy_label(self) -> str:
        return self._profile_name or self._render_message(_AGENT_FALLBACK_LABEL.bind())

    def _render_message(self, reference: MessageRef) -> str:
        return render_str(widget_localizer(self), reference)

    def compose(self) -> ComposeResult:
        if self._is_intermediate:
            self._processed_text = process_think_tags(
                self._source_text,
                intermediate=True,
                render_message=self._render_message,
            )
            self._processed_text_stale = False
        with AgentHeaderRow():
            yield _AgentHeader(self._copy_label(), self._header_text())
            copy_button = AgentCopyButton(
                tooltip=self._render_message(_AGENT_RESPONSE_COPY_TOOLTIP.bind()),
                text=self._render_message(_AGENT_RESPONSE_COPY_BUTTON.bind()),
            )
            copy_button.display = self._should_show_copy_button()
            yield copy_button
        self._md_widget = VirtualizedMarkdown(self.text)
        # The body starts from the current text, so a queued sync has nothing to hand it.
        self._body_stale = False
        yield self._md_widget
        if not self._is_final:
            yield Static(Text(" \u258d", style="bold green"), classes="agent-cursor")

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Agent content is copied by its header/body children."""
        return None

    def render_lines(self, crop: Region) -> list[Strip]:
        """Render the repeated message chrome without per-row style work.

        AgentMessage only paints a left border, padding, and transparent
        background; child widgets render the header/body text. Every row of
        that chrome is identical, so a clipped parent scroll can reuse one
        strip instead of asking Textual's generic style renderer to rebuild
        it for each visible markdown row.
        """
        if not self.is_attached:
            return [Strip.blank(crop.width) for _ in crop.line_range]

        if crop.width <= 0 or crop.height <= 0:
            return []
        if not self._can_reuse_chrome_strip():
            return super().render_lines(crop)

        cache_key = (
            crop.x,
            crop.width,
            self.size.width,
            self.app.theme,
            str(self.app.console.color_system),
            # Private Textual cache key: this fast path is intentionally tied
            # to Textual's current style invalidation model and covered by TUI
            # render-cache tests.
            self.styles._cache_key,
        )
        cached = self._background_strip_cache
        if cached is None or cache_key != self._background_strip_cache_key:
            cached = Widget.render_lines(self, Region(crop.x, 0, crop.width, 1))[0]
            self._background_strip_cache = cached
            self._background_strip_cache_key = cache_key
        return [cached] * crop.height

    def _can_reuse_chrome_strip(self) -> bool:
        """Return true when every AgentMessage chrome row is visually identical."""
        styles = self.styles
        if styles.gutter.height or self._border_title is not None or self._border_subtitle is not None:
            return False
        if any(edge for edge, _color in styles.outline):
            return False
        if styles.has_rule("hatch") and styles.hatch != "none":
            return False
        if styles.tint.a or styles.background_tint.a:
            return False
        if styles.text_opacity != 1.0 or self.opacity != 1.0:
            return False
        if styles.line_pad:
            return False
        return styles.text_align in {"start", "left"}

    def notify_style_update(self) -> None:
        self._background_strip_cache_key = None
        self._background_strip_cache = None
        super().notify_style_update()

    def watch_collapsed(self, collapsed: bool) -> None:
        if self._md_widget is not None:
            self._md_widget.display = not collapsed
        with contextlib.suppress(Exception):
            self.query_one(_AgentHeader).update(self._header_text())

    def on__agent_header_clicked(self) -> None:
        if self._is_final:
            self.collapsed = not self.collapsed

    def set_timestamp(self, timestamp: str) -> None:
        """Update the header timestamp."""
        if not timestamp or timestamp == self._ts:
            return
        self._ts = timestamp
        with contextlib.suppress(Exception):
            self.query_one(_AgentHeader).update(self._header_text())

    def _should_show_copy_button(self) -> bool:
        """Return true when this widget represents a final raw agent response."""
        return (
            self._is_final
            and not self._is_intermediate
            and not self._is_compressed
            and not self._is_structured_completion
            and bool(self.format_agent_response_copy().strip())
        )

    def _show_copy_button_if_ready(self) -> None:
        if not self._should_show_copy_button():
            return
        with contextlib.suppress(Exception):
            self.query_one(AgentCopyButton).display = True

    def format_agent_response_copy(self) -> str:
        """Return the raw response text copied by the inline button."""
        return self.text

    def copy_agent_response(self) -> None:
        """Copy this finalized agent response to the available clipboards."""
        payload = self.format_agent_response_copy()
        if not payload.strip():
            return
        copy_text_to_clipboards(self.app, payload)
        localizer = widget_localizer(self)
        self.notify(
            render_str(localizer, _AGENT_RESPONSE_COPIED.bind()),
            title=render_str(localizer, COPIED_TITLE.bind()),
            timeout=2,
            markup=False,
        )

    def on_agent_copy_button_clicked(self, event: AgentCopyButton.Clicked) -> None:
        """Handle clicks from the finalized-response copy affordance."""
        event.stop()
        if not self._should_show_copy_button():
            return
        self.copy_agent_response()

    def stream_update(self, text: str, is_final: bool = False, *, timestamp: str = "") -> None:
        """Update the message text (for streaming).

        The main turn replays its buffered answer as a synchronous burst of
        cumulative updates, one per line. A non-final update only records the
        newest text: the markdown body catches up once per message-loop turn,
        so a burst costs one think-tag pass and one markdown update instead of
        one per line, while ``text`` always reads the newest text. A final
        update hands the body its text before the cursor and copy button change.
        """
        self._source_text = text
        self._processed_text_stale = True
        self._body_stale = True
        self._is_final = is_final
        if timestamp:
            self.set_timestamp(timestamp)
        if is_final:
            self._sync_body()
            try:
                cursor = self.query_one(".agent-cursor")
                cursor.remove()
            except Exception:
                pass
            self._show_copy_button_if_ready()
        elif not self._body_sync_scheduled:
            self._body_sync_scheduled = self.call_later(self._run_scheduled_body_sync)
            if not self._body_sync_scheduled:
                # The message pump is closing and refused the callback.
                self._sync_body()

    def _run_scheduled_body_sync(self) -> None:
        self._body_sync_scheduled = False
        self._sync_body()

    def _sync_body(self) -> None:
        """Hand the markdown body the current text if it has not seen it yet."""
        if not self._body_stale:
            return
        self._body_stale = False
        if self._md_widget is not None:
            self._md_widget.update(self.text)


class SystemMessage(Static):
    """System notification — dim, unobtrusive."""

    DEFAULT_CSS = """
    SystemMessage {
        margin: 1 0;
        padding: 0 1;
        color: $text-muted;
        height: auto;
    }
    SystemMessage.-warning {
        color: $warning;
    }
    """

    def __init__(self, text: str, *, warning: bool = False) -> None:
        super().__init__(Text(f"  ◦ {text}"))
        self.set_class(warning, "-warning")


class ConversationStatusAction:
    """Messages emitted by inline failed/interrupted conversation actions."""

    class Pressed(Message):
        """Posted when the inline retry/continue action is clicked."""


class ErrorMessage(Widget):
    """Error display with red accent."""

    DEFAULT_CSS = """
    ErrorMessage {
        margin: 1 0;
        padding: 0 1 0 2;
        border-left: thick $tui-border-status-error $border-opacity;
        height: auto;
    }
    ErrorMessage > .status-body {
        height: auto;
        color: $error;
    }
    ErrorMessage > .status-action-row {
        height: 1;
        margin: 1 0 0 0;
    }
    ErrorMessage > .status-action-row > Button {
        height: 1;
        min-width: 10;
        border: none;
        background: $error;
        color: $button-color-foreground;
        text-style: bold;
    }
    ErrorMessage > .status-action-row > Button:hover,
    ErrorMessage > .status-action-row > Button:focus,
    ErrorMessage > .status-action-row > Button.-active {
        background: $foreground;
        color: $background;
        text-style: bold;
    }
    """

    def __init__(self, text: str, *, action_label: str | None = None) -> None:
        self._text = text
        self._action_label = action_label
        super().__init__()

    def _render_text(self) -> Text:
        t = Text()
        t.append(render_str(widget_localizer(self), INTERRUPTED_ERROR_HEADER_MESSAGE.bind()), style="bold")
        t.append("\n")
        t.append(self._text)
        return t

    def compose(self) -> ComposeResult:
        yield Static(self._render_text(), classes="status-body")
        if self._action_label is not None:
            with Horizontal(classes="status-action-row"):
                yield Button(self._action_label, compact=True, classes="status-action-btn")

    def on_mount(self) -> None:
        self._size_action_button()

    def _size_action_button(self) -> None:
        if self._action_label is None:
            return
        button = self.query_one(".status-action-btn", Button)
        width = len(self._action_label) + 4
        button.styles.width = width
        button.styles.min_width = width

    def hide_action(self) -> None:
        """Hide the inline retry action after it has been used."""
        for row in self.query(".status-action-row"):
            row.display = False

    def set_action_disabled(self, disabled: bool) -> None:
        """Prevent repeat clicks while the retry is being admitted."""
        for button in self.query(".status-action-btn").results(Button):
            button.disabled = disabled

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not event.button.has_class("status-action-btn"):
            return
        event.stop()
        if event.button.disabled:
            return
        self.post_message(ConversationStatusAction.Pressed())


class RetryMessage(Static):
    """Transient retry notice — shown while the executor waits before retrying."""

    DEFAULT_CSS = """
    RetryMessage {
        margin: 1 0;
        padding: 0 1 0 2;
        border-left: thick $tui-border-status-error $border-opacity;
        height: auto;
        color: $error;
    }
    """

    def __init__(self, message: str, attempt: int, max_attempts: int, delay_seconds: int) -> None:
        self._message = message
        self._attempt = attempt
        self._max_attempts = max_attempts
        self._delay = delay_seconds
        super().__init__()

    def render(self) -> Text:
        t = Text()
        t.append(self._render_message(INTERRUPTED_ERROR_HEADER_MESSAGE.bind()), style="bold")
        t.append("\n")
        t.append(
            self._render_message(
                _RETRY_MESSAGE.bind(
                    message=DisplayBlock(self._message),
                    delay_seconds=self._delay,
                    attempt=self._attempt,
                    max_attempts=self._max_attempts,
                )
            ),
        )
        return t

    def _render_message(self, reference: MessageRef) -> str:
        return render_str(widget_localizer(self), reference)


class InterruptedMessage(Widget):
    """Interruption notice — yellow for user interrupts, red for errors."""

    DEFAULT_CSS = """
    InterruptedMessage {
        margin: 1 0;
        padding: 0 1 0 2;
        border-left: thick $tui-border-status-warning $border-opacity;
        height: auto;
    }
    InterruptedMessage.-error {
        border-left: thick $tui-border-status-error $border-opacity;
    }
    InterruptedMessage.-error > .status-action-row > Button {
        background: $error;
    }
    InterruptedMessage > .status-body {
        height: auto;
        color: $warning;
    }
    InterruptedMessage.-error > .status-body {
        color: $error;
    }
    InterruptedMessage > .status-action-row {
        height: 1;
        margin: 1 0 0 0;
    }
    InterruptedMessage > .status-action-row > Button {
        height: 1;
        min-width: 10;
        border: none;
        background: $warning;
        color: $button-color-foreground;
        text-style: bold;
    }
    InterruptedMessage > .status-action-row > Button:hover,
    InterruptedMessage > .status-action-row > Button:focus,
    InterruptedMessage > .status-action-row > Button.-active {
        background: $foreground;
        color: $background;
        text-style: bold;
    }
    """

    def __init__(
        self,
        reason: str = "Execution interrupted",
        source: str = "user",
        *,
        action_label: str | None = None,
        header: str | None = None,
    ) -> None:
        if header is None:
            copy = resolve_interrupted_message_copy(reason, source)
            header = copy.header
            reason = copy.reason
        self._header = header
        self._reason = reason
        self._source = source
        self._action_label = action_label
        super().__init__()
        if source == "error":
            self.add_class("-error")

    def _render_text(self) -> Text:
        t = Text()
        t.append(self._header, style="bold")
        t.append("\n")
        t.append(self._reason)
        return t

    def compose(self) -> ComposeResult:
        yield Static(self._render_text(), classes="status-body")
        if self._action_label is not None:
            with Horizontal(classes="status-action-row"):
                yield Button(self._action_label, compact=True, classes="status-action-btn")

    def on_mount(self) -> None:
        self._size_action_button()

    def _size_action_button(self) -> None:
        if self._action_label is None:
            return
        button = self.query_one(".status-action-btn", Button)
        width = Text(self._action_label).cell_len + 4
        button.styles.width = width
        button.styles.min_width = width

    def hide_action(self) -> None:
        """Hide the inline retry/continue action after it has been used."""
        for row in self.query(".status-action-row"):
            row.display = False

    def set_action_disabled(self, disabled: bool) -> None:
        """Prevent repeat clicks while the retry is being admitted."""
        for button in self.query(".status-action-btn").results(Button):
            button.disabled = disabled

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not event.button.has_class("status-action-btn"):
            return
        event.stop()
        if event.button.disabled:
            return
        self.post_message(ConversationStatusAction.Pressed())


class ThinkingIndicator(Widget):
    """Loading indicator with label text."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        margin: 1 0;
        padding: 0 1;
        height: 1;
        layout: horizontal;
    }
    ThinkingIndicator > LoadingIndicator {
        width: 5;
        height: 1;
        color: $accent;
    }
    ThinkingIndicator > .thinking-label {
        width: auto;
        height: 1;
        color: $accent;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        from textual.widgets import LoadingIndicator

        yield LoadingIndicator()
        yield Static("thinking", classes="thinking-label", id="thinking-label")

    def update_label(self, label: str) -> None:
        """Change the displayed text."""
        import contextlib

        with contextlib.suppress(Exception):
            self.query_one("#thinking-label", Static).update(Text(label))
