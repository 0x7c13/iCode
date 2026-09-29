# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Renderers for ``edit_file`` and ``write_file`` tools.

Before/after file content is provided by the mutation tracking system
via ``file_snapshot``. Large snapshots may be disk-backed refs and
are resolved only when the inline diff/copy/detail view needs them.

edit_file: split DiffView (side-by-side before/after).
write_file: unified DiffView (new file content).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from contextlib import suppress
from difflib import unified_diff
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Static

import chrys.app.tui.widgets.file_preview as _file_preview
from chrys.app.tui.i18n import render_str, render_text, widget_localizer
from chrys.app.tui.support.gc_freeze import GcAbsorbReason, GcAbsorbRequested
from chrys.app.tui.util.static_update import update_static_in_place
from chrys.app.tui.util.visibility import is_widget_shown_on_active_screen
from chrys.app.tui.widgets.chat.file_snapshot import FileSnapshotPayload, FileSnapshotRef
from chrys.app.tui.widgets.chat.tool_call import (
    TOOL_CARD_REJECTED,
    BaseToolCard,
    ToolCardHeader,
    fmt_duration,
    tool_result_render_status,
)
from chrys.app.tui.widgets.chat.tool_view_builders import ToolViewDiff, build_output_views
from chrys.foundation.i18n import msg
from chrys.foundation.tool_kinds import KIND_FILESYSTEM_WRITE

if TYPE_CHECKING:
    from textual.app import ComposeResult

_FILE_EDIT_WRITING = msg("tui.tool_card.file_edit.writing", fallback="writing")
_FILE_EDIT_EDITING = msg("tui.tool_card.file_edit.editing", fallback="editing")
_FILE_EDIT_ERROR = msg("tui.tool_card.file_edit.status.error", fallback="error")
_FILE_EDIT_CHANGES = msg("tui.tool_card.file_edit.status.changes", fallback="changes")
_FILE_EDIT_CHANGE_COUNTS = msg(
    "tui.tool_card.file_edit.status.change_counts",
    fallback="+{additions}, -{removals}",
)
_FILE_EDIT_LINES_WRITTEN = msg(
    "tui.tool_card.file_edit.status.lines_written",
    fallback="{lines} lines written",
)
_FILE_EDIT_OVERWRITE = msg("tui.tool_card.file_edit.overwrite", fallback="(overwrite)")
_FILE_EDIT_DIFF = msg("tui.tool_card.file_edit.section.diff", fallback="Diff")
_FILE_EDIT_DIFF_PREVIEW = msg(
    "tui.tool_card.file_edit.section.diff_preview",
    fallback="Diff Preview",
)
_FILE_EDIT_FULL_DIFF_OMITTED = msg(
    "tui.tool_card.file_edit.full_diff_omitted",
    fallback=(
        "@@ Full diff omitted: snapshot is {total_chars} chars (limit {limit_chars}); "
        "showing bounded diff around first change @@"
    ),
)
_FILE_EDIT_BOUNDED_PREVIEW_UNCHANGED = msg(
    "tui.tool_card.file_edit.bounded_preview_unchanged",
    fallback="@@ Bounded preview contains no changed lines @@",
)


_WRITE_FILE_LINE_COUNT_SCAN_CHARS = 64 * 1024
_FILE_COPY_FULL_DIFF_MAX_CHARS = 64 * 1024
_FILE_COPY_PREVIEW_LINES = 12
_FILE_COPY_CONTEXT_LINES = 3
_FILE_COPY_DIFF_SCAN_CHUNK_CHARS = 4096
_EDIT_FILE_INLINE_MAX_DISPLAY_LINES = 24
_INLINE_DIFF_MOUNT_DELAY_SECONDS = 0.2
_WRITE_FILE_RESULT_LINES_RE = re.compile(r"\((\d+) lines?\)")

logger = logging.getLogger(__name__)


def _write_file_line_count(result: str, after: str) -> str:
    """Return the written line count without materializing all lines."""
    if match := _WRITE_FILE_RESULT_LINES_RE.search(result):
        return match.group(1)
    if not after:
        return "0"
    scan_end = min(len(after), _WRITE_FILE_LINE_COUNT_SCAN_CHARS)
    newlines = after.count("\n", 0, scan_end)
    if scan_end == len(after):
        count = newlines + (0 if after.endswith("\n") else 1)
        return str(count)
    count = newlines if after[scan_end - 1] == "\n" else newlines + 1
    return f"{count}+"


def _line_ending_sensitive_change(before: str, after: str, *, changed_index: int | None = None) -> bool:
    """Return true when content lines match but terminators differ."""
    if changed_index is not None:
        before_char = before[changed_index] if changed_index < len(before) else ""
        after_char = after[changed_index] if changed_index < len(after) else ""
        return before_char in {"\r", "\n", ""} and after_char in {"\r", "\n", ""}
    if not before or not after:
        return False
    return before != after and before.splitlines() == after.splitlines()


def _split_line_ending(line: str) -> tuple[str, str]:
    """Return ``(body, label)`` for a possibly terminated line chunk."""
    if line.endswith("\r\n"):
        return line[:-2], "CRLF"
    if line.endswith("\n"):
        return line[:-1], "LF"
    if line.endswith("\r"):
        return line[:-1], "CR"
    return line, "none"


def _format_eol_marker(label: str) -> str:
    """Return a visible line-ending marker for copied diffs."""
    return " [no newline]" if label == "none" else f" [EOL: {label}]"


def _copy_diff_source_lines(text: str, *, show_eol_markers: bool) -> list[str]:
    """Return source lines for copy-only diffs, optionally exposing EOLs."""
    if not show_eol_markers:
        return text.splitlines()
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        body, eol = _split_line_ending(line)
        lines.append(_file_preview.sanitize_preview_line(body) + _format_eol_marker(eol))
    return lines


def _first_mismatch_index(before: str, after: str) -> int | None:
    """Return the first differing character index without splitting snapshots."""
    common_len = min(len(before), len(after))
    pos = 0
    while pos < common_len:
        end = min(pos + _FILE_COPY_DIFF_SCAN_CHUNK_CHARS, common_len)
        if before[pos:end] == after[pos:end]:
            pos = end
            continue
        for idx in range(pos, end):
            if before[idx] != after[idx]:
                return idx
    if len(before) != len(after):
        return common_len
    return None


def _line_start_with_context(text: str, index: int, context_lines: int) -> int:
    """Return the line start before *index*, including bounded context."""
    if not text:
        return 0
    index = min(max(index, 0), len(text))
    start = text.rfind("\n", 0, index) + 1
    for _ in range(context_lines):
        if start == 0:
            break
        prev_newline = text.rfind("\n", 0, start - 1)
        start = 0 if prev_newline < 0 else prev_newline + 1
    return start


def _sanitize_bounded_diff_line(line: str, changed_offset: int | None, max_chars: int) -> str:
    """Return one terminal-safe bounded line for diff previews."""
    if len(line) <= max_chars:
        return _file_preview.sanitize_preview_line(line)

    if changed_offset is None:
        return _file_preview.sanitize_preview_line(line[:max_chars]) + "..."

    changed_offset = min(max(changed_offset, 0), len(line))
    budget = max(1, max_chars - 6)
    start = max(0, changed_offset - (budget // 2))
    end = min(len(line), start + budget)
    start = max(0, end - budget)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(line) else ""
    return prefix + _file_preview.sanitize_preview_line(line[start:end]) + suffix


def _bounded_changed_window(
    text: str,
    changed_index: int,
    *,
    max_lines: int,
    max_line_chars: int,
    show_eol_markers: bool = False,
) -> list[str]:
    """Return bounded lines around the actual changed region."""
    if not text or max_lines <= 0 or max_line_chars <= 0:
        return []

    changed_index = min(max(changed_index, 0), len(text))
    pos = _line_start_with_context(text, changed_index, _FILE_COPY_CONTEXT_LINES)
    lines: list[str] = []
    while pos <= len(text) and len(lines) < max_lines:
        newline = text.find("\n", pos)
        if newline < 0:
            line_end = len(text)
            next_pos = len(text) + 1
        else:
            line_end = newline
            next_pos = newline + 1

        changed_offset: int | None = None
        if pos <= changed_index <= line_end:
            changed_offset = changed_index - pos
        body, eol = _split_line_ending(text[pos:next_pos])
        line = _sanitize_bounded_diff_line(body, changed_offset, max_line_chars)
        if show_eol_markers:
            line += _format_eol_marker(eol)
        lines.append(line)

        if newline < 0:
            break
        pos = next_pos

    return lines


class _FileToolCall(BaseToolCard):
    """Base renderer for file mutation tools (edit_file, write_file).

    Before/after content is supplied via ``file_snapshot`` in
    ``set_complete()`` — sourced from the mutation tracking system
    during live execution and from serialized blobs during replay.
    """

    DEFAULT_CSS = """
    _FileToolCall {
        height: auto;
        padding: 0 0 0 2;
        margin: 0 0 1 0;
    }
    _FileToolCall #ft-label {
        height: auto;
    }
    _FileToolCall > #ft-panel {
        height: auto;
        margin: 0 0 0 2;
        border: round $tui-border-primary 50%;
        border-title-color: $text;
        border-title-style: bold;
        border-subtitle-align: right;
        border-subtitle-color: $text-muted;
        padding: 0 1;
    }
    _FileToolCall.-success > #ft-panel {
        border: round $tui-border-success 30%;
        border-title-color: $success;
        border-subtitle-color: $success;

        border-title-style: not bold;
    }
    _FileToolCall.-error > #ft-panel {
        border: round $tui-border-error 50%;
        border-title-color: $text-error;
        border-subtitle-color: $text-error;

        border-title-style: not bold;
    }
    _FileToolCall.-rejected > #ft-panel {
        border: round $tui-border-warning 50%;
        border-title-color: $warning;
        border-subtitle-color: $warning;

        border-title-style: not bold;
    }
    _FileToolCall #ft-spinner {
        height: auto;
    }
    _FileToolCall #ft-content {
        height: auto;
        display: none;
    }
    _FileToolCall.-done #ft-spinner {
        display: none;
    }
    _FileToolCall.-done #ft-content {
        display: block;
    }
    """

    _SPINNERS: ClassVar[str] = "◐◓◑◒"

    def __init__(
        self,
        call_id: str,
        tool_name: str,
        args_summary: str = "",
        args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(call_id, tool_name, args_summary, args=args)
        self._spin_idx = 0
        self._file_path = self.args.get("path", "")
        self._before_content = ""
        self._after_content: str | None = None
        self._snapshot_ref: FileSnapshotRef | None = None
        self._diff_pending = False
        self._diff_mounting = False
        self._diff_timer: Timer | None = None
        self._content_release_epoch = 0
        self._timer: Timer | None = None

    def _label_suffix(self) -> str:
        """Extra text for the label. Override in subclasses."""
        return ""

    def _label_text(self, duration_ms: int = 0) -> Text:
        t = Text()
        t.append("• ", style="bold")
        t.append(self.tool_name, style="bold")
        if duration_ms:
            t.append(f" ({fmt_duration(duration_ms)})", style="dim")
        if self._file_path:
            t.append(f" {self._file_path}", style="dim")
        suffix = self._label_suffix()
        if suffix:
            t.append(f" {suffix}", style="dim italic")
        return t

    def compose(self) -> ComposeResult:
        yield ToolCardHeader(self._label_text(), id="ft-label")
        panel = Widget(id="ft-panel")
        panel.border_title = Text(os.path.basename(self._file_path)) if self._file_path else Text(self.tool_name)
        yield panel

    def on_mount(self) -> None:
        panel = self.query_one("#ft-panel")
        panel.mount(Static(self._render_spinner(), id="ft-spinner"))
        panel.mount(Widget(id="ft-content"))
        self._timer = self.set_interval(0.12, self._spin)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        if self._diff_timer is not None:
            self._diff_timer.stop()

    def _spin(self) -> None:
        if self.status == "running":
            self._spin_idx = (self._spin_idx + 1) % len(self._SPINNERS)
            # Only a shown card repaints (see ``ToolCall._spin``); the first tick after it shows paints it.
            if not is_widget_shown_on_active_screen(self):
                return
            with suppress(Exception):
                update_static_in_place(self.query_one("#ft-spinner", Static), self._render_spinner())

    def _render_spinner(self) -> Text:
        t = Text()
        t.append(f"{self._SPINNERS[self._spin_idx]} ", style="yellow")
        t.append(render_str(widget_localizer(self), _FILE_EDIT_WRITING.bind()), style="yellow")
        return t

    def _get_after_content(self) -> str:
        """Return the after-mutation file content from snapshot."""
        self._ensure_snapshot_loaded()
        return self._after_content if self._after_content is not None else ""

    def _has_snapshot_data(self) -> bool:
        """Return true when a text snapshot is available inline or by disk ref."""
        return self._after_content is not None or self._snapshot_ref is not None

    def _ensure_snapshot_loaded(self) -> None:
        """Load a disk-backed snapshot ref on demand."""
        if self._snapshot_ref is None or self._after_content is not None:
            return
        self._before_content, self._after_content = self._snapshot_ref.resolve()

    async def _ensure_snapshot_loaded_async(self) -> None:
        """Load a disk-backed snapshot ref without blocking the UI loop."""
        if self._snapshot_ref is None or self._after_content is not None:
            return
        before, after = await asyncio.to_thread(self._snapshot_ref.resolve)
        if self._snapshot_ref is not None and self._after_content is None:
            self._before_content = before
            self._after_content = after

    async def _copy_after_snapshot_load(self) -> None:
        """Load a disk-backed snapshot before copying tool details."""
        await self._ensure_snapshot_loaded_async()
        self.copy_tool_execution()

    async def _open_view_after_snapshot_load(self) -> None:
        """Load a disk-backed snapshot before opening the full detail view."""
        await self._ensure_snapshot_loaded_async()
        self.open_tool_view()

    def _defer_tool_copy(self) -> bool:
        """Schedule copy handling when a disk-backed snapshot must be loaded first."""
        if self._snapshot_ref is not None and self._after_content is None:
            self.run_worker(self._copy_after_snapshot_load(), exclusive=True, group="file-snapshot-actions")
            return True
        return False

    def _defer_tool_view(self) -> bool:
        """Schedule view handling when a disk-backed snapshot must be loaded first."""
        if self._snapshot_ref is not None and self._after_content is None:
            self.run_worker(self._open_view_after_snapshot_load(), exclusive=True, group="file-snapshot-actions")
            return True
        return False

    def _tool_copy_full_diff_section(self) -> tuple[str, str, str] | None:
        """Return the full mutation diff for small snapshots."""
        path = self._file_path or self.tool_name
        after = self._get_after_content()
        show_eol_markers = _line_ending_sensitive_change(self._before_content, after)
        diff_lines = unified_diff(
            _copy_diff_source_lines(self._before_content, show_eol_markers=show_eol_markers),
            _copy_diff_source_lines(after, show_eol_markers=show_eol_markers),
            fromfile=f"before/{path}",
            tofile=f"after/{path}",
            lineterm="",
        )
        diff_text = "\n".join(diff_lines)
        if diff_text:
            return (self._render_message(_FILE_EDIT_DIFF.bind()), "diff", diff_text)
        return None

    def _tool_copy_preview_diff_section(self) -> tuple[str, str, str] | None:
        """Return a bounded real diff around the first changed region."""
        after = self._get_after_content()
        before = self._before_content
        path = self._file_path or self.tool_name
        total_chars = len(before) + len(after)
        changed_index = _first_mismatch_index(before, after)
        if changed_index is None:
            return None
        show_eol_markers = _line_ending_sensitive_change(before, after, changed_index=changed_index)
        before_lines = _bounded_changed_window(
            before,
            max_lines=_FILE_COPY_PREVIEW_LINES,
            max_line_chars=_file_preview.WRITE_FILE_PREVIEW_MAX_LINE_CHARS,
            changed_index=changed_index,
            show_eol_markers=show_eol_markers,
        )
        after_lines = _bounded_changed_window(
            after,
            max_lines=_FILE_COPY_PREVIEW_LINES,
            max_line_chars=_file_preview.WRITE_FILE_PREVIEW_MAX_LINE_CHARS,
            changed_index=changed_index,
            show_eol_markers=show_eol_markers,
        )
        lines = [
            f"--- before/{path}",
            f"+++ after/{path}",
            self._render_message(
                _FILE_EDIT_FULL_DIFF_OMITTED.bind(
                    total_chars=total_chars,
                    limit_chars=_FILE_COPY_FULL_DIFF_MAX_CHARS,
                )
            ),
        ]
        preview_diff = unified_diff(
            before_lines,
            after_lines,
            lineterm="",
        )
        with suppress(StopIteration):
            next(preview_diff)
            next(preview_diff)
        lines.extend(preview_diff)
        if len(lines) == 3:
            lines.append(self._render_message(_FILE_EDIT_BOUNDED_PREVIEW_UNCHANGED.bind()))
        return (self._render_message(_FILE_EDIT_DIFF_PREVIEW.bind()), "diff", "\n".join(lines))

    def _tool_copy_diff_section(self) -> tuple[str, str, str] | None:
        """Return the copy diff section, bounded for oversized snapshots."""
        after = self._get_after_content()
        before = self._before_content
        if len(before) + len(after) > _FILE_COPY_FULL_DIFF_MAX_CHARS:
            return self._tool_copy_preview_diff_section()
        return self._tool_copy_full_diff_section()

    def _tool_copy_sections(self) -> list[tuple[str, str, str]]:
        """Include the mutation diff in copy payloads when snapshot data exists."""
        sections = super()._tool_copy_sections()
        if self.has_class("-rejected") or self._result_is_error():
            return sections
        if not self._has_snapshot_data():
            return sections

        diff_section = self._tool_copy_diff_section()
        if diff_section is not None:
            sections.append(diff_section)
        return sections

    def _tool_view_output_widgets(self) -> list[Widget]:
        """Show the full, untruncated mutation diff in the view modal."""
        if self.has_class("-rejected") or self._result_is_error() or not self._has_snapshot_data():
            return super()._tool_view_output_widgets()
        after = self._get_after_content()
        # No content at all (e.g. empty write) — DiffView would render blank.
        if not self._before_content and not after:
            return super()._tool_view_output_widgets()
        # No-op mutation (identical snapshot) — DiffView has no opcodes to show,
        # so fall back to the textual result rather than a blank tab.
        if self._before_content == after:
            return super()._tool_view_output_widgets()
        # Oversized snapshots (e.g. large/minified write_file output, or a big
        # line-ending-only rewrite) would make a full diff run expensive highlight
        # + opcode work and mount thousands of lines. Fall back to the bounded
        # preview diff the copy path already uses for the same threshold. This is
        # checked before the EOL branch so large EOL-only changes are bounded too;
        # _tool_copy_diff_section()'s preview path is itself EOL-aware.
        if len(self._before_content) + len(after) > _FILE_COPY_FULL_DIFF_MAX_CHARS:
            section = self._tool_copy_diff_section()
            if section is not None:
                return build_output_views(
                    [section],
                    dark=self._view_dark(),
                    render_message=self._render_message,
                )
            return super()._tool_view_output_widgets()
        # Line-ending-only changes vanish under DiffView's splitlines(); use the
        # EOL-marked unified diff so the change stays visible.
        if _line_ending_sensitive_change(self._before_content, after):
            section = self._tool_copy_full_diff_section()
            if section is not None:
                return build_output_views(
                    [section],
                    dark=self._view_dark(),
                    render_message=self._render_message,
                )
            return super()._tool_view_output_widgets()

        return [
            ToolViewDiff(
                self._file_path or self.tool_name,
                self._before_content,
                after,
                render_message=self._render_message,
            )
        ]

    def _build_diff(self) -> Widget | Static | None:
        """Build the diff widget. Override in subclasses."""
        return None

    def _is_in_collapsed_tool_group(self) -> bool:
        """Return true when this tool sits inside a collapsed tool group."""
        from chrys.app.tui.widgets.chat.tool_call import ToolGroup

        return any(isinstance(ancestor, ToolGroup) and ancestor.collapsed for ancestor in self.ancestors)

    def _has_inline_diff_child(self) -> bool:
        """Return true when the inline content container already has a mounted child.

        A child that ``release_collapsed_content`` already removed stays in the
        node list until its prune completes; it must not count as mounted or a
        rebuild racing that prune would skip itself and strand the card.
        """
        with suppress(Exception):
            content = self.query_one("#ft-content")
            return any(not getattr(child, "_pruning", False) for child in content.children)
        return False

    def release_collapsed_content(self) -> None:
        """Drop expensive inline diff widgets while a completed group is collapsed.

        The raw snapshots remain on this widget, so expanding the group can
        rebuild the inline preview on demand and the detailed view/copy paths
        still have full content.
        """
        if (
            self.status != "complete"
            or self.has_class("-rejected")
            or self._result_is_error()
            or not self._has_snapshot_data()
        ):
            return
        self._diff_pending = True
        self._content_release_epoch += 1
        if self._diff_timer is not None:
            self._diff_timer.stop()
            self._diff_timer = None
        self.add_class("-done")
        with suppress(Exception):
            content = self.query_one("#ft-content")
            for child in list(content.children):
                child.remove()
        if self._snapshot_ref is not None:
            self._before_content = ""
            self._after_content = None

    def _update_title(self, result: str, *, defer_snapshot: bool = False) -> None:
        """Update panel border title on completion. Override in subclasses."""

    def set_complete(
        self,
        result: str,
        duration_ms: int = 0,
        *,
        file_snapshot: FileSnapshotPayload | None = None,
        lazy: bool = False,
        **kwargs: Any,
    ) -> None:
        """Mark as complete and mount diff view.

        *file_snapshot* supplies ``(before, after)`` content from the
        mutation tracking system (both live and replay paths).

        If *lazy* is True, defer diff construction entirely until
        ``mount_diff_if_pending()`` is called (used during replay when the
        tool group is collapsed).
        """
        if file_snapshot is not None:
            if isinstance(file_snapshot, FileSnapshotRef):
                self._snapshot_ref = file_snapshot
                self._before_content = ""
                self._after_content = None
            else:
                self._snapshot_ref = None
                self._before_content = file_snapshot[0]
                self._after_content = file_snapshot[1]
        self.result_text = result
        self.duration_ms = duration_ms
        self.status = "complete"
        metadata = kwargs.get("metadata")
        self.metadata = metadata if isinstance(metadata, dict) else {}
        if self._timer is not None:
            self._timer.stop()

        # Update label
        with suppress(Exception):
            self.query_one("#ft-label", Static).update(self._label_text(duration_ms))

        self.approval = kwargs.get("approval")
        render_status = self._result_render_status()
        self.status = render_status
        if render_status == "rejected":
            self.add_class("-rejected")
            panel = self.query_one("#ft-panel")
            panel.border_title = Text(os.path.basename(self._file_path))
            panel.border_subtitle = render_text(widget_localizer(self), TOOL_CARD_REJECTED.bind())
            with suppress(Exception):
                content = self.query_one("#ft-content")
                content.mount(Static(Text(result)))
            self.add_class("-done")
            self._show_tool_copy_button()
        elif render_status == "error":
            self.add_class("-error")
            panel = self.query_one("#ft-panel")
            panel.border_title = Text(os.path.basename(self._file_path))
            panel.border_subtitle = render_text(widget_localizer(self), _FILE_EDIT_ERROR.bind())
            with suppress(Exception):
                content = self.query_one("#ft-content")
                content.mount(Static(Text(result)))
            self.add_class("-done")
            self._show_tool_copy_button()
        else:
            self.add_class("-success")
            defer_snapshot = isinstance(file_snapshot, FileSnapshotRef)
            self._update_title(result, defer_snapshot=defer_snapshot)
            self._show_tool_copy_button()
            if lazy or defer_snapshot:
                # Store flag — diff will be built on first expand
                self._diff_pending = True
            else:
                # Defer diff mounting — syntax highlighting is expensive and
                # blocks the event loop. A short timer lets fast-completing
                # tool groups collapse first, avoiding work for historical
                # diffs that are no longer visible.
                self._diff_timer = self.set_timer(_INLINE_DIFF_MOUNT_DELAY_SECONDS, self._schedule_mount_diff)

    async def mount_diff_if_pending(self) -> bool:
        """Build and mount the diff if it was deferred (lazy mode).

        Called when the parent ToolGroup is first expanded after replay.
        """
        if not self._diff_pending or self.has_class("-rejected") or self._result_is_error():
            return False
        return await self._mount_diff_async(post_absorb=False)

    def _result_render_status(self) -> str:
        return tool_result_render_status(
            self.result_text,
            self.approval,
            self.tool_kind or KIND_FILESYSTEM_WRITE,
            self.metadata,
        )

    def _result_is_error(self) -> bool:
        return self._result_render_status() == "error"

    async def mount_pending_content(self) -> bool:
        """Build and mount deferred inline content after the parent group expands."""
        return await self.mount_diff_if_pending()

    def _schedule_mount_diff(self) -> None:
        """Schedule async diff mounting after the short collapse grace period."""
        self._diff_timer = None
        if not self.is_attached:
            self._diff_pending = True
            return
        self.call_later(self._mount_live_diff_async)

    async def _mount_live_diff_async(self) -> None:
        """Mount a deferred live diff and request absorb after prewarming."""
        await self._mount_diff_async(post_absorb=True)

    async def _mount_diff_async(self, *, post_absorb: bool) -> bool:
        """Build and mount the diff widget, computing highlights off-thread."""
        if not self.is_attached:
            self._diff_pending = True
            self.add_class("-done")
            return False
        self._diff_timer = None
        if self._diff_mounting:
            return False
        if self._has_inline_diff_child():
            self._diff_pending = False
            self.add_class("-done")
            return False
        if self._is_in_collapsed_tool_group():
            self._diff_pending = True
            self.add_class("-done")
            return False
        self._diff_mounting = True
        release_epoch = self._content_release_epoch
        mounted = False
        mount_error: Exception | None = None
        try:
            await self._ensure_snapshot_loaded_async()
            if self._is_in_collapsed_tool_group():
                self._diff_pending = True
                return False
            diff_widget = self._build_diff()
            if diff_widget is not None:
                from chrys.app.tui.widgets.diff_view.inline import InlineUnifiedDiffLines
                from chrys.app.tui.widgets.diff_view.protocols import SupportsPrepare

                # Pre-compute Pygments highlighting + SequenceMatcher in a
                # worker thread so compose() finds cached results instantly.
                if isinstance(diff_widget, SupportsPrepare):
                    try:
                        await diff_widget.prepare()
                    except Exception:
                        diff_widget = Static(Text(self.result_text))
                if self._is_in_collapsed_tool_group():
                    self._diff_pending = True
                    return False
                if self._has_inline_diff_child():
                    self._diff_pending = False
                    return False
                try:
                    content = self.query_one("#ft-content")
                    await content.mount(diff_widget)
                    mounted = True
                    if isinstance(diff_widget, InlineUnifiedDiffLines):
                        if post_absorb:

                            def prewarm_and_request_absorb() -> None:
                                diff_widget.prewarm_visible_rows()
                                if (
                                    self.is_attached
                                    and diff_widget.is_attached
                                    and not self._is_in_collapsed_tool_group()
                                ):
                                    self.post_message(GcAbsorbRequested(GcAbsorbReason.STABLE_CONTENT_MOUNTED))

                            self.call_after_refresh(prewarm_and_request_absorb)
                        else:
                            self.call_after_refresh(diff_widget.prewarm_visible_rows)
                    elif post_absorb:
                        self.post_message(GcAbsorbRequested(GcAbsorbReason.STABLE_CONTENT_MOUNTED))
                except Exception as exc:
                    mount_error = exc
            if release_epoch != self._content_release_epoch:
                # The group collapsed while the mount was in flight and
                # ``release_collapsed_content`` already dropped the widget we
                # just attached. Keep the diff deferred so the next expand
                # rebuilds it, and rebuild right away if the group is already
                # expanded again (the expand pass saw us mid-mount and skipped).
                self._diff_pending = True
                if not self._is_in_collapsed_tool_group():
                    self.call_later(self._mount_diff_async, post_absorb=post_absorb)
                return False
            if diff_widget is not None and not mounted:
                # The mount itself failed. Keep the diff deferred so the next
                # expand (or explicit mount request) retries instead of leaving
                # the card without inline content and nothing left to rebuild.
                logger.warning(
                    "Inline diff mount failed for %s; deferring the rebuild to the next expand.",
                    self.call_id,
                    exc_info=mount_error,
                )
                self._diff_pending = True
                return False
            self._diff_pending = False
            return mounted
        finally:
            self._diff_mounting = False
            self.add_class("-done")

    def set_error(self, error: str) -> None:
        """Mark as failed."""
        self.result_text = error
        self.status = "error"
        if self._timer is not None:
            self._timer.stop()

        panel = self.query_one("#ft-panel")
        panel.border_title = Text(os.path.basename(self._file_path))
        panel.border_subtitle = render_text(widget_localizer(self), _FILE_EDIT_ERROR.bind())
        self.add_class("-error")
        with suppress(Exception):
            content = self.query_one("#ft-content")
            content.mount(Static(Text(error)))
        self.add_class("-done")
        self._show_tool_copy_button()


# ---------------------------------------------------------------------------
# EditFileToolCall — split diff view
# ---------------------------------------------------------------------------


class EditFileToolCall(_FileToolCall):
    """Renderer for ``edit_file`` — full-file split DiffView."""

    def _render_spinner(self) -> Text:
        t = Text()
        t.append(f"{self._SPINNERS[self._spin_idx]} ", style="yellow")
        t.append(render_str(widget_localizer(self), _FILE_EDIT_EDITING.bind()), style="yellow")
        return t

    def _build_diff(self) -> Widget | Static | None:
        from chrys.app.tui.widgets.diff_view.inline import InlineUnifiedDiffLines

        after = self._get_after_content()
        if not self._before_content and not after:
            return Static(Text(self.result_text))
        path = self._file_path
        return InlineUnifiedDiffLines(
            path,
            path,
            self._before_content,
            after,
            auto_height=True,
            max_display_lines=_EDIT_FILE_INLINE_MAX_DISPLAY_LINES,
        )

    def _update_title(self, result: str, *, defer_snapshot: bool = False) -> None:
        panel = self.query_one("#ft-panel")
        panel.border_title = Text(os.path.basename(self._file_path))
        if defer_snapshot:
            panel.border_subtitle = render_text(widget_localizer(self), _FILE_EDIT_CHANGES.bind())
            return
        # Compute counts cheaply via difflib without creating a full DiffView.
        # The DiffView for display is built separately in _build_diff().
        import difflib

        after = self._get_after_content()
        before_lines = self._before_content.splitlines()
        after_lines = after.splitlines()
        sm = difflib.SequenceMatcher(lambda c: c in {" ", "\t"}, before_lines, after_lines, autojunk=True)
        additions = 0
        removals = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "delete":
                removals += i2 - i1
            elif tag == "replace":
                additions += j2 - j1
                removals += i2 - i1
            elif tag == "insert":
                additions += j2 - j1
        panel.border_subtitle = render_text(
            widget_localizer(self),
            _FILE_EDIT_CHANGE_COUNTS.bind(additions=additions, removals=removals),
        )


# ---------------------------------------------------------------------------
# WriteFileToolCall — unified diff view
# ---------------------------------------------------------------------------


class WriteFileToolCall(_FileToolCall):
    """Renderer for ``write_file`` — unified DiffView showing new content."""

    def _label_suffix(self) -> str:
        overwrite = self.args.get("overwrite", False)
        return render_str(widget_localizer(self), _FILE_EDIT_OVERWRITE.bind()) if overwrite else ""

    _PREVIEW_LINES = 10

    def _build_diff(self) -> Widget | Static | None:
        from chrys.app.tui.widgets.diff_view.inline import InlineUnifiedDiffLines

        after = self._get_after_content()
        if not after:
            return Static(Text(self.result_text)) if self.result_text else None
        preview_text = _file_preview.write_file_preview_text(
            after,
            max_lines=self._PREVIEW_LINES,
            max_line_chars=_file_preview.WRITE_FILE_PREVIEW_MAX_LINE_CHARS,
        )
        path = self._file_path
        return InlineUnifiedDiffLines(path, path, "", preview_text, max_display_lines=10)

    def _update_title(self, result: str, *, defer_snapshot: bool = False) -> None:
        panel = self.query_one("#ft-panel")
        panel.border_title = Text(os.path.basename(self._file_path))
        if defer_snapshot and _WRITE_FILE_RESULT_LINES_RE.search(result) is None:
            total_lines = "?"
        else:
            after = "" if defer_snapshot else self._get_after_content()
            total_lines = _write_file_line_count(result, after)
        panel.border_subtitle = render_text(
            widget_localizer(self),
            _FILE_EDIT_LINES_WRITTEN.bind(lines=total_lines),
        )
