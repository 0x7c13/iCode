# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the log viewer handler's snapshot / trim / filter semantics."""

from __future__ import annotations

import logging

import pytest
from textual.app import App, ComposeResult
from textual.widgets import RichLog, Static, TabbedContent

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.logs.screen import (
    MAX_DISPLAY_LINE_CHARS,
    MAX_DISPLAY_LINES,
    MODULE_ORDER,
    LogViewerScreen,
    _truncate_display_line,
    _TuiLogHandler,
    install_log_handler,
)
from chrys.app.tui.widgets.hatch import HatchedEmptyState
from chrys.foundation.config.settings import Settings


def _record(name: str, msg: str, level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=0,
        msg=msg,
        args=None,
        exc_info=None,
    )


class _LogViewerApp(App):
    locale_controller = LocaleController(Settings(locale="en"))

    def compose(self) -> ComposeResult:
        yield Static("placeholder")


def test_truncate_display_line_leaves_bounded_lines_unchanged() -> None:
    line = "x" * MAX_DISPLAY_LINE_CHARS

    assert _truncate_display_line(line) == line


def test_truncate_display_line_bounds_long_lines_with_suffix() -> None:
    line = "x" * (MAX_DISPLAY_LINE_CHARS + 1)
    truncated = _truncate_display_line(line)

    assert len(truncated) == MAX_DISPLAY_LINE_CHARS
    assert truncated.endswith("...")


def test_install_log_handler_respects_explicitly_pinned_logger_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """App-wiring level pins must survive the install-time DEBUG escalation of whitelisted loggers."""
    from chrys.app.tui.screens.logs import screen as screen_mod

    root_logger = logging.getLogger()
    saved_levels = {name: logging.getLogger(name).level for name in screen_mod.MODULE_WHITELIST}
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.NOTSET)
    monkeypatch.setattr(screen_mod, "_handler", None)

    handler = screen_mod.install_log_handler()
    try:
        assert logging.getLogger("openai").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.DEBUG
    finally:
        root_logger.removeHandler(handler)
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)


def test_install_log_handler_reattaches_existing_singleton() -> None:
    root_logger = logging.getLogger()
    handler = install_log_handler()
    was_attached = handler in root_logger.handlers

    root_logger.removeHandler(handler)
    try:
        assert handler not in root_logger.handlers
        assert install_log_handler() is handler
        assert handler in root_logger.handlers
    finally:
        if handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        if was_attached:
            root_logger.addHandler(handler)


class TestTuiLogHandler:
    def test_snapshot_returns_new_lines_incrementally(self) -> None:
        h = _TuiLogHandler()
        h.setFormatter(logging.Formatter("%(message)s"))

        for i in range(3):
            h.emit(_record("chrys", f"m{i}"))

        lines, total = h.snapshot("chrys", 0)
        assert lines == ["m0", "m1", "m2"]
        assert total == 3

        # Emitting more only yields the new ones on the next snapshot
        h.emit(_record("chrys", "m3"))
        lines, total = h.snapshot("chrys", total)
        assert lines == ["m3"]
        assert total == 4

    def test_snapshot_survives_buffer_trim(self) -> None:
        """After buffer trim, subsequent emissions must still be delivered."""
        h = _TuiLogHandler()
        h.setFormatter(logging.Formatter("%(message)s"))

        # Fill past the trim threshold (2 * MAX_DISPLAY_LINES).
        # Before the fix, the flushed index became absolute and the
        # buffer-relative slice silently returned [] for new records.
        for i in range(MAX_DISPLAY_LINES * 2 + 10):
            h.emit(_record("chrys", f"m{i}"))

        # Drain everything observed so far.
        _, total = h.snapshot("chrys", 0)
        assert total == MAX_DISPLAY_LINES * 2 + 10

        # New record after trim MUST be observable.
        h.emit(_record("chrys", "fresh"))
        lines, new_total = h.snapshot("chrys", total)
        assert lines == ["fresh"]
        assert new_total == total + 1

    def test_snapshot_unknown_module_returns_empty(self) -> None:
        h = _TuiLogHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        lines, total = h.snapshot("does-not-exist", 0)
        assert lines == []
        assert total == 0

    def test_modules_with_logs_groups_by_top_level(self) -> None:
        h = _TuiLogHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        h.emit(_record("chrys.orchestration.engine.engine", "a"))
        h.emit(_record("chrys.app.tui.app", "b"))
        h.emit(_record("httpx._client", "c"))
        assert h.modules_with_logs() == ["chrys", "httpx"]

    def test_non_whitelisted_modules_are_dropped(self) -> None:
        h = _TuiLogHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        h.emit(_record("markdown_it.parser", "noise"))
        h.emit(_record("some.third.party", "noise"))
        h.emit(_record("chrys.core", "real"))
        assert h.modules_with_logs() == ["chrys"]
        lines, total = h.snapshot("markdown_it", 0)
        assert lines == []
        assert total == 0

    def test_snapshot_filters_by_min_level(self) -> None:
        """min_level drops lower-severity lines but keeps the monotonic total."""
        h = _TuiLogHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        h.emit(_record("chrys", "dbg", logging.DEBUG))
        h.emit(_record("chrys", "info", logging.INFO))
        h.emit(_record("chrys", "warn", logging.WARNING))

        debug_lines, total = h.snapshot("chrys", 0, min_level=logging.DEBUG)
        assert debug_lines == ["dbg", "info", "warn"]
        assert total == 3

        info_lines, total = h.snapshot("chrys", 0, min_level=logging.INFO)
        assert info_lines == ["info", "warn"]
        assert total == 3

        warn_lines, total = h.snapshot("chrys", 0, min_level=logging.WARNING)
        assert warn_lines == ["warn"]
        assert total == 3

    def test_snapshot_limit_returns_tail_without_changing_total(self) -> None:
        h = _TuiLogHandler()
        h.setFormatter(logging.Formatter("%(message)s"))

        for i in range(5):
            h.emit(_record("chrys", f"m{i}"))

        lines, total = h.snapshot("chrys", 0, limit=2)
        assert lines == ["m3", "m4"]
        assert total == 5

    def test_snapshot_limit_is_applied_after_level_filter(self) -> None:
        h = _TuiLogHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        h.emit(_record("chrys", "warn-0", logging.WARNING))
        h.emit(_record("chrys", "debug-0", logging.DEBUG))
        h.emit(_record("chrys", "warn-1", logging.WARNING))
        h.emit(_record("chrys", "debug-1", logging.DEBUG))

        lines, total = h.snapshot("chrys", 0, min_level=logging.WARNING, limit=2)
        assert lines == ["warn-0", "warn-1"]
        assert total == 4


@pytest.mark.asyncio
async def test_every_log_tab_scrollbar_is_one_cell_and_flush_right() -> None:
    handler = _TuiLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    for module in MODULE_ORDER:
        for index in range(100):
            handler.emit(_record(module, f"{module}-{index}"))

    screen = LogViewerScreen()
    screen._handler = handler

    async with _LogViewerApp().run_test(size=(120, 20)) as pilot:
        await pilot.app.push_screen(screen)
        await pilot.pause()

        tabs = screen.query_one("#lv-tabs", TabbedContent)
        container = screen.query_one("#lv-container")
        for module in MODULE_ORDER:
            tabs.active = f"lv-tab-{module}"
            await pilot.pause()

            log_widget = screen.query_one(f"#lv-log-{module}", RichLog)
            assert log_widget.show_vertical_scrollbar is True
            assert log_widget.styles.scrollbar_size_vertical == 1
            assert log_widget.vertical_scrollbar.region.right == container.content_region.right


@pytest.mark.asyncio
async def test_empty_state_stays_visible_after_filtered_handler_buffer_trim() -> None:
    h = _TuiLogHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    h.emit(_record("chrys", "warn", logging.WARNING))

    screen = LogViewerScreen()
    screen._handler = h
    screen._filter_level = logging.WARNING

    async with _LogViewerApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await pilot.pause()

        log_widget = screen.query_one("#lv-log-chrys", RichLog)
        empty = screen.query_one("#lv-empty-chrys", HatchedEmptyState)
        assert bool(log_widget.display)
        assert not bool(empty.display)

        for index in range(MAX_DISPLAY_LINES * 2 + 10):
            h.emit(_record("chrys", f"debug-{index}", logging.DEBUG))

        screen._refresh_logs()
        await pilot.pause()

        assert bool(log_widget.display)
        assert not bool(empty.display)


@pytest.mark.asyncio
async def test_openai_records_render_from_initial_buffer() -> None:
    h = _TuiLogHandler()
    h.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    h.emit(_record("openai", "request started", logging.INFO))

    screen = LogViewerScreen()
    screen._handler = h

    async with _LogViewerApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await pilot.pause()

        log_widget = screen.query_one("#lv-log-openai", RichLog)
        empty = screen.query_one("#lv-empty-openai", HatchedEmptyState)
        assert screen._rendered_any_lines["openai"] is True
        assert bool(log_widget.display)
        assert not bool(empty.display)


@pytest.mark.asyncio
async def test_inactive_tabs_render_lazily_on_activation() -> None:
    h = _TuiLogHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    h.emit(_record("chrys", "initial", logging.INFO))
    h.emit(_record("httpx", "lazy", logging.INFO))

    screen = LogViewerScreen()
    screen._handler = h

    async with _LogViewerApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await pilot.pause()

        assert screen._rendered_any_lines["chrys"] is True
        assert "httpx" not in screen._rendered_any_lines

        tabs = screen.query_one("#lv-tabs", TabbedContent)
        tabs.active = "lv-tab-httpx"
        await pilot.pause()

        assert screen._rendered_any_lines["httpx"] is True


@pytest.mark.asyncio
async def test_flush_module_queues_truncated_display_lines_for_inactive_tab() -> None:
    h = _TuiLogHandler()
    h.setFormatter(logging.Formatter("%(message)s"))

    screen = LogViewerScreen()
    screen._handler = h

    async with _LogViewerApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await pilot.pause()

        full_line = "x" * (MAX_DISPLAY_LINE_CHARS + 200)
        h.emit(_record("chrys", full_line, logging.INFO))

        screen._active_module = "httpx"
        screen._flush_module("chrys")

        assert screen._pending_lines["chrys"] == [_truncate_display_line(full_line)]
        assert len(screen._pending_lines["chrys"][0]) == MAX_DISPLAY_LINE_CHARS

        lines, _ = h.snapshot("chrys", 0)
        assert lines == [full_line]


@pytest.mark.asyncio
async def test_copy_logs_uses_full_lines_when_display_is_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    h = _TuiLogHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    full_line = "request-body=" + ("x" * (MAX_DISPLAY_LINE_CHARS + 200))
    h.emit(_record("chrys", full_line, logging.INFO))

    screen = LogViewerScreen()
    screen._handler = h

    async with _LogViewerApp().run_test(size=(120, 40)) as pilot:
        copied: list[str] = []
        # Patch the shared TUI clipboard helper before invoking the action.
        monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)

        await pilot.app.push_screen(screen)
        await pilot.pause()

        screen.action_copy_logs()

        assert pilot.app.clipboard == full_line
        assert copied == [full_line]
