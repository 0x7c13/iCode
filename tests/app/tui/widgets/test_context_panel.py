# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for ContextPanel: default context window, usage state gauges, compressed-block rendering and reflow, and localization."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.events import Resize
from textual.widgets import RichLog, Static

from chrys.app.tui import i18n as tui_i18n
from chrys.app.tui.i18n import LocaleController, LocaleSwitchStatus
from chrys.app.tui.theme import TuiVariableDefaultsMixin
from chrys.app.tui.widgets.sidebar.context import ContextPanel, ContextUsageState, _CompressedBlock
from chrys.app.tui.widgets.sidebar.panel import SidebarPanel
from chrys.foundation.config.settings import Settings
from tests.support.tui_helpers import (
    WidgetApp,
)
from tests.support.waiting import wait_for


class ContextPanelApp(TuiVariableDefaultsMixin, App):
    def compose(self) -> ComposeResult:
        yield ContextPanel()


class SidebarContextPanelApp(TuiVariableDefaultsMixin, App):
    def compose(self) -> ComposeResult:
        yield SidebarPanel()


async def test_context_panel_renders_default_context_window_before_usage_events() -> None:
    async with ContextPanelApp().run_test(size=(46, 24)) as pilot:
        await pilot.pause()

        usage_text = pilot.app.query_one("#ctx-usage-text", Static).render().plain

    assert usage_text == "0 / 200,000 (0.0%)"


def test_context_panel_usage_state_preserves_main_gauge_for_sub_agent_totals() -> None:
    cp = ContextPanel()
    main_state = ContextUsageState.with_window(
        used_tokens=57_700,
        max_context_tokens=200_000,
        total_session_tokens=236_100,
        total_session_input_tokens=100_000,
        total_session_output_tokens=136_100,
        total_session_cache_hit_tokens=4_200,
    )

    cp.watch_usage_state(main_state)

    assert cp._current_used == 57_700
    assert cp._current_max == 200_000
    assert cp._usage_history[-1] == pytest.approx(28.85)
    assert cp._total_session_tokens == 236_100

    cp.watch_usage_state(
        ContextUsageState.session_totals_only(
            main_state,
            fallback_used_tokens=99_999,
            total_session_tokens=253_535,
            total_session_input_tokens=110_000,
            total_session_output_tokens=143_535,
            total_session_cache_hit_tokens=5_000,
        )
    )

    assert cp._current_used == 57_700
    assert cp._current_max == 200_000
    assert len(cp._usage_history) == 2
    assert cp._usage_history[0] == 0.0
    assert cp._usage_history[-1] == pytest.approx(28.85)
    assert cp._total_session_tokens == 253_535
    assert cp._total_session_input_tokens == 110_000
    assert cp._total_session_output_tokens == 143_535
    assert cp._total_session_cache_hit_tokens == 5_000

    remounted = ContextPanel()
    remounted.watch_usage_state(
        ContextUsageState.session_totals_only(
            main_state,
            fallback_used_tokens=99_999,
            total_session_tokens=300_000,
        )
    )

    assert remounted._current_used == 57_700
    assert remounted._current_max == 200_000
    assert remounted._usage_history == [0.0]
    assert remounted._total_session_tokens == 300_000


def test_context_panel_compressed_block_title_includes_turn_range_and_context_id() -> None:
    panel = ContextPanel()

    def title(context_id: str, turn_range: tuple[int, int]) -> str:
        return panel._compressed_block_title(
            _CompressedBlock(context_id=context_id, summary="", freed_messages=0, turn_range=turn_range)
        )

    assert title("ctx_930083b2", (1, 4)) == "Turn 1-4 (ctx_930083b2)"
    assert title("ctx_single", (3, 3)) == "Turn 3 (ctx_single)"
    assert title("ctx_legacy", (0, 0)) == "ctx_legacy"


async def test_context_panel_compressed_blocks_rerender_in_active_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    controller = LocaleController(Settings(locale="en"))

    async with WidgetApp(lambda: ContextPanel(locale_controller=controller)).run_test(size=(60, 24)) as pilot:
        panel = pilot.app.query_one(ContextPanel)
        log = pilot.app.query_one("#ctx-snapshots", RichLog)
        panel.add_compressed_block("ctx_a", "summary", 5000, (1, 4))
        panel.add_compressed_block("ctx_b", "solo", 1, (6, 6))
        await wait_for(
            lambda: any("Turn 6 (ctx_b)" in line.text for line in log.lines),
            pilot=pilot,
            description="initial compressed block reflow",
        )

        joined = " ".join(line.text.strip() for line in log.lines)
        assert "Turn 1-4 (ctx_a)  5,000 messages" in joined
        assert "Turn 6 (ctx_b)  1 message" in joined

        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        panel.refresh_localization()
        await wait_for(
            lambda: bool(log.lines and "第" in log.lines[0].text),
            pilot=pilot,
            description="localized compressed block reflow",
        )

        joined = " ".join(line.text.strip() for line in log.lines)
        assert "第 1-4 轮 (ctx_a)  5,000 条消息" in joined
        assert "第 6 轮 (ctx_b)  1 条消息" in joined

        # A block recorded while zh-Hans is already active renders zh directly.
        panel.add_compressed_block("ctx_c", "fresh", 2, (7, 8))
        await wait_for(
            lambda: any("ctx_c" in line.text for line in log.lines),
            pilot=pilot,
            description="new localized compressed block reflow",
        )
        joined = " ".join(line.text.strip() for line in log.lines)
        assert "第 7-8 轮 (ctx_c)  2 条消息" in joined


async def test_context_panel_wraps_compressed_summary_at_visible_width() -> None:
    summary = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa"
    async with ContextPanelApp().run_test(size=(46, 24)) as pilot:
        panel = pilot.app.query_one(ContextPanel)
        log = pilot.app.query_one("#ctx-snapshots", RichLog)
        assert log.can_focus
        panel.add_compressed_block("ctx_930083b2", summary, turn_range=(1, 4))
        await wait_for(
            lambda: bool(log.lines),
            pilot=pilot,
            description="compressed summary reflow",
        )

        visible_width = log.scrollable_content_region.width
        lines = [line.text.rstrip() for line in log.lines]
        assert visible_width < 78
        assert all(line.cell_length <= visible_width for line in log.lines)
        assert lines[0] == "\u2022 Turn 1-4 (ctx_930083b2)"
        assert " ".join(line.strip() for line in lines[1:]) == summary


async def test_context_panel_reflows_block_written_while_tab_is_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chrys.app.tui.widgets.sidebar.buddy.current_buddy", lambda: None)
    summary = " ".join(f"word{index:03d}" for index in range(80))

    async with SidebarContextPanelApp().run_test(size=(120, 40)) as pilot:
        sidebar = pilot.app.query_one(SidebarPanel)
        log = pilot.app.query_one("#ctx-snapshots", RichLog)

        sidebar.focus_tab("tab-context")
        await pilot.pause()
        assert log.scrollable_content_region.width > 0

        sidebar.focus_tab("tab-toc")
        await pilot.pause()
        assert log.scrollable_content_region.width == 0

        sidebar.context_panel.add_compressed_block("ctx_hidden", summary, turn_range=(5, 7))
        assert log.lines == []

        sidebar.focus_tab("tab-context")
        # The re-shown tab needs a full layout/resize/reflow chain; loaded
        # macOS CI workers have overrun the default window.
        await wait_for(
            lambda: bool(log.scrollable_content_region.width > 0 and log.lines),
            pilot=pilot,
            timeout=15.0,
            description="hidden compressed block reflow",
        )

        visible_width = log.scrollable_content_region.width
        lines = [line.text.rstrip() for line in log.lines]
        assert visible_width > 0
        assert all(line.cell_length <= visible_width for line in log.lines)
        assert lines[0] == "\u2022 Turn 5-7 (ctx_hidden)"
        assert " ".join(line.strip() for line in lines[1:]) == summary


async def test_context_panel_coalesces_resize_reflows(monkeypatch: pytest.MonkeyPatch) -> None:
    async with ContextPanelApp().run_test(size=(46, 24)) as pilot:
        panel = pilot.app.query_one(ContextPanel)
        log = pilot.app.query_one("#ctx-snapshots", RichLog)
        panel.add_compressed_block("ctx_resize", "summary", turn_range=(8, 8))
        await wait_for(
            lambda: bool(log.lines),
            pilot=pilot,
            description="initial resize fixture reflow",
        )

        reflow_widths: list[int] = []
        log_type = type(log)
        original_reflow = log_type._reflow

        def record_reflow(widget: object, width: int) -> None:
            reflow_widths.append(width)
            original_reflow(widget, width)

        monkeypatch.setattr(log_type, "_reflow", record_reflow)
        log._last_render_width = 0
        resize = Resize(log.size, log.virtual_size)
        log.on_resize(resize)
        log.on_resize(resize)

        assert reflow_widths == []
        # The flush runs via call_after_refresh, i.e. only after the next
        # screen refresh — poll instead of betting on a single pause.
        await wait_for(lambda: reflow_widths, pilot=pilot, description="coalesced reflow")
        assert reflow_widths == [log.scrollable_content_region.width]
