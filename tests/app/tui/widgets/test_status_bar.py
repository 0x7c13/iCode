# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for StatusBar: localization of status/trail/tooltip, layout and flash trails, snapshot restore, and details-click messages."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Static

from chrys.app.tui import i18n as tui_i18n
from chrys.app.tui.i18n import LocaleController, LocaleSwitchStatus
from chrys.app.tui.widgets import ChrysLoadingIndicator
from chrys.app.tui.widgets.chrome.status_bar import (
    STATUS_INTERRUPTED,
    STATUS_SESSION_RESTORED,
    STATUS_THINKING,
    StatusBar,
)
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.types import (
    AgentRuntimeDetails,
    RuntimeHookDetails,
    RuntimeHookSourceDetails,
)
from tests.support.tui_helpers import (
    WidgetApp,
    make_click,
)
from tests.support.waiting import wait_for


class _DetailsClickApp(App):
    """StatusBar host that records the DetailsClicked messages it receives."""

    def __init__(self) -> None:
        self.details_clicks: list[StatusBar.DetailsClicked] = []
        super().__init__()

    def compose(self) -> ComposeResult:
        yield StatusBar()

    @on(StatusBar.DetailsClicked)
    def _record_details_click(self, event: StatusBar.DetailsClicked) -> None:
        self.details_clicks.append(event)


async def test_status_bar_status_reactive_refreshes_text() -> None:
    async with WidgetApp(StatusBar).run_test() as pilot:
        status_bar = pilot.app.query_one(StatusBar)
        status_bar.status = "Thinking"
        await pilot.pause()

        assert pilot.app.query_one("#status-text", Static).render().plain == "Thinking"


async def test_status_bar_relocalizes_status_tool_trail_tooltip_and_literal_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from chrys.app.tui.screens.main.model_indicator import ModelIndicatorState
    from chrys.app.tui.screens.main.runtime_info import RegistryRuntimeInfoProvider
    from chrys.app.tui.widgets.chrome import status_bar as status_bar_module

    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    now = [100.0]
    monkeypatch.setattr(status_bar_module.time, "monotonic", lambda: now[0])
    status_bar: StatusBar | None = None

    async with WidgetApp(lambda: StatusBar(locale_controller=controller)).run_test() as pilot:
        status_bar = pilot.app.query_one(StatusBar)
        assert status_bar in controller._surfaces
        status_bar.set_profile("Code Agent")
        status_bar.set_model(
            ModelIndicatorState(
                label="Test Model",
                tooltip="",
                mode="select",
                profile_id="test-model",
                visible=True,
            )
        )
        status_bar.start_run()
        now[0] = 161.0
        status_bar.add_tool_call()
        runtime_info = RegistryRuntimeInfoProvider(  # type: ignore[arg-type]
            SimpleNamespace(agent_registry=None)
        )
        status_bar.set_tool_info(
            runtime_info.format_tool_info(
                ["read_file", "write_file"],
                ["review"],
                memory_files=["AGENTS.md"],
                runtime_details=AgentRuntimeDetails(
                    hook_sources=[
                        RuntimeHookSourceDetails(
                            scope="project",
                            hooks=[
                                RuntimeHookDetails(id="guard", enabled=True),
                                RuntimeHookDetails(id="notify", enabled=True),
                                RuntimeHookDetails(id="disabled", enabled=False),
                            ],
                        )
                    ]
                ),
            )
        )
        status_bar.show(STATUS_THINKING.bind())

        assert status_bar.query_one("#agent-label", Static).render().plain == "Agent"
        assert status_bar.query_one("#model-label", Static).render().plain == "Model"
        assert status_bar.query_one("#status-text", Static).render().plain == "Thinking"
        assert status_bar.query_one("#status-trail", Static).render().plain == "  (1m 1s · 1 tool call)"
        assert (
            status_bar.query_one("#status-tool-info", Static).render().plain == "2 tools · 1 skill · 2 hooks · 1 file"
        )

        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        assert status_bar.query_one("#agent-label", Static).render().plain == "智能体"
        assert status_bar.query_one("#model-label", Static).render().plain == "模型"
        assert status_bar.query_one("#status-text", Static).render().plain == "正在思考"
        assert status_bar.query_one("#status-trail", Static).render().plain == "  (1分 1秒 · 1 次工具调用)"
        tool_info = status_bar.query_one("#status-tool-info", Static)
        assert tool_info.render().plain == "2 个工具 · 1 个 Skill · 2 个钩子 · 1 个文件"
        assert tool_info.tooltip is not None
        assert tool_info.tooltip.plain == "点击查看详情"

        status_bar.show("provider payload")
        assert controller.switch_locale("en").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        assert status_bar.query_one("#status-text", Static).render().plain == "provider payload"

    assert status_bar is not None
    assert status_bar not in controller._surfaces


async def test_status_bar_relocalizes_active_flash_and_snapshot_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)

    async with WidgetApp(lambda: StatusBar(locale_controller=controller)).run_test() as pilot:
        status_bar = pilot.app.query_one(StatusBar)
        status_bar.set_tool_info((STATUS_SESSION_RESTORED.bind(session_id="abc123"),))
        status_bar.show(STATUS_THINKING.bind())
        snapshot = status_bar.snapshot()

        status_bar.flash(STATUS_INTERRUPTED.bind(), caution=True)
        assert status_bar.query_one("#status-flash", Static).render().plain == "Interrupted by user"

        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        assert status_bar.query_one("#status-flash", Static).render().plain == "已由用户中断"

        status_bar.restore(snapshot)
        await pilot.pause()
        assert status_bar.query_one("#status-text", Static).render().plain == "正在思考"
        assert status_bar.query_one("#status-tool-info", Static).render().plain == "会话已恢复：abc123"  # noqa: RUF001


async def test_status_bar_completed_flash_restores_localized_elapsed_from_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.app.tui.widgets.chrome import status_bar as status_bar_module

    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    now = [100.0]
    monkeypatch.setattr(status_bar_module.time, "monotonic", lambda: now[0])

    async with WidgetApp(lambda: StatusBar(locale_controller=controller)).run_test() as pilot:
        status_bar = pilot.app.query_one(StatusBar)
        status_bar.start_run()
        now[0] = 161.0
        status_bar.flash_completed()
        snapshot = status_bar.snapshot()
        assert status_bar.query_one("#status-flash", Static).render().plain == "Completed in 1m 1s"

        status_bar.flash("shell payload", warn=True)
        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        status_bar.restore(snapshot)

        assert status_bar.query_one("#status-flash", Static).render().plain == "已在 1分 1秒 内完成"


async def test_status_bar_localizes_tool_count_choices_and_elapsed_formats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.app.tui.widgets.chrome import status_bar as status_bar_module

    controller = LocaleController(Settings(locale="zh-Hans"))
    now = [117.0]
    monkeypatch.setattr(status_bar_module.time, "monotonic", lambda: now[0])

    async with WidgetApp(lambda: StatusBar(locale_controller=controller)).run_test() as pilot:
        status_bar = pilot.app.query_one(StatusBar)
        status_bar._start_time = 100.0
        status_bar._tool_count = 1
        status_bar.show(STATUS_THINKING.bind())
        assert status_bar.query_one("#status-trail", Static).render().plain == "  (17秒 · 1 次工具调用)"

        now[0] = 221.0
        status_bar._tool_count = 2
        status_bar.refresh_localization()
        assert status_bar.query_one("#status-trail", Static).render().plain == "  (2分 1秒 · 2 次工具调用)"


async def test_status_bar_show_hide() -> None:
    async with WidgetApp(StatusBar).run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        assert sb.styles.padding.left == 0
        assert sb.query_one(".status-selectors").styles.padding.right == 1
        assert sb.query_one(".status-selectors").display is False
        assert sb.query_one(".status-body").styles.padding.left == 0
        assert sb.shown is False
        assert sb.visible is False

        sb.show("thinking...")
        assert sb.shown is True
        assert sb.visible is True
        assert sb.status == "thinking..."
        assert sb.query_one(".status-run").visible is True
        assert sb.query_one(".status-flash-bar").visible is False

        sb.show("running: read_file")
        assert sb.status == "running: read_file"

        sb.flash("done")
        assert sb.query_one(".status-run").visible is False
        assert sb.query_one(".status-flash-bar").visible is True

        sb.hide()
        assert sb.shown is False
        assert sb.visible is False
        assert sb.query_one(".status-run").visible is False
        assert sb.query_one(".status-flash-bar").visible is False


async def test_status_bar_flash_text_gets_full_row_when_trail_is_empty() -> None:
    """The auto-width trail must leave the whole row to the flash message."""

    async with WidgetApp(StatusBar).run_test(size=(80, 6)) as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.flash("M" * 55 + "TAIL-END-MARKER")
        await pilot.pause()
        await pilot.pause()
        strips = pilot.app.screen._compositor.render_strips()
        frame = "\n".join(strip.text for strip in strips)
        # A half-row flash box (the old 1fr/1fr split) clips this tail.
        assert "TAIL-END-MARKER" in frame


async def test_status_bar_run_trail_hugs_status_text() -> None:
    """The elapsed/tool-call trail sits right after the status label.

    A flexible status-text width strands the trail mid-row, far from the
    label it annotates; the tool info keeps the right edge of the row.
    """

    async with WidgetApp(StatusBar).run_test(size=(100, 6)) as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.start_run()
        sb.add_tool_call()
        sb.set_tool_info("MODEL-MARKER")
        sb.show("Running: explore_agent")
        await pilot.pause()
        await pilot.pause()

        text_widget = sb.query_one("#status-text", Static)
        trail_widget = sb.query_one("#status-trail", Static)
        assert text_widget.region.width == len("Running: explore_agent")
        assert trail_widget.region.x == text_widget.region.right

        strips = pilot.app.screen._compositor.render_strips()
        row = next(strip.text for strip in strips if "Running:" in strip.text)
        # Trail text is adjacent (its two leading spaces are part of it) and
        # the persistent tool info stays right-aligned.
        assert "Running: explore_agent  (" in row
        assert "tool call)" in row
        assert row.rstrip().endswith("MODEL-MARKER")


async def test_status_bar_warn_flash_suppresses_tool_trail() -> None:
    """Warn flashes (shell mode instructions) get the row to themselves."""

    async with WidgetApp(StatusBar).run_test(size=(80, 6)) as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_tool_info("TOOLTRAIL-MARKER")

        sb.flash("regular note")
        await pilot.pause()
        await pilot.pause()
        strips = pilot.app.screen._compositor.render_strips()
        frame = "\n".join(strip.text for strip in strips)
        assert "TOOLTRAIL-MARKER" in frame

        sb.flash("shell mode notice", warn=True)
        await pilot.pause()
        await pilot.pause()
        assert sb.query_one("#status-flash").region.width == sb.query_one(".status-body").content_region.width
        strips = pilot.app.screen._compositor.render_strips()
        frame = "\n".join(strip.text for strip in strips)
        assert "TOOLTRAIL-MARKER" not in frame


async def test_status_bar_snapshot_restore_preserves_run_state() -> None:
    async with WidgetApp(StatusBar).run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_tool_info("3 tools")
        sb.start_run()
        sb.add_tool_call()
        sb.show("Thinking")
        snapshot = sb.snapshot()

        sb.set_tool_info("")
        sb.start_run()
        sb.show("Restoring Session")
        sb.restore(snapshot)
        await pilot.pause()

        assert sb.status == "Thinking"
        assert sb._start_time == snapshot["start_time"]
        assert sb._tool_count == snapshot["tool_count"]
        assert sb._tool_trail == "3 tools"


async def test_status_bar_idle_restore_clears_shell_flash_before_tool_info_updates() -> None:
    async with WidgetApp(StatusBar).run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_profile("Code Agent")
        sb.set_tool_info("Original runtime")
        sb.clear_status()
        snapshot = sb.snapshot()

        sb.flash("Shell mode", warn=True)
        sb.restore(snapshot)
        sb.set_tool_info("Updated runtime")
        await pilot.pause()

        assert sb._flash is None
        assert not sb.has_class("-warn")
        tool_info = sb.query_one("#status-tool-info", Static)
        assert tool_info.visible is True
        assert tool_info.render().plain == "Updated runtime"


async def test_status_bar_selectors_only_restore_does_not_poison_next_run_visibility() -> None:
    async with WidgetApp(StatusBar).run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.hide()
        sb.set_profile("Code Agent")
        snapshot = sb.snapshot()
        assert snapshot["content_shown"] is False
        assert snapshot["idle_shown"] is False

        sb.flash("Shell mode", warn=True)
        sb.restore(snapshot)
        sb.show("Thinking")
        await pilot.pause()

        assert sb.query_one("ChrysLoadingIndicator").visible is True
        assert sb.query_one("#status-text", Static).visible is True
        assert sb.query_one("#status-text", Static).render().plain == "Thinking"


async def test_status_bar_hidden_restore_recovers_configured_idle_chrome() -> None:
    async with WidgetApp(StatusBar).run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_profile("Code Agent")
        sb.hide()
        snapshot = sb.snapshot()

        sb.flash("Shell mode", warn=True)
        sb.restore(snapshot)
        await pilot.pause()

        assert sb.visible is True
        assert sb._flash is None
        assert sb.query_one("#profile-tag", Static).render().plain == "Code Agent"
        assert sb.query_one("#status-text", Static).visible is False


async def test_status_bar_tooltip_is_click_hint_only() -> None:
    async with WidgetApp(StatusBar).run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_tool_info("3 tools")

        sb.show("Loading")

        tool_info = sb.query_one("#status-tool-info", Static)
        assert tool_info.tooltip is not None
        assert tool_info.tooltip.plain == "Click for details"


async def test_status_bar_empty_runtime_trail_removes_mouse_details_entry() -> None:
    from types import SimpleNamespace

    from chrys.app.tui.screens.main.runtime_info import RegistryRuntimeInfoProvider

    runtime_info = RegistryRuntimeInfoProvider(  # type: ignore[arg-type]
        SimpleNamespace(agent_registry=None)
    )
    trail = runtime_info.format_tool_info(
        [],
        [],
        runtime_details=AgentRuntimeDetails(
            hook_sources=[
                RuntimeHookSourceDetails(
                    scope="global",
                    hooks=[RuntimeHookDetails(id="disabled", enabled=False)],
                )
            ]
        ),
    )
    assert trail == ()

    app = _DetailsClickApp()
    async with app.run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_tool_info(trail)
        sb.show("Loading")
        await pilot.pause()

        tool_info = sb.query_one("#status-tool-info", Static)
        assert tool_info.render().plain == ""
        assert tool_info.styles.pointer == "default"
        assert tool_info.tooltip is None

        event = make_click(sb, screen_x=tool_info.region.x, screen_y=tool_info.region.y)
        sb.on_click(event)
        await pilot.pause()

    assert event._no_default_action is False
    assert event._stop_propagation is False
    assert app.details_clicks == []


async def test_status_bar_set_tool_info_refreshes_visible_flash_trail() -> None:
    async with WidgetApp(StatusBar).run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.flash("Completed in 1s")
        sb.set_tool_info("11 tools · 1 skill")
        await pilot.pause()

        flash_trail = sb.query_one("#status-flash-trail", Static)
        assert flash_trail.render().plain == "11 tools · 1 skill"


@pytest.mark.parametrize(
    ("reveal_trail", "target_id"),
    [
        pytest.param(lambda sb: sb.show("Loading"), "#status-tool-info", id="tool-info"),
        pytest.param(lambda sb: sb.flash("Profile: Code", trail="3 tools"), "#status-flash-trail", id="flash-trail"),
    ],
)
async def test_status_bar_posts_details_clicked(reveal_trail: Callable[[StatusBar], None], target_id: str) -> None:
    """Clicking either trail rendering of the tool info posts DetailsClicked and consumes the click."""
    app = _DetailsClickApp()
    async with app.run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_tool_info("3 tools")
        reveal_trail(sb)
        await pilot.pause()

        target = sb.query_one(target_id, Static)
        event = make_click(sb, screen_x=target.region.x, screen_y=target.region.y)
        sb.on_click(event)
        await pilot.pause()

    assert event._no_default_action is True
    assert event._stop_propagation is True
    assert len(app.details_clicks) == 1


async def test_status_bar_spinner_ticks_only_while_it_is_painted() -> None:
    """The bar swaps faces through layout-free visibility flips; their Show/Hide park and restart the spinner's timer."""
    async with WidgetApp(StatusBar).run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        indicator = sb.query_one(ChrysLoadingIndicator)
        timer = indicator._auto_refresh_timer
        assert timer is not None
        assert indicator.visible is False
        await wait_for(lambda: not timer._active.is_set(), pilot=pilot, description="the hidden spinner is parked")

        sb.show("Thinking")
        assert indicator.visible is True
        await wait_for(timer._active.is_set, pilot=pilot, description="the running face animates the spinner")

        sb.flash("Done")
        assert indicator.visible is False
        await wait_for(lambda: not timer._active.is_set(), pilot=pilot, description="the flash face parks the spinner")

        sb.show("Thinking again")
        await wait_for(timer._active.is_set, pilot=pilot, description="the spinner animates again")
        sb.hide()
        await wait_for(lambda: not timer._active.is_set(), pilot=pilot, description="the hidden bar parks the spinner")
