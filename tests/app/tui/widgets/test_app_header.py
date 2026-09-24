# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for AppHeader: product title punctuation, approval badge visibility/reactives/relocalization, and badge click handling."""

from __future__ import annotations

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.selection import SELECT_ALL
from textual.widgets import Static

from chrys.app.tui import i18n as tui_i18n
from chrys.app.tui.i18n import LocaleController, LocaleSwitchStatus
from chrys.app.tui.widgets.chrome.app_header import AppHeader
from chrys.foundation.branding import format_app_version_title
from chrys.foundation.config.settings import Settings
from chrys.service.approval.policy import ApprovalMode
from tests.support.tui_helpers import (
    WidgetApp,
    make_click,
)


class AppHeaderApp(App):
    def compose(self) -> ComposeResult:
        yield AppHeader()


class AppHeaderWithoutApprovalApp(App):
    def compose(self) -> ComposeResult:
        yield AppHeader(show_approval_badge=False)


async def test_app_header_title_uses_product_punctuation() -> None:
    from chrys import __version__

    async with AppHeaderApp().run_test() as pilot:
        title = pilot.app.query_one("#header-title", Static)
        assert title.render().plain == format_app_version_title(__version__)


async def test_app_header_can_hide_approval_badge() -> None:
    async with AppHeaderWithoutApprovalApp().run_test() as pilot:
        assert list(pilot.app.query("#approval-badge")) == []


async def test_app_header_reactives_refresh_title_and_approval_badge() -> None:
    async with AppHeaderApp().run_test() as pilot:
        header = pilot.app.query_one(AppHeader)
        header.subtitle_parts = ("Code", "mock-model")
        header.approval_mode = ApprovalMode.AUTO
        await pilot.pause()

        title = pilot.app.query_one("#header-title", Static)
        badge = pilot.app.query_one("#approval-badge", Static)

        assert title.render().plain.endswith("Code \u2502 mock-model")
        assert badge.render().plain == " APPROVAL MODE: AUTO "
        assert badge.has_class("approval-auto")
        assert badge.allow_select is False
        pilot.app.screen.selections = {badge: SELECT_ALL}
        await pilot.pause()
        assert badge.text_selection is None
        assert pilot.app.screen.get_selected_text() == ""


async def test_app_header_relocalizes_current_approval_mode_and_unregisters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    header: AppHeader | None = None

    async with WidgetApp(lambda: AppHeader(locale_controller=controller)).run_test() as pilot:
        header = pilot.app.query_one(AppHeader)
        badge = header.query_one("#approval-badge", Static)
        assert header in controller._surfaces
        assert badge.render().plain == " APPROVAL MODE: MANUAL "

        result = controller.switch_locale("zh-Hans")
        assert result.status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        assert badge.render().plain == " 审批模式：手动 "  # noqa: RUF001

        for mode, expected in (
            (ApprovalMode.AUTO, " 审批模式：自动 "),  # noqa: RUF001
            (ApprovalMode.BYPASS, " 审批模式：绕过 "),  # noqa: RUF001
            (ApprovalMode.MANUAL, " 审批模式：手动 "),  # noqa: RUF001
        ):
            header.approval_mode = mode
            await pilot.pause()
            assert badge.render().plain == expected

    assert header is not None
    assert header not in controller._surfaces


async def test_app_header_approval_badge_click_consumes_event() -> None:
    messages: list[AppHeader.ApprovalBadgeClicked] = []

    class HeaderClickApp(App):
        def compose(self) -> ComposeResult:
            yield AppHeader()

        @on(AppHeader.ApprovalBadgeClicked)
        def on_approval_badge_clicked(self, event: AppHeader.ApprovalBadgeClicked) -> None:
            messages.append(event)

    async with HeaderClickApp().run_test() as pilot:
        header = pilot.app.query_one(AppHeader)
        badge = pilot.app.query_one("#approval-badge", Static)
        event = make_click(header, screen_x=badge.region.x, screen_y=badge.region.y)
        header.on_click(event)
        await pilot.pause()

    assert event._no_default_action is True
    assert event._stop_propagation is True
    assert len(messages) == 1
