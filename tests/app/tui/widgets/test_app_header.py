# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for AppHeader: product title punctuation, approval badge visibility/reactives/relocalization, and badge click handling."""

from __future__ import annotations

import pytest
from rich.cells import cell_len
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
from tests.support.waiting import wait_for, wait_until


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


def _reviewing(header: AppHeader) -> Static:
    return header.query_one("#approval-reviewing", Static)


def _review_label(header: AppHeader) -> str:
    """The review label, after checking its padding and spinner frame."""
    text = _reviewing(header).render().plain
    assert text.startswith(" ") and text.endswith(" ")
    frame, label = text.strip().split(" ", 1)
    assert frame in "◐◓◑◒"
    return label


async def test_app_header_shows_reviewing_beside_an_unchanged_badge_while_calls_are_under_review() -> None:
    async with AppHeaderApp().run_test(size=(100, 4)) as pilot:
        header = pilot.app.query_one(AppHeader)
        header.approval_mode = ApprovalMode.AUTO
        await pilot.pause()
        badge = header.query_one("#approval-badge", Static)
        reviewing = _reviewing(header)
        assert not reviewing.visible

        for count, label in ((2, "Reviewing"), (1, "Reviewing"), (0, None)):
            header.set_auto_review_count(count)
            await pilot.pause()
            assert reviewing.visible is (label is not None)
            assert badge.render().plain == " APPROVAL MODE: AUTO "
            if label is not None:
                assert _review_label(header) == label
                assert reviewing.styles.width is not None
                assert reviewing.styles.width.value == cell_len(reviewing.render().plain)


async def test_app_header_review_count_set_before_mount_shows_once_mounted() -> None:
    header = AppHeader()
    header.set_auto_review_count(3)

    async with WidgetApp(lambda: header).run_test() as pilot:
        await pilot.pause()
        assert _reviewing(header).visible
        assert _review_label(header) == "Reviewing"
        assert header.query_one("#approval-badge", Static).render().plain == " APPROVAL MODE: MANUAL "


async def test_app_header_unchanged_review_count_leaves_the_header_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    async with AppHeaderApp().run_test() as pilot:
        header = pilot.app.query_one(AppHeader)
        header.set_auto_review_count(1)
        await pilot.pause()
        refreshes: list[None] = []
        monkeypatch.setattr(header, "_refresh_review_count", lambda: refreshes.append(None))

        header.set_auto_review_count(1)
        header.set_auto_review_count(2)

        assert refreshes == [None]


async def test_app_header_review_spinner_turns_only_while_calls_are_under_review() -> None:
    class _CountingHeader(AppHeader):
        ticks = 0

        def _spin_review(self) -> None:
            self.ticks += 1
            super()._spin_review()

    header = _CountingHeader()
    async with WidgetApp(lambda: header).run_test() as pilot:
        assert not await wait_until(lambda: header.ticks > 0, timeout=0.5, pilot=pilot)

        header.set_auto_review_count(1)
        first_frame = _reviewing(header).render().plain[1]
        await wait_for(
            lambda: _reviewing(header).render().plain[1] != first_frame,
            pilot=pilot,
            description="the review spinner turning",
        )

        header.set_auto_review_count(0)
        stopped_at = header.ticks
        assert not await wait_until(lambda: header.ticks > stopped_at, timeout=0.5, pilot=pilot)


async def test_app_header_review_label_is_localized(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)

    async with WidgetApp(lambda: AppHeader(locale_controller=controller)).run_test() as pilot:
        header = pilot.app.query_one(AppHeader)
        header.approval_mode = ApprovalMode.AUTO
        header.set_auto_review_count(2)
        await pilot.pause()

        controller.switch_locale("zh-Hans")

        assert _review_label(header) == "审查中"
        assert header.query_one("#approval-badge", Static).render().plain == " 审批模式：自动 "  # noqa: RUF001
