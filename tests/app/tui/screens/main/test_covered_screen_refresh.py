# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A translucent dialog over a live MainScreen receives each underlay repaint once."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from textual import __version__ as textual_version
from textual._compositor import LayoutUpdate
from textual._context import visible_screen_stack
from textual.geometry import Region
from textual.widgets import Static

from chrys.app.tui.screens.dialogs.tool_view import ToolDetailModal
from chrys.app.tui.screens.main import screen as main_screen_module
from chrys.app.tui.screens.main.screen import MainScreen
from chrys.app.tui.widgets.chat.messages import AgentMessage, UserMessage
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chat.renderers.sub_agent import SubAgentToolCall
from chrys.app.tui.widgets.chrome.status_bar import StatusBar
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from tests.support.pilot_barrier import screen_is_settled
from tests.support.tui_app_harness import make_chrys_app
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from pathlib import Path

    from rich.style import Style
    from textual.screen import Screen


def test_the_background_refresh_fork_matches_the_pinned_textual() -> None:
    """A Textual upgrade must explicitly re-audit the private background refresh fork."""
    assert textual_version == main_screen_module.TEXTUAL_BACKGROUND_REFRESH_FORK_VERSION


def _rows(regions: tuple[Region, ...]) -> set[int]:
    return {y for region in regions for y in region.line_range}


def _frame(screen: Screen[object]) -> list[list[tuple[str, Style | None]]]:
    """*screen*'s composited frame over the screens beneath it, from its render caches."""
    token = visible_screen_stack.set(screen.app._background_screens)
    try:
        strips = screen._compositor.render_strips()
    finally:
        visible_screen_stack.reset(token)
    return [[(segment.text, segment.style) for segment in strip] for strip in strips]


def _painted(screen: Screen[object]) -> bool:
    """*screen* holds no repaint it has yet to composite."""
    return not (screen._repaint_required or screen._dirty_widgets or screen._compositor._dirty_regions)


async def test_live_card_under_translucent_dialog_forwards_only_fresh_damage(tmp_path: Path) -> None:
    """Underlay repaints keep reaching the dialog, each carrying only its own rows, and none go stale.

    Card spinners park under an overlay, so the test repaints the card itself, as its
    other live updates (elapsed time, streamed activity) do.
    """
    app = make_chrys_app(tmp_path)
    async with app.run_test(size=(160, 44)) as pilot:
        main = app.screen
        assert isinstance(main, MainScreen)
        panel = main.query_one(ChatPanel)
        transcript: list[UserMessage | AgentMessage] = []
        for index in range(16):
            transcript.extend(
                (
                    UserMessage(f"Question {index}\nwith a second line"),
                    AgentMessage(f"## Answer {index}\n\nA paragraph with `code` and **emphasis**."),
                )
            )
        card = SubAgentToolCall("live-card", "explore_agent", args={"prompt": "inspect the codebase"})
        await panel.mount(*transcript, card)
        await wait_for(
            lambda: all(markdown.source for markdown in panel.query(VirtualizedMarkdown)),
            pilot=pilot,
            description="populated transcript composition",
        )
        panel.scroll_end(animate=False)
        await wait_for(
            lambda: card in main._compositor.visible_widgets and card._timer is not None,
            pilot=pilot,
            description="the running card is on screen and ticking",
        )

        modal = ToolDetailModal(title="Details", input_widgets=[], output_widgets=[Static("Output")])
        await app.push_screen(modal)
        await wait_for(
            lambda: app.screen is modal and modal.is_mounted and screen_is_settled(app, modal),
            pilot=pilot,
            description="the dialog is up over MainScreen",
        )
        assert main in app._background_screens
        assert modal.styles.background.a < 1

        timer = card._timer
        assert timer is not None
        with patch.object(modal, "refresh", autospec=True, side_effect=modal.refresh) as refresh:

            def forwarded() -> list[set[int]]:
                return [_rows(call.args) for call in refresh.call_args_list if call.args]

            # Damage the underlay outside the card once, with the spinner held.
            timer.pause()
            status_bar = main.query_one(StatusBar)
            status_bar.set_profile("Reviewer", "profile changed under the dialog")
            status_rows = set(status_bar.region.line_range)
            await wait_for(
                lambda: (
                    any(rows & status_rows for rows in forwarded())
                    and screen_is_settled(app, main)
                    and not main._dirty_widgets
                    and not main._compositor._dirty_regions
                ),
                pilot=pilot,
                description="the status change has reached the dialog",
            )

            # Then repaint only the card, once per update.
            seeded = len(forwarded())
            for repaint in range(1, 7):
                card.refresh()
                await wait_for(
                    lambda repaint=repaint: len(forwarded()) >= seeded + repaint,
                    pilot=pilot,
                    description=f"card repaint {repaint} reaches the dialog",
                )
            card_rows = set(card.region.line_range)
            repaints = forwarded()[seeded:]
            assert all(rows and rows <= card_rows for rows in repaints), (
                f"after the status change, underlay repaints forwarded rows {repaints}, "
                f"not only the repainted card's rows {sorted(card_rows)}"
            )

            await wait_for(
                lambda: (
                    screen_is_settled(app, main)
                    and screen_is_settled(app, modal)
                    and _painted(main)
                    and _painted(modal)
                ),
                pilot=pilot,
                description="the last card repaint is composited on the dialog",
            )
        cached = _frame(modal)
        modal.refresh()
        fresh = _frame(modal)
        stale_rows = [y for y, (shown, current) in enumerate(zip(cached, fresh, strict=True)) if shown != current]
        assert stale_rows == [], f"the dialog still shows old underlay content on rows {stale_rows}"

        with patch.object(app, "_display", autospec=True, side_effect=app._display) as display:
            await app.pop_screen()
            await wait_for(
                lambda: any(
                    call.args[0] is main and isinstance(call.args[1], LayoutUpdate) for call in display.call_args_list
                ),
                pilot=pilot,
                description="MainScreen repaints its whole frame once the dialog closes",
            )
        full_updates = [
            call.args[1]
            for call in display.call_args_list
            if call.args[0] is main and isinstance(call.args[1], LayoutUpdate)
        ]
        assert full_updates[0].region == Region(0, 0, *app.size)
