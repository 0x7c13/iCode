# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the fork-session dialog widget."""

from __future__ import annotations

from types import SimpleNamespace


def test_fork_session_dialog_only_allows_dismiss_after_result() -> None:
    from chrys.app.tui.screens.dialogs.fork_session import ForkSessionDialog

    dialog = ForkSessionDialog()
    assert dialog._allow_click_outside_dismiss() is False
    assert dialog._default_dismiss_result() is None

    dialog.set_success("fork-1234")
    assert dialog._allow_click_outside_dismiss() is True
    assert dialog._default_dismiss_result() == "stay"

    dialog = ForkSessionDialog()
    dialog.set_error("Fork failed")
    assert dialog._allow_click_outside_dismiss() is True
    assert dialog._default_dismiss_result() is None


async def test_fork_session_dialog_omits_new_window_button_when_unavailable() -> None:
    from textual.app import App

    from chrys.app.tui.screens.dialogs.fork_session import ForkSessionDialog

    app = App()
    dialog = ForkSessionDialog("fork-1234", show_new_window=False)

    async with app.run_test():
        await app.push_screen(dialog)

        assert list(dialog.query("#fork-session-new-window")) == []
        assert list(dialog.query("#fork-session-switch"))
        assert list(dialog.query("#fork-session-stay"))


def test_fork_session_focus_skips_buttons_in_hidden_groups() -> None:
    from chrys.app.tui.screens.dialogs.fork_session import ForkSessionDialog

    class _FakeParent:
        def __init__(self, *, display: bool) -> None:
            self.display = display

    class _FakeButton:
        def __init__(self, *, parent: _FakeParent, has_focus: bool = False) -> None:
            self.display = True
            self.parent = parent
            self.has_focus = has_focus
            self.focused = False

        def focus(self) -> None:
            self.focused = True

    visible_group = _FakeParent(display=True)
    hidden_group = _FakeParent(display=False)
    switch = _FakeButton(parent=visible_group)
    stay = _FakeButton(parent=visible_group, has_focus=True)
    ok = _FakeButton(parent=hidden_group)
    dialog = SimpleNamespace(
        query=lambda _cls: [switch, stay, ok],
        _button_is_visible=ForkSessionDialog._button_is_visible,
    )

    ForkSessionDialog._focus_relative(dialog, 1)

    assert switch.focused is True
    assert ok.focused is False
    assert ForkSessionDialog._button_is_visible(ok) is False
