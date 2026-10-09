# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Terminal protocols preserve newline, submission, selection and paste in InputBar."""

from __future__ import annotations

import pytest
from textual._xterm_parser import XTermParser
from textual.app import App, ComposeResult
from textual.widgets import TextArea

from chrys.app.tui.widgets.chrome.input_bar import InputBar
from chrys.foundation.patches import textual_extended_keys
from chrys.foundation.patches.textual_kitty_keyboard import apply_runtime_patch as apply_kitty_patch
from chrys.foundation.patches.textual_windows_keys import get_parser_class
from tests.support.keyboard_patches import isolated_keyboard_patches as isolated_keyboard_patches
from tests.support.waiting import wait_for


@pytest.fixture(autouse=True)
def _production_keyboard_patch(isolated_keyboard_patches: None) -> None:
    apply_kitty_patch()
    textual_extended_keys.apply_runtime_patch()


class _InputApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[str] = []

    def compose(self) -> ComposeResult:
        yield InputBar()

    def on_input_bar_user_submitted(self, event: InputBar.UserSubmitted) -> None:
        self.submitted.append(event.text)


@pytest.mark.parametrize(
    ("windows", "sequence"),
    [
        (False, "\x1b[13;2u"),
        (False, "\n"),
        (False, "\x1b[27;2;13~"),
        (False, "\x1b[27;5;106~"),
        (False, "\x1b[13;2;13u"),
        (False, "\x1b[13;5u\x1b[13;5:3u"),
        (False, "\x1b[27;5;13~"),
        (True, "\x1b[13;28;13;1;16;1_\x1b[13;28;13;0;16;1_"),
        (True, "\x1b[13;28;13;1;272;1_"),
        (True, "\x1b[74;36;10;1;8;1_"),
        (True, "\x1b[13;28;10;1;8;1_\x1b[13;28;10;0;8;1_"),
    ],
)
async def test_raw_protocol_reaches_icode_newline_then_plain_enter_submits(windows: bool, sequence: str) -> None:
    app = _InputApp()
    async with app.run_test() as pilot:
        input_bar = app.query_one(InputBar)
        area = input_bar.query_one(TextArea)
        area.focus()
        await pilot.press("a")
        parser = get_parser_class()() if windows else XTermParser()
        for character in sequence:
            for event in parser.feed(character):
                app.post_message(event)
        await wait_for(lambda: input_bar.value == "a\n", pilot=pilot, description="protocol newline reaches InputBar")
        assert not app.submitted
        await pilot.press("b")
        submit = "\x1b[13;28;13;1;0;1_\x1b[13;28;13;0;0;1_" if windows else "\r"
        for event in parser.feed(submit):
            app.post_message(event)
        await wait_for(lambda: bool(app.submitted), pilot=pilot, description="plain Enter submits")
        assert app.submitted == ["a\nb"]


@pytest.mark.parametrize("windows", [False, True])
async def test_extended_typing_shortcuts_and_paste_reach_input_bar(windows: bool) -> None:
    """Extended reporting must not lose capitals, punctuation or Ctrl shortcuts."""
    app = _InputApp()
    async with app.run_test() as pilot:
        input_bar = app.query_one(InputBar)
        input_bar.query_one(TextArea).focus()
        parser = get_parser_class()() if windows else XTermParser()
        typing = (
            "\x1b[65;30;65;1;16;1_\x1b[49;2;33;1;16;1_\x1b[0;0;24403;1;0;1_\x1b[0;0;21069;1;0;1_"
            if windows
            else "\x1b[27;2;65~\x1b[27;2;33~\x1b[32;;24403:21069u"
        )
        for event in parser.feed(typing):
            app.post_message(event)
        await wait_for(lambda: input_bar.value == "A!当前", pilot=pilot, description="extended text is inserted")
        for event in parser.feed("\x1b[65;30;1;1;8;1_" if windows else "\x1b[27;5;97~"):
            app.post_message(event)
        await pilot.pause()
        assert input_bar.query_one(TextArea).selected_text == "A!当前"
        paste = "\x1b[200~first\nsecond\x1b[201~"
        if windows:
            paste = "".join(f"\x1b[0;0;{ord(char)};1;0;1_" for char in paste)
        for event in parser.feed(paste):
            app.post_message(event)
        await wait_for(lambda: input_bar.value == "first\nsecond", pilot=pilot, description="paste replaces selection")
        assert not app.submitted
        for event in parser.feed("\x1b[13;28;13;1;0;1_" if windows else "\r"):
            app.post_message(event)
        await wait_for(lambda: bool(app.submitted), pilot=pilot, description="pasted draft is submitted")
        assert app.submitted == ["first\nsecond"]


@pytest.mark.parametrize("native_letters", [False, True])
async def test_windows_paste_with_native_control_records_reaches_input_bar(native_letters: bool) -> None:
    app = _InputApp()
    async with app.run_test() as pilot:
        input_bar = app.query_one(InputBar)
        input_bar.query_one(TextArea).focus()
        parser = get_parser_class()()
        first = "\x1b[65;30;97;1;0;1_" if native_letters else "a"
        last = "\x1b[66;48;98;1;0;1_" if native_letters else "b"
        paste = "\x1b[200~" + first + "\x1b[13;28;13;1;0;1_\x1b[9;15;9;1;0;1_" + last + "\x1b[201~"
        for character in paste:
            for event in parser.feed(character):
                app.post_message(event)
        await wait_for(
            lambda: input_bar.value == "a\n\tb", pilot=pilot, description="native multiline paste reaches input"
        )
        assert not app.submitted
        for event in parser.feed("\x1b[13;28;13;1;0;1_"):
            app.post_message(event)
        await wait_for(lambda: bool(app.submitted), pilot=pilot, description="Enter submits the complete pasted draft")
        assert app.submitted == ["a\n\tb"]


@pytest.mark.parametrize("windows", [False, True])
async def test_legacy_editing_chords_reach_input_bar(windows: bool) -> None:
    app = _InputApp()
    async with app.run_test() as pilot:
        input_bar = app.query_one(InputBar)
        area = input_bar.query_one(TextArea)
        area.focus()
        await pilot.press("a", "space", "t", "w", "o")
        parser = get_parser_class()() if windows else XTermParser()
        for sequence, expected in (
            ("\x1b[8;14;8;1;2;1_" if windows else "\x1b[27;3;127~", "a "),
            ("\x1b[72;35;8;1;8;1_" if windows else "\x1b[27;5;104~", "a"),
        ):
            for event in parser.feed(sequence):
                app.post_message(event)
            await pilot.pause()
            assert input_bar.value == expected

        for event in parser.feed("\x1b[73;23;9;1;8;1_" if windows else "\x1b[27;5;105~"):
            app.post_message(event)
        await pilot.pause()
        assert not area.has_focus
        assert not app.submitted
        area.focus()
        await pilot.pause()
        for event in parser.feed("\x1b[77;50;13;1;8;1_" if windows else "\x1b[27;5;109~"):
            app.post_message(event)
        await pilot.pause()
        assert app.submitted == ["a"]


@pytest.mark.parametrize("windows", [False, True])
async def test_alt_b_and_alt_f_move_the_input_bar_cursor_by_word(windows: bool) -> None:
    app = _InputApp()
    async with app.run_test() as pilot:
        area = app.query_one(InputBar).query_one(TextArea)
        area.focus()
        area.insert("alpha beta gamma")
        parser = get_parser_class()() if windows else XTermParser()
        alt_b, alt_f = (
            ("\x1b[66;48;98;1;2;1_", "\x1b[70;33;102;1;2;1_") if windows else ("\x1b[27;3;98~", "\x1b[27;3;102~")
        )
        for sequence, column in ((alt_b, 11), (alt_b, 6), (alt_f, 10)):
            for event in parser.feed(sequence):
                app.post_message(event)
            await pilot.pause()
            assert area.cursor_location == (0, column)
