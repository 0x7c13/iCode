# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What the terminal sends the program: keys in each encoding, pastes, pointer reports and focus."""

from __future__ import annotations

import pytest

from chrys.app.tui.terminal.emulator import (
    FOCUS_IN,
    FOCUS_OUT,
    KeyProtocol,
    MouseEncoding,
    MouseTracking,
    PointerAction,
    encode_key,
    encode_paste,
    encode_pointer,
)

_EVERY_PROTOCOL = list(KeyProtocol)
# The encodings that are xterm's own. The kitty protocol is somebody else's, with other ideas.
_XTERM_PROTOCOLS = [KeyProtocol.LEGACY, KeyProtocol.MODIFY_OTHER_KEYS_1, KeyProtocol.MODIFY_OTHER_KEYS_2]

# -- keys that are not characters --------------------------------------------------------------------


@pytest.mark.parametrize("protocol", _EVERY_PROTOCOL)
@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("up", "\x1b[A"),
        ("down", "\x1b[B"),
        ("right", "\x1b[C"),
        ("left", "\x1b[D"),
        ("home", "\x1b[H"),
        ("end", "\x1b[F"),
        ("shift+up", "\x1b[1;2A"),
        ("alt+down", "\x1b[1;3B"),
        ("alt+shift+right", "\x1b[1;4C"),
        ("ctrl+left", "\x1b[1;5D"),
        ("ctrl+shift+left", "\x1b[1;6D"),
        ("ctrl+alt+home", "\x1b[1;7H"),
        ("ctrl+alt+shift+end", "\x1b[1;8F"),
        ("shift+f1", "\x1b[1;2P"),
        ("alt+f2", "\x1b[1;3Q"),
        ("ctrl+f4", "\x1b[1;5S"),
        ("insert", "\x1b[2~"),
        ("delete", "\x1b[3~"),
        ("pageup", "\x1b[5~"),
        ("pagedown", "\x1b[6~"),
        ("f5", "\x1b[15~"),
        ("f6", "\x1b[17~"),
        ("f7", "\x1b[18~"),
        ("f8", "\x1b[19~"),
        ("f9", "\x1b[20~"),
        ("f10", "\x1b[21~"),
        ("f11", "\x1b[23~"),
        ("f12", "\x1b[24~"),
        ("shift+delete", "\x1b[3;2~"),
        ("ctrl+pageup", "\x1b[5;5~"),
        ("alt+f5", "\x1b[15;3~"),
        ("ctrl+shift+f12", "\x1b[24;6~"),
    ],
)
def test_function_keys_every_protocol_spells_alike(key: str, expected: str, protocol: KeyProtocol) -> None:
    assert encode_key(key, None, protocol=protocol) == expected


@pytest.mark.parametrize("protocol", _XTERM_PROTOCOLS)
@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("f1", "\x1bOP"),
        ("f2", "\x1bOQ"),
        ("f3", "\x1bOR"),
        ("f4", "\x1bOS"),
        ("shift+f3", "\x1b[1;2R"),
        ("ctrl+f3", "\x1b[1;5R"),
        ("f13", "\x1b[25~"),
        ("f14", "\x1b[26~"),
        ("f15", "\x1b[28~"),
        ("f16", "\x1b[29~"),
        ("f17", "\x1b[31~"),
        ("f18", "\x1b[32~"),
        ("f19", "\x1b[33~"),
        ("f20", "\x1b[34~"),
        ("ctrl+f13", "\x1b[25;5~"),
        # xterm's Meta is the eighth bit, and it has none for Super or Hyper.
        ("meta+up", "\x1b[1;9A"),
        ("meta+shift+delete", "\x1b[3;10~"),
        ("super+up", "\x1b[A"),
        ("hyper+ctrl+up", "\x1b[1;5A"),
    ],
)
def test_function_keys_as_xterm_spells_them(key: str, expected: str, protocol: KeyProtocol) -> None:
    assert encode_key(key, None, protocol=protocol) == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        # No SS3 any more: a key that types nothing is a control sequence, always.
        ("f1", "\x1b[P"),
        ("f2", "\x1b[Q"),
        ("f4", "\x1b[S"),
        # And no F3 that reads as a cursor position report.
        ("f3", "\x1b[13~"),
        ("shift+f3", "\x1b[13;2~"),
        ("ctrl+f3", "\x1b[13;5~"),
        # Past F12 the numbers are the protocol's own.
        ("f13", "\x1b[57376u"),
        ("f20", "\x1b[57383u"),
        ("f21", "\x1b[57384u"),
        ("f35", "\x1b[57398u"),
        ("ctrl+f13", "\x1b[57376;5u"),
        # So are the modifier bits after Ctrl: Super, Hyper, Meta.
        ("super+up", "\x1b[1;9A"),
        ("hyper+up", "\x1b[1;17A"),
        ("meta+up", "\x1b[1;33A"),
        ("hyper+ctrl+up", "\x1b[1;21A"),
        ("meta+shift+delete", "\x1b[3;34~"),
        ("super+f1", "\x1b[1;9P"),
    ],
)
def test_function_keys_as_the_kitty_protocol_spells_them(key: str, expected: str) -> None:
    assert encode_key(key, None, protocol=KeyProtocol.KITTY) == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("up", "\x1bOA"),
        ("down", "\x1bOB"),
        ("right", "\x1bOC"),
        ("left", "\x1bOD"),
        ("home", "\x1bOH"),
        ("end", "\x1bOF"),
        # A modified cursor key has one spelling only.
        ("ctrl+up", "\x1b[1;5A"),
        ("shift+home", "\x1b[1;2H"),
        # And application mode is about the cursor keys alone.
        ("f1", "\x1bOP"),
        ("delete", "\x1b[3~"),
        ("enter", "\r"),
    ],
)
@pytest.mark.parametrize("protocol", _XTERM_PROTOCOLS)
def test_application_cursor_keys(key: str, expected: str, protocol: KeyProtocol) -> None:
    assert encode_key(key, None, application_cursor_keys=True, protocol=protocol) == expected


@pytest.mark.parametrize("key", ["up", "down", "right", "left", "home", "end"])
def test_kitty_protocol_knows_no_application_cursor_keys(key: str) -> None:
    plain = encode_key(key, None, protocol=KeyProtocol.KITTY)

    assert plain is not None
    assert plain.startswith("\x1b[")
    assert encode_key(key, None, application_cursor_keys=True, protocol=KeyProtocol.KITTY) == plain


@pytest.mark.parametrize("key", ["f21", "print_screen", "scroll_lock", "caps_lock", "ctrl+f21", "no such key"])
def test_key_that_is_nothing_a_terminal_knows_sends_nothing(key: str) -> None:
    assert encode_key(key, None) is None


@pytest.mark.parametrize("key", ["f36", "print_screen", "no such key"])
def test_key_that_is_nothing_the_kitty_protocol_was_taught_here_sends_nothing(key: str) -> None:
    assert encode_key(key, None, protocol=KeyProtocol.KITTY) is None


# -- keys that are characters, the way terminals have always sent them -------------------------------


@pytest.mark.parametrize(
    ("key", "character", "expected"),
    [
        ("a", "a", "a"),
        ("A", "A", "A"),
        ("z", None, "z"),
        ("shift+a", None, "A"),
        ("shift+a", "A", "A"),
        ("5", "5", "5"),
        ("é", "é", "é"),
        ("你", "你", "你"),
        ("space", " ", " "),
        ("enter", "\r", "\r"),
        ("enter", None, "\r"),
        ("tab", "\t", "\t"),
        ("escape", None, "\x1b"),
        ("backspace", None, "\x7f"),
        ("backspace", "\x08", "\x7f"),
        ("shift+enter", None, "\r"),
        ("shift+space", " ", " "),
        ("shift+backspace", None, "\x7f"),
        ("shift+tab", None, "\x1b[Z"),
        ("ctrl+shift+tab", None, "\x1b[Z"),
        ("alt+shift+tab", None, "\x1b\x1b[Z"),
        ("ctrl+enter", None, "\r"),
        ("ctrl+space", None, "\x00"),
        ("ctrl+backspace", None, "\x08"),
        ("alt+enter", None, "\x1b\r"),
        ("alt+backspace", None, "\x1b\x7f"),
        ("alt+escape", None, "\x1b\x1b"),
        ("alt+a", None, "\x1ba"),
        ("alt+shift+a", None, "\x1bA"),
        # A terminal that tells the keys apart names the key and says Shift was held. What the two
        # type together is not in the name; it is taken to be what a US keyboard would make of them.
        ("shift+1", None, "!"),
        ("alt+shift+1", None, "\x1b!"),
        ("alt+shift+slash", None, "\x1b?"),
        ("alt+shift+left_square_bracket", None, "\x1b{"),
        ("alt+shift+é", None, "\x1bÉ"),
        ("alt+shift+ß", None, "\x1bß"),
        ("alt+shift+exclamation_mark", None, "\x1b!"),
        ("alt+ctrl+a", None, "\x1b\x01"),
        ("alt+space", " ", "\x1b "),
        ("alt+你", "你", "\x1b你"),
    ],
)
def test_text_keys(key: str, character: str | None, expected: str) -> None:
    assert encode_key(key, character) == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        *((f"ctrl+{chr(letter)}", chr(letter - 0x60)) for letter in range(ord("a"), ord("z") + 1)),
        ("ctrl+A", "\x01"),
        ("ctrl+shift+a", "\x01"),
        ("ctrl+at", "\x00"),
        ("ctrl+@", "\x00"),
        ("ctrl+grave_accent", "\x00"),
        ("ctrl+2", "\x00"),
        ("ctrl+left_square_bracket", "\x1b"),
        ("ctrl+[", "\x1b"),
        ("ctrl+3", "\x1b"),
        ("ctrl+backslash", "\x1c"),
        ("ctrl+4", "\x1c"),
        ("ctrl+right_square_bracket", "\x1d"),
        ("ctrl+5", "\x1d"),
        ("ctrl+circumflex_accent", "\x1e"),
        ("ctrl+6", "\x1e"),
        ("ctrl+underscore", "\x1f"),
        ("ctrl+minus", "\x1f"),
        ("ctrl+slash", "\x1f"),
        ("ctrl+7", "\x1f"),
        ("ctrl+8", "\x7f"),
        ("ctrl+question_mark", "\x7f"),
        ("ctrl+left_curly_bracket", "\x1b"),
        ("ctrl+vertical_line", "\x1c"),
        ("ctrl+right_curly_bracket", "\x1d"),
        ("ctrl+tilde", "\x1e"),
        # With Shift, it is what the two type together that Ctrl folds, not what the key types alone.
        ("ctrl+shift+2", "\x00"),
        ("ctrl+shift+6", "\x1e"),
        ("ctrl+shift+minus", "\x1f"),
        ("ctrl+shift+slash", "\x7f"),
        ("ctrl+shift+left_square_bracket", "\x1b"),
        ("ctrl+shift+backslash", "\x1c"),
        ("ctrl+shift+right_square_bracket", "\x1d"),
        ("ctrl+shift+grave_accent", "\x1e"),
        ("ctrl+shift+1", "!"),
        ("ctrl+shift+3", "#"),
        ("ctrl+shift+8", "*"),
        ("ctrl+shift+comma", "<"),
        ("ctrl+alt+shift+3", "\x1b#"),
        # No control shares these keys: they send what they would have sent anyway.
        ("ctrl+1", "1"),
        ("ctrl+9", "9"),
        ("ctrl+0", "0"),
        ("ctrl+comma", ","),
        ("ctrl+full_stop", "."),
        ("ctrl+é", "é"),
    ],
)
def test_control_folds_a_character_into_a_c0_control(key: str, expected: str) -> None:
    assert encode_key(key, None) == expected


@pytest.mark.parametrize(
    ("name", "character"),
    [
        ("slash", "/"),
        ("backslash", "\\"),
        ("at", "@"),
        ("minus", "-"),
        ("plus", "+"),
        ("underscore", "_"),
        ("less_than_sign", "<"),
        ("greater_than_sign", ">"),
        ("full_stop", "."),
        ("comma", ","),
        ("colon", ":"),
        ("semicolon", ";"),
        ("apostrophe", "'"),
        ("quotation_mark", '"'),
        ("exclamation_mark", "!"),
        ("question_mark", "?"),
        ("number_sign", "#"),
        ("dollar_sign", "$"),
        ("percent_sign", "%"),
        ("ampersand", "&"),
        ("asterisk", "*"),
        ("equals_sign", "="),
        ("circumflex_accent", "^"),
        ("grave_accent", "`"),
        ("tilde", "~"),
        ("vertical_line", "|"),
        ("left_parenthesis", "("),
        ("right_parenthesis", ")"),
        ("left_square_bracket", "["),
        ("right_square_bracket", "]"),
        ("left_curly_bracket", "{"),
        ("right_curly_bracket", "}"),
    ],
)
def test_punctuation_is_found_by_its_name(name: str, character: str) -> None:
    assert encode_key(name, None) == character
    assert encode_key(f"alt+{name}", None) == f"\x1b{character}"


def test_text_the_key_produced_is_what_is_sent() -> None:
    # Whatever the name says: a layout or a dead key may have made something else of it.
    assert encode_key("a", "ä") == "ä"
    assert encode_key("1", "!") == "!"
    assert encode_key("shift+1", "!") == "!"
    assert encode_key("alt+a", "å") == "\x1bå"
    # Except where the terminal, not the keyboard, decides what the key means.
    assert encode_key("ctrl+a", "a") == "\x01"
    assert encode_key("enter", "\n") == "\r"


def test_input_method_commits_several_characters_as_one_key() -> None:
    for protocol in _EVERY_PROTOCOL:
        assert encode_key("你好", "你好", protocol=protocol) == "你好"
        assert encode_key("é", "é", protocol=protocol) == "é"


def test_text_with_a_plus_sign_in_it_is_not_a_modified_key() -> None:
    for protocol in _EVERY_PROTOCOL:
        assert encode_key("+", "+", protocol=protocol) == "+"
        assert encode_key("1+1", "1+1", protocol=protocol) == "1+1"
        assert encode_key("c++", "c++", protocol=protocol) == "c++"


# -- keys the classic encoding cannot tell apart -----------------------------------------------------


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("escape", "\x1b[27u"),
        ("shift+escape", "\x1b[27;2u"),
        ("alt+escape", "\x1b[27;3u"),
        ("shift+enter", "\x1b[13;2u"),
        ("ctrl+enter", "\x1b[13;5u"),
        ("alt+enter", "\x1b[13;3u"),
        ("ctrl+shift+enter", "\x1b[13;6u"),
        ("shift+tab", "\x1b[9;2u"),
        ("ctrl+tab", "\x1b[9;5u"),
        ("shift+backspace", "\x1b[127;2u"),
        ("ctrl+backspace", "\x1b[127;5u"),
        ("alt+backspace", "\x1b[127;3u"),
        ("shift+space", "\x1b[32;2u"),
        ("ctrl+space", "\x1b[32;5u"),
        ("ctrl+a", "\x1b[97;5u"),
        ("ctrl+shift+a", "\x1b[97;6u"),
        ("alt+a", "\x1b[97;3u"),
        ("ctrl+alt+a", "\x1b[97;7u"),
        ("super+a", "\x1b[97;9u"),
        ("hyper+a", "\x1b[97;17u"),
        ("meta+a", "\x1b[97;33u"),
        ("ctrl+super+a", "\x1b[97;13u"),
        ("meta+shift+a", "\x1b[97;34u"),
        ("super+enter", "\x1b[13;9u"),
        # A key is named by what it types unshifted; the capital says Shift was held.
        ("alt+A", "\x1b[97;4u"),
        ("ctrl+Z", "\x1b[122;6u"),
        # Which is how such keys arrive here, and how they leave: no layout is asked what Shift made.
        ("ctrl+shift+1", "\x1b[49;6u"),
        ("alt+shift+slash", "\x1b[47;4u"),
        ("ctrl+i", "\x1b[105;5u"),
        ("ctrl+m", "\x1b[109;5u"),
        ("ctrl+left_square_bracket", "\x1b[91;5u"),
        ("ctrl+1", "\x1b[49;5u"),
        ("ctrl+comma", "\x1b[44;5u"),
        ("alt+你", "\x1b[20320;3u"),
    ],
)
def test_kitty_protocol_disambiguates(key: str, expected: str) -> None:
    assert encode_key(key, None, protocol=KeyProtocol.KITTY) == expected


@pytest.mark.parametrize(
    ("key", "character", "expected"),
    [
        ("a", "a", "a"),
        ("shift+a", "A", "A"),
        ("A", "A", "A"),
        ("exclamation_mark", "!", "!"),
        ("你", "你", "你"),
        ("enter", "\r", "\r"),
        ("tab", "\t", "\t"),
        ("backspace", None, "\x7f"),
        ("space", " ", " "),
    ],
)
def test_kitty_protocol_leaves_text_as_text(key: str, character: str | None, expected: str) -> None:
    assert encode_key(key, character, protocol=KeyProtocol.KITTY) == expected


_SPELLED_OUT_AT_EITHER_LEVEL = [
    ("shift+enter", "\x1b[27;2;13~"),
    ("ctrl+enter", "\x1b[27;5;13~"),
    ("alt+enter", "\x1b[27;3;13~"),
    ("ctrl+shift+enter", "\x1b[27;6;13~"),
    ("alt+shift+enter", "\x1b[27;4;13~"),
    # xterm's own example of level 1: ESC is not put before Tab, so Alt has no other spelling.
    ("alt+tab", "\x1b[27;3;9~"),
    ("ctrl+tab", "\x1b[27;5;9~"),
    ("alt+shift+tab", "\x1b[27;4;9~"),
    ("ctrl+shift+tab", "\x1b[27;6;9~"),
    ("alt+escape", "\x1b[27;3;27~"),
    ("ctrl+alt+escape", "\x1b[27;7;27~"),
    # Ctrl with a character that has no control character to become.
    ("ctrl+1", "\x1b[27;5;49~"),
    ("ctrl+9", "\x1b[27;5;57~"),
    ("ctrl+comma", "\x1b[27;5;44~"),
    ("ctrl+full_stop", "\x1b[27;5;46~"),
    ("ctrl+alt+comma", "\x1b[27;7;44~"),
    # xterm has no "-" among the control characters, and makes DEL of "?" too late to count.
    ("ctrl+minus", "\x1b[27;5;45~"),
    # With Shift it is what the two type together that xterm goes by: the code it reports...
    ("ctrl+shift+1", "\x1b[27;6;33~"),
    ("ctrl+shift+9", "\x1b[27;6;40~"),
    ("ctrl+shift+comma", "\x1b[27;6;60~"),
    ("ctrl+shift+equals_sign", "\x1b[27;6;43~"),
    ("ctrl+shift+slash", "\x1b[27;6;63~"),
    # ...and what is asked whether Ctrl has a control character for it. "3" has, "#" has none.
    ("ctrl+shift+3", "\x1b[27;6;35~"),
    ("ctrl+shift+8", "\x1b[27;6;42~"),
    # Meta has no classic spelling at all.
    ("meta+a", "\x1b[27;9;97~"),
    ("meta+shift+a", "\x1b[27;10;65~"),
]
# Keys with a well-known spelling, which level 1 lets them keep although it drops a modifier or is
# some other key's as well, and level 2 does not.
_SPELLED_OUT_AT_LEVEL_2_ONLY = [
    # xterm's own example of level 2.
    ("shift+tab", "\x1b[27;2;9~"),
    ("shift+escape", "\x1b[27;2;27~"),
    ("ctrl+escape", "\x1b[27;5;27~"),
    ("shift+backspace", "\x1b[27;2;127~"),
    ("alt+backspace", "\x1b[27;3;127~"),
    ("ctrl+shift+backspace", "\x1b[27;6;127~"),
    ("shift+space", "\x1b[27;2;32~"),
    ("ctrl+space", "\x1b[27;5;32~"),
    ("alt+space", "\x1b[27;3;32~"),
    ("alt+a", "\x1b[27;3;97~"),
    ("alt+shift+a", "\x1b[27;4;65~"),
    ("ctrl+a", "\x1b[27;5;97~"),
    ("ctrl+shift+a", "\x1b[27;6;65~"),
    ("ctrl+alt+a", "\x1b[27;7;97~"),
    ("ctrl+alt+shift+z", "\x1b[27;8;90~"),
    ("ctrl+i", "\x1b[27;5;105~"),
    ("ctrl+m", "\x1b[27;5;109~"),
    ("ctrl+left_square_bracket", "\x1b[27;5;91~"),
    ("ctrl+2", "\x1b[27;5;50~"),
    ("ctrl+slash", "\x1b[27;5;47~"),
    # "@", "^", "_", "{" and "~" have a control character each, which level 1 leaves them.
    ("ctrl+shift+2", "\x1b[27;6;64~"),
    ("ctrl+shift+6", "\x1b[27;6;94~"),
    ("ctrl+shift+minus", "\x1b[27;6;95~"),
    ("ctrl+shift+left_square_bracket", "\x1b[27;6;123~"),
    ("ctrl+shift+grave_accent", "\x1b[27;6;126~"),
    ("alt+shift+1", "\x1b[27;4;33~"),
    ("ctrl+alt+shift+2", "\x1b[27;8;64~"),
    ("alt+你", "\x1b[27;3;20320~"),
]
_NEVER_SPELLED_OUT = [
    ("a", "a"),
    ("shift+a", "A"),
    ("exclamation_mark", "!"),
    ("enter", None),
    ("tab", None),
    ("escape", None),
    ("backspace", None),
    # Not a chord to xterm but the other backspace.
    ("ctrl+backspace", None),
    ("你", "你"),
]


@pytest.mark.parametrize("protocol", [KeyProtocol.MODIFY_OTHER_KEYS_1, KeyProtocol.MODIFY_OTHER_KEYS_2])
@pytest.mark.parametrize(("key", "expected"), _SPELLED_OUT_AT_EITHER_LEVEL)
def test_modify_other_keys_spells_out_what_has_no_classic_spelling(
    key: str, expected: str, protocol: KeyProtocol
) -> None:
    assert encode_key(key, None, protocol=protocol) == expected


@pytest.mark.parametrize(("key", "expected"), _SPELLED_OUT_AT_LEVEL_2_ONLY)
def test_modify_other_keys_level_2_spells_out_the_well_known_keys_too(key: str, expected: str) -> None:
    classic = encode_key(key, None)

    assert classic is not None
    assert encode_key(key, None, protocol=KeyProtocol.MODIFY_OTHER_KEYS_1) == classic
    assert encode_key(key, None, protocol=KeyProtocol.MODIFY_OTHER_KEYS_2) == expected


@pytest.mark.parametrize("protocol", [KeyProtocol.MODIFY_OTHER_KEYS_1, KeyProtocol.MODIFY_OTHER_KEYS_2])
@pytest.mark.parametrize(("key", "character"), _NEVER_SPELLED_OUT)
def test_modify_other_keys_leaves_text_and_bare_keys_alone(
    key: str, character: str | None, protocol: KeyProtocol
) -> None:
    classic = encode_key(key, character)

    assert classic is not None
    assert encode_key(key, character, protocol=protocol) == classic


# -- paste -------------------------------------------------------------------------------------------


def test_bracketed_paste_is_marked_and_otherwise_untouched() -> None:
    assert encode_paste("one\ntwo\r\nthree\t你好", bracketed=True) == "\x1b[200~one\ntwo\r\nthree\t你好\x1b[201~"
    assert encode_paste("", bracketed=True) == "\x1b[200~\x1b[201~"


def test_bracketed_paste_cannot_end_itself() -> None:
    pasted = "harmless\x1b[201~; rm -rf ~\n"

    assert encode_paste(pasted, bracketed=True) == "\x1b[200~harmless; rm -rf ~\n\x1b[201~"


def test_unbracketed_paste_arrives_as_if_typed() -> None:
    assert encode_paste("one\ntwo\r\nthree\rfour", bracketed=False) == "one\rtwo\rthree\rfour"
    assert encode_paste("", bracketed=False) == ""


# -- pointer -----------------------------------------------------------------------------------------

_TRACKING_ON = [tracking for tracking in MouseTracking if tracking is not MouseTracking.OFF]


@pytest.mark.parametrize("action", list(PointerAction))
@pytest.mark.parametrize("encoding", list(MouseEncoding))
def test_nothing_is_reported_unless_asked_for(action: PointerAction, encoding: MouseEncoding) -> None:
    assert encode_pointer(MouseTracking.OFF, encoding, action, x=0, y=0, button=0) is None


@pytest.mark.parametrize(
    ("tracking", "action", "button", "reported"),
    [
        (MouseTracking.PRESS, PointerAction.PRESS, 0, True),
        (MouseTracking.PRESS, PointerAction.RELEASE, 0, False),
        (MouseTracking.PRESS, PointerAction.MOVE, 0, False),
        (MouseTracking.PRESS, PointerAction.WHEEL_UP, None, False),
        (MouseTracking.PRESS_RELEASE, PointerAction.PRESS, 0, True),
        (MouseTracking.PRESS_RELEASE, PointerAction.RELEASE, 0, True),
        (MouseTracking.PRESS_RELEASE, PointerAction.MOVE, 0, False),
        (MouseTracking.PRESS_RELEASE, PointerAction.MOVE, None, False),
        (MouseTracking.PRESS_RELEASE, PointerAction.WHEEL_DOWN, None, True),
        (MouseTracking.DRAG, PointerAction.PRESS, 0, True),
        (MouseTracking.DRAG, PointerAction.RELEASE, 0, True),
        (MouseTracking.DRAG, PointerAction.MOVE, 0, True),
        (MouseTracking.DRAG, PointerAction.MOVE, None, False),
        (MouseTracking.DRAG, PointerAction.WHEEL_UP, None, True),
        (MouseTracking.MOTION, PointerAction.PRESS, 0, True),
        (MouseTracking.MOTION, PointerAction.RELEASE, 0, True),
        (MouseTracking.MOTION, PointerAction.MOVE, 0, True),
        (MouseTracking.MOTION, PointerAction.MOVE, None, True),
        (MouseTracking.MOTION, PointerAction.WHEEL_LEFT, None, True),
    ],
)
def test_tracking_mode_decides_what_is_reported(
    tracking: MouseTracking, action: PointerAction, button: int | None, reported: bool
) -> None:
    report = encode_pointer(tracking, MouseEncoding.SGR, action, x=0, y=0, button=button)

    assert (report is not None) == reported


@pytest.mark.parametrize(
    ("action", "button", "expected"),
    [
        (PointerAction.PRESS, 0, "\x1b[<0;11;6M"),
        (PointerAction.PRESS, 1, "\x1b[<1;11;6M"),
        (PointerAction.PRESS, 2, "\x1b[<2;11;6M"),
        (PointerAction.RELEASE, 0, "\x1b[<0;11;6m"),
        (PointerAction.RELEASE, 2, "\x1b[<2;11;6m"),
        (PointerAction.RELEASE, None, "\x1b[<3;11;6m"),
        (PointerAction.MOVE, 0, "\x1b[<32;11;6M"),
        (PointerAction.MOVE, 2, "\x1b[<34;11;6M"),
        (PointerAction.MOVE, None, "\x1b[<35;11;6M"),
        (PointerAction.WHEEL_UP, None, "\x1b[<64;11;6M"),
        (PointerAction.WHEEL_DOWN, None, "\x1b[<65;11;6M"),
        (PointerAction.WHEEL_LEFT, None, "\x1b[<66;11;6M"),
        (PointerAction.WHEEL_RIGHT, None, "\x1b[<67;11;6M"),
        # The wheel is its own button, whatever else is held.
        (PointerAction.WHEEL_UP, 0, "\x1b[<64;11;6M"),
    ],
)
def test_sgr_reports(action: PointerAction, button: int | None, expected: str) -> None:
    assert encode_pointer(MouseTracking.MOTION, MouseEncoding.SGR, action, x=10, y=5, button=button) == expected


@pytest.mark.parametrize(
    ("action", "button", "code"),
    [
        (PointerAction.PRESS, 0, 0),
        (PointerAction.PRESS, 2, 2),
        # Which button was let go is more than these formats can say.
        (PointerAction.RELEASE, 0, 3),
        (PointerAction.RELEASE, 2, 3),
        (PointerAction.MOVE, 0, 32),
        (PointerAction.MOVE, None, 35),
        (PointerAction.WHEEL_UP, None, 64),
        (PointerAction.WHEEL_DOWN, None, 65),
    ],
)
def test_reports_in_the_older_formats(action: PointerAction, button: int | None, code: int) -> None:
    def report(encoding: MouseEncoding) -> str | None:
        return encode_pointer(MouseTracking.MOTION, encoding, action, x=10, y=5, button=button)

    assert report(MouseEncoding.DEFAULT) == f"\x1b[M{chr(code + 32)}{chr(11 + 32)}{chr(6 + 32)}"
    assert report(MouseEncoding.UTF8) == f"\x1b[M{chr(code + 32)}{chr(11 + 32)}{chr(6 + 32)}"
    assert report(MouseEncoding.URXVT) == f"\x1b[{code + 32};11;6M"


@pytest.mark.parametrize(
    ("shift", "alt", "ctrl", "bits"),
    [
        (False, False, False, 0),
        (True, False, False, 4),
        (False, True, False, 8),
        (False, False, True, 16),
        (True, True, True, 28),
    ],
)
def test_modifiers_held_are_reported(shift: bool, alt: bool, ctrl: bool, bits: int) -> None:
    def report(tracking: MouseTracking, action: PointerAction, button: int | None) -> str | None:
        return encode_pointer(
            tracking, MouseEncoding.SGR, action, x=0, y=0, button=button, shift=shift, alt=alt, ctrl=ctrl
        )

    assert report(MouseTracking.PRESS_RELEASE, PointerAction.PRESS, 0) == f"\x1b[<{bits};1;1M"
    assert report(MouseTracking.DRAG, PointerAction.MOVE, 1) == f"\x1b[<{33 + bits};1;1M"
    assert report(MouseTracking.MOTION, PointerAction.WHEEL_DOWN, None) == f"\x1b[<{65 + bits};1;1M"
    # X10 compatibility mode is the press and the button, nothing more.
    assert report(MouseTracking.PRESS, PointerAction.PRESS, 0) == "\x1b[<0;1;1M"


def _as_typed(report: str | None) -> bytes:
    """The bytes a report is to the program: a byte that is not text rides in it as `surrogateescape` has it."""
    assert report is not None
    return report.encode("utf-8", "surrogateescape")


@pytest.mark.parametrize(
    ("encoding", "last", "typed"),
    [
        (MouseEncoding.DEFAULT, 222, b"\x1b[M \xff\xff"),
        (MouseEncoding.UTF8, 2014, "\x1b[M ߿߿".encode()),
    ],
)
def test_one_value_per_coordinate_only_counts_so_far(encoding: MouseEncoding, last: int, typed: bytes) -> None:
    def press(x: int, y: int) -> str | None:
        return encode_pointer(MouseTracking.PRESS_RELEASE, encoding, PointerAction.PRESS, x=x, y=y, button=0)

    assert _as_typed(press(last, last)) == typed
    assert press(last + 1, 0) is None
    assert press(0, last + 1) is None


def test_original_format_spends_a_byte_on_a_coordinate_where_the_utf8_format_spends_a_character() -> None:
    def press(encoding: MouseEncoding, x: int) -> bytes:
        report = encode_pointer(MouseTracking.PRESS_RELEASE, encoding, PointerAction.PRESS, x=x, y=0, button=0)
        return _as_typed(report)

    # Up to column 95 a coordinate is ASCII, and the two formats are the same bytes.
    assert press(MouseEncoding.DEFAULT, 94) == press(MouseEncoding.UTF8, 94) == b"\x1b[M \x7f!"
    assert press(MouseEncoding.DEFAULT, 100) == b"\x1b[M \x85!"
    assert press(MouseEncoding.UTF8, 100) == b"\x1b[M \xc2\x85!"


@pytest.mark.parametrize("encoding", [MouseEncoding.SGR, MouseEncoding.URXVT])
def test_numeric_formats_count_as_far_as_the_screen_goes(encoding: MouseEncoding) -> None:
    report = encode_pointer(MouseTracking.PRESS_RELEASE, encoding, PointerAction.PRESS, x=4999, y=2999, button=0)

    assert report is not None
    assert ";5000;3000M" in report


@pytest.mark.parametrize("tracking", _TRACKING_ON)
def test_every_tracking_mode_reports_a_press(tracking: MouseTracking) -> None:
    assert encode_pointer(tracking, MouseEncoding.DEFAULT, PointerAction.PRESS, x=0, y=0, button=0) == "\x1b[M !!"


# -- focus -------------------------------------------------------------------------------------------


def test_focus_reports() -> None:
    assert (FOCUS_IN, FOCUS_OUT) == ("\x1b[I", "\x1b[O")
