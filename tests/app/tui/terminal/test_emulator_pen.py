# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Graphic rendition (SGR onto a pen) and character sets (designations and shifts)."""

from __future__ import annotations

from functools import reduce
from operator import or_

import pytest

from chrys.app.tui.terminal.emulator import DEFAULT_PEN, Attribute, Pen, Rgb, TerminalEmulator
from chrys.app.tui.terminal.emulator.charsets import Charsets
from chrys.app.tui.terminal.emulator.pen import apply_sgr

_EVERYTHING = Pen(
    foreground=1,
    background=Rgb(1, 2, 3),
    attributes=Attribute.BOLD | Attribute.ITALIC | Attribute.UNDERLINE | Attribute.STRIKE,
    link="https://example.com",
)


# -- attributes --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "attribute"),
    [
        ("1", Attribute.BOLD),
        ("2", Attribute.DIM),
        ("3", Attribute.ITALIC),
        ("4", Attribute.UNDERLINE),
        ("5", Attribute.BLINK),
        ("6", Attribute.BLINK),
        ("7", Attribute.REVERSE),
        ("8", Attribute.CONCEAL),
        ("9", Attribute.STRIKE),
        ("21", Attribute.DOUBLE_UNDERLINE),
        ("53", Attribute.OVERLINE),
    ],
)
def test_attribute_is_switched_on(code: str, attribute: Attribute) -> None:
    assert apply_sgr(DEFAULT_PEN, code) == Pen(attributes=attribute)


@pytest.mark.parametrize(
    ("code", "cleared"),
    [
        ("22", Attribute.BOLD | Attribute.DIM),
        ("23", Attribute.ITALIC),
        ("24", Attribute.UNDERLINE | Attribute.DOUBLE_UNDERLINE),
        ("25", Attribute.BLINK),
        ("27", Attribute.REVERSE),
        ("28", Attribute.CONCEAL),
        ("29", Attribute.STRIKE),
        ("55", Attribute.OVERLINE),
    ],
)
def test_attribute_is_switched_off(code: str, cleared: Attribute) -> None:
    everything = reduce(or_, Attribute)

    assert apply_sgr(Pen(attributes=everything), code).attributes == everything & ~cleared


@pytest.mark.parametrize(
    ("parameters", "attributes"),
    [
        ("4:0", Attribute.NONE),
        ("4:1", Attribute.UNDERLINE),
        ("4:2", Attribute.DOUBLE_UNDERLINE),
        ("4:3", Attribute.UNDERLINE),
        ("4:5", Attribute.UNDERLINE),
    ],
)
def test_underline_shape(parameters: str, attributes: Attribute) -> None:
    # Whatever the shape asked for, the pen holds one kind of underline at a time.
    underlined = Pen(attributes=Attribute.UNDERLINE | Attribute.DOUBLE_UNDERLINE | Attribute.BOLD)

    assert apply_sgr(underlined, parameters).attributes == attributes | Attribute.BOLD


@pytest.mark.parametrize("parameters", ["", "0", "00"])
def test_reset_keeps_nothing_but_the_link(parameters: str) -> None:
    assert apply_sgr(_EVERYTHING, parameters) == Pen(link="https://example.com")


def test_codes_apply_in_order() -> None:
    assert apply_sgr(DEFAULT_PEN, "1;31;42") == Pen(1, 2, Attribute.BOLD)
    assert apply_sgr(DEFAULT_PEN, "1;31;0;3") == Pen(attributes=Attribute.ITALIC)
    # An omitted parameter is zero, which is reset.
    assert apply_sgr(DEFAULT_PEN, "1;;3") == Pen(attributes=Attribute.ITALIC)


@pytest.mark.parametrize("code", ["10", "26", "50", "99", "65535", "99999999999999999999"])
def test_unknown_codes_are_skipped(code: str) -> None:
    assert apply_sgr(_EVERYTHING, code) == _EVERYTHING
    assert apply_sgr(DEFAULT_PEN, f"{code};3") == Pen(attributes=Attribute.ITALIC)


# -- colors ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        ("30", Pen(foreground=0)),
        ("37", Pen(foreground=7)),
        ("90", Pen(foreground=8)),
        ("97", Pen(foreground=15)),
        ("40", Pen(background=0)),
        ("47", Pen(background=7)),
        ("100", Pen(background=8)),
        ("107", Pen(background=15)),
        ("38;5;196", Pen(foreground=196)),
        ("48;5;0", Pen(background=0)),
        ("38:5:196", Pen(foreground=196)),
        ("38;2;10;20;30", Pen(foreground=Rgb(10, 20, 30))),
        ("48;2;0;0;0", Pen(background=Rgb(0, 0, 0))),
        ("38:2:10:20:30", Pen(foreground=Rgb(10, 20, 30))),
        ("38:2::10:20:30", Pen(foreground=Rgb(10, 20, 30))),
        ("48:2:1:10:20:30", Pen(background=Rgb(10, 20, 30))),
    ],
)
def test_colors(parameters: str, expected: Pen) -> None:
    assert apply_sgr(DEFAULT_PEN, parameters) == expected


def test_default_colors() -> None:
    assert apply_sgr(_EVERYTHING, "39") == _EVERYTHING._replace(foreground=None)
    assert apply_sgr(_EVERYTHING, "49") == _EVERYTHING._replace(background=None)


def test_legacy_extended_color_spends_its_operands() -> None:
    assert apply_sgr(DEFAULT_PEN, "38;5;196;1") == Pen(foreground=196, attributes=Attribute.BOLD)
    assert apply_sgr(DEFAULT_PEN, "38;2;1;2;3;4") == Pen(foreground=Rgb(1, 2, 3), attributes=Attribute.UNDERLINE)


def test_sub_parameter_extended_color_spends_nothing_else() -> None:
    assert apply_sgr(DEFAULT_PEN, "38:5:196;1") == Pen(foreground=196, attributes=Attribute.BOLD)
    assert apply_sgr(DEFAULT_PEN, "38:2::1:2:3;5;4") == Pen(
        foreground=Rgb(1, 2, 3), attributes=Attribute.BLINK | Attribute.UNDERLINE
    )


@pytest.mark.parametrize("parameters", ["38;5;256;1", "38;2;256;0;0;1", "38;2;0;0;99999;1", "38:5:256;1", "38:2:1:2;1"])
def test_unusable_color_is_skipped_with_its_operands(parameters: str) -> None:
    # The operands must not be read as codes of their own: 5 would blink, 2 would dim.
    assert apply_sgr(DEFAULT_PEN, parameters) == Pen(attributes=Attribute.BOLD)


@pytest.mark.parametrize("parameters", ["38", "38;5", "38;2;1;2", "48", "58"])
def test_truncated_extended_color_changes_nothing(parameters: str) -> None:
    assert apply_sgr(_EVERYTHING, parameters) == _EVERYTHING


def test_underline_color_is_consumed_and_ignored() -> None:
    assert apply_sgr(DEFAULT_PEN, "58;5;196;1") == Pen(attributes=Attribute.BOLD)
    assert apply_sgr(DEFAULT_PEN, "58;2;1;2;3;1") == Pen(attributes=Attribute.BOLD)
    assert apply_sgr(DEFAULT_PEN, "58:2::1:2:3;1") == Pen(attributes=Attribute.BOLD)


def test_eraser_keeps_only_the_background() -> None:
    assert _EVERYTHING.eraser == Pen(background=Rgb(1, 2, 3))
    assert Pen(foreground=1, attributes=Attribute.REVERSE, link="x").eraser is DEFAULT_PEN


# -- character sets ----------------------------------------------------------------------------------


def test_special_graphics_draw_lines() -> None:
    charsets = Charsets()
    charsets.designate(0, "0")

    assert charsets.translate("lqqk x mqqj") == "┌──┐ │ └──┘"
    assert charsets.translate("ABC 123") == "ABC 123"


@pytest.mark.parametrize(
    ("designator", "ascii_text", "expected"),
    [
        ("B", "#@[\\]^_`{|}~", "#@[\\]^_`{|}~"),
        ("A", "#1", "£1"),
        ("K", "@[\\]{|}~", "§ÄÖÜäöüß"),
        ("R", "#@[\\]{|}~", "£à°ç§éùè¨"),
        ("f", "#", "£"),
        ("Z", "#@[\\]{|}", "£§¡Ñ¿°ñç"),
        ("H", "@[\\]^`{|}~", "ÉÄÖÅÜéäöåü"),
        ("7", "[", "Ä"),
        ("E", "[\\]", "ÆØÅ"),
        ("6", "[\\]", "ÆØÅ"),
        ("`", "[\\]", "ÆØÅ"),
        ("C", "[\\]", "ÄÖÅ"),
        ("5", "[\\]", "ÄÖÅ"),
        ("Q", "@[\\]", "àâçê"),
        ("9", "@[\\]", "àâçê"),
        ("Y", "#@[", "£§°"),
        ("4", "#@[", "£¾ĳ"),
        ("=", "#@[", "ùàé"),
    ],
)
def test_national_set_replaces_only_the_open_positions(designator: str, ascii_text: str, expected: str) -> None:
    charsets = Charsets()
    charsets.designate(0, designator)

    assert charsets.translate(ascii_text) == expected
    assert charsets.translate("plain text 09") == "plain text 09"


@pytest.mark.parametrize("designator", ["<", "%5"])
def test_supplemental_set_is_the_upper_half_of_latin_1_as_dec_had_it(designator: str) -> None:
    charsets = Charsets()
    charsets.designate(0, designator)

    assert charsets.translate("!1Aa") == "¡±Áá"
    # Where DEC's set predates Latin-1 and differs from it.
    assert charsets.translate("(W]w}") == "¤ŒŸœÿ"
    # Positions DEC left unassigned, and the space that no 94-character set covers.
    assert charsets.translate("$&P~ ") == "$&P~ "


def test_unknown_designator_means_ascii() -> None:
    charsets = Charsets()
    charsets.designate(0, "0")
    charsets.designate(0, "?")

    assert charsets.translate("lqk") == "lqk"


def test_locking_shift_selects_the_slot_that_prints() -> None:
    charsets = Charsets()
    charsets.designate(1, "0")
    charsets.designate(3, "A")

    assert charsets.translate("q#") == "q#"
    charsets.lock(1)
    assert charsets.translate("q#") == "─#"
    charsets.lock(3)
    assert charsets.translate("q#") == "q£"
    charsets.lock(0)
    assert charsets.translate("q#") == "q#"


def test_single_shift_covers_one_character() -> None:
    charsets = Charsets()
    charsets.designate(2, "0")
    charsets.designate(0, "A")

    charsets.shift_once(2)

    assert charsets.translate("q#q") == "─£q"
    assert charsets.translate("q") == "q"


def test_snapshot_restores_designations_and_locking_shift() -> None:
    charsets = Charsets()
    charsets.designate(1, "0")
    charsets.lock(1)
    saved = charsets.snapshot()

    charsets.designate(1, "B")
    charsets.lock(0)
    charsets.shift_once(2)
    charsets.restore(saved)

    assert charsets.translate("qq") == "──"


def _printed(stream: str) -> str:
    emulator = TerminalEmulator(40, 3)
    emulator.feed(stream)
    return emulator.buffer.screen_text[0]


def test_emulator_designates_and_shifts() -> None:
    assert _printed("\x1b(0lqk\x1b(Blqk") == "┌─┐lqk"
    assert _printed("\x1b)0q\x0eq\x0fq") == "q─q"
    assert _printed("\x1b*0\x1bnq") == "─"
    assert _printed("\x1b+0\x1boq") == "─"
    assert _printed("\x1b(%5!") == "¡"


def test_emulator_single_shifts() -> None:
    assert _printed("\x1b*0\x1bNqq") == "─q"
    assert _printed("\x1b+A\x1bO##") == "£#"
    # The shifted character may be the first of a later read.
    emulator = TerminalEmulator(40, 3)
    emulator.feed("\x1b*0\x1bN")
    emulator.feed("q")
    emulator.feed("q")
    assert emulator.buffer.screen_text[0] == "─q"


def test_emulator_accepts_right_half_invocations_and_ignores_them() -> None:
    assert _printed("\x1b)0\x1b~\x1b}\x1b|q") == "q"


def test_emulator_saves_character_sets_with_the_cursor() -> None:
    assert _printed("\x1b(0\x1b7\x1b(B\x1b8q") == "─"
    assert _printed("\x1b)0\x0e\x1b7\x0f\x1b8\rq") == "─"


def test_emulator_resets_character_sets() -> None:
    assert _printed("\x1b(0\x1b[!pq") == "q"
    assert _printed("\x1b(0\x1bcq") == "q"
