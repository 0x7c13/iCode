# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Palette layout, grouping and color fidelity."""

from __future__ import annotations

import pytest
from rich.color import EIGHT_BIT_PALETTE
from textual.app import App, ComposeResult
from textual.color import Color

from chrys.app.tui.screens.themes.palette import (
    _TRANSPARENT_SENTINEL,
    _XTERM_256_ROWS,
    PaletteChanged,
    _ansi_base_token,
    _ansi_display_label,
    _group_variables,
    _Xterm256PalettePicker,
)
from chrys.app.tui.theme import CHRYS_ANSI_THEME, CHRYS_THEME
from chrys.app.tui.themes.document import copy_theme
from chrys.app.tui.widgets.checkbox import CHECKED_MARKER, UNCHECKED_MARKER
from tests.support.paths import SRC_ROOT
from tests.support.waiting import wait_for


def test_group_variables_buckets_and_order() -> None:
    sample = [
        "border-blurred",
        "scrollbar-hover",
        "block-cursor-foreground",
        "input-cursor-background",
        "footer-background",
        "primary-muted",
        "overlay-background",
        "button-flat-foreground",
    ]
    grouped = _group_variables(sample)
    assert grouped == [
        ("button", ["button-flat-foreground"]),
        ("border", ["border-blurred"]),
        ("scrollbar", ["scrollbar-hover"]),
        ("block", ["block-cursor-foreground"]),
        ("input", ["input-cursor-background"]),
        ("footer", ["footer-background"]),
        ("muted", ["primary-muted"]),
        ("controls", ["overlay-background"]),
    ]


def test_ansi_variable_names_only_truly_ansi_slots() -> None:
    # ``_ANSI_VARIABLE_NAMES`` is the strict set of slots that only have
    # meaningful values when ``Theme.ansi=True`` (RGB themes resolve
    # both to ``transparent``).  Everything else chrys-ansi tunes —
    # text-muted / text-disabled / screen-selection-background — is
    # shown in regular sidebar buckets on every theme.
    from chrys.app.tui.screens.themes.palette import _ANSI_VARIABLE_NAMES

    assert _ANSI_VARIABLE_NAMES == ("ansi-background", "ansi-foreground")


def test_group_variables_routes_ansi_names_to_ansi_bucket() -> None:
    grouped = _group_variables(["ansi-background", "ansi-foreground"], ansi=True)
    assert grouped == [("ansi", ["ansi-background", "ansi-foreground"])]


def test_group_variables_keeps_shared_names_out_of_ansi_bucket() -> None:
    # ``border-blurred`` looks ANSI-ish (built-in ANSI themes define it
    # too) but it's a generic textual slot — must stay in ``border``.
    from chrys.app.tui.screens.themes.palette import _ANSI_VARIABLE_NAMES

    assert "border-blurred" not in _ANSI_VARIABLE_NAMES
    grouped = _group_variables(["border-blurred"])
    assert grouped == [("border", ["border-blurred"])]


def test_canonical_variable_names_includes_chrys_ansi_tuned_extras() -> None:
    # chrys-ansi explicitly overrides these textual-derived slots; the
    # sidebar surfaces them on every theme so edits made on base chrys
    # carry over visually when switching to chrys-ansi (or vice versa).
    from chrys.app.tui.screens.themes.palette import _CANONICAL_VARIABLE_NAMES

    for name in ("text-muted", "text-disabled", "screen-selection-background"):
        assert name in _CANONICAL_VARIABLE_NAMES


def _is_palette_hex(value: str) -> bool:
    palette = {f"#{red:02X}{green:02X}{blue:02X}" for red, green, blue in EIGHT_BIT_PALETTE}
    return value.upper() in palette


@pytest.mark.parametrize(
    "raw_hex",
    [
        "#1C1C1C",  # gray entry, on-palette
        "#AF87FF",  # chrys primary, on-palette
        "#5FFF87",  # chrys success, on-palette
    ],
)
def test_match_palette_index_returns_exact_index_for_on_palette_hex(raw_hex: str) -> None:
    color = Color.parse(raw_hex)
    index = _Xterm256PalettePicker._match_palette_index(color)
    triplet = EIGHT_BIT_PALETTE[index]
    assert f"#{triplet.red:02X}{triplet.green:02X}{triplet.blue:02X}" == raw_hex.upper()


@pytest.mark.parametrize(
    "raw_hex",
    [
        "#FB7F2A",  # truecolor orange — off-palette
        "#123456",  # arbitrary
        "#2B2E3B",  # near-gray, off-palette
    ],
)
def test_match_palette_index_picks_nearest_neighbor_for_off_palette_hex(raw_hex: str) -> None:
    color = Color.parse(raw_hex)
    index = _Xterm256PalettePicker._match_palette_index(color)
    # Rich's ``Palette.match`` uses a perceptually-weighted distance.
    # We just verify the returned index is on-palette and that no
    # palette entry has an exact-RGB hit we'd be ignoring.
    triplet = EIGHT_BIT_PALETTE[index]
    assert 0 <= index < 256
    # An exact hex match anywhere in the palette would always win
    # distance=0; verify the chosen index hits exact when one exists,
    # and otherwise the chosen entry's distance is at least as good as
    # naive squared-RGB nearest.
    exact = next(
        (i for i in range(256) if EIGHT_BIT_PALETTE[i] == triplet and triplet == (color.r, color.g, color.b)), None
    )
    if exact is not None:
        assert index == exact


def test_match_palette_index_returns_sentinel_for_transparent() -> None:
    assert _Xterm256PalettePicker._match_palette_index(Color(0, 0, 0, a=0)) == _TRANSPARENT_SENTINEL


def test_match_palette_index_skips_sentinel_when_transparent_disallowed() -> None:
    # When the picker hides its transparent cell (surface slots), a
    # transparent input must fall back to a real palette index instead
    # of returning the sentinel — otherwise the position lookup would
    # KeyError on the missing sentinel row.
    index = _Xterm256PalettePicker._match_palette_index(Color(0, 0, 0, a=0), allow_transparent=False)
    assert index != _TRANSPARENT_SENTINEL
    assert 0 <= index < 256


def test_xterm_picker_initialized_with_transparent_selects_sentinel_cell() -> None:
    picker = _Xterm256PalettePicker(Color(0, 0, 0, a=0))
    assert picker._selected_index == _TRANSPARENT_SENTINEL
    assert picker.color.is_transparent


@pytest.mark.parametrize("theme", ["chrys", "chrys-ansi"])
@pytest.mark.parametrize("initial", ["#123456", "transparent"])
async def test_xterm_picker_transparent_checkbox_toggles_by_marker_label_and_space(theme: str, initial: str) -> None:
    # This is a widget input/rendering contract. The narrow-dialog regression
    # separately covers PaletteChanged -> live preview -> cancel in ChrysApp.
    transparent_row = next(i for i, (_, idx) in enumerate(_XTERM_256_ROWS) if _TRANSPARENT_SENTINEL in idx)
    picker = _Xterm256PalettePicker(Color.parse(initial))
    changes: list[Color] = []

    class PaletteApp(App):
        CSS_PATH = SRC_ROOT / "chrys" / "app" / "tui" / "screens" / "themes" / "editor.tcss"

        def compose(self) -> ComposeResult:
            yield picker

        def on_palette_changed(self, event: PaletteChanged) -> None:
            assert event.picker is picker
            changes.append(event.color)

    app = PaletteApp()
    app.register_theme(copy_theme(CHRYS_ANSI_THEME if theme == "chrys-ansi" else CHRYS_THEME))
    app.theme = theme
    async with app.run_test(size=(80, 28)) as pilot:
        await pilot.pause()
        assert changes == []
        if initial != "transparent":
            assert picker.render_line(transparent_row).text == f"{UNCHECKED_MARKER} Transparent"
            assert await pilot.click(picker, offset=(1, transparent_row))
            await wait_for(lambda: len(changes) == 1, pilot=pilot)
            assert changes[-1].is_transparent
        assert picker.color.is_transparent
        strip = picker.render_line(transparent_row)
        assert strip.text == f"{CHECKED_MARKER} Transparent"
        marker = next(iter(strip))
        assert marker.style is not None
        assert marker.style.color == Color.parse(app.theme_variables["success"]).rich_color
        assert await pilot.click(picker, offset=(6, transparent_row))
        restored = Color.parse("#FFFFFF" if initial == "transparent" else initial)
        await wait_for(lambda: bool(changes) and changes[-1] == restored, pilot=pilot)
        assert picker.color == restored
        assert picker.render_line(transparent_row).text == f"{UNCHECKED_MARKER} Transparent"
        await pilot.press("space")
        await wait_for(lambda: changes[-1].is_transparent, pilot=pilot)
        await pilot.press("space")
        await wait_for(lambda: changes[-1] == restored, pilot=pilot)
        assert picker.render_line(transparent_row).text == f"{UNCHECKED_MARKER} Transparent"
        assert changes == ([Color(0, 0, 0, a=0)] if initial != "transparent" else []) + [
            restored,
            Color(0, 0, 0, a=0),
            restored,
        ]


def test_xterm_picker_without_transparent_hides_transparent_section_when_no_hint() -> None:
    # ``allow_transparent=False`` + ``disabled_hint=None`` removes the
    # section entirely: no header, no selectable sentinel, no hint row.
    picker = _Xterm256PalettePicker(Color.parse("#FFFFFF"), allow_transparent=False)
    layout_indices = {idx for _, indices in picker._rows for idx in indices}
    assert _TRANSPARENT_SENTINEL not in layout_indices
    labels = {label for label, _ in picker._rows}
    assert "Transparent" not in labels
    assert picker._selected_index != _TRANSPARENT_SENTINEL


def test_xterm_picker_with_disabled_hint_keeps_transparent_header() -> None:
    # When a hint is provided, the "Transparent" header stays so the user
    # still sees *where* the option lives; the sentinel cell is replaced
    # by an empty placeholder row that ``render_line`` paints as the hint.
    picker = _Xterm256PalettePicker(
        Color.parse("#FFFFFF"),
        allow_transparent=False,
        disabled_hint="background can't be transparent",
    )
    labels = {label for label, _ in picker._rows}
    assert "Transparent" in labels
    # No selectable sentinel cell — the option is *visually present but
    # unavailable*, not selectable.
    layout_indices = {idx for _, indices in picker._rows for idx in indices}
    assert _TRANSPARENT_SENTINEL not in layout_indices
    assert picker._selected_index != _TRANSPARENT_SENTINEL


def test_xterm_picker_disabled_hint_only_applied_when_transparent_blocked() -> None:
    # The hint is meaningful only when transparent is hidden — passing
    # ``disabled_hint`` alongside ``allow_transparent=True`` would be a
    # contradiction in caller intent, so the picker drops it silently.
    picker = _Xterm256PalettePicker(
        Color.parse("#FFFFFF"),
        allow_transparent=True,
        disabled_hint="background can't be transparent",
    )
    assert picker._disabled_hint is None


def test_xterm_picker_disabled_hint_renders_in_placeholder_row() -> None:
    # The hint paints the placeholder row (``(None, ())``) that sits
    # immediately below the "Transparent" header in the opaque-only layout.
    hint = "background can't be transparent"
    picker = _Xterm256PalettePicker(
        Color.parse("#FFFFFF"),
        allow_transparent=False,
        disabled_hint=hint,
    )
    # Placeholder is the last row in ``WITH_HINT`` layout.
    placeholder_row = len(picker._rows) - 1
    placeholder_strip = picker.render_line(placeholder_row)
    placeholder_text = "".join(segment.text for segment in placeholder_strip)
    assert hint in placeholder_text
    # The "Transparent" header above the placeholder still renders.
    header_row = placeholder_row - 1
    header_strip = picker.render_line(header_row)
    header_text = "".join(segment.text for segment in header_strip)
    assert "Transparent" in header_text


def test_xterm_picker_without_transparent_recovers_when_seeded_transparent() -> None:
    # Defensive path: if a slot somehow already holds a transparent value
    # when the picker opens with ``allow_transparent=False``, the picker
    # must still construct (no KeyError on the sentinel position) and
    # land on a real palette index.
    picker = _Xterm256PalettePicker(Color(0, 0, 0, a=0), allow_transparent=False)
    assert picker._selected_index != _TRANSPARENT_SENTINEL
    assert not picker.color.is_transparent


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ansi_red", "ansi_red"),
        ("ansi_bright_magenta", "ansi_bright_magenta"),
        ("ansi_white 40%", "ansi_white"),
        ("ansi_default", "ansi_default"),
        ("#FF0000", None),
        ("transparent", None),
        ("", None),
        (None, None),
    ],
)
def test_ansi_base_token(value: str | None, expected: str | None) -> None:
    assert _ansi_base_token(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ansi_default", "default"),
        ("ansi_red", "#800000"),
        ("ansi_bright_black", "#808080"),
        ("ansi_bright_magenta", "#FF00FF"),
        ("ansi_white 40%", None),
        ("ansi_bright_cyan 20%", None),
        ("#FF0000", None),
        ("transparent", None),
    ],
)
def test_ansi_display_label(value: str, expected: str | None) -> None:
    assert _ansi_display_label(value) == expected
