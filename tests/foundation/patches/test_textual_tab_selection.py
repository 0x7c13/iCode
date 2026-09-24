# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the Textual tab selection offset patch."""

from __future__ import annotations

from rich.text import Text
from textual.app import App, ComposeResult
from textual.content import Content
from textual.geometry import Offset
from textual.selection import Selection
from textual.style import Style
from textual.visual import RenderOptions
from textual.widgets import Static

from chrys.foundation.patches.textual_tab_selection import (
    _RUNTIME_PATCH_MARKER,
    _RUNTIME_PATCH_TEXTUAL_VERSION,
    _expand_tabs_with_offsets,
    _pad_source_offsets,
    _runtime_compositor_patch_compatible,
    apply_runtime_patch,
)


class _TabStaticApp(App):
    CSS = """
    Static {
        width: 20;
        height: 1;
        padding: 0;
        margin: 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(Text("A\tB"))


class _LinePadTabStaticApp(App):
    CSS = """
    Static {
        width: 20;
        height: 1;
        padding: 0;
        margin: 0;
        line-pad: 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("A\tB")


class _PaddedStaticApp(App):
    CSS = """
    Static {
        width: 20;
        height: 1;
        padding-left: 2;
        padding-right: 0;
        margin: 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("abc")


async def test_runtime_patch_maps_expanded_tab_cells_to_source_offsets() -> None:
    """Textual Static selection extracts from source text, not rendered text."""
    apply_runtime_patch()
    app = _TabStaticApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        widget = app.query_one(Static)
        origin = widget.region.offset

        offsets = [app.screen.get_widget_and_offset_at(origin.x + cell, origin.y)[1] for cell in (0, 1, 7, 8, 9)]

        assert offsets == [
            Offset(0, 0),
            Offset(1, 0),
            Offset(1, 0),
            Offset(2, 0),
            Offset(3, 0),
        ]
        app.screen.selections = {widget: Selection(Offset(2, 0), Offset(3, 0))}
        assert app.screen.get_selected_text() == "B"


async def test_runtime_patch_keeps_tab_offsets_with_line_pad() -> None:
    apply_runtime_patch()
    app = _LinePadTabStaticApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        widget = app.query_one(Static)
        origin = widget.region.offset

        offsets = [app.screen.get_widget_and_offset_at(origin.x + cell, origin.y)[1] for cell in (0, 2, 9, 10, 11)]

        assert offsets == [
            Offset(0, 0),
            Offset(0, 0),
            Offset(1, 0),
            Offset(2, 0),
            Offset(3, 0),
        ]
        app.screen.selections = {widget: Selection(Offset(2, 0), Offset(3, 0))}
        assert app.screen.get_selected_text() == "B"


async def test_runtime_patch_preserves_left_padding_selection_start() -> None:
    """Textual 8.2.7 maps selection from left padding to the line start."""
    apply_runtime_patch()
    app = _PaddedStaticApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        widget = app.query_one(Static)
        origin = widget.region.offset

        assert app.screen.get_widget_and_offset_at(origin.x, origin.y)[1] == Offset(0, 0)


def test_plain_tab_expansion_preserves_textual_unstyled_spacing() -> None:
    apply_runtime_patch()

    expanded, offsets = _expand_tabs_with_offsets(Content("漢\tB"), 8)

    assert expanded.plain == "漢\tB".expandtabs(8)
    assert offsets == (0, 1, 1, 1, 1, 1, 1, 1, 2, 3)


def test_styled_tab_expansion_matches_textual_unstyled_spacing() -> None:
    apply_runtime_patch()

    expanded, offsets = _expand_tabs_with_offsets(Content("样例\t文本").stylize(Style.parse("bold")), 8)

    assert expanded.plain == "样例\t文本".expandtabs(8)
    assert offsets == (0, 1, 2, 2, 2, 2, 2, 2, 3, 4, 5)


async def test_static_text_tab_rendering_does_not_shift_when_selected() -> None:
    apply_runtime_patch()

    class _CjkTabStaticApp(App):
        CSS = """
        Static {
            width: 40;
            height: 1;
            padding: 0;
            margin: 0;
        }
        """

        def compose(self) -> ComposeResult:
            yield Static(Text("样例\t文本"))

    def row_text(widget: Static) -> str:
        return "".join(segment.text for segment in widget.render_line(0)._segments).rstrip()

    app = _CjkTabStaticApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        widget = app.query_one(Static)
        before = row_text(widget)

        app.screen.selections = {widget: Selection(Offset(0, 0), Offset(5, 0))}
        await pilot.pause()
        after = row_text(widget)

    assert before == "样例\t文本".expandtabs(8)
    assert after == before


def test_pad_source_offsets_accepts_empty_maps() -> None:
    assert _pad_source_offsets((), 2, 2) == ()


def test_runtime_compositor_guard_accepts_pinned_textual_version() -> None:
    assert _runtime_compositor_patch_compatible()


def test_runtime_compositor_guard_target_matches_installed_textual() -> None:
    """The guard constant must track the Textual version chrys actually pins."""
    import textual

    assert textual.__version__ == _RUNTIME_PATCH_TEXTUAL_VERSION


def test_runtime_compositor_guard_rejects_mismatched_textual_version(monkeypatch) -> None:
    import textual

    monkeypatch.setattr(textual, "__version__", "0.0.0-unexpected")

    assert not _runtime_compositor_patch_compatible()


def test_runtime_patch_version_gate_precedes_private_api_access(monkeypatch, caplog) -> None:
    import textual
    import textual._compositor as compositor_mod

    monkeypatch.setattr(textual, "__version__", "9.0.0")
    monkeypatch.delattr(compositor_mod, "Compositor")

    apply_runtime_patch()

    assert "loaded Textual is not the pinned 8.2.7" in caplog.text


def test_runtime_patch_repairs_partial_marker_state(monkeypatch) -> None:
    apply_runtime_patch()

    import textual.content as content_mod
    import textual.style as style_mod

    original = style_mod.Style.rich_style_with_offset

    def unmarked_rich_style_with_offset(self, x, y):
        return original(self, x, y)

    monkeypatch.setattr(style_mod.Style, "rich_style_with_offset", unmarked_rich_style_with_offset)
    setattr(content_mod.Content, _RUNTIME_PATCH_MARKER, True)

    apply_runtime_patch()

    patched = style_mod.Style.rich_style_with_offset
    assert getattr(patched, _RUNTIME_PATCH_MARKER, False)
    assert style_mod.Style().rich_style_with_offset(0, 0, offset_map=(0,)).meta["offset_map"] == (0,)


def test_apply_offsets_rebases_source_maps_to_rendered_offsets() -> None:
    """Chrys line-API widgets extract selections from their rendered strips."""
    apply_runtime_patch()
    strip = Content("A\tB").render_strips(
        20,
        None,
        Style(),
        RenderOptions(Style.parse, {}),
    )[0]

    rebased = strip.apply_offsets(2, 4)
    first_style = rebased._segments[0].style

    assert first_style is not None
    assert first_style.meta["offset"] == (2, 4)
    assert first_style.meta["offset_map"] == tuple(range(2, 12))
