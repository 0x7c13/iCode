# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Color fidelity and real keyboard/mouse input in the independent picker."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.color import Color
from textual.containers import VerticalScroll
from textual.widgets import Input

from chrys.app.tui.widgets.color_picker import ColorEditContext, ColorPicker, ColorValue
from chrys.app.tui.widgets.color_picker.controls import ColorComparison, ColorPlane
from tests.support.waiting import wait_for


class PickerApp(App):
    def __init__(self, expression: str = "#aB123480", *, allow_alpha: bool = True) -> None:
        super().__init__()
        self.picker = ColorPicker(
            expression, allow_alpha=allow_alpha, context=ColorEditContext("document", "var:color", "transaction")
        )
        self.changes: list[ColorPicker.Changed] = []

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield self.picker

    def on_color_picker_changed(self, event: ColorPicker.Changed) -> None:
        self.changes.append(event)


@pytest.mark.parametrize("raw", ["#123", "#1234", "#aB1234", "#aB123480", "ansi_white 40%", "transparent"])
def test_source_spelling_and_alpha_survive_an_unchanged_value(raw: str) -> None:
    value = ColorValue.from_expression(raw)
    assert value.expression == raw
    assert value.with_expression(raw) == value


def test_hue_and_saturation_survive_black_and_gray() -> None:
    black = ColorValue.from_expression("#000000").with_hsv(0.5, 0.75, 0)
    assert black.expression == "#000000"
    cyan = black.with_hsv(black.hue, black.saturation, 1)
    assert cyan.color.g == cyan.color.b == 255
    assert cyan.color.r < 100
    gray = cyan.with_rgb(128, 128, 128, 1)
    assert gray.hue == 0.5


def test_unedited_rgb_does_not_round_trip_through_hsv() -> None:
    value = ColorValue.from_expression("#FFB86C")
    assert value.with_rgb(*value.color.rgb, value.color.a) is value
    assert value.with_alpha(0.5).color.rgb == (255, 184, 108)


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 0.9, 0.123456])
def test_numeric_opacity_round_trips_without_eight_bit_truncation(alpha: float) -> None:
    value = ColorValue.from_expression("#FFB86C").with_alpha(alpha)
    parsed = Color.parse(value.expression)
    assert parsed.a == pytest.approx(alpha)
    assert parsed.rgb == value.color.rgb


async def test_mount_resize_and_focusing_do_not_emit_edits() -> None:
    app = PickerApp()
    async with app.run_test(size=(100, 38)) as pilot:
        app.picker.query_one(Input).focus()
        await pilot.press("tab", "shift+tab")
        await pilot.resize_terminal(52, 24)
        # The App dispatches a resize from a debounce timer, which Pilot's
        # settle barrier does not wait for.
        await wait_for(lambda: app.picker.has_class("compact"), pilot=pilot, description="compact picker")
        assert app.changes == []
        assert app.picker.value == "#aB123480"


async def test_partial_input_keeps_last_value_and_valid_input_carries_context() -> None:
    app = PickerApp()
    async with app.run_test(size=(100, 38)) as pilot:
        field = app.picker.query_one("#color-expression", Input)
        field.focus()
        await pilot.pause()
        field.action_select_all()
        await pilot.press("#", "1", "2")
        await wait_for(lambda: not app.picker.valid, pilot=pilot)
        assert app.changes == []
        assert app.picker.value == "#aB123480"
        await pilot.press("3", "4")
        await wait_for(lambda: app.picker.value == "#1234", pilot=pilot)
        assert app.picker.state.color.a == pytest.approx(0x44 / 255)
        assert app.changes[-1].context == ColorEditContext("document", "var:color", "transaction")


async def test_channel_edit_preserves_other_channels_and_alpha() -> None:
    app = PickerApp("#FFB86C80")
    async with app.run_test(size=(100, 38)) as pilot:
        field = app.picker.query_one("#channel-r", Input)
        field.focus()
        await pilot.pause()
        field.action_select_all()
        await pilot.press("1", "2", "7")
        await wait_for(lambda: app.picker.state.color.r == 127, pilot=pilot)
        assert app.picker.state.color == Color(127, 184, 108, 128 / 255)


async def test_mouse_drag_outside_plane_clamps_and_releases_capture() -> None:
    app = PickerApp("#FF0000")
    async with app.run_test(size=(100, 38)) as pilot:
        plane = app.picker.query_one("#color-sv", ColorPlane)
        await pilot.mouse_down(plane, offset=(3, 3))
        await pilot.hover(plane, offset=(plane.size.width + 5, plane.size.height + 3))
        await pilot.mouse_up(plane, offset=(plane.size.width + 5, plane.size.height + 3))
        await pilot.pause()
        assert app.mouse_captured is None
        assert app.picker.state.saturation == 1
        assert app.picker.state.color.hsv.v == 0
        assert len(app.changes) >= 2


async def test_comparison_drag_does_not_select_or_copy_widget_name() -> None:
    app = PickerApp()
    async with app.run_test(size=(100, 38)) as pilot:
        comparison = app.picker.query_one(ColorComparison)
        app.screen.set_focus(None)
        app.copy_to_clipboard("previous clipboard")
        await pilot.mouse_down(comparison, offset=(2, 0))
        await pilot.hover(comparison, offset=(comparison.size.width - 2, 2))
        await pilot.mouse_up(comparison, offset=(comparison.size.width - 2, 2))
        assert not app.screen.selections
        assert not app.screen.get_selected_text()
        await pilot.press("super+c")
        assert app.clipboard == "previous clipboard"
        assert app.changes == []


async def test_keyboard_opacity_and_restore_preserve_rgb() -> None:
    app = PickerApp("#FFB86C")
    async with app.run_test(size=(100, 38)) as pilot:
        plane = app.picker.query_one("#color-alpha", ColorPlane)
        plane.focus()
        await pilot.press("shift+left")
        await wait_for(lambda: app.picker.state.color.a < 1, pilot=pilot)
        assert app.picker.state.color.rgb == (255, 184, 108)
        assert app.picker.state.color.a == pytest.approx(0.9)
        app.picker.restore_original()
        await pilot.pause()
        assert app.picker.value == "#FFB86C"


async def test_opaque_field_rejects_alpha_without_changing_preview() -> None:
    app = PickerApp("#112233", allow_alpha=False)
    async with app.run_test(size=(100, 38)) as pilot:
        app.picker.query_one("#color-expression", Input).value = "#1234"
        await wait_for(lambda: not app.picker.valid, pilot=pilot)
        assert app.picker.value == "#112233"
        assert not app.picker.query("#color-alpha")


async def test_one_cell_plane_renders_without_dividing_by_zero() -> None:
    app = PickerApp()
    async with app.run_test(size=(100, 38)) as pilot:
        plane = app.picker.query_one("#color-sv", ColorPlane)
        plane.styles.width = 3
        plane.styles.height = 3
        await pilot.pause()
        assert plane.content_size == (1, 1)
        assert plane.render_line(0).cell_length == 1
