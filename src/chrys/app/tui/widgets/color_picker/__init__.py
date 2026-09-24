# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""An independent color editor. The caller owns transactions and theme previews."""

from __future__ import annotations

import math

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.color import Color
from textual.containers import HorizontalGroup, VerticalGroup
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Label

from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.widgets.input import EnhancedInput as Input
from chrys.foundation.i18n import MessageDef, msg

from .controls import ColorComparison, ColorPlane
from .model import ColorEditContext, ColorValue

_EXPRESSION = msg("tui.color_picker.expression", fallback="Color expression / HEX")
_COMPARISON = msg("tui.color_picker.comparison", fallback="Original / Preview · checkerboard and actual background")
_HUE = msg("tui.color_picker.hue", fallback="Hue")
_ALPHA = msg("tui.color_picker.alpha", fallback="Opacity")
_INVALID = msg("tui.color_picker.invalid", fallback="Enter a valid color; the last valid preview is kept.")
_OPAQUE = msg("tui.color_picker.opaque", fallback="This field requires an opaque RGB color.")
_RANGE = msg("tui.color_picker.range", fallback="Use RGB 0-255, H 0-360, S/V/A 0-100.")


class ColorPicker(Widget):
    """RGB, HSV and alpha controls which preserve untouched source expressions."""

    DEFAULT_CSS = """
    ColorPicker {
        height: auto;
        & > HorizontalGroup { height: auto; }
        .planes { width: 2fr; padding-right: 2; height: auto; }
        .numbers { width: 1fr; min-width: 24; height: auto; padding: 1; background: $foreground 4%; }
        .channels { height: auto; margin-top: 1; }
        .channel { width: 1fr; height: auto; margin-right: 1; }
        .channel:last-child { margin-right: 0; }
        .channel Label { text-align: center; width: 1fr; }
        Label { color: $text-muted; }
        Input {
            width: 1fr;
            height: 1;
            border: none;
            background: $foreground 8%;
            padding: 0 1;
        }
        Input:focus { border: none; background: $foreground 12%; }
        ColorComparison { margin-bottom: 1; }
        .validation-error { color: $error; background: transparent; height: auto; min-height: 1; width: 1fr; }
        .caption { height: auto; width: 1fr; }
        &.compact > HorizontalGroup { layout: vertical; }
        &.compact .planes { width: 1fr; padding-right: 0; }
        &.compact .numbers { width: 1fr; }
        &.compact ColorPlane.sv { height: 7; }
    }
    """

    class Changed(Message):
        """Emitted only for user edits, never by mounting or synchronizing controls."""

        def __init__(self, picker: ColorPicker, value: str) -> None:
            super().__init__()
            self.picker = picker
            self.value = value
            self.context = picker.context

    class ValidityChanged(Message):
        """Let a containing dialog reflect invalid input in its confirm action."""

        def __init__(self, picker: ColorPicker) -> None:
            super().__init__()
            self.picker = picker
            self.context = picker.context
            self.valid = picker.valid

    def __init__(
        self,
        value: str,
        *,
        original: str | None = None,
        seed: Color | None = None,
        background: Color | None = None,
        allow_alpha: bool = True,
        context: ColorEditContext | None = None,
    ) -> None:
        super().__init__()
        self.state = ColorValue.from_expression(value, seed=seed)
        self.original = self.state if original is None else ColorValue.from_expression(original, seed=seed)
        self.background = background or Color(32, 32, 32)
        self.allow_alpha = allow_alpha
        self.valid = True
        self.context = context
        self._input_values: dict[str, str] = {}
        self._error_message: MessageDef | None = None

    @property
    def value(self) -> str:
        return self.state.expression

    def compose(self) -> ComposeResult:
        self._input_values = {
            "color-expression": self.value,
            **{f"channel-{key}": value for key, value in self._values().items()},
        }
        localizer = widget_localizer(self)
        yield Label(Text(render_str(localizer, _COMPARISON.bind())), id="color-comparison-label", classes="caption")
        comparison = ColorComparison(self.original.color, self.background)
        comparison.color = self.state.color
        yield comparison
        with HorizontalGroup():
            with VerticalGroup(classes="planes"):
                yield ColorPlane("sv", self.state, id="color-sv")
                yield Label(Text(render_str(localizer, _HUE.bind())), id="color-hue-label")
                yield ColorPlane("hue", self.state, id="color-hue")
                if self.allow_alpha:
                    yield Label(Text(render_str(localizer, _ALPHA.bind())), id="color-alpha-label")
                    yield ColorPlane("alpha", self.state, id="color-alpha")
            with VerticalGroup(classes="numbers"):
                yield Label(
                    Text(render_str(localizer, _EXPRESSION.bind())), id="color-expression-label", classes="caption"
                )
                yield Input(self.value, id="color-expression")
                groups = [("r", "g", "b"), ("h", "s", "v")]
                if self.allow_alpha:
                    groups.append(("a",))
                for names in groups:
                    with HorizontalGroup(classes="channels"):
                        for name in names:
                            with VerticalGroup(classes="channel"):
                                yield Label(Text(name.upper()))
                                yield Input(self._values()[name], id=f"channel-{name}")
        yield Label(Text(""), id="color-error", classes="validation-error")

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < 62, "compact")

    def refresh_localization(self) -> None:
        """Keep input text, invalid edits, focus and plane selection intact."""
        localizer = widget_localizer(self)
        for name, definition in (
            ("comparison", _COMPARISON),
            ("hue", _HUE),
            ("alpha", _ALPHA),
            ("expression", _EXPRESSION),
        ):
            if name != "alpha" or self.allow_alpha:
                self.query_one(f"#color-{name}-label", Label).update(Text(render_str(localizer, definition.bind())))
        self._show_error(self._error_message)

    def _show_error(self, message: MessageDef | None) -> None:
        self._error_message = message
        self.query_one("#color-error", Label).update(
            Text(render_str(widget_localizer(self), message.bind()) if message is not None else "")
        )

    def _values(self) -> dict[str, str]:
        r, g, b = self.state.color.rgb
        return {
            "r": str(r),
            "g": str(g),
            "b": str(b),
            "h": f"{self.state.hue * 360:.2f}".rstrip("0").rstrip(".") or "0",
            "s": f"{self.state.saturation * 100:.2f}".rstrip("0").rstrip(".") or "0",
            "v": f"{self.state.color.hsv.v * 100:.2f}".rstrip("0").rstrip(".") or "0",
            "a": f"{self.state.color.a * 100:.2f}".rstrip("0").rstrip(".") or "0",
        }

    def _publish(self, value: ColorValue, *, source: str | None = None) -> None:
        if not self.allow_alpha and (value.color.a != 1 or value.color.ansi is not None):
            self._set_valid(False)
            self._show_error(_OPAQUE)
            return
        old = self.state
        self.state = value
        self._set_valid(True)
        self._show_error(None)
        values = {"color-expression": value.expression, **{f"channel-{k}": v for k, v in self._values().items()}}
        with self.prevent(Input.Changed):
            for control in self.query(Input):
                if control.id != source:
                    control.value = values[control.id or ""]
                self._input_values[control.id or ""] = control.value
        for plane in self.query(ColorPlane):
            plane.set_value(value)
        comparison = self.query_one(ColorComparison)
        if comparison.color != value.color:
            comparison.color = value.color
            comparison.refresh()
        if old.expression != value.expression:
            self.post_message(self.Changed(self, value.expression))

    def _set_valid(self, valid: bool) -> None:
        if valid != self.valid:
            self.valid = valid
            self.post_message(self.ValidityChanged(self))

    @on(ColorPlane.Adjusted)
    def _plane_adjusted(self, event: ColorPlane.Adjusted) -> None:
        event.stop()
        self._publish(event.value)

    @on(Input.Changed)
    def _input_changed(self, event: Input.Changed) -> None:
        event.stop()
        if not self.is_mounted or event.value != event.input.value:
            return
        control_id = event.input.id or ""
        if self._input_values.get(control_id) == event.value:
            return
        self._input_values[control_id] = event.value
        value = None
        if control_id == "color-expression":
            value = self.state.with_expression(event.value)
        else:
            # Each channel changes independently; typing a partial number never
            # forces the other six fields through rounded display values.
            try:
                number = float(event.value)
            except ValueError:
                number = math.nan
            channel = control_id.removeprefix("channel-")
            limit = 255 if channel in {"r", "g", "b"} else 360 if channel == "h" else 100
            if math.isfinite(number) and 0 <= number <= limit:
                if channel in {"r", "g", "b"} and number.is_integer():
                    rgb = dict(zip(("r", "g", "b"), self.state.color.rgb, strict=True))
                    rgb[channel] = int(number)
                    value = self.state.with_rgb(rgb["r"], rgb["g"], rgb["b"], self.state.color.a)
                elif channel == "a":
                    value = self.state.with_alpha(number / 100)
                elif channel in {"h", "s", "v"}:
                    hsv = {"h": self.state.hue, "s": self.state.saturation, "v": self.state.color.hsv.v}
                    hsv[channel] = number / limit
                    value = self.state.with_hsv(hsv["h"], hsv["s"], hsv["v"])
        if value is None:
            self._set_valid(False)
            message = _INVALID if control_id == "color-expression" else _RANGE
            self._show_error(message)
        else:
            self._publish(value, source=control_id)

    def restore_original(self) -> None:
        """Restore the input exactly, including expressions and original precision."""
        self._publish(self.original)


__all__ = ["ColorEditContext", "ColorPicker", "ColorValue"]
