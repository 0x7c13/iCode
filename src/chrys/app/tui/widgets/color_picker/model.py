# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Lossless input expressions and explicitly edited RGB/HSV values."""

from __future__ import annotations

from dataclasses import dataclass

from textual.color import Color, ColorParseError


@dataclass(frozen=True)
class ColorEditContext:
    """Opaque caller correlation, carried unchanged on every edit message."""

    document: str
    field: str
    transaction: str


def parse_color(value: str) -> Color | None:
    """Return a concrete color, leaving automatic CSS expressions to the caller."""
    try:
        color = Color.parse(value)
    except ColorParseError:
        return None
    return None if color.auto else color


def _edited_expression(color: Color) -> str:
    if color.a == 1:
        return color.hex
    # Numeric opacity should not silently truncate to an 8-bit HEX alpha.
    alpha = f"{color.a:.16f}".rstrip("0").rstrip(".")
    return f"rgba({color.r}, {color.g}, {color.b}, {alpha})"


@dataclass(frozen=True)
class ColorValue:
    """Keep the original spelling until the user actually changes the color.

    Hue is independent of RGB so black and gray do not erase a hue selection.
    Previewing an expression never serializes it through an RGB approximation.
    """

    expression: str
    color: Color
    hue: float
    saturation: float

    @classmethod
    def from_expression(cls, expression: str, *, seed: Color | None = None) -> ColorValue:
        color = parse_color(expression) or seed or Color(255, 255, 255)
        hsv = color.hsv
        return cls(expression, color, hsv.h, hsv.s)

    def with_expression(self, expression: str) -> ColorValue | None:
        if expression == self.expression:
            return self
        color = parse_color(expression)
        if color is None:
            return None
        hsv = color.hsv
        return ColorValue(expression, color, hsv.h if hsv.s and hsv.v else self.hue, hsv.s)

    def with_rgb(self, red: int, green: int, blue: int, alpha: float) -> ColorValue:
        color = Color(red, green, blue, alpha).clamped
        if color == self.color:
            return self
        hsv = color.hsv
        return ColorValue(_edited_expression(color), color, hsv.h if hsv.s and hsv.v else self.hue, hsv.s)

    def with_hsv(self, hue: float, saturation: float, value: float) -> ColorValue:
        hue = max(0.0, min(1.0, hue))
        saturation = max(0.0, min(1.0, saturation))
        value = max(0.0, min(1.0, value))
        if (hue, saturation, value) == (self.hue, self.saturation, self.color.hsv.v):
            return self
        color = Color.from_hsv(hue, saturation, value).with_alpha(self.color.a)
        # Selecting a hue while black is meaningful, but isn't a color edit yet.
        expression = self.expression if color == self.color else _edited_expression(color)
        return ColorValue(expression, color, hue, saturation)

    def with_alpha(self, alpha: float) -> ColorValue:
        alpha = max(0.0, min(1.0, alpha))
        if alpha == self.color.a:
            return self
        color = self.color.with_alpha(alpha)
        return ColorValue(_edited_expression(color), color, self.hue, self.saturation)
