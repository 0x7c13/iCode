# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Keyboard and mouse surfaces for the Chrys color picker."""

from __future__ import annotations

from typing import ClassVar, Literal

from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.binding import Binding
from textual.color import Color
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget

from .model import ColorValue

Axis = Literal["sv", "hue", "alpha"]


class ColorPlane(Widget, can_focus=True):
    """A focusable color surface with bounded dragging, including outside its bounds."""

    ALLOW_SELECT = False
    DEFAULT_CSS = """
    ColorPlane {
        height: 3;
        border: blank $tui-border-foreground;
        &:focus { border: solid $tui-border-foreground; }
        &.sv { height: 12; }
    }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("left", "step(-1, 0, 1)", show=False),
        Binding("right", "step(1, 0, 1)", show=False),
        Binding("up", "step(0, -1, 1)", show=False),
        Binding("down", "step(0, 1, 1)", show=False),
        Binding("shift+left", "step(-1, 0, 10)", show=False),
        Binding("shift+right", "step(1, 0, 10)", show=False),
        Binding("shift+up", "step(0, -1, 10)", show=False),
        Binding("shift+down", "step(0, 1, 10)", show=False),
        Binding("home", "edge(0)", show=False),
        Binding("end", "edge(1)", show=False),
    ]

    class Adjusted(Message):
        """A user edit; rendering and initialization never emit this message."""

        def __init__(self, plane: ColorPlane, value: ColorValue) -> None:
            super().__init__()
            self.plane = plane
            self.value = value

    def __init__(self, axis: Axis, value: ColorValue, *, id: str) -> None:
        super().__init__(id=id, classes=axis)
        self.axis = axis
        self.value = value
        self._dragging = False

    def set_value(self, value: ColorValue) -> None:
        if value != self.value:
            self.value = value
            self.refresh()

    def _coordinates(self) -> tuple[float, float]:
        if self.axis == "sv":
            return self.value.saturation, 1 - self.value.color.hsv.v
        return (self.value.hue if self.axis == "hue" else self.value.color.a), 0.5

    def _edit(self, x: float, y: float) -> None:
        x, y = max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
        if self.axis == "sv":
            value = self.value.with_hsv(self.value.hue, x, 1 - y)
        elif self.axis == "hue":
            value = self.value.with_hsv(x, self.value.saturation, self.value.color.hsv.v)
        else:
            value = self.value.with_alpha(x)
        if value != self.value:
            self.set_value(value)
            self.post_message(self.Adjusted(self, value))

    def action_step(self, dx: int, dy: int, multiplier: int) -> None:
        x, y = self._coordinates()
        if self.axis == "sv":
            self._edit(x + dx * multiplier / 100, y + dy * multiplier / 100)
        else:
            self._edit(x + (dx - dy) * multiplier / (360 if self.axis == "hue" else 100), y)

    def action_edge(self, edge: int) -> None:
        self._edit(edge, self._coordinates()[1])

    def _edit_at_mouse(self, event: events.MouseEvent) -> None:
        region = self.content_region
        self._edit(
            (event.screen_x - region.x) / max(1, region.width - 1),
            (event.screen_y - region.y) / max(1, region.height - 1),
        )

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1:
            return
        event.stop()
        event.prevent_default()
        self.focus()
        self._dragging = True
        self.capture_mouse()
        self._edit_at_mouse(event)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._dragging:
            event.stop()
            self._edit_at_mouse(event)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._dragging:
            event.stop()
            self._edit_at_mouse(event)
            self._dragging = False
            self.release_mouse()

    def on_blur(self) -> None:
        if self._dragging:
            self._dragging = False
            self.release_mouse()

    def on_unmount(self) -> None:
        self.on_blur()

    def render_line(self, y: int) -> Strip:
        width, height = self.content_size
        point_x, point_y = self._coordinates()
        cursor = round(point_x * max(0, width - 1)), round(point_y * max(0, height - 1))
        segments: list[Segment] = []
        for x in range(width):
            fraction = x / max(1, width - 1)
            if self.axis == "sv":
                color = Color.from_hsv(self.value.hue, fraction, 1 - y / max(1, height - 1))
            elif self.axis == "hue":
                color = Color.from_hsv(fraction, 1, 1)
            else:
                tile = Color(190, 190, 190) if (x // 2 + y) % 2 else Color(90, 90, 90)
                color = tile.blend(self.value.color.with_alpha(1), fraction)
            style = Style(color=color.get_contrast_text().with_alpha(1).rich_color, bgcolor=color.rich_color)
            segments.append(Segment("◆" if (x, y) == cursor else " ", style))
        return Strip(segments)


class ColorComparison(Widget):
    """Original and candidate over checkerboard and the caller's real background."""

    ALLOW_SELECT = False
    DEFAULT_CSS = "ColorComparison { height: 3; }"

    def __init__(self, original: Color, background: Color) -> None:
        super().__init__()
        self.original = original
        self.color = original
        self.background = background

    def render_line(self, y: int) -> Strip:
        width = self.content_size.width
        segments: list[Segment] = []
        for x in range(width):
            color = self.original if x < width // 2 else self.color
            tile = Color(180, 180, 180) if (x // 2 + y) % 2 else Color(80, 80, 80)
            background = tile if y < 2 else self.background
            composed = background.blend(color.with_alpha(1), color.a)
            segments.append(Segment(" ", Style(bgcolor=composed.rich_color)))
        return Strip(segments)
