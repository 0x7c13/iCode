# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Native horizontal graph navigation placed below the workflow actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import events
from textual.message import Message
from textual.scrollbar import ScrollBar, ScrollBarRender, ScrollMessage

from chrys.app.tui.widgets.workflow.graph import WorkflowGraph

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult


class _HiddenScrollBarRender(ScrollBarRender):
    """Leave native scrollbar space blank, including with terminal ANSI colors."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        return ()


class WorkflowScrollBar(ScrollBar):
    """Mirror the viewport's native scrollbar, keeping one owner of scroll state."""

    DEFAULT_CSS = """
    WorkflowScrollBar { display: none; width: 1fr; height: 1; margin-top: 1; }
    WorkflowScrollBar.-overflow { display: block; }
    WorkflowScrollBar.-vertical-scrollbar { margin: 1 2 0 0; }
    WorkflowPanel.-empty WorkflowScrollBar { display: none; }
    """

    def __init__(self, graph: WorkflowGraph) -> None:
        self.graph = graph
        super().__init__(vertical=False)

    def on_mount(self) -> None:
        native = self.graph.horizontal_scrollbar
        # Keep native geometry so Textual continues to own sizing and clamping.
        # Its reserved row becomes the space between the canvas and the actions.
        # System scrollbar painting bypasses visibility. Suppress its renderer
        # rather than blending reverse-video ink into ANSI default colors.
        # Visibility still excludes the original track from mouse hit testing.
        native.styles.visibility = "hidden"
        native.renderer = _HiddenScrollBarRender  # ty: ignore[invalid-attribute-access]  # Textual explicitly supports per-instance renderers.
        self.watch(native, "window_virtual_size", self._sync_virtual_size)
        self.watch(native, "window_size", self._sync_window_size)
        self.watch(native, "position", self._sync_position)
        self.watch(self.graph, "show_horizontal_scrollbar", self._sync_visibility)
        self.watch(self.graph, "show_vertical_scrollbar", self._sync_vertical_scrollbar)

    def _sync_virtual_size(self, value: int) -> None:
        self.window_virtual_size = value

    def _sync_window_size(self, value: int) -> None:
        self.window_size = value

    def _sync_position(self, value: float) -> None:
        self.position = value

    def _sync_visibility(self, visible: bool) -> None:
        self.set_class(visible, "-overflow")

    def _sync_vertical_scrollbar(self, visible: bool) -> None:
        self.set_class(visible, "-vertical-scrollbar")

    def post_message(self, message: Message) -> bool:
        if isinstance(
            message,
            (
                ScrollMessage,
                events.MouseScrollDown,
                events.MouseScrollUp,
                events.MouseScrollLeft,
                events.MouseScrollRight,
            ),
        ):
            return self.graph.post_message(message)
        return super().post_message(message)
