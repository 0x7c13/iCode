# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A real Textual app around one `Terminal`, for tests of the widget.

Geometry, scrolling, focus and the pointer are the widget's contract with Textual, so the tests
drive them through a running app instead of patching regions onto a detached widget.
"""

from __future__ import annotations

from textual import events, on
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.message import Message
from textual.pilot import Pilot
from textual.selection import Selection
from textual.widget import Widget
from textual.widgets import Button

from chrys.app.tui.terminal.widget import Terminal

# The vertical scrollbar gutter the terminal keeps open at all times.
GUTTER_WIDTH = 2


class TerminalApp(App[None]):
    """One terminal filling the screen, above a one-line button that can take the focus away."""

    CSS = """
    Terminal { width: 100%; height: 1fr; }
    Button { height: 1; min-width: 4; border: none; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.terminal = Terminal()
        self.terminal_messages: list[Message] = []

    def compose(self) -> ComposeResult:
        yield self.terminal
        yield Button("x", id="elsewhere")

    @on(Terminal.SizeChanged)
    @on(Terminal.AlternateScreenChanged)
    @on(Terminal.EscapeExited)
    @on(Terminal.DirectoryChanged)
    @on(Terminal.CommandSubmitted)
    def _record(self, message: Message) -> None:
        self.terminal_messages.append(message)

    def messages_of[MessageT: Message](self, kind: type[MessageT]) -> list[MessageT]:
        return [message for message in self.terminal_messages if isinstance(message, kind)]


class Stdin:
    """Stands in for the program's input: records what the terminal sends it."""

    def __init__(self) -> None:
        self.writes: list[str] = []

    async def __call__(self, data: str) -> None:
        self.writes.append(data)

    @property
    def text(self) -> str:
        return "".join(self.writes)


def connect_stdin(terminal: Terminal) -> Stdin:
    stdin = Stdin()
    terminal.set_write_to_stdin(stdin)
    return stdin


def shown_lines(terminal: Terminal) -> list[str]:
    """What the widget draws, line by line, without the padding."""
    height = terminal.scrollable_content_region.height
    return [terminal.render_line(y).text.rstrip() for y in range(height)]


def select_text(terminal: Terminal, start: Offset, end: Offset) -> Selection:
    """Select from ``start`` to ``end`` in the terminal, the way a drag over it ends up doing."""
    selection = Selection(start, end)
    selections: dict[Widget, Selection] = {terminal: selection}
    terminal.screen.selections = selections
    return selection


def numbered_lines(count: int) -> str:
    """``count`` lines of output, each ended the way a program behind a PTY ends them."""
    return "".join(f"line {number}\r\n" for number in range(count))


async def post_mouse_event(
    pilot: Pilot[None],
    terminal: Terminal,
    kind: type[events.MouseEvent],
    x: int,
    y: int,
    *,
    button: int = 0,
    shift: bool = False,
    meta: bool = False,
    ctrl: bool = False,
) -> None:
    """Deliver a pointer event at cell ``(x, y)`` of the terminal, the way the driver would.

    Pilot has no wheel, no motion with a button held and no release of a named button. The event
    enters at the app, in screen coordinates, so capture and forwarding work as they do live; the
    cell may lie outside the terminal, as it does when a drag leaves it.
    """
    origin = terminal.region.offset
    screen_x, screen_y = origin.x + x, origin.y + y
    pilot.app.post_message(
        kind(
            None,
            screen_x,
            screen_y,
            0,
            0,
            button,
            shift,
            meta,
            ctrl,
            screen_x=screen_x,
            screen_y=screen_y,
        )
    )
    await pilot.pause()
