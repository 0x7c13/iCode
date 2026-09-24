# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The terminal itself: what each control does to the screen, and what the terminal says back.

`TerminalEmulator` is a display-less VT220-class terminal with the xterm extensions programs rely
on today. Feed it a program's output and it keeps two screens of styled cells current; read the
active `ScreenBuffer` to draw them. It owns no I/O: replies the protocol owes the program come back
from `feed` for the caller to deliver.

Sequences are named here as the standards name them (ECMA-48, the DEC VT510 manual, xterm's
ctlseqs); each handler states the mnemonic it implements.
"""

from __future__ import annotations

import base64
import binascii
import re
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Final, NamedTuple
from urllib.parse import unquote, urlsplit

from rich.cells import cell_len, get_character_cell_size

from chrys.app.tui.terminal.emulator.buffer import ScreenBuffer
from chrys.app.tui.terminal.emulator.charsets import Charsets, CharsetSnapshot
from chrys.app.tui.terminal.emulator.parser import SequenceParser
from chrys.app.tui.terminal.emulator.pen import DEFAULT_PEN, Pen, apply_sgr

DEFAULT_HISTORY_LIMIT: Final = 10_000

_ZERO_WIDTH_JOINER: Final = "\u200d"
_TAB_INTERVAL: Final = 8
_MAX_PARAMETER: Final = 0xFFFF

# DEC private modes that are plain switches. The alternate screen and the mouse modes are not:
# several numbers drive one piece of state, so they are handled by name below.
_APPLICATION_CURSOR: Final = 1
_ORIGIN: Final = 6
_AUTOWRAP: Final = 7
_CURSOR_BLINK: Final = 12
_CURSOR_VISIBLE: Final = 25
_APPLICATION_KEYPAD: Final = 66
_FOCUS_REPORTING: Final = 1004
_ALTERNATE_SCROLL: Final = 1007
_BRACKETED_PASTE: Final = 2004
_SWITCHES: Final = frozenset(
    {
        _APPLICATION_CURSOR,
        _ORIGIN,
        _AUTOWRAP,
        _CURSOR_BLINK,
        _CURSOR_VISIBLE,
        _APPLICATION_KEYPAD,
        _FOCUS_REPORTING,
        _ALTERNATE_SCROLL,
        _BRACKETED_PASTE,
    }
)
# xterm leaves alternate scroll off, but every terminal people use today turns the wheel into
# arrow keys on the alternate screen unless told not to, and pagers are read with that in hand.
_SWITCHES_ON_AT_RESET: Final = frozenset({_AUTOWRAP, _CURSOR_VISIBLE, _ALTERNATE_SCROLL})

_INSERT: Final = 4
_NEWLINE: Final = 20

# 47 and 1047 only switch screens; 1049 also saves the cursor going in and restores it coming out.
_ALTERNATE_SCREENS: Final = frozenset({47, 1047, 1049})
_ALTERNATE_SCREEN_SAVING_CURSOR: Final = 1049

# Of the kitty keyboard protocol's progressive enhancements only "disambiguate escape codes" is
# offered. A program reads back which flags took, so asking for more degrades cleanly.
_KITTY_DISAMBIGUATE: Final = 1
_KITTY_STACK_LIMIT: Final = 16
_XTERM_MODIFY_OTHER_KEYS: Final = 4

_DRIVE_PATH = re.compile(r"/[A-Za-z]:(/|$)")


class MouseTracking(Enum):
    """Which pointer events the program asked to hear about. Values are the DEC private modes."""

    OFF = 0
    PRESS = 9
    PRESS_RELEASE = 1000
    DRAG = 1002
    MOTION = 1003


class MouseEncoding(Enum):
    """How pointer reports are spelled. Values are the DEC private modes."""

    DEFAULT = 0
    UTF8 = 1005
    SGR = 1006
    URXVT = 1015


class KeyProtocol(Enum):
    """How keys the classic encoding cannot tell apart are spelled, once a program asks."""

    LEGACY = "legacy"
    MODIFY_OTHER_KEYS_1 = "modify_other_keys_1"
    """xterm's ``CSI 27 ; modifiers ; code ~``, switched on with XTMODKEYS, for the keys that have
    no classic spelling of their own. A key with a well-known one keeps it, less a modifier or not."""
    MODIFY_OTHER_KEYS_2 = "modify_other_keys_2"
    """The same spelling for every modified key, those with a well-known one included."""
    KITTY = "kitty"
    """The kitty keyboard protocol's ``CSI code ; modifiers u``."""


class CursorShape(Enum):
    BLOCK = "block"
    UNDERLINE = "underline"
    BAR = "bar"


# By XTMODKEYS level. Level 3 also spells out keys held with nothing, which no program here has been
# seen to want; it is answered with the most there is.
_MODIFY_OTHER_KEYS_LEVELS: Final = (
    KeyProtocol.LEGACY,
    KeyProtocol.MODIFY_OTHER_KEYS_1,
    KeyProtocol.MODIFY_OTHER_KEYS_2,
)
_TRACKING_MODES: Final = {tracking.value: tracking for tracking in MouseTracking if tracking.value}
_ENCODING_MODES: Final = {encoding.value: encoding for encoding in MouseEncoding if encoding.value}
_CURSOR_SHAPES: Final = (
    CursorShape.BLOCK,
    CursorShape.BLOCK,
    CursorShape.BLOCK,
    CursorShape.UNDERLINE,
    CursorShape.UNDERLINE,
    CursorShape.BAR,
    CursorShape.BAR,
)
"""By DECSCUSR parameter. Each shape comes blinking and steady, and zero is the default."""
_STEADY_CURSOR_PARAMETERS: Final = {CursorShape.BLOCK: 2, CursorShape.UNDERLINE: 4, CursorShape.BAR: 6}
"""What DECRQSS reports: the cursor is drawn without blinking, whichever variant was asked for."""


@dataclass(frozen=True, slots=True)
class DirectoryChanged:
    """The shell reported its working directory."""

    path: str


@dataclass(frozen=True, slots=True)
class CommandSubmitted:
    """The shell reported the command line it is about to run."""

    command: str


type TerminalEvent = DirectoryChanged | CommandSubmitted


@dataclass(frozen=True, slots=True)
class Update:
    """What one `TerminalEmulator.feed` changed."""

    damaged: set[int] | None
    """Rows of the active buffer to redraw, as indices into its ``rows``; ``None`` means all."""
    trimmed: int
    """History rows dropped from the front of the active buffer: every later row moved up by this."""
    replies: str
    """Bytes the protocol owes the program (status reports and the like), to write to its input."""
    events: tuple[TerminalEvent, ...]


class _SavedCursor(NamedTuple):
    x: int
    y: int
    pending_wrap: bool
    pen: Pen
    charsets: CharsetSnapshot
    origin: bool


@lru_cache(maxsize=1024)
def _parse_parameters(parameters: str) -> tuple[int, ...]:
    """Numeric parameters, omitted ones as zero. Sub-parameters mean nothing outside SGR."""
    return tuple(min(int(field.partition(":")[0] or 0), _MAX_PARAMETER) for field in parameters.split(";"))


def _awaits_joined_character(cluster: str) -> bool:
    """Whether the cluster ends in a zero-width joiner that has not had its character yet.

    A joiner takes the one character after it, whatever that is, and a joiner taken that way is
    spent: it joins nothing itself. That is how `rich.cells` measures, nonsense included.
    """
    return (len(cluster) - len(cluster.rstrip(_ZERO_WIDTH_JOINER))) % 2 == 1


def _decode_payload(payload: str) -> str | None:
    """The text of a shell-integration report, which may travel as ``b64:`` to survive any content."""
    if not payload.startswith("b64:"):
        return payload
    try:
        return base64.b64decode(payload[4:], validate=True).decode("utf-8", "replace")
    except binascii.Error, ValueError:
        return None


class TerminalEmulator:
    """A terminal without a display."""

    def __init__(self, columns: int = 80, lines: int = 24, *, history_limit: int = DEFAULT_HISTORY_LIMIT) -> None:
        self.columns = columns
        self.lines = lines
        self._history_limit = history_limit
        self._parser = SequenceParser(self)
        self._replies: list[str] = []
        self._events: list[TerminalEvent] = []
        self._history_dropped = 0
        """History rows the program threw away (ED 3, RIS) that no `Update` has reported yet."""
        self._control_sequences = self._control_sequence_table()
        self._escapes = self._escape_table()
        self._commands = self._operating_system_command_table()
        self._reset()

    def _reset(self) -> None:
        self._primary = ScreenBuffer(self.columns, self.lines, history_limit=self._history_limit)
        self._alternate = ScreenBuffer(self.columns, self.lines, history_limit=0)
        self.buffer = self._primary
        self.pen = DEFAULT_PEN
        self.cursor_shape = CursorShape.BLOCK
        self.mouse_tracking = MouseTracking.OFF
        self.mouse_encoding = MouseEncoding.DEFAULT
        self._charsets = Charsets()
        self._switches = set(_SWITCHES_ON_AT_RESET)
        self._insert = False
        self._newline = False
        self._tab_stops = set(range(_TAB_INTERVAL, self.columns, _TAB_INTERVAL))
        self._saved_cursors: dict[bool, _SavedCursor] = {}
        self._modify_other_keys = 0
        # One stack per screen, so a full-screen program that dies without popping its flags
        # does not leave the shell behind it reading keys it cannot parse.
        self._kitty_flags: dict[bool, list[int]] = {False: [0], True: [0]}
        self._last_printed = ""
        self._anchor_pending = False

    # -- state a front end reads -------------------------------------------------------------

    @property
    def alternate_screen(self) -> bool:
        return self.buffer is self._alternate

    @property
    def cursor_visible(self) -> bool:
        return _CURSOR_VISIBLE in self._switches

    @property
    def application_cursor_keys(self) -> bool:
        return _APPLICATION_CURSOR in self._switches

    @property
    def bracketed_paste(self) -> bool:
        return _BRACKETED_PASTE in self._switches

    @property
    def focus_reporting(self) -> bool:
        return _FOCUS_REPORTING in self._switches

    @property
    def alternate_scroll(self) -> bool:
        return _ALTERNATE_SCROLL in self._switches

    @property
    def key_protocol(self) -> KeyProtocol:
        if self._kitty_flags[self.alternate_screen][-1] & _KITTY_DISAMBIGUATE:
            return KeyProtocol.KITTY
        return _MODIFY_OTHER_KEYS_LEVELS[min(self._modify_other_keys, len(_MODIFY_OTHER_KEYS_LEVELS) - 1)]

    # -- driving -----------------------------------------------------------------------------

    def feed(self, text: str) -> Update:
        """Interpret the next piece of the program's output."""
        buffer = self.buffer
        cursor_before = (buffer.cursor_index, buffer.cursor.x, self.cursor_visible, self.cursor_shape)
        self._parser.feed(text)
        trimmed = 0
        if self.buffer is self._primary:
            # What the limit takes now and what the program threw away itself. Both wait for the
            # primary screen to be the one on show: it is its rows that a front end numbers.
            trimmed = self._history_dropped + self._primary.trim_history()
            self._history_dropped = 0
        if self.buffer is buffer:
            cursor_after = (buffer.cursor_index, buffer.cursor.x, self.cursor_visible, self.cursor_shape)
            if cursor_after != cursor_before:
                # The cursor is drawn with its row, so moving it dirties the row it left too.
                buffer.damage(cursor_before[0])
                buffer.damage(cursor_after[0])
        update = Update(self.buffer.take_damage(), trimmed, "".join(self._replies), tuple(self._events))
        self._replies.clear()
        self._events.clear()
        return update

    def resize(self, columns: int, lines: int, *, reflow: bool = True, host_repaints: bool = False) -> None:
        """Change the screen size.

        Args:
            columns: New width.
            lines: New height.
            reflow: Rewrap the primary screen and its history to the new width.
            host_repaints: Whoever feeds us keeps a screen of its own, reflows that, and repaints
                ours with absolute positions afterwards (Windows ConPTY does). The screen origin
                then stays where that host believes it is, and the first position it sends is
                taken as the truth about which row the cursor is on.
        """
        self._tab_stops.update(range(-(-self.columns // _TAB_INTERVAL) * _TAB_INTERVAL, columns, _TAB_INTERVAL))
        self.columns, self.lines = columns, lines
        primary = self._primary
        cursor = primary.cursor
        saved = self._saved_cursors.get(False)
        if saved is not None and (saved.x, saved.y) != (cursor.x, cursor.y):
            saved = None
        primary.resize(columns, lines, reflow=reflow, pull_history=not host_repaints)
        if saved is not None:
            # Saved where the cursor is, as entering the alternate screen saves it: it goes where
            # the cursor went, or leaving that screen would put the shell back on the wrong row.
            self._saved_cursors[False] = saved._replace(x=cursor.x, y=cursor.y, pending_wrap=cursor.pending_wrap)
        self._alternate.resize(columns, lines, reflow=False, pull_history=False)
        if host_repaints:
            if reflow and primary.used_height <= lines:
                # Everything fits on one screen, which is how the host lays it out too.
                primary.anchor_cursor_row(primary.cursor_index)
            self._anchor_pending = not self.alternate_screen

    # -- SequenceHandler ---------------------------------------------------------------------

    def print(self, text: str) -> None:
        text = self._charsets.translate(text)
        self._anchor_pending = False
        if text.isascii() and not _awaits_joined_character(self._cell_behind_cursor()[1]):
            self._last_printed = text[-1]
            self._write(text)
        elif cells := self._cells(text):
            self._last_printed = next(cell for cell in reversed(cells) if cell)
            self._write(cells)

    def execute(self, control: str) -> None:
        cursor = self.buffer.cursor
        if control == "\n" or control in "\v\f":
            self._line_feed()
        elif control == "\r":
            cursor.x = 0
            cursor.pending_wrap = False
        elif control == "\b":
            cursor.x = max(cursor.x - 1, 0)
            cursor.pending_wrap = False
        elif control == "\t":
            self._tab_forward(1)
        elif control == "\x0e":
            self._charsets.lock(1)
        elif control == "\x0f":
            self._charsets.lock(0)
        # BEL and the rest have no effect here: the bell belongs to the outer terminal, and
        # ringing it for an embedded program would badge the whole application.

    def escape(self, intermediates: str, final: str) -> None:
        if intermediates and intermediates[0] in "()*+":
            self._charsets.designate("()*+".index(intermediates[0]), intermediates[1:] + final)
        elif (handler := self._escapes.get(intermediates + final)) is not None:
            handler()

    def control_sequence(self, marker: str, parameters: str, intermediates: str, final: str) -> None:
        if final == "m" and not marker and not intermediates:
            self.pen = apply_sgr(self.pen, parameters)
        elif (handler := self._control_sequences.get(marker + intermediates + final)) is not None:
            handler(_parse_parameters(parameters))

    def operating_system_command(self, payload: str) -> None:
        command, _, argument = payload.partition(";")
        if (handler := self._commands.get(command)) is not None:
            handler(argument)

    def device_control(self, payload: str) -> None:
        """DECRQSS, the one device control string programs send to probe a terminal."""
        if not payload.startswith("$q"):
            return
        setting = payload[2:]
        if setting == "r":
            report = f"{self.buffer.margin_top + 1};{self.buffer.margin_bottom + 1}r"
        elif setting == " q":
            report = f"{_STEADY_CURSOR_PARAMETERS[self.cursor_shape]} q"
        else:
            self._replies.append("\x1bP0$r\x1b\\")
            return
        self._replies.append(f"\x1bP1$r{report}\x1b\\")

    # -- printing ----------------------------------------------------------------------------

    def _cells(self, text: str) -> list[str]:
        """Split text into cells the way the renderer will measure it.

        A zero-width character joins the cell before it, a character after a zero-width joiner
        joins too, and a variation selector may widen its base: exactly the rules `rich.cells`
        measures by, so a row's cell count always equals the rendered width of its text.
        """
        cells: list[str] = []
        for character in text:
            base = len(cells) - 1 if cells and cells[-1] else len(cells) - 2
            if base < 0:
                if self._joins_cell_behind_cursor(character):
                    continue
                width = get_character_cell_size(character)
            elif _awaits_joined_character(cells[base]):
                cells[base] += character
                continue
            elif not (width := get_character_cell_size(character)):
                cells[base] += character
                if base == len(cells) - 1 and cell_len(cells[base]) == 2:
                    cells.append("")
                continue
            cells.append(character)
            if width == 2:
                cells.append("")
        return cells

    def _joins_cell_behind_cursor(self, character: str) -> bool:
        """Whether a character opening a run belongs to the cell already written before the cursor.

        That is the case for a zero-width character, and for anything following a zero-width
        joiner, when the two halves of a sequence arrive in separate runs.
        """
        column, behind = self._cell_behind_cursor()
        if get_character_cell_size(character) and not _awaits_joined_character(behind):
            return False
        if behind:
            # The cluster is what was printed last now, and what REP repeats.
            self._last_printed = behind + character
            self._rewrite_cell(column, self._last_printed)
        return True

    def _cell_behind_cursor(self) -> tuple[int, str]:
        """The column and the text of the cell before the cursor, ``""`` where there is none.

        With the last column just written the cursor is past it in all but position, so that
        column is the one behind it, whether or not the next character is going to wrap.
        """
        cursor = self.buffer.cursor
        cells = self.buffer.row(cursor.y).cells
        column = cursor.x if cursor.pending_wrap else cursor.x - 1
        if 0 < column < len(cells) and not cells[column]:
            column -= 1
        return column, cells[column] if 0 <= column < len(cells) else ""

    def _rewrite_cell(self, column: int, cluster: str) -> None:
        """Put ``cluster`` in the cursor's row, where the cell at ``column`` holds its beginning.

        A variation selector makes a narrow character wide. What is left then is what printing
        the cluster in one piece leaves: a program's output is cut wherever a read happens to
        end, and that must not show on the screen.
        """
        columns = self.columns
        cursor = self.buffer.cursor
        row = self.buffer.edit(cursor.y)
        pen = row.pens[column]
        width = cell_len(cluster)
        if width == cell_len(row.cells[column]):
            # Rewritten whole, second cell and all: half a double-width cell would be blanked.
            row.put(column, [cluster, *[""] * (width - 1)], pen)
        elif cursor.pending_wrap:
            # The last column has no room for it: it is printed again whole, to land where it
            # would have, on the next row or, without autowrap, over the last two columns.
            row.erase(column, columns, DEFAULT_PEN, columns)
            self._write([cluster, ""], pen)
        else:
            if self._insert:
                row.insert(cursor.x, [""], pen, columns)
            row.put(column, [cluster, ""], pen)
            cursor.x += 1
            if cursor.x >= columns:
                cursor.x = columns - 1
                cursor.pending_wrap = True

    def _write(self, cells: Sequence[str], pen: Pen | None = None) -> None:
        """Print cells at the cursor, in the current pen unless another is given."""
        buffer, columns = self.buffer, self.columns
        if pen is None:
            pen = self.pen
        cursor = buffer.cursor
        autowrap = _AUTOWRAP in self._switches
        position, total = 0, len(cells)
        while position < total:
            if cursor.pending_wrap:
                if autowrap:
                    self._wrap()
                else:
                    # Nowhere to go: the last column is written over.
                    cursor.pending_wrap = False
            end = min(position + columns - cursor.x, total)
            if end < total and not cells[end]:
                # A double-width character does not fit in what is left of the row.
                end -= 1
                if end == position:
                    if not cursor.x:
                        # A screen one column wide has no room for it anywhere.
                        position += 2
                    elif autowrap:
                        self._wrap()
                    else:
                        # Written over the last two columns, as a narrow one is over the last.
                        cursor.x -= 1
                    continue
            row = buffer.edit(cursor.y)
            if self._insert:
                row.insert(cursor.x, cells[position:end], pen, columns)
            else:
                row.put(cursor.x, cells[position:end], pen)
            cursor.x += end - position
            position = end
            if cursor.x >= columns:
                # Whether that is a wrap is for the next character to find out: until then it only
                # says which cell was printed last, which a character completing it has to know.
                cursor.x = columns - 1
                cursor.pending_wrap = True

    def _wrap(self) -> None:
        buffer = self.buffer
        cursor = buffer.cursor
        row = buffer.row(cursor.y)
        cursor.x = 0
        buffer.index(self.pen.eraser)
        # Flagged only once the cursor has moved on. Scrolling, which may be how it moved, unflags
        # the row it leaves behind; and below the margins on the last row it does not move at all.
        row.wrapped = buffer.row(cursor.y) is not row

    def _line_feed(self) -> None:
        self.buffer.index(self.pen.eraser)
        if self._newline:
            self.buffer.cursor.x = 0

    # -- cursor ------------------------------------------------------------------------------

    def _move_to(self, x: int, y: int) -> None:
        """Absolute positioning. Under origin mode rows count from, and stay inside, the margins."""
        buffer = self.buffer
        if _ORIGIN in self._switches:
            y = min(buffer.margin_top + max(y, 0), buffer.margin_bottom)
        buffer.cursor.y = min(max(y, 0), self.lines - 1)
        buffer.cursor.x = min(max(x, 0), self.columns - 1)
        buffer.cursor.pending_wrap = False

    def _move_by(self, columns: int, rows: int) -> None:
        """Relative movement, which stops at a margin the cursor starts inside of."""
        buffer = self.buffer
        cursor = buffer.cursor
        if rows < 0:
            limit = buffer.margin_top if cursor.y >= buffer.margin_top else 0
            cursor.y = max(cursor.y + rows, limit)
        elif rows:
            limit = buffer.margin_bottom if cursor.y <= buffer.margin_bottom else self.lines - 1
            cursor.y = min(cursor.y + rows, limit)
        cursor.x = min(max(cursor.x + columns, 0), self.columns - 1)
        cursor.pending_wrap = False

    def _row_in_origin(self) -> int:
        """The cursor row as reports and absolute positioning count it."""
        buffer = self.buffer
        return buffer.cursor.y - (buffer.margin_top if _ORIGIN in self._switches else 0)

    def _tab_forward(self, count: int) -> None:
        cursor = self.buffer.cursor
        for _ in range(count):
            cursor.x = min((stop for stop in self._tab_stops if stop > cursor.x), default=self.columns - 1)
        cursor.x = min(cursor.x, self.columns - 1)

    def _tab_backward(self, count: int) -> None:
        cursor = self.buffer.cursor
        for _ in range(count):
            cursor.x = max((stop for stop in self._tab_stops if stop < cursor.x), default=0)
        cursor.pending_wrap = False

    def _save_cursor(self) -> None:
        cursor = self.buffer.cursor
        self._saved_cursors[self.alternate_screen] = _SavedCursor(
            cursor.x, cursor.y, cursor.pending_wrap, self.pen, self._charsets.snapshot(), _ORIGIN in self._switches
        )

    def _restore_cursor(self) -> None:
        saved = self._saved_cursors.get(self.alternate_screen)
        if saved is None:
            # Nothing saved: DEC specifies the power-up state, home with default rendition.
            self._switches.discard(_ORIGIN)
            self._move_to(0, 0)
            self.pen = self.pen._replace(foreground=None, background=None, attributes=DEFAULT_PEN.attributes)
            return
        cursor = self.buffer.cursor
        cursor.x = min(saved.x, self.columns - 1)
        cursor.y = min(saved.y, self.lines - 1)
        cursor.pending_wrap = saved.pending_wrap and cursor.x == self.columns - 1
        self.pen = saved.pen._replace(link=self.pen.link)
        self._charsets.restore(saved.charsets)
        self._set_switch(_ORIGIN, saved.origin)

    # -- screens and modes -------------------------------------------------------------------

    def _set_switch(self, mode: int, enabled: bool) -> None:
        if enabled:
            self._switches.add(mode)
        else:
            self._switches.discard(mode)

    def _set_private_mode(self, mode: int, enabled: bool) -> None:
        if mode in _SWITCHES:
            self._set_switch(mode, enabled)
            if mode == _ORIGIN:
                self._move_to(0, 0)
        elif mode in _ALTERNATE_SCREENS:
            self._show_alternate_screen(enabled, save_cursor=mode == _ALTERNATE_SCREEN_SAVING_CURSOR)
        elif (tracking := _TRACKING_MODES.get(mode)) is not None:
            if enabled or self.mouse_tracking is tracking:
                self.mouse_tracking = tracking if enabled else MouseTracking.OFF
        elif (encoding := _ENCODING_MODES.get(mode)) is not None:
            if enabled or self.mouse_encoding is encoding:
                self.mouse_encoding = encoding if enabled else MouseEncoding.DEFAULT

    def _private_mode_status(self, mode: int) -> int:
        """DECRPM status: 1 set, 2 reset, 0 for a mode this terminal does not have."""
        if mode in _SWITCHES:
            enabled = mode in self._switches
        elif mode in _ALTERNATE_SCREENS:
            enabled = self.alternate_screen
        elif mode in _TRACKING_MODES:
            enabled = self.mouse_tracking.value == mode
        elif mode in _ENCODING_MODES:
            enabled = self.mouse_encoding.value == mode
        else:
            return 0
        return 1 if enabled else 2

    def _show_alternate_screen(self, show: bool, *, save_cursor: bool) -> None:
        if show == self.alternate_screen:
            return
        if show:
            if save_cursor:
                self._save_cursor()
            primary_cursor = self._primary.cursor
            self._alternate = ScreenBuffer(self.columns, self.lines, history_limit=0)
            self._kitty_flags[True] = [0]
            self._alternate.cursor.x, self._alternate.cursor.y = primary_cursor.x, primary_cursor.y
            self.buffer = self._alternate
        else:
            self.buffer = self._primary
            if save_cursor:
                self._restore_cursor()
        self.buffer.damage_everything()

    def _soft_reset(self) -> None:
        """DECSTR: back to sane modes without touching what is on the screen."""
        self._switches.discard(_ORIGIN)
        self._switches.discard(_APPLICATION_CURSOR)
        self._switches.discard(_APPLICATION_KEYPAD)
        self._switches.update(_SWITCHES_ON_AT_RESET)
        self._insert = False
        self.pen = self.pen._replace(foreground=None, background=None, attributes=DEFAULT_PEN.attributes)
        self.cursor_shape = CursorShape.BLOCK
        self._charsets = Charsets()
        self._saved_cursors.clear()
        self.buffer.set_margins(0, self.lines - 1)

    def _hard_reset(self) -> None:
        """RIS: the state the terminal was constructed in, history gone."""
        self._history_dropped += self._primary.top
        self._reset()
        self._parser.reset()

    # -- dispatch tables ---------------------------------------------------------------------

    def _escape_table(self) -> dict[str, Callable[[], None]]:
        return {
            "7": self._save_cursor,  # DECSC
            "8": self._restore_cursor,  # DECRC
            "D": lambda: self.buffer.index(self.pen.eraser),  # IND
            "E": self._next_line,  # NEL
            "H": lambda: self._tab_stops.add(self.buffer.cursor.x),  # HTS
            "M": lambda: self.buffer.reverse_index(self.pen.eraser),  # RI
            "N": lambda: self._charsets.shift_once(2),  # SS2
            "O": lambda: self._charsets.shift_once(3),  # SS3
            "n": lambda: self._charsets.lock(2),  # LS2
            "o": lambda: self._charsets.lock(3),  # LS3
            "c": self._hard_reset,  # RIS
            "=": lambda: self._switches.add(_APPLICATION_KEYPAD),  # DECKPAM
            ">": lambda: self._switches.discard(_APPLICATION_KEYPAD),  # DECKPNM
            "#8": lambda: self.buffer.fill("E"),  # DECALN
        }

    def _control_sequence_table(self) -> dict[str, Callable[[tuple[int, ...]], None]]:
        return {
            "@": self._insert_characters,
            "A": lambda p: self._move_by(0, -(p[0] or 1)),  # CUU
            "B": lambda p: self._move_by(0, p[0] or 1),  # CUD
            "C": lambda p: self._move_by(p[0] or 1, 0),  # CUF
            "D": lambda p: self._move_by(-(p[0] or 1), 0),  # CUB
            "E": lambda p: self._move_by(-self.columns, p[0] or 1),  # CNL
            "F": lambda p: self._move_by(-self.columns, -(p[0] or 1)),  # CPL
            "G": lambda p: self._move_to(p[0] - 1, self._row_in_origin()),  # CHA
            "H": self._cursor_position,
            "I": lambda p: self._tab_forward(p[0] or 1),  # CHT
            "J": self._erase_in_display,
            "?J": self._erase_in_display,  # DECSED; nothing here is protected, so it erases like ED
            "K": self._erase_in_line,
            "?K": self._erase_in_line,  # DECSEL
            "L": lambda p: self._edit_lines(self.buffer.insert_lines, p),  # IL
            "M": lambda p: self._edit_lines(self.buffer.delete_lines, p),  # DL
            "P": self._delete_characters,
            "S": lambda p: self.buffer.scroll_up(p[0] or 1, self.pen.eraser),  # SU
            "T": lambda p: self.buffer.scroll_down(p[0] or 1, self.pen.eraser),  # SD
            "X": self._erase_characters,
            "Z": lambda p: self._tab_backward(p[0] or 1),  # CBT
            "`": lambda p: self._move_to(p[0] - 1, self._row_in_origin()),  # HPA
            "a": lambda p: self._move_by(p[0] or 1, 0),  # HPR
            "b": self._repeat,
            "c": self._report_primary_attributes,
            ">c": self._report_secondary_attributes,
            "d": self._line_position_absolute,
            "e": lambda p: self._move_by(0, p[0] or 1),  # VPR
            "f": self._cursor_position,  # HVP
            "g": self._clear_tab_stops,
            "h": lambda p: self._set_modes(p, enabled=True),  # SM
            "l": lambda p: self._set_modes(p, enabled=False),  # RM
            "?h": lambda p: self._set_private_modes(p, enabled=True),  # DECSET
            "?l": lambda p: self._set_private_modes(p, enabled=False),  # DECRST
            "n": self._report_status,
            "?n": self._report_status_extended,
            "!p": lambda _: self._soft_reset(),  # DECSTR
            "$p": self._report_mode,
            "?$p": self._report_private_mode,
            " q": self._set_cursor_shape,
            "r": self._set_scrolling_region,
            "s": lambda _: self._save_cursor(),  # SCOSC
            "t": self._window_operation,
            "u": lambda _: self._restore_cursor(),  # SCORC
            ">m": lambda p: self._set_modify_other_keys(p, p[1] if len(p) > 1 else 0),  # XTMODKEYS
            ">n": lambda p: self._set_modify_other_keys(p, 0),
            ">u": self._push_kitty_flags,
            "<u": self._pop_kitty_flags,
            "=u": self._set_kitty_flags,
            "?u": lambda _: self._replies.append(f"\x1b[?{self._kitty_flags[self.alternate_screen][-1]}u"),
        }

    def _operating_system_command_table(self) -> dict[str, Callable[[str], None]]:
        return {
            # Titles (OSC 0 and 2) fall through with everything else unlisted: accepted, ignored.
            "7": self._report_directory_url,
            "8": self._set_hyperlink,
            "2025": self._report_directory,
            "2026": self._report_command,
        }

    # -- control sequences -------------------------------------------------------------------

    def _cursor_position(self, parameters: tuple[int, ...]) -> None:
        """CUP / HVP."""
        row = (parameters[0] or 1) - 1
        self._anchor_to_host(row)
        self._move_to((parameters[1] or 1) - 1 if len(parameters) > 1 else 0, row)

    def _line_position_absolute(self, parameters: tuple[int, ...]) -> None:
        """VPA."""
        row = (parameters[0] or 1) - 1
        self._anchor_to_host(row)
        self._move_to(self.buffer.cursor.x, row)

    def _anchor_to_host(self, row: int) -> None:
        """Believe the first row a repainting host names after a resize; see `resize`."""
        if self._anchor_pending:
            self._anchor_pending = False
            self.buffer.anchor_cursor_row(row)

    def _next_line(self) -> None:
        """NEL."""
        self.buffer.index(self.pen.eraser)
        self.buffer.cursor.x = 0

    def _erase_in_display(self, parameters: tuple[int, ...]) -> None:
        """ED."""
        buffer = self.buffer
        cursor = buffer.cursor
        eraser = self.pen.eraser
        cursor.pending_wrap = False
        if parameters[0] == 0:
            buffer.edit(cursor.y).erase(cursor.x, self.columns, eraser, self.columns)
            buffer.erase_rows(cursor.y + 1, self.lines - 1, eraser)
        elif parameters[0] == 1:
            buffer.erase_rows(0, cursor.y - 1, eraser)
            buffer.edit(cursor.y).erase(0, cursor.x + 1, eraser, self.columns)
        elif parameters[0] == 2:
            buffer.erase_rows(0, self.lines - 1, eraser)
        elif parameters[0] == 3:
            self._history_dropped += buffer.clear_history()

    def _erase_in_line(self, parameters: tuple[int, ...]) -> None:
        """EL."""
        cursor = self.buffer.cursor
        cursor.pending_wrap = False
        start, end = ((cursor.x, self.columns), (0, cursor.x + 1), (0, self.columns))[min(parameters[0], 2)]
        self.buffer.edit(cursor.y).erase(start, end, self.pen.eraser, self.columns)

    def _erase_characters(self, parameters: tuple[int, ...]) -> None:
        """ECH."""
        cursor = self.buffer.cursor
        cursor.pending_wrap = False
        end = min(cursor.x + (parameters[0] or 1), self.columns)
        self.buffer.edit(cursor.y).erase(cursor.x, end, self.pen.eraser, self.columns)

    def _insert_characters(self, parameters: tuple[int, ...]) -> None:
        """ICH."""
        cursor = self.buffer.cursor
        cursor.pending_wrap = False
        blanks = " " * min(parameters[0] or 1, self.columns - cursor.x)
        self.buffer.edit(cursor.y).insert(cursor.x, blanks, self.pen.eraser, self.columns)

    def _delete_characters(self, parameters: tuple[int, ...]) -> None:
        """DCH."""
        cursor = self.buffer.cursor
        cursor.pending_wrap = False
        self.buffer.edit(cursor.y).delete(cursor.x, parameters[0] or 1, self.pen.eraser, self.columns)

    def _edit_lines(self, operation: Callable[[int, Pen], None], parameters: tuple[int, ...]) -> None:
        """IL / DL, which leave the cursor at the start of its row."""
        operation(parameters[0] or 1, self.pen.eraser)
        self.buffer.cursor.x = 0
        self.buffer.cursor.pending_wrap = False

    def _repeat(self, parameters: tuple[int, ...]) -> None:
        """REP."""
        if self._last_printed:
            repeated = [self._last_printed] * min(parameters[0] or 1, self.columns * self.lines)
            self._write([cell for cluster in repeated for cell in (cluster, *[""] * (cell_len(cluster) - 1))])

    def _clear_tab_stops(self, parameters: tuple[int, ...]) -> None:
        """TBC."""
        if parameters[0] == 0:
            self._tab_stops.discard(self.buffer.cursor.x)
        elif parameters[0] == 3:
            self._tab_stops.clear()

    def _set_private_modes(self, parameters: tuple[int, ...], *, enabled: bool) -> None:
        """DECSET / DECRST."""
        for mode in parameters:
            self._set_private_mode(mode, enabled)

    def _set_modes(self, parameters: tuple[int, ...], *, enabled: bool) -> None:
        """SM / RM."""
        for mode in parameters:
            if mode == _INSERT:
                self._insert = enabled
            elif mode == _NEWLINE:
                self._newline = enabled

    def _set_scrolling_region(self, parameters: tuple[int, ...]) -> None:
        """DECSTBM."""
        top = (parameters[0] or 1) - 1
        bottom = (parameters[1] if len(parameters) > 1 and parameters[1] else self.lines) - 1
        bottom = min(bottom, self.lines - 1)
        if top < bottom:
            self.buffer.set_margins(top, bottom)
            self._move_to(0, 0)

    def _set_cursor_shape(self, parameters: tuple[int, ...]) -> None:
        """DECSCUSR."""
        if parameters[0] < len(_CURSOR_SHAPES):
            self.cursor_shape = _CURSOR_SHAPES[parameters[0]]

    def _set_modify_other_keys(self, parameters: tuple[int, ...], level: int) -> None:
        if parameters[0] == _XTERM_MODIFY_OTHER_KEYS:
            self._modify_other_keys = level

    def _push_kitty_flags(self, parameters: tuple[int, ...]) -> None:
        stack = self._kitty_flags[self.alternate_screen]
        stack.append(parameters[0] & _KITTY_DISAMBIGUATE)
        del stack[1:-_KITTY_STACK_LIMIT]

    def _pop_kitty_flags(self, parameters: tuple[int, ...]) -> None:
        stack = self._kitty_flags[self.alternate_screen]
        del stack[max(len(stack) - (parameters[0] or 1), 0) :]
        if not stack:
            # Emptied, which the protocol says resets every flag: those set in place on the
            # bottom entry, never pushed, go too.
            stack.append(0)

    def _set_kitty_flags(self, parameters: tuple[int, ...]) -> None:
        stack = self._kitty_flags[self.alternate_screen]
        flags = parameters[0] & _KITTY_DISAMBIGUATE
        how = parameters[1] if len(parameters) > 1 else 1
        stack[-1] = flags if how <= 1 else stack[-1] | flags if how == 2 else stack[-1] & ~flags

    def _window_operation(self, parameters: tuple[int, ...]) -> None:
        """XTWINOPS. Only the size report: nothing else about a window applies to a widget."""
        if parameters[0] == 18:
            self._replies.append(f"\x1b[8;{self.lines};{self.columns}t")

    # -- reports -----------------------------------------------------------------------------

    def _report_primary_attributes(self, parameters: tuple[int, ...]) -> None:
        """DA1: a VT220 with ANSI color."""
        if parameters[0] == 0:
            self._replies.append("\x1b[?62;22c")

    def _report_secondary_attributes(self, parameters: tuple[int, ...]) -> None:
        """DA2: terminal type VT220, no firmware version worth testing against."""
        if parameters[0] == 0:
            self._replies.append("\x1b[>1;0;0c")

    def _report_status(self, parameters: tuple[int, ...]) -> None:
        """DSR: operating status, or the cursor position (CPR)."""
        if parameters[0] == 5:
            self._replies.append("\x1b[0n")
        elif parameters[0] == 6:
            self._replies.append(f"\x1b[{self._row_in_origin() + 1};{self.buffer.cursor.x + 1}R")

    def _report_status_extended(self, parameters: tuple[int, ...]) -> None:
        """DECXCPR."""
        if parameters[0] == 6:
            self._replies.append(f"\x1b[?{self._row_in_origin() + 1};{self.buffer.cursor.x + 1}R")

    def _report_mode(self, parameters: tuple[int, ...]) -> None:
        """DECRQM for an ANSI mode. The answer (DECRPM) says 1 for set, 2 for reset, 0 for unknown."""
        mode = parameters[0]
        states = {_INSERT: self._insert, _NEWLINE: self._newline}
        status = 0 if mode not in states else 1 if states[mode] else 2
        self._replies.append(f"\x1b[{mode};{status}$y")

    def _report_private_mode(self, parameters: tuple[int, ...]) -> None:
        """DECRQM for a DEC private mode."""
        self._replies.append(f"\x1b[?{parameters[0]};{self._private_mode_status(parameters[0])}$y")

    # -- operating system commands -----------------------------------------------------------

    def _set_hyperlink(self, argument: str) -> None:
        """OSC 8 ; params ; URI. An empty URI closes the link."""
        _, _, uri = argument.partition(";")
        self.pen = self.pen._replace(link=uri or None)

    def _report_directory_url(self, url: str) -> None:
        """OSC 7, the ``file://host/path`` convention shells use to announce their directory."""
        try:
            parts = urlsplit(url)
        except ValueError:
            # Brackets around something that is no IP address, say. Nowhere a shell can be.
            return
        host = (parts.hostname or "").partition(".")[0]
        # A shell on the far side of ssh reports paths that mean nothing on this machine.
        if parts.scheme == "file" and host in ("", "localhost", socket.gethostname().lower().partition(".")[0]):
            path = unquote(parts.path)
            # A drive path is written file:///C:/..., where the leading slash belongs to the URL.
            self._events.append(DirectoryChanged(path[1:] if _DRIVE_PATH.match(path) else path))

    def _report_directory(self, payload: str) -> None:
        if path := _decode_payload(payload):
            self._events.append(DirectoryChanged(path))

    def _report_command(self, payload: str) -> None:
        # An empty command is Enter at an empty prompt: nothing ran.
        if command := _decode_payload(payload):
            self._events.append(CommandSubmitted(command))
