# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Theme editor palette, variable groups and color swatches."""

from __future__ import annotations

from itertools import batched
from typing import TYPE_CHECKING, ClassVar

from rich.color import EIGHT_BIT_PALETTE
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.color import Color
from textual.geometry import Region
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Button

from chrys.app.tui.binding_display import CLOSE_BINDING, localized_binding
from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.theme import (
    BUTTON_COLOR_VARIABLES,
    CHRYS_COLOR_NAMES,
)
from chrys.app.tui.widgets.checkbox import CHECKED_MARKER, UNCHECKED_MARKER
from chrys.app.tui.widgets.color_picker.model import ColorEditContext, parse_color
from chrys.foundation.i18n import MessageRef

from . import messages as M

if TYPE_CHECKING:
    from chrys.app.tui.i18n import LocaleController

_SURFACE_COLOR_NAMES = ("background", "surface", "panel", "boost")
_THEME_COLOR_NAMES = (*CHRYS_COLOR_NAMES, *_SURFACE_COLOR_NAMES)
# Textual's ANSI generator always emits transparent surface/panel/boost,
# irrespective of these Theme fields. ANSI controls have dedicated slots below.
_ANSI_THEME_COLOR_NAMES = (*CHRYS_COLOR_NAMES, "background")

# Deliberate editing surface, independent of whichever overrides a built-in or
# imported YAML happens to carry. Advanced/unknown keys are retained by the
# document and writer, but do not become unexplained or ineffective UI rows.
_ANSI_VARIABLE_NAMES = ("ansi-background", "ansi-foreground")
# RGB foreground-muted only supplies the default H6 color; expose H6 directly.
# ANSI also consumes it in controls, so its generic slot remains useful there.
# RGB H4/H5 follow foreground; ANSI instead defaults both to cyan.
_ANSI_ONLY_EDITOR_VARIABLES = frozenset(
    (
        *_ANSI_VARIABLE_NAMES,
        "foreground-muted",
        "control-background-muted",
        "control-foreground-muted",
        "markdown-h4-color",
        "markdown-h5-color",
    )
)
_VARIABLE_GROUPS = (
    ("button", BUTTON_COLOR_VARIABLES),
    ("border", ("border", "border-blurred", "border-color")),
    ("scrollbar", ("scrollbar", "scrollbar-hover", "scrollbar-active", "scrollbar-background")),
    (
        "block",
        (
            "block-cursor-foreground",
            "block-cursor-background",
            "block-cursor-blurred-background",
            "block-hover-background",
        ),
    ),
    ("input", ("input-cursor-background", "input-cursor-foreground", "input-selection-background")),
    ("footer", ("footer-background", "footer-key-foreground", "footer-description-foreground")),
    ("markdown", tuple(f"markdown-h{level}-color" for level in range(1, 7))),
    ("muted", ("primary-muted", "error-muted", "warning-muted", "success-muted", "foreground-muted", "text-muted")),
    (
        "controls",
        (
            "control-background",
            "control-disabled-background",
            "control-background-muted",
            "control-foreground-muted",
            "overlay-background",
            "markdown-block-background",
        ),
    ),
    ("misc", ("text-disabled", "screen-selection-background", "tool-group-title-color", "hatch-color")),
    ("ansi", _ANSI_VARIABLE_NAMES),
)
_CANONICAL_VARIABLE_NAMES = tuple(name for _group, names in _VARIABLE_GROUPS for name in names)
_BUCKET_ORDER = tuple(group for group, _names in _VARIABLE_GROUPS)
_RESET_BUTTON_LABEL = "↺"
_ANSI_COLOR_TOKENS = (
    "ansi_default",
    "ansi_black",
    "ansi_red",
    "ansi_green",
    "ansi_yellow",
    "ansi_blue",
    "ansi_magenta",
    "ansi_cyan",
    "ansi_white",
    "ansi_bright_black",
    "ansi_bright_red",
    "ansi_bright_green",
    "ansi_bright_yellow",
    "ansi_bright_blue",
    "ansi_bright_magenta",
    "ansi_bright_cyan",
    "ansi_bright_white",
)


def _group_variables(var_names: list[str], *, ansi: bool = False) -> list[tuple[str, list[str]]]:
    """Show supported colors in stable groups, omitting mode-inapplicable slots."""
    requested = set(var_names)
    if not ansi:
        requested -= _ANSI_ONLY_EDITOR_VARIABLES
    return [
        (group, visible)
        for group, names in _VARIABLE_GROUPS
        if (visible := [name for name in names if name in requested])
    ]


def _try_parse_color(value: str) -> Color | None:
    """Resolve concrete colors while retaining automatic expressions as text."""
    return parse_color(value)


def _ansi_base_token(value: str | None) -> str | None:
    """Return the supported ANSI token at the front of ``value``."""
    if not value:
        return None
    parts = value.strip().split(maxsplit=1)
    return parts[0] if parts and parts[0] in _ANSI_COLOR_TOKENS else None


def _ansi_display_label(value: str) -> str | None:
    """Show palette slots as approximate HEX without changing theme data.

    Terminal-default remains a keyword because it has no fixed palette color.
    Composite expressions are outside this formatter's scope.
    """
    token = _ansi_base_token(value)
    if token is None or value.strip() != token:
        return None
    return "default" if token == "ansi_default" else Color.parse(token).hex6


def _palette_color(index: int) -> Color:
    """Return xterm-256 palette index as a Textual ``Color``."""
    triplet = EIGHT_BIT_PALETTE[index]
    return Color(triplet.red, triplet.green, triplet.blue)


def _palette_sort_key(index: int) -> tuple[float, float, float]:
    """Group the 216-color cube perceptually instead of by xterm index."""
    h, s, v = _palette_color(index).hsv
    return h, -s, -v


# Sentinel for the dedicated "transparent" cell rendered alongside the
# 256 palette entries.  Negative so it can't collide with any real
# palette index across the ``_XTERM_256_POSITION_BY_INDEX*`` maps.
_TRANSPARENT_SENTINEL = -1

# Three xterm-256 layouts share a base prefix.  The picker picks one in
# ``_Xterm256PalettePicker.__init__`` based on whether transparent is a
# valid choice for the slot being edited:
#   - ``_XTERM_256_ROWS``         : full layout w/ selectable Transparent checkbox.
#   - ``_XTERM_256_ROWS_BASE``    : no Transparent section at all.
#   - ``_XTERM_256_ROWS_WITH_HINT``: Transparent header + empty placeholder
#                                    row (``(None, ())``); ``render_line``
#                                    paints the disabled hint there.
# The four surface slots (background / surface / panel / boost) use one of
# the two no-transparent layouts because writing ``"transparent"`` to
# ``theme.surface`` crashes Textual's TextArea theme apply — rich's color
# parser rejects the keyword that Textual itself accepts.
_XTERM_ANSI_SECTION = tuple(range(16))
_XTERM_CUBE_SECTION = tuple(sorted(range(16, 232), key=_palette_sort_key))
_XTERM_GRAYSCALE_SECTION = tuple(range(232, 256))
_XTERM_256_ROWS_BASE: tuple[tuple[str | None, tuple[int, ...]], ...] = (
    ("ANSI 16", ()),
    (None, _XTERM_ANSI_SECTION),
    ("", ()),
    ("Color cube (hue grouped)", ()),
    *((None, tuple(row)) for row in batched(_XTERM_CUBE_SECTION, 18, strict=True)),
    ("", ()),
    ("Grayscale", ()),
    *((None, tuple(row)) for row in batched(_XTERM_GRAYSCALE_SECTION, 12, strict=True)),
)
_XTERM_256_ROWS: tuple[tuple[str | None, tuple[int, ...]], ...] = (
    *_XTERM_256_ROWS_BASE,
    ("", ()),
    ("Transparent", (_TRANSPARENT_SENTINEL,)),
)
# Placeholder row uses ``indices=()`` so ``_selectable_rows`` excludes it.
_XTERM_256_ROWS_WITH_HINT: tuple[tuple[str | None, tuple[int, ...]], ...] = (
    *_XTERM_256_ROWS_BASE,
    ("", ()),
    ("Transparent", ()),
    (None, ()),
)


def _build_position_map(rows: tuple[tuple[str | None, tuple[int, ...]], ...]) -> dict[int, tuple[int, int]]:
    return {index: (row, column) for row, (_, indices) in enumerate(rows) for column, index in enumerate(indices)}


_XTERM_256_POSITION_BY_INDEX = _build_position_map(_XTERM_256_ROWS)
# Shared by BASE and WITH_HINT: neither carries a selectable sentinel.
_XTERM_256_POSITION_BY_INDEX_BASE = _build_position_map(_XTERM_256_ROWS_BASE)

_XTERM_CELL_STYLES: tuple[Style, ...] = tuple(
    Style.from_color(
        _palette_color(i).get_contrast_text().with_alpha(1.0).rich_color,
        _palette_color(i).rich_color,
    )
    for i in range(256)
)
_ANSI_SWATCH_STYLES: tuple[Style, ...] = tuple(
    Style.from_color(bgcolor=Color.parse(token).rich_color) for token in _ANSI_COLOR_TOKENS
)


class _ResetButton(Button):
    """Per-row reset glyph.

    ``target`` is ``"color:<name>"`` or ``"var:<name>"`` so one handler
    can dispatch on the prefix. ``_refresh_dirty_state`` toggles the
    ``--clean`` class; the CSS uses it to hide the glyph on unedited rows.
    """

    # Mirror of ``_ColorSwatchButton`` bindings: ``left`` hops back to
    # the swatch, ``up``/``down`` move to the neighbouring row's swatch
    # (returning the user to the main navigation column).
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("left", "focus_swatch", show=False),
        Binding("up", "navigate_row(-1)", show=False),
        Binding("down", "navigate_row(1)", show=False),
    ]

    def __init__(self, target: str) -> None:
        super().__init__(_RESET_BUTTON_LABEL, classes="--reset-color --clean")
        # Textual's CSS parser rejects line-pad: 0; the public style accepts it.
        self.styles.line_pad = 0
        self.target = target

    def _find_swatch(self) -> _ColorSwatchButton | None:
        screen = self.screen
        if screen is None:
            return None
        for swatch in screen.query(_ColorSwatchButton):
            if swatch._reset_target() == self.target:
                return swatch
        return None

    def action_focus_swatch(self) -> None:
        if (swatch := self._find_swatch()) is not None:
            swatch.focus()

    def action_navigate_row(self, delta: int) -> None:
        # Up/down on a reset returns to the swatch column, then moves
        # one row in that direction — same end state as if the user had
        # pressed Left first, then Up/Down on the swatch.
        if (swatch := self._find_swatch()) is not None:
            swatch.action_move_focus(delta)


class _ColorSwatchButton(Button):
    """Shared rendering for one-row color swatches.

    Non-hex values (``transparent``, ``ansi_white 40%``, ...) keep the
    button clickable so the picker can replace them with a hex value;
    the ``--non-hex`` class only marks them visually.
    """

    # Keyboard navigation between rows + lateral hop to the row's reset
    # button.  Bindings live on the swatch itself so they only fire when
    # a swatch is focused — no priority/global side effects.
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "move_focus(-1)", show=False),
        Binding("down", "move_focus(1)", show=False),
        Binding("right", "focus_reset", show=False),
    ]

    def __init__(self, value: str | None) -> None:
        super().__init__("")
        # Let transparent and RGBA labels fit the compact swatch column.
        self.styles.line_pad = 0
        self.set_value(value)

    def _reset_target(self) -> str | None:
        """``_ResetButton.target`` for the row's reset glyph.  Subclasses set."""
        return None

    def action_move_focus(self, delta: int) -> None:
        """Focus the previous/next swatch (clamped at the ends)."""
        screen = self.screen
        if screen is None:
            return
        # ``query`` matches subclasses, so this picks up both
        # ``_FlatThemeColorButton`` and ``_VariableSwatchButton`` rows in
        # DOM order — same order the user sees on screen.
        swatches = list(screen.query(_ColorSwatchButton))
        try:
            idx = swatches.index(self)
        except ValueError:
            return
        next_idx = max(0, min(idx + delta, len(swatches) - 1))
        swatches[next_idx].focus()

    def action_focus_reset(self) -> None:
        """Hop to this row's reset glyph, but only when it's visible.

        Clean rows hide their reset via ``--clean`` (visibility: hidden);
        focusing a hidden widget would be a confusing no-op, so we skip.
        """
        target = self._reset_target()
        if target is None:
            return
        screen = self.screen
        if screen is None:
            return
        for reset in screen.query(_ResetButton):
            if reset.target == target and not reset.has_class("--clean"):
                reset.focus()
                return

    def set_value(self, value: str | None) -> None:
        if value is None:
            self.label = Text(M.UNSET)
            self.tooltip = None
            self.add_class("--unset")
            self.remove_class("--non-hex")
            self.styles.background = None
            return
        ansi_label = _ansi_display_label(value)
        self.label = Text(ansi_label or value)
        self.tooltip = value if ansi_label is not None else None
        self.remove_class("--unset")
        color = _try_parse_color(value)
        if color is None and (token := _ansi_base_token(value)) is not None:
            color = Color.parse(token)
        if color is None:
            self.add_class("--non-hex")
            self.styles.background = None
            return
        self.remove_class("--non-hex")
        self.styles.background = color


class _FlatThemeColorButton(_ColorSwatchButton):
    """Single-row swatch for a ``Theme.<color>`` attribute."""

    def __init__(self, value: str | None, color_name: str) -> None:
        self.color_name = color_name
        super().__init__(value)

    def _reset_target(self) -> str:
        return f"color:{self.color_name}"


class _VariableSwatchButton(_ColorSwatchButton):
    """Single-row swatch for a ``theme.variables`` entry."""

    def __init__(self, var_name: str, value: str | None) -> None:
        self.var_name = var_name
        super().__init__(value)

    def _reset_target(self) -> str:
        return f"var:{self.var_name}"


class _DismissableModal(BaseDialog):
    """``ModalScreen`` with shared escape-to-close + backdrop click dismiss.

    Both modal screens in this file follow the same "click outside the
    dialog or press escape to dismiss" idiom; this base captures it so
    subclasses only describe their dialog content.
    """

    BINDINGS: ClassVar[list] = [localized_binding("escape", "close", CLOSE_BINDING)]

    def __init__(self, *, locale_controller: LocaleController | None = None) -> None:
        super().__init__()
        self._locale_controller = locale_controller
        self._dismiss_requested = False

    def on_mount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.register_surface(self)

    def on_unmount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.unregister_surface(self)

    def refresh_localization(self) -> None:
        """Subclasses replace display text while preserving their edit state."""

    def action_close(self) -> None:
        self._dismiss_requested = True
        if self.app.screen is self:
            self.dismiss()

    def on_screen_resume(self) -> None:
        # A theme menu may cover this dialog when selection cancels its edit.
        # Only pop our own screen after that covering menu has gone away.
        if self._dismiss_requested and self.app.screen is self:
            self.dismiss()

    def _dismiss_clicked_outside(self) -> None:
        self.action_close()


class PaletteChanged(Message):
    """A user selection from the discrete palette."""

    def __init__(self, picker: _Xterm256PalettePicker, color: Color) -> None:
        super().__init__()
        self.picker = picker
        self.color = color
        self.context = picker.context


class _AnsiTokenPicker(Widget, can_focus=True):
    """Palette picker for Textual's terminal ANSI color tokens."""

    ALLOW_SELECT = False
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "move(-1)", show=False),
        Binding("down", "move(1)", show=False),
    ]

    class Changed(Message):
        """Posted when the selected ANSI token changes."""

        def __init__(self, picker: _AnsiTokenPicker, token: str) -> None:
            super().__init__()
            self.picker = picker
            self.token = token
            self.context = picker.context

    def __init__(self, value: str | None, *, context: ColorEditContext | None = None) -> None:
        super().__init__()
        self.context = context
        token = _ansi_base_token(value) or "ansi_default"
        self._selected_index = _ANSI_COLOR_TOKENS.index(token)
        self._selection_explicit = value == self.token

    @property
    def token(self) -> str:
        return _ANSI_COLOR_TOKENS[self._selected_index]

    @property
    def selection_region(self) -> Region:
        return Region(0, self._selected_index, 30, 1)

    def render_line(self, y: int) -> Strip:
        if y >= len(_ANSI_COLOR_TOKENS):
            return Strip.blank(self.size.width)
        token = _ANSI_COLOR_TOKENS[y]
        marker = ">" if y == self._selected_index else " "
        label_style = Style(bold=y == self._selected_index)
        return Strip(
            [
                Segment(f"{marker} ", label_style),
                Segment("    ", _ANSI_SWATCH_STYLES[y]),
                Segment(f" {token}", label_style),
            ]
        )

    def render(self) -> Text:
        return Text(self.token)

    def action_move(self, dy: int) -> None:
        self._select_index(min(max(self._selected_index + dy, 0), len(_ANSI_COLOR_TOKENS) - 1), preview=True)

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1:
            return
        offset = event.get_content_offset(self)
        if offset is None or offset.y >= len(_ANSI_COLOR_TOKENS):
            return
        event.stop()
        event.prevent_default()
        self.focus(scroll_visible=False)
        self._select_index(offset.y, preview=True)

    def _select_index(self, index: int, *, preview: bool) -> None:
        # No-op guard avoids a full CSS reparse when the user re-selects
        # the current row (arrow clamped at edge, click on selected row).
        if index == self._selected_index and self._selection_explicit:
            return
        self._selected_index = index
        self._selection_explicit = True
        self.refresh()
        if preview:
            self.post_message(self.Changed(self, self.token))


class _Xterm256PalettePicker(Widget, can_focus=True):
    """Select exact xterm entries without RGB quantization on each input event."""

    ALLOW_SELECT = False
    COMPONENT_CLASSES: ClassVar[set[str]] = {"palette--transparent", "palette--transparent-selected"}
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("left", "move(-1, 0)", show=False),
        Binding("right", "move(1, 0)", show=False),
        Binding("up", "move(0, -1)", show=False),
        Binding("down", "move(0, 1)", show=False),
        Binding("space", "toggle_transparent", show=False),
    ]

    _CELL_WIDTH = 4

    def __init__(
        self,
        color: Color,
        *,
        allow_transparent: bool = True,
        disabled_hint: MessageRef | str | None = None,
        context: ColorEditContext | None = None,
    ) -> None:
        super().__init__()
        self.context = context
        # See ``_XTERM_256_ROWS_BASE`` for why callers disable transparent.
        # With ``disabled_hint`` we keep the "Transparent" header visible
        # and paint the hint below it, explaining why the option is unavailable.
        # Without a hint, the section is removed silently.
        self._disabled_hint = disabled_hint if not allow_transparent else None
        if allow_transparent:
            self._rows = _XTERM_256_ROWS
            position_map = _XTERM_256_POSITION_BY_INDEX
        elif self._disabled_hint:
            self._rows = _XTERM_256_ROWS_WITH_HINT
            position_map = _XTERM_256_POSITION_BY_INDEX_BASE
        else:
            self._rows = _XTERM_256_ROWS_BASE
            position_map = _XTERM_256_POSITION_BY_INDEX_BASE
        self._selectable_rows = tuple(row for row, (_, indices) in enumerate(self._rows) if indices)
        self._selected_index = self._match_palette_index(color, allow_transparent=allow_transparent)
        self._selected_row, self._selected_column = position_map[self._selected_index]
        self._selected_color = (
            Color(0, 0, 0, a=0)
            if self._selected_index == _TRANSPARENT_SENTINEL
            else _palette_color(self._selected_index)
        )
        # Preserve off-palette/alpha colors when transparency is toggled off.
        # A transparent seed has no earlier color, so start with opaque white.
        self._last_nontransparent_color = Color(255, 255, 255) if color.is_transparent else color
        # A nearest-color highlight is not an explicit edit. Mounting preserves
        # the original; clicking that same cell must select its exact color.
        self._selection_explicit = color == self.color

    @property
    def color(self) -> Color:
        return self._selected_color

    @property
    def selection_region(self) -> Region:
        return Region(self._selected_column * self._CELL_WIDTH, self._selected_row, self._CELL_WIDTH, 1)

    def refresh_localization(self) -> None:
        self.refresh()

    def render_line(self, y: int) -> Strip:
        # CSS height matches the full layout; shorter layouts pad with
        # ``rich_style`` so the parent surface doesn't bleed through.
        if y >= len(self._rows):
            return Strip.blank(self.size.width, style=self.rich_style)
        label, indices = self._rows[y]
        if _TRANSPARENT_SENTINEL in indices:
            selected = self._selected_index == _TRANSPARENT_SENTINEL
            marker = CHECKED_MARKER if selected else UNCHECKED_MARKER
            component = "palette--transparent-selected" if selected else "palette--transparent"
            return Strip(
                [
                    Segment(marker, self.get_component_rich_style(component)),
                    Segment(f" {M.text(self, M.TRANSPARENT)}", self.rich_style),
                ]
            )
        # WITH_HINT placeholder row — only emitted when ``_disabled_hint`` is set.
        if label is None and not indices:
            if self._disabled_hint is None:
                raise RuntimeError("A disabled palette row requires hint text.")
            hint = self._disabled_hint
            text = render_str(widget_localizer(self), hint) if isinstance(hint, MessageRef) else hint
            return Strip([Segment(text, Style(dim=True, italic=True))])
        if label is not None:
            definition = {
                "Color cube (hue grouped)": M.CUBE,
                "Grayscale": M.GRAYSCALE,
                "Transparent": M.TRANSPARENT,
            }.get(label)
            display = M.text(self, definition) if definition is not None else label
            return Strip([Segment(display, Style(dim=not label, bold=bool(label)))])

        segments: list[Segment] = []
        for index in indices:
            text = f"{index:03d} " if index == self._selected_index else " " * self._CELL_WIDTH
            segments.append(Segment(text, _XTERM_CELL_STYLES[index]))
        return Strip(segments)

    def render(self) -> Text:
        if self._selected_index == _TRANSPARENT_SENTINEL:
            return Text("transparent")
        return Text(f"{self._selected_index:03d} {self.color.hex}")

    def action_move(self, dx: int, dy: int) -> None:
        if dy:
            row = self._move_row(dy)
            column = min(self._selected_column, len(self._rows[row][1]) - 1)
            self._select_cell(row, column, preview=True)
            return
        row_indices = self._rows[self._selected_row][1]
        column = min(max(self._selected_column + dx, 0), len(row_indices) - 1)
        self._select_cell(self._selected_row, column, preview=True)

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1:
            return
        offset = event.get_content_offset(self)
        if offset is None:
            return
        if offset.y >= len(self._rows):
            return
        indices = self._rows[offset.y][1]
        if not indices:
            return
        if _TRANSPARENT_SENTINEL in indices:
            if offset.x >= self.render_line(offset.y).cell_length:
                return
            event.stop()
            event.prevent_default()
            self.focus(scroll_visible=False)
            self.action_toggle_transparent()
            return
        column = offset.x // self._CELL_WIDTH
        if column >= len(indices):
            return
        event.stop()
        event.prevent_default()
        self.focus(scroll_visible=False)
        self._select_cell(offset.y, column, preview=True)

    def _move_row(self, dy: int) -> int:
        current_position = self._selectable_rows.index(self._selected_row)
        next_position = min(max(current_position + dy, 0), len(self._selectable_rows) - 1)
        return self._selectable_rows[next_position]

    def action_toggle_transparent(self) -> None:
        row = next((row for row, (_, indices) in enumerate(self._rows) if _TRANSPARENT_SENTINEL in indices), None)
        if row is None:
            return
        if self._selected_index != _TRANSPARENT_SENTINEL:
            self._select_cell(row, 0, preview=True)
        else:
            color = self._last_nontransparent_color
            index = self._match_palette_index(color, allow_transparent=False)
            row, column = _XTERM_256_POSITION_BY_INDEX[index]
            self._select_cell(row, column, preview=True, restored_color=color)

    def _select_cell(self, row: int, column: int, *, preview: bool, restored_color: Color | None = None) -> None:
        # See ``_AnsiTokenPicker._select_index`` for the no-op rationale.
        if (row, column) == (self._selected_row, self._selected_column) and self._selection_explicit:
            return
        self._selected_row = row
        self._selected_column = column
        self._selected_index = self._rows[row][1][column]
        cell_color = (
            Color(0, 0, 0, a=0)
            if self._selected_index == _TRANSPARENT_SENTINEL
            else _palette_color(self._selected_index)
        )
        self._selected_color = cell_color if restored_color is None else restored_color
        self._selection_explicit = self.color == cell_color
        if not self.color.is_transparent:
            self._last_nontransparent_color = self.color
        self.refresh()
        if preview:
            self.post_message(PaletteChanged(self, self.color))

    @staticmethod
    def _match_palette_index(color: Color, *, allow_transparent: bool = True) -> int:
        if color.is_transparent and allow_transparent:
            return _TRANSPARENT_SENTINEL
        # Transparent input with no sentinel cell falls through to the
        # nearest palette entry — BASE map has no sentinel key.
        return EIGHT_BIT_PALETTE.match((color.r, color.g, color.b))
