# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Validate candidate themes without touching the running application's styles.

The adapter targets the project's pinned Textual 8.2.7. Private CSS invalidation
and source metadata live here; widgets only submit whole Theme snapshots. The
application suppresses preference writes while an editor preview is active.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.color import Color
from textual.css.stylesheet import Stylesheet, StylesheetError, StylesheetParseError
from textual.css.tokenizer import TokenError
from textual.theme import Theme

from chrys.app.tui.i18n import render_str
from chrys.app.tui.theme import with_tui_css_variables
from chrys.foundation.i18n import Localizer, msg

from .document import SURFACES, copy_theme

if TYPE_CHECKING:
    from chrys.app.tui.app import ChrysApp

_INVALID_CSS = msg("tui.theme_editor.invalid_css", fallback="Invalid CSS value{field}: {value}")


def theme_variables(theme: Theme, defaults: dict[str, str]) -> dict[str, str]:
    variables = with_tui_css_variables(theme, {**defaults, **theme.to_color_system().generate(), **theme.variables})
    return variables


def css_error(error: Exception, localizer: Localizer | None = None) -> str:
    if isinstance(error, StylesheetParseError):
        # Textual's HelpText summaries embed Rich markup alongside raw user
        # values. Build a plain diagnostic from tokens instead of interpreting
        # that mixed string or exposing its formatting tags in a Label.
        messages = []
        localizer = localizer or Localizer("en")
        for rule in error.errors.rules:
            for token, _message in rule.errors:
                reference = token.referenced_by
                variable = f" (${reference.name})" if reference is not None else ""
                messages.append(render_str(localizer, _INVALID_CSS.bind(field=variable, value=repr(token.value))))
        return "; ".join(dict.fromkeys(messages))[:700]
    return str(error)[:700] or type(error).__name__


def validate_theme(theme: Theme, defaults: dict[str, str], live: Stylesheet) -> dict[str, str]:
    """Check only styles already in memory, without touching the live sheet.

    New screen/widget styles are checked by ThemeStylesheet when loaded; the
    preview's recovery hook keeps the previous valid theme if they reject it.
    Never scan or parse application/library source files to open the editor.
    """
    for name in SURFACES:
        value = getattr(theme, name)
        if value is not None and Color.parse(value).a != 1:
            # TextArea's Rich fallback may consume Theme.surface directly,
            # outside CSS parsing. Keep the editor's existing opaque-surface policy.
            raise ValueError(f"{name} requires an opaque color")
    variables = theme_variables(theme, defaults)
    probe = Stylesheet(variables=variables)
    probe.source.update(live.source)
    probe.parse()
    return variables


class ThemePreview:
    """Last-valid publication with bounded history for newly discovered CSS."""

    def __init__(self, app: ChrysApp) -> None:
        self.app = app
        # Opening the editor adopts the already displayed theme. Revalidating
        # and republishing it would parse all loaded CSS and restyle the entire
        # transcript before the panel can paint, despite no theme change.
        self.current: Theme | None = copy_theme(app.current_theme)
        self.variables: dict[str, str] | None = app.get_css_variables().copy()
        self.history: list[Theme] = []
        self.baseline: Theme | None = copy_theme(app.current_theme)
        self._error: Exception | None = None
        self._error_field = ""
        self.field = ""

    @property
    def error(self) -> str:
        """Render retained diagnostics in the active language without revalidating."""
        if self._error is None:
            return ""
        diagnostic = css_error(self._error, self.app.locale_controller.localizer)
        return f"{self._error_field}: {diagnostic}" if self._error_field else diagnostic

    def checkpoint(self) -> None:
        """Retain the pre-gesture theme even after a long stream of drag updates."""
        if self.current is not None:
            self.baseline = copy_theme(self.current)

    def check(self, theme: Theme, field: str = "") -> dict[str, str] | None:
        try:
            result = validate_theme(theme, self.app.get_theme_variable_defaults(), self.app.stylesheet)
        except (StylesheetError, TokenError, ValueError, TypeError, AttributeError) as error:
            self._error = error
            self._error_field = field
            return None
        self._error = None
        return result

    def show(self, theme: Theme, field: str = "") -> bool:
        if theme == self.current:
            self._error = None
            return True
        variables = self.check(theme, field)
        if variables is None:
            return False
        if field != self.field:
            self.checkpoint()
        if self.current is not None:
            self.history.append(copy_theme(self.current))
            del self.history[:-20]
        self.field = field
        self.current = copy_theme(theme)
        self.variables = variables
        self._refresh()
        return True

    def _refresh(self) -> None:
        theme = self.current
        if theme is None:
            raise RuntimeError("Refreshing the theme preview requires a current theme.")
        app = self.app
        classes = {name: False for name in app.classes if name.startswith("-theme-")}
        classes.update(
            {
                f"-theme-{theme.name}": True,
                "-dark-mode": theme.dark,
                "-light-mode": not theme.dark,
            }
        )
        app.update_classes(classes, update=False)
        app._refresh_truecolor_filter(app.ansi_theme)
        app._invalidate_css()
        app.refresh_css(animate=False)
        app.theme_changed_signal.publish(theme)

    def recover(self, stylesheet: Stylesheet, error: Exception) -> bool:
        """A lazily introduced style rejects the candidate, without theme switching.

        Ordinary application syntax errors still propagate: none of the previous
        themes can make them valid. Never delegate an editing failure to Chrys's
        production fallback-to-chrys behavior.
        """
        candidates = [*reversed(self.history)]
        if self.baseline is not None:
            candidates.append(self.baseline)
        for previous in candidates:
            try:
                variables = validate_theme(previous, self.app.get_theme_variable_defaults(), stylesheet)
            except StylesheetError, TokenError, ValueError, TypeError, AttributeError:
                continue
            self._error = error
            self._error_field = self.field
            self.current = copy_theme(previous)
            self.variables = variables
            stylesheet.set_variables(variables)
            self.app.call_next(self._refresh)
            return True
        return False
