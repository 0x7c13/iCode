# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Theme-adaptive, background-transparent syntax highlighting helpers.

``rich`` ``Syntax`` bakes a Pygments theme background into every token, which
clashes with the Textual surface and looks like a hard black box on light
themes. :class:`TransparentPygmentsTheme` strips those backgrounds so the
widget's own (theme-driven) background shows through, and :func:`transparent_syntax`
picks a Pygments style that matches the active dark/light theme.
"""

from __future__ import annotations

from pygments.token import Error, Generic
from rich.style import Style as RichStyle
from rich.syntax import PygmentsSyntaxTheme, Syntax
from textual.highlight import HighlightTheme


class NoErrorHighlightTheme(HighlightTheme):
    """Textual highlight theme that does not style Pygments error tokens."""

    STYLES: dict[tuple[str, ...], str] = {  # noqa: RUF012 - Textual requires a mutable theme mapping.
        token_type: style
        for token_type, style in HighlightTheme.STYLES.items()
        if token_type not in {Error, Generic.Error}
    }


class TransparentPygmentsTheme(PygmentsSyntaxTheme):
    """``PygmentsSyntaxTheme`` variant that strips **all** ``bgcolor`` values.

    ``PygmentsSyntaxTheme.get_style_for_token`` falls back to the theme's
    ``background_color`` for every token (even ones that don't define a
    background), which bleeds the Pygments background into every segment.
    This subclass removes that fallback so the widget's own background
    (from the Textual theme) shows through instead.
    """

    def __init__(self, theme: str) -> None:
        super().__init__(theme)
        self._clean_cache: dict[tuple[str, ...], RichStyle] = {}

    def get_style_for_token(self, token_type: tuple[str, ...]) -> RichStyle:
        try:
            return self._clean_cache[token_type]
        except KeyError:
            style = super().get_style_for_token(token_type)
            if style.bgcolor is not None:
                style = RichStyle(
                    color=style.color,
                    bold=style.bold,
                    dim=style.dim,
                    italic=style.italic,
                    underline=style.underline,
                    overline=style.overline,
                    strike=style.strike,
                    link=style.link,
                )
            self._clean_cache[token_type] = style
            return style

    def get_background_style(self) -> RichStyle:
        return RichStyle.null()


def transparent_syntax(
    code: str,
    language: str,
    *,
    dark: bool,
    line_numbers: bool = False,
    start_line: int = 1,
    word_wrap: bool = True,
) -> Syntax:
    """Build a ``Syntax`` with a transparent, theme-matched Pygments style."""
    theme = TransparentPygmentsTheme("native" if dark else "default")
    return Syntax(
        code,
        language,
        theme=theme,
        line_numbers=line_numbers,
        start_line=start_line,
        word_wrap=word_wrap,
    )
