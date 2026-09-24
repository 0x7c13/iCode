# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the background-transparent syntax theme helpers."""

from __future__ import annotations

from pygments.token import Token
from rich.style import Style
from rich.syntax import Syntax
from textual.highlight import HighlightTheme

from chrys.app.tui.widgets.syntax_theme import NoErrorHighlightTheme, TransparentPygmentsTheme, transparent_syntax


def test_background_style_is_null() -> None:
    assert TransparentPygmentsTheme("native").get_background_style() == Style.null()


def test_token_styles_drop_background() -> None:
    theme = TransparentPygmentsTheme("native")
    # Keyword normally carries the theme background; it must be stripped.
    assert theme.get_style_for_token(Token.Keyword).bgcolor is None
    assert theme.get_style_for_token(Token.Name).bgcolor is None


def test_no_error_highlight_theme_drops_error_token_styles() -> None:
    assert Token.Error in HighlightTheme.STYLES
    assert Token.Generic.Error in HighlightTheme.STYLES
    assert Token.Error not in NoErrorHighlightTheme.STYLES
    assert Token.Generic.Error not in NoErrorHighlightTheme.STYLES


def test_transparent_syntax_returns_syntax_with_options() -> None:
    syntax = transparent_syntax("x = 1\n", "python", dark=True, line_numbers=True, start_line=5)
    assert isinstance(syntax, Syntax)
    assert isinstance(syntax._theme, TransparentPygmentsTheme)


def test_transparent_syntax_highlight_has_no_token_backgrounds() -> None:
    syntax = transparent_syntax("def f():\n    return 1\n", "python", dark=False)
    highlighted = syntax.highlight("def f():\n    return 1\n")
    for span in highlighted.spans:
        if isinstance(span.style, Style):
            assert span.style.bgcolor is None
