# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Recover user-theme failures at Textual's actual CSS parsing boundary."""

from __future__ import annotations

from collections.abc import Callable

from textual.css.stylesheet import Stylesheet, StylesheetError
from textual.css.tokenizer import TokenError


class ThemeStylesheet(Stylesheet):
    """Cover initial parsing, theme refreshes, and lazily mounted widget CSS."""

    def __init__(self, *, variables: dict[str, str], recover: Callable[[Stylesheet, Exception], bool]) -> None:
        super().__init__(variables=variables)
        self._recover = recover

    def copy(self) -> ThemeStylesheet:
        # Textual replaces the live stylesheet with a copy after CSS hot reload.
        stylesheet = ThemeStylesheet(variables=self._variables.copy(), recover=self._recover)
        stylesheet.source = self.source.copy()
        return stylesheet

    def parse(self) -> None:
        try:
            super().parse()
        except (StylesheetError, TokenError) as error:
            if not self._recover(self, error):
                raise
            super().parse()

    def reparse(self) -> None:
        # Textual's reparse uses a plain temporary Stylesheet internally.
        try:
            super().reparse()
        except (StylesheetError, TokenError) as error:
            if not self._recover(self, error):
                raise
            super().reparse()
