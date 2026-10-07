# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: use empty style for block border outer when parent bg is terminal default.

Problem
-------
The ``block`` border type uses half-block characters (▄/▀) whose "other
half" is rendered with the parent widget's background.  When the parent
background is ``ansi_default``, Textual emits ``Style(bgcolor=default)``
which Rich renders as ``\\x1b[49m`` (SGR default background).  Some
terminals render this as opaque black instead of true transparent for
half-block characters, creating visible black lines in block borders.

Solution
--------
When ``base_background`` is ``ansi_default`` (ansi == -1) or fully
transparent (a == 0), return an empty ``Style()`` for the outer style.
An empty style has no bgcolor at all — Rich emits no background escape
code, letting the terminal render its native background in those cells.

Widget imports StylesCache before Chrys bootstrap. Replace the live cached
method as well, so the first process does not retain the old method or results.

The live method also fixes a retention: Textual caches ``get_inner_outer`` per
instance, so its 1024-entry LRU keeps that many StylesCaches of removed widgets
alive, each with its rendered lines. The result depends only on the colors, so
the replacement caches on those alone.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from chrys.foundation.patches.patcher import FilePatch, register

_RUNTIME_PATCH_MARKER = "_chrys_transparent_block_border"
_RUNTIME_PATCH_TEXTUAL_VERSION = "8.2.7"
logger = logging.getLogger(__name__)


def apply_runtime_patch() -> None:
    """Update already-imported StylesCache classes and discard stale results."""
    try:
        import textual
        from textual._styles_cache import StylesCache
        from textual.style import Style
    except ImportError:
        return
    if textual.__version__ != _RUNTIME_PATCH_TEXTUAL_VERSION:
        logger.warning("Skipping block border patch for unsupported Textual %s", textual.__version__)
        return
    if getattr(StylesCache.get_inner_outer, _RUNTIME_PATCH_MARKER, False):
        return

    @lru_cache(1024)
    def inner_outer(base_background: Any, background: Any) -> tuple[Any, Any]:
        is_default = getattr(base_background, "ansi", None) == -1
        outer = Style() if (is_default or base_background.a == 0) else Style(background=base_background)
        return Style(background=base_background + background), outer

    def get_inner_outer(self: Any, base_background: Any, background: Any) -> tuple[Any, Any]:
        return inner_outer(base_background, background)

    setattr(get_inner_outer, _RUNTIME_PATCH_MARKER, True)
    StylesCache.get_inner_outer = get_inner_outer


_OLD = """\
    def get_inner_outer(
        cls, base_background: Color, background: Color
    ) -> tuple[Style, Style]:
        \"\"\"Get inner and outer background colors.\"\"\"
        return (
            Style(background=base_background + background),
            Style(background=base_background),
        )"""

_NEW = """\
    def get_inner_outer(
        cls, base_background: Color, background: Color
    ) -> tuple[Style, Style]:
        \"\"\"Get inner and outer background colors.\"\"\"
        # --- Chrys patch: transparent parent → empty outer style ---
        # When base_background is ansi_default or transparent, emit no
        # bgcolor so the terminal renders its native background.
        is_default = getattr(base_background, "ansi", None) == -1
        outer = Style() if (is_default or base_background.a == 0) else Style(background=base_background)
        return (
            Style(background=base_background + background),
            outer,
        )"""

register(
    FilePatch(
        package="textual",
        module_file="_styles_cache.py",
        old_fragment=_OLD,
        new_fragment=_NEW,
        description="Use empty style for block border outer on transparent terminals",
    ),
)
