# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Heading anchors — which ``#fragment`` a markdown heading answers to.

Links are written for GitHub, whose heading ids keep every letter, CJK
included, and drop punctuation: "智能体、模型与请求" is ``#智能体模型与请求``
and "在本机启动 iCode" is ``#在本机启动-icode``. Textual's own slug removes
CJK entirely, so under it those two headings would become ``""`` and
``-icode``, and two different headings could share an id. A fragment is
therefore matched against GitHub's id first, then Textual's slug, then the
raw heading title.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from textual._slug import TrackedSlugs

# Besides letters and marks, GitHub keeps decimal digits, letter numbers
# (Ⅻ) and connector punctuation (``_``), but drops other numbers (², ½).
_KEPT_CATEGORIES = frozenset({"Nd", "Nl", "Pc"})


def github_slug(title: str) -> str:
    """Return GitHub's id for a heading, before duplicate numbering.

    Lowercase, keep word characters, ``-`` and spaces, and turn each space
    into a hyphen.
    """
    kept = "".join(char for char in title.lower() if char in "- " or _is_word_character(char))
    return kept.replace(" ", "-")


def _is_word_character(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in "LM" or category in _KEPT_CATEGORIES


class _GitHubSlugs:
    """GitHub's duplicate numbering: the second "Usage" becomes ``usage-1``."""

    def __init__(self) -> None:
        self._occurrences: dict[str, int] = {}

    def slug(self, title: str) -> str:
        original = result = github_slug(title)
        while result in self._occurrences:
            self._occurrences[original] += 1
            result = f"{original}-{self._occurrences[original]}"
        self._occurrences[result] = 0
        return result


def find_heading(titles: Sequence[str], fragment: str) -> int | None:
    """Return the index of the heading *fragment* names, or ``None``.

    *titles* are the plain heading texts in document order, *fragment* the
    decoded link target after ``#``.
    """
    if not fragment:
        return None
    github = _GitHubSlugs()
    textual = TrackedSlugs()
    keys = [(github.slug(title), textual.slug(title), title) for title in titles]
    for tier in range(3):
        for index, key in enumerate(keys):
            if key[tier] == fragment:
                return index
    return None
