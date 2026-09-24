# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Heading anchors: GitHub's ids first, then Textual's slug, then the raw title."""

from __future__ import annotations

import pytest

from chrys.app.tui.widgets.markdown.anchors import find_heading, github_slug


@pytest.mark.parametrize(
    ("title", "slug"),
    [
        ("Connect MCP servers", "connect-mcp-servers"),
        ("智能体、模型与请求", "智能体模型与请求"),
        ("在本机启动 iCode", "在本机启动-icode"),
        ("Use `icode run`", "use-icode-run"),
        ("Hooks: before_tool_call (gated)", "hooks-before_tool_call-gated"),
        ("A -- B", "a----b"),
        ("x² and ½", "x-and-"),
        ("Ⅻ‿end", "ⅻ‿end"),
    ],
)
def test_github_slug_keeps_word_characters_hyphens_and_spaces(title: str, slug: str) -> None:
    assert github_slug(title) == slug


def test_duplicate_headings_are_numbered_like_github() -> None:
    titles = ["Usage", "Usage", "Usage-1", "Usage"]
    assert find_heading(titles, "usage") == 0
    assert find_heading(titles, "usage-1") == 1
    # "Usage-1" collides with the second heading's id, so it is numbered in turn.
    assert find_heading(titles, "usage-1-1") == 2
    assert find_heading(titles, "usage-2") == 3


def test_github_id_wins_over_an_earlier_textual_slug() -> None:
    # Textual's slug for the first heading drops its CJK text and leaves
    # "-icode", which is GitHub's id for the second heading.
    titles = ["在本机启动 iCode", "· iCode"]
    assert find_heading(titles, "-icode") == 1


def test_textual_slug_and_raw_title_are_fallbacks() -> None:
    titles = ["在本机启动 iCode", "What's new?"]
    assert find_heading(titles, "-icode") == 0
    assert find_heading(titles, "What's new?") == 1


@pytest.mark.parametrize("fragment", ["", "missing"])
def test_unknown_or_empty_fragment_finds_nothing(fragment: str) -> None:
    assert find_heading(["Intro"], fragment) is None
