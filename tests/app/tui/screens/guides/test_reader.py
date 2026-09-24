# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the user-guide markdown reader (H1, loading, display names)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.app.tui.screens.guides.index import GuideIndex, GuideTopic, load_guide_index
from chrys.app.tui.screens.guides.reader import (
    branch_display_name,
    extract_h1,
    read_topic_markdown,
    resolve_guide_link,
    topic_display_name,
)

_INDEX = """\
locales:
  - zh-Hans
  - en
topics:
  - id: intro
    path: start/what-is-chrys.md
"""


def _docs(tmp_path: Path) -> Path:
    (tmp_path / "index.yaml").write_text(_INDEX, encoding="utf-8")
    (tmp_path / "zh-Hans" / "start").mkdir(parents=True)
    (tmp_path / "zh-Hans" / "start" / "what-is-chrys.md").write_text("# 什么是 Chrys\n\n正文内容。", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("# Title\n\nbody", "Title"),
        ("  #  Spaced  \nbody", "Spaced"),
        ("## Subheading\nbody", None),
        ("no heading at all", None),
        ("", None),
    ],
)
def test_extract_h1(markdown: str, expected: str | None) -> None:
    assert extract_h1(markdown) == expected


def test_read_topic_markdown_reads_current_locale(tmp_path: Path) -> None:
    index = load_guide_index(_docs(tmp_path))
    topic = index.topics[0]

    assert read_topic_markdown(tmp_path, index, topic, "zh-Hans") == "# 什么是 Chrys\n\n正文内容。"


def test_read_topic_markdown_missing_locale_returns_none(tmp_path: Path) -> None:
    index = load_guide_index(_docs(tmp_path))

    assert read_topic_markdown(tmp_path, index, index.topics[0], "en") is None


def test_read_topic_markdown_missing_file_returns_none(tmp_path: Path) -> None:
    index = load_guide_index(_docs(tmp_path))
    missing = GuideTopic(id="missing", path="start/nope.md", labels=None, children=())

    assert read_topic_markdown(tmp_path, index, missing, "zh-Hans") is None


def test_read_topic_markdown_refuses_escape_above_locale_root(tmp_path: Path) -> None:
    index = load_guide_index(_docs(tmp_path))
    (tmp_path / "secret.md").write_text("secret", encoding="utf-8")
    escaping = GuideTopic(id="escape", path="../secret.md", labels=None, children=())

    assert read_topic_markdown(tmp_path, index, escaping, "zh-Hans") is None


def test_read_topic_markdown_refuses_locale_root_above_docs_root(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (tmp_path / "secret.md").write_text("secret", encoding="utf-8")
    index = GuideIndex(locales=("..",), default_topic_id=None, topics=())
    topic = GuideTopic(id="escape", path="secret.md", labels=None, children=())

    assert read_topic_markdown(docs_root, index, topic, "..") is None


def test_topic_display_name_fallback_chain() -> None:
    labeled = GuideTopic(id="a", path="dir/name.md", labels={"zh-Hans": "标签"}, children=())
    assert topic_display_name(labeled, locale="zh-Hans", default_locale="zh-Hans", current_h1=None) == "标签"

    unlabeled = GuideTopic(id="a", path="dir/name.md", labels=None, children=())
    assert topic_display_name(unlabeled, locale="zh-Hans", default_locale="zh-Hans", current_h1="当前H1") == "当前H1"
    # No content in the active language: degrade to the file name, never to
    # another language's H1.
    assert topic_display_name(unlabeled, locale="en", default_locale="zh-Hans", current_h1=None) == "name"
    id_only = GuideTopic(id="a", path=None, labels=None, children=())
    assert topic_display_name(id_only, locale="zh-Hans", default_locale="zh-Hans", current_h1=None) == "a"


def test_labels_resolve_current_locale_then_default_locale() -> None:
    bilingual = GuideTopic(id="a", path="dir/name.md", labels={"zh-Hans": "配置", "en": "Configuration"}, children=())

    assert (
        topic_display_name(bilingual, locale="en", default_locale="zh-Hans", current_h1="连接 MCP") == "Configuration"
    )
    assert topic_display_name(bilingual, locale="zh-Hans", default_locale="zh-Hans", current_h1="连接 MCP") == "配置"
    # Labels beat the H1; unknown locales fall back to the default locale.
    assert topic_display_name(bilingual, locale="de", default_locale="zh-Hans", current_h1="连接 MCP") == "配置"

    default_only = GuideTopic(id="a", path="dir/name.md", labels={"zh-Hans": "默认标签"}, children=())
    assert topic_display_name(default_only, locale="en", default_locale="zh-Hans", current_h1="H1") == "默认标签"
    assert topic_display_name(default_only, locale="de", default_locale="de", current_h1="H1") == "H1"


def test_branch_display_name_label_override() -> None:
    branch = GuideTopic(
        id="configuration",
        path=None,
        labels={"zh-Hans": "配置", "en": "Configuration"},
        children=(
            GuideTopic(id="mcp", path="guides/configuration/mcp.md", labels=None, children=()),
            GuideTopic(id="tools", path="guides/configuration/tools.md", labels=None, children=()),
        ),
    )

    assert branch_display_name(branch, locale="zh-Hans", default_locale="zh-Hans") == "配置"
    assert branch_display_name(branch, locale="en", default_locale="zh-Hans") == "Configuration"


def test_branch_display_name_common_directory_fallback() -> None:
    branch = GuideTopic(
        id="daily_use",
        path=None,
        labels=None,
        children=(
            GuideTopic(id="workspaces", path="guides/daily-use/workspaces.md", labels=None, children=()),
            GuideTopic(id="sessions", path="guides/daily-use/sessions.md", labels=None, children=()),
        ),
    )

    assert branch_display_name(branch, locale="en", default_locale="zh-Hans") == "daily-use"


def test_branch_display_name_id_fallback() -> None:
    empty_branch = GuideTopic(id="empty", path=None, labels=None, children=())
    assert branch_display_name(empty_branch, locale="zh-Hans", default_locale="zh-Hans") == "empty"

    scattered = GuideTopic(
        id="scattered",
        path=None,
        labels=None,
        children=(
            GuideTopic(id="a", path="guides/a/one.md", labels=None, children=()),
            GuideTopic(id="b", path="docs/b/two.md", labels=None, children=()),
        ),
    )
    assert branch_display_name(scattered, locale="zh-Hans", default_locale="zh-Hans") == "scattered"


def _link_index() -> GuideIndex:
    """A topic tree mirroring the docs layout for link-resolution tests."""
    intro = GuideTopic(id="intro", path="start/what-is-chrys.md", labels=None, children=())
    getting_started = GuideTopic(id="getting_started", path="start/getting-started.md", labels=None, children=())
    mcp = GuideTopic(id="mcp", path="guides/configuration/mcp.md", labels=None, children=())
    agents = GuideTopic(id="agents", path="guides/configuration/agents.md", labels=None, children=())
    configuration = GuideTopic(id="configuration", path=None, labels=None, children=(mcp, agents))
    return GuideIndex(
        locales=("zh-Hans", "en"),
        default_topic_id="intro",
        topics=(intro, getting_started, configuration),
    )


def test_resolve_guide_link_same_directory() -> None:
    index = _link_index()
    target, anchor = resolve_guide_link(index, index.topics[0], "getting-started.md")

    assert target is not None and target.id == "getting_started"
    assert anchor == ""


def test_resolve_guide_link_same_directory_dot_prefix() -> None:
    index = _link_index()
    target, _anchor = resolve_guide_link(index, index.topics[0], "./getting-started.md")

    assert target is not None and target.id == "getting_started"


def test_resolve_guide_link_cross_directory() -> None:
    index = _link_index()
    target, _anchor = resolve_guide_link(index, index.topics[0], "../guides/configuration/mcp.md")

    assert target is not None and target.id == "mcp"


def test_resolve_guide_link_anchor_split() -> None:
    index = _link_index()
    target, anchor = resolve_guide_link(index, index.topics[0], "../guides/configuration/mcp.md#连接-mcp-服务器")

    assert target is not None and target.id == "mcp"
    assert anchor == "连接-mcp-服务器"


def test_resolve_guide_link_page_internal_anchor_is_not_cross_document() -> None:
    index = _link_index()
    target, anchor = resolve_guide_link(index, index.topics[0], "#为智能体配置子智能体")

    assert target is None
    assert anchor == ""


@pytest.mark.parametrize(
    "href",
    [
        "https://example.com/guide.md",
        "http://localhost/docs.md",
        "mailto:someone@example.com",
        "",
    ],
)
def test_resolve_guide_link_external_or_empty_href_never_resolves(href: str) -> None:
    index = _link_index()
    assert resolve_guide_link(index, index.topics[0], href) == (None, "")


def test_resolve_guide_link_unindexed_file_returns_none() -> None:
    index = _link_index()
    # ../acp/chrys-acp.md exists on disk but is not in the topic tree.
    target, anchor = resolve_guide_link(index, index.topics[0], "../acp/chrys-acp.md")

    assert target is None
    assert anchor == ""


def test_resolve_guide_link_escape_above_root_never_matches() -> None:
    index = _link_index()
    assert resolve_guide_link(index, index.topics[0], "../../secret.md") == (None, "")
    assert resolve_guide_link(index, index.topics[0], "/absolute/guide.md") == (None, "")


def test_resolve_guide_link_branch_source_returns_none() -> None:
    index = _link_index()
    branch = index.topics[2]
    assert branch.is_branch
    assert resolve_guide_link(index, branch, "mcp.md") == (None, "")
