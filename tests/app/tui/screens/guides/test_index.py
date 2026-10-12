# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the user-guide docs index (``docs/index.yaml``) parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.app.tui.screens.guides.index import (
    GuideIndexError,
    default_topic_id,
    iter_leaf_topics,
    language_cycle,
    load_guide_index,
)

_VALID_INDEX = """\
default: mcp
locales:
  - zh-Hans
  - en
topics:
  - id: intro
    path: start/what-is-chrys.md
  - id: configuration
    children:
      - id: mcp
        path: guides/configuration/mcp.md
"""


def _write(docs_root: Path, content: str) -> Path:
    docs_root.mkdir(parents=True, exist_ok=True)
    (docs_root / "index.yaml").write_text(content, encoding="utf-8")
    return docs_root


def test_parse_preserves_order_and_structure(tmp_path: Path) -> None:
    index = load_guide_index(_write(tmp_path, _VALID_INDEX))

    assert index.locales == ("zh-Hans", "en")
    assert index.default_topic_id == "mcp"
    assert len(index.topics) == 2
    intro, configuration = index.topics
    assert intro.is_leaf and intro.path == "start/what-is-chrys.md"
    assert configuration.is_branch
    assert configuration.children[0].id == "mcp"
    assert configuration.children[0].path == "guides/configuration/mcp.md"


def test_language_cycle_follows_locales_order(tmp_path: Path) -> None:
    index = load_guide_index(_write(tmp_path, _VALID_INDEX))

    assert language_cycle(index) == ("zh-Hans", "en")


def test_default_topic_id_prefers_explicit_then_first_leaf(tmp_path: Path) -> None:
    index = load_guide_index(_write(tmp_path, _VALID_INDEX))
    assert default_topic_id(index) == "mcp"

    explicit_default = _VALID_INDEX.replace("default: mcp", "default: intro")
    assert default_topic_id(load_guide_index(_write(tmp_path, explicit_default))) == "intro"

    no_default = "\n".join(line for line in _VALID_INDEX.splitlines() if not line.startswith("default:"))
    assert default_topic_id(load_guide_index(_write(tmp_path, no_default))) == "intro"


def test_default_topic_falls_back_when_id_unknown(tmp_path: Path) -> None:
    unknown = _VALID_INDEX.replace("default: mcp", "default: nope")
    assert default_topic_id(load_guide_index(_write(tmp_path, unknown))) == "intro"


def test_iter_leaf_topics_walks_document_order(tmp_path: Path) -> None:
    index = load_guide_index(_write(tmp_path, _VALID_INDEX))

    leaves = [(topic_id, topic.path) for topic_id, topic in iter_leaf_topics(index.topics)]
    assert leaves == [
        ("intro", "start/what-is-chrys.md"),
        ("mcp", "guides/configuration/mcp.md"),
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("topics:\n  - path: a.md\n", "missing 'locales'"),
        ("locales:\n  - zh-Hans\n", "missing 'topics'"),
        ("locales: []\ntopics: []\n", "must list at least one locale"),
        ("locales:\n  - en\n  - ' en '\ntopics: []\n", "duplicate locale 'en'"),
        ("locales:\n  - zh-Hans\ntopics:\n  - path: a.md\n    children: []\n", "both 'path' and 'children'"),
        (
            "locales:\n  - zh-Hans\ntopics:\n  - id: x\n",
            r"needs 'path' \(leaf\) or 'children' \(branch\)",
        ),
        ("not-a-mapping\n", "must be a mapping"),
    ],
)
def test_malformed_index_raises(tmp_path: Path, content: str, message: str) -> None:
    with pytest.raises(GuideIndexError, match=message):
        load_guide_index(_write(tmp_path, content))


def test_index_that_is_not_utf8_raises(tmp_path: Path) -> None:
    (tmp_path / "index.yaml").write_bytes(b"locales:\n  - \xff\xfe\ntopics: []\n")
    with pytest.raises(GuideIndexError, match="Guide index unreadable"):
        load_guide_index(tmp_path)


def test_missing_index_raises(tmp_path: Path) -> None:
    with pytest.raises(GuideIndexError, match="Guide index missing"):
        load_guide_index(tmp_path)
