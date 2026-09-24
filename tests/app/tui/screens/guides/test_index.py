# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the user-guide docs index (``docs/index.yaml``) parsing."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from chrys.app.tui.screens.guides import index as index_module
from chrys.app.tui.screens.guides.index import (
    BUNDLED_DOCS_DIRNAME,
    GuideIndexError,
    default_topic_id,
    iter_leaf_topics,
    language_cycle,
    load_guide_index,
    resolve_docs_root,
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


_REPO_ROOT = Path(__file__).resolve().parents[5]


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


def test_resolve_docs_root_prefers_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = _write(tmp_path / "docs", _VALID_INDEX)
    monkeypatch.setenv("CHRYS_DOCS_ROOT", str(docs))

    assert resolve_docs_root() == docs


def test_resolve_docs_root_reports_an_override_without_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong override shows the guide's missing-docs state, not another copy."""
    monkeypatch.setenv("CHRYS_DOCS_ROOT", str(tmp_path / "does-not-exist"))

    assert resolve_docs_root() is None


def test_resolve_docs_root_finds_repository_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHRYS_DOCS_ROOT", raising=False)

    assert resolve_docs_root() == _REPO_ROOT / "docs"


def _installed_module(site_packages: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the resolver at an ``index.py`` inside a fake installed package."""
    module_dir = site_packages / "chrys" / "app" / "tui" / "screens" / "guides"
    module_dir.mkdir(parents=True)
    monkeypatch.setattr(index_module, "__file__", str(module_dir / "index.py"))
    monkeypatch.delenv("CHRYS_DOCS_ROOT", raising=False)
    return module_dir


def test_resolve_docs_root_prefers_the_copy_bundled_in_the_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_dir = _installed_module(tmp_path / "site-packages", monkeypatch)
    # A checkout-shaped docs tree around the install loses to the bundled copy.
    _write(tmp_path / "docs", _VALID_INDEX)
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    assert resolve_docs_root() == tmp_path / "docs"

    bundled = _write(module_dir / BUNDLED_DOCS_DIRNAME, _VALID_INDEX)

    assert resolve_docs_root() == bundled


def test_resolve_docs_root_ignores_docs_outside_a_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _installed_module(tmp_path / "site-packages", monkeypatch)
    _write(tmp_path / "docs", _VALID_INDEX)

    assert resolve_docs_root() is None


def test_wheel_bundles_docs_where_the_resolver_looks() -> None:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    module_dir = Path(index_module.__file__).resolve().parent.relative_to(_REPO_ROOT / "src")

    assert force_include["docs"] == (module_dir / BUNDLED_DOCS_DIRNAME).as_posix()
