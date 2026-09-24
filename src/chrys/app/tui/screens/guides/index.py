# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""User-guide docs index — YAML topic parsing and docs-root resolution.

Pure logic with no Textual dependency: models the ``docs/index.yaml`` topic
tree and locale ordering, and locates the docs root at runtime. The dialog
content is read from the files referenced here; nothing is hardcoded in code.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

INDEX_FILENAME = "index.yaml"
BUNDLED_DOCS_DIRNAME = "_docs"
"""Where built wheels carry ``docs/``, next to this module (``pyproject.toml`` force-include)."""
_CHECKOUT_DEPTH = 6
"""``<checkout>/src/chrys/app/tui/screens/guides/index.py`` sits this many directories below the checkout."""


class GuideIndexError(Exception):
    """Raised when the guide index mapping table is missing or malformed."""


@dataclass(frozen=True, slots=True)
class GuideTopic:
    """One node of the guide topic tree (branch or leaf)."""

    id: str | None
    """Stable identifier, referenced by the ``default`` field."""
    path: str | None
    """Leaf-only: markdown path relative to the locale docs subdirectory."""
    labels: Mapping[str, str] | None
    """Per-locale display-name overrides, keyed by entries in ``locales``."""
    children: tuple[GuideTopic, ...]
    """Branch-only: nested topics."""

    @property
    def is_leaf(self) -> bool:
        return self.path is not None

    @property
    def is_branch(self) -> bool:
        return bool(self.children)


@dataclass(frozen=True, slots=True)
class GuideIndex:
    """Parsed ``docs/index.yaml`` topic index."""

    locales: tuple[str, ...]
    """Ordered locales, also used as docs subdirectory names."""
    default_topic_id: str | None
    topics: tuple[GuideTopic, ...]


def resolve_docs_root() -> Path | None:
    """Locate the docs root, or ``None`` when unavailable.

    Resolution order:
    1. ``CHRYS_DOCS_ROOT``, for previewing edited docs.  A value without an
       ``index.yaml`` reports the docs as missing instead of quietly showing
       another copy;
    2. the copy bundled into built wheels next to this module;
    3. the source checkout's ``docs/`` (an editable install runs from ``src/``).
    """
    override = os.environ.get("CHRYS_DOCS_ROOT")
    if override:
        candidate = Path(override).expanduser()
        return candidate if (candidate / INDEX_FILENAME).is_file() else None
    here = Path(__file__).resolve()
    bundled = here.parent / BUNDLED_DOCS_DIRNAME
    if (bundled / INDEX_FILENAME).is_file():
        return bundled
    checkout = here.parents[_CHECKOUT_DEPTH]
    if (checkout / "pyproject.toml").is_file() and (checkout / "docs" / INDEX_FILENAME).is_file():
        return checkout / "docs"
    return None


def load_guide_index(docs_root: Path) -> GuideIndex:
    """Parse and validate ``docs_root/index.yaml`` into a :class:`GuideIndex`.

    Raises :class:`GuideIndexError` with a stable English diagnostic for
    missing files or schema violations; the caller renders a localized error
    state instead of crashing the dialog.
    """
    index_path = Path(docs_root) / INDEX_FILENAME
    if not index_path.is_file():
        raise GuideIndexError(f"Guide index missing: {index_path}")
    try:
        raw = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise GuideIndexError(f"Guide index unreadable: {index_path}") from error
    if not isinstance(raw, dict):
        raise GuideIndexError(f"Guide index must be a mapping: {index_path}")

    locales = _parse_locales(raw.get("locales"), index_path)
    default_topic_id = _parse_default(raw.get("default"), index_path)
    topics = _parse_topics(raw.get("topics"), index_path)
    return GuideIndex(locales=locales, default_topic_id=default_topic_id, topics=topics)


def language_cycle(index: GuideIndex) -> tuple[str, ...]:
    """Return the ordered locales the space key cycles through."""
    return index.locales


def default_topic_id(index: GuideIndex) -> str | None:
    """Resolve the default leaf: the ``default`` field, else the first leaf."""
    if index.default_topic_id is not None and _find_leaf(index.topics, index.default_topic_id) is not None:
        return index.default_topic_id
    first = next(iter_leaf_topics(index.topics), None)
    return first[0] if first is not None else None


def iter_leaf_topics(topics: tuple[GuideTopic, ...]) -> Iterator[tuple[str | None, GuideTopic]]:
    """Yield ``(id, topic)`` for every leaf in document order.

    Leaves without an ``id`` yield ``(None, topic)``; callers that only need
    the topic object may ignore the first element.
    """
    for topic in topics:
        if topic.is_leaf:
            yield topic.id, topic
        else:
            yield from iter_leaf_topics(topic.children)


def _find_leaf(topics: tuple[GuideTopic, ...], topic_id: str) -> GuideTopic | None:
    for leaf_id, topic in iter_leaf_topics(topics):
        if leaf_id == topic_id:
            return topic
    return None


def _parse_locales(raw: object, index_path: Path) -> tuple[str, ...]:
    if raw is None:
        raise GuideIndexError(f"Guide index missing 'locales': {index_path}")
    if not isinstance(raw, list):
        raise GuideIndexError(f"Guide index 'locales' must be a list: {index_path}")
    locales: list[str] = []
    for locale in raw:
        if not isinstance(locale, str) or not locale.strip():
            raise GuideIndexError(f"Guide index 'locales' has an invalid locale: {index_path}")
        normalized = locale.strip()
        if normalized in locales:
            raise GuideIndexError(f"Guide index 'locales' has duplicate locale {normalized!r}: {index_path}")
        locales.append(normalized)
    if not locales:
        raise GuideIndexError(f"Guide index 'locales' must list at least one locale: {index_path}")
    return tuple(locales)


def _parse_default(raw: object, index_path: Path) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise GuideIndexError(f"Guide index 'default' must be a non-empty string: {index_path}")
    return raw.strip()


def _parse_topics(raw: object, index_path: Path) -> tuple[GuideTopic, ...]:
    if raw is None:
        raise GuideIndexError(f"Guide index missing 'topics': {index_path}")
    if not isinstance(raw, list):
        raise GuideIndexError(f"Guide index 'topics' must be a list: {index_path}")
    return tuple(_parse_topic(item, index_path, depth=0) for item in raw)


def _parse_topic(raw: object, index_path: Path, *, depth: int) -> GuideTopic:
    if depth > 8:
        raise GuideIndexError(f"Guide index topic nesting too deep: {index_path}")
    if not isinstance(raw, dict):
        raise GuideIndexError(f"Guide index topic must be a mapping: {index_path}")
    topic_id = raw.get("id")
    if topic_id is not None and (not isinstance(topic_id, str) or not topic_id.strip()):
        raise GuideIndexError(f"Guide index topic 'id' must be a non-empty string: {index_path}")
    path = raw.get("path")
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise GuideIndexError(f"Guide index topic 'path' must be a non-empty string: {index_path}")
    labels_raw = raw.get("labels")
    if labels_raw is not None and not isinstance(labels_raw, dict):
        raise GuideIndexError(f"Guide index topic 'labels' must be a mapping: {index_path}")
    labels: Mapping[str, str] | None = None
    if labels_raw:
        for locale, label in labels_raw.items():
            if not isinstance(locale, str) or not locale.strip():
                raise GuideIndexError(f"Guide index topic 'labels' has an invalid locale key: {index_path}")
            if not isinstance(label, str) or not label.strip():
                raise GuideIndexError(f"Guide index topic 'labels' has an empty label for {locale!r}: {index_path}")
        labels = {locale.strip(): label.strip() for locale, label in labels_raw.items()}
    children_raw = raw.get("children")
    if children_raw is not None and not isinstance(children_raw, list):
        raise GuideIndexError(f"Guide index topic 'children' must be a list: {index_path}")
    has_path = path is not None
    # 'children' declared (even as an empty list) alongside 'path' is a
    # contradiction — refuse presence of both keys, not just non-empty lists.
    if has_path and children_raw is not None:
        raise GuideIndexError(f"Guide index topic has both 'path' and 'children': {index_path}")
    has_children = children_raw is not None and len(children_raw) > 0
    if not has_path and not has_children:
        raise GuideIndexError(f"Guide index topic needs 'path' (leaf) or 'children' (branch): {index_path}")
    children = tuple(_parse_topic(child, index_path, depth=depth + 1) for child in children_raw) if has_children else ()
    return GuideTopic(
        id=topic_id.strip() if isinstance(topic_id, str) else None,
        path=path.strip() if isinstance(path, str) else None,
        labels=labels,
        children=children,
    )
