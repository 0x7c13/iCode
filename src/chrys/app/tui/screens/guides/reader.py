# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""User-guide markdown reading — H1 extraction, file loading, display names.

Pure logic with no Textual dependency.  Document content is always read from
the filesystem at the moment it is needed; a missing file yields ``None`` so
the dialog can show its localized "no content" state.
"""

from __future__ import annotations

import posixpath
from pathlib import Path, PurePosixPath

from chrys.app.tui.screens.guides.index import GuideIndex, GuideTopic, iter_leaf_topics


def extract_h1(markdown: str | None) -> str | None:
    """Return the first top-level ATX heading (``# ...``) text, if any.

    ``None`` input (a missing file) yields ``None`` — the caller falls back
    through its display-name chain instead of crashing.
    """
    if markdown is None:
        return None
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def read_topic_markdown(docs_root: Path, index: GuideIndex, topic: GuideTopic, locale: str) -> str | None:
    """Read the markdown for *topic* in *locale*, or ``None`` when unavailable.

    Paths are resolved against the locale subdirectory and confined to it:
    a ``path`` escaping the docs root (e.g. ``../``) is refused.  A file that
    cannot be read or is not UTF-8 counts as unavailable too.
    """
    if topic.path is None:
        return None
    if locale not in index.locales:
        return None
    try:
        docs_root = Path(docs_root).resolve()
        locale_root = (docs_root / locale).resolve()
        candidate = (locale_root / topic.path).resolve()
        if (
            not locale_root.is_relative_to(docs_root)
            or not candidate.is_relative_to(locale_root)
            or not candidate.is_file()
        ):
            return None
        return candidate.read_text(encoding="utf-8")
    except OSError, UnicodeDecodeError:
        return None


def resolve_guide_link(
    index: GuideIndex,
    source_topic: GuideTopic,
    href: str,
) -> tuple[GuideTopic | None, str]:
    """Resolve a markdown relative link from *source_topic* to a guide leaf.

    Returns ``(topic, anchor)`` where *topic* is the leaf whose ``path`` the
    link resolves to — or ``None`` for page-internal ``#anchor`` links,
    external URLs, and paths outside the topic tree — and *anchor* is the
    ``#fragment`` suffix (or ``""``).

    Resolution is purely lexical over ``PurePosixPath`` (never
    ``Path.resolve()``, which would resolve against the process cwd) and
    relative to the source topic's directory.  Topic paths are
    language-independent, so the link resolves identically in every document
    language; the caller loads the target with the current language.
    """
    if not href or source_topic.path is None:
        return None, ""
    stripped = href.strip()
    if stripped.startswith("#") or "://" in stripped:
        return None, ""
    path_part, _separator, fragment = stripped.partition("#")
    if not path_part:
        return None, ""
    source_dir = PurePosixPath(source_topic.path).parent
    # normpath folds "." / ".." lexically; an escaped prefix (e.g. ``../x``
    # from the docs root) or an absolute href can never equal a topic path.
    normalized = posixpath.normpath(str(source_dir / PurePosixPath(path_part)))
    for _leaf_id, topic in iter_leaf_topics(index.topics):
        if topic.path == normalized:
            return topic, fragment
    return None, ""


def _labels_label(topic: GuideTopic, *, locale: str, default_locale: str) -> str | None:
    """Return the per-locale display-name override, or ``None``.

    Resolution: exact *locale* first, then the *default_locale* (the docs
    root's primary language).  Unknown locales are skipped, so pre-declaring
    a label for a future language never breaks existing ones.
    """
    if not topic.labels:
        return None
    if locale in topic.labels:
        return topic.labels[locale]
    if default_locale in topic.labels:
        return topic.labels[default_locale]
    return None


def topic_display_name(topic: GuideTopic, *, locale: str, default_locale: str, current_h1: str | None) -> str:
    """Resolve a leaf topic's display name via the documented fallback chain.

    Order: per-locale ``labels`` → current-language H1 → filename stem →
    topic id.  The current-language H1 is the only H1 consulted: a topic
    without content in the active language degrades to its file name rather
    than borrowing another language's title.
    """
    labeled = _labels_label(topic, locale=locale, default_locale=default_locale)
    if labeled:
        return labeled
    if current_h1:
        return current_h1
    if topic.path:
        return Path(topic.path).stem
    if topic.id:
        return topic.id
    return ""


def branch_display_name(topic: GuideTopic, *, locale: str, default_locale: str) -> str:
    """Resolve a branch node's display name.

    Branches have no per-language file, so there is no H1 fallback.  Order:
    per-locale ``labels`` → the common directory segment shared by the
    branch's descendant files (e.g. ``daily-use``) → topic id.
    """
    labeled = _labels_label(topic, locale=locale, default_locale=default_locale)
    if labeled:
        return labeled
    directory = _branch_directory_name(topic)
    if directory:
        return directory
    if topic.id:
        return topic.id
    return ""


def _branch_directory_name(topic: GuideTopic) -> str | None:
    """Return the last shared directory segment of the branch's leaf files.

    For leaves under ``guides/configuration/*`` this is ``configuration``;
    for ``guides/daily-use/*`` it is ``daily-use``.  ``None`` when the
    branch has no leaf files or the files share no directory.
    """
    paths = [Path(leaf.path) for _leaf_id, leaf in iter_leaf_topics((topic,)) if leaf.path]
    if not paths:
        return None
    common = list(paths[0].parts[:-1])
    for path in paths[1:]:
        common = _common_prefix(common, list(path.parts[:-1]))
        if not common:
            return None
    return common[-1] if common else None


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    prefix: list[str] = []
    for left, right in zip(a, b, strict=False):
        if left != right:
            break
        prefix.append(left)
    return prefix
