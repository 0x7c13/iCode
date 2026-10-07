# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: Textual's LRU cache keeps its entries in an ``OrderedDict``.

Problem
-------
``textual.cache.LRUCache`` links its entries into a circular list of four-item Python lists
(``[prev, next, key, value]``), so every cache holding an entry is a reference cycle. Every
widget carries one (``_box_model_cache``) and every DOM node another (``_query_one_cache``), and
Chrys's markdown, diff and terminal views keep their rendered lines in them. An entry is only
freed by a cyclic collection, every entry adds a tracked container the collector traverses, and
under ``gc.freeze()`` an entry cleared from a frozen cache stays in the permanent generation
until the next unfreeze.

Solution
--------
Keep the entries in an ``OrderedDict`` in order of use: eviction pops its first entry and a hit
moves the key to its end. The cache holds no cycle, so an evicted, discarded or cleared entry is
freed by its last reference. Lookups test membership before reading, because a miss is common
and raising ``KeyError`` for it costs twice a hit. ``keys()`` returns a snapshot: a hit reorders
the cache, and ``Log`` reads entries while it iterates the keys. As upstream's, ``set`` leaves
an existing key's value as it is.

One difference is deliberate. Upstream's cache, once it has evicted, evicts on every later insert
(its ``_full`` flag never resets), so ``grow()`` and a larger ``maxsize`` stop taking effect, and
a zero ``maxsize`` fails its second insert with ``KeyError``. This cache evicts only past
``maxsize``: a capacity change applies to the next insert, and a zero-capacity cache stays empty.

The runtime install must precede the first cache, as bootstrap does: a cache built by the
upstream ``__init__`` keeps linked entries that the new methods cannot read, and fails at its
first lookup.
"""

from __future__ import annotations

import logging

from chrys.foundation.patches.patcher import FilePatch, register
from chrys.foundation.patches.staged_members import StagedSourceDriftError, members_installed, stage_members

_RUNTIME_PATCH_MARKER = "_chrys_lru_acyclic"
_RUNTIME_PATCH_TEXTUAL_VERSION = "8.2.7"
_RUNTIME_MEMBERS = {"LRUCache": ["__init__", "clear", "keys", "set", "__setitem__", "get", "__getitem__", "discard"]}
logger = logging.getLogger(__name__)

_IMPORT_OLD = """\
from typing import TYPE_CHECKING, Dict, Generic, KeysView, TypeVar, overload
"""

_IMPORT_NEW = """\
from collections import OrderedDict
from typing import TYPE_CHECKING, Generic, KeysView, TypeVar, overload
"""

_INIT_OLD = """\
        self._cache: Dict[CacheKey, list[object]] = {}
        self._full = False
        self._head: list[object] = []
"""

_INIT_NEW = """\
        self._cache: OrderedDict[CacheKey, CacheValue] = OrderedDict()
"""

_CLEAR_OLD = """\
        self._cache.clear()
        self._full = False
        self._head = []

    def keys(self) -> KeysView[CacheKey]:
        \"\"\"Get cache keys.\"\"\"
        # Mostly for tests
        return self._cache.keys()
"""

_CLEAR_NEW = """\
        self._cache.clear()

    def keys(self) -> KeysView[CacheKey]:
        \"\"\"Get cache keys.\"\"\"
        # A snapshot: reading an entry reorders the cache, which a live view would not survive.
        return dict.fromkeys(self._cache).keys()
"""

_SET_OLD = """\
        if self._cache.get(key) is None:
            head = self._head
            if not head:
                # First link references itself
                self._head[:] = [head, head, key, value]
            else:
                # Add a new root to the beginning
                self._head = [head[0], head, key, value]
                # Updated references on previous root
                head[0][1] = self._head  # type: ignore[index]
                head[0] = self._head
            self._cache[key] = self._head

            if self._full or len(self._cache) > self._maxsize:
                # Cache is full, we need to evict the oldest one
                self._full = True
                head = self._head
                last = head[0]
                last[0][1] = head  # type: ignore[index]
                head[0] = last[0]  # type: ignore[index]
                del self._cache[last[2]]  # type: ignore[index]
"""

_SET_NEW = """\
        cache = self._cache
        if key not in cache:
            cache[key] = value
            if len(cache) > self._maxsize:
                # Cache is full, we need to evict the oldest one
                cache.popitem(last=False)
"""

_GET_OLD = """\

        if (link := self._cache.get(key)) is None:
            self.misses += 1
            return default
        if link is not self._head:
            # Remove link from list
            link[0][1] = link[1]  # type: ignore[index]
            link[1][0] = link[0]  # type: ignore[index]
            head = self._head
            # Move link to head of list
            link[0] = head[0]
            link[1] = head
            self._head = head[0][1] = head[0] = link  # type: ignore[index]
        self.hits += 1
        return link[3]  # type: ignore[return-value]

    def __getitem__(self, key: CacheKey) -> CacheValue:
        link = self._cache.get(key)
        if (link := self._cache.get(key)) is None:
            self.misses += 1
            raise KeyError(key)
        if link is not self._head:
            link[0][1] = link[1]  # type: ignore[index]
            link[1][0] = link[0]  # type: ignore[index]
            head = self._head
            link[0] = head[0]
            link[1] = head
            self._head = head[0][1] = head[0] = link  # type: ignore[index]
        self.hits += 1
        return link[3]  # type: ignore[return-value]
"""

_GET_NEW = """\
        cache = self._cache
        if key in cache:
            cache.move_to_end(key)
            self.hits += 1
            return cache[key]
        self.misses += 1
        return default

    def __getitem__(self, key: CacheKey) -> CacheValue:
        cache = self._cache
        if key in cache:
            cache.move_to_end(key)
            self.hits += 1
            return cache[key]
        self.misses += 1
        raise KeyError(key)
"""

_DISCARD_OLD = """\
        if key not in self._cache:
            return
        link = self._cache[key]

        # Remove link from list
        link[0][1] = link[1]  # type: ignore[index]
        link[1][0] = link[0]  # type: ignore[index]
        # Remove link from cache

        if self._head[2] == key:
            self._head = self._head[1]  # type: ignore[assignment]
            if self._head[2] == key:  # type: ignore[index]
                self._head = []

        del self._cache[key]
        self._full = False
"""

_DISCARD_NEW = """\
        self._cache.pop(key, None)
"""


def _patch(old: str, new: str, description: str) -> FilePatch:
    return FilePatch(
        package="textual",
        module_file="cache.py",
        old_fragment=old,
        new_fragment=new,
        description=description,
    )


_PATCHES = (
    _patch(_IMPORT_OLD, _IMPORT_NEW, "textual.cache imports OrderedDict"),
    _patch(_INIT_OLD, _INIT_NEW, "LRUCache.__init__ builds an OrderedDict"),
    _patch(_CLEAR_OLD, _CLEAR_NEW, "LRUCache.clear drops no links; keys() is a snapshot"),
    _patch(_SET_OLD, _SET_NEW, "LRUCache.set evicts the OrderedDict's oldest entry"),
    _patch(_GET_OLD, _GET_NEW, "LRUCache.get and __getitem__ move a hit to the end"),
    _patch(_DISCARD_OLD, _DISCARD_NEW, "LRUCache.discard pops the entry"),
)

register(*_PATCHES)


def apply_runtime_patch() -> None:
    """Install the ``OrderedDict`` LRU methods in the current process, before any cache exists."""
    try:
        from collections import OrderedDict

        import textual
        import textual.cache as cache_mod
    except ImportError:
        return
    if textual.__version__ != _RUNTIME_PATCH_TEXTUAL_VERSION:
        logger.warning(
            "Skipping Textual LRU runtime patch: loaded Textual is not the pinned %s that the patch targets.",
            _RUNTIME_PATCH_TEXTUAL_VERSION,
        )
        return
    if members_installed(cache_mod, _RUNTIME_MEMBERS, _RUNTIME_PATCH_MARKER):
        return
    try:
        staged = stage_members(cache_mod, _PATCHES, _RUNTIME_MEMBERS, label="LRU")
    except StagedSourceDriftError as exc:
        logger.warning("Skipping Textual LRU runtime patch: %s", exc)
        return
    vars(cache_mod).setdefault("OrderedDict", OrderedDict)
    staged.install(_RUNTIME_PATCH_MARKER)
