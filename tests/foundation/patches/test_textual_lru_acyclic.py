# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the patch that keeps Textual's LRU cache entries in an ``OrderedDict``."""

from __future__ import annotations

import gc
import weakref
from typing import TYPE_CHECKING, Any

import pytest
from textual import cache as cache_module
from textual.cache import LRUCache

from chrys.foundation.patches import textual_lru_acyclic
from chrys.foundation.patches.patcher import FilePatch
from chrys.foundation.patches.staged_members import members_installed, stage_patched_source

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class _Value:
    pass


@pytest.fixture
def no_cyclic_collection() -> Iterator[None]:
    """Only reference counting frees objects, so a cycle keeps its entries alive."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _drop_cache(cache: Any) -> None:
    del cache


_RELEASES: dict[str, Callable[[Any], None]] = {
    "evict": lambda cache: [cache.set(key, _Value()) for key in range(cache.maxsize)],
    "discard": lambda cache: cache.discard("tracked"),
    "clear": lambda cache: cache.clear(),
    "drop-cache": _drop_cache,
}


def _released_by_last_reference(cache_type: type[Any], release: Callable[[Any], None]) -> bool:
    cache = cache_type(4)
    value = _Value()
    ref = weakref.ref(value)
    cache.set("tracked", value)
    cache.set("other", _Value())
    assert cache.get("tracked") is value
    del value
    release(cache)
    del cache
    return ref() is None


def test_the_installed_textual_is_patched() -> None:
    assert members_installed(
        cache_module, textual_lru_acyclic._RUNTIME_MEMBERS, textual_lru_acyclic._RUNTIME_PATCH_MARKER
    )


@pytest.mark.usefixtures("no_cyclic_collection")
@pytest.mark.parametrize("release", list(_RELEASES))
def test_a_released_entry_is_freed_by_its_last_reference(release: str) -> None:
    assert _released_by_last_reference(LRUCache, _RELEASES[release])


def test_the_least_recently_used_entry_is_evicted() -> None:
    cache: LRUCache[str, int] = LRUCache(3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert cache.get("a") == 1
    assert cache["b"] == 2

    cache["d"] = 4

    assert list(cache.keys()) == ["a", "b", "d"]
    assert (cache.hits, cache.misses) == (2, 0)


def test_misses_count_and_keep_upstream_results() -> None:
    cache: LRUCache[str, int] = LRUCache(2)

    assert cache.get("absent") is None
    assert cache.get("absent", 7) == 7
    with pytest.raises(KeyError, match="absent"):
        cache["absent"]
    cache.discard("absent")

    assert (cache.hits, cache.misses) == (0, 3)
    assert not cache


def test_setting_an_existing_key_keeps_its_value_as_upstream_does() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    cache["a"] = 1
    cache["a"] = 2

    assert cache["a"] == 1
    assert len(cache) == 1


def test_entries_can_be_read_while_iterating_the_keys() -> None:
    """``Log`` rebuilds its line cache from ``keys()`` while reading every entry."""
    cache: LRUCache[int, int] = LRUCache(8)
    for key in range(4):
        cache[key] = key * 10
    keys = cache.keys()

    assert {key: cache[key] for key in keys if key > 0} == {1: 10, 2: 20, 3: 30}


def test_capacity_changes_apply_to_the_next_insert() -> None:
    cache: LRUCache[int, int] = LRUCache(1)
    cache[1] = 1
    cache.grow(2)
    cache[2] = 2
    assert list(cache.keys()) == [1, 2]

    cache.maxsize = 1
    cache[3] = 3

    assert list(cache.keys()) == [2, 3]


def test_growing_a_cache_that_has_evicted_raises_its_capacity() -> None:
    """Upstream's cache keeps evicting on every insert once it has evicted, whatever its size."""
    cache: LRUCache[int, int] = LRUCache(1)
    cache[1] = 1
    cache[2] = 2
    assert list(cache.keys()) == [2]

    cache.grow(3)
    cache[3] = 3
    cache[4] = 4

    assert list(cache.keys()) == [2, 3, 4]


def test_a_zero_capacity_cache_stays_empty() -> None:
    """Upstream's fails its second insert with ``KeyError``."""
    cache: LRUCache[str, int] = LRUCache(0)
    cache["a"] = 1
    cache["b"] = 2

    assert not cache
    assert cache.get("b") is None
    with pytest.raises(KeyError):
        cache["b"]


@pytest.mark.usefixtures("no_cyclic_collection")
def test_upstream_lru_entries_wait_for_a_cyclic_collection() -> None:
    """Pins that the patch is still needed; drop the patch once a Textual upgrade makes this fail."""
    upstream = [
        FilePatch(
            package="textual",
            module_file="cache.py",
            old_fragment=patch.new_fragment,
            new_fragment=patch.old_fragment,
            description=f"Restore upstream: {patch.description}",
        )
        for patch in textual_lru_acyclic._PATCHES
    ]
    namespace: dict[str, Any] = {"__name__": "chrys_upstream_textual_cache"}
    exec(compile(stage_patched_source(cache_module, upstream), cache_module.__file__ or "cache.py", "exec"), namespace)
    upstream_cache = namespace["LRUCache"]

    assert not _released_by_last_reference(upstream_cache, _RELEASES["drop-cache"])
