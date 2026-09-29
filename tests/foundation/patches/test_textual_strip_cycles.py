# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the patch that stops a Textual strip from caching itself."""

from __future__ import annotations

import gc
import weakref
from typing import TYPE_CHECKING

import pytest
from rich.segment import Segment
from textual import strip as strip_module
from textual.strip import Strip

from chrys.foundation.patches import textual_strip_cycles
from chrys.foundation.patches.patcher import FilePatch
from chrys.foundation.patches.staged_members import install_patched_members

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class _WeakStrip(Strip):
    __slots__ = ("__weakref__",)


@pytest.fixture
def no_cyclic_collection() -> Iterator[None]:
    """Only reference counting frees objects, so a cycle keeps its strip alive."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@pytest.fixture
def restore_strip_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Strip, "crop_extend", Strip.crop_extend)
    monkeypatch.setattr(Strip, "divide", Strip.divide)


def _painted_at_own_width(strip: Strip) -> None:
    strip.crop_extend(0, strip.cell_length, None)
    strip.divide([strip.cell_length])


def _outlives_its_last_reference(paint: Callable[[Strip], None]) -> bool:
    strip = _WeakStrip([Segment("line")], 4)
    ref = weakref.ref(strip)
    paint(strip)
    del strip
    return ref() is not None


@pytest.mark.usefixtures("no_cyclic_collection")
def test_a_strip_painted_at_its_own_width_is_freed_by_its_last_reference() -> None:
    textual_strip_cycles.apply_runtime_patch()

    assert not _outlives_its_last_reference(_painted_at_own_width)


def test_other_results_are_still_cached() -> None:
    textual_strip_cycles.apply_runtime_patch()
    strip = Strip([Segment("line")], 4)

    assert strip.crop_extend(1, 6, None) is strip.crop_extend(1, 6, None)
    assert strip.divide([2, 4]) is strip.divide([2, 4])
    assert strip.crop_extend(0, 4, None) is strip
    assert strip.divide([4]) == [strip]


@pytest.mark.usefixtures("no_cyclic_collection", "restore_strip_methods")
def test_upstream_strips_cache_themselves() -> None:
    """Pins that the patch is still needed; drop the patch once a Textual upgrade makes this fail."""
    upstream = [
        FilePatch(
            package="textual",
            module_file="strip.py",
            old_fragment=patch.new_fragment,
            new_fragment=patch.old_fragment,
            description=f"Restore upstream: {patch.description}",
        )
        for patch in textual_strip_cycles._PATCHES
    ]
    assert install_patched_members(
        strip_module,
        upstream,
        {"Strip": ["crop_extend", "divide"]},
        marker="_upstream",
        label="test",
    )

    assert _outlives_its_last_reference(lambda strip: strip.crop_extend(0, strip.cell_length, None))
    assert _outlives_its_last_reference(lambda strip: strip.divide([strip.cell_length]))
