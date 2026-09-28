# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the patch that clears the caches a removed node could stay in."""

from __future__ import annotations

import gc
import inspect
import logging
import weakref
from typing import TYPE_CHECKING

import pytest
from textual._node_list import NodeList
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from chrys.foundation.patches import textual_removed_node_caches
from chrys.foundation.patches.textual_removed_node_caches import (
    _RUNTIME_PATCH_TEXTUAL_VERSION,
    apply_runtime_patch,
    removal_clears_node_caches,
)
from tests.support.pilot_barrier import screen_is_settled
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from textual.pilot import Pilot


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        with Vertical(id="parent"):
            yield Static("kept", id="kept")
            with Vertical(id="branch"):
                yield Static("leaf", id="leaf")


async def _removed_leaf(app: _Host, pilot: Pilot[None], *, holder: str) -> weakref.ref[Static]:
    """Remove ``#leaf`` after *holder* cached it, and return a weak handle on it once settled."""
    await wait_for(lambda: screen_is_settled(app, app.screen), pilot=pilot, description="settled layout")
    branch = app.query_one("#branch", Vertical)
    if holder == "query":
        # The parent's lookup runs before the removal, as a card looks up the prompt it removes.
        leaf = app.query_one("#parent", Vertical).query_one("#leaf", Static)
    else:
        # Only layout caches hold the leaf: its parent's arrangement and displayed children.
        leaf = next(child for child in branch.children if isinstance(child, Static))
        assert branch._arrangement_cache, "the branch has arranged the leaf"
    leaf_ref = weakref.ref(leaf)
    await leaf.remove()
    del leaf
    await wait_for(lambda: screen_is_settled(app, app.screen), pilot=pilot, description="settled after removal")
    assert not branch.children
    gc.collect()
    return leaf_ref


@pytest.mark.parametrize("holder", ["query", "layout"])
async def test_upstream_keeps_a_removed_node_in_its_caches(monkeypatch: pytest.MonkeyPatch, holder: str) -> None:
    """Pins that the patch is still needed; drop the patch once a Textual upgrade makes this fail."""
    apply_runtime_patch()
    monkeypatch.setattr(NodeList, "_remove", inspect.unwrap(NodeList._remove))
    app = _Host()
    async with app.run_test() as pilot:
        leaf_ref = await _removed_leaf(app, pilot, holder=holder)

        assert leaf_ref() is not None


@pytest.mark.parametrize("holder", ["query", "layout"])
async def test_a_removed_node_is_freed_although_a_cache_held_it(holder: str) -> None:
    apply_runtime_patch()
    app = _Host()
    async with app.run_test() as pilot:
        leaf_ref = await _removed_leaf(app, pilot, holder=holder)

        assert leaf_ref() is None


async def test_a_removal_that_finds_no_node_keeps_the_caches() -> None:
    apply_runtime_patch()
    app = _Host()
    async with app.run_test() as pilot:
        await wait_for(lambda: screen_is_settled(app, app.screen), pilot=pilot, description="settled layout")
        parent = app.query_one("#parent", Vertical)
        kept = parent.query_one("#kept", Static)
        cached = list(parent._query_one_cache.keys())
        assert cached

        parent._nodes._remove(Static("stranger"))
        assert list(parent._query_one_cache.keys()) == cached

        await kept.remove()
        assert not parent._query_one_cache


def test_runtime_patch_is_idempotent() -> None:
    apply_runtime_patch()
    patched = NodeList._remove

    apply_runtime_patch()

    assert NodeList._remove is patched
    assert inspect.unwrap(NodeList._remove) is not patched
    assert removal_clears_node_caches() is True


def test_an_unpatched_removal_reports_it(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_runtime_patch()
    monkeypatch.setattr(NodeList, "_remove", inspect.unwrap(NodeList._remove))

    assert removal_clears_node_caches() is False


def test_runtime_patch_version_tracks_the_pinned_textual() -> None:
    import textual

    assert textual.__version__ == _RUNTIME_PATCH_TEXTUAL_VERSION


def test_another_textual_release_runs_unpatched(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The patch reads private Textual attributes, so another release keeps stock removal (and GC freeze detaching)."""
    import textual

    monkeypatch.setattr(NodeList, "_remove", inspect.unwrap(NodeList._remove))
    monkeypatch.setattr(textual, "__version__", "9.0.0")

    with caplog.at_level(logging.WARNING, logger=textual_removed_node_caches.__name__):
        apply_runtime_patch()

    assert NodeList._remove is inspect.unwrap(NodeList._remove)
    assert removal_clears_node_caches() is False
    assert "loaded Textual is not the pinned 8.2.7" in caplog.text
