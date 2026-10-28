# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the patch that makes a widget sized to its children measure them anew once one shows, hides or mounts."""

from __future__ import annotations

import inspect
import logging

import pytest
from textual._node_list import NodeList
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from chrys.foundation.patches import textual_box_model_children, textual_reflow_reuse
from chrys.foundation.patches.textual_box_model_children import _RUNTIME_PATCH_TEXTUAL_VERSION, apply_runtime_patch
from tests.support.pilot_barrier import screen_is_settled
from tests.support.waiting import wait_for


class _Host(App[None]):
    CSS = "#entry { height: auto; }"

    def compose(self) -> ComposeResult:
        with Vertical(id="entry"):
            yield Static("title")
            yield Static("one\ntwo\nthree", id="body")
        yield Static("below")


async def _entry_height_laid_out_before_the_child_idles(change: str) -> tuple[int, int]:
    """The entry's height before and after *change* to its children, in a layout that runs before their pumps idle,
    as one another widget asked for can."""
    app = _Host()
    async with app.run_test() as pilot:
        body = app.query_one("#body", Static)
        body.display = change != "shows"
        await wait_for(lambda: screen_is_settled(app, app.screen), pilot=pilot, description="settled layout")
        entry = app.query_one("#entry", Vertical)
        before = entry.region.height
        if change == "mounts":
            entry.mount(Static("four\nfive"))
        else:
            body.display = change == "shows"
        app.screen._refresh_layout()
        return before, entry.region.height


@pytest.mark.parametrize(("change", "heights"), [("hides", (4, 1)), ("shows", (1, 4)), ("mounts", (4, 6))])
async def test_an_entry_sized_to_its_children_is_measured_anew_once_one_shows_hides_or_mounts(
    change: str, heights: tuple[int, int]
) -> None:
    apply_runtime_patch()

    assert await _entry_height_laid_out_before_the_child_idles(change) == heights


@pytest.mark.parametrize(("change", "before"), [("hides", 4), ("shows", 1), ("mounts", 4)])
async def test_upstream_measures_an_entry_at_the_size_it_had(
    monkeypatch: pytest.MonkeyPatch, change: str, before: int
) -> None:
    """Pins that the patch is still needed; drop the patch once a Textual upgrade makes this fail."""
    apply_runtime_patch()
    monkeypatch.setattr(NodeList, "updated", inspect.unwrap(NodeList.updated))

    assert await _entry_height_laid_out_before_the_child_idles(change) == (before, before)


def test_a_tree_built_before_it_mounts_stores_no_layout_count() -> None:
    """A transcript precomposed detached was never measured: its widgets keep the count on the class."""
    apply_runtime_patch()
    entry, title = Vertical(), Static("title")

    entry._nodes._append(title)

    assert "_layout_updates" not in vars(entry)


def test_runtime_patch_wraps_the_reusing_reflows_child_list_update_once() -> None:
    apply_runtime_patch()
    patched = NodeList.updated

    apply_runtime_patch()
    textual_reflow_reuse.apply_runtime_patch()

    assert NodeList.updated is patched
    assert getattr(patched, textual_box_model_children._RUNTIME_PATCH_MARKER) is True
    assert vars(inspect.unwrap(patched)).get(textual_reflow_reuse._RUNTIME_PATCH_MARKER) is True


def test_runtime_patch_version_tracks_the_pinned_textual() -> None:
    import textual

    assert textual.__version__ == _RUNTIME_PATCH_TEXTUAL_VERSION


def test_another_textual_release_runs_unpatched(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The patch reads private Textual attributes, so another release keeps the upstream child-list update."""
    import textual

    upstream = inspect.unwrap(NodeList.updated)
    monkeypatch.setattr(NodeList, "updated", upstream)
    monkeypatch.setattr(textual, "__version__", "9.0.0")

    with caplog.at_level(logging.WARNING, logger=textual_box_model_children.__name__):
        apply_runtime_patch()

    assert NodeList.updated is upstream
    assert "loaded Textual is not the pinned 8.2.7" in caplog.text
