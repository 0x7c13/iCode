# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Permanent-generation residue probes for transcript render LRUs."""

from __future__ import annotations

import gc
import weakref

import pytest
from rich.segment import Segment
from textual.app import App, ComposeResult
from textual.events import Resize
from textual.geometry import Offset, Region, Size
from textual.selection import Selection
from textual.strip import Strip

from chrys.app.tui.support.gc_freeze import DetachedLruCache
from chrys.app.tui.widgets.diff_view import DiffView
from chrys.app.tui.widgets.diff_view.code import CodeColumn
from chrys.app.tui.widgets.diff_view.gutter import GutterColumn
from chrys.app.tui.widgets.diff_view.unified import UnifiedDiffLines
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown


class _Marker:
    pass


def _tracked_strip() -> tuple[Strip, weakref.ReferenceType[_Marker]]:
    marker = _Marker()
    marker_ref = weakref.ref(marker)
    return Strip([Segment("x", control=marker)], 1), marker_ref  # type: ignore[arg-type]


_MARKDOWN = """
# Cache probe

| Name | Value |
| --- | --- |
| alpha | a value that wraps at narrower widths |
| beta | another value |

```python
for index in range(8):
    print(index)
```
"""


class _MarkdownCacheApp(App):
    def compose(self) -> ComposeResult:
        yield VirtualizedMarkdown(_MARKDOWN)


def _populate_markdown_caches(markdown: VirtualizedMarkdown) -> None:
    markdown._frame_visible_height = max(1, markdown._total_lines)
    for line in range(markdown._total_lines):
        markdown.render_line(line)

    assert markdown._line_cache
    assert markdown._table_strips
    assert markdown._fence_line_strips
    assert markdown._fence_source_line_strips


def _track_markdown_caches(markdown: VirtualizedMarkdown) -> list[weakref.ReferenceType[_Marker]]:
    marker_refs: list[weakref.ReferenceType[_Marker]] = []
    strip, marker_ref = _tracked_strip()
    markdown._line_cache[(-1, -1)] = strip
    marker_refs.append(marker_ref)
    strip, marker_ref = _tracked_strip()
    markdown._table_strips[(-1, -1)] = strip
    marker_refs.append(marker_ref)
    strip, marker_ref = _tracked_strip()
    markdown._fence_line_strips[(-1, -1)] = strip
    marker_refs.append(marker_ref)
    strip, marker_ref = _tracked_strip()
    markdown._fence_source_line_strips[(-1, -1)] = [strip]
    marker_refs.append(marker_ref)
    del strip
    del marker_ref
    return marker_refs


@pytest.mark.asyncio
async def test_markdown_gc_participant_detaches_and_renews_render_lrus() -> None:
    app = _MarkdownCacheApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)
        _populate_markdown_caches(markdown)
        old_caches = (
            markdown._line_cache,
            markdown._table_strips,
            markdown._fence_line_strips,
            markdown._fence_source_line_strips,
        )
        capacities = tuple(cache.maxsize for cache in old_caches)

        markdown.prepare_for_gc_freeze()
        assert isinstance(markdown._line_cache, DetachedLruCache)
        assert isinstance(markdown._table_strips, DetachedLruCache)
        assert isinstance(markdown._fence_line_strips, DetachedLruCache)
        assert isinstance(markdown._fence_source_line_strips, DetachedLruCache)
        gc.collect()
        gc.freeze()
        try:
            markdown.after_gc_freeze()
            new_caches = (
                markdown._line_cache,
                markdown._table_strips,
                markdown._fence_line_strips,
                markdown._fence_source_line_strips,
            )
            assert all(new is not old for new, old in zip(new_caches, old_caches, strict=True))
            assert tuple(cache.maxsize for cache in new_caches) == capacities
            del new_caches

            marker_refs = _track_markdown_caches(markdown)
            markdown.prepare_for_gc_freeze()
            gc.collect()
            assert all(marker_ref() is None for marker_ref in marker_refs)
            markdown.after_gc_freeze()
        finally:
            gc.unfreeze()
            gc.collect()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidation", ["style", "selection-width"])
async def test_populated_markdown_lru_residue_waits_for_unfreeze(
    monkeypatch: pytest.MonkeyPatch,
    invalidation: str,
) -> None:
    app = _MarkdownCacheApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)
        _populate_markdown_caches(markdown)
        marker_refs = _track_markdown_caches(markdown)
        old_width = markdown._width_at_last_layout

        gc.collect()
        gc.freeze()
        try:
            if invalidation == "style":
                markdown.notify_style_update()
            else:
                selection = Selection(Offset(0, 0), Offset(8, 1))
                app.screen.selections = {markdown: selection}
                markdown.selection_updated(selection)
                markdown.render_lines(Region(0, 0, markdown.size.width, min(2, markdown._total_lines)))
                assert all(marker_ref() is not None for marker_ref in marker_refs)

                new_width = max(1, old_width - 3)
                monkeypatch.setattr(
                    type(markdown),
                    "scrollable_content_region",
                    property(lambda _self: Region(0, 0, new_width, 24)),
                )
                markdown.on_resize(Resize(Size(new_width, 24), Size(old_width, 24)))

            assert not markdown._line_cache
            assert not markdown._table_strips
            assert not markdown._fence_line_strips
            assert not markdown._fence_source_line_strips
            gc.collect()
            assert all(marker_ref() is not None for marker_ref in marker_refs)
        finally:
            app.screen.selections = {}
            gc.unfreeze()
            gc.collect()

        assert all(marker_ref() is None for marker_ref in marker_refs)


@pytest.mark.asyncio
async def test_populated_markdown_line_lru_eviction_releases_without_unfreeze() -> None:
    app = _MarkdownCacheApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)
        _populate_markdown_caches(markdown)
        cache = markdown._line_cache
        cache.clear()
        strip, marker_ref = _tracked_strip()
        tracked_key = (-1, -1)
        cache[tracked_key] = strip
        del strip

        gc.collect()
        gc.freeze()
        try:
            for index in range(cache.maxsize):
                cache[(index, 1)] = Strip.blank(1)
            assert tracked_key not in cache
            gc.collect()
            assert marker_ref() is None
        finally:
            gc.unfreeze()
            gc.collect()

        assert marker_ref() is None


class _DiffCacheApp(App):
    CSS = "DiffView { height: 10; }"

    def compose(self) -> ComposeResult:
        flat = DiffView("before.py", "after.py", "alpha\nbeta\n", "alpha\nBETA\ngamma\n", id="flat")
        flat.split = False
        flat.auto_height = True
        flat.show_scrollbars = False
        yield flat

        split = DiffView("before.py", "after.py", "alpha\nbeta\n", "alpha\nBETA\ngamma\n", id="split")
        split.split = True
        yield split


def _populate_diff_caches(app: _DiffCacheApp) -> tuple[UnifiedDiffLines, list[CodeColumn], list[GutterColumn]]:
    flat = app.query_one("#flat", DiffView).query_one(UnifiedDiffLines)
    flat.render_lines(Region(0, 0, max(1, flat.size.width), min(3, len(flat.rows))))
    code_columns = list(app.query_one("#split", DiffView).query(CodeColumn))
    gutter_columns = list(app.query_one("#split", DiffView).query(GutterColumn))
    for code_column in code_columns:
        code_column.render_lines(Region(0, 0, max(1, code_column.size.width), 2))
    for gutter_column in gutter_columns:
        gutter_column.render_lines(Region(0, 0, max(1, gutter_column.size.width), 2))

    assert flat._strip_cache
    assert code_columns and all(column._strip_cache for column in code_columns)
    assert gutter_columns and all(column._strip_cache for column in gutter_columns)
    return flat, code_columns, gutter_columns


def _track_diff_caches(
    flat: UnifiedDiffLines,
    code_columns: list[CodeColumn],
    gutter_columns: list[GutterColumn],
) -> list[weakref.ReferenceType[_Marker]]:
    marker_refs: list[weakref.ReferenceType[_Marker]] = []
    strip, marker_ref = _tracked_strip()
    flat._strip_cache[-1] = strip
    marker_refs.append(marker_ref)
    for index, code_column in enumerate(code_columns):
        strip, marker_ref = _tracked_strip()
        code_column._strip_cache[-index - 1] = strip
        marker_refs.append(marker_ref)
    for index, gutter_column in enumerate(gutter_columns):
        strip, marker_ref = _tracked_strip()
        gutter_column._strip_cache[-index - 1] = strip
        marker_refs.append(marker_ref)
    del strip
    del marker_ref
    return marker_refs


@pytest.mark.asyncio
async def test_diff_gc_participants_detach_and_renew_render_lrus() -> None:
    app = _DiffCacheApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        flat, code_columns, gutter_columns = _populate_diff_caches(app)
        surfaces = [flat, *code_columns, *gutter_columns]
        old_caches = [
            flat._strip_cache,
            *(column._strip_cache for column in code_columns),
            *(column._strip_cache for column in gutter_columns),
        ]
        capacities = [cache.maxsize for cache in old_caches]

        for surface in surfaces:
            surface.prepare_for_gc_freeze()
        assert isinstance(flat._strip_cache, DetachedLruCache)
        assert all(isinstance(column._strip_cache, DetachedLruCache) for column in code_columns)
        assert all(isinstance(column._strip_cache, DetachedLruCache) for column in gutter_columns)
        gc.collect()
        gc.freeze()
        try:
            for surface in surfaces:
                surface.after_gc_freeze()
            new_caches = [
                flat._strip_cache,
                *(column._strip_cache for column in code_columns),
                *(column._strip_cache for column in gutter_columns),
            ]
            assert all(new is not old for new, old in zip(new_caches, old_caches, strict=True))
            assert [cache.maxsize for cache in new_caches] == capacities
            del new_caches

            marker_refs = _track_diff_caches(flat, code_columns, gutter_columns)
            for surface in surfaces:
                surface.prepare_for_gc_freeze()
            gc.collect()
            assert all(marker_ref() is None for marker_ref in marker_refs)
            for surface in surfaces:
                surface.after_gc_freeze()
        finally:
            gc.unfreeze()
            gc.collect()


@pytest.mark.asyncio
async def test_populated_diff_lru_style_invalidation_waits_for_unfreeze() -> None:
    app = _DiffCacheApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        flat, code_columns, gutter_columns = _populate_diff_caches(app)
        marker_refs = _track_diff_caches(flat, code_columns, gutter_columns)

        gc.collect()
        gc.freeze()
        try:
            flat.notify_style_update()
            for code_column in code_columns:
                code_column.notify_style_update()
            for gutter_column in gutter_columns:
                gutter_column.notify_style_update()

            assert not flat._strip_cache
            assert all(not column._strip_cache for column in code_columns)
            assert all(not column._strip_cache for column in gutter_columns)
            gc.collect()
            assert all(marker_ref() is not None for marker_ref in marker_refs)
        finally:
            gc.unfreeze()
            gc.collect()

        assert all(marker_ref() is None for marker_ref in marker_refs)


@pytest.mark.asyncio
async def test_populated_diff_selection_and_width_residue_waits_for_unfreeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _DiffCacheApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        flat, code_columns, gutter_columns = _populate_diff_caches(app)
        marker_refs = _track_diff_caches(flat, code_columns, gutter_columns)
        selection = Selection(Offset(0, 0), Offset(8, 1))

        gc.collect()
        gc.freeze()
        try:
            app.screen.selections = {flat: selection, **dict.fromkeys(code_columns, selection)}
            flat.render_lines(Region(0, 0, max(1, flat.size.width), 2))
            for code_column in code_columns:
                code_column.render_lines(Region(0, 0, max(1, code_column.size.width), 2))
            assert all(marker_ref() is not None for marker_ref in marker_refs)

            flat.set_annotations(not flat.annotations)
            old_widths = {column: column._get_render_width() for column in code_columns}
            monkeypatch.setattr(CodeColumn, "_get_render_width", lambda self: max(1, old_widths[self] - 1))
            for code_column in code_columns:
                code_column.render_lines(Region(0, 0, max(1, code_column.size.width), 2))
            for gutter_column in gutter_columns:
                gutter_column.notify_style_update()

            assert not flat._strip_cache
            assert all(not column._strip_cache for column in code_columns)
            assert all(not column._strip_cache for column in gutter_columns)
            gc.collect()
            assert all(marker_ref() is not None for marker_ref in marker_refs)
        finally:
            app.screen.selections = {}
            gc.unfreeze()
            gc.collect()

        assert all(marker_ref() is None for marker_ref in marker_refs)


@pytest.mark.asyncio
async def test_populated_diff_strip_lru_eviction_release_without_unfreeze() -> None:
    app = _DiffCacheApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        flat, code_columns, gutter_columns = _populate_diff_caches(app)
        caches = [flat._strip_cache, code_columns[0]._strip_cache, gutter_columns[0]._strip_cache]
        refs: list[weakref.ReferenceType[_Marker]] = []
        for cache in caches:
            cache.clear()
            strip, ref = _tracked_strip()
            cache[-1] = strip
            refs.append(ref)
        del strip
        del ref

        gc.collect()
        gc.freeze()
        try:
            for cache in caches:
                for index in range(cache.maxsize):
                    cache[index] = Strip.blank(1)
                assert -1 not in cache
            gc.collect()
            assert all(ref() is None for ref in refs)
        finally:
            gc.unfreeze()
            gc.collect()

        assert all(ref() is None for ref in refs)
