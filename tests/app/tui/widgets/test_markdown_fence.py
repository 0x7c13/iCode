# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the lazy fence-strip path in :class:`VirtualizedMarkdown`.

``_render_block_line`` used to call ``Content.render_strips`` once per visible
row of a fenced code block, discarding all but one strip and making the first
scroll through an N-line fence O(N^2). Later it rendered the whole fence when
the first row became visible, which moved the stall to one large scroll tick.
These tests pin the row-level cache behavior so neither regression returns.
"""

from __future__ import annotations

import pytest
from rich.style import Style
from textual.app import App, ComposeResult
from textual.content import Content, Span
from textual.events import Resize
from textual.geometry import Region, Size
from textual.strip import Strip

from chrys.app.tui.widgets.markdown import widget as widget_module
from chrys.app.tui.widgets.markdown.blocks import MarkdownBlock
from chrys.app.tui.widgets.markdown.parser import _FENCE_CLOSED_META, _create_markdown_parser
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown, _content_slice, _line_ranges


def _fence_markdown(n_lines: int) -> str:
    body = "\n".join(f"line_{i:03d} = {i} * 2" for i in range(n_lines))
    return f"# Heading\n\n```python\n{body}\n```\n"


def _offscreen_fence_markdown(n_lines: int) -> str:
    prelude = "\n\n".join(f"Prelude paragraph {i}." for i in range(20))
    return f"{prelude}\n\n{_fence_markdown(n_lines)}"


def _offscreen_table_markdown(n_rows: int) -> str:
    prelude = "\n\n".join(f"Prelude paragraph {i}." for i in range(20))
    rows = "\n".join(f"| row {i} | value {i} with wrapping text |" for i in range(n_rows))
    table = f"| Name | Value |\n| --- | --- |\n{rows}\n"
    return f"{prelude}\n\n{table}"


def _long_paragraph_markdown(n_words: int) -> str:
    return " ".join(f"word_{i:03d}" for i in range(n_words))


@pytest.mark.parametrize(
    ("source", "closed"),
    [
        ("```mermaid\nA --> B\n```", True),
        ("```mermaid\nA --> B", False),
        ("```mermaid\nA --> B\n````", True),
        ("~~~mermaid\nA --> B\n~~~", True),
        ("> ```mermaid\n> A --> B\n> ```", True),
        ("- item\n\n  ```mermaid\n  A --> B\n  ```", True),
    ],
)
def test_fence_tokens_record_whether_the_source_closed(source: str, closed: bool) -> None:
    token = next(token for token in _create_markdown_parser().parse(source) if token.type == "fence")

    assert token.meta[_FENCE_CLOSED_META] is closed


def test_content_slice_preserves_overlapping_unsorted_spans() -> None:
    """Fence row slicing must not depend on Textual span ordering."""
    content = Content(
        "abcdef",
        [
            Span(4, 6, "red"),
            Span(0, 6, "base"),
            Span(1, 3, "blue"),
        ],
    )

    sliced = _content_slice(content, 2, 5)

    assert sliced.plain == "cde"
    assert sliced.spans == [
        Span(2, 3, "red"),
        Span(0, 3, "base"),
        Span(0, 1, "blue"),
    ]


class _MdApp(App):
    def __init__(self, markdown: str) -> None:
        super().__init__()
        self._md = markdown
        self.widget: VirtualizedMarkdown | None = None

    def compose(self) -> ComposeResult:
        self.widget = VirtualizedMarkdown(self._md)
        yield self.widget


async def test_layout_blocks_defers_fence_line_strips_until_first_render() -> None:
    """Layout measures fences without pre-rendering offscreen code rows."""
    app = _MdApp(_offscreen_fence_markdown(20))
    async with app.run_test(size=(80, 5)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        fence_indices = [i for i, b in enumerate(md._blocks) if b.block_type == "fence"]
        assert fence_indices, "expected at least one fence block"
        idx = fence_indices[0]
        assert idx in md._fence_layouts, "layout should record fence height metadata"
        assert not md._fence_line_strips
        assert not md._fence_source_line_strips

        block = md._blocks[idx]
        info = md._block_line_info[idx]
        width = md.scrollable_content_region.width
        md._render_block_line(block, info, info.start_line + info.top_margin + block.padding_top, width)

        assert (idx, 0) in md._fence_line_strips
        assert (idx, 0) in md._fence_source_line_strips


async def test_layout_blocks_defers_table_strips_until_first_render() -> None:
    """Layout measures tables without pre-rendering offscreen table rows."""
    app = _MdApp(_offscreen_table_markdown(20))
    async with app.run_test(size=(80, 5)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        table_idx = next(i for i, block in enumerate(md._blocks) if block.block_type == "table")
        assert table_idx in md._table_layouts, "layout should record table height metadata"
        table_strip_keys = list(md._table_strips.keys())
        assert not [key for key in table_strip_keys if key[0] == table_idx]

        block = md._blocks[table_idx]
        info = md._block_line_info[table_idx]
        width = md.scrollable_content_region.width
        md._render_block_line(block, info, info.start_line + info.top_margin, width)

        assert (table_idx, 0) in md._table_strips


async def test_render_line_calls_render_strips_once_per_fence_source_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fence renders only the source lines that become visible."""
    app = _MdApp(_fence_markdown(40))
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        fence_idx = next(i for i, b in enumerate(md._blocks) if b.block_type == "fence")
        block = md._blocks[fence_idx]
        info = md._block_line_info[fence_idx]
        md._fence_line_strips.clear()
        md._fence_source_line_strips.clear()

        calls = {"n": 0}
        original = Content.render_strips

        def _spy(self: Content, *args, **kwargs):
            calls["n"] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Content, "render_strips", _spy)

        # Render every line of the fence directly; each source line should
        # render once, not force the entire fence to render on the first row.
        width = md.scrollable_content_region.width
        for line in range(info.start_line, info.start_line + info.height):
            md._render_block_line(block, info, line, width)

        assert calls["n"] == 40, "fence cache should render each source line once"


async def test_render_line_output_identical_with_cache() -> None:
    """Pre-cached strips must produce the same output as the cold-cache path."""
    app = _MdApp(_fence_markdown(8))
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        fence_idx = next(i for i, b in enumerate(md._blocks) if b.block_type == "fence")
        block = md._blocks[fence_idx]
        info = md._block_line_info[fence_idx]
        width = md.scrollable_content_region.width

        # Pick a line known to be inside the fence body (skip top padding).
        body_line = info.start_line + info.top_margin + block.padding_top + 2
        md._fence_line_strips.clear()
        md._fence_source_line_strips.clear()
        assert (fence_idx, 2) not in md._fence_line_strips
        cached_strip = md._render_block_line(block, info, body_line, width)
        assert (fence_idx, 2) in md._fence_line_strips

        # Drop the cached strips and render again; should match.
        md._fence_line_strips.clear()
        md._fence_source_line_strips.clear()
        fresh_strip = md._render_block_line(block, info, body_line, width)

        assert cached_strip.cell_length == fresh_strip.cell_length
        assert [(s.text, s.style) for s in cached_strip._segments] == [(s.text, s.style) for s in fresh_strip._segments]


async def test_render_line_calls_render_strips_once_per_long_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Long paragraph blocks should warm once, then serve rows from cached strips."""
    app = _MdApp(_long_paragraph_markdown(500))
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        block_idx = next(i for i, b in enumerate(md._blocks) if b.block_type == "paragraph")
        block = md._blocks[block_idx]
        info = md._block_line_info[block_idx]
        width = md.scrollable_content_region.width
        md._frame_visible_height = 5
        md._block_strips.clear()

        calls = {"n": 0}
        original = block.content.render_strips

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(block.content, "render_strips", _spy)

        for line in range(info.start_line, info.start_line + info.height):
            md._render_block_line(block, info, line, width)

        assert calls["n"] == 1, "long block cache should avoid per-row render_strips calls"
        assert block_idx in md._block_strips


async def test_short_block_does_not_use_block_strip_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocks shorter than the visible crop stay on the existing per-line cache path."""
    app = _MdApp("short paragraph")
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        block_idx = next(i for i, b in enumerate(md._blocks) if b.block_type == "paragraph")
        block = md._blocks[block_idx]
        info = md._block_line_info[block_idx]
        width = md.scrollable_content_region.width
        md._frame_visible_height = 20
        md._block_strips.clear()

        calls = {"n": 0}
        original = block.content.render_strips

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(block.content, "render_strips", _spy)

        for line in range(info.start_line, info.start_line + info.height):
            md._render_block_line(block, info, line, width)

        assert calls["n"] == info.content_height
        assert not md._block_strips


async def test_long_block_render_output_identical_with_cache() -> None:
    """Cached paragraph strips must match the fallback per-row render path."""
    app = _MdApp(_long_paragraph_markdown(200))
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        block_idx = next(i for i, b in enumerate(md._blocks) if b.block_type == "paragraph")
        block = md._blocks[block_idx]
        info = md._block_line_info[block_idx]
        width = md.scrollable_content_region.width
        body_line = info.start_line + info.top_margin + min(2, info.content_height - 1)

        md._frame_visible_height = 5
        md._block_strips.clear()
        cached_strip = md._render_block_line(block, info, body_line, width)
        assert block_idx in md._block_strips

        md._frame_visible_height = info.content_height + 1
        md._block_strips.clear()
        fresh_strip = md._render_block_line(block, info, body_line, width)

        assert cached_strip.cell_length == fresh_strip.cell_length
        assert [(s.text, s.style) for s in cached_strip._segments] == [(s.text, s.style) for s in fresh_strip._segments]


async def test_long_block_strip_cache_invalidates_on_style_update() -> None:
    """Theme / style changes must drop cached long-block strips."""
    app = _MdApp(_long_paragraph_markdown(200))
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        block_idx = next(i for i, b in enumerate(md._blocks) if b.block_type == "paragraph")
        block = md._blocks[block_idx]
        info = md._block_line_info[block_idx]
        width = md.scrollable_content_region.width
        md._frame_visible_height = 5
        md._render_block_line(block, info, info.start_line, width)
        assert md._block_strips, "expected long-block strips after first render"

        md.notify_style_update()
        assert not md._block_strips, "notify_style_update should drop cached long-block strips"


async def test_long_block_strip_cache_invalidates_on_update() -> None:
    """``update()`` rebuilds blocks; long-block strips must not survive."""
    app = _MdApp(_long_paragraph_markdown(200))
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        block_idx = next(i for i, b in enumerate(md._blocks) if b.block_type == "paragraph")
        block = md._blocks[block_idx]
        info = md._block_line_info[block_idx]
        width = md.scrollable_content_region.width
        md._frame_visible_height = 5
        md._render_block_line(block, info, info.start_line, width)
        assert md._block_strips, "expected long-block strips after first render"
        stale = dict(md._block_strips)

        await md.update("replacement paragraph")
        # A refresh may render the new document before the awaiter resumes, so
        # the check is for the old document's strips only.
        assert all(md._block_strips.get(index) is not strips for index, strips in stale.items()), (
            "update() should drop cached long-block strips"
        )


async def test_converge_layout_relayouts_when_scrollbar_shifts_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-layout when a scrollbar appearing during layout shifts the width.

    Regression guard for the 1-column offset on resize: ``_layout_blocks``
    sets ``virtual_size`` and can toggle the vertical scrollbar, shrinking
    ``scrollable_content_region.width`` by one.  A single layout would leave
    cached strips and the heights in ``_block_line_info`` sized for the
    pre-scrollbar width while the direct render path paints at the new width.
    ``_converge_layout`` must re-run until the width settles so they agree.
    """
    app = _MdApp(_long_paragraph_markdown(200))
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        # Model the scrollbar toggling the content area from 50 → 49 the first
        # time the widget lays out, then holding at 49.
        state = {"width": 50}
        real_layout = type(md)._layout_blocks
        calls = {"n": 0}

        def fake_layout(self: VirtualizedMarkdown) -> None:
            calls["n"] += 1
            real_layout(self)
            state["width"] = 49

        monkeypatch.setattr(
            type(md),
            "scrollable_content_region",
            property(lambda self: Region(0, 0, state["width"], 12)),
        )
        monkeypatch.setattr(type(md), "_layout_blocks", fake_layout)

        md._width_at_last_layout = -1  # force a fresh layout
        md._converge_layout()

        assert calls["n"] >= 2, "must re-layout after the scrollbar shifts the width"
        assert md._width_at_last_layout == 49, "layout width must match the scrollbar-adjusted viewport"


async def test_layout_blocks_invalidates_direct_lines_cache() -> None:
    """A width re-layout must not reuse a stale direct-render crop cache."""
    app = _MdApp("| A | B |\n|---|---|\n| one | two |\n")
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        md._lines_cache_result = [Strip.blank(1)]

        md._layout_blocks()

        assert md._lines_cache_result is None, "layout changes must invalidate full-crop cached strips"


async def test_content_height_relayouts_for_measured_width() -> None:
    """Auto-height measurement must not reuse a height computed at an old width.

    Replay can build markdown inside hidden/collapsed tool content, then later
    expand it at the real chat width.  Textual asks ``get_content_height`` for
    the current width during that layout pass, so the markdown widget must
    measure against that width instead of returning a stale ``virtual_size``.
    """
    md = VirtualizedMarkdown()
    md._blocks = md._build_blocks(_long_paragraph_markdown(80))
    md._layout_blocks(80)
    wide_height = md.virtual_size.height

    narrow_height = md.get_content_height(Size(20, 12), Size(20, 12), 20)

    assert narrow_height > wide_height
    assert md._width_at_last_layout == 20
    assert md.virtual_size.height == narrow_height


async def test_resize_eagerly_relayouts_before_next_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resize should update markdown layout before parent scroll/paint fast paths."""
    app = _MdApp("| A | B |\n|---|---|\n| one | two |\n")
    async with app.run_test(size=(50, 12)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        old_width = md.scrollable_content_region.width
        new_width = max(1, old_width - 3)
        state = {"width": new_width}
        refresh_calls: list[dict[str, object]] = []

        monkeypatch.setattr(
            type(md),
            "scrollable_content_region",
            property(lambda self: Region(0, 0, state["width"], 12)),
        )
        monkeypatch.setattr(md, "refresh", lambda *args, **kwargs: refresh_calls.append(kwargs))

        md._width_at_last_layout = old_width
        md._lines_cache_result = [Strip.blank(1)]
        md.on_resize(Resize(Size(new_width, 12), Size(old_width, 12)))

        assert md._width_at_last_layout == new_width
        assert md._lines_cache_result is None
        assert refresh_calls[-1] == {"layout": True}


async def test_notify_style_update_clears_fence_line_strips() -> None:
    """Theme / style changes must drop cached strips so they re-render with new colors."""
    app = _MdApp(_fence_markdown(10))
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        fence_idx = next(i for i, b in enumerate(md._blocks) if b.block_type == "fence")
        block = md._blocks[fence_idx]
        info = md._block_line_info[fence_idx]
        width = md.scrollable_content_region.width
        md._render_block_line(block, info, info.start_line + info.top_margin + block.padding_top, width)
        assert md._fence_line_strips, "expected fence line strips after first render"
        assert md._fence_source_line_strips, "expected fence source-line strips after first render"

        md.notify_style_update()
        assert not md._fence_line_strips, "notify_style_update should drop cached fence line strips"
        assert not md._fence_source_line_strips, "notify_style_update should drop source-line strips"


# ---------------------------------------------------------------------------
# Block-style cache (Tier 3): memoizes _get_block_style per block index.
# ---------------------------------------------------------------------------


async def test_get_block_style_cached_is_memoized(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_get_block_style`` runs at most once per block, not once per row."""
    app = _MdApp(_fence_markdown(40))
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None
        md._block_style_cache.clear()

        calls = {"n": 0}
        original = type(md)._get_block_style

        def _spy(self, block):
            calls["n"] += 1
            return original(self, block)

        monkeypatch.setattr(type(md), "_get_block_style", _spy)

        fence_idx = next(i for i, b in enumerate(md._blocks) if b.block_type == "fence")
        block = md._blocks[fence_idx]
        info = md._block_line_info[fence_idx]
        width = md.scrollable_content_region.width

        for line in range(info.start_line, info.start_line + info.height):
            md._render_block_line(block, info, line, width)

        # First call populates the cache; every subsequent row hits the cache.
        # (_render_padding_line is called for padding rows with the same
        # block_style passed in; it does not re-invoke _get_block_style.)
        assert calls["n"] == 1, f"expected 1 _get_block_style call, got {calls['n']}"


async def test_get_block_style_cached_returns_same_result_as_uncached() -> None:
    """Cached lookup must agree with the uncached helper for every block."""
    markdown = "# Heading\n\nParagraph text.\n\n> quoted text\n>\n> nested\n\n```python\nx = 1\n```\n\n- one\n- two\n"
    app = _MdApp(markdown)
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None
        md._block_style_cache.clear()

        for index, block in enumerate(md._blocks):
            cached = md._get_block_style_cached(index, block)
            fresh = md._get_block_style(block)
            assert cached is fresh or cached.rich_style == fresh.rich_style, (
                f"block {index} ({block.block_type}) cached style diverged from fresh"
            )


async def test_block_style_cache_invalidates_on_style_update() -> None:
    """``notify_style_update`` must drop the block-style cache."""
    app = _MdApp(_fence_markdown(5))
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        fence_idx = next(i for i, b in enumerate(md._blocks) if b.block_type == "fence")
        block = md._blocks[fence_idx]
        md._get_block_style_cached(fence_idx, block)
        assert fence_idx in md._block_style_cache

        md.notify_style_update()
        assert not md._block_style_cache, "notify_style_update should drop block-style cache"


async def test_block_style_cache_invalidates_on_update() -> None:
    """``update()`` rebuilds blocks; the cache must not survive across the rebuild."""
    app = _MdApp(_fence_markdown(5))
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None

        # Prime the cache with entries no render of the new document can produce.
        stale = Style(italic=True)
        for index, block in enumerate(md._blocks):
            md._get_block_style_cached(index, block)
            md._block_style_cache[index] = stale
        assert md._block_style_cache

        await md.update("# Different\n\nentirely new content\n")
        # A refresh may render the new document before the awaiter resumes and
        # cache its own styles, so the check is for the old entries only.
        assert all(style is not stale for style in md._block_style_cache.values()), (
            "update() should drop block-style cache"
        )


_FENCE_HEIGHT_PROBE_LINES = (
    "",
    "short",
    "x" * 19,
    "x" * 20,
    "x" * 21,
    "界" * 9 + "x",
    "界" * 10,
    "界" * 10 + "x",
    "a\tb",
    "\t\tindented",
    "word " * 5,
    "trailing" + " " * 16,
    "😀" * 10,
    "😀" * 11,
    "é" * 20,
    "alpha beta gamma delta epsilon zeta",
)


def _probe_fence_block(markdown: VirtualizedMarkdown) -> MarkdownBlock:
    body = "\n".join(_FENCE_HEIGHT_PROBE_LINES)
    blocks = markdown._build_blocks(f"```text\n{body}\n```\n")
    return next(block for block in blocks if block.block_type == "fence")


@pytest.mark.parametrize(
    ("width", "line_pad"),
    # Content itself cannot wrap to a zero-cell width, so every case keeps a
    # positive width once both line pads are taken off.
    [(1, 0), (2, 0), (7, 0), (20, 0), (21, 0), (40, 0), (3, 1), (7, 1), (22, 1), (40, 1)],
)
def test_fence_layout_rows_match_content_wrapping(width: int, line_pad: int) -> None:
    """The one-row shortcut agrees with Content's own wrap on every probe line."""
    markdown = VirtualizedMarkdown()
    block = _probe_fence_block(markdown)
    rules = {"line_pad": line_pad}
    text = block.content.plain

    expected_starts: list[int] = []
    row = 0
    for start, end in _line_ranges(text):
        expected_starts.append(row)
        row += max(1, Content(text[start:end], strip_control_codes=False).get_height(rules, width))

    layout = markdown._build_fence_layout(block, width, rules)

    assert layout.line_starts == expected_starts
    assert layout.total_height == row


def test_fence_layout_builds_content_only_for_lines_that_may_wrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measuring a fence must not build a Content for every line that fits."""
    markdown = VirtualizedMarkdown()
    overlong = "y" * 90
    body = "\n".join([*(f"fits_{index} = {index}" for index in range(50)), overlong, "a\tb"])
    block = next(block for block in markdown._build_blocks(f"```python\n{body}\n```\n") if block.block_type == "fence")
    built: list[str] = []

    def counting_content(text: str, *args: object, **kwargs: bool) -> Content:
        built.append(text)
        return Content(text, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(widget_module, "Content", counting_content)
    layout = markdown._build_fence_layout(block, 40, {})

    assert built == [overlong, "a\tb"]
    assert layout.total_height == 50 + Content(overlong).get_height({}, 40) + 1
