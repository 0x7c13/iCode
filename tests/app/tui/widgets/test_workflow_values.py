# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Node values render ``text`` as Markdown or literal lines and ``data`` as fields, each behind its own tab."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import pytest
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, Tab, Tabs

from chrys.app.tui.widgets.chat.tool_view_builders import ToolViewContent
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.app.tui.widgets.workflow import values
from chrys.app.tui.widgets.workflow.values import (
    ShownValue,
    ValueDocument,
    ValueTab,
    WorkflowValueView,
    record_values,
    shown_value,
    value_tabs,
)
from tests.support.tui_helpers import BusyWidget, click_when_settled, rich_plain
from tests.support.waiting import wait_for, wait_until

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.widget import Widget

_TEXT_TABS = (ValueTab.MARKDOWN, ValueTab.PLAIN)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plain words", ShownValue("plain words")),
        ({"text": "summary", "data": {"score": 1}}, ShownValue("summary", {"score": 1})),
        ({"text": "summary"}, ShownValue("summary")),
        ({"text": "summary", "extra": 1}, ShownValue(data={"text": "summary", "extra": 1})),
        ({"text": 3, "data": None}, ShownValue(data={"text": 3, "data": None})),
        ([1, 2], ShownValue(data=[1, 2])),
        (None, ShownValue()),
    ],
    ids=["str", "envelope", "text-only", "extra-key", "non-str-text", "list", "missing"],
)
def test_shown_value_reads_envelopes_and_keeps_every_other_shape_whole(value: Any, expected: ShownValue) -> None:
    assert shown_value(value) == expected


def test_record_values_title_join_sources_by_their_node() -> None:
    assert record_values({"value": {"text": "one"}}) == (ShownValue("one"),)
    assert record_values(
        {"sources": [{"node_id": "left", "value": {"text": "a", "data": [1]}}, {"value": "b"}, "loose"]}
    ) == (ShownValue("a", [1], "left"), ShownValue("b"), ShownValue("loose"))


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        (ValueDocument((ShownValue("text"),)), _TEXT_TABS),
        (ValueDocument((ShownValue(),)), _TEXT_TABS),
        (ValueDocument((ShownValue(data=[1]),)), (ValueTab.DATA,)),
        (ValueDocument((ShownValue("text", 0),), emits=((1, "m"),)), (*_TEXT_TABS, ValueTab.DATA, ValueTab.PROGRESS)),
        (ValueDocument((ShownValue(data=1), ShownValue("b"))), (*_TEXT_TABS, ValueTab.DATA)),
        (ValueDocument(emits=((1, "m"),)), (ValueTab.PROGRESS,)),
        (ValueDocument(placeholder="none"), ()),
    ],
    ids=["text", "empty-value", "data-only", "all", "mixed-join", "emits-only", "nothing"],
)
def test_value_tabs_offer_only_what_the_document_holds(document: ValueDocument, expected: tuple[ValueTab, ...]) -> None:
    assert value_tabs(document) == expected


class _Harness(App[None]):
    def compose(self) -> ComposeResult:
        yield WorkflowValueView(None)


def _tab_labels(view: WorkflowValueView) -> list[str]:
    return [tab.label_text for tab in view.query(Tab)]


def _body(view: WorkflowValueView) -> list[Static]:
    return list(view.query(".workflow-value-body Static").results(Static))


async def _shown(app: _Harness, document: ValueDocument, tab: ValueTab | None = None) -> WorkflowValueView:
    view = app.query_one(WorkflowValueView)
    if tab is not None:
        view.selected = tab
    await view.show(document)
    return view


async def test_text_opens_as_markdown_below_the_notice_and_its_tabs() -> None:
    app = _Harness()
    async with app.run_test(size=(100, 40)):
        view = await _shown(
            app,
            ValueDocument(
                (ShownValue("## Findings\n- first", {"count": 2}),),
                emits=((1, "started"),),
                notice="Structured data was dropped.",
            ),
        )
        assert list(view.children) == [view.header, view.body] and view.scroll is None
        notice, tabs = view.header.children
        assert isinstance(notice, Static) and str(notice.content) == "Structured data was dropped."
        assert isinstance(tabs, Tabs) and tabs.active == "workflow-value-markdown"
        assert _tab_labels(view) == ["Markdown", "Plain text", "Data", "Progress messages"]
        assert view.body.query_one(VirtualizedMarkdown).source == "## Findings\n- first"
        assert not view.query(".tool-view-params") and not view.query(".workflow-value-emits")


async def test_plain_tab_keeps_literal_lines_without_trailing_blank_rows() -> None:
    app = _Harness()
    async with app.run_test(size=(100, 40)):
        view = await _shown(app, ValueDocument((ShownValue("  indented\n# not a heading\n\n"),)), ValueTab.PLAIN)
        assert not view.query(VirtualizedMarkdown)
        assert [str(widget.content) for widget in _body(view)] == ["  indented\n# not a heading"]


async def test_data_tab_shows_fields_and_progress_tab_numbers_messages() -> None:
    app = _Harness()
    async with app.run_test(size=(100, 40)):
        document = ValueDocument(
            (ShownValue("text", {"count": 2, "path": "[red]src[/red]"}),), emits=((1, "started"), (2, "done"))
        )
        view = await _shown(app, document, ValueTab.DATA)
        params = view.query_one(".tool-view-params", Static).content
        assert isinstance(params, Table)
        assert rich_plain(params).splitlines() == ["count  2", "path   [red]src[/red]"]
        await view.select(ValueTab.PROGRESS)
        emits = view.query_one(".workflow-value-emits", Static).content
        assert isinstance(emits, Table)
        assert rich_plain(emits).splitlines() == ["1 started", "2 done"]
        assert not view.query(".tool-view-params")


@pytest.mark.parametrize(
    ("data", "renderable"), [([1, "two"], Syntax), ("literal text", Text), (7, Syntax)], ids=["list", "str", "number"]
)
async def test_data_without_text_opens_on_its_only_tab(data: Any, renderable: type) -> None:
    app = _Harness()
    async with app.run_test(size=(100, 40)):
        view = await _shown(app, ValueDocument((ShownValue(data=data),)))
        assert _tab_labels(view) == ["Data"]
        assert view.shown_tab is ValueTab.DATA and view.selected is ValueTab.MARKDOWN
        [body] = _body(view)
        assert isinstance(body.content, renderable)


async def test_oversized_data_renders_as_literal_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(values, "HIGHLIGHT_MAX_CHARS", 20)
    app = _Harness()
    async with app.run_test(size=(100, 40)):
        view = await _shown(app, ValueDocument((ShownValue(data={"key": "a long enough value"}),)))
        assert not view.query(".tool-view-params")
        assert [str(widget.content) for widget in _body(view)] == ['{\n  "key": "a long enough value"\n}']


async def test_empty_values_and_documents_say_so() -> None:
    app = _Harness()
    async with app.run_test(size=(100, 40)):
        view = await _shown(app, ValueDocument((ShownValue(),)))
        assert _tab_labels(view) == ["Markdown", "Plain text"]
        assert [str(widget.content) for widget in _body(view)] == ["(empty)"]
        await view.show(ValueDocument(placeholder="No record available."))
        assert not view.query(Tabs)
        assert [str(widget.content) for widget in view.query(Static)] == ["No record available."]


class _NoViewTabsHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield WorkflowValueView(None, view_tabs=False)


async def test_a_view_without_view_tabs_shows_its_first_tab_below_the_notice() -> None:
    app = _NoViewTabsHarness()
    async with app.run_test(size=(100, 40)):
        view = app.query_one(WorkflowValueView)
        await view.show(ValueDocument((ShownValue("## Findings", {"count": 2}),), notice="Only the summary is shown."))
        assert not view.query(Tabs)
        [notice] = view.header.children
        assert isinstance(notice, Static) and str(notice.content) == "Only the summary is shown."
        assert view.body.query_one(VirtualizedMarkdown).source == "## Findings"
        await view.show(ValueDocument((ShownValue(data=[1, "two"]),)))
        assert not view.query(Tabs) and not view.header.children
        assert view.shown_tab is ValueTab.DATA
        [body] = _body(view)
        assert isinstance(body.content, Syntax)


async def test_join_sources_render_in_boxes_titled_by_node_on_every_tab() -> None:
    app = _Harness()
    async with app.run_test(size=(100, 40)):
        document = ValueDocument((ShownValue("a", source="left"), ShownValue(data=[2], source="right")))
        view = await _shown(app, document)
        boxes = list(view.query(".workflow-value-source"))
        assert [str(box.border_title) for box in boxes] == ["left", "right"]
        assert boxes[0].query_one(VirtualizedMarkdown).source == "a"
        assert str(boxes[1].query_one(Static).content) == "(empty)"
        await view.select(ValueTab.DATA)
        boxes = list(view.query(".workflow-value-source"))
        assert [str(box.border_title) for box in boxes] == ["left", "right"]
        assert str(boxes[0].query_one(Static).content) == "(empty)"
        assert isinstance(boxes[1].query_one(Static).content, Syntax)


async def test_a_pane_keeps_its_tab_across_documents_and_rebuilds_only_what_changed() -> None:
    app = _Harness()
    async with app.run_test(size=(100, 40)):
        document = ValueDocument((ShownValue("same", {"k": 1}),))
        view = await _shown(app, document)
        tabs, markdown = view.query_one(Tabs), view.query_one(VirtualizedMarkdown)
        await view.show(ValueDocument((ShownValue("same", {"k": 1}),)))
        assert view.query_one(VirtualizedMarkdown) is markdown

        await view.select(ValueTab.DATA)
        assert not markdown.is_attached and view.query_one(Tabs) is tabs
        assert tabs.active == "workflow-value-data"
        # Another record with the same tabs keeps the tab bar too.
        await view.show(ValueDocument((ShownValue("other", {"k": 2}),)))
        assert view.query_one(Tabs) is tabs
        assert rich_plain(view.query_one(".tool-view-params", Static).content).splitlines() == ["k  2"]
        # A document without data falls back to its first tab but remembers the reader's choice.
        await view.show(ValueDocument((ShownValue("next"),)))
        assert view.shown_tab is ValueTab.MARKDOWN and view.selected is ValueTab.DATA
        assert view.query_one(Tabs).active == "workflow-value-markdown"
        await view.show(document)
        assert view.query_one(Tabs).active == "workflow-value-data" and view.query(".tool-view-params")


async def test_clicking_a_tab_switches_the_body() -> None:
    app = _Harness()
    async with app.run_test(size=(100, 40)) as pilot:
        view = await _shown(app, ValueDocument((ShownValue("**bold**"),)))
        tabs = view.query_one(Tabs)
        await click_when_settled(pilot, tabs.query_one("#workflow-value-plain", Tab))
        await wait_for(
            lambda: [str(widget.content) for widget in _body(view)] == ["**bold**"],
            pilot=pilot,
            description="plain text tab shown",
        )
        assert view.selected is ValueTab.PLAIN and view.query_one(Tabs) is tabs and app.focused is tabs


class _ScrollHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield WorkflowValueView(None, Static("after", id="trailing"), scroll=True)


async def test_a_scrolling_view_keeps_its_tab_bar_and_trailing_widgets_in_place() -> None:
    app = _ScrollHarness()
    async with app.run_test(size=(60, 20)) as pilot:
        view = app.query_one(WorkflowValueView)
        scroll, trailing = view.scroll, app.query_one("#trailing", Static)
        assert scroll is not None and list(view.children) == [view.header, scroll]
        await view.show(ValueDocument((ShownValue("\n".join(f"line {index}" for index in range(60)), {"k": 1}),)))
        tabs = view.query_one(Tabs)
        # The scrollbar that the overflow turns on is placed by a later layout pass.
        await wait_for(
            lambda: scroll.max_scroll_y > 0 and scroll.vertical_scrollbar.region.height > 0,
            pilot=pilot,
            description="content overflows the view and its scrollbar is placed",
        )
        assert scroll.region.y == tabs.region.bottom + 1
        assert scroll.vertical_scrollbar.region.y == scroll.region.y
        tabs_y = tabs.region.y
        scroll.scroll_end(animate=False, immediate=True)
        await wait_for(lambda: scroll.scroll_y == scroll.max_scroll_y, pilot=pilot)
        assert tabs.region.y == tabs_y

        # Another tab opens at its top; the widgets after the content survive every rebuild.
        await view.select(ValueTab.DATA)
        assert scroll.scroll_y == 0 and list(scroll.children) == [view.body, trailing]
        await view.show(ValueDocument(placeholder="No record available."))
        assert not view.header.children and list(scroll.children) == [view.body, trailing]
        assert [str(widget.content) for widget in _body(view)] == ["No record available."]


class _ActivationSpy(WorkflowValueView):
    def __init__(self) -> None:
        super().__init__(None)
        self.activations: list[str] = []

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        self.activations.append(event.tab.id or "")


class _SpyHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield _ActivationSpy()


async def test_rebuilds_move_the_tab_bar_without_posting_activations() -> None:
    app = _SpyHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        view = app.query_one(_ActivationSpy)
        # Mount a bar, move it, then replace it with one on its first tab.
        await view.show(ValueDocument((ShownValue("text", {"k": 1}),)))
        await view.select(ValueTab.DATA)
        await view.show(ValueDocument((ShownValue("text"),), emits=((1, "m"),)))
        await click_when_settled(pilot, view.query_one("#workflow-value-plain", Tab))
        await wait_for(
            lambda: view.activations == ["workflow-value-plain"],
            pilot=pilot,
            description="the reader's click is the only activation",
        )


async def test_a_refresh_that_replaces_the_tab_bar_keeps_the_readers_tab() -> None:
    app = _Harness()
    async with app.run_test(size=(100, 40)) as pilot:
        view = await _shown(app, ValueDocument((ShownValue("text", {"k": 1}),)), ValueTab.DATA)
        # The first refresh has no data, so it mounts a bar on its first tab; the second brings
        # the data back before that bar's mount is handled.
        await asyncio.gather(
            view.show(ValueDocument((ShownValue("text"),), emits=((1, "m"),))),
            view.show(ValueDocument((ShownValue("text", {"k": 2}),), emits=((1, "m"),))),
        )
        assert not await wait_until(lambda: view.selected is not ValueTab.DATA, pilot=pilot, timeout=0.5)
        assert view.query_one(Tabs).active == "workflow-value-data"
        assert rich_plain(view.query_one(".tool-view-params", Static).content).splitlines() == ["k  2"]


async def test_activations_from_a_replaced_tab_bar_are_ignored() -> None:
    app = _Harness()
    async with app.run_test(size=(100, 40)) as pilot:
        view = await _shown(app, ValueDocument((ShownValue("text", {"k": 1}),)))
        old = view.query_one(Tabs)
        data = old.query_one("#workflow-value-data", Tab)
        await view.show(ValueDocument((ShownValue("text"),), emits=((1, "m"),)))
        assert view.query_one(Tabs) is not old
        # A click on the old bar that was still on its way when the refresh replaced the bar.
        view.post_message(Tabs.TabActivated(old, data))
        assert not await wait_until(lambda: view.selected is not ValueTab.MARKDOWN, pilot=pilot, timeout=0.5)
        await click_when_settled(pilot, view.query_one("#workflow-value-plain", Tab))
        await wait_for(lambda: view.selected is ValueTab.PLAIN, pilot=pilot, description="the current bar switches")


async def test_switching_again_while_widgets_are_being_removed_leaves_none_behind() -> None:
    app = _Harness()
    async with app.run_test(size=(100, 40)) as pilot:
        view = await _shown(app, ValueDocument((ShownValue("text", {"k": 1}),)))
        tabs, markdown = view.query_one(Tabs), view.query_one(VirtualizedMarkdown)
        tabs.focus()
        busy = BusyWidget()
        holder = Vertical(busy)
        await view.body.mount(holder)
        busy.hold()
        try:
            await wait_for(lambda: busy.holding, description="the busy widget holds its message loop")
            tabs.action_next_tab()
            await wait_for(lambda: not markdown.is_attached, description="the switch to Plain is removing the body")
            # The switch to Data cancels the one to Plain while it waits for the busy widget.
            tabs.action_next_tab()
            await wait_for(lambda: view.selected is ValueTab.DATA, description="the second switch is handled")
        finally:
            busy.release.set()
        await wait_for(lambda: bool(view.body.query(".tool-view-params")), pilot=pilot, description="data shown")
        assert not holder.is_attached and [type(child) for child in view.body.children] == [ToolViewContent]
        assert view.query_one(Tabs) is tabs and tabs.active == "workflow-value-data" and app.focused is tabs
        tabs.action_previous_tab()
        await wait_for(
            lambda: [str(widget.content) for widget in _body(view)] == ["text"],
            pilot=pilot,
            description="the view still switches",
        )


class _HeaderProbe(WorkflowValueView):
    """Calls ``on_bar_built`` as soon as a rebuild has built a tab bar, just before it mounts the bar."""

    def __init__(self) -> None:
        super().__init__(None)
        self.on_bar_built: Callable[[], None] | None = None

    def _header_widgets(self, document: ValueDocument, tab: ValueTab | None) -> list[Widget]:
        widgets = super()._header_widgets(document, tab)
        if self.on_bar_built is not None:
            self.on_bar_built()
        return widgets


class _ProbeHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield _HeaderProbe()


def _after_turns(turns: int, action: Callable[[], object]) -> None:
    """Run ``action`` once ``turns`` more loop turns have passed, waking no task in between.

    The turns pick how far the bar's nested mount has got when ``action`` removes it.
    """
    if turns:
        asyncio.get_running_loop().call_soon(_after_turns, turns - 1, action)
    else:
        action()


@pytest.mark.parametrize("turns", range(6))
async def test_removing_the_view_while_its_tab_bar_mounts_leaves_the_app_running(turns: int) -> None:
    """The node dialog closes while its records load, so a value's tab bar never gets its tabs."""
    app = _ProbeHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        view = app.query_one(_HeaderProbe)
        view.on_bar_built = lambda: _after_turns(turns, view.remove)
        await view.show(ValueDocument((ShownValue("text", {"k": 1}),)))
        await wait_for(lambda: not view.is_attached, pilot=pilot, description="the view is removed")
        assert app.is_running
    # Leaving run_test re-raises what took the App down, such as "No Tab with id ...".


@pytest.mark.parametrize("turns", range(6))
async def test_a_new_header_can_replace_a_bar_left_mounting_by_a_cancelled_show(turns: int) -> None:
    """Selecting another attempt cancels the record loader while it mounts a value's tab bar."""
    app = _ProbeHarness()
    async with app.run_test(size=(100, 40)):
        view = app.query_one(_HeaderProbe)
        showing = asyncio.create_task(view.show(ValueDocument((ShownValue("text", {"k": 1}),))))
        view.on_bar_built = lambda: _after_turns(turns, showing.cancel)
        with contextlib.suppress(asyncio.CancelledError):
            await showing
        view.on_bar_built = None
        # The notice changes the header, so this rebuild removes the half-mounted bar.
        await view.show(ValueDocument((ShownValue("other"),), notice="dropped"))
        notice, tabs = view.header.children
        assert isinstance(notice, Static) and str(notice.content) == "dropped"
        assert isinstance(tabs, Tabs) and tabs.active == "workflow-value-markdown"
        assert _tab_labels(view) == ["Markdown", "Plain text"]
        assert view.body.query_one(VirtualizedMarkdown).source == "other"
