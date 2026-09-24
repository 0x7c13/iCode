# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Readable node values: tabs show ``text`` as Markdown or literal lines, and ``data`` as fields."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rich.table import Table
from rich.text import Text
from textual import on
from textual.containers import Vertical, VerticalGroup, VerticalScroll
from textual.widgets import Static, Tab, Tabs

from chrys.app.tui.util.removal import remove_children_shielded
from chrys.app.tui.util.source_text import sanitize_source_text
from chrys.app.tui.widgets.chat.tool_view_builders import (
    TOOL_VIEW_EMPTY,
    ToolViewContent,
    build_code_view,
    build_params_view,
)
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.app.tui.widgets.markdown.parser import create_line_break_markdown_parser
from chrys.app.tui.widgets.workflow import text

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.widget import Widget

    from chrys.app.tui.i18n import LocaleController
    from chrys.foundation.i18n import MessageRef

HIGHLIGHT_MAX_CHARS = 200_000
"""Larger data renders as literal JSON: highlighting and table layout run on the UI thread."""

_TAB_ID_PREFIX = "workflow-value-"


class ValueTab(StrEnum):
    MARKDOWN = "markdown"
    PLAIN = "plain"
    DATA = "data"
    PROGRESS = "progress"


_TAB_LABELS = {
    ValueTab.MARKDOWN: text.VALUE_MARKDOWN,
    ValueTab.PLAIN: text.VALUE_PLAIN,
    ValueTab.DATA: text.VALUE_DATA,
    ValueTab.PROGRESS: text.VALUE_PROGRESS,
}


@dataclass(frozen=True, slots=True)
class ShownValue:
    """One ``WorkflowValue``; ``source`` names the join input it arrived from."""

    text: str = ""
    data: Any = None
    source: str = ""


@dataclass(frozen=True, slots=True)
class ValueDocument:
    """Everything one record pane shows; ``placeholder`` stands in when there is nothing else."""

    values: tuple[ShownValue, ...] = ()
    emits: tuple[tuple[int, str], ...] = ()
    notice: str = ""
    placeholder: str = ""


def shown_value(value: Any, *, source: str = "") -> ShownValue:
    """Read a stored ``{text, data}`` envelope; any other shape is kept whole as data."""
    if isinstance(value, str):
        return ShownValue(value, source=source)
    if isinstance(value, dict) and isinstance(value.get("text"), str) and value.keys() <= {"text", "data"}:
        return ShownValue(value["text"], value.get("data"), source)
    return ShownValue(data=value, source=source)


def record_values(record: Mapping[str, Any]) -> tuple[ShownValue, ...]:
    """A node record's ``value``, or each of a join's ``sources`` titled by its node."""
    sources = record.get("sources")
    if not isinstance(sources, list):
        return (shown_value(record.get("value")),)
    return tuple(
        shown_value(source.get("value"), source=str(source.get("node_id", "")))
        if isinstance(source, dict)
        else shown_value(source)
        for source in sources
    )


def value_tabs(document: ValueDocument) -> tuple[ValueTab, ...]:
    """The tabs with something to show: text only when a value has some, or has no data either."""
    tabs: list[ValueTab] = []
    if any(value.text or value.data is None for value in document.values):
        tabs += (ValueTab.MARKDOWN, ValueTab.PLAIN)
    if any(value.data is not None for value in document.values):
        tabs.append(ValueTab.DATA)
    if document.emits:
        tabs.append(ValueTab.PROGRESS)
    return tuple(tabs)


def _header_key(document: ValueDocument) -> tuple[str, tuple[ValueTab, ...]]:
    return document.notice, value_tabs(document)


class WorkflowValueView(Vertical):
    """Value tabs over the shown tab's content; each view keeps the reader's tab while its records change.

    With ``scroll`` the tab bar stays put above a scrolling area holding the content, followed by
    ``trailing`` widgets that the view never rebuilds.
    """

    DEFAULT_CSS = """
    WorkflowValueView { width: 100%; height: auto; }
    WorkflowValueView.-scroll { height: 1fr; }
    WorkflowValueView .workflow-value-header,
    WorkflowValueView .workflow-value-body { width: 100%; height: auto; }
    WorkflowValueView .workflow-value-scroll { height: 1fr; scrollbar-size-vertical: 1; }
    WorkflowValueView .workflow-value-notice {
        width: 100%; height: auto; margin: 0 0 1 0; padding: 0 1;
        border-left: outer $tui-border-warning; background: $warning 10%; color: $text-warning;
    }
    WorkflowValueView .workflow-value-tabs { margin: 0 0 1 0; }
    WorkflowValueView .workflow-value-body .tool-view-md,
    WorkflowValueView .workflow-value-body .tool-view-text,
    WorkflowValueView .workflow-value-body .tool-view-code,
    WorkflowValueView .workflow-value-body .tool-view-params,
    WorkflowValueView .workflow-value-body .tool-view-section-title { padding: 0 1; }
    WorkflowValueView .workflow-value-empty { color: $text-muted; text-style: italic; }
    WorkflowValueView .workflow-value-source {
        width: 100%; height: auto; margin: 0 0 1 0;
        border: round $tui-border-primary 40%; border-title-color: $text-primary; border-title-style: bold;
    }
    WorkflowValueView .workflow-value-body > *:last-child,
    WorkflowValueView .workflow-value-body ToolViewContent > *:last-child { margin-bottom: 0; }
    WorkflowValueView .workflow-value-emits { width: 100%; height: auto; padding: 0 1; }
    WorkflowValueView .workflow-value-placeholder { width: 100%; height: auto; color: $text-muted; }
    """

    def __init__(
        self, locale: LocaleController | None, *trailing: Widget, scroll: bool = False, id: str | None = None
    ) -> None:
        super().__init__(id=id, classes="-scroll" if scroll else "")
        self._locale = locale
        self._trailing = trailing
        self.header = VerticalGroup(classes="workflow-value-header")
        """The notice and the tab bar."""
        self.body = VerticalGroup(classes="workflow-value-body")
        """The shown tab's content, or the placeholder."""
        self.scroll = VerticalScroll(classes="workflow-value-scroll") if scroll else None
        self.document = ValueDocument()
        self.selected = ValueTab.MARKDOWN
        """The reader's tab; a document without it shows its first tab instead."""
        self._shown: tuple[ValueDocument, ValueTab | None] | None = None
        self._header_shown: tuple[str, tuple[ValueTab, ...]] | None = None
        self._tabs: Tabs | None = None
        """The tab bar of the header; activations from a bar the header has replaced are ignored."""
        self._lock = asyncio.Lock()

    def compose(self) -> ComposeResult:
        yield self.header
        if self.scroll is None:
            yield self.body
            yield from self._trailing
            return
        with self.scroll:
            yield self.body
            yield from self._trailing

    @property
    def shown_tab(self) -> ValueTab | None:
        tabs = value_tabs(self.document)
        return self.selected if self.selected in tabs else next(iter(tabs), None)

    async def show(self, document: ValueDocument) -> None:
        """Display the latest document; a caller that waited for an older rebuild renders the newest one."""
        self.document = document
        await self._rebuild()

    async def select(self, tab: ValueTab) -> None:
        self.selected = tab
        await self._rebuild()

    @on(Tabs.TabActivated, ".workflow-value-tabs")
    def tab_activated(self, event: Tabs.TabActivated) -> None:
        """The reader's choice: rebuilds change the tab bar without posting activations."""
        event.stop()
        tab = ValueTab((event.tab.id or "").removeprefix(_TAB_ID_PREFIX))
        if event.tabs is not self._tabs or tab is self.shown_tab:
            return
        # Recorded now, so that the next activation compares against it before the rebuild runs.
        self.selected = tab
        # Rebuilding awaits mounts that this view's own message pump completes.
        self.run_worker(self._rebuild(), group="workflow-value-tab", exclusive=True)

    async def _rebuild(self) -> None:
        async with self._lock:
            document, tab = self.document, self.shown_tab
            shown = self._shown
            if (document, tab) == shown or not self.is_attached:
                return
            # A cancelled rebuild must not leave a stale match behind.
            self._shown = None
            with self.app.batch_update():
                header = _header_key(document)
                if header != self._header_shown:
                    self._header_shown = self._tabs = None
                    await remove_children_shielded(self.header)
                    # Removal runs through the message loop, which may have detached this view meanwhile.
                    if not self.header.is_attached:
                        return
                    await self.header.mount_all(self._header_widgets(document, tab))
                    self._header_shown = header
                elif self._tabs is not None and tab is not None:
                    # The same tabs: keep the tab bar, and the keyboard focus on it.
                    with self.prevent(Tabs.TabActivated):
                        self._tabs.active = _TAB_ID_PREFIX + tab
                await remove_children_shielded(self.body)
                if not self.body.is_attached:
                    return
                await self.body.mount_all(self._body_widgets(document, tab))
                if self.scroll is not None and shown is not None and shown[0] == document:
                    # Another tab of the same record opens at its top.
                    self.scroll.scroll_home(animate=False, immediate=True)
            self._shown = (document, tab)

    def _text(self, message: MessageRef) -> str:
        return text.render(message, self._locale)

    def _dark(self) -> bool:
        return self.app.current_theme.dark

    def _header_widgets(self, document: ValueDocument, tab: ValueTab | None) -> list[Widget]:
        widgets: list[Widget] = []
        if document.notice:
            widgets.append(Static(Text(document.notice), classes="workflow-value-notice"))
        if tab is not None:
            tabs = [
                Tab(Text(self._text(_TAB_LABELS[option].bind())), id=_TAB_ID_PREFIX + option)
                for option in value_tabs(document)
            ]
            # A bar posts an activation for its first tab when it mounts; that is not the reader's choice.
            with self.prevent(Tabs.TabActivated):
                self._tabs = Tabs(*tabs, active=_TAB_ID_PREFIX + tab, classes="workflow-value-tabs")
            widgets.append(self._tabs)
        return widgets

    def _body_widgets(self, document: ValueDocument, tab: ValueTab | None) -> list[Widget]:
        if tab is None:
            return [Static(Text(document.placeholder), classes="workflow-value-placeholder")]
        return self._tab_widgets(document, tab)

    def _tab_widgets(self, document: ValueDocument, tab: ValueTab) -> list[Widget]:
        if tab is ValueTab.PROGRESS:
            table = Table.grid(padding=(0, 1))
            table.add_column(style="dim", justify="right", no_wrap=True)
            table.add_column(overflow="fold")
            for ordinal, line in document.emits:
                table.add_row(str(ordinal), Text(sanitize_source_text(line, tab_size=8)))
            return [Static(table, classes="workflow-value-emits")]
        widgets: list[Widget] = []
        for value in document.values:
            content = ToolViewContent(self._value_widgets(value, tab))
            if value.source:
                box = Vertical(content, classes="workflow-value-source")
                box.border_title = Text(value.source)
                widgets.append(box)
            else:
                widgets.append(content)
        return widgets

    def _value_widgets(self, value: ShownValue, tab: ValueTab) -> list[Widget]:
        if tab is ValueTab.DATA:
            return self._empty() if value.data is None else self._data_widgets(value.data)
        if not value.text:
            return self._empty()
        if tab is ValueTab.MARKDOWN:
            # Values are often line-oriented program output, so newlines stay line breaks.
            return [
                VirtualizedMarkdown(
                    sanitize_source_text(value.text),
                    classes="tool-view-md",
                    parser_factory=create_line_break_markdown_parser,
                )
            ]
        return [build_code_view("text", value.text.rstrip("\r\n"), render_message=self._text)]

    def _data_widgets(self, data: Any) -> list[Widget]:
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        if len(pretty) > HIGHLIGHT_MAX_CHARS:
            return [build_code_view("text", pretty, render_message=self._text)]
        if isinstance(data, dict) and data:
            return build_params_view(data, dark=self._dark(), render_message=self._text)
        if isinstance(data, str):
            return [build_code_view("text", data, render_message=self._text)]
        return [build_code_view("json", pretty, dark=self._dark(), render_message=self._text)]

    def _empty(self) -> list[Widget]:
        return [Static(Text(self._text(TOOL_VIEW_EMPTY.bind())), classes="tool-view-text workflow-value-empty")]
