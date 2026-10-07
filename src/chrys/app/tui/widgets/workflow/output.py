# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded display of workflow status facts; complete records remain in the run store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from rich.style import Style
from rich.text import Text
from textual import on
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static

from chrys.app.tui.widgets.page_navigator import PageNavigator
from chrys.app.tui.widgets.workflow import text
from chrys.foundation.events import types as events

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController
    from chrys.app.tui.widgets.workflow.projector import ObservedRun


OUTPUT_PAGE_SIZE = 32_768


@dataclass(frozen=True, slots=True)
class WorkflowOutputText:
    node_id: str
    text: str
    summary_only: bool = False

    def render(self, locale: LocaleController | None) -> str:
        note = text.render(text.OUTPUT_SUMMARY_ONLY.bind(), locale) if self.summary_only else ""
        return "\n".join(part for part in (self.node_id, note, self.text) if part)


class WorkflowOutputView(Vertical):
    """Keep every result accessible while bounding the text rendered on each page."""

    DEFAULT_CSS = """
    WorkflowOutputView { height: 1fr; }
    WorkflowOutputView #workflow-outputs-scroll { height: 1fr; padding: 0; scrollbar-size-vertical: 1; }
    WorkflowOutputView #workflow-outputs-scroll > Static { margin: 0 1; height: auto; }
    /* Whole margins: a lone margin-top or margin-bottom would zero the side margins above. */
    WorkflowOutputView #workflow-outputs-scroll > #workflow-iterations { display: none; margin: 0 1 1 1; }
    WorkflowOutputView #workflow-outputs-scroll > #workflow-outputs { margin: 1 1 0 1; }
    WorkflowOutputView #workflow-output-pages { width: 1fr; margin: 1 0; padding: 0 1; }
    """

    def __init__(self, locale: LocaleController | None) -> None:
        super().__init__()
        self._locale = locale
        self._results: tuple[WorkflowOutputText, ...] | None = None
        self._output = ""
        self._page = 0
        self._page_ends: list[int] = []

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="workflow-outputs-scroll"):
            yield Static(id="workflow-iterations")
            yield WorkflowStatusOutput(self._locale)
            yield Static(id="workflow-outputs")
        pages = PageNavigator(self._locale, id="workflow-output-pages")
        pages.display = False
        yield pages

    def clear(self) -> None:
        self.show_iterations({})
        self._results = None
        self._output = ""
        self._page = 0
        self._page_ends.clear()
        self.query_one("#workflow-outputs", Static).update(Text(""))
        self.query_one("#workflow-output-pages").display = False
        self.query_one(WorkflowStatusOutput).clear()

    def show_iterations(self, iterations: dict[str, tuple[int, int]]) -> None:
        label = "  ".join(
            f"{loop_id} {text.iteration_label(iteration, maximum, self._locale)}"
            for loop_id, (iteration, maximum) in iterations.items()
        )
        widget = self.query_one("#workflow-iterations", Static)
        widget.display = bool(label)
        widget.update(Text(label), layout=False)

    def show_outputs(self, results: tuple[WorkflowOutputText, ...]) -> None:
        if results == self._results:
            return
        self._results = results
        self._page = 0
        self._paginate()
        self._show_page()

    def _paginate(self) -> None:
        if self._results is None:
            raise RuntimeError("Paginating workflow outputs requires loaded results.")
        output = "\n\n".join(result.render(self._locale) for result in self._results)
        self._output = output = output or text.render(text.NO_OUTPUTS.bind(), self._locale)
        self._page_ends = []
        start = 0
        while start < len(output):
            end = min(start + OUTPUT_PAGE_SIZE, len(output))
            if end < len(output):
                # Prefer complete lines without leaving a nearly empty page before a long line.
                newline = output.rfind("\n", start, end)
                if newline >= start + OUTPUT_PAGE_SIZE // 2:
                    end = newline + 1
            self._page_ends.append(end)
            start = end
        self._page = min(self._page, len(self._page_ends) - 1)

    def refresh_localization(self) -> None:
        self.query_one(WorkflowStatusOutput).refresh_localization()
        self.query_one(PageNavigator).refresh_localization()
        if self._results is not None:
            self._paginate()
            self._show_page()

    def _show_page(self) -> None:
        pages = len(self._page_ends)
        start = self._page_ends[self._page - 1] if self._page else 0
        end = self._page_ends[self._page] if pages else 0
        self.query_one("#workflow-outputs", Static).update(
            Text(text.render(text.OUTPUTS.bind(), self._locale) + "\n" + self._output[start:end])
        )
        navigator = self.query_one(PageNavigator)
        navigator.display = pages > 1
        navigator.show(self._page + 1, pages)

    @on(PageNavigator.Changed, "#workflow-output-pages")
    def change_page(self, event: PageNavigator.Changed) -> None:
        event.stop()
        page = event.page - 1
        if 0 <= page < len(self._page_ends):
            self.screen.set_focus(self.query_one("#workflow-outputs-scroll", VerticalScroll), scroll_visible=False)
            self._page = page
            self._show_page()
            self.call_after_refresh(self._scroll_to_output, self._output, self._page)

    def _scroll_to_output(self, output: str, page: int) -> None:
        if not self.is_attached or self._output is not output or self._page != page:
            return
        self.query_one("#workflow-outputs-scroll", VerticalScroll).scroll_to_widget(
            self.query_one("#workflow-outputs"), top=True, animate=False
        )


def status_output(run: ObservedRun | None, locale: LocaleController | None, *, warning: Style, error: Style) -> Text:
    content = Text()
    content.append(text.render(text.STATUS_MESSAGES.bind(), locale), style="bold")
    content.append("\n", style="")
    if run is None:
        content.append(text.state_label("idle", locale), style="dim")
        return content
    content.append(text.shown(run.started.title) + "\n", style="")
    for notice in run.notices.values():
        content.append(notice.message + "\n", style=warning)
    for event in run.facts:
        if isinstance(event, events.WorkflowNodeStateChanged):
            content.append(
                f"{event.activation_id} · {text.state_label(event.state, locale)}"
                + (f" · {event.error}" if event.error else "")
                + "\n",
                style=error if event.error else "",
            )
        elif isinstance(event, events.WorkflowNodeOutput) and event.kind == events.WORKFLOW_OUTPUT_EMIT:
            content.append(f"{event.activation_id}: {event.summary_text}\n", style="")
        elif isinstance(event, events.WorkflowLoopIteration):
            content.append(f"{event.loop_id} #{event.iteration} · {event.verdict}\n", style="")
        elif isinstance(event, events.WorkflowRunFinished):
            content.append(text.state_label(event.outcome, locale) + "\n", style="bold")
            if event.error:
                content.append(event.error + "\n", style=error)
    # A trailing newline would render a blank row below the last fact on top of the outputs' margin.
    content.rstrip()
    return content


class WorkflowStatusOutput(Static):
    """Repaint cached facts on theme/locale changes as well as new run events."""

    COMPONENT_CLASSES: ClassVar[set[str]] = {"workflow-output--warning", "workflow-output--error"}
    DEFAULT_CSS = """
    WorkflowStatusOutput .workflow-output--warning { color: $warning; }
    WorkflowStatusOutput .workflow-output--error { color: $error; }
    """

    def __init__(self, locale: LocaleController | None) -> None:
        super().__init__("", id="workflow-status-output")
        self._locale = locale
        self._run: ObservedRun | None = None
        self._revision: tuple[str, int] | None = None

    def clear(self) -> None:
        self._run = None
        self._revision = None
        self.update(Text(""))

    def show_run(self, run: ObservedRun | None) -> None:
        self._run = run
        revision = (run.started.run_id, run.fact_count) if run else ("", 0)
        if revision != self._revision:
            self._revision = revision
            self.refresh_localization()

    def refresh_localization(self) -> None:
        if self._revision is not None:
            self.update(
                status_output(
                    self._run,
                    self._locale,
                    warning=self.get_component_rich_style("workflow-output--warning"),
                    error=self.get_component_rich_style("workflow-output--error"),
                )
            )

    def notify_style_update(self) -> None:
        super().notify_style_update()
        if self.is_mounted:
            # Resolve colors only once Textual has rebuilt the component styles.
            self.call_later(self.refresh_localization)
