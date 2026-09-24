# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Run input and output text sit one column in from each side; the scrollbar stays on the border."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, TabbedContent

from chrys.app.tui.theme import TuiVariableDefaultsMixin
from chrys.app.tui.widgets.workflow.output import WorkflowOutputText, WorkflowOutputView, WorkflowStatusOutput
from chrys.app.tui.widgets.workflow.panel import WorkflowPanel
from tests.support.pilot_barrier import screen_is_settled
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from textual.widget import Widget


class Harness(TuiVariableDefaultsMixin, App):
    def compose(self) -> ComposeResult:
        panel = WorkflowPanel()
        panel.display = True
        yield panel


_LONG_TEXT = "\n".join(f"line {number} " + "word " * 20 for number in range(60))


def _inset(text: Widget, scroll: VerticalScroll) -> tuple[int, int]:
    content = scroll.scrollable_content_region
    return text.region.x - content.x, content.right - text.region.right


def _overflows_against_the_border(scroll: VerticalScroll, panel: WorkflowPanel) -> bool:
    return (
        scroll.show_vertical_scrollbar
        and scroll.region.right == panel.content_region.right
        and scroll.scrollable_content_region.right == scroll.region.right - 1
    )


async def test_run_input_and_output_text_keep_a_column_from_each_side() -> None:
    app = Harness()
    async with app.run_test(size=(60, 24)) as pilot:
        panel = app.query_one(WorkflowPanel)
        panel.remove_class("-empty")
        views = panel.query_one("#workflow-run", TabbedContent)

        views.active = "workflow-input-tab"
        run_input = panel.query_one("#workflow-run-input", Static)
        run_input.update(Text(_LONG_TEXT))
        input_scroll = panel.query_one("#workflow-input-scroll", VerticalScroll)
        # The scrollbar appears one layout before the text is laid out again beside it, so the
        # text keeps its full width until the screen settles.
        await wait_for(
            lambda: _overflows_against_the_border(input_scroll, panel) and screen_is_settled(app, app.screen),
            pilot=pilot,
            description="the run input overflows its pane",
        )
        assert _inset(run_input, input_scroll) == (1, 1)

        views.active = "workflow-output-tab"
        output = panel.query_one(WorkflowOutputView)
        output.show_iterations({"loop": (1, 3)})
        output.query_one(WorkflowStatusOutput).show_run(None)
        panel.show_outputs((WorkflowOutputText("node", _LONG_TEXT),))
        output_scroll = output.query_one("#workflow-outputs-scroll", VerticalScroll)
        texts = list(output_scroll.query_children(Static))
        assert [text.id for text in texts] == ["workflow-iterations", "workflow-status-output", "workflow-outputs"]
        await wait_for(
            lambda: _overflows_against_the_border(output_scroll, panel) and screen_is_settled(app, app.screen),
            pilot=pilot,
            description="the run outputs overflow their pane",
        )
        assert [_inset(text, output_scroll) for text in texts] == [(1, 1)] * 3
