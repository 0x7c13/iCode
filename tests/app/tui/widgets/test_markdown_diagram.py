# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Integration tests for Mermaid blocks in VirtualizedMarkdown."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from markdown_it import MarkdownIt
from markdown_it.rules_block import fence as markdown_it_fence
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.selection import SELECT_ALL

from chrys.app.tui.widgets.markdown.diagram.messages import DiagramOpenRequested
from chrys.app.tui.widgets.markdown.diagram.model import DiagnosticSeverity
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from chrys.foundation.i18n import Localizer, MessageRef

if TYPE_CHECKING:
    from chrys.app.tui.widgets.markdown.diagram.model import CompiledDiagram


_FLOWCHART = """```mermaid
flowchart LR
    Request --> Engine
    Engine --> Model
```
"""


class _DiagramMarkdownApp(App):
    CSS = "VirtualizedMarkdown { padding: 0; }"

    def __init__(self, markdown: str, *, parser_factory: Callable[[], MarkdownIt] | None = None) -> None:
        super().__init__()
        self._markdown = markdown
        self._parser_factory = parser_factory
        self.opened: CompiledDiagram | None = None

    def compose(self) -> ComposeResult:
        yield VirtualizedMarkdown(self._markdown, parser_factory=self._parser_factory)

    def on_diagram_open_requested(self, event: DiagramOpenRequested) -> None:
        self.opened = event.diagram
        event.stop()


async def test_closed_mermaid_fence_becomes_diagram_block() -> None:
    app = _DiagramMarkdownApp(_FLOWCHART)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        assert [block.block_type for block in markdown._blocks] == ["diagram"]
        diagram = markdown._blocks[0].diagram
        assert diagram is not None
        assert diagram.kind.value == "flowchart"

        rendered = [
            "".join(segment.text for segment in markdown.render_line(y)._segments).rstrip()
            for y in range(markdown.virtual_size.height)
        ]
        assert any("Request" in row for row in rendered)
        assert any("Open full diagram" in row for row in rendered)


@pytest.mark.parametrize(
    ("kind", "body", "expected"),
    [
        ("pie", 'pie title 浏览器市场份额\n"Chrome": 65\n"Safari": 35', "浏览器市场份额"),
        ("xychart", "xychart\nx-axis [Q1, Q2]\nbar [10, 20]", "Q1"),
        ("quadrant", "quadrantChart\nCandidate: [0.25, 0.75]", "Candidate"),
        ("treemap", 'treemap-beta\n"Products"\n    "Desktop": 40', "Desktop"),
        ("er", "erDiagram\nStudent ||--o{ Enrollment : enrolls", "Enrollment"),
        ("journey", "journey\nsection Buy\nChoose: 5: User", "Choose"),
        ("timeline", "timeline\n2026: Release", "Release"),
        ("kanban", "kanban\n  todo[Todo]\n    task[Task]", "Task"),
        ("mindmap", "mindmap\n  root((Root))\n    Child", "Root"),
        ("gantt", "gantt\ndateFormat YYYY-MM-DD\nWork: 2026-01-01, 7d", "Work"),
        ("git", 'gitGraph\ncommit id: "Initial"', "Initial"),
        ("packet", 'packet-beta\n0-31: "Field"', "Field"),
        ("block", "block-beta\na[Block]", "Block"),
        ("architecture", "architecture-beta\nservice api(server)[API]", "API"),
        ("sankey", "sankey-beta\nSource,Target,10", "Source"),
        ("requirement", "requirementDiagram\nrequirement Req {\nid: R1\n}", "R1"),
        ("c4", 'C4Context\nSystem(sys, "System")', "System"),
    ],
)
async def test_supported_mermaid_fences_become_virtualized_diagram_blocks(
    kind: str,
    body: str,
    expected: str,
) -> None:
    app = _DiagramMarkdownApp(f"```mermaid\n{body}\n```")
    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        assert [block.block_type for block in markdown._blocks] == ["diagram"]
        diagram = markdown._blocks[0].diagram
        assert diagram is not None
        assert diagram.kind.value == kind
        assert expected in "\n".join(diagram.rows)


async def test_diagram_selection_copies_rendered_preview_while_dialog_owns_source_copy() -> None:
    app = _DiagramMarkdownApp(_FLOWCHART)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        selected = markdown.get_selection(SELECT_ALL)

        assert selected is not None
        text, separator = selected
        assert separator == "\n"
        assert "Request" in text
        assert "Open full diagram" in text
        assert "flowchart LR" not in text


@pytest.mark.parametrize(
    "source",
    [
        "```mermaid\nflowchart LR\nA --> B",
        "~~~mermaid\nflowchart LR\nA --> B",
    ],
)
async def test_streaming_open_mermaid_fence_stays_code(source: str) -> None:
    app = _DiagramMarkdownApp(source)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        assert [block.block_type for block in markdown._blocks] == ["fence"]
        assert markdown._blocks[0].diagram is None


async def test_custom_fence_rule_fails_closed_for_streaming_mermaid() -> None:
    def parser_factory() -> MarkdownIt:
        parser = MarkdownIt("commonmark")

        def proxied_fence(*args, **kwargs):
            return markdown_it_fence(*args, **kwargs)

        parser.block.ruler.at("fence", proxied_fence)
        return parser

    app = _DiagramMarkdownApp("```mermaid\nflowchart LR\nA --> B", parser_factory=parser_factory)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        assert [block.block_type for block in markdown._blocks] == ["fence"]
        assert markdown._blocks[0].diagram is None


@pytest.mark.parametrize(
    "body",
    [
        "flowchart LR\nA -->",
        "flowchart ZZ\nA --> B",
        "unknownDiagram\n    root((Unsupported))",
        "xychart\ny-axis 0 --> 10\nline [20, 30]",
        "xychart horizontal\ny-axis 0 --> 10\nbar [5, 8]\nline [-10, 20]",
    ],
)
async def test_mermaid_render_error_falls_back_to_virtualized_source_fence(body: str) -> None:
    source = f"```mermaid\n{body}\n```"
    app = _DiagramMarkdownApp(source)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        assert [block.block_type for block in markdown._blocks] == ["fence"]
        block = markdown._blocks[0]
        assert block.diagram is None
        assert block.code_language == "mermaid"
        assert body in block.content.plain
        assert not markdown._diagram_action_lines


async def test_mermaid_warning_still_renders_the_diagram() -> None:
    source = """```mermaid
flowchart LR
A[First]
A[Second]
```"""
    app = _DiagramMarkdownApp(source)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        assert [block.block_type for block in markdown._blocks] == ["diagram"]
        diagram = markdown._blocks[0].diagram
        assert diagram is not None
        assert [diagnostic.severity for diagnostic in diagram.diagnostics] == [DiagnosticSeverity.WARNING]
        assert "Second" in "\n".join(diagram.rows)


async def test_ignored_flowchart_decorations_keep_the_structural_diagram_visible() -> None:
    source = """```mermaid
flowchart LR
classDef active fill:#f00
subgraph Runtime
A[Decode]:::active --> B[Execute]
end
click A "https://example.com"
```"""
    app = _DiagramMarkdownApp(source)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        assert [block.block_type for block in markdown._blocks] == ["diagram"]
        diagram = markdown._blocks[0].diagram
        assert diagram is not None
        assert {diagnostic.severity for diagnostic in diagram.diagnostics} == {DiagnosticSeverity.WARNING}
        assert "Decode" in "\n".join(diagram.rows)
        assert "Execute" in "\n".join(diagram.rows)


async def test_flattened_sequence_fragments_keep_messages_visible() -> None:
    source = """```mermaid
sequenceDiagram
A->>+B: request
alt success
B-->>-A: response
else failure
B--xA: error
end
```"""
    app = _DiagramMarkdownApp(source)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        assert [block.block_type for block in markdown._blocks] == ["diagram"]
        diagram = markdown._blocks[0].diagram
        assert diagram is not None
        assert {diagnostic.severity for diagnostic in diagram.diagnostics} == {DiagnosticSeverity.WARNING}
        rendered = "\n".join(diagram.rows)
        assert "request" in rendered
        assert "response" in rendered
        assert "error" in rendered


async def test_frontmatter_metadata_and_notes_keep_the_diagram_visible() -> None:
    source = """```mermaid
---
title: State with note
---
stateDiagram-v2
accTitle: State lifecycle
state fork_state <<fork>>
Ready --> fork_state
fork_state --> Done
note right of Done : visible note
```
"""
    app = _DiagramMarkdownApp(source)
    async with app.run_test(size=(70, 28)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        assert [block.block_type for block in markdown._blocks] == ["diagram"]
        diagram = markdown._blocks[0].diagram
        assert diagram is not None
        rendered = "\n".join(diagram.rows)
        assert "━" * 9 in rendered
        assert "📝 visible note" in rendered
        assert not diagram.diagnostics


async def test_sequence_notes_and_links_keep_the_diagram_visible_without_link_actions() -> None:
    source = """```mermaid
sequenceDiagram
autonumber 2 0.5
participant A
participant B
Note over A,B: shared context
link A: docs @ https://example.com
A->>B: request
```
"""
    app = _DiagramMarkdownApp(source)
    async with app.run_test(size=(70, 28)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        assert [block.block_type for block in markdown._blocks] == ["diagram"]
        diagram = markdown._blocks[0].diagram
        assert diagram is not None
        rendered = "\n".join(diagram.rows)
        assert rendered.count("📝 shared context") == 2
        assert "2. request" in rendered
        assert "https://example.com" not in rendered
        assert {diagnostic.severity for diagnostic in diagram.diagnostics} == {DiagnosticSeverity.WARNING}


async def test_diagram_open_request_carries_compiled_source() -> None:
    app = _DiagramMarkdownApp(_FLOWCHART)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        action_line = next(iter(markdown._diagram_action_lines))
        await pilot.click(VirtualizedMarkdown, offset=(markdown.size.width - 3, action_line))
        await pilot.pause()

        assert app.opened is not None
        assert app.opened.source.startswith("flowchart LR")


async def test_diagram_action_uses_hand_pointer_only_over_its_text() -> None:
    app = _DiagramMarkdownApp(_FLOWCHART)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)
        action_line = next(iter(markdown._diagram_action_lines))

        await pilot.hover(VirtualizedMarkdown, offset=(markdown.size.width - 3, action_line))
        assert markdown.styles.pointer == "pointer"

        await pilot.hover(VirtualizedMarkdown, offset=(1, action_line))
        assert markdown.styles.pointer == "default"


async def test_scrolled_diagram_action_hit_test_uses_virtual_line() -> None:
    source = "```mermaid\nflowchart TB\n" + " --> ".join(f"N{i}" for i in range(12)) + "\n```"
    app = _DiagramMarkdownApp(source)
    async with app.run_test(size=(60, 8)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)
        action_line = next(iter(markdown._diagram_action_lines))

        markdown.scroll_to(y=action_line, animate=False, force=True, immediate=True)
        await pilot.pause()
        viewport_y = action_line - round(markdown.scroll_offset.y)
        assert markdown.scroll_offset.y > 0
        assert 0 <= viewport_y < markdown.size.height
        block = markdown._blocks[0]
        content_width = (
            markdown._width_at_last_layout
            - block.indent
            - block.padding_left
            - block.padding_right
            - len(block.border_left)
            - 1
        )
        action_x = block.indent + block.padding_left + len(block.border_left) + content_width - 1

        assert markdown._diagram_action_at(Offset(action_x, viewport_y)) == 0
        await pilot.click(VirtualizedMarkdown, offset=(action_x, viewport_y))
        await pilot.pause()
        assert app.opened is not None


async def test_diagram_action_hitbox_keeps_rendered_locale_width(monkeypatch: pytest.MonkeyPatch) -> None:
    from chrys.app.tui.widgets.markdown import widget as widget_module

    localizer = Localizer("en")
    monkeypatch.setattr(widget_module, "widget_localizer", lambda _widget: localizer)
    app = _DiagramMarkdownApp(_FLOWCHART)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)
        action_line = next(iter(markdown._diagram_action_lines))
        rendered = markdown.render_line(action_line)
        action_width = markdown._diagram_action_widths[action_line]
        assert rendered.text.rstrip().endswith("Open full diagram")

        assert localizer.switch_locale("zh-Hans") is None
        block = markdown._blocks[0]
        content_width = (
            markdown._width_at_last_layout
            - block.indent
            - block.padding_left
            - block.padding_right
            - len(block.border_left)
            - 1
        )
        action_start = block.indent + block.padding_left + len(block.border_left) + content_width - action_width

        assert markdown._diagram_action_at(Offset(action_start, action_line)) == 0
        await pilot.click(VirtualizedMarkdown, offset=(action_start, action_line))
        await pilot.pause()
        assert app.opened is not None


async def test_streamed_reparse_reuses_closed_diagram_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    from chrys.app.tui.widgets.markdown import widget as widget_module

    calls = 0
    original = widget_module.compile_mermaid

    def counted_compile(
        source: str,
        *,
        render_message: Callable[[MessageRef], str] | None = None,
    ):
        nonlocal calls
        calls += 1
        return original(source, render_message=render_message)

    monkeypatch.setattr(widget_module, "compile_mermaid", counted_compile)
    app = _DiagramMarkdownApp(_FLOWCHART)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)
        await markdown.update(_FLOWCHART)
        await markdown.append("\nTrailing paragraph.")

        assert calls == 1


async def test_diagram_action_is_bottom_right_without_dimensions_or_full_width_underline() -> None:
    source = "```mermaid\nflowchart LR\n" + " --> ".join(f"N{i}[Node {i}]" for i in range(8)) + "\n```"
    app = _DiagramMarkdownApp(source)
    async with app.run_test(size=(32, 20)) as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)
        layout = markdown._diagram_layouts[0]

        assert layout.preview_height > 0
        assert markdown._diagram_action_lines
        action_line = next(iter(markdown._diagram_action_lines))
        strip = markdown.render_line(action_line)
        assert strip.text.rstrip().endswith("Open full diagram")
        assert "flowchart" not in strip.text
        assert "\N{MULTIPLICATION SIGN}" not in strip.text

        underlined = [segment for segment in strip._segments if segment.style is not None and segment.style.underline]
        assert [segment.text for segment in underlined] == ["Open full diagram"]
        action_index = strip._segments.index(underlined[0])
        action_style = underlined[0].style
        background_style = strip._segments[action_index - 1].style
        assert action_style is not None
        assert background_style is not None
        assert action_style.bgcolor == background_style.bgcolor
