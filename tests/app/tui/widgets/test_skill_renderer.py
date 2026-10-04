# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for skill tool renderers."""

from __future__ import annotations

import sys

import pytest
from textual.app import App, ComposeResult

from chrys.app.tui.widgets.chat import tool_renderers
from chrys.app.tui.widgets.chat.renderers.skill import SkillToolCall
from chrys.app.tui.widgets.chat.tool_call import BaseToolCard, ToolCardHeader, ToolGroup
from chrys.app.tui.widgets.chat.tool_renderers import create_tool_widget
from chrys.foundation.tool_result_metadata import PARTIAL_OUTPUT_LABEL, TOOL_FAILED_METADATA_KEY

_RENDERER_MODULES = (
    "chrys.app.tui.widgets.chat.renderers.ask_user",
    "chrys.app.tui.widgets.chat.renderers.execute",
    "chrys.app.tui.widgets.chat.renderers.file_edit",
    "chrys.app.tui.widgets.chat.renderers.read_file",
    "chrys.app.tui.widgets.chat.renderers.search",
    "chrys.app.tui.widgets.chat.renderers.skill",
    "chrys.app.tui.widgets.chat.renderers.sleep",
    "chrys.app.tui.widgets.chat.renderers.sub_agent",
)
_MISSING = object()


class _SkillToolApp(App):
    def __init__(self, tool_name: str, args: dict) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._args = args

    def compose(self) -> ComposeResult:
        yield SkillToolCall("c1", self._tool_name, args=self._args)


class _NoSplitlinesStr(str):
    """Sentinel for large output where preview code must avoid full splitting."""

    def splitlines(self, keepends: bool = False) -> list[str]:
        raise AssertionError("skill renderer split the full output")

    def count(self, sub: str, start: int | None = None, end: int | None = None) -> int:
        if sub == "\n" and (start is None or end is None or end > 64 * 1024):
            raise AssertionError("skill renderer counted the full output")
        return super().count(sub, 0 if start is None else start, len(self) if end is None else end)


def test_factory_routes_skill_tool_names_to_skill_renderer() -> None:
    for name in ("load_skill", "read_skill_resource", "run_skill_script"):
        widget = create_tool_widget("c1", name, "", args={"skill_name": "docs"})
        assert isinstance(widget, SkillToolCall)


def test_skill_renderer_uses_base_tool_card_contract() -> None:
    tool = SkillToolCall("c1", "load_skill", args={"skill_name": "docs"})

    assert isinstance(tool, BaseToolCard)
    assert tool.call_id == "c1"
    assert tool.tool_name == "load_skill"
    assert tool.status == "running"
    assert tool.result_text == ""
    assert tool.duration_ms == 0
    assert tool.args == {"skill_name": "docs"}


def test_lazy_loader_imports_skill_renderer_module() -> None:
    parent_module = sys.modules["chrys.app.tui.widgets.chat.renderers"]
    parent_attrs = vars(parent_module)
    saved_parent_attrs = {
        module_name.rsplit(".", 1)[1]: parent_attrs.get(module_name.rsplit(".", 1)[1], _MISSING)
        for module_name in _RENDERER_MODULES
    }
    saved_registry = dict(tool_renderers._REGISTRY)
    saved_kind_registry = dict(tool_renderers._KIND_REGISTRY)
    saved_loaded = tool_renderers._loaded
    saved_modules = {module_name: sys.modules.pop(module_name, _MISSING) for module_name in _RENDERER_MODULES}
    for attr in saved_parent_attrs:
        parent_attrs.pop(attr, None)
    try:
        tool_renderers._REGISTRY.clear()
        tool_renderers._KIND_REGISTRY.clear()
        tool_renderers._loaded = False

        widget = create_tool_widget("c1", "load_skill", "", args={"skill_name": "docs"})

        assert widget.__class__.__name__ == "SkillToolCall"
    finally:
        for module_name, module in saved_modules.items():
            if module is _MISSING:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = module
        for attr, value in saved_parent_attrs.items():
            if value is _MISSING:
                parent_attrs.pop(attr, None)
            else:
                parent_attrs[attr] = value
        tool_renderers._REGISTRY.clear()
        tool_renderers._REGISTRY.update(saved_registry)
        tool_renderers._KIND_REGISTRY.clear()
        tool_renderers._KIND_REGISTRY.update(saved_kind_registry)
        tool_renderers._loaded = saved_loaded


def test_load_skill_result_extracts_metadata_and_instruction_preview() -> None:
    result = """\
---
name: docs
description: Documentation workflow
---

Use this skill to write docs.

<skill_dir>/repo/.agents/skills/docs</skill_dir>
<resources>
  <resource name="references/style.md" description="Style guide"/>
</resources>
<scripts>
  <script name="scripts/build.py"/>
</scripts>
"""
    tool = SkillToolCall("c1", "load_skill", args={"skill_name": "docs"})

    display, subtitle, is_error = tool._format_result(result)

    assert not is_error
    assert subtitle == "1 resource, 1 script"
    plain = display.plain
    assert "dir: /repo/.agents/skills/docs" in plain
    assert "resources: references/style.md" in plain
    assert "scripts: scripts/build.py" in plain
    assert "Use this skill to write docs." in plain
    assert "name: docs" not in plain


def test_load_skill_metadata_ignores_instruction_examples() -> None:
    result = """\
<name>docs</name>
<description>Documentation workflow</description>

<instructions>
Use <script name="example.py"/> in examples only.
</instructions>

<scripts>
  <script name="scripts/build.py"/>
</scripts>
"""
    tool = SkillToolCall("c1", "load_skill", args={"skill_name": "docs"})

    display, subtitle, is_error = tool._format_result(result)

    assert not is_error
    assert subtitle == "1 script"
    assert display.plain.startswith("scripts: scripts/build.py\n\n")
    assert "example.py" in display.plain
    assert "scripts/build.py" in display.plain


def test_load_skill_result_ignores_self_closing_empty_metadata_blocks() -> None:
    result = """\
<name>docs</name>
<description>Documentation workflow</description>

<instructions>
Use docs.
</instructions>

<resources />
<scripts />
"""
    tool = SkillToolCall("c1", "load_skill", args={"skill_name": "docs"})

    display, subtitle, is_error = tool._format_result(result)

    assert not is_error
    assert subtitle == "loaded"
    assert "Use docs." in display.plain
    assert "<resources />" not in display.plain
    assert "<scripts />" not in display.plain
    assert "resources:" not in display.plain
    assert "scripts:" not in display.plain


def test_code_skill_metadata_ignores_description_examples() -> None:
    result = """\
<name>docs</name>
<description>Example <resource name="description-only.md"/> syntax.</description>

<instructions>
Use docs.
</instructions>

<resources>
  <resource name="references/style.md"/>
</resources>
"""
    tool = SkillToolCall("c1", "load_skill", args={"skill_name": "docs"})

    display, subtitle, is_error = tool._format_result(result)

    assert not is_error
    assert subtitle == "1 resource"
    assert "description-only.md" not in display.plain
    assert "references/style.md" in display.plain


def test_file_skill_metadata_uses_appended_skill_dir_block() -> None:
    result = """\
---
name: docs
description: Documentation workflow
---

Mention <skill_dir>/example/from/instructions</skill_dir> as literal syntax.

<skill_dir>/repo/.agents/skills/docs</skill_dir>
<resources>
  <resource name="references/style.md"/>
</resources>
"""
    tool = SkillToolCall("c1", "load_skill", args={"skill_name": "docs"})

    display, subtitle, is_error = tool._format_result(result)

    assert not is_error
    assert subtitle == "1 resource"
    assert "dir: /repo/.agents/skills/docs" in display.plain
    assert "Mention <skill_dir>/example/from/instructions</skill_dir>" in display.plain


def test_malformed_skill_metadata_degrades_to_instruction_preview() -> None:
    result = """\
---
name: broken
description: Has malformed metadata
---

Follow these instructions even when metadata below is malformed.

<skill_dir>/repo/.agents/skills/broken</skill_dir>
<resources>
  <resource description="missing name"/>
  <resource name="unterminated>
</resources>
<scripts>
  <script description="missing name"/>
</scripts>
"""
    tool = SkillToolCall("c1", "load_skill", args={"skill_name": "broken"})

    display, subtitle, is_error = tool._format_result(result)

    assert not is_error
    assert subtitle == "loaded"
    assert "dir: /repo/.agents/skills/broken" in display.plain
    assert "Follow these instructions" in display.plain


def test_escaped_resource_and_script_names_are_displayed_plainly() -> None:
    result = """\
<name>docs</name>
<description>Documentation workflow</description>

<instructions>
Use docs.
</instructions>

<resources>
  <resource name="references/style &amp; tone.md"/>
</resources>
<scripts>
  <script name="scripts/build &amp; check.py"/>
</scripts>
"""
    tool = SkillToolCall("c1", "load_skill", args={"skill_name": "docs"})

    display, subtitle, is_error = tool._format_result(result)

    assert not is_error
    assert subtitle == "1 resource, 1 script"
    assert "references/style & tone.md" in display.plain
    assert "scripts/build & check.py" in display.plain


def test_missing_skill_tool_args_still_mounts_cleanly() -> None:
    tool = SkillToolCall("c1", "read_skill_resource", args={})

    assert tool._label_suffix() == ""
    assert tool._panel_title() == "resource"


def test_script_result_detects_nonzero_exit_code() -> None:
    tool = SkillToolCall("c1", "run_skill_script", args={"skill_name": "docs", "script_name": "scripts/build.py"})

    display, subtitle, is_error = tool._format_result("failed\n[stderr]\nboom\n[exit_code: 42]")

    assert is_error
    assert subtitle == "exit 42"
    assert "[exit_code" not in display.plain
    assert "boom" in display.plain


def test_script_result_exit_suffix_removal_strips_only_trailing_newlines() -> None:
    tool = SkillToolCall("c1", "run_skill_script", args={"skill_name": "docs", "script_name": "scripts/build.py"})

    display, subtitle, is_error = tool._format_result("  indented\nbody\n\n[exit_code: 0]")

    assert not is_error
    assert subtitle == "exit 0"
    assert display.plain == "  indented\nbody"


def test_timed_out_script_result_previews_what_the_script_printed() -> None:
    tool = SkillToolCall("c1", "run_skill_script", args={"skill_name": "docs", "script_name": "scripts/hang.py"})
    printed = "\n".join(f"line {i}" for i in range(400))

    display, subtitle, is_error = tool._format_result(
        f"Error: Script 'hang.py' timed out after 30s.\n{PARTIAL_OUTPUT_LABEL}\n{printed}"
    )

    assert is_error
    assert subtitle == "error"
    lines = "\n".join(f"line {i}" for i in range(8))
    assert display.plain == f"Error: Script 'hang.py' timed out after 30s.\n{lines}\n..."


def test_script_error_without_output_is_shown_whole() -> None:
    tool = SkillToolCall("c1", "run_skill_script", args={"skill_name": "docs", "script_name": "scripts/x.py"})
    error = "Error: Script 'x.py' failed to start.\nfirst detail\nsecond detail"

    display, _subtitle, is_error = tool._format_result(error)

    assert is_error
    assert display.plain == error


def test_large_resource_preview_does_not_split_full_output() -> None:
    result = _NoSplitlinesStr("heading\n" + ("x" * 1_000_000))
    tool = SkillToolCall("c1", "read_skill_resource", args={"skill_name": "docs", "resource_name": "large.md"})

    display, subtitle, is_error = tool._format_result(result)

    assert not is_error
    assert subtitle == "2+ lines, 1000008 chars"
    assert display.plain.startswith("heading\n")
    assert len(display.plain) < 400
    assert display.plain.endswith("...")


@pytest.mark.asyncio
async def test_load_skill_mounts_and_shows_copy_button_after_completion() -> None:
    async with _SkillToolApp("load_skill", {"skill_name": "docs"}).run_test() as pilot:
        tool = pilot.app.query_one(SkillToolCall)
        header = tool.query_one(ToolCardHeader)
        assert header.actions_visible is False

        tool.set_complete("<instructions>\nUse docs.\n</instructions>", 25)
        await pilot.pause()

        assert tool.status == "complete"
        assert tool.query_one("#skill-panel").border_subtitle == "loaded"
        assert header.actions_visible is True
        assert "Use docs." in tool.query_one("#skill-panel").render().plain


@pytest.mark.asyncio
async def test_skill_renderer_records_rejected_completion_approval() -> None:
    async with _SkillToolApp("load_skill", {"skill_name": "docs"}).run_test() as pilot:
        tool = pilot.app.query_one(SkillToolCall)

        tool.set_complete("Error: rejected", duration_ms=10, approval="user_rejected")
        await pilot.pause()

        assert tool.approval == "user_rejected"
        assert tool.has_class("-rejected")
        assert tool.query_one("#skill-panel").border_subtitle == "Rejected"


@pytest.mark.asyncio
async def test_skill_renderer_uses_structured_failure_metadata() -> None:
    async with _SkillToolApp("load_skill", {"skill_name": "docs"}).run_test() as pilot:
        tool = pilot.app.query_one(SkillToolCall)

        tool.set_complete(
            "<instructions>\nUse docs.\n</instructions>",
            duration_ms=10,
            metadata={TOOL_FAILED_METADATA_KEY: True},
        )
        await pilot.pause()

        assert tool.status == "error"
        assert tool.has_class("-error")
        assert not tool.has_class("-success")
        assert tool.query_one("#skill-panel").border_subtitle == "loaded"


@pytest.mark.asyncio
async def test_skill_renderer_works_inside_tool_group() -> None:
    class GroupApp(App):
        def compose(self) -> ComposeResult:
            yield ToolGroup()

    async with GroupApp().run_test() as pilot:
        group = pilot.app.query_one(ToolGroup)
        await group.add_tool("c1", "read_skill_resource", "", args={"skill_name": "docs", "resource_name": "style.md"})
        await pilot.pause()

        tool = pilot.app.query_one(SkillToolCall)
        group.complete_tool("c1", "Prefer active voice.", 5)
        await pilot.pause()

        assert tool.status == "complete"
        assert tool.query_one("#skill-panel").border_subtitle == "1 line, 20 chars"
        assert "Prefer active voice." in tool.query_one("#skill-panel").render().plain
