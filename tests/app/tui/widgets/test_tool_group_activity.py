# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Live tool-group activity header and default-collapse behavior."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chat.tool_call import ToolGroup, ToolGroupTitle, _tool_activity_reference
from chrys.app.tui.widgets.loading import ChrysLoadingIndicator
from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.tool_kinds import (
    KIND_ASK_USER,
    KIND_FILESYSTEM_READ,
    KIND_FILESYSTEM_WRITE,
    KIND_SEARCH,
    KIND_SHELL,
    KIND_SKILL,
    KIND_SUB_AGENT,
)


class _PanelApp(App):
    def __init__(self, *, expanded: bool) -> None:
        self.expanded = expanded
        super().__init__()

    def compose(self) -> ComposeResult:
        yield ChatPanel(tool_groups_expanded=lambda: self.expanded)


def _title(group: ToolGroup) -> str:
    return group.query_one(ToolGroupTitle).render().plain


@pytest.mark.asyncio
async def test_activity_title_renders_a_real_ellipsis_at_narrow_width() -> None:
    async with _PanelApp(expanded=False).run_test(size=(28, 12)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_tool_start(
            "shell",
            "zsh",
            KIND_SHELL,
            args={"reason": "Running an intentionally long focused validation command"},
        )
        await pilot.pause()

        title = panel.query_one(ToolGroupTitle)
        rendered = title.render()
        assert rendered.plain.endswith("…")
        assert rendered.cell_len <= title.content_size.width


@pytest.mark.asyncio
async def test_live_group_activity_survives_manual_expand_and_falls_back_by_recency() -> None:
    async with _PanelApp(expanded=False).run_test(size=(100, 30)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_tool_start(
            "shell",
            "zsh",
            KIND_SHELL,
            args={"command": "pytest", "reason": "Running focused tests"},
        )
        await pilot.pause()

        group = panel.query_one(ToolGroup)
        indicator = group.query_one(ChrysLoadingIndicator)
        assert group.collapsed is True
        assert _title(group) == "Running focused tests"
        assert indicator.display is True
        assert list(group.query_one("#tg-header").children) == [indicator, group.query_one(ToolGroupTitle)]
        assert indicator.styles.color == group.query_one(ToolGroupTitle).styles.color

        await pilot.click(ToolGroupTitle)
        await pilot.pause()
        assert group.collapsed is False
        assert _title(group) == "Running focused tests"
        assert indicator.display is True

        await panel.add_tool_start(
            "read",
            "read_file",
            KIND_FILESYSTEM_READ,
            args={"path": "src/chrys/app.py"},
        )
        assert _title(group) == "Reading src/chrys/app.py"
        panel.update_tool_args("shell", {"command": "pytest", "reason": "Updated shell work"})
        assert _title(group) == "Reading src/chrys/app.py"

        await panel.add_tool_result("read", "read_file", "done")
        assert _title(group) == "Updated shell work"
        assert indicator.display is True

        await panel.add_tool_result("shell", "zsh", "done")
        assert _title(group) == "Updated shell work"
        assert indicator.display is True
        assert indicator._auto_refresh_timer._active.is_set() is True

        await panel.add_tool_start("write", "write_file", KIND_FILESYSTEM_WRITE, args={"path": "report.md"})
        assert _title(group) == "Writing report.md"
        assert indicator.display is True
        assert indicator._auto_refresh_timer._active.is_set() is True

        await panel.add_tool_result("write", "write_file", "done")
        assert _title(group) == "Writing report.md"
        assert indicator.display is True

        await panel.add_agent_message("Moving on", is_intermediate=True)
        assert _title(group).startswith("▶ Tools (3/3")
        assert indicator.display is False
        assert indicator._auto_refresh_timer._active.is_set() is False

        group.on_tool_group_title_clicked()
        await pilot.pause()
        assert group.collapsed is False
        assert _title(group).startswith("▼ Tools (3/3")


@pytest.mark.asyncio
async def test_bare_chat_panel_keeps_compatibility_default_expanded() -> None:
    class _BarePanelApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with _BarePanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_tool_start("read", "read_file", KIND_FILESYSTEM_READ, args={"path": "README.md"})
        assert panel.query_one(ToolGroup).collapsed is False


@pytest.mark.asyncio
async def test_default_getter_is_read_for_each_new_group_without_rewriting_existing_state() -> None:
    async with _PanelApp(expanded=False).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_tool_start("first", "read_file", KIND_FILESYSTEM_READ, args={"path": "one.txt"})
        first = panel.query_one(ToolGroup)
        assert first.collapsed is True

        panel._end_tool_group()
        pilot.app.expanded = True
        await panel.add_tool_start("second", "read_file", KIND_FILESYSTEM_READ, args={"path": "two.txt"})
        groups = list(panel.query(ToolGroup))

        assert groups == [first, panel._tool_groups_by_call_id["second"]]
        assert [group.collapsed for group in groups] == [True, False]


@pytest.mark.asyncio
async def test_shell_without_reason_hides_command_and_skill_activity_uses_safe_fields() -> None:
    async with _PanelApp(expanded=False).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_tool_start("shell", "zsh", KIND_SHELL, args={"command": "secret --token value"})
        group = panel.query_one(ToolGroup)

        assert _title(group) == "Running shell command"
        assert "secret" not in _title(group)

        await panel.add_tool_result("shell", "zsh", "done")
        await panel.add_tool_start(
            "skill",
            "run_skill_script",
            KIND_SKILL,
            args={"skill_name": "documents", "script_name": "render_docx.py"},
        )
        assert _title(group) == "Running skill script documents/render_docx.py"


@pytest.mark.asyncio
async def test_fold_target_ignores_locked_groups() -> None:
    class _FoldApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with _FoldApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        locked = ToolGroup()
        other = ToolGroup()
        other.collapsed = True
        await panel.mount(locked, other)
        locked.lock_collapse_for("ask")

        assert panel.toggle_fold_all() is False
        assert locked.collapsed is False
        assert other.collapsed is False


@pytest.mark.asyncio
async def test_paused_sub_agent_expands_and_locks_group_until_resume() -> None:
    async with _PanelApp(expanded=False).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_tool_start("parent", "Explore", KIND_SUB_AGENT, args={"task": "inspect"})
        panel.link_sub_agent_invocation("parent", "invocation", "Explore Agent")
        group = panel.query_one(ToolGroup)

        assert group.collapsed is True
        assert group.collapse_locked is False

        panel.sub_agent_paused("invocation", "acp_transport", "connection closed", 2)
        await pilot.pause()

        assert group.collapsed is False
        assert group.collapse_locked is True
        group.collapsed = True
        assert group.collapsed is False

        panel.sub_agent_resumed_after_pause("invocation")
        assert group.collapsed is False
        assert group.collapse_locked is False
        group.collapsed = True
        assert group.collapsed is True


@pytest.mark.asyncio
async def test_sub_agent_inner_activity_updates_parent_header_and_retains_latest_state() -> None:
    async with _PanelApp(expanded=False).run_test(size=(100, 30)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_tool_start("parent", "Explore", KIND_SUB_AGENT, args={"task": "inspect"})
        panel.link_sub_agent_invocation("parent", "invocation", "Explore Agent")

        group = panel.query_one(ToolGroup)
        assert _title(group) == "Running Explore Agent"

        panel.add_sub_agent_message("Explore Agent", "invocation", "Let me inspect the notes.")

        assert _title(group) == "Running Explore Agent: Let me inspect the notes."

        await panel.add_sub_agent_tool_start(
            "Explore",
            "invocation",
            "read_file",
            {"path": "notes.txt", "content": "do not retain"},
            "inner",
            tool_kind=KIND_FILESYSTEM_READ,
        )

        indicator = group.query_one(ChrysLoadingIndicator)
        assert _title(group) == "Running Explore Agent: Reading notes.txt"
        assert group._tool_records["parent"].nested_activities["inner"].args == {"path": "notes.txt"}

        panel._end_tool_group()
        assert _title(group) == "Running Explore Agent: Reading notes.txt"
        assert indicator.display is True

        panel.complete_sub_agent_tool("Explore", "invocation", "inner", "done", 1)
        assert _title(group) == "Running Explore Agent: Reading notes.txt"
        assert group._tool_records["parent"].nested_activities == {}

        await panel.add_tool_result("parent", "Explore", "done")
        assert _title(group).startswith("▶ Tools (1/1")
        assert indicator.display is False


@pytest.mark.asyncio
async def test_system_retry_and_interrupt_are_tool_group_presentation_boundaries() -> None:
    async with _PanelApp(expanded=False).run_test(size=(100, 30)) as pilot:
        panel = pilot.app.query_one(ChatPanel)

        await panel.add_tool_start("system", "read_file", KIND_FILESYSTEM_READ, args={"path": "system.txt"})
        first = panel.query_one(ToolGroup)
        await panel.add_tool_result("system", "read_file", "done")
        assert _title(first) == "Reading system.txt"
        await panel.add_system("Environment changed")
        assert _title(first).startswith("▶ Tools (1/1")

        await panel.add_tool_start("retry", "read_file", KIND_FILESYSTEM_READ, args={"path": "retry.txt"})
        second = panel._tool_groups_by_call_id["retry"]
        await panel.add_tool_result("retry", "read_file", "done")
        assert _title(second) == "Reading retry.txt"
        await panel.add_retry("Trying again", 1, 2, 0)
        assert _title(second).startswith("▶ Tools (1/1")

        await panel.add_tool_start("interrupt", "read_file", KIND_FILESYSTEM_READ, args={"path": "active.txt"})
        third = panel._tool_groups_by_call_id["interrupt"]
        assert _title(third) == "Reading active.txt"
        await panel.add_interrupted()
        assert _title(third).startswith("▶ Tools (1/1")
        assert third.query_one(ChrysLoadingIndicator).display is False


@pytest.mark.asyncio
async def test_replay_group_has_no_live_indicator() -> None:
    class _ReplayGroupApp(App):
        def compose(self) -> ComposeResult:
            yield ToolGroup(live=False)

    async with _ReplayGroupApp().run_test() as pilot:
        group = pilot.app.query_one(ToolGroup)
        assert list(group.query(ChrysLoadingIndicator)) == []


def test_loading_indicator_pause_and_resume_are_safe_before_mount() -> None:
    indicator = ChrysLoadingIndicator()
    indicator.pause_animation()
    indicator.resume_animation()


@pytest.mark.parametrize(
    ("tool_name", "tool_kind", "args", "expected"),
    [
        ("edit_file", KIND_FILESYSTEM_WRITE, {"path": "notes.md"}, "Editing notes.md"),
        ("grep", KIND_SEARCH, {"pattern": "TODO"}, "Searching for TODO"),
        ("glob", KIND_SEARCH, {"pattern": "*.py"}, "Finding *.py"),
        ("load_skill", KIND_SKILL, {"skill_name": "documents"}, "Loading skill documents"),
        (
            "read_skill_resource",
            KIND_SKILL,
            {"skill_name": "documents", "resource_name": "guide.md"},
            "Reading skill resource documents/guide.md",
        ),
        ("ask_user", KIND_ASK_USER, {"question": "Continue?"}, "Waiting for your answer"),
        ("custom_tool", "mcp", {}, "Running custom_tool"),
    ],
)
def test_activity_copy_uses_kind_specific_safe_arguments(
    tool_name: str,
    tool_kind: str,
    args: dict[str, str],
    expected: str,
) -> None:
    activity = _tool_activity_reference(tool_name, tool_kind, args)

    assert (activity if isinstance(activity, str) else format_message(activity)) == expected
