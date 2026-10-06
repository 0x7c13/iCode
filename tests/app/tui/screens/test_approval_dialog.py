# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the approval dialog."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalGroup
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from chrys.app.tui.screens.dialogs.approval import ApprovalDialog
from chrys.app.tui.screens.dialogs.approval import body as approval_body_module
from chrys.app.tui.screens.dialogs.approval.body import (
    ApprovalBody,
    ApprovalBodyBuilder,
    create_approval_body,
)
from chrys.app.tui.theme import CHRYS_ANSI_THEME
from chrys.app.tui.widgets import StableAutoHeightScroll
from chrys.app.tui.widgets.loading import ChrysLoadingIndicator
from chrys.app.tui.widgets.text_area import EnhancedTextArea
from chrys.foundation.i18n import MessageRef
from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.tool_kinds import KIND_FILESYSTEM_WRITE, KIND_MCP, KIND_SUB_AGENT
from chrys.service.approval.judge import JudgeVerdict
from tests.support.paths import SRC_ROOT
from tests.support.waiting import wait_for

_CHRYS_CSS = SRC_ROOT / "chrys" / "app" / "tui" / "chrys.tcss"


@pytest.mark.asyncio
async def test_create_approval_body_rejects_invalid_builder_result(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_name = "invalid-approval-body-result"

    def invalid_builder(_tool_kind: str, _args: dict[str, Any], _workspace_cwd: str | None) -> object:
        return "not-an-approval-body"

    monkeypatch.setitem(
        approval_body_module._REGISTRY,
        tool_name,
        cast("ApprovalBodyBuilder", invalid_builder),
    )

    assert await create_approval_body(tool_name, KIND_MCP, {}) is None


def _normalized_eol(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


class _AnsiDialogHost(App):
    CSS_PATH = str(_CHRYS_CSS)

    def __init__(self) -> None:
        super().__init__()
        self.register_theme(CHRYS_ANSI_THEME)
        self.theme = "chrys-ansi"

    def compose(self) -> ComposeResult:
        yield Static("placeholder")


@pytest.mark.asyncio
async def test_approval_dialog_max_height_matches_dialog_standard() -> None:
    dialog = ApprovalDialog(caller_name="", tool_name="zsh")

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        container = dialog.query_one("#approval-container", VerticalGroup)
        inner = dialog.query_one("#approval-inner", StableAutoHeightScroll)
        assert str(container.styles.max_height) == "85h"
        assert inner.styles.padding.left == 1
        assert inner.styles.padding.right == 0
        assert inner.styles.scrollbar_gutter == "stable"
        assert inner.can_focus is False


@pytest.mark.asyncio
async def test_bridged_approval_header_shows_action_label_not_raw_title() -> None:
    """ACP-bridged requests render a friendly action header; the remote title
    is suppressed when it merely repeats an argument box."""
    command = 'which cloc scc tokei 2>/dev/null; echo "---"'
    dialog = ApprovalDialog(
        caller_name="claude_code",
        tool_name=f"acp:`{command}`",
        args={"command": command, "description": "Check available tools"},
        presentation_kind="shell",
    )

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        header = dialog.query_one("#approval-tool", Static)
        assert "Run command" in header.render().plain
        assert command not in header.render().plain
        assert not dialog.query("#approval-remote-title")


@pytest.mark.asyncio
async def test_bridged_approval_shows_novel_remote_title_and_unknown_kind_label() -> None:
    dialog = ApprovalDialog(
        caller_name="claude_code",
        tool_name="acp:Update workspace settings",
        args={"scope": "user"},
        presentation_kind="mystery-kind",
    )

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        header = dialog.query_one("#approval-tool", Static)
        assert "Remote tool" in header.render().plain
        title_line = dialog.query_one("#approval-remote-title", Static)
        assert "Update workspace settings" in title_line.render().plain


@pytest.mark.asyncio
async def test_bridged_approval_suppresses_title_echoing_the_header_chip() -> None:
    """A remote title that merely repeats the chip label adds nothing."""
    dialog = ApprovalDialog(
        caller_name="chrys_code",
        tool_name="acp:run command",
        args={"other": "value"},
        presentation_kind="shell",
    )

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        assert "Run command" in dialog.query_one("#approval-tool", Static).render().plain
        assert not dialog.query("#approval-remote-title")


@pytest.mark.asyncio
async def test_bridged_approval_shows_description_arg_as_detail_line() -> None:
    """Remote agents carry their rationale in a "description" arg; it renders
    under the header like the native shell "reason" line, not as an arg box."""
    command = "which cloc scc tokei"
    dialog = ApprovalDialog(
        caller_name="claude_code",
        tool_name=f"acp:`{command}`",
        args={"command": command, "description": "Check available LOC tools"},
        presentation_kind="shell",
    )

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        detail = dialog.query_one("#approval-detail", Static)
        assert "Check available LOC tools" in detail.render().plain
        box_labels = [str(box.border_title) for box in dialog.query(".approval-arg-box")]
        # The description was consumed by the detail line; command keeps its box.
        assert box_labels == ["command"]


@pytest.mark.asyncio
async def test_bridged_filesystem_approval_uses_presentation_kind_for_detail() -> None:
    """Bridged requests publish tool_kind="" with the chrys kind in
    presentation_kind — the kind-specific detail keys must still apply, so a
    remote read shows its path under the header instead of as an arg box."""
    dialog = ApprovalDialog(
        caller_name="claude_code",
        tool_name="acp:a.py",
        args={"path": "a.py"},
        presentation_kind="filesystem.read",
    )

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        detail = dialog.query_one("#approval-detail", Static)
        assert detail.render().plain == "a.py"
        # Path consumed by the detail line: no arg boxes remain, and the
        # remote title (the same path) is suppressed as redundant.
        assert not list(dialog.query(".approval-arg-box"))
        assert not list(dialog.query("#approval-remote-title"))


def test_detail_line_prefers_kind_specific_reason_over_description() -> None:
    dialog = ApprovalDialog(
        caller_name="",
        tool_name="zsh",
        tool_kind="shell",
        args={"command": "ls", "reason": "List files", "description": "generic"},
    )
    assert dialog._detail == "List files"
    assert dialog._detail_key == "reason"


@pytest.mark.asyncio
async def test_approval_dialog_scrollbar_does_not_change_argument_width() -> None:
    dialog = ApprovalDialog(
        caller_name="",
        tool_name="shell",
        args={"command": ("word " * 150).strip()},
    )

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        inner = dialog.query_one("#approval-inner", StableAutoHeightScroll)
        argument = dialog.query_one(".approval-arg-box", Static)
        await wait_for(
            lambda: argument.size.width > 0 and not inner.show_vertical_scrollbar,
            pilot=pilot,
            description="approval body to settle without a scrollbar",
        )
        assert inner.show_vertical_scrollbar is False
        width_without_scrollbar = argument.size.width

        await pilot.resize_terminal(100, 25)
        await wait_for(
            lambda: inner.show_vertical_scrollbar,
            pilot=pilot,
            description="approval body to show its vertical scrollbar",
        )

        assert inner.show_vertical_scrollbar is True
        assert argument.size.width == width_without_scrollbar


@pytest.mark.asyncio
async def test_approval_dialog_reason_allows_five_content_lines() -> None:
    dialog = ApprovalDialog(caller_name="", tool_name="zsh")

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        input_area = dialog.query_one("#approval-reason", EnhancedTextArea)
        assert str(input_area.styles.max_height) == "7"


@pytest.mark.asyncio
async def test_approval_dialog_buttons_show_keyboard_shortcuts() -> None:
    dialog = ApprovalDialog(caller_name="", tool_name="zsh")

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        assert dialog.query_one("#approval-yes", Button).label.plain == "Approve (Y)"
        assert dialog.query_one("#approval-no", Button).label.plain == "Decline (N)"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("y", (True, "", None)),
        ("Y", (True, "", None)),
        ("n", (False, "", None)),
        ("N", (False, "", None)),
    ],
)
@pytest.mark.asyncio
async def test_approval_dialog_y_n_shortcuts_submit_decision(
    key: str,
    expected: tuple[bool, str, dict[str, object] | None],
) -> None:
    dialog = ApprovalDialog(caller_name="", tool_name="zsh")
    results: list[tuple[bool, str, dict[str, object] | None]] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        await pilot.press(key)
        await pilot.pause()

    assert results == [expected]


@pytest.mark.asyncio
async def test_approval_dialog_y_n_in_reason_inserts_text() -> None:
    dialog = ApprovalDialog(caller_name="", tool_name="zsh")
    results: list[tuple[bool, str, dict[str, object] | None]] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        input_area = dialog.query_one("#approval-reason", EnhancedTextArea)
        input_area.focus()
        await wait_for(lambda: input_area.has_focus, pilot=pilot, description="control focus before interaction")
        await pilot.pause()

        await pilot.press("y", "n")
        await pilot.pause()

        assert input_area.text == "yn"
        assert results == []


@pytest.mark.asyncio
async def test_approval_dialog_enter_does_not_press_focused_buttons() -> None:
    dialog = ApprovalDialog(caller_name="", tool_name="zsh")
    results: list[tuple[bool, str, dict[str, object] | None]] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        approve = dialog.query_one("#approval-yes", Button)
        decline = dialog.query_one("#approval-no", Button)
        assert approve.has_focus

        await pilot.press("enter")
        await pilot.pause()
        assert results == []

        await pilot.press("right")
        await pilot.pause()
        assert decline.has_focus

        await pilot.press("enter")
        await pilot.pause()
        assert results == []

        await pilot.press("n")
        await pilot.pause()

    assert results == [(False, "", None)]


@pytest.mark.asyncio
async def test_approval_dialog_decline_returns_optional_reason() -> None:
    """Decline should include the optional reason text in the modal result."""
    dialog = ApprovalDialog(
        caller_name="",
        tool_name="zsh",
        tool_kind="shell",
        args={"command": "rm generated.tmp"},
    )
    results: list[tuple[bool, str, dict[str, object] | None]] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        assert dialog.query_one("#approval-yes", Button).has_focus

        input_area = dialog.query_one("#approval-reason", EnhancedTextArea)
        input_area.insert("Use a narrower command")
        input_area.focus()
        await wait_for(lambda: input_area.has_focus, pilot=pilot, description="control focus before interaction")
        await pilot.press("ctrl+a")
        await pilot.pause()
        assert input_area.selected_text == "Use a narrower command"

        input_area.clear()
        input_area.insert("Use a narrower command")
        await pilot.press("enter")
        await pilot.pause()
        assert input_area.text == "Use a narrower command\n"
        assert results == []

        dialog.query_one("#approval-no", Button).press()
        await pilot.pause()

    assert results == [(False, "Use a narrower command", None)]


@pytest.mark.asyncio
async def test_approval_dialog_reason_placeholder_is_dim_in_chrys_ansi() -> None:
    dialog = ApprovalDialog(caller_name="", tool_name="zsh")

    app = _AnsiDialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        input_area = dialog.query_one("#approval-reason", EnhancedTextArea)
        assert input_area.placeholder == "Reason (optional, sent to agent on Decline)"
        placeholder_style = input_area.get_visual_style("text-area--placeholder").rich_style
        assert placeholder_style.color is not None
        assert placeholder_style.color.number == 8


@pytest.mark.asyncio
async def test_approval_dialog_judge_uses_chrys_loading_indicator() -> None:
    dialog = ApprovalDialog(caller_name="", tool_name="zsh", judging=True)

    app = _AnsiDialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        assert isinstance(dialog.query_one("#approval-judge-loading"), ChrysLoadingIndicator)


@pytest.mark.asyncio
async def test_approval_dialog_separator_is_dim_in_chrys_ansi() -> None:
    dialog = ApprovalDialog(caller_name="", tool_name="zsh", judging=True)

    app = _AnsiDialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        separator_style = dialog.query_one("#approval-separator", Static).rich_style
        assert separator_style.color is not None
        assert separator_style.color.number == 8


@pytest.mark.asyncio
async def test_approval_dialog_approve_returns_empty_reason() -> None:
    """Approval without a decline reason returns an empty reason."""
    dialog = ApprovalDialog(caller_name="", tool_name="read_file", judging=True)
    results: list[tuple[bool, str, dict[str, object] | None]] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        dialog.query_one("#approval-yes", Button).press()
        await pilot.pause()

    assert results == [(True, "", None)]


@pytest.mark.asyncio
async def test_approval_dialog_custom_body_hides_selected_args() -> None:
    dialog = ApprovalDialog(
        caller_name="",
        tool_name="write_file",
        tool_kind="filesystem.write",
        args={"path": "out.txt", "content": "secret\n"},
        approval_body=ApprovalBody(
            widgets=[Static("diff body", id="custom-approval-body")],
            hidden_arg_keys=frozenset({"content"}),
        ),
    )

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        dialog.query_one("#custom-approval-body", Static)
        assert len(dialog.query("#approval-args")) == 0


@pytest.mark.asyncio
async def test_approval_diff_preview_uses_dialog_scroller() -> None:
    from chrys.app.tui.screens.dialogs.approval.bodies.file_edit import ApprovalDiffPreview
    from chrys.app.tui.widgets.diff_view import DiffView
    from chrys.app.tui.widgets.diff_view.code import CodeColumn

    dialog = ApprovalDialog(
        caller_name="",
        tool_name="edit_file",
        tool_kind="filesystem.write",
        args={"path": "example.py", "old_string": "a", "new_string": "b"},
        approval_body=ApprovalBody(
            widgets=[
                ApprovalDiffPreview(
                    "example.py",
                    "before\n" * 30,
                    "after\n" * 30,
                    title="Planned Diff",
                )
            ],
            hidden_arg_keys=frozenset({"old_string", "new_string"}),
        ),
    )

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await wait_for(
            lambda: len(dialog.query(CodeColumn)),
            timeout=20,
            pilot=pilot,
            description="approval diff contents mounted",
        )

        diff = dialog.query_one(DiffView)
        assert diff.auto_height is True
        assert diff.max_display_lines is None
        assert diff.show_scrollbars is True
        code_column = dialog.query_one(CodeColumn)
        assert code_column.styles.overflow_y == "hidden"
        assert code_column.styles.scrollbar_size_vertical == 0


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("write_file", "copy target"),
        ("edit_file", "old target"),
    ],
)
@pytest.mark.asyncio
async def test_file_approval_dialog_right_click_copies_diff_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_name: str,
    expected: str,
) -> None:
    from textual.events import MouseDown
    from textual.geometry import Offset
    from textual.selection import Selection

    from chrys.app.tui.widgets.diff_view.code import CodeColumn

    copied: list[str] = []
    monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)

    if tool_name == "write_file":
        args = {"path": str(tmp_path / "created.txt"), "content": "copy target\nsecond line\n"}
    else:
        target = tmp_path / "existing.txt"
        target.write_text("old target\nsecond line\n", encoding="utf-8")
        args = {"path": str(target), "old_string": "old target", "new_string": "new target"}

    body = await create_approval_body(tool_name, KIND_FILESYSTEM_WRITE, args)
    assert body is not None
    dialog = ApprovalDialog(
        caller_name="",
        tool_name=tool_name,
        tool_kind=KIND_FILESYSTEM_WRITE,
        args=args,
        approval_body=body,
    )

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await wait_for(
            lambda: len(dialog.query(CodeColumn)),
            timeout=20,
            pilot=pilot,
            description="approval diff code column mounted",
        )

        code_column = dialog.query_one(CodeColumn)
        dialog.selections = {code_column: Selection(Offset(0, 0), Offset(len(expected), 0))}
        # Extraction needs the column's content laid out, not merely mounted;
        # poll the selection until it yields the text the click will copy.
        await wait_for(
            lambda: dialog.get_selected_text() == expected,
            pilot=pilot,
            description="selection extractable",
        )

        # Exercise the dialog's real screen-forwarding seam at a stable,
        # non-TextArea coordinate. The short write-file diff may be clipped
        # behind the Reason editor, so CodeColumn.region is not a reliable
        # pointer target even after its selectable content is ready.
        dialog._forward_event(
            MouseDown(
                None,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=3,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=0,
                screen_y=0,
            )
        )

        assert app.clipboard == expected
        assert copied == [expected]
        assert dialog.get_selected_text() is None


@pytest.mark.asyncio
async def test_write_file_approval_body_keeps_full_multiline_content(tmp_path: Path) -> None:
    from chrys.app.tui.screens.dialogs.approval.bodies.file_edit import ApprovalDiffPreview

    content = "\n".join(f"line {i}" for i in range(1, 31))

    body = await create_approval_body(
        "write_file",
        "filesystem.write",
        {"path": str(tmp_path / "large.html"), "content": content},
    )

    assert body is not None
    assert body.hidden_arg_keys == frozenset({"content"})
    assert len(body.widgets) == 1
    preview = body.widgets[0]
    assert isinstance(preview, ApprovalDiffPreview)
    assert _normalized_eol(preview._after) == content
    assert "line 30" in preview._after


@pytest.mark.asyncio
async def test_file_approval_body_resolves_relative_paths_against_workspace_cwd(tmp_path: Path) -> None:
    from chrys.app.tui.screens.dialogs.approval.bodies.file_edit import ApprovalDiffPreview

    target = tmp_path / "existing.txt"
    target.write_text("old\ncontent\n", encoding="utf-8")

    write_body = await create_approval_body(
        "write_file",
        KIND_FILESYSTEM_WRITE,
        {"path": "created.txt", "content": "new\ncontent\n"},
        workspace_cwd=str(tmp_path),
    )
    edit_body = await create_approval_body(
        "edit_file",
        KIND_FILESYSTEM_WRITE,
        {"path": "existing.txt", "old_string": "old", "new_string": "new"},
        workspace_cwd=str(tmp_path),
    )

    assert write_body is not None
    write_preview = write_body.widgets[0]
    assert isinstance(write_preview, ApprovalDiffPreview)
    assert write_preview._path == str(tmp_path / "created.txt")

    assert edit_body is not None
    edit_preview = edit_body.widgets[0]
    assert isinstance(edit_preview, ApprovalDiffPreview)
    assert edit_preview._path == str(target)
    assert edit_preview._before == "old\ncontent\n"
    assert edit_preview._after == "new\ncontent\n"


@pytest.mark.asyncio
async def test_write_file_approval_body_scoped_to_filesystem_write(tmp_path: Path) -> None:
    write_body = await create_approval_body(
        "write_file",
        KIND_MCP,
        {"path": str(tmp_path / "large.html"), "content": "new"},
    )
    edit_body = await create_approval_body(
        "edit_file",
        KIND_MCP,
        {"path": str(tmp_path / "large.html"), "old_string": "old", "new_string": "new"},
    )

    assert write_body is None
    assert edit_body is None


@pytest.mark.asyncio
async def test_sub_agent_approval_body_edits_prompt() -> None:
    body = await create_approval_body(
        "explore_agent",
        KIND_SUB_AGENT,
        {"prompt": "inspect src"},
    )

    assert body is not None
    assert body.hidden_arg_keys == frozenset({"prompt"})

    dialog = ApprovalDialog(
        caller_name="Code",
        tool_name="explore_agent",
        tool_kind=KIND_SUB_AGENT,
        args={"prompt": "inspect src"},
        approval_body=body,
    )
    results: list[tuple[bool, str, dict[str, object] | None]] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        prompt_input = dialog.query_one("#approval-sub-agent-prompt-input", EnhancedTextArea)
        prompt_input.clear()
        prompt_input.insert("inspect src and tests")
        dialog.query_one("#approval-yes", Button).press()
        await pilot.pause()

    assert results == [(True, "", {"prompt": "inspect src and tests"})]


@pytest.mark.asyncio
async def test_sub_agent_approval_body_falls_back_after_name_builder_declines() -> None:
    """Name builder returns None on kind mismatch; kind builder fills in."""
    body = await create_approval_body(
        "write_file",
        KIND_SUB_AGENT,
        {"prompt": "inspect src"},
    )

    assert body is not None
    assert isinstance(body.detail, MessageRef)
    assert format_message(body.detail) == "Review sub-agent delegation"
    assert body.hidden_arg_keys == frozenset({"prompt"})
    assert body.modified_args is not None


@pytest.mark.asyncio
async def test_write_file_overwrite_approval_body_shows_existing_content(tmp_path: Path) -> None:
    from chrys.app.tui.screens.dialogs.approval.bodies.file_edit import ApprovalDiffPreview

    target = tmp_path / "existing.txt"
    target.write_text("old\ncontent\n", encoding="utf-8")

    body = await create_approval_body(
        "write_file",
        KIND_FILESYSTEM_WRITE,
        {"path": str(target), "content": "new\ncontent\n", "overwrite": True},
    )

    assert body is not None
    assert len(body.widgets) == 1
    preview = body.widgets[0]
    assert isinstance(preview, ApprovalDiffPreview)
    assert preview._before == "old\ncontent\n"
    assert _normalized_eol(preview._after) == "new\ncontent\n"
    assert str(preview.border_title) == "Planned Diff"


@pytest.mark.asyncio
async def test_approval_dialog_manual_reason_disables_approve_until_empty() -> None:
    """Manual mode treats raw reason text as decline-only intent."""
    dialog = ApprovalDialog(caller_name="", tool_name="read_file")
    results: list[tuple[bool, str, dict[str, object] | None]] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        approve = dialog.query_one("#approval-yes", Button)
        input_area = dialog.query_one("#approval-reason", EnhancedTextArea)

        assert approve.has_focus
        input_area.insert("Use decline instead")
        await pilot.pause()

        assert approve.disabled is True
        assert dialog.query_one("#approval-no", Button).has_focus

        approve.press()
        await pilot.pause()
        assert results == []

        input_area.clear()
        await pilot.pause()
        assert input_area.text == ""
        assert approve.disabled is False

        approve.press()
        await pilot.pause()

    assert results == [(True, "", None)]


@pytest.mark.asyncio
async def test_approval_dialog_manual_whitespace_reason_disables_approve() -> None:
    """Whitespace is still raw reason text, so manual Approve is disabled."""
    dialog = ApprovalDialog(caller_name="", tool_name="read_file")

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        dialog.query_one("#approval-reason", EnhancedTextArea).insert("   \n")
        await pilot.pause()

        assert dialog.query_one("#approval-yes", Button).disabled is True


@pytest.mark.asyncio
async def test_approval_dialog_auto_reason_disables_approve_until_empty() -> None:
    """Reason text is decline-only in AUTO mode too."""
    dialog = ApprovalDialog(caller_name="", tool_name="read_file", judging=True)
    results: list[tuple[bool, str, dict[str, object] | None]] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        approve = dialog.query_one("#approval-yes", Button)
        dialog.query_one("#approval-reason", EnhancedTextArea).insert("This should not block auto mode")
        await pilot.pause()

        assert approve.disabled is True
        approve.press()
        await pilot.pause()
        assert results == []

        dialog.query_one("#approval-reason", EnhancedTextArea).clear()
        await pilot.pause()

        assert approve.disabled is False
        approve.press()
        await pilot.pause()

    assert results == [(True, "", None)]


_FLAG = JudgeVerdict(approved=False, reason="Deletes files outside the workspace")


def _assert_shows_flag(dialog: ApprovalDialog) -> None:
    judge_area = dialog.query_one("#approval-judge", VerticalGroup)
    assert judge_area.has_class("judge-flagged")
    assert str(judge_area.border_title) == "Flagged by Auto-Review"
    assert dialog.query_one("#approval-judge-loading", ChrysLoadingIndicator).display is False
    concern = dialog.query_one("#approval-concern", Static)
    assert concern.display is True
    assert str(concern.content) == _FLAG.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("delivered", ["at_construction", "before_mount"])
async def test_a_flag_that_beats_the_dialog_opens_it_flagged_with_the_reason_focused(delivered: str) -> None:
    """It may pop up while the user is typing: a stray ``y`` must land in the reason, not approve."""
    if delivered == "at_construction":
        dialog = ApprovalDialog(caller_name="", tool_name="rm", judging=True, verdict=_FLAG)
    else:
        dialog = ApprovalDialog(caller_name="", tool_name="rm", judging=True)
        dialog.receive_verdict(_FLAG)
    results: list[tuple[bool, str, dict[str, object] | None]] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        reason = dialog.query_one("#approval-reason", EnhancedTextArea)
        await wait_for(lambda: reason.has_focus, pilot=pilot, description="reason input focused")
        _assert_shows_flag(dialog)

        await pilot.press("y", "n")
        await pilot.pause()

        assert reason.text == "yn"
        assert results == []


@pytest.mark.asyncio
async def test_a_flag_for_an_open_dialog_leaves_the_focus_where_it_is() -> None:
    dialog = ApprovalDialog(caller_name="", tool_name="rm", judging=True)

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        approve = dialog.query_one("#approval-yes", Button)
        await wait_for(lambda: approve.has_focus, pilot=pilot, description="approve button focused")

        dialog.receive_verdict(_FLAG)
        await pilot.pause()

        _assert_shows_flag(dialog)
        assert approve.has_focus


@pytest.mark.asyncio
async def test_approval_dialog_cancellation_dismisses_with_no_decision() -> None:
    dialog = ApprovalDialog(caller_name="ACP child", tool_name="read_file", judging=True)
    results: list[tuple[bool, str, dict[str, object] | None] | None] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        dialog.dismiss_due_to_cancellation()
        await wait_for(
            lambda: results == [None],
            pilot=pilot,
            description="cancelled approval dialog callback",
        )

    assert dialog.is_dismissed is True
    assert dialog.user_decision_submitted is False
    assert results == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("turns", [0, 1, 2, 3, 5])
async def test_approval_dialog_cancellation_while_mounting_closes_once_mounted(turns: int) -> None:
    """Removing a dialog that is still composing mounts it without the buttons it focuses."""
    dialog = ApprovalDialog(caller_name="ACP child", tool_name="read_file", judging=True)
    results: list[tuple[bool, str, dict[str, object] | None] | None] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        app.push_screen(dialog, callback=results.append)
        for _ in range(turns):
            await asyncio.sleep(0)

        dialog.dismiss_due_to_cancellation()
        await wait_for(lambda: results == [None], pilot=pilot, description="cancelled approval dialog callback")

        assert dialog not in app.screen_stack
    assert dialog.user_decision_submitted is False


@pytest.mark.asyncio
async def test_approval_dialog_auto_approval_while_covered_preserves_result_and_top_screen() -> None:
    dialog = ApprovalDialog(caller_name="ACP child", tool_name="read_file", judging=True)
    results: list[tuple[bool, str, dict[str, object] | None] | None] = []
    cover_results: list[object] = []

    class _CoverScreen(ModalScreen[str]):
        def compose(self) -> ComposeResult:
            yield Static("covering question")

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        cover = _CoverScreen()
        await app.push_screen(cover, callback=cover_results.append)
        await pilot.pause()

        dialog.receive_verdict(JudgeVerdict(approved=True, reason="safe"))
        await pilot.pause()

        assert app.screen is cover
        assert results == []
        assert dialog.is_dismissed is True

        cover.dismiss("answered")
        await wait_for(
            lambda: results == [(True, "", None)],
            pilot=pilot,
            description="deferred auto-approval dismissal on resume",
        )

    assert cover_results == ["answered"]
    assert results == [(True, "", None)]


@pytest.mark.asyncio
async def test_approval_dialog_cancellation_rechecks_top_screen_on_stale_resume() -> None:
    dialog = ApprovalDialog(caller_name="ACP child", tool_name="read_file", judging=True)
    results: list[tuple[bool, str, dict[str, object] | None] | None] = []
    cover_results: list[object] = []

    class _CoverScreen(ModalScreen[str]):
        def compose(self) -> ComposeResult:
            yield Static("covering question")

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        first_cover = _CoverScreen()
        await app.push_screen(first_cover, callback=cover_results.append)
        await pilot.pause()

        dialog.dismiss_due_to_cancellation()
        first_cover.dismiss("first")
        second_cover = _CoverScreen()
        await app.push_screen(second_cover, callback=cover_results.append)
        await pilot.pause()

        assert app.screen is second_cover
        assert results == []
        assert dialog.is_dismissed is True

        second_cover.dismiss("second")
        await wait_for(
            lambda: results == [None],
            pilot=pilot,
            description="deferred cancellation after replacement cover",
        )

    assert cover_results == ["first", "second"]
    assert results == [None]


@pytest.mark.asyncio
async def test_approval_dialog_reason_supports_shift_enter_newline() -> None:
    """The reason field supports explicit multiline input without submitting."""
    dialog = ApprovalDialog(caller_name="", tool_name="zsh")
    results: list[tuple[bool, str, dict[str, object] | None]] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        input_area = dialog.query_one("#approval-reason", EnhancedTextArea)
        input_area.insert("Line one")
        input_area.focus()
        await wait_for(lambda: input_area.has_focus, pilot=pilot, description="control focus before interaction")
        await pilot.press("shift+enter")
        await pilot.pause()

        assert input_area.text == "Line one\n"
        assert results == []


@pytest.mark.asyncio
async def test_approval_dialog_reason_newline_keeps_cursor_visible_after_height_cap() -> None:
    dialog = ApprovalDialog(caller_name="", tool_name="zsh")

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test(size=(130, 30)) as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        input_area = dialog.query_one("#approval-reason", EnhancedTextArea)
        input_area.focus()
        await wait_for(lambda: input_area.has_focus, pilot=pilot, description="control focus before interaction")
        await pilot.press("1")
        for line in range(2, 7):
            await pilot.press("enter")
            await pilot.press(str(line))
        await pilot.pause()

        assert input_area.document.line_count == 6
        assert input_area.content_size.height == 5

        await pilot.press("enter")
        await pilot.pause()

        cursor_y = input_area.cursor_location[0]
        scroll_y = round(input_area.scroll_y)
        assert scroll_y <= cursor_y < scroll_y + input_area.content_size.height
