# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tool output stays literal from real producers through cards and detail views."""

from __future__ import annotations

import shlex
import sys
from typing import TYPE_CHECKING

import pytest
from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Static

from chrys.app.tui.widgets.chat.renderers.execute import ExecuteToolCall
from chrys.app.tui.widgets.chat.renderers.search import GlobToolCall, GrepToolCall
from chrys.app.tui.widgets.chat.renderers.skill import SkillToolCall
from chrys.app.tui.widgets.chat.tool_call import BaseToolCard
from chrys.app.tui.widgets.chat.tool_view_builders import ToolViewContent, build_code_view, build_params_view
from chrys.foundation.models.session_env import SessionEnvironment
from chrys.service.skills import runner as runner_module
from chrys.service.skills.loader import load_file_skill
from chrys.service.skills.model import Skill
from chrys.service.skills.provider import ChrysSkillsProvider
from chrys.service.skills.runner import SubprocessScriptRunner
from chrys.service.tools.builtins.search import grep
from chrys.service.tools.builtins.shell import ShellTools, shell_progress_callback
from tests.support.tui_helpers import LocalizedApp, rich_plain
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from pathlib import Path

    from textual.widget import Widget

_PAYLOAD = "needle \x1b[24Dvisible \x1bPcontrol\x1b\\ \x9b24D\x07\x7f\n\t中文 end-marker\n"
_ERROR = "Error: " + _PAYLOAD
_CONTROL = "\x1b[24D"
_SELECTORS = {
    "execute": "#exec-panel",
    "grep": "#search-panel",
    "glob": "#search-panel",
    "load_skill": "#skill-panel",
    "read_skill_resource": "#skill-panel",
    "run_skill_script": "#skill-panel",
}


class _OutputApp(LocalizedApp):
    CSS = ".tool-view-md { height: 16; }"

    def __init__(self, card: BaseToolCard) -> None:
        super().__init__()
        self.card = card

    def compose(self) -> ComposeResult:
        yield self.card


def _rendered_text(widget: Widget) -> str:
    # Strip.text contains the actual Textual-rendered characters without the
    # legitimate style/position escape sequences added by the terminal writer.
    return "\n".join(widget.render_line(y).text for y in range(widget.size.height))


def _assert_literal(text: str) -> None:
    assert "end-marker" in text
    _assert_no_controls(text)


def _assert_no_controls(text: str) -> None:
    assert not any((ord(char) < 32 and char != "\n") or 0x7F <= ord(char) <= 0x9F for char in text)


async def _check_card_and_details(card: BaseToolCard, result: str, *, error: bool = False) -> None:
    async with _OutputApp(card).run_test(size=(160, 60)) as pilot:
        if error:
            card.set_error(result)
        else:
            card.set_complete(result)
        panel = card.query_one(_SELECTORS[card.tool_name], Static)
        await wait_for(
            lambda: "end-marker" in _rendered_text(panel), pilot=pilot, description="tool output is rendered"
        )
        _assert_literal(_rendered_text(panel))
        assert isinstance(panel.content, Text)
        _assert_literal(panel.content.plain)
        assert "中文 end-marker" in panel.content.plain

        # Exercise the same builders used by the modal, including Markdown for
        # loaded skills, rather than merely re-rendering the compact card.
        details = card._tool_view_output_widgets()
        await pilot.app.screen.mount(ToolViewContent(details))
        await wait_for(
            lambda: any("end-marker" in _rendered_text(widget) for widget in details),
            pilot=pilot,
            description="full tool detail output is rendered",
        )
        _assert_literal("\n".join(_rendered_text(widget) for widget in details))
        assert card.result_text == result
        assert card._tool_copy_sections()[0][2] == result


@pytest.mark.parametrize("tool_name", ["grep", "load_skill", "read_skill_resource", "run_skill_script"])
async def test_real_file_outputs_are_literal_in_cards_and_details(
    tool_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill_dir = tmp_path / "display-test"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: display-test\ndescription: Display test\n---\n" + _PAYLOAD, encoding="utf-8"
    )
    resource_path = skill_dir / "reference.txt"
    resource_path.write_text(_PAYLOAD, encoding="utf-8")
    (skill_dir / "emit.py").write_text(
        f"import sys\nsys.stdout.buffer.write({_PAYLOAD.encode('utf-8')!r})\n", encoding="utf-8"
    )
    if tool_name == "grep":
        result = await grep("needle", path=str(resource_path), context_lines=1)
        card = GrepToolCall("search", "grep", args={"pattern": "needle"})
    else:
        skill = load_file_skill(str(skill_dir), script_extensions=[".py"])
        assert isinstance(skill, Skill)

        async def load_skills() -> list[Skill]:
            return [skill]

        # Execute the real script runner without resolving/downloading a new
        # interpreter; the test owns this process and its temporary input file.
        monkeypatch.setattr(runner_module, "_find_python_runner", lambda: [sys.executable])
        provider = ChrysSkillsProvider(load_skills, script_runner=SubprocessScriptRunner(timeout=10))
        await provider.initialize()
        if tool_name == "load_skill":
            result = await provider._load_skill(provider._skills, skill.name)
        elif tool_name == "read_skill_resource":
            result = await provider._read_skill_resource(provider._skills, skill.name, "reference.txt")
        else:
            result = await provider._run_skill_script_chrys(provider._skills, skill.name, "emit.py")
        card = SkillToolCall("skill", tool_name, args={"skill_name": skill.name})

    assert "\x1b" in result  # The producer has not already made this display safe.
    await _check_card_and_details(card, result)


async def test_real_shell_stream_completion_and_replay_neutralize_remaining_controls(tmp_path: Path) -> None:
    script = tmp_path / "emit.py"
    script.write_text(f"import sys\nsys.stdout.buffer.write({_PAYLOAD.encode('utf-8')!r})\n", encoding="utf-8")
    runtime = SessionEnvironment.capture()
    if runtime.platform.shell.name in {"pwsh", "powershell"}:
        executable = sys.executable.replace("'", "''")
        script_arg = str(script).replace("'", "''")
        command = f"& '{executable}' '{script_arg}'"
    elif runtime.platform.shell.name == "cmd":
        command = f'"{sys.executable}" "{script}"'
    else:
        command = shlex.join([sys.executable, str(script)])

    card = ExecuteToolCall("shell", "execute", args={"command": command})
    progress: list[str] = []
    async with _OutputApp(card).run_test(size=(160, 40)) as pilot:

        async def collect(lines: list[str]) -> None:
            progress.extend(lines)
            card.update_progress(lines)
            content = card.query_one("#exec-panel", Static).content
            assert isinstance(content, Text)
            assert "\x1b" not in content.plain
            assert "\x9b" not in content.plain

        token = shell_progress_callback.set(collect)
        try:
            result = await ShellTools(runtime).execute(command, reason="display test", timeout=10)
        finally:
            shell_progress_callback.reset(token)
        assert "[exit_code: 0]" in result
        assert _CONTROL not in result  # Common ANSI was already stripped upstream.
        assert "\x1bP" in result  # Other escape families still reach the display boundary.
        assert any("\x1bP" in line for line in progress)
        panel = card.query_one("#exec-panel", Static)
        await wait_for(
            lambda: "end-marker" in _rendered_text(panel), pilot=pilot, description="shell progress is rendered"
        )
        _assert_literal(_rendered_text(panel))
        card.set_complete(result)
        assert isinstance(panel.content, Text)
        _assert_literal(panel.content.plain)
        state = card.compact_display_state()
        assert state == {"progress_lines": progress}

    replay = ExecuteToolCall("replay", "execute", args={"command": command})
    assert state is not None
    replay.restore_compact_display_state(state)
    await _check_card_and_details(replay, result)
    await _check_card_and_details(ExecuteToolCall("final", "execute", args={"command": command}), result)


@pytest.mark.parametrize("tool_name", list(_SELECTORS))
@pytest.mark.parametrize("error", [False, True], ids=["error-result", "set-error"])
async def test_error_results_and_error_callbacks_are_literal(tool_name: str, *, error: bool) -> None:
    if tool_name == "execute":
        card = ExecuteToolCall("error", tool_name)
    elif tool_name == "grep":
        card = GrepToolCall("error", tool_name)
    elif tool_name == "glob":
        card = GlobToolCall("error", tool_name)
    else:
        card = SkillToolCall("error", tool_name)
    await _check_card_and_details(card, _ERROR, error=error)


async def test_skill_metadata_is_literal_after_entity_decoding_and_keeps_styles() -> None:
    result = (
        _PAYLOAD + "\n<skill_dir>/skills/&amp; \x1b[24Ddocs</skill_dir>\n"
        '<resources><resource name="ref\x1b[24D.md"/></resources>\n'
        '<scripts><script name="run\x1b[24D.py"/></scripts>\n'
    )
    card = SkillToolCall("metadata", "load_skill", args={"skill_name": "docs" + _CONTROL})
    await _check_card_and_details(card, result)
    display, _, _ = card._format_result(result)
    _assert_literal(display.plain)
    assert "/skills/& �[24Ddocs" in display.plain
    cyan_text = "".join(display.plain[span.start : span.end] for span in display.spans if span.style == "cyan")
    assert "/skills/& �[24Ddocs" in cyan_text
    assert "ref�[24D.md" in cyan_text
    assert "run�[24D.py" in cyan_text
    assert _CONTROL not in card._label_text().plain


async def test_search_result_titles_and_file_paths_are_literal() -> None:
    card = GlobToolCall("glob", "glob", args={"pattern": "*" + _CONTROL})
    result = f"Found 1 file(s) matching '*' in /root{_CONTROL}\n\n/root/{_PAYLOAD}"
    await _check_card_and_details(card, result)
    assert _CONTROL not in card._label_text().plain


@pytest.mark.parametrize("language", ["text", "python", "markdown"])
async def test_detail_builder_filters_controls_with_language_appropriate_indentation(language: str) -> None:
    async with LocalizedApp().run_test(size=(100, 30)) as pilot:
        widget = build_code_view(language, "first\r\n\t中文 end-marker\r" + _CONTROL)
        widget.styles.height = 10
        await pilot.app.screen.mount(widget)
        await wait_for(
            lambda: "end-marker" in _rendered_text(widget), pilot=pilot, description="detail body is rendered"
        )
        _assert_literal(_rendered_text(widget))
        if isinstance(widget, Static):
            indent = 8 if language == "text" else 4
            assert " " * indent + "中文 end-marker" in rich_plain(widget.content)


async def test_shell_input_and_detail_parameters_do_not_bypass_output_filtering() -> None:
    args = {"command": _PAYLOAD, "reason": "reason" + _CONTROL}
    card = ExecuteToolCall("input", "execute", args=args)
    async with _OutputApp(card).run_test(size=(160, 40)) as pilot:
        panel = card.query_one("#exec-cmd", Static)
        await wait_for(
            lambda: "end-marker" in _rendered_text(panel), pilot=pilot, description="command input is rendered"
        )
        _assert_literal(_rendered_text(panel))
        for widget in card._tool_view_input_widgets():
            assert isinstance(widget, Static)
            assert "\x1b" not in rich_plain(widget.content)
        assert card.args == args

        params = {"name" + _CONTROL: _PAYLOAD, "single": "end-marker" + _CONTROL, "list": [_PAYLOAD]}
        for widget in build_params_view(params):
            assert isinstance(widget, Static)
            _assert_no_controls(rich_plain(widget.content))
