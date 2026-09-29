# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shell integration: how each shell is launched, and what real shells then report to the emulator."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol
from unittest.mock import patch

import pytest

from chrys.app.tui.terminal import shell_integration
from chrys.app.tui.terminal.emulator import CommandSubmitted, DirectoryChanged, TerminalEmulator
from chrys.app.tui.terminal.pty_process import PtyProcess, spawn_pty_process
from chrys.app.tui.terminal.shell_integration import ShellLaunch, prepare_shell_launch
from chrys.foundation.platform import get_platform
from tests.support.waiting import wait_for, with_wait_deadline

# A live shell test may spend its whole wait budget and then still has a shell to end.
pytestmark = pytest.mark.timeout(120)

# -- the launch ----------------------------------------------------------------------------------


def test_zsh_reads_our_startup_files_which_read_the_users(tmp_path: Path) -> None:
    user_zdotdir = str(tmp_path / "my 'zsh' files")
    env = {"ZDOTDIR": user_zdotdir, "HOME": str(tmp_path)}

    launch = prepare_shell_launch("/bin/zsh -f", env)

    assert launch.argv == ["/bin/zsh", "-f"]
    assert launch.scratch_directory is not None
    scratch = Path(launch.scratch_directory)
    assert env == {"ZDOTDIR": str(scratch), "HOME": str(tmp_path)}
    zshenv = (scratch / ".zshenv").read_text(encoding="utf-8")
    assert f"ZDOTDIR={shlex.quote(user_zdotdir)}\n" in zshenv
    assert 'source "$ZDOTDIR/.zshenv"' in zshenv
    zshrc = (scratch / ".zshrc").read_text(encoding="utf-8")
    assert 'source "$ZDOTDIR/.zshrc"' in zshrc
    assert "precmd_functions+=(__chrys_precmd)" in zshrc
    assert "preexec_functions+=(__chrys_preexec)" in zshrc

    launch.clean_up()
    launch.clean_up()
    assert not scratch.exists()


def test_zsh_without_a_zdotdir_looks_in_the_home_the_shell_will_have(tmp_path: Path) -> None:
    env = {"HOME": str(tmp_path / "home")}

    launch = prepare_shell_launch("zsh", env)
    try:
        assert launch.scratch_directory is not None
        zshenv = Path(launch.scratch_directory, ".zshenv").read_text(encoding="utf-8")
        assert f"ZDOTDIR={shlex.quote(str(tmp_path / 'home'))}\n" in zshenv
    finally:
        launch.clean_up()


def test_bash_gets_its_rcfile_option_ahead_of_every_other_argument() -> None:
    env: dict[str, str] = {}

    launch = prepare_shell_launch("bash --noprofile -i", env)
    try:
        assert launch.scratch_directory is not None
        rcfile = Path(launch.scratch_directory, "init.bash")
        assert launch.argv == ["bash", "--rcfile", str(rcfile), "--noprofile", "-i"]
        assert env == {}
        script = rcfile.read_text(encoding="utf-8")
        assert script.startswith("[[ -f ~/.bashrc ]] && source ~/.bashrc\n")
        assert "trap '__chrys_preexec' DEBUG" in script
        assert "b64:" in script
    finally:
        launch.clean_up()
    assert not rcfile.exists()


@pytest.mark.parametrize(("command", "failed_file"), [("bash", "init.bash"), ("zsh", ".zshenv"), ("zsh", ".zshrc")])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_failed_startup_file_write_removes_scratch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, failed_file: str, error_type: type[BaseException]
) -> None:
    monkeypatch.setattr(shell_integration.tempfile, "tempdir", str(tmp_path))
    env = {"HOME": str(tmp_path / "home"), "ZDOTDIR": str(tmp_path / "user-zsh")}
    original_env = env.copy()
    real_write_text = Path.write_text
    written: list[Path] = []
    error = error_type("startup write failed")

    def failing_write_text(
        path: Path, data: str, encoding: str | None = None, errors: str | None = None, newline: str | None = None
    ) -> int:
        result = real_write_text(path, data, encoding=encoding, errors=errors, newline=newline)
        written.append(path)
        if path.name == failed_file:
            raise error
        return result

    with (
        patch.object(Path, "write_text", autospec=True, side_effect=failing_write_text),
        pytest.raises(error_type, match="startup write failed") as caught,
    ):
        prepare_shell_launch(command, env)

    assert caught.value is error
    assert written[-1].name == failed_file
    assert list(tmp_path.iterdir()) == []
    assert env == original_env


def test_fish_gets_its_hooks_as_an_init_command() -> None:
    launch = prepare_shell_launch("fish --private", {})

    assert launch.argv[:3] == ["fish", "--private", "-C"]
    assert "--on-event fish_preexec" in launch.argv[3]
    assert "--on-event fish_prompt" in launch.argv[3]
    assert "b64:" in launch.argv[3]
    assert launch.scratch_directory is None


@pytest.mark.parametrize("command", ["pwsh", "powershell", "pwsh-preview -NoLogo", "PowerShell.EXE"])
def test_powershell_gets_its_hooks_as_a_command_it_stays_open_after(command: str) -> None:
    launch = prepare_shell_launch(command, {})

    assert launch.argv[:-3] == command.split()
    assert launch.argv[-3:-1] == ["-NoExit", "-Command"]
    assert "b64:" in launch.argv[-1]
    assert "function prompt" in launch.argv[-1]
    assert launch.scratch_directory is None


def test_a_path_with_spaces_that_exists_is_a_program_not_a_command_line(tmp_path: Path) -> None:
    program = tmp_path / "Program Files" / "PowerShell 7" / "pwsh.exe"
    program.parent.mkdir(parents=True)
    program.touch()

    launch = prepare_shell_launch(str(program), {})

    assert launch.argv == [str(program), "-NoExit", "-Command", launch.argv[-1]]


def test_a_quoted_program_path_with_spaces_is_one_argument_without_its_quotes() -> None:
    launch = prepare_shell_launch('"/opt/My Shells/nu" --login', {})

    assert launch == ShellLaunch(["/opt/My Shells/nu", "--login"])


def test_the_shell_is_recognized_by_its_file_name_whatever_the_directory_or_case() -> None:
    fish = prepare_shell_launch(str(Path("/opt/zsh-tools/bin/FISH")), {})
    unknown = prepare_shell_launch("/opt/bash-tools/bin/nu --login", {})

    assert fish.argv[1] == "-C"
    assert unknown == ShellLaunch(["/opt/bash-tools/bin/nu", "--login"])


def test_a_shell_without_hooks_starts_as_it_is() -> None:
    env = {"KEEP": "1"}

    launch = prepare_shell_launch("sh -i", env)

    assert launch == ShellLaunch(["sh", "-i"])
    assert env == {"KEEP": "1"}
    launch.clean_up()


def test_a_command_line_that_cannot_be_split_is_taken_as_the_program() -> None:
    assert prepare_shell_launch('nu "unterminated', {}).argv == ['nu "unterminated']
    assert prepare_shell_launch("", {}).argv == [""]


# -- real shells ---------------------------------------------------------------------------------

_LIVE_BUDGET = 45.0
_LIVE_WAIT = 30.0
_CLOSE_WAIT = 10.0

posix_only = pytest.mark.skipif(get_platform().is_windows, reason="drives shells through a POSIX pseudo-terminal")


class LiveShell:
    """A real shell on a pseudo-terminal, shown on a real emulator."""

    def __init__(self, process: PtyProcess, launch: ShellLaunch) -> None:
        self.process = process
        self.launch = launch
        self.emulator = TerminalEmulator(120, 30)
        self.events: list[DirectoryChanged | CommandSubmitted] = []
        self._pump = asyncio.create_task(self._show_output())

    async def _show_output(self) -> None:
        while output := await self.process.read():
            update = self.emulator.feed(output)
            self.events.extend(update.events)
            if update.replies:
                # Line editors ask the terminal questions and wait for the answers.
                await self.process.write(update.replies)

    @property
    def directories(self) -> list[str]:
        return [event.path for event in self.events if isinstance(event, DirectoryChanged)]

    @property
    def commands(self) -> list[str]:
        return [event.command for event in self.events if isinstance(event, CommandSubmitted)]

    @property
    def screen(self) -> list[str]:
        buffer = self.emulator.buffer
        return [row.text.rstrip() for row in buffer.rows]

    async def prompt(self, count: int = 1) -> None:
        """Wait until the shell has reported its directory, as it does ahead of every prompt."""
        await wait_for(
            lambda: len(self.directories) >= count or self._pump.done(),
            timeout=_LIVE_WAIT,
            description=f"prompt number {count}",
        )
        assert not self._pump.done(), f"shell ended early; screen: {self.screen}"

    async def enter(self, line: str) -> None:
        """Type a line and wait for the prompt that follows it."""
        prompts = len(self.directories)
        assert await self.process.write(f"{line}\r")
        await self.prompt(prompts + 1)

    async def close(self) -> None:
        self.process.kill()
        try:
            async with asyncio.timeout(_CLOSE_WAIT):
                await self._pump
        finally:
            self._pump.cancel()
            await self.process.close()
            self.launch.clean_up()


class StartShell(Protocol):
    async def __call__(self, name: str) -> LiveShell: ...


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    return home


@pytest.fixture
async def start_shell(tmp_path: Path, home: Path) -> AsyncIterator[StartShell]:
    started: list[LiveShell] = []
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def start(name: str) -> LiveShell:
        program = shutil.which(name)
        if program is None:
            pytest.skip(f"{name} is not installed")
        inherited = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "USER", "LOGNAME", "SYSTEMROOT")
        env = {name: os.environ[name] for name in inherited if name in os.environ}
        env.update(
            HOME=str(home),
            TERM="xterm-256color",
            XDG_CONFIG_HOME=str(home / ".config"),
            XDG_DATA_HOME=str(home / ".local" / "share"),
            XDG_CACHE_HOME=str(home / ".cache"),
            BASH_SILENCE_DEPRECATION_WARNING="1",
            POWERSHELL_TELEMETRY_OPTOUT="1",
            POWERSHELL_UPDATECHECK="Off",
        )
        launch = prepare_shell_launch(program, env)
        try:
            process = await spawn_pty_process(launch.argv, env=env, cwd=str(workspace), columns=120, lines=30)
        except BaseException:
            launch.clean_up()
            raise
        shell = LiveShell(process, launch)
        started.append(shell)
        await shell.prompt()
        return shell

    yield start
    for shell in started:
        await shell.close()


@posix_only
@pytest.mark.parametrize("name", ["zsh", "bash", "fish", "pwsh"])
@with_wait_deadline(_LIVE_BUDGET)
async def test_live_shell_reports_its_directory_and_each_command_line(
    start_shell: StartShell, tmp_path: Path, name: str
) -> None:
    workspace = os.path.realpath(tmp_path / "workspace")
    # Everything a naive report would trip over: a separator of the report itself, blanks, non-ASCII.
    target = Path(workspace, "with space; and 中文")
    target.mkdir()

    shell = await start_shell(name)
    assert shell.directories == [workspace]
    assert shell.commands == []

    await shell.enter("")
    assert shell.directories == [workspace, workspace]
    assert shell.commands == []

    await shell.enter("cd with*")
    assert shell.directories[-1] == str(target)
    assert shell.commands == ["cd with*"]

    pipeline = "echo one two | sort ; echo three"
    await shell.enter(pipeline)
    assert shell.commands == ["cd with*", pipeline]
    assert shell.directories[-1] == str(target)
    assert not any("b64:" in line or "__chrys" in line for line in shell.screen)


@posix_only
@with_wait_deadline(_LIVE_BUDGET)
async def test_live_bash_leaves_the_users_prompt_command_working_and_unreported(
    start_shell: StartShell, home: Path
) -> None:
    (home / ".bashrc").write_text(
        "PROMPT_COMMAND='__mine_status=$?; __mine_prompts=$((__mine_prompts + 1)); '\nHISTCONTROL=ignoreboth\n",
        encoding="utf-8",
    )

    shell = await start_shell("bash")
    await shell.enter("false")
    await shell.enter("echo seen=$__mine_status")
    await shell.enter("echo seen=$__mine_status")
    await shell.enter(" echo hidden from history | cat")

    # The user's prompt command still sees the status of the command line, and is never taken for one.
    assert shell.commands[:3] == ["false", "echo seen=$__mine_status", "echo seen=$__mine_status"]
    # Kept out of the history on purpose, so only its first simple command is known.
    assert shell.commands[3] == "echo hidden from history"
    assert len(shell.commands) == 4
    assert shell.screen.count("seen=1") == 1
    assert shell.screen.count("seen=0") == 1


@posix_only
@with_wait_deadline(_LIVE_BUDGET)
async def test_live_zsh_still_reads_the_users_own_startup_files(start_shell: StartShell, home: Path) -> None:
    moved = home / "zsh files"
    moved.mkdir()
    (home / ".zshenv").write_text(f"export FROM_ZSHENV=yes\nZDOTDIR={shlex.quote(str(moved))}\n", encoding="utf-8")
    (moved / ".zshrc").write_text("FROM_ZSHRC=yes\nPS1='%% '\n", encoding="utf-8")

    shell = await start_shell("zsh")
    await shell.enter('echo "env=$FROM_ZSHENV rc=$FROM_ZSHRC"')
    await shell.enter('[[ $ZDOTDIR == "$HOME/zsh files" ]] && echo zdotdir-restored')

    assert "env=yes rc=yes" in shell.screen
    assert "zdotdir-restored" in shell.screen
