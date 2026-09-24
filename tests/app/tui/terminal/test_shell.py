# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""`Shell`: one process piped to one terminal, from spawn to exit, against a scripted PTY."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from textual.message import Message

import chrys.app.tui.terminal.shell as shell_module
from chrys import __version__
from chrys.app.tui.terminal.pty_process import PtyProcess
from chrys.app.tui.terminal.shell import Shell, ShellFinished
from chrys.app.tui.terminal.widget import Terminal
from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.i18n import Localizer, MessageRef
from chrys.foundation.i18n.formatting import format_message
from tests.support.waiting import wait_for

type Operation = tuple[str, str] | tuple[str, int, int]


class FakeTerminal:
    """The part of `Terminal` a shell touches."""

    def __init__(self, width: int = 80, height: int = 24) -> None:
        self.width = width
        self.height = height
        self.host_repaints = False
        self.stdin_writer: Callable[[str], Awaitable[object]] | None = None
        self.output: list[str] = []
        self.messages: list[Message] = []
        self.on_write: Callable[[str], Awaitable[None]] | None = None

    def set_write_to_stdin(self, write_to_stdin: Callable[[str], Awaitable[object]] | None) -> None:
        self.stdin_writer = write_to_stdin

    async def write(self, text: str) -> bool:
        if self.on_write is not None:
            await self.on_write(text)
        self.output.append(text)
        return True

    def post_message(self, message: Message) -> bool:
        self.messages.append(message)
        return True

    @property
    def finished_announcements(self) -> int:
        return sum(isinstance(message, ShellFinished) for message in self.messages)


class FakePtyProcess:
    """A `PtyProcess` whose output the test supplies and whose input the test reads back."""

    def __init__(self, size: tuple[int, int] | None = (80, 24), *, repaints_after_resize: bool = False) -> None:
        self.size = size
        self.repaints_after_resize = repaints_after_resize
        self.output: asyncio.Queue[str | BaseException] = asyncio.Queue()
        self.operations: list[Operation] = []
        self.kills = 0
        self.closes = 0
        self.kill_ends_output = True
        self.resize_succeeds = True
        self.on_resize: Callable[[int, int], Awaitable[None]] | None = None

    async def read(self) -> str:
        item = await self.output.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def write(self, data: str) -> bool:
        self.operations.append(("write", data))
        return True

    async def resize(self, columns: int, lines: int) -> bool:
        self.operations.append(("resize", columns, lines))
        if self.on_resize is not None:
            await self.on_resize(columns, lines)
        if self.resize_succeeds:
            self.size = (columns, lines)
        return self.resize_succeeds

    def kill(self) -> None:
        self.kills += 1
        if self.kill_ends_output:
            self.output.put_nowait("")

    async def close(self) -> None:
        self.closes += 1

    @property
    def writes(self) -> list[str]:
        return [operation[1] for operation in self.operations if len(operation) == 2]


@dataclass(frozen=True)
class SpawnCall:
    argv: list[str]
    env: dict[str, str]
    cwd: str
    columns: int
    lines: int


class Spawner:
    """Stands in for `spawn_pty_process`."""

    def __init__(self, process: FakePtyProcess | None = None, *, error: BaseException | None = None) -> None:
        self.process = process or FakePtyProcess()
        self.error = error
        self.calls: list[SpawnCall] = []
        self.gate: asyncio.Event | None = None

    async def spawn(
        self, argv: Sequence[str], *, env: Mapping[str, str], cwd: str, columns: int, lines: int
    ) -> PtyProcess:
        self.calls.append(SpawnCall(list(argv), dict(env), cwd, columns, lines))
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.process


@pytest.fixture
def spawner(monkeypatch: pytest.MonkeyPatch) -> Spawner:
    spawner = Spawner()
    monkeypatch.setattr(shell_module, "spawn_pty_process", spawner.spawn)
    return spawner


def _shell(
    terminal: FakeTerminal,
    *,
    shell_command: str = "sh",
    working_directory: str = "/",
    on_error: Callable[[MessageRef], None] | None = None,
) -> Shell:
    return Shell(cast("Terminal", terminal), working_directory, shell_command, on_error)


async def _started(shell: Shell) -> Shell:
    shell.start()
    await shell.wait_for_ready()
    return shell


# -- start ---------------------------------------------------------------------------------------


async def test_start_spawns_at_the_terminal_size_and_wires_the_two_together(spawner: Spawner) -> None:
    spawner.process = FakePtyProcess((132, 30), repaints_after_resize=True)
    terminal = FakeTerminal(132, 30)
    shell = _shell(terminal, working_directory="/work")
    assert shell.pty_size is None

    await _started(shell)
    try:
        [call] = spawner.calls
        assert (call.argv, call.cwd, call.columns, call.lines) == (["sh"], "/work", 132, 30)
        assert terminal.host_repaints is True
        assert terminal.stdin_writer == shell.write_input_queued
        assert shell.pty_size == (132, 30)
        assert not shell.is_finished
    finally:
        await shell.close()


async def test_invalid_first_measurement_spawns_at_the_default_size(spawner: Spawner) -> None:
    terminal = FakeTerminal(0, 1)
    shell = await _started(_shell(terminal))
    try:
        assert (spawner.calls[0].columns, spawner.calls[0].lines) == (80, 24)
        await shell.send("pwd", 0, 1)
        assert spawner.process.operations == [("write", "pwd\r")]
        assert shell.pty_size == (80, 24)
    finally:
        await shell.close()


async def test_output_reaches_the_terminal_in_order(spawner: Spawner) -> None:
    terminal = FakeTerminal()
    shell = await _started(_shell(terminal))
    try:
        for chunk in ("one", "two", "three"):
            spawner.process.output.put_nowait(chunk)
        await wait_for(lambda: len(terminal.output) == 3, description="all output shown")
        assert terminal.output == ["one", "two", "three"]
    finally:
        await shell.close()


async def test_output_the_terminal_chokes_on_is_logged_and_the_shell_lives_on(
    spawner: Spawner, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = FakeTerminal()
    logged: list[str] = []
    errors: list[MessageRef] = []

    def record(message: str) -> None:
        logged.append(message)

    async def choke(text: str) -> None:
        if text == "bad":
            raise ValueError("cannot draw")

    monkeypatch.setattr(shell_module, "log", record)
    terminal.on_write = choke
    shell = await _started(_shell(terminal, on_error=errors.append))
    try:
        spawner.process.output.put_nowait("bad")
        spawner.process.output.put_nowait("good")
        await wait_for(lambda: terminal.output == ["good"], description="output after the bad chunk")
        assert any("cannot draw" in message for message in logged)
        assert not shell.is_finished
        assert errors == []
    finally:
        await shell.close()


# -- the end -------------------------------------------------------------------------------------


async def test_end_of_output_finishes_the_shell_and_announces_it_once(spawner: Spawner) -> None:
    terminal = FakeTerminal()
    shell = await _started(_shell(terminal, shell_command="bash"))
    rcfile = Path(spawner.calls[0].argv[2])
    assert rcfile.is_file()

    spawner.process.output.put_nowait("bye")
    spawner.process.output.put_nowait("")
    await wait_for(lambda: shell.is_finished, description="shell finished")
    await shell.close()

    assert terminal.output == ["bye"]
    assert terminal.finished_announcements == 1
    assert spawner.process.closes == 1
    assert spawner.process.kills == 0
    assert shell.pty_size is None
    assert not rcfile.parent.exists()


async def test_spawn_failure_is_reported_and_wakes_whoever_waits_for_the_shell(spawner: Spawner) -> None:
    spawner.error = OSError("boom")
    terminal = FakeTerminal()
    errors: list[MessageRef] = []
    shell = _shell(terminal, shell_command="bash", on_error=errors.append)

    await shell.run()
    await shell.wait_for_ready()

    assert shell.is_finished
    assert [format_message(error) for error in errors] == ["Unable to start shell: boom"]
    assert Localizer("zh-Hans").render(errors[0]) == "无法启动 Shell：boom"  # noqa: RUF001
    assert terminal.finished_announcements == 1
    assert terminal.stdin_writer is None
    assert shell.pty_size is None
    assert not Path(spawner.calls[0].argv[2]).parent.exists()
    # Nothing to send to: neither of these may hang or raise.
    await shell.send("pwd", 80, 24)
    await shell.interrupt()


async def test_backend_constructor_failure_is_reported_the_same_way(spawner: Spawner) -> None:
    spawner.error = RuntimeError("WinPTY is unavailable on this platform")
    terminal = FakeTerminal(0, 1)
    errors: list[MessageRef] = []
    shell = _shell(terminal, on_error=errors.append)

    await shell.run()
    await shell.wait_for_ready()

    assert shell.is_finished
    assert [format_message(error) for error in errors] == [
        "Unable to start shell: WinPTY is unavailable on this platform"
    ]
    assert terminal.finished_announcements == 1


async def test_a_failure_while_running_is_reported_and_still_releases_the_process(spawner: Spawner) -> None:
    terminal = FakeTerminal()
    errors: list[MessageRef] = []
    shell = await _started(_shell(terminal, on_error=errors.append))

    spawner.process.output.put_nowait(RuntimeError("pipe broke"))
    await wait_for(lambda: shell.is_finished, description="shell finished")
    await shell.close()

    assert [format_message(error) for error in errors] == ["Shell failed: pipe broke"]
    assert Localizer("zh-Hans").render(errors[0]).endswith("pipe broke")
    assert spawner.process.closes == 1
    assert terminal.finished_announcements == 1


async def test_a_failure_without_a_listener_is_not_an_error_of_its_own(spawner: Spawner) -> None:
    spawner.error = OSError("boom")
    terminal = FakeTerminal()
    shell = _shell(terminal)

    await shell.run()

    assert shell.is_finished
    assert terminal.finished_announcements == 1


# -- send, interrupt, resize ---------------------------------------------------------------------


async def test_send_waits_for_the_shell_then_resizes_before_it_types(spawner: Spawner) -> None:
    spawner.gate = asyncio.Event()
    shell = _shell(FakeTerminal())
    shell.start()
    sending = asyncio.create_task(shell.send("ls -la", 100, 30))
    try:
        await wait_for(lambda: spawner.calls, description="spawn requested")
        assert not sending.done()
        assert spawner.process.operations == []

        spawner.gate.set()
        await sending

        assert spawner.process.operations == [("resize", 100, 30), ("write", "ls -la\r")]
        assert shell.pty_size == (100, 30)
    finally:
        sending.cancel()
        await shell.close()


async def test_a_hidden_terminals_empty_measurement_stands_for_the_last_real_size(spawner: Spawner) -> None:
    spawner.process = FakePtyProcess((132, 30))
    shell = await _started(_shell(FakeTerminal(132, 30)))
    try:
        await shell.send("pwd", 0, 1)
        assert spawner.process.operations == [("write", "pwd\r")]

        await shell.resize(120, 40)
        # The program lost track of its size; the next resize has to tell it again.
        spawner.process.size = None
        await shell.resize(0, 1)

        assert spawner.process.operations[1:] == [("resize", 120, 40), ("resize", 120, 40)]
        assert shell.pty_size == (120, 40)
    finally:
        await shell.close()


async def test_resize_to_the_size_the_program_already_has_is_skipped(spawner: Spawner) -> None:
    shell = await _started(_shell(FakeTerminal()))
    try:
        await shell.resize(80, 24)
        assert spawner.process.operations == []
    finally:
        await shell.close()


async def test_concurrent_resizes_to_one_size_reach_the_program_once(spawner: Spawner) -> None:
    release = asyncio.Event()
    entered = asyncio.Event()

    async def slow_resize(columns: int, lines: int) -> None:
        del columns, lines
        entered.set()
        await release.wait()

    spawner.process.on_resize = slow_resize
    shell = await _started(_shell(FakeTerminal()))
    resizes = [asyncio.create_task(shell.resize(100, 30)) for _ in range(2)]
    try:
        await entered.wait()
        # The second call is parked on the lock by now, not inside the program's resize.
        await asyncio.sleep(0)
        assert spawner.process.operations == [("resize", 100, 30)]
        release.set()
        await asyncio.gather(*resizes)

        assert spawner.process.operations == [("resize", 100, 30)]
        assert shell.pty_size == (100, 30)
    finally:
        release.set()
        for resize in resizes:
            resize.cancel()
        await shell.close()


async def test_resize_before_the_program_exists_claims_no_size(spawner: Spawner) -> None:
    shell = _shell(FakeTerminal())

    await shell.resize(100, 24)

    assert shell.pty_size is None
    assert spawner.calls == []


async def test_a_size_the_program_could_not_be_told_is_not_claimed_and_is_retried(spawner: Spawner) -> None:
    spawner.process = FakePtyProcess(None)
    spawner.process.resize_succeeds = False
    shell = await _started(_shell(FakeTerminal()))
    try:
        assert shell.pty_size is None
        await shell.resize(80, 24)
        assert shell.pty_size is None

        spawner.process.resize_succeeds = True
        await shell.resize(80, 24)

        assert spawner.process.operations == [("resize", 80, 24), ("resize", 80, 24)]
        assert shell.pty_size == (80, 24)
    finally:
        await shell.close()


async def test_interrupt_sends_ctrl_c_without_resizing_first(spawner: Spawner) -> None:
    shell = await _started(_shell(FakeTerminal(100, 20)))
    try:
        spawner.process.size = (80, 24)
        await shell.interrupt()
        assert spawner.process.operations == [("write", "\x03")]
    finally:
        await shell.close()


# -- queued input --------------------------------------------------------------------------------


async def test_queued_input_that_piled_up_goes_out_as_one_write(spawner: Spawner) -> None:
    shell = await _started(_shell(FakeTerminal()))
    try:
        for key in "abc":
            await shell.write_input_queued(key)
        await wait_for(lambda: spawner.process.writes, description="queued input written")
        assert spawner.process.operations == [("write", "abc")]

        await shell.write_input_queued("d")
        await wait_for(lambda: len(spawner.process.writes) == 2, description="later input written")
        assert spawner.process.writes == ["abc", "d"]
    finally:
        await shell.close()


async def test_queued_input_tells_the_program_the_terminal_size_first(spawner: Spawner) -> None:
    terminal = FakeTerminal()
    shell = await _started(_shell(terminal))
    try:
        terminal.width, terminal.height = 100, 20
        await shell.write_input_queued("x")
        await wait_for(lambda: spawner.process.writes, description="queued input written")
        assert spawner.process.operations == [("resize", 100, 20), ("write", "x")]
        assert shell.pty_size == (100, 20)
    finally:
        await shell.close()


async def test_queued_input_waits_out_a_terminal_that_resizes_while_being_synced(spawner: Spawner) -> None:
    terminal = FakeTerminal()
    shell = await _started(_shell(terminal))

    async def resize_again(columns: int, lines: int) -> None:
        if (columns, lines) == (100, 20):
            terminal.width, terminal.height = 120, 30

    spawner.process.on_resize = resize_again
    try:
        terminal.width, terminal.height = 100, 20
        await shell.write_input_queued("x")
        await wait_for(lambda: spawner.process.writes, description="queued input written")
        assert spawner.process.operations == [("resize", 100, 20), ("resize", 120, 30), ("write", "x")]
    finally:
        await shell.close()


async def test_input_queued_before_the_shell_exists_is_dropped(spawner: Spawner) -> None:
    shell = _shell(FakeTerminal())
    await shell.write_input_queued("early")
    await _started(shell)
    try:
        await shell.write_input_queued("")
        await shell.write_input_queued("y")
        await wait_for(lambda: spawner.process.writes, description="queued input written")
        assert spawner.process.writes == ["y"]
    finally:
        await shell.close()


async def test_input_queued_after_the_shell_ended_is_dropped(spawner: Spawner) -> None:
    shell = await _started(_shell(FakeTerminal()))
    spawner.process.output.put_nowait("")
    await wait_for(lambda: shell.is_finished, description="shell finished")
    await shell.close()

    await shell.write_input_queued("late")
    await shell.write("late")
    await shell.resize(100, 30)

    assert spawner.process.operations == []


# -- terminate and close -------------------------------------------------------------------------


async def test_terminate_kills_once_and_is_not_announced_as_the_shell_exiting(spawner: Spawner) -> None:
    terminal = FakeTerminal()
    shell = await _started(_shell(terminal, shell_command="bash"))
    scratch = Path(spawner.calls[0].argv[2]).parent

    shell.terminate()
    shell.terminate()

    assert shell.is_finished
    assert spawner.process.kills == 1
    assert shell.pty_size is None
    assert not scratch.exists()
    await shell.write_input_queued("late")

    await shell.close()
    assert spawner.process.closes == 1
    assert spawner.process.operations == []
    # The owner ended it; "the shell exited" is for a shell that ends by itself.
    assert terminal.finished_announcements == 0


async def test_terminate_drops_input_that_was_still_queued(spawner: Spawner) -> None:
    release = asyncio.Event()

    async def stuck_resize(columns: int, lines: int) -> None:
        del columns, lines
        await release.wait()

    terminal = FakeTerminal()
    shell = await _started(_shell(terminal))
    spawner.process.on_resize = stuck_resize
    terminal.width = 100
    try:
        await shell.write_input_queued("x")
        await wait_for(lambda: spawner.process.operations, description="size sync started")

        shell.terminate()
        release.set()
        await shell.close()

        assert spawner.process.writes == []
    finally:
        release.set()


async def test_a_shell_terminated_before_it_ran_never_spawns(spawner: Spawner) -> None:
    terminal = FakeTerminal()
    shell = _shell(terminal)
    shell.terminate()

    await shell.run()
    await shell.wait_for_ready()

    assert shell.is_finished
    assert spawner.calls == []
    assert terminal.finished_announcements == 0


async def test_a_shell_terminated_while_spawning_kills_the_late_process(spawner: Spawner) -> None:
    spawner.gate = asyncio.Event()
    terminal = FakeTerminal()
    shell = _shell(terminal)
    shell.start()
    await wait_for(lambda: spawner.calls, description="spawn requested")

    shell.terminate()
    await shell.wait_for_ready()
    spawner.gate.set()
    await shell.close()

    assert spawner.process.kills == 1
    assert spawner.process.closes == 1
    assert terminal.stdin_writer is None
    assert shell.pty_size is None
    assert terminal.output == []


async def test_close_waits_for_the_shell_task_to_wind_down(spawner: Spawner) -> None:
    shell = await _started(_shell(FakeTerminal()))
    task = shell._task
    assert task is not None

    await shell.close()

    assert task.done() and not task.cancelled()
    assert spawner.process.closes == 1
    assert shell._task is None
    await shell.close()


async def test_close_cancels_a_shell_task_that_will_not_end(spawner: Spawner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell_module, "_CLOSE_TASK_TIMEOUT", 0.05)
    spawner.process.kill_ends_output = False
    shell = await _started(_shell(FakeTerminal()))
    task = shell._task
    assert task is not None

    await shell.close()

    assert task.cancelled()
    assert spawner.process.kills == 1
    assert spawner.process.closes == 1
    assert shell._task is None


async def test_close_from_inside_the_shell_task_does_not_wait_for_itself(spawner: Spawner) -> None:
    terminal = FakeTerminal()
    shell = _shell(terminal)

    async def close_on_output(text: str) -> None:
        if text == "quit":
            await shell.close()

    terminal.on_write = close_on_output
    await _started(shell)
    task = shell._task
    assert task is not None

    spawner.process.output.put_nowait("quit")
    await wait_for(task.done, description="shell task done")

    assert task.exception() is None
    assert shell.is_finished
    assert spawner.process.kills == 1
    assert spawner.process.closes == 1


# -- what the shell is started with --------------------------------------------------------------


async def test_the_shell_is_told_which_terminal_it_runs_in_not_which_one_we_run_in(
    spawner: Spawner, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in shell_module._OUTER_TERMINAL_VARIABLES:
        monkeypatch.setenv(name, "from-the-outer-terminal")
    for name in ("FORCE_COLOR", "TTY_COMPATIBLE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CHRYS_TEST_PASSES_THROUGH", "yes")

    shell = await _started(_shell(FakeTerminal()))
    await shell.close()

    env = spawner.calls[0].env
    assert env["TERM"] == "xterm-256color"
    assert env["COLORTERM"] == "truecolor"
    assert env["CLICOLOR"] == "1"
    assert env["CHRYS"] == "1"
    assert env["PYTHONUTF8"] == "1"
    assert env["TERM_PROGRAM"] == APP_DISPLAY_NAME
    assert env["TERM_PROGRAM_VERSION"] == __version__
    assert env["CHRYS_TEST_PASSES_THROUGH"] == "yes"
    assert "from-the-outer-terminal" not in env.values()
    # Both make programs color output that is not going to a terminal at all.
    assert "FORCE_COLOR" not in env
    assert "TTY_COMPATIBLE" not in env


def test_the_default_shell_is_the_users_own(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = FakeTerminal()
    monkeypatch.setattr(shell_module, "IS_WINDOWS", False)
    monkeypatch.setenv("SHELL", "/opt/bin/custom-sh")
    assert _shell(terminal, shell_command="").shell_command == "/opt/bin/custom-sh"
    assert _shell(terminal, shell_command="fish").shell_command == "fish"

    monkeypatch.delenv("SHELL")
    assert _shell(terminal, shell_command="").shell_command == "sh"

    monkeypatch.setattr(shell_module, "IS_WINDOWS", True)
    monkeypatch.setenv("SHELL", "/opt/bin/custom-sh")
    assert _shell(terminal, shell_command="").shell_command == "pwsh"
