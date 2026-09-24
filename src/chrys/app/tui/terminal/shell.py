# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The shell behind a `Terminal`: starts it, pipes the two together, and ends it."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from textual import log
from textual.message import Message

from chrys import __version__
from chrys.app.tui.terminal._pty_backend import IS_WINDOWS
from chrys.app.tui.terminal.pty_process import PtyProcess, spawn_pty_process
from chrys.app.tui.terminal.shell_integration import ShellLaunch, prepare_shell_launch
from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.platform.runtime_paths import reorder_path_demoting_runtime

if TYPE_CHECKING:
    from chrys.app.tui.terminal.widget import Terminal

_CLOSE_TASK_TIMEOUT: Final = 2.0
_DEFAULT_PTY_SIZE: Final = (80, 24)

# How the terminal we run in introduced itself to its programs. The shell is in a different
# terminal now, and a program that believed these would speak the outer one's private protocols
# to us. Apple's session variable also makes zsh announce a "restored session" on every start.
_OUTER_TERMINAL_VARIABLES: Final = (
    "TERM_PROGRAM",
    "TERM_PROGRAM_VERSION",
    "TERM_SESSION_ID",
    "ITERM_SESSION_ID",
    "KITTY_WINDOW_ID",
    "VTE_VERSION",
    "WT_SESSION",
    "WT_PROFILE_ID",
)

_SHELL_FAILED = msg("tui.terminal.shell.failed", fallback="Shell failed: {error}")
_SHELL_START_FAILED = msg("tui.terminal.shell.start_failed", fallback="Unable to start shell: {error}")


@dataclass
class ShellFinished(Message):
    """The shell process exited."""


def _shell_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in _OUTER_TERMINAL_VARIABLES:
        env.pop(name, None)
    env.update(
        TERM="xterm-256color",
        COLORTERM="truecolor",
        CLICOLOR="1",
        TERM_PROGRAM=APP_DISPLAY_NAME,
        TERM_PROGRAM_VERSION=__version__,
        # Lets a program, ours included, tell that it runs inside this app.
        CHRYS="1",
        # Output is decoded as UTF-8, which Python on Windows does not write unless told to.
        PYTHONUTF8="1",
    )
    reorder_path_demoting_runtime(env)
    return env


class Shell:
    """One shell process, shown in one `Terminal`."""

    def __init__(
        self,
        terminal: Terminal,
        working_directory: str,
        shell_command: str = "",
        on_error: Callable[[MessageRef], None] | None = None,
    ) -> None:
        self.terminal = terminal
        self.working_directory = working_directory
        self.shell_command = shell_command or ("pwsh" if IS_WINDOWS else os.environ.get("SHELL", "sh"))
        self.on_error = on_error
        self._process: PtyProcess | None = None
        self._launch: ShellLaunch | None = None
        self._task: asyncio.Task[None] | None = None
        self._finished = False
        self._ready = asyncio.Event()
        self._requested_size = _DEFAULT_PTY_SIZE
        self._resize_lock = asyncio.Lock()
        self._input: asyncio.Queue[str] | None = None
        self._input_writer: asyncio.Task[None] | None = None

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def pty_size(self) -> tuple[int, int] | None:
        """The size the shell was last told of; ``None`` while it has been told of none."""
        return None if self._process is None else self._process.size

    async def wait_for_ready(self) -> None:
        """Wait until the shell takes input, or has turned out never to."""
        await self._ready.wait()

    def start(self) -> None:
        assert self._task is None
        self._task = asyncio.create_task(self.run(), name=repr(self))

    # -- talking to the shell ----------------------------------------------------------------------

    async def send(self, command: str, width: int, height: int) -> None:
        """Run ``command`` as if typed at the prompt of a ``width`` by ``height`` terminal."""
        await self._ready.wait()
        if self._process is None:
            return
        # The size first: a prompt drawn for another width smears when the command is echoed.
        await self.resize(width, height)
        await self.write(f"{command}\r")

    async def interrupt(self) -> None:
        """Send Ctrl+C."""
        await self.write("\x03")

    async def write(self, data: str) -> None:
        """Type ``data`` into the shell at once."""
        if (process := self._process) is not None:
            await process.write(data)

    async def write_input_queued(self, data: str) -> None:
        """Type ``data`` into the shell after whatever is already waiting.

        Keys, pastes and pointer reports come faster than the shell takes them one by one. Queued,
        everything that piled up during a write goes out as the next.
        """
        if self._process is None or not data:
            return
        if self._input is None:
            self._input = asyncio.Queue()
        self._input.put_nowait(data)
        if self._input_writer is None or self._input_writer.done():
            self._input_writer = asyncio.create_task(self._write_queued_input(self._input))

    async def _write_queued_input(self, queue: asyncio.Queue[str]) -> None:
        while not self._finished:
            pending = [await queue.get()]
            # One turn of the loop lets the rest of a burst arrive.
            await asyncio.sleep(0)
            with suppress(asyncio.QueueEmpty):
                while True:
                    pending.append(queue.get_nowait())
            await self._settle_size()
            await self.write("".join(pending))

    async def _settle_size(self) -> None:
        """Have the shell know the terminal's size before input makes it redraw its prompt."""
        while True:
            size = (self.terminal.width, self.terminal.height)
            await self.resize(*size)
            if (self.terminal.width, self.terminal.height) == size:
                return

    async def resize(self, width: int, height: int) -> None:
        """Tell the shell its terminal has a new size."""
        async with self._resize_lock:
            size = self._positive_size(width, height)
            if (process := self._process) is not None and process.size != size:
                await process.resize(*size)

    def _positive_size(self, width: int, height: int) -> tuple[int, int]:
        """The size to use: a hidden widget measures 0x0, which stands for the last real size."""
        if width > 0 and height > 0:
            self._requested_size = (width, height)
        return self._requested_size

    # -- life cycle --------------------------------------------------------------------------------

    def terminate(self) -> None:
        """End the shell now. `close` also waits for it to be gone."""
        self._stop_input_writer()
        if self._finished:
            return
        if (process := self._process) is not None:
            self._process = None
            process.kill()
        self._mark_finished()

    async def close(self) -> None:
        self.terminate()
        if (task := self._task) is None:
            return
        try:
            if task is not asyncio.current_task() and not task.done():
                await asyncio.wait_for(task, timeout=_CLOSE_TASK_TIMEOUT)
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        except Exception as error:
            log(f"shell did not close cleanly: {error!r}")
        finally:
            if self._task is task:
                self._task = None

    async def run(self) -> None:
        """Start the shell and show its output until it ends."""
        try:
            await self._run()
        except Exception as error:
            log(f"shell failed: {error!r}")
            self._report(_SHELL_FAILED.bind(error=str(error)))
        finally:
            self._stop_input_writer()
            self._mark_finished(announce=True)

    async def _run(self) -> None:
        if self._finished:
            return
        env = _shell_environment()
        self._launch = launch = prepare_shell_launch(self.shell_command, env)
        columns, lines = self._positive_size(self.terminal.width, self.terminal.height)
        try:
            process = await spawn_pty_process(
                launch.argv, env=env, cwd=self.working_directory, columns=columns, lines=lines
            )
        except Exception as error:
            log(f"unable to start shell: {error!r}")
            self._report(_SHELL_START_FAILED.bind(error=str(error)))
            return
        try:
            if self._finished:
                # Terminated while it was starting.
                process.kill()
                return
            self._process = process
            self.terminal.host_repaints = process.repaints_after_resize
            self.terminal.set_write_to_stdin(self.write_input_queued)
            self._ready.set()
            while output := await process.read():
                try:
                    await self.terminal.write(output)
                except Exception as error:
                    # One chunk the terminal chokes on must not take the shell down with it.
                    log(f"terminal rejected shell output: {error!r}")
        finally:
            self._process = None
            await process.close()

    def _report(self, error: MessageRef) -> None:
        if self.on_error is not None:
            self.on_error(error)

    def _stop_input_writer(self) -> None:
        if self._input_writer is not None:
            self._input_writer.cancel()
            self._input_writer = None
        self._input = None

    def _mark_finished(self, *, announce: bool = False) -> None:
        already_finished = self._finished
        self._finished = True
        # Whoever waits for the shell to be ready should stop: it never will be.
        self._ready.set()
        if self._launch is not None:
            self._launch.clean_up()
            self._launch = None
        if announce and not already_finished:
            self.terminal.post_message(ShellFinished())
