# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shell execution tools — instance-based tools that hold SessionEnvironment.

Uses Chrys kernel instance tool support: @tool on class methods binds to the
instance via the descriptor protocol, giving each tool access to self._runtime
for working directory, platform info, etc.

On Unix, commands run with stdout/stderr attached to a PTY and
``TERM=dumb`` so programs see terminal output streams but suppress most ANSI
formatting.  Stdin is ``/dev/null`` and the PTY is not installed as a
controlling terminal, so interactive terminal reads fail instead of triggering
job-control stops.  On Windows, commands run with stdout/stderr pipes (no PTY
available).

Each output stream is captured within a byte limit that keeps its head and
tail (``foundation/platform/output_capture.py``), and the result is truncated
to ``max_tokens`` with a 1:2 head:tail ratio.  Every backend first assembles
the same canonical result shape; when a session directory is available, the
cleaned result is spilled there before the bounded view is returned.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import struct
import sys
import time
from collections import deque
from contextlib import AsyncExitStack, suppress
from contextvars import ContextVar
from typing import TYPE_CHECKING, Annotated, Any, cast

from chrys.foundation.platform.output_capture import BoundedCapture
from chrys.foundation.platform.paths import resolve_workspace_path
from chrys.foundation.platform.process import (
    MissingWorkingDirectoryError,
    SubprocessStoppedError,
    decode_subprocess_output,
    kill_windows_process_tree,
    managed_subprocess,
    raise_if_missing_cwd,
    signal_process_session,
    wait_for_subprocess,
)
from chrys.foundation.platform.pty_output import PtyOutputProtocol
from chrys.foundation.platform.runtime_paths import reorder_path_demoting_runtime
from chrys.foundation.platform.win_console import preserve_console_mode
from chrys.foundation.text.tool_output import (
    process_carriage_returns as _process_carriage_returns,
)
from chrys.foundation.text.tool_output import (
    strip_ansi as _strip_ansi,
)
from chrys.foundation.tool_result_metadata import (
    PARTIAL_OUTPUT_LABEL,
    SHELL_BACKGROUND_TASK_METADATA_KEY,
    SHELL_EXIT_CODE_METADATA_KEY,
    SHELL_TIMED_OUT_METADATA_KEY,
    SHELL_TIMEOUT_SECONDS_METADATA_KEY,
    TOOL_ERRORED_METADATA_KEY,
    result_text_exit_code,
    result_text_without_exit_code,
)
from chrys.foundation.util.once_close import OnceClose
from chrys.service.mutations.trace import discard_trace_file, prefix_trace_file, shell_trace_argv
from chrys.service.tools.approval_targets import (
    CHANGED_AFTER_APPROVAL,
    ApprovedTargets,
    approved_cwd,
    physical_dir,
)
from chrys.service.tools.background import current_scope, stop_command
from chrys.service.tools.kinds import KIND_SHELL, set_tool_kind, tool
from chrys.service.tools.result_metadata import record_tool_success, tool_error
from chrys.service.tools.spill import bound_process_output
from chrys.service.tools.workspace_paths import missing_base_cwd_error, working_dir_missing_error

if sys.platform != "win32":
    import fcntl
    import pty
    import select
    import signal
    import termios

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

    from chrys.foundation.models.session_env import SessionEnvironment
    from chrys.foundation.platform import ShellInfo
    from chrys.foundation.platform.output_capture import CapturedOutput
    from chrys.service.tools.background import BackgroundTask, BackgroundTasks
    from chrys.service.tools.builtins.shell_filter import ShellCommandFilter

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 8000

SHELL_REASON_DESCRIPTION = (
    "Brief user-facing explanation of why this command needs to run (1-2 sentences). "
    "Write it in the same language as the user's latest prompt whenever possible. "
    "This reason is shown to the user and recorded in audit logs, so state the command's intent plainly "
    "without private reasoning or hidden context."
)

# ---------------------------------------------------------------------------
# Shell execution context
# ---------------------------------------------------------------------------

shell_progress_callback: ContextVar[Callable[[list[str]], Awaitable[None]] | None] = ContextVar(
    "shell_progress_callback", default=None
)
"""Contextvar for streaming shell output lines to the TUI.

Set by ``ToolEventMiddleware`` before shell tool execution.  The callback
receives batches of cleaned output lines (ANSI-stripped, \\r handled).
"""

shell_result_metadata: ContextVar[dict[str, object] | None] = ContextVar("shell_result_metadata", default=None)
"""Mutable metadata buffer set by ``ToolEventMiddleware`` for shell calls."""


def _record_shell_exit_code(returncode: int) -> None:
    metadata = shell_result_metadata.get(None)
    if metadata is not None:
        metadata[SHELL_EXIT_CODE_METADATA_KEY] = returncode


def _record_shell_timeout(timeout: int) -> None:
    metadata = shell_result_metadata.get(None)
    if metadata is not None:
        metadata[SHELL_TIMED_OUT_METADATA_KEY] = True
        metadata[SHELL_TIMEOUT_SECONDS_METADATA_KEY] = timeout


def _record_shell_error() -> None:
    metadata = shell_result_metadata.get(None)
    if metadata is not None:
        metadata[TOOL_ERRORED_METADATA_KEY] = True


def _record_background_task(task: BackgroundTask) -> None:
    metadata = shell_result_metadata.get(None)
    if metadata is not None:
        metadata[SHELL_BACKGROUND_TASK_METADATA_KEY] = {
            "id": task.id,
            "pid": task.pid,
            "trigger": task.trigger,
            "log": str(task.log_path),
        }
    # Moving to the background is no failure, unlike the timeout that kills a command.
    record_tool_success()


def _working_dir_not_found(resolved: str, working_dir: str) -> str:
    return tool_error(
        "working_dir_not_found",
        f"working_dir not found — {resolved}",
        details={"working_dir": working_dir, "resolved_path": resolved},
    )


_PROGRESS_THROTTLE_INTERVAL = 0.2
"""Minimum seconds between progress emissions to avoid flooding the TUI."""

_PROGRESS_LINE_LIMIT = 64 * 1024
"""Most of an unfinished output line held for progress; its start is dropped past this."""

_PROGRESS_PENDING_LIMIT = 256 * 1024
"""Most characters of progress lines held between emissions; the oldest lines go first."""


class _ProgressLines:
    """Output lines waiting for the next progress emission, within a size limit.

    Dropping lines here only thins the live progress display; the command's
    result comes from its output capture.
    """

    def __init__(self) -> None:
        self._lines: deque[str] = deque()
        self._size = 0

    def __bool__(self) -> bool:
        return bool(self._lines)

    def add(self, line: str) -> None:
        # Each line counts its newline too, so blank lines fill the limit.
        self._lines.append(line)
        self._size += len(line) + 1
        while self._size > _PROGRESS_PENDING_LIMIT and len(self._lines) > 1:
            self._size -= len(self._lines.popleft()) + 1

    def take(self) -> list[str]:
        lines = list(self._lines)
        self._lines.clear()
        self._size = 0
        return lines


_PTY_DRAIN_GRACE = 2.0
"""Seconds to let the PTY reader catch up before the last slave fd is closed."""


async def _await_pty_master_drained(master_fd: int, *, grace: float = _PTY_DRAIN_GRACE) -> None:
    """Yield until the PTY reader has taken everything the tty still holds.

    macOS drops whatever is still buffered in a tty the moment its last slave fd
    closes, so releasing the parent's copy the instant the child exits discards
    the output of any command whose reader callback has not been scheduled yet —
    a busy event loop turns ``echo hello`` into a bare ``[exit_code: 0]``.  The
    child has already exited by the time this runs, so nothing new can arrive:
    once the master stops reporting readable, the reader holds everything that
    was written.  ``grace`` bounds the wait so a reader paused by flow control
    cannot stall the command.
    """
    # Unlike select(), poll() accepts descriptors above FD_SETSIZE (1024).
    poller = select.poll()
    poller.register(master_fd, select.POLLIN)
    deadline = time.monotonic() + grace
    while True:
        try:
            events = poller.poll(0)
            if not events or any(event & select.POLLNVAL for _, event in events):
                return
        except OSError:
            return
        if time.monotonic() >= deadline:
            return
        await asyncio.sleep(0)


def _signal_command(proc: asyncio.subprocess.Process, signum: int) -> None:
    """Send *signum* to every process of a POSIX command, never to Chrys's own process group.

    The command was spawned with ``setsid``, so it leads its own session and
    process group: the session reaches descendants that moved to a group of
    their own, the group and then the shell itself are what is left when the
    session's members cannot be listed — only while the shell lives, as its
    pid may name another process once it was reaped.
    """
    if signal_process_session(proc.pid, signum) or proc.returncode is not None:
        return
    posix_os = cast(Any, os)
    with suppress(ProcessLookupError, PermissionError, OSError):
        child_pgid = posix_os.getpgid(proc.pid)
        if child_pgid == proc.pid and child_pgid != posix_os.getpgrp():
            posix_os.killpg(child_pgid, signum)
            return
    with suppress(ProcessLookupError, OSError):
        os.kill(proc.pid, signum)


def _clean_output(text: str) -> str:
    return _process_carriage_returns(_strip_ansi(text))


def _format_partial_output(output: CapturedOutput | None, stderr: CapturedOutput | None = None) -> str:
    """Assemble what a command printed before its timeout, or ``""`` when it printed nothing."""
    parts: list[str] = []
    body = output.text(_clean_output) if output is not None else ""
    if body:
        parts.append(f"{PARTIAL_OUTPUT_LABEL}\n{body}")
    stderr_clean = stderr.text(_clean_output) if stderr is not None else ""
    if stderr_clean:
        parts.append(f"[partial stderr]\n{stderr_clean}")
    return "\n".join(parts)


_PIPEFAIL_SHELLS = frozenset({"bash", "zsh", "git_bash"})
"""Shells launched with ``-o pipefail`` so pipelines propagate upstream failures.

Without pipefail a pipeline's status is its last stage's, so
``make 2>&1 | tail -20`` reports success even when the build failed.
Deliberately conservative: a basename cannot prove capability (a ``ksh``
may be ksh88 without pipefail, ``sh`` may be an old dash), and a wrong
guess makes the shell reject its argv — every command would fail.  Shells
not listed here keep their native pipeline semantics unchanged.
"""


def _pipefail_args(shell: ShellInfo) -> tuple[str, ...]:
    """Return extra launch args enabling pipefail for shells known to support it."""
    return ("-o", "pipefail") if shell.name in _PIPEFAIL_SHELLS else ()


def _shell_argv_variants(shell: ShellInfo, command: str) -> list[tuple[str, ...]]:
    """Launch argvs to try in order: traced first (when armed), then bare.

    The mutation pipeline hands an fsatrace wrapper prefix over via
    contextvar; wrapping happens at
    argv level — no shell-quoting hazards.  Spawn failure of the traced
    variant must degrade to the bare one: tracing may never change
    user-visible shell behavior.
    """
    base = (shell.path, *_pipefail_args(shell), *shell.args, command)
    prefix = shell_trace_argv.get()
    if prefix:
        return [(*prefix, *base), base]
    return [base]


def _build_subprocess_env() -> dict[str, str]:
    """Build a colour-suppressed environment for shell subprocesses.

    Shared by both the PTY (Unix) and pipe (Windows) execution paths.
    """
    env = os.environ.copy()
    env["TERM"] = "dumb"
    env["CHRYS"] = "1"
    env["NO_COLOR"] = "1"
    # Force UTF-8 output from Python subprocesses (and other tools
    # that respect these variables) so CJK text is never garbled.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Prevent git (and other tools) from launching a pager, which
    # would block forever waiting for interactive input.
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    # Suppress colour-forcing variables
    env.pop("FORCE_COLOR", None)
    env.pop("COLORTERM", None)
    env.pop("CLICOLOR_FORCE", None)
    # Shell commands intentionally mirror the user's environment more closely
    # than managed hook/skill/MCP subprocesses, so keep PYTHONHOME/PYTHONPATH.
    reorder_path_demoting_runtime(env)
    return env


_POWERSHELL_UTF8_PREAMBLE = "\n".join(
    (
        "$__chrysUtf8 = [System.Text.UTF8Encoding]::new($false)",
        "try {",
        "  [Console]::InputEncoding = $__chrysUtf8",
        "  [Console]::OutputEncoding = $__chrysUtf8",
        "} catch { }",
        "$OutputEncoding = $__chrysUtf8",
    )
)


def _prepare_pipe_command(command: str, shell: ShellInfo) -> str:
    """Adjust commands for pipe-based shells before execution."""
    if shell.name not in ("pwsh", "powershell"):
        return command
    return f"{_POWERSHELL_UTF8_PREAMBLE}\n{command}"


_READ_CHUNK_BYTES = 64 * 1024

_STDOUT = 0
"""The stream progress lines come from: the merged output of a PTY, or stdout."""


def _trace_file(argv: Sequence[str]) -> str | None:
    """The file fsatrace writes when *argv* is the traced launch."""
    prefix = shell_trace_argv.get()
    return prefix_trace_file(prefix) if prefix and tuple(argv[: len(prefix)]) == prefix else None


class _Command:
    """A running shell command: its process, the tasks reading its output, and what they kept.

    The readers never wait on anyone. Until :meth:`hand_off` they keep the head
    and tail of each stream and queue the first one's lines for live progress;
    after it every chunk goes to the sink handed over. So a command can outlive
    the tool call that started it without ever blocking on a full pipe or PTY.
    """

    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        streams: Sequence[asyncio.StreamReader],
        cleanup: AsyncExitStack,
        *,
        after_exit: Callable[[], Awaitable[None]] | None = None,
        trace_file: str | None = None,
    ) -> None:
        self.proc = proc
        self._captures = tuple(BoundedCapture() for _ in streams)
        self._progress = _ProgressLines() if shell_progress_callback.get(None) is not None else None
        self._line = bytearray()
        self._sink: Callable[[int, bytes], None] | None = None
        self._cleanup = cleanup
        self._trace_file = trace_file
        self._close_once = OnceClose(self._close)
        loop = asyncio.get_running_loop()
        # Fresh contexts: the readers may outlive the tool call and must not carry its state.
        self._parts = [
            loop.create_task(self._read(index, stream), context=contextvars.Context())
            for index, stream in enumerate(streams)
        ]
        self._parts.append(loop.create_task(self._exit(after_exit), context=contextvars.Context()))
        self.finished = asyncio.gather(*self._parts)
        """Done once the shell exited and every output stream ended."""
        # Closing cancels the parts, which ends the gather with an error nobody reads.
        self.finished.add_done_callback(lambda finished: finished.cancelled() or finished.exception())

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def returncode(self) -> int:
        return self.proc.returncode if self.proc.returncode is not None else -1

    def captured(self) -> tuple[CapturedOutput, ...]:
        """What each stream kept so far."""
        return tuple(capture.snapshot() for capture in self._captures)

    def hand_off(self, sink: Callable[[int, bytes], None]) -> tuple[CapturedOutput, ...]:
        """Send all further output to *sink*, returning what was kept until now."""
        self._sink = sink
        self._progress = None
        if self._trace_file is not None:
            # fsatrace writes its trace when the command ends, after the tool call stopped looking.
            self._cleanup.callback(discard_trace_file, self._trace_file)
        # The command may run on for hours: let go of what the tool call kept.
        kept, self._captures = self.captured(), ()
        return kept

    async def follow(self, show: Callable[[list[str]], Awaitable[None]] | None) -> int:
        """Wait for the command to end, passing its output lines to *show* as they arrive."""
        interval = _PROGRESS_THROTTLE_INTERVAL if show is not None else None
        while True:
            done, _ = await asyncio.wait({self.finished}, timeout=interval)
            if show is not None and self._progress:
                await show(self._progress.take())
            if done:
                self.finished.result()
                return self.returncode

    async def wait(self) -> int:
        """Wait for the command to end and return its shell's exit code."""
        await asyncio.wait({self.finished})
        if not self.finished.cancelled() and self.finished.exception() is not None:
            logger.debug("Reading a background command's output failed", exc_info=self.finished.exception())
        return self.returncode

    async def terminate(self, *, force: bool) -> None:
        """Ask every process of the command to stop, or kill them all when *force* is set."""
        if sys.platform == "win32":
            # A console process tree has no polite stop here: both kill it.
            await kill_windows_process_tree(self.proc.pid)
            if self.proc.returncode is None:
                with suppress(ProcessLookupError, OSError):
                    self.proc.kill()
        else:
            _signal_command(self.proc, signal.SIGKILL if force else signal.SIGTERM)

    async def close(self) -> None:
        """Stop reading the output and release the process, killing what of the command still runs.

        Cancel-safe: the close runs to its end even when its caller is cancelled.
        """
        await self._close_once()

    async def _close(self) -> None:
        try:
            ended = self.finished.done() and not self.finished.cancelled() and self.finished.exception() is None
            if not ended:
                await self.terminate(force=True)
            for part in self._parts:
                part.cancel()
            await asyncio.wait(self._parts)
        finally:
            await self._cleanup.aclose()

    async def _read(self, stream: int, reader: asyncio.StreamReader) -> None:
        while chunk := await reader.read(_READ_CHUNK_BYTES):
            self._feed(stream, chunk)
        if stream == _STDOUT and self._progress is not None and self._line.strip():
            last = _strip_ansi(decode_subprocess_output(bytes(self._line).rstrip(b"\r\n")))
            if last:
                self._progress.add(last)

    async def _exit(self, after_exit: Callable[[], Awaitable[None]] | None) -> None:
        await self.proc.wait()
        if after_exit is not None:
            await after_exit()

    def _feed(self, stream: int, chunk: bytes) -> None:
        if self._sink is not None:
            self._sink(stream, chunk)
            return
        self._captures[stream].feed(chunk)
        if stream != _STDOUT or self._progress is None:
            return
        self._line += chunk
        if b"\n" in chunk:
            *lines, rest = self._line.split(b"\n")
            # Completed blank lines stay: the tool card shows the streamed lines once the command ends.
            for line in lines:
                self._progress.add(_strip_ansi(decode_subprocess_output(bytes(line).rstrip(b"\r"))))
            self._line = rest
        if len(self._line) > _PROGRESS_LINE_LIMIT:
            del self._line[:-_PROGRESS_LINE_LIMIT]


async def _spawn_pty(command: str, shell: ShellInfo, cwd: str) -> _Command:
    """Start *command* with its output on a PTY (Unix only)."""
    master, slave = pty.openpty()
    try:
        # Set terminal size to 80x24
        size = struct.pack("HHHH", 24, 80, 0, 0)
        fcntl.ioctl(slave, termios.TIOCSWINSZ, size)

        # Non-blocking master
        posix_os = cast(Any, os)
        flags = fcntl.fcntl(master, fcntl.F_GETFL)
        fcntl.fcntl(master, fcntl.F_SETFL, flags | posix_os.O_NONBLOCK)

        env = _build_subprocess_env()

        def _setup_pty() -> None:
            posix_os.setsid()

        # fsatrace (when armed) execs between the PTY and the shell:
        # stdio fds are inherited, ``setsid`` applies to fsatrace, and
        # the shell inherits its process group — draining and
        # group-kill behave exactly as untraced.
        argv_variants = _shell_argv_variants(shell, command)
        proc: asyncio.subprocess.Process | None = None
        for index, argv in enumerate(argv_variants):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=slave,
                    stderr=slave,
                    cwd=cwd,
                    env=env,
                    preexec_fn=_setup_pty,
                )
                break
            except OSError as exc:
                raise_if_missing_cwd(cwd, exc)
                if index == len(argv_variants) - 1:
                    raise
                logger.debug("Traced shell spawn failed; degrading to untraced", exc_info=True)
        if proc is None:
            raise RuntimeError("No shell process was created by the configured spawn variants.")
    except BaseException:
        # Clean up both fds if subprocess creation fails
        os.close(slave)
        os.close(master)
        raise

    # Keep the parent's ``slave`` fd open until the drain has finished.  On
    # macOS, if every slave fd closes (child exit + the parent closing its
    # copy) before the async reader has drained the master, the PTY can
    # reach EOF and the child's final output is silently dropped.  Holding
    # the slave open here keeps the master from reaching EOF until we close
    # it ourselves — after the child has exited and its output is read.
    slave_open = True

    def release_slave() -> None:
        nonlocal slave_open
        if slave_open:
            slave_open = False
            with suppress(OSError):
                os.close(slave)

    async def drained() -> None:
        # Once the child exits, release the parent's slave fd so the PTY
        # master reaches EOF *after* the child's buffered output has been
        # flushed.  Waiting for the reader to catch up first is what makes
        # that flush real: see _await_pty_master_drained.
        await _await_pty_master_drained(master)
        release_slave()

    cleanup = AsyncExitStack()
    cleanup.callback(release_slave)
    master_file = os.fdopen(master, "rb", 0)
    cleanup.callback(master_file.close)
    reader = asyncio.StreamReader()
    run = _Command(proc, (reader,), cleanup, after_exit=drained, trace_file=_trace_file(argv))
    try:
        # Async reader on PTY master (same pattern as TUI shell)
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: PtyOutputProtocol(reader), master_file
        )
    except BaseException:
        await run.close()
        raise
    cleanup.callback(transport.close)
    return run


async def _spawn_pipe(command: str, shell: ShellInfo, cwd: str) -> _Command:
    """Start *command* with its output on pipes (the Windows backend)."""
    env = _build_subprocess_env()
    argv_variants = _shell_argv_variants(shell, _prepare_pipe_command(command, shell))
    cleanup = AsyncExitStack()
    proc: asyncio.subprocess.Process | None = None
    for index, argv in enumerate(argv_variants):
        try:
            proc = await cleanup.enter_async_context(
                managed_subprocess(
                    *argv,
                    # stdin=DEVNULL so the child's STD_INPUT_HANDLE is not the
                    # outer chrys's console input handle on Windows.  Without this,
                    # a grandchild TUI (e.g. ``uv run icode`` inside PowerShell)
                    # can call ``SetConsoleMode`` on the inherited handle and wipe
                    # the outer TUI's mouse/keyboard flags.  Harmless on Unix.
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                )
            )
            break
        except MissingWorkingDirectoryError:
            raise
        except OSError:
            # Spawn failure surfaces at context entry, before the
            # cleanup registers — safe to retry the bare argv.
            if index == len(argv_variants) - 1:
                raise
            logger.debug("Traced shell spawn failed; degrading to untraced", exc_info=True)
    if proc is None:
        raise RuntimeError("No shell process was created by the configured spawn variants.")
    if proc.stdout is None or proc.stderr is None:
        await cleanup.aclose()
        raise RuntimeError("The shell process has no output pipes.")
    return _Command(proc, (proc.stdout, proc.stderr), cleanup, trace_file=_trace_file(argv))


def _with_stop_notes(result: str, stopped: Sequence[BackgroundTask]) -> str:
    """Say which background tasks stopped, before the exit code that must end *result*."""
    if not stopped:
        return result
    notes = [
        f"[Stopped background task {task.id}]"
        if task.ended.is_set()
        else f"[Background task {task.id} is still stopping]"
        for task in stopped
    ]
    exit_code = result_text_exit_code(result)
    if exit_code is None:
        return "\n".join((result, *notes))
    body = result_text_without_exit_code(result)
    return "\n".join((*([body] if body else []), *notes, f"[exit_code: {exit_code}]"))


SHELL_TOOL_NAMES = frozenset(
    {"bash", "zsh", "fish", "sh", "dash", "ksh", "tcsh", "csh", "pwsh", "powershell", "cmd", "git_bash"}
)
"""All possible names for shell command tools.

Shell command tools are named after their shell (e.g. ``zsh``, ``pwsh``, ``powershell``)
by ``ShellTools.tools()``.  This set covers all shells that
``platform.py`` can detect.  Used by TUI renderers for tool-specific display.
"""


class ShellTools:
    """Instance-based shell tools backed by SessionEnvironment.

    Each @tool-decorated method is a FunctionTool that binds to the instance,
    so it can access self._runtime for cwd and platform info.

    Usage::

        runtime = SessionEnvironment.capture()
        shell = ShellTools(runtime)
        tools = shell.tools()  # → [shell.<detected shell name>]
    """

    def __init__(
        self,
        runtime: SessionEnvironment,
        command_filter: ShellCommandFilter | None = None,
        shell: ShellInfo | None = None,
        session_dir: Path | None = None,
        background_tasks: BackgroundTasks | None = None,
        *,
        session_wide_background: bool = False,
    ) -> None:
        """*session_wide_background*: the agent's background commands run until the session ends, not until it finishes."""
        self._runtime = runtime
        self._filter = command_filter
        self._shell = shell or runtime.platform.shell
        self._session_dir = session_dir
        self._tasks = background_tasks
        self._session_wide_background = session_wide_background

    @property
    def shell(self) -> ShellInfo:
        """Return the shell configuration used by this tool instance."""
        return self._shell

    def approval_targets(self, arguments: dict[str, object]) -> ApprovedTargets:
        """The directory a command will run in; an unresolvable one is left to ``execute``."""
        working_dir = arguments.get("working_dir")
        if working_dir is not None and not isinstance(working_dir, str):
            return ApprovedTargets()
        try:
            return ApprovedTargets(cwd=physical_dir(self._command_cwd(working_dir)))
        except OSError, ValueError:
            return ApprovedTargets()

    def _command_cwd(self, working_dir: str | None) -> str:
        return resolve_workspace_path(working_dir, base_cwd=self._runtime.cwd) if working_dir else self._runtime.cwd

    async def _bound_result(
        self, canonical: str, budget: int, captures: Sequence[CapturedOutput] = (), *, lead: str = ""
    ) -> str:
        return await bound_process_output(self._session_dir, "shell", canonical, budget, captures, lead=lead)

    def tools(self) -> list:
        """Return all tool instances bound to this object.

        Dynamically sets descriptions to include detected shell info so the
        LLM knows which shell/version it is writing commands for.
        """
        shell = self._shell
        shell_label = f"{shell.name} {shell.version}".strip() if shell.version else shell.name
        if shell.name in ("pwsh", "powershell"):
            output_guidance = (
                "For large expected output, use PowerShell-native limiting/search commands "
                "such as Select-Object -First/-Last, Select-String, or Get-Content -Tail. "
                "To read a file's content, use the read_file tool instead of Get-Content (if present). "
            )
        elif shell.name in _PIPEFAIL_SHELLS:
            output_guidance = (
                "For large expected output, pipe through tail -n N or grep. "
                "Pipelines fail when any stage fails (pipefail is enabled by default). "
                "Avoid head and grep -q/-m in pipelines — they exit early and can kill "
                "the upstream command with SIGPIPE, often causing the pipeline to exit "
                "with status 141; if that happens, rerun without the early-exiting "
                "stage to see the real status. "
            )
        else:
            output_guidance = "For large expected output, pipe through head, tail, or grep. "

        if self._tasks is not None:
            timeout_guidance = (
                "Commands run with a timeout (default 30s). Give commands whose results or file changes "
                "you need now — installs, builds, tests, formatters, code generators — a timeout long "
                "enough for them to finish. If a command is still running when its timeout expires, it is "
                "not killed: it moves to the background and keeps running, and you get the output so far, "
                "its pid, the path of a log file that receives all further output, and the command that "
                "stops it. To start a server, watcher, or other long-running process, give it a short "
                "timeout (about 5-15 seconds) so startup errors still appear in the result; `&` and "
                "`nohup` are not needed. When a background command finishes, its log ends with a line "
                "starting with `[icode]`, such as `[icode] process exited with code N`. To wait for it, "
                "call sleep, which returns early with the end of the log when one of your background "
                "commands finishes. To stop it, run "
                "the shell command shown in the result, which creates its stop file (use the shell for "
                "this, even though files are otherwise written with write_file); this stops every process "
                "the command started, which killing the pid may not. Put commands that need the stopped "
                "command's port or files in your next response, not in the same command or the same "
                "batch of calls. "
                + (
                    "Running background commands are listed in a system reminder at the start of each "
                    "turn and are stopped when the session ends. "
                    if self._session_wide_background
                    else "Background commands you start are stopped when you finish, so do not report "
                    "one as still running. "
                )
            )
            handle_note = ""
        else:
            timeout_guidance = (
                "Commands run with a configurable timeout (default 30s). "
                "Consider a larger timeout for builds, tests, downloads, and other long-running commands. "
            )
            handle_note = "No asynchronous task handle is provided. "

        execute = self.execute
        execute.name = shell.name
        set_tool_kind(execute, KIND_SHELL)
        description = (
            f"Execute a command in {shell_label} ({shell.path}). "
            f"{timeout_guidance}"
            "This call waits for the command's result. Calls in the same response are dispatched "
            "together with no guaranteed execution order. The calling agent resumes only after the "
            "entire batch returns. Calls that depend on this result or could conflict through shared "
            "state must go in a later response. "
            f"{handle_note}"
            "Output is truncated to max_tokens (default 8000) preserving the first "
            "and last portions with a truncation notice in the middle. Very large output "
            "keeps only its beginning and end; the middle cannot be recovered. When possible, "
            "the kept output is saved under the current session directory. "
            f"{output_guidance}"
            "Set reason to a brief user-facing intent in the same language as the user's latest prompt; "
            "it is shown to the user and audit logs. "
            "Use timeout, working_dir, and max_tokens to control runtime, cwd, and output size. "
            "Do NOT use the shell to read/write/edit files or search code — "
            "use the dedicated read_file, write_file, edit_file, grep, and glob tools instead. "
            "Do NOT run interactive/TUI programs (vim, nano, less, htop, top) "
            "or commands requiring user input (passwords, y/n prompts). "
            "Use non-interactive flags when available "
            "(e.g. --yes, --no-pager, --no-input, DEBIAN_FRONTEND=noninteractive)."
        )
        platform_info = self._runtime.platform
        if shell.name == "powershell":
            description += (
                " IMPORTANT: This is Windows PowerShell 5.x, NOT bash/sh/cmd. "
                "You MUST use PowerShell syntax — do NOT write bash/Unix commands. "
                "Platform: Windows — paths use backslash (C:\\Users\\name\\project), "
                "home dir is $env:USERPROFILE, "
                "temp dir is $env:TEMP, program files at $env:ProgramFiles. "
                "Command chaining: use ';' to separate commands — '&&' and '||' "
                "are NOT supported and will cause syntax errors. "
                "For conditional execution: "
                "if ($LASTEXITCODE -eq 0) { next-command }. "
                "Environment variables: $env:VAR (NOT $VAR or %VAR%). "
                "Comparison operators: -eq, -ne, -gt, -lt, -ge, -le, -like, -match "
                "(NOT ==, !=, >, <). "
                "Logical operators: -and, -or, -not/! (NOT &&, ||). "
                "Null output: | Out-Null or > $null (NOT > /dev/null). "
                "Line continuation: backtick ` at end of line (NOT \\). "
                'Here-strings: @" must end its line, "@ must start its own line '
                "(NOT << 'EOF'). "
                "Prefer full cmdlet names over aliases: Get-ChildItem, Select-String, "
                "Remove-Item -Recurse -Force, Set-Location, Copy-Item, and Move-Item. "
                "If you must use Get-Content in Windows PowerShell 5.x, specify "
                "-Encoding UTF8; otherwise BOM-less UTF-8 can be decoded as the ANSI "
                "code page and non-ASCII text can be corrupted. "
                "Quoting: double quotes interpolate variables ($var expands), "
                "single quotes are literal. "
                "Exit codes: check $LASTEXITCODE for native commands, "
                "$? for cmdlet success. "
                "Examples: "
                "Get-ChildItem C:\\project\\src -Filter *.py -Recurse. "
                "Select-String -Path .\\*.log -Pattern 'error'. "
                "$env:PATH -split ';'. "
                "Test-Path C:\\Users\\name\\file.txt. "
                "Join-Path $env:USERPROFILE 'Documents' 'project'."
            )
        elif shell.name == "pwsh":
            if platform_info.is_windows:
                platform_note = (
                    "Platform: Windows — paths use backslash "
                    "(C:\\Users\\name\\project) or forward slash. "
                    "Home dir: $HOME (e.g. C:\\Users\\name), "
                    "temp dir: $env:TEMP, program files: $env:ProgramFiles. "
                    "Path separator: ';'."
                )
                path_examples = (
                    "Get-ChildItem C:\\project\\src -Filter *.py -Recurse. "
                    "Join-Path $HOME 'Documents' 'project'. "
                    "Test-Path C:\\Users\\name\\file.txt. "
                    "$env:PATH -split ';'."
                )
            elif platform_info.is_macos:
                platform_note = (
                    "Platform: macOS — paths use forward slash "
                    "(/Users/name/project). "
                    "Home dir: $HOME (e.g. /Users/name), "
                    "temp dir: $env:TMPDIR or /tmp. "
                    "Path separator: ':'."
                )
                path_examples = (
                    "Get-ChildItem ~/project/src -Filter *.py -Recurse. "
                    "Join-Path $HOME 'Documents' 'project'. "
                    "Test-Path /usr/local/bin/python3. "
                    "$env:PATH -split ':'."
                )
            else:
                platform_note = (
                    "Platform: Linux — paths use forward slash "
                    "(/home/name/project). "
                    "Home dir: $HOME (e.g. /home/name), "
                    "temp dir: /tmp. "
                    "Path separator: ':'."
                )
                path_examples = (
                    "Get-ChildItem ~/project/src -Filter *.py -Recurse. "
                    "Join-Path $HOME 'Documents' 'project'. "
                    "Test-Path /usr/bin/python3. "
                    "$env:PATH -split ':'."
                )
            description += (
                " IMPORTANT: This is PowerShell 7+ (cross-platform), NOT bash/sh. "
                "You MUST use PowerShell syntax — do NOT write bash/Unix commands. "
                f"{platform_note} "
                "Command chaining: '&&' and '||' are supported as pipeline chain "
                "operators (success/failure), ';' separates independent statements. "
                "Environment variables: $env:VAR (NOT $VAR or export VAR=val). "
                "Comparison operators: -eq, -ne, -gt, -lt, -ge, -le, -like, -match "
                "(NOT ==, !=, >, <). "
                "Logical operators: -and, -or, -not/! (do NOT use && or || as logical operators). "
                "Null output: | Out-Null or > $null (NOT > /dev/null). "
                "Line continuation: backtick ` at end of line (NOT \\). "
                'Here-strings: @" must end its line, "@ must start its own line '
                "(NOT << 'EOF'). "
                "Ternary: $cond ? $trueVal : $falseVal. "
                "Null-coalescing: $a ?? $default. "
                "Prefer full cmdlet names over aliases: Get-ChildItem, Select-String, "
                "Remove-Item -Recurse -Force, Set-Location, Copy-Item, and Move-Item. "
                "Quoting: double quotes interpolate variables ($var expands), "
                "single quotes are literal. "
                "Exit codes: check $LASTEXITCODE for native commands, "
                "$? for cmdlet success. "
                f"Examples: {path_examples}"
            )
        execute.description = description

        tools = [execute]
        if self._runtime.working_dirs:
            list_dirs = self.list_workspace_dirs
            set_tool_kind(list_dirs, KIND_SHELL)
            tools.append(list_dirs)
        return tools

    @tool(kind=KIND_SHELL)
    async def execute(
        self,
        command: Annotated[str, "The command to execute in the platform's default shell."],
        reason: Annotated[str, SHELL_REASON_DESCRIPTION],
        timeout: Annotated[int, "Timeout in seconds. Default: 30."] = 30,
        working_dir: Annotated[
            str | None,
            "Working directory for the command. Relative paths are resolved against the session workspace. "
            "Default: session workspace directory.",
        ] = None,
        max_tokens: Annotated[
            int,
            "Token budget for command output. Large outputs are truncated to fit.",
        ] = _DEFAULT_MAX_TOKENS,
    ) -> str:
        """Execute a shell command on the current platform."""
        budget = max(100, max_tokens)
        tasks = self._tasks
        # Validate command against filter policy (if configured); the command
        # this shell is given to stop a background task always passes.
        if self._filter is not None and not (tasks is not None and tasks.is_stop_command(command, self._shell.name)):
            result = self._filter.validate(command)
            if not result.allowed:
                _record_shell_error()
                return tool_error(
                    "command_blocked",
                    f"command blocked — {result.reason}",
                    details={"command": command, "reason": result.reason},
                )

        if working_dir:
            missing_base = missing_base_cwd_error(working_dir, self._runtime.cwd)
            if missing_base is not None:
                _record_shell_error()
                return missing_base
            cwd = self._command_cwd(working_dir)
            if not os.path.isdir(cwd):
                _record_shell_error()
                return _working_dir_not_found(cwd, working_dir)
        else:
            cwd = self._runtime.cwd
        pinned_cwd = approved_cwd()
        if pinned_cwd is not None:
            if await asyncio.to_thread(physical_dir, cwd) != pinned_cwd:
                _record_shell_error()
                return tool_error(
                    "working_dir_changed",
                    CHANGED_AFTER_APPROVAL.format(target="Working directory"),
                    details={"working_dir": cwd},
                )
            # Start in the checked directory itself: a link in the path given
            # could still be retargeted between this check and the spawn.
            cwd = pinned_cwd

        if tasks is None:
            return await self._run(command, reason, cwd, timeout, budget, working_dir)
        # A stop file may come from another tool: stop its task before a new
        # command can reach for the same port or files.
        stopped = await tasks.stop_requested()
        result = await self._run(command, reason, cwd, timeout, budget, working_dir)
        stopped += [task for task in await tasks.stop_requested() if task not in stopped]
        return _with_stop_notes(result, stopped)

    async def _run(
        self, command: str, reason: str, cwd: str, timeout: int, budget: int, working_dir: str | None
    ) -> str:
        shell = self._shell
        try:
            if sys.platform == "win32":
                # Windows: snapshot outer TUI's console mode so a child TUI
                # (e.g. the agent running ``uv run icode``) cannot leave the
                # outer chrys without mouse/keyboard input after it exits.
                # See ``preserve_console_mode`` for the full rationale.
                with preserve_console_mode():
                    stdout, returncode, stderr = await self._execute_pipe(command, shell, cwd, timeout, reason)
                body = stdout.text(_clean_output)
                stderr_clean = stderr.text(_clean_output)
                exit_suffix = f"[exit_code: {returncode}]"
                parts: list[str] = []
                if body:
                    parts.append(body)
                if stderr_clean:
                    parts.append(f"[stderr]\n{stderr_clean}")
                parts.append(exit_suffix)
                _record_shell_exit_code(returncode)
                return await self._bound_result("\n".join(parts), budget, (stdout, stderr))
            merged, returncode = await self._execute_pty(command, shell, cwd, timeout, reason)
            body = merged.text(_clean_output)
            suffix = f"[exit_code: {returncode}]"
            parts = [body, suffix] if body else [suffix]
            _record_shell_exit_code(returncode)
            return await self._bound_result("\n".join(parts), budget, (merged,))
        except _Timeout as exc:
            if exc.task is not None:
                return await self._backgrounded_result(exc.task, exc.captures, timeout, budget)
            _record_shell_timeout(timeout)
            message = f"Error: command timed out after {timeout} seconds."
            if self._tasks is not None and self._tasks.full:
                message += (
                    " The session already runs as many background commands as it can, so it was stopped"
                    " instead of moving to the background; stop one first."
                )
            partial = _format_partial_output(exc.output, exc.stderr)
            if not partial:
                return message
            return await self._bound_result(partial, budget, exc.captures, lead=f"{message}\n")
        except SubprocessStoppedError:
            _record_shell_error()
            return tool_error("process_stopped", "command entered stopped state and was terminated.")
        except MissingWorkingDirectoryError as exc:
            _record_shell_error()
            if working_dir:
                return _working_dir_not_found(exc.path, working_dir)
            return working_dir_missing_error(exc.path)
        except FileNotFoundError:
            _record_shell_error()
            return tool_error("shell_not_found", f"shell not found at {shell.path}", details={"shell_path": shell.path})
        except Exception as e:
            _record_shell_error()
            return tool_error(
                "command_execution_failed", f"failed to execute command — {e}", details={"command": command}
            )

    async def _backgrounded_result(
        self, task: BackgroundTask, captures: tuple[CapturedOutput, ...], timeout: int, budget: int
    ) -> str:
        _record_background_task(task)
        notice = (
            f"[Still running in the background: timeout of {timeout}s reached]\n"
            f"Task {task.id} (pid {task.pid}) keeps running; do not run the command again.\n"
            f'Its output goes to {task.log_path}, which ends with a line starting with "[icode]" '
            "when the command finishes.\n"
            f"To stop it, run: {stop_command(self._shell.name, task.stop_path)}"
        )
        partial = _format_partial_output(*captures)
        if not partial:
            return notice
        return f"{await self._bound_result(partial, budget, captures)}\n{notice}"

    # ------------------------------------------------------------------
    # Execution backends
    # ------------------------------------------------------------------

    async def _execute_pty(
        self, command: str, shell: ShellInfo, cwd: str, timeout: int | float, reason: str = ""
    ) -> tuple[CapturedOutput, int]:
        """Run *command* in a PTY (Unix only). Returns ``(output, returncode)``."""
        run = await _spawn_pty(command, shell, cwd)
        returncode = await self._follow(run, command, reason, timeout)
        (output,) = run.captured()
        return output, returncode

    async def _execute_pipe(
        self, command: str, shell: ShellInfo, cwd: str, timeout: int | float, reason: str = ""
    ) -> tuple[CapturedOutput, int, CapturedOutput]:
        """Run *command* with pipes (Windows fallback). Returns ``(stdout, returncode, stderr)``."""
        run = await _spawn_pipe(command, shell, cwd)
        returncode = await self._follow(run, command, reason, timeout)
        stdout, stderr = run.captured()
        return stdout, returncode, stderr

    async def _follow(self, run: _Command, command: str, reason: str, timeout: int | float) -> int:
        """Wait up to *timeout* for *run* to end and return its exit code.

        Past the timeout the command moves to the background when the session
        takes it, and is killed otherwise. Leaving any other way — an error,
        or the call being cancelled — kills it too.
        """
        handed_off = False
        try:
            # Spawned with setsid off Windows: the session and group both carry the shell's pid.
            owner = None if sys.platform == "win32" else run.pid
            try:
                return await wait_for_subprocess(
                    run.follow(shell_progress_callback.get(None)),
                    timeout=timeout,
                    process_group_id=owner,
                    process_session_id=owner,
                )
            except TimeoutError:
                captures = run.captured()
                task = (
                    self._tasks.adopt(run, command=command, reason=reason, trigger="timeout", scope=current_scope.get())
                    if self._tasks is not None
                    else None
                )
                handed_off = task is not None
                raise _Timeout(*captures, task=task) from None
        finally:
            if not handed_off:
                await run.close()

    # ------------------------------------------------------------------
    # Other tools
    # ------------------------------------------------------------------

    @tool
    def list_workspace_dirs(self) -> str:
        """List all working directories in the current workspace.

        Returns directory paths with their labels and primary status.
        Useful for understanding the project structure across multiple repositories.
        """
        if not self._runtime.working_dirs:
            return f"No workspace active. Current directory: {self._runtime.cwd}"
        lines = [f"Workspace directories (cwd: {self._runtime.cwd}):"]
        for wd in self._runtime.working_dirs:
            marker = " (primary)" if wd.is_primary else ""
            label = f" [{wd.label}]" if wd.label else ""
            exists = " ok" if wd.exists else " MISSING"
            lines.append(f"  {wd.path}{label}{marker}{exists}")
        return "\n".join(lines)


class _Timeout(Exception):
    """Internal sentinel — keeps timeout handling out of the backends.

    *task* is where the command moved when it was not killed.
    """

    def __init__(
        self,
        output: CapturedOutput | None = None,
        stderr: CapturedOutput | None = None,
        *,
        task: BackgroundTask | None = None,
    ) -> None:
        super().__init__()
        self.output = output
        self.stderr = stderr
        self.task = task

    @property
    def captures(self) -> tuple[CapturedOutput, ...]:
        return tuple(capture for capture in (self.output, self.stderr) if capture is not None)
