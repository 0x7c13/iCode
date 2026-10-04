# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Subprocess runner for hooks.

Three execution modes are dispatched here:

* ``blocking`` / ``async`` — spawn with stdio pipes via
  :func:`chrys.foundation.platform.process.managed_subprocess` so cleanup is
  guaranteed on cancellation; await the process and parse its result
  file (path passed via ``CHRYS_HOOK_RESULT``).  The async-vs-blocking
  distinction is purely about *who* awaits — :class:`HookManager`
  decides — so the runner just exposes a single ``run_and_wait`` entry
  point.
* ``fire_and_forget`` with ``detach: false`` — spawn the same way and
  return immediately; the manager observes completion for logging/outbox
  updates but does not await it at turn or session boundaries.
* ``fire_and_forget`` with ``detach: true`` — spawn via raw
  :class:`subprocess.Popen` with platform-specific detach flags
  (``setsid`` / ``DETACHED_PROCESS``).  Stdio is redirected to a
  log file under ``<config_dir>/hooks/logs/``.  Once spawned chrys
  forgets about the child entirely; it survives ``engine.shutdown()``.

Hook input/output flows through temp files:

* ``CHRYS_HOOK_PAYLOAD_FILE`` points at the JSON event payload.  We do not
  put the payload directly in the environment because tool arguments/results
  can exceed OS env limits.
* ``CHRYS_HOOK_RESULT`` points at the optional JSON decision file.  Empty /
  missing result file means "no opinion" — the manager treats it as
  ``action: allow`` with no modifications.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from chrys.foundation.platform import windows_program_files_dirs
from chrys.foundation.platform.output_capture import BoundedCapture, drain_process_pipes
from chrys.foundation.platform.process import (
    SubprocessStoppedError,
    managed_subprocess,
    wait_for_subprocess,
)
from chrys.foundation.platform.runtime_paths import (
    is_frozen_runtime,
    reorder_path_demoting_runtime,
    strip_python_runtime_overrides,
    which_excluding_runtime,
)

if TYPE_CHECKING:
    from chrys.service.hooks.outbox import OutboxJob
    from chrys.service.hooks.schema import HookConfig

logger = logging.getLogger(__name__)

HOOK_RESULT_FILE_MAX_BYTES = 1 * 1024 * 1024

HOOK_OUTPUT_CAPTURE_LIMIT_BYTES = 256 * 1024
"""Bytes kept of a waited-for hook's stdout and of its stderr; decisions come from the result file."""

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class HookResult:
    """Raw outcome of a single hook subprocess.

    The :class:`HookManager` consumes this to build a :class:`HookDecision`;
    callers outside the manager rarely look at it directly.
    """

    hook_id: str
    exit_code: int | None
    """``None`` for ``fire_and_forget`` (spawn-and-forget — we never wait)
    and for timeouts that hit the SIGKILL fallback.
    """
    timed_out: bool = False
    decision: dict[str, Any] = field(default_factory=dict)
    """Parsed JSON from ``CHRYS_HOOK_RESULT``; empty dict if the file
    was absent / empty / malformed.
    """
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0


@dataclass(frozen=True)
class HookInvocation:
    """Fully resolved subprocess invocation for one hook run."""

    cmd: list[str]
    env: dict[str, str]
    cwd: str
    result_path: Path
    payload_path: Path


# ---------------------------------------------------------------------------
# Interpreter detection (mirrors chrys/service/skills/runner.py)
# ---------------------------------------------------------------------------

_INTERPRETERS: dict[str, list[str]] = {
    ".js": ["node"],
    ".mjs": ["node"],
    ".ts": ["npx", "tsx"],
    ".rb": ["ruby"],
    ".pl": ["perl"],
}
_SHELL_SCRIPT_EXTS = frozenset({".sh", ".bash", ".zsh"})


def _find_python_runner() -> list[str]:
    runner = _find_python_runner_on_path(fallback_to_runtime=False)
    if runner is not None:
        return runner
    runner = _find_python_runner_on_path(fallback_to_runtime=True)
    if runner is not None:
        return runner
    return [sys.executable]


def _find_python_runner_on_path(*, fallback_to_runtime: bool) -> list[str] | None:
    uv = which_excluding_runtime("uv", fallback_to_runtime=fallback_to_runtime)
    if uv:
        return [uv, "run"]
    for name in ("python3", "python"):
        path = which_excluding_runtime(name, fallback_to_runtime=fallback_to_runtime)
        if path:
            return [path]
    return None


def _find_powershell() -> str:
    for name in ("pwsh", "powershell"):
        path = shutil.which(name)
        if path:
            return path
    return "pwsh"


def _find_bash() -> str:
    for name in ("bash", "sh"):
        path = shutil.which(name)
        if path:
            return path
    for base in windows_program_files_dirs():
        git_bash = os.path.join(base, "Git", "bin", "bash.exe")
        if os.path.isfile(git_bash):
            return git_bash
    return "bash"


def _build_script_command(script_path: Path) -> list[str]:
    suffix = script_path.suffix.lower()
    if suffix == ".py":
        return [*_find_python_runner(), str(script_path)]
    if suffix == ".ps1":
        return [_find_powershell(), "-NoProfile", "-File", str(script_path)]
    if suffix in _SHELL_SCRIPT_EXTS:
        return [_find_bash(), str(script_path)]
    interpreter = _INTERPRETERS.get(suffix)
    if interpreter is not None:
        return [*interpreter, str(script_path)]
    return [*_find_python_runner(), str(script_path)]


# ---------------------------------------------------------------------------
# Path / template helpers
# ---------------------------------------------------------------------------


def _expand_template(value: str, *, workspace_cwd: str, chrys_home: Path, session_id: str, profile: str) -> str:
    """Substitute the ``${var}`` placeholders documented on :class:`HookRun`."""
    if not value:
        return value
    return (
        value.replace("${workspace_cwd}", workspace_cwd)
        .replace("${chrys_home}", str(chrys_home))
        .replace("${session_id}", session_id)
        .replace("${profile}", profile)
    )


def _resolve_script_path(rel_or_abs: str, hooks_dir: Path) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (hooks_dir / p).resolve()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class HookRunner:
    """Executes hooks as subprocesses.

    One instance per :class:`HookManager`; thread-safe by virtue of
    everything running on a single event loop.  All paths derived from
    ``hooks_dir`` are resolved eagerly so the runner doesn't re-walk
    the filesystem on every dispatch.
    """

    def __init__(self, *, hooks_dir: Path) -> None:
        self._hooks_dir = hooks_dir
        self._log_dir = hooks_dir / "logs"

    @property
    def hooks_dir(self) -> Path:
        return self._hooks_dir

    async def run_and_wait(self, hook: HookConfig, payload: dict[str, Any]) -> HookResult:
        """Spawn *hook* with stdio pipes and await it (with timeout).

        Used for ``blocking`` and ``async`` modes.  The async variant
        only differs in *who* awaits — the manager does, after dispatch
        returns — so this entry point is shared.
        """
        start = datetime.now(UTC)
        result = HookResult(hook_id=hook.id, exit_code=None)
        invocation: HookInvocation | None = None
        try:
            invocation = self._build_invocation(hook, payload)
            async with managed_subprocess(
                *invocation.cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=invocation.cwd,
                env=invocation.env,
            ) as proc:
                process_group_id = getattr(proc, "pid", None) if sys.platform != "win32" else None
                stdout = BoundedCapture(HOOK_OUTPUT_CAPTURE_LIMIT_BYTES)
                stderr = BoundedCapture(HOOK_OUTPUT_CAPTURE_LIMIT_BYTES)
                await wait_for_subprocess(
                    drain_process_pipes(proc, stdout, stderr),
                    timeout=hook.execution.timeout_seconds,
                    process_group_id=process_group_id,
                )
                result.exit_code = proc.returncode
                result.stdout = stdout.snapshot().text()
                result.stderr = stderr.snapshot().text()
        except TimeoutError:
            result.timed_out = True
            logger.warning(
                "Hook %r timed out after %.1fs",
                hook.id,
                hook.execution.timeout_seconds,
            )
            return result
        except SubprocessStoppedError:
            result.timed_out = True
            result.stderr = "Hook process entered stopped state and was terminated."
            logger.warning("Hook %r entered stopped state and was terminated", hook.id)
            return result
        except (FileNotFoundError, OSError, ValueError) as exc:
            result.exit_code = None
            result.stderr = str(exc)
        finally:
            if invocation is not None:
                result.decision = _read_result_file(invocation.result_path)
                _cleanup_invocation_files(invocation)
            result.duration_ms = int((datetime.now(UTC) - start).total_seconds() * 1000)
        return result

    # ── internal ─────────────────────────────────────────────────────

    def _build_invocation(
        self,
        hook: HookConfig,
        payload: dict[str, Any],
    ) -> HookInvocation:
        """Build the (argv, env, cwd) triple for a hook invocation."""
        workspace_cwd = payload.get("cwd") or os.getcwd()
        session_id = payload.get("session_id") or ""
        profile = payload.get("profile") or ""
        chrys_home = self._hooks_dir.parent

        run = hook.run
        if run.type == "script":
            script = _resolve_script_path(run.path, self._hooks_dir)
            cmd = [*_build_script_command(script), *run.args]
        elif run.type == "command":
            cmd = [*run.argv, *run.args]
        else:  # shell
            cmd = self._build_shell_command(run.shell)

        cwd = (
            _expand_template(
                run.cwd,
                workspace_cwd=workspace_cwd,
                chrys_home=chrys_home,
                session_id=session_id,
                profile=profile,
            )
            or workspace_cwd
        )

        env = os.environ.copy()
        strip_python_runtime_overrides(env)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["CHRYS_HOOK_ID"] = hook.id
        env["CHRYS_HOOK_EVENT"] = str(payload.get("event", ""))
        tmp_dir = self._hooks_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        payload_fd = -1
        result_fd = -1
        payload_path_obj: Path | None = None
        result_path_obj: Path | None = None
        try:
            # Payload file: hooks read event data here.  Avoid putting large
            # tool args/results in the environment.
            payload_fd, payload_path = tempfile.mkstemp(
                prefix=f"{_safe_temp_prefix(hook.id)}-payload-",
                suffix=".json",
                dir=tmp_dir,
            )
            payload_path_obj = Path(payload_path)
            _set_owner_only(payload_fd)
            with os.fdopen(payload_fd, "w", encoding="utf-8") as f:
                payload_fd = -1
                json.dump(payload, f, default=str)
            env["CHRYS_HOOK_PAYLOAD_FILE"] = payload_path

            # Result file: hooks write their decision here so stdout/stderr
            # are free for logs.  The runner deletes the file after read.
            result_fd, result_path = tempfile.mkstemp(
                prefix=f"{_safe_temp_prefix(hook.id)}-result-",
                suffix=".json",
                dir=tmp_dir,
            )
            result_path_obj = Path(result_path)
            _set_owner_only(result_fd)
            os.close(result_fd)
            result_fd = -1
            env["CHRYS_HOOK_RESULT"] = result_path
        except BaseException:
            if payload_fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(payload_fd)
            if result_fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(result_fd)
            for cleanup_path in (payload_path_obj, result_path_obj):
                if cleanup_path is not None:
                    with contextlib.suppress(OSError):
                        cleanup_path.unlink(missing_ok=True)
            raise
        if result_path_obj is None or payload_path_obj is None:
            msg = "failed to create hook payload/result temp files"
            raise RuntimeError(msg)

        for k, v in run.env.items():
            env[k] = _expand_template(
                v,
                workspace_cwd=workspace_cwd,
                chrys_home=chrys_home,
                session_id=session_id,
                profile=profile,
            )
        reorder_path_demoting_runtime(env)
        return HookInvocation(
            cmd=cmd,
            env=env,
            cwd=cwd,
            result_path=result_path_obj,
            payload_path=payload_path_obj,
        )

    def _build_shell_command(self, shell_str: str) -> list[str]:
        """Wrap a shell-syntax string in the user's shell.

        On POSIX we use ``$SHELL`` (fallback ``/bin/sh``); on Windows we
        use ``cmd.exe /c``.  Users who need a specific shell should pick
        ``type: command`` + an explicit argv.
        """
        if sys.platform == "win32":
            return ["cmd.exe", "/c", shell_str]
        shell = os.environ.get("SHELL") or "/bin/sh"
        return [shell, "-c", shell_str]

    async def spawn_detached(
        self,
        hook: HookConfig,
        payload: dict[str, Any],
        *,
        job: OutboxJob | None = None,
    ) -> None:
        """Spawn a detached worker, redirecting stdio to a log file.

        The worker runs the real hook command, updates durable outbox state,
        and cleans temp files.  On POSIX the worker starts a new session; on
        Windows it starts in a detached process group.  Either way it outlives
        the parent Chrys process.
        """
        invocation = self._build_invocation(hook, payload)
        invocation_path: Path | None = None
        log_file: Any | None = None

        try:
            invocation_path = self._write_detached_invocation(invocation, hook_id=hook.id, job=job)
            log_path = self._log_path_for(hook, payload)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = _open_owner_only_append(log_path)
            worker_env = os.environ.copy()
            strip_python_runtime_overrides(worker_env)
            if is_frozen_runtime(worker_env):
                worker_cmd = [sys.executable, "-s", "-m", "chrys.service.hooks.detached_worker", str(invocation_path)]
                worker_env["PYTHONNOUSERSITE"] = "1"
            else:
                worker_cmd = [sys.executable, "-m", "chrys.service.hooks.detached_worker", str(invocation_path)]
            worker_env["PYTHONUTF8"] = "1"
            worker_env["PYTHONIOENCODING"] = "utf-8"

            kwargs: dict[str, Any] = {
                "stdin": subprocess.DEVNULL,
                "stdout": log_file,
                "stderr": log_file,
                "cwd": str(self._hooks_dir),
                "env": worker_env,
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True

            await asyncio.to_thread(subprocess.Popen, worker_cmd, **kwargs)
        except BaseException:
            _cleanup_invocation_files(invocation)
            if invocation_path is not None:
                with contextlib.suppress(OSError):
                    invocation_path.unlink(missing_ok=True)
            raise
        finally:
            # The worker has dup'd the FD; closing here doesn't affect it.
            if log_file is not None:
                with contextlib.suppress(OSError):
                    log_file.close()

    def _write_detached_invocation(
        self,
        invocation: HookInvocation,
        *,
        hook_id: str,
        job: OutboxJob | None,
    ) -> Path:
        tmp_dir = self._hooks_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(
            prefix=f"{_safe_temp_prefix(hook_id)}-detached-",
            suffix=".json",
            dir=tmp_dir,
        )
        _set_owner_only(fd)
        path = Path(raw_path)
        data = {
            "cmd": invocation.cmd,
            "env": invocation.env,
            "cwd": invocation.cwd,
            "result_path": str(invocation.result_path),
            "payload_path": str(invocation.payload_path),
            "outbox_root": str(self._hooks_dir / "outbox") if job is not None else "",
            "job": job.__dict__ if job is not None else None,
        }
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = -1
                json.dump(data, f)
        except BaseException:
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            raise
        return path

    def _log_path_for(self, hook: HookConfig, payload: dict[str, Any]) -> Path:
        session_id = _safe_temp_prefix(str(payload.get("session_id") or "no-session"))
        hook_id = _safe_temp_prefix(hook.id)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return self._log_dir / session_id / f"{hook_id}.{ts}.log"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_result_file(path: Path) -> dict[str, Any]:
    """Parse ``CHRYS_HOOK_RESULT``; return ``{}`` on absent/empty/invalid."""
    try:
        with path.open("rb") as result_file:
            raw = result_file.read(HOOK_RESULT_FILE_MAX_BYTES + 1)
    except OSError:
        return {}
    if len(raw) > HOOK_RESULT_FILE_MAX_BYTES:
        logger.warning(
            "Hook result file %s exceeded the %d-byte limit and was ignored",
            path,
            HOOK_RESULT_FILE_MAX_BYTES,
        )
        return {}
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return {}
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Hook result file %s contained invalid JSON: %s", path, exc)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("Hook result file %s must be a JSON object, got %s", path, type(parsed).__name__)
        return {}
    return parsed


def _cleanup_invocation_files(invocation: HookInvocation) -> None:
    """Remove temp payload/result files for a completed non-detached invocation."""
    for path in (invocation.result_path, invocation.payload_path):
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


def _safe_temp_prefix(value: str) -> str:
    """Return a filesystem-safe prefix for temp files derived from hook ids."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value) or "hook"


def _set_owner_only(fd: int) -> None:
    """Best-effort 0600 permissions for hook temp files that may carry secrets."""
    try:
        os.fchmod(fd, 0o600)
    except AttributeError, OSError:
        return


def _open_owner_only_append(path: Path) -> BinaryIO:
    """Open *path* for append with best-effort owner-only permissions."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(path, flags, 0o600)
    try:
        _set_owner_only(fd)
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
        return os.fdopen(fd, "ab")
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
