# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Subprocess-based skill script runner.

Implements the chrys :class:`~chrys.service.skills.model.SkillScriptRunner`
protocol using async subprocess execution, suitable for chrys's async
architecture.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chrys.foundation.platform import windows_program_files_dirs
from chrys.foundation.platform.output_capture import BoundedCapture, CapturedOutput, drain_process_pipes
from chrys.foundation.platform.process import (
    MissingWorkingDirectoryError,
    SubprocessStoppedError,
    managed_subprocess,
    wait_for_subprocess,
)
from chrys.foundation.platform.runtime_paths import (
    reorder_path_demoting_runtime,
    strip_python_runtime_overrides,
    which_excluding_runtime,
)
from chrys.foundation.platform.win_console import preserve_console_mode
from chrys.foundation.text.tool_output import process_carriage_returns, strip_ansi
from chrys.foundation.tool_result_metadata import PARTIAL_OUTPUT_LABEL
from chrys.service.skills.constants import DEFAULT_SCRIPT_RESULT_MAX_TOKENS
from chrys.service.tools.result_metadata import record_process_result, record_process_timeout, tool_error
from chrys.service.tools.spill import bound_process_output
from chrys.service.tools.workspace_paths import working_dir_missing_error

if TYPE_CHECKING:
    from chrys.foundation.models.session_env import SessionEnvironment
    from chrys.service.skills.model import Skill, SkillScript

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Interpreter resolution
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
    """Find a Python script runner: ``uv run`` first (supports PEP 723 inline script metadata), system python next, chrys' own as fallback."""
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
    """Find PowerShell: ``pwsh`` (v7+) first, ``powershell`` (v5.1) as fallback."""
    for name in ("pwsh", "powershell"):
        path = shutil.which(name)
        if path:
            return path
    return "pwsh"  # let it fail with a clear FileNotFoundError


def _find_bash() -> str:
    """Find a bash-compatible shell.

    On Windows, ``sh``/``bash`` aren't on PATH by default.  Falls back to
    Git Bash (``Git\\bin\\bash.exe``) if available, discovering the
    Program Files directories from the environment (so non-``C:`` Windows
    installs work too).
    """
    for name in ("bash", "sh"):
        path = shutil.which(name)
        if path:
            return path
    # Git Bash fallback (Windows) — honours the actual system install drive.
    for base in windows_program_files_dirs():
        git_bash = os.path.join(base, "Git", "bin", "bash.exe")
        if os.path.isfile(git_bash):
            return git_bash
    return "bash"  # let it fail with a clear FileNotFoundError


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


def _build_command(script_path: Path) -> list[str]:
    """Build the command list for a script based on its file extension."""
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


def _flag_name(key: str) -> str:
    """Return the CLI flag name for an ``args`` key."""
    return key if key.startswith("-") else f"--{key}"


def _args_to_flags(args: dict[str, Any]) -> list[str]:
    """Convert an args dict to CLI flags.

    Keys without leading dashes become ``--key``. Keys that already begin
    with ``-`` pass through verbatim, which lets callers express short and
    combined flags. Lists expand to ``--key v1 v2 ...`` (argparse
    ``nargs='+'`` style). Empty lists, ``None``, and ``False`` are all
    omitted — falsy "no value" inputs uniformly skip the flag.
    """
    flags: list[str] = []
    for key, value in args.items():
        flag = _flag_name(key)
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        elif value is None:
            continue
        elif isinstance(value, list):
            if not value:
                continue
            flags.append(flag)
            flags.extend(str(v) for v in value)
        else:
            flags.append(flag)
            flags.append(str(value))
    return flags


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------


def _clean_output(text: str) -> str:
    return process_carriage_returns(strip_ansi(text))


def _script_output(stdout: CapturedOutput, stderr: CapturedOutput, *, returncode: int | None = None) -> str:
    """Assemble a script's captured output and a nonzero exit code into one text."""
    parts: list[str] = []
    if stdout.seen:
        parts.append(stdout.text(_clean_output))
    if stderr.seen:
        parts.append(f"[stderr]\n{stderr.text(_clean_output)}")
    if returncode:
        parts.append(f"[exit_code: {returncode}]")
    return "\n".join(parts).strip()


class SubprocessScriptRunner:
    """Async skill script runner using subprocess execution.

    Satisfies the chrys :class:`~chrys.service.skills.model.SkillScriptRunner`
    protocol; ``ChrysSkillsProvider.run_skill_script`` invokes it with the
    full ``args``/``arguments``/``cwd`` signature.

    Resolves the script's absolute path from :attr:`SkillScript.full_path`,
    verifies it still lives under the owning :class:`Skill` directory, converts
    args to CLI flags, and runs via ``asyncio.create_subprocess_exec``.
    """

    def __init__(
        self,
        timeout: int = 600,
        runtime: SessionEnvironment | None = None,
        session_dir: Path | None = None,
    ) -> None:
        self._timeout = timeout
        self._runtime = runtime
        self._session_dir = session_dir

    async def __call__(
        self,
        skill: Skill,
        script: SkillScript,
        args: dict[str, Any] | list[str] | None = None,
        arguments: list[str] | None = None,
        cwd: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run a file-based skill script as a subprocess.

        Args:
            skill: The file-based skill that owns the script.
            script: The file-based script to run.  Its ``full_path`` is
                expected to be absolute (validated at discovery time and
                re-checked here before execution).
            args: Optional keyword arguments forwarded as CLI flags (--key value),
                or a legacy/list-style positional argument array. Dict-derived
                flags are appended after ``arguments``.
            arguments: Optional ordered argument tokens passed before dict-derived
                flags.
            cwd: Optional absolute working directory for the script. When
                ``None``, defaults to the primary working directory from
                ``SessionEnvironment`` (or the script's parent directory if no
                runtime is bound). Relative paths are rejected.
            max_tokens: Estimated-token budget for the returned output. Values
                below 100 are clamped to 100.

        Returns:
            Captured output, or an error message string.
        """
        if not skill.path or not str(skill.path).strip():
            return tool_error("skill_path_missing", f"Skill '{skill.name}' has no directory path.")

        if not script.full_path:
            return tool_error("script_path_missing", f"Script '{script.name}' has no resolved file path.")

        try:
            skill_dir = Path(skill.path).resolve()
            script_path = Path(script.full_path).resolve()
            script_path.relative_to(skill_dir)
        except ValueError:
            return tool_error(
                "script_path_outside_skill",
                f"Script path escapes skill directory: {script.full_path}",
                details={"skill_name": skill.name, "script_name": script.name, "script_path": script.full_path},
            )
        except OSError as e:
            return tool_error(
                "script_path_resolution_failed",
                f"Failed to resolve script path '{script.full_path}': {e}",
                details={"skill_name": skill.name, "script_name": script.name, "script_path": script.full_path},
            )

        # A project skill can live inside the deleted working directory: report
        # the directory, not its script, so the model stops instead of retrying.
        if cwd is None and self._runtime is not None and self._runtime.cwd and not os.path.isdir(self._runtime.cwd):
            return working_dir_missing_error(self._runtime.cwd)

        if not script_path.is_file():
            return tool_error(
                "script_file_not_found",
                f"Script file not found: {script_path}",
                details={"skill_name": skill.name, "script_name": script.name, "script_path": str(script_path)},
            )

        resolved_cwd = self._resolve_cwd(cwd, script_path)
        if resolved_cwd.startswith("Error: "):
            return resolved_cwd

        cmd = _build_command(script_path)
        positional_args = [str(a) for a in arguments] if arguments else []
        flag_args: list[str] = []
        if isinstance(args, list):
            positional_args = [str(a) for a in args] + positional_args
        elif isinstance(args, dict) and args:
            flag_args = _args_to_flags(args)
        if positional_args:
            cmd.extend(positional_args)
        if flag_args:
            cmd.extend(flag_args)

        # Force UTF-8 output from Python (and other interpreters that
        # respect these variables) so CJK text is never garbled.
        env = os.environ.copy()
        # Skills have no declarative env override, so keep PYTHONPATH for
        # shared helpers while still stripping PYTHONHOME to protect PyApp tools.
        strip_python_runtime_overrides(env, strip_pythonpath=False)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        reorder_path_demoting_runtime(env)

        budget = max(100, DEFAULT_SCRIPT_RESULT_MAX_TOKENS if max_tokens is None else max_tokens)
        stdout_capture = BoundedCapture()
        stderr_capture = BoundedCapture()
        try:
            # Skill scripts may invoke a TUI. ``preserve_console_mode`` guards
            # the outer chrys's Windows console mode against child mutation;
            # ``stdin=DEVNULL`` stops the child inheriting the outer console
            # input handle.  See chrys.foundation.platform.win_console for the full story.
            with preserve_console_mode():
                async with managed_subprocess(
                    *cmd,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=resolved_cwd,
                    env=env,
                ) as proc:
                    process_group_id = getattr(proc, "pid", None) if sys.platform != "win32" else None
                    await wait_for_subprocess(
                        drain_process_pipes(proc, stdout_capture, stderr_capture),
                        timeout=self._timeout,
                        process_group_id=process_group_id,
                    )

                captures = (stdout_capture.snapshot(), stderr_capture.snapshot())
                record_process_result(proc.returncode or 0)
                canonical = _script_output(*captures, returncode=proc.returncode) or "(no output)"
                return await bound_process_output(self._session_dir, "skill", canonical, budget, captures)

        except TimeoutError:
            record_process_timeout(self._timeout)
            error = tool_error(
                "script_timeout",
                f"Script '{script.name}' timed out after {self._timeout}s.",
                retryable=True,
                details={"skill_name": skill.name, "script_name": script.name, "timeout_seconds": self._timeout},
            )
            captures = (stdout_capture.snapshot(), stderr_capture.snapshot())
            partial = _script_output(*captures)
            if not partial:
                return error
            # The output stays out of the failure record.
            return await bound_process_output(
                self._session_dir, "skill", partial, budget, captures, lead=f"{error}\n{PARTIAL_OUTPUT_LABEL}\n"
            )
        except SubprocessStoppedError:
            return tool_error(
                "process_stopped",
                f"Script '{script.name}' entered stopped state and was terminated.",
                details={"skill_name": skill.name, "script_name": script.name},
            )
        except MissingWorkingDirectoryError as e:
            if cwd is None:
                return working_dir_missing_error(e.path)
            return tool_error("invalid_cwd", f"'cwd' is not an existing directory: {e.path}", details={"cwd": e.path})
        except FileNotFoundError:
            return tool_error(
                "interpreter_not_found",
                f"Interpreter not found for script '{script.name}'.",
                details={"skill_name": skill.name, "script_name": script.name},
            )
        except OSError as e:
            return tool_error(
                "script_execution_failed",
                f"Failed to execute script '{script.name}': {e}",
                details={"skill_name": skill.name, "script_name": script.name},
            )

    def _resolve_cwd(self, cwd: str | None, script_path: Path) -> str:
        """Resolve the working directory for a script run.

        Returns the cwd string on success, or an ``"Error: ..."`` message on
        failure (caller short-circuits and returns it to the agent).

        Resolution order:
            1. Explicit ``cwd`` from the agent — must be absolute and must
               exist as a directory.
            2. ``self._runtime.cwd`` if a ``SessionEnvironment`` was bound.
            3. Legacy fallback: ``script_path.parent`` (for standalone runner
               use without a runtime).
        """
        if cwd is not None:
            if not os.path.isabs(cwd):
                return tool_error("invalid_cwd", f"'cwd' must be an absolute path, got: {cwd}", details={"cwd": cwd})
            if not os.path.isdir(cwd):
                return tool_error("invalid_cwd", f"'cwd' is not an existing directory: {cwd}", details={"cwd": cwd})
            return cwd
        if self._runtime is not None:
            return self._runtime.cwd
        return str(script_path.parent)
