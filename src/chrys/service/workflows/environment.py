# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which interpreter runs a workflow, and the facts probed from it.

Three data-only objects and one owner. :func:`parse_environment_request`
reads the file's PEP 723 ``script`` block lexically into an
:class:`EnvironmentRequest`; :func:`plan_environment` turns that into a
:class:`DefaultPlan` (the chrys interpreter) or a :class:`ByoPlan` (a virtual
environment or executable the file points at) without running anything;
:class:`WorkflowEnvironmentManager.prepare` probes the planned interpreter and
is the only source of a :class:`PreparedEnvironment`, whose fingerprint pins
the interpreter, its version and platform and the SDK build, not the packages
installed in it. Every failure here is a pre-run failure: the run never
starts, and the message is written for the user.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.platform import get_platform
from chrys.service.workflows.interpreter import InterpreterError, probe_interpreter
from chrys.service.workflows.values import canonical_json

# The reference expression from the inline script metadata specification.
_METADATA_BLOCK = re.compile(r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$")
_CLAUSE = re.compile(r"^(?P<op>~=|==|!=|<=|>=|<|>)\s*(?P<version>[0-9]+(?:\.[0-9]+)*)(?P<wildcard>\.\*)?$")

DEPENDENCIES_NEED_BYO = (
    f"This version of {APP_DISPLAY_NAME} does not install workflow dependencies. Create a virtual environment with them "
    "installed and point `[tool.chrys] python` at it, or reach the tool through subprocess or an agent node."
)


class WorkflowEnvironmentError(RuntimeError):
    """The workflow cannot be given an environment; the message is user-facing and the run never starts.

    ``line`` and ``column`` (1-based, characters) place a fault in the file's ``script`` metadata block.
    """

    def __init__(self, message: str, *, line: int | None = None, column: int | None = None) -> None:
        super().__init__(message)
        self.line = line
        self.column = column


@dataclass(frozen=True, slots=True)
class EnvironmentRequest:
    """What the file declares in its ``script`` metadata block; reading it ran no user code."""

    requires_python: str | None
    dependencies: tuple[str, ...]
    python: str | None


@dataclass(frozen=True, slots=True)
class DefaultPlan:
    """Run on the chrys interpreter."""

    request: EnvironmentRequest
    interpreter: str


@dataclass(frozen=True, slots=True)
class ByoPlan:
    """Run on the interpreter the file points at; chrys starts it and installs nothing into it."""

    request: EnvironmentRequest
    interpreter: str


EnvironmentPlan = DefaultPlan | ByoPlan


@dataclass(frozen=True, slots=True)
class PreparedEnvironment:
    """Facts about the interpreter that will run the worker; the fingerprint is what ledgers and snapshots pin."""

    mode: Literal["default", "byo"]
    executable: str
    implementation: str
    python_version: str
    platform: str
    machine: str
    libc: str
    sdk_digest: str
    environment_fingerprint: str


def parse_environment_request(source: bytes) -> EnvironmentRequest:
    """Read the ``script`` metadata block; no block is an empty request, a malformed one an error."""
    text = _metadata_text(source)
    blocks = [match for match in _METADATA_BLOCK.finditer(text) if match.group("type") == "script"]
    if not blocks:
        return EnvironmentRequest(None, (), None)
    if len(blocks) > 1:
        raise WorkflowEnvironmentError(
            "The workflow file declares more than one `script` metadata block.", line=_line_of(text, blocks[1])
        )
    block_line = _line_of(text, blocks[0])
    lines = blocks[0].group("content").splitlines(keepends=True)
    content = "".join(line[2:] if line.startswith("# ") else line[1:] for line in lines)
    try:
        metadata = tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        # The block's content starts on the line after `# /// script`; each line lost its "# " (or "#") prefix.
        inside = 0 < exc.lineno <= len(lines)
        raise WorkflowEnvironmentError(
            f"The workflow file's `script` metadata is not valid TOML: {exc.msg}",
            line=block_line + exc.lineno if inside else block_line,
            column=exc.colno + (2 if lines[exc.lineno - 1].startswith("# ") else 1) if inside else None,
        ) from exc
    requires_python = _optional_string(metadata.get("requires-python"), "requires-python", block_line)
    dependencies = metadata.get("dependencies", [])
    if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
        raise WorkflowEnvironmentError("The workflow file's `dependencies` must be a list of strings.", line=block_line)
    tool = metadata.get("tool", {})
    if not isinstance(tool, dict):
        raise WorkflowEnvironmentError("The workflow file's `[tool]` must be a table.", line=block_line)
    chrys_tool = tool.get("chrys", {})
    if not isinstance(chrys_tool, dict):
        raise WorkflowEnvironmentError("The workflow file's `[tool.chrys]` must be a table.", line=block_line)
    python = _optional_string(chrys_tool.get("python"), "[tool.chrys] python", block_line)
    return EnvironmentRequest(requires_python, tuple(dependencies), python)


def metadata_block_line(source: bytes) -> int | None:
    """The line of the file's first ``# /// script`` block, if it has one."""
    text = _metadata_text(source)
    block = next((match for match in _METADATA_BLOCK.finditer(text) if match.group("type") == "script"), None)
    return _line_of(text, block) if block is not None else None


def _metadata_text(source: bytes) -> str:
    # Read as text mode would, with universal newlines: a CRLF file would otherwise hide its block from the regex.
    return source.decode("utf-8", errors="replace").removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def _line_of(text: str, block: re.Match[str]) -> int:
    return text.count("\n", 0, block.start()) + 1


def plan_environment(
    request: EnvironmentRequest, *, entry_path: Path, default_interpreter: str = sys.executable
) -> EnvironmentPlan:
    """Choose the interpreter; a relative ``[tool.chrys] python`` is taken from the workflow file's own directory."""
    if request.python is None:
        if request.dependencies:
            raise WorkflowEnvironmentError(DEPENDENCIES_NEED_BYO)
        return DefaultPlan(request, default_interpreter)
    target = entry_path.parent / request.python
    if target.is_dir():
        if not (target / "pyvenv.cfg").is_file():
            raise WorkflowEnvironmentError(
                f"`[tool.chrys] python` points at {target}, which is not a virtual environment (no pyvenv.cfg)."
            )
        executable = target / "Scripts" / "python.exe" if get_platform().is_windows else target / "bin" / "python"
        if not executable.is_file():
            raise WorkflowEnvironmentError(f"The virtual environment at {target} has no interpreter at {executable}.")
    elif target.is_file():
        executable = target
    else:
        raise WorkflowEnvironmentError(f"`[tool.chrys] python` points at {target}, which does not exist.")
    return ByoPlan(request, str(executable))


def satisfies_requires_python(version: tuple[int, ...], specifier: str) -> bool:
    """The version-specifier subset a ``requires-python`` line uses: comparison clauses, ``.*`` wildcards, ``~=``.

    Versions are compared by zero-padded release segment: a pre-release interpreter (``3.15.0rc1``) is judged as its
    final release, ``3.15.0``, by every clause.
    """
    # Every clause is read before the verdict: an unsupported one is the declaration's fault, not the interpreter's.
    satisfied = True
    for raw in specifier.split(","):
        clause = raw.strip()
        if not clause:
            continue
        match = _CLAUSE.match(clause)
        if match is None:
            raise WorkflowEnvironmentError(f"Unsupported `requires-python` clause {clause!r}.")
        operator = match.group("op")
        wanted = tuple(int(part) for part in match.group("version").split("."))
        if match.group("wildcard"):
            if operator not in {"==", "!="}:
                raise WorkflowEnvironmentError(f"Unsupported `requires-python` clause {clause!r}.")
            matched = _release_prefix(version, len(wanted)) == wanted
            if matched != (operator == "=="):
                satisfied = False
        elif operator == "~=":
            if len(wanted) < 2:
                raise WorkflowEnvironmentError(f"Unsupported `requires-python` clause {clause!r}.")
            if _compare(version, wanted) < 0 or _release_prefix(version, len(wanted) - 1) != wanted[:-1]:
                satisfied = False
        elif not _holds(operator, _compare(version, wanted)):
            satisfied = False
    return satisfied


class WorkflowEnvironmentManager:
    """The one place a :class:`PreparedEnvironment` comes from: probe, check, fingerprint. Nothing is installed."""

    def __init__(self, *, sdk_digest: str) -> None:
        self._sdk_digest = sdk_digest

    async def prepare(self, plan: EnvironmentPlan) -> PreparedEnvironment:
        mode: Literal["default", "byo"] = "byo" if isinstance(plan, ByoPlan) else "default"
        # Anchored once, before the probe, so a relative plan is never looked up on PATH and the interpreter probed
        # is the one recorded.
        executable = _anchored(plan.interpreter)
        try:
            probe = await probe_interpreter(executable)
        except InterpreterError as exc:
            raise WorkflowEnvironmentError(str(exc)) from exc
        requires = plan.request.requires_python
        if requires is not None and not satisfies_requires_python(probe.version_tuple, requires):
            raise WorkflowEnvironmentError(
                f"The {mode} interpreter {executable!r} is Python {probe.python_version}, "
                f"but the workflow requires {requires!r}."
            )
        facts: dict[str, Any] = {
            "mode": mode,
            "executable": executable,
            "implementation": probe.implementation,
            "python_version": probe.python_version,
            "platform": probe.platform,
            "machine": probe.machine,
            "libc": probe.libc,
            "sdk_digest": self._sdk_digest,
        }
        # surrogatepass: the executable is filesystem identity, and a surrogateescaped path must fingerprint, not raise.
        fingerprint = _sha256(canonical_json(facts).encode("utf-8", "surrogatepass"))
        return PreparedEnvironment(environment_fingerprint=fingerprint, **facts)


def _anchored(candidate: str) -> str:
    """The absolute path of *candidate* as the filesystem reads it, keeping the executable's own symlink.

    The directory part is made physical, so a ``..`` after a symlinked directory goes where the OS goes and not where a
    lexical collapse would; the last component is left alone, because a virtual environment's python is a symlink
    whose location selects the venv.
    """
    return os.path.join(os.path.realpath(os.path.dirname(candidate) or "."), os.path.basename(candidate))


def _optional_string(value: Any, name: str, line: int) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise WorkflowEnvironmentError(f"The workflow file's `{name}` must be a string.", line=line)


def _release_prefix(version: tuple[int, ...], width: int) -> tuple[int, ...]:
    """The first *width* release components, zero-padded: ``(3, 15)`` is ``(3, 15, 0)`` when three are wanted."""
    return (version + (0,) * width)[:width]


def _compare(actual: tuple[int, ...], wanted: tuple[int, ...]) -> int:
    width = max(len(actual), len(wanted))
    left = _release_prefix(actual, width)
    right = _release_prefix(wanted, width)
    return (left > right) - (left < right)


def _holds(operator: str, comparison: int) -> bool:
    return {
        "==": comparison == 0,
        "!=": comparison != 0,
        "<": comparison < 0,
        "<=": comparison <= 0,
        ">": comparison > 0,
        ">=": comparison >= 0,
    }[operator]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
