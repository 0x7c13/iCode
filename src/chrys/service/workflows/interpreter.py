# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Version probe for the interpreter that will run a workflow worker.

The host file is compiled as a whole before any of it executes, so a too-old
interpreter fails on syntax before it could report anything. The probe is a
fixed one-liner in the lowest common syntax, run with ``-I -S`` so neither the
user's site-packages nor environment variables take part, and it only prints
the facts the floor check and the environment snapshot need.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import subprocess
from dataclasses import dataclass

from chrys.service.workflows.protocol import PYTHON_FLOOR

PROBE_TIMEOUT = 15.0  # seconds
_RELEASE_DIGITS = re.compile(r"[0-9]+")
PROBE_SOURCE = (
    "import json,platform,sys;"
    "sys.stdout.write(json.dumps({'python_version':platform.python_version(),"
    "'implementation':platform.python_implementation(),'platform':sys.platform,"
    "'machine':platform.machine(),'libc':platform.libc_ver()[0]}))"
)


class InterpreterError(RuntimeError):
    """The candidate interpreter cannot host a worker; the message is user-facing."""


@dataclass(frozen=True, slots=True)
class InterpreterProbe:
    executable: str
    python_version: str
    implementation: str
    platform: str
    machine: str
    libc: str

    @property
    def version_tuple(self) -> tuple[int, ...]:
        """The numeric release segment: a pre-release or a post-tag build keeps its patch number.

        ``3.9.1rc1`` is ``(3, 9, 1)``, its final release, and ``3.12.4+`` is ``(3, 12, 4)``.
        """
        return tuple(
            int(match.group()) for part in self.python_version.split(".")[:3] if (match := _RELEASE_DIGITS.match(part))
        )


async def _reap(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    await process.wait()


async def probe_interpreter(executable: str, *, timeout: float = PROBE_TIMEOUT) -> InterpreterProbe:
    """Run the probe and enforce the 3.9 floor; raises :class:`InterpreterError`."""
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            "-I",
            "-S",
            "-c",
            PROBE_SOURCE,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise InterpreterError(f"Cannot start interpreter {executable!r}: {exc}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.CancelledError:
        await _reap(process)
        raise
    except TimeoutError as exc:
        await _reap(process)
        raise InterpreterError(f"Interpreter {executable!r} did not answer the version probe in {timeout:g}s.") from exc
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip().splitlines()
        tail = detail[-1] if detail else f"exit status {process.returncode}"
        raise InterpreterError(f"Interpreter {executable!r} failed the version probe: {tail}")
    try:
        facts = json.loads(stdout.decode("utf-8"))
        probe = InterpreterProbe(
            executable=executable,
            python_version=str(facts["python_version"]),
            implementation=str(facts["implementation"]),
            platform=str(facts["platform"]),
            machine=str(facts["machine"]),
            libc=str(facts["libc"]),
        )
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise InterpreterError(f"Interpreter {executable!r} returned an unreadable probe result.") from exc
    if probe.version_tuple[:2] < PYTHON_FLOOR:
        floor = ".".join(str(part) for part in PYTHON_FLOOR)
        raise InterpreterError(
            f"Interpreter {executable!r} is Python {probe.python_version}; workflows need Python {floor} or newer."
        )
    return probe
