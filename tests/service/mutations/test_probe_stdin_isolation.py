# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Non-interactive probe subprocesses must never inherit the process stdin.

Under ACP the process stdin is the JSON-RPC pipe. A probe child that inherits
it can block reading protocol traffic — and swallow whatever it reads. The
mutation-layer Git probe helpers covered here (and the tracer self-test)
therefore hand their child ``stdin=DEVNULL``: the child sees EOF at once, and
the parent's stdin is left untouched. The startup-time shell/Git version
probes are covered in ``tests/foundation/platform``.

The behavioral tests below stand a ``python -c`` payload in for ``git`` (the
helpers only need an executable path, and ``python -c`` reads stdin the way a
misbehaving wrapper would), point the process stdin at a pipe holding one
JSON-RPC line, and check that the child consumed nothing and the line is still
there afterwards.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import chrys.service.mutations.trace as trace_mod
from chrys.service.mutations import git_state
from chrys.service.mutations.git_calibrator import GitDiffCalibrator
from chrys.service.mutations.trace import FSATRACE_PATH_ENV, resolve_fsatrace

_PROTOCOL_LINE = b'{"jsonrpc":"2.0","id":7,"method":"session/new","params":{}}\n'

# Reads all of stdin (blocks until EOF), then reports how much it consumed.
_READ_STDIN_AND_REPORT = "import sys; data = sys.stdin.buffer.read(); sys.stdout.write('consumed=%d' % len(data))"
_READ_STDIN_AND_REPORT_NUL = (
    "import sys; data = sys.stdin.buffer.read(); sys.stdout.buffer.write(b'consumed=%d\\0' % len(data))"
)


@contextlib.contextmanager
def _process_stdin_from(fd: int) -> Iterator[None]:
    """Make *fd* the process stdin that child processes would inherit.

    POSIX children inherit descriptor 0; Windows children inherit the
    ``STD_INPUT_HANDLE`` slot, which is what ``subprocess`` reads when no
    ``stdin`` is given.
    """
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        # Typed signatures: without them ctypes treats the HANDLE as a C int,
        # which can truncate it on 64-bit Windows and restore a bogus handle.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        kernel32.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
        kernel32.SetStdHandle.restype = wintypes.BOOL
        std_input_handle = wintypes.DWORD(-10 & 0xFFFFFFFF)
        previous = kernel32.GetStdHandle(std_input_handle)
        if not kernel32.SetStdHandle(std_input_handle, msvcrt.get_osfhandle(fd)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            if not kernel32.SetStdHandle(std_input_handle, previous):
                raise ctypes.WinError(ctypes.get_last_error())
        return
    saved = os.dup(0)
    try:
        os.dup2(fd, 0)
        yield
    finally:
        os.dup2(saved, 0)
        os.close(saved)


@contextlib.contextmanager
def _protocol_stdin() -> Iterator[int]:
    """Process stdin = a pipe holding one JSON-RPC line, write end already closed.

    Yields the read end so the test can check the line is still there once
    the probe has run: a child that inherited the pipe would have drained it.
    """
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, _PROTOCOL_LINE)
        os.close(write_fd)
        with _process_stdin_from(read_fd):
            yield read_fd
    finally:
        os.close(read_fd)


@pytest.fixture
def python_as_git(monkeypatch: pytest.MonkeyPatch) -> str:
    """Resolve ``git`` to the interpreter so ``args=["-c", code]`` runs *code*."""
    monkeypatch.setattr(git_state.shutil, "which", lambda _name: sys.executable)
    return sys.executable


def test_run_git_gives_the_child_devnull_stdin_and_leaves_the_protocol_pipe_alone(
    tmp_path: Path, python_as_git: str
) -> None:
    with _protocol_stdin() as read_fd:
        result = git_state._run_git(str(tmp_path), ["-c", _READ_STDIN_AND_REPORT], timeout=30.0)
        assert result is not None
        assert result.returncode == 0
        assert result.stdout == b"consumed=0"
        assert os.read(read_fd, len(_PROTOCOL_LINE) + 16) == _PROTOCOL_LINE


def test_run_git_nul_stream_gives_the_child_devnull_stdin_and_leaves_the_protocol_pipe_alone(
    tmp_path: Path, python_as_git: str
) -> None:
    fields: list[bytes] = []

    def _consume(field: bytes) -> bool:
        fields.append(field)
        return True

    with _protocol_stdin() as read_fd:
        result = git_state._run_git_nul_stream(
            str(tmp_path), ["-c", _READ_STDIN_AND_REPORT_NUL], timeout=30.0, consume=_consume
        )
        assert result.returncode == 0
        assert fields == [b"consumed=0"]
        assert os.read(read_fd, len(_PROTOCOL_LINE) + 16) == _PROTOCOL_LINE


def test_calibrator_namelist_gives_the_child_devnull_stdin_and_leaves_the_protocol_pipe_alone(
    tmp_path: Path, python_as_git: str
) -> None:
    calibrator = GitDiffCalibrator(str(tmp_path))
    out: set[str] = set()
    with _protocol_stdin() as read_fd:
        assert calibrator._run_git_namelist(["-c", _READ_STDIN_AND_REPORT_NUL], out) is True
        assert out == {os.path.normpath(os.path.join(calibrator._root, "consumed=0"))}
        assert os.read(read_fd, len(_PROTOCOL_LINE) + 16) == _PROTOCOL_LINE


def test_tracer_self_test_gives_the_probe_devnull_stdin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The self-test wraps ``python -c`` in an operator-supplied tracer binary;
    neither may inherit the process stdin. Captured at the call site: the
    tracer is whatever the operator points at, so no stand-in can run here."""
    seen: dict[str, Any] = {}

    def _fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen["argv"] = argv
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, returncode=1, stdout=b"", stderr=b"")

    fake_subprocess = SimpleNamespace(
        run=_fake_run,
        DEVNULL=subprocess.DEVNULL,
        PIPE=subprocess.PIPE,
        CompletedProcess=subprocess.CompletedProcess,
    )
    monkeypatch.setattr(trace_mod, "subprocess", fake_subprocess)
    monkeypatch.setattr(trace_mod, "_probe_done", False)
    monkeypatch.setattr(trace_mod, "_probe_result", None)
    fake_binary = tmp_path / "fsatrace"
    fake_binary.write_text("")
    fake_binary.chmod(0o755)
    monkeypatch.setenv(FSATRACE_PATH_ENV, str(fake_binary))

    assert resolve_fsatrace(force_probe=True) is None
    assert seen["argv"][0] == str(fake_binary)
    assert seen["stdin"] is subprocess.DEVNULL
