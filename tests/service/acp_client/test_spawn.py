# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ACP spawn environment, validation, and stderr-capture tests."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from chrys.foundation.platform.process import ManagedStdioProcess, MissingWorkingDirectoryError
from chrys.service.acp_client import AcpConfigError, AcpSpawnError
from chrys.service.acp_client.spawn import (
    DEPTH_ENV_VAR,
    STDERR_LOG_QUOTA_BYTES,
    STDERR_TAIL_BYTES,
    build_child_environment,
)

from .helpers import connected_client, make_spec


def test_child_environment_is_trimmed_casefold_deduplicated_and_depth_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/base")
    monkeypatch.setenv("UNDECLARED_SECRET", "must-not-leak")
    spec = make_spec(
        tmp_path,
        depth=2,
        extra_env={"Path": "/override", "TOKEN": "secret", "chrys_acp_subagent_depth": "99"},
    )

    env = build_child_environment(spec)

    assert env["PATH"] == "/override"
    assert sum(key.casefold() == "path" for key in env) == 1
    assert env[DEPTH_ENV_VAR] == "3"
    assert sum(key.casefold() == DEPTH_ENV_VAR.casefold() for key in env) == 1
    assert "UNDECLARED_SECRET" not in env


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("command", "bad\x00command"),
        ("cwd", "bad\x00cwd"),
    ],
)
async def test_spawn_rechecks_nul_in_command_and_cwd(tmp_path: Path, field: str, value: str) -> None:
    spec = make_spec(tmp_path)
    values = {
        "command": spec.command,
        "cwd": spec.cwd,
        "stderr_log_path": spec.stderr_log_path,
        "args": spec.args,
        "env": spec.env,
    }
    values[field] = value
    invalid = type(spec)(**values)

    from chrys.service.acp_client.spawn import spawn_acp_process

    with pytest.raises(AcpConfigError, match="NUL"):
        await spawn_acp_process(invalid)


@pytest.mark.parametrize(
    "overrides",
    [
        {"cwd": "relative"},
        {"args": ("bad\x00argument",)},
        {"env": {"TOKEN": "bad\x00value"}},
        {"handshake_timeout_seconds": True},
        {"idle_timeout_seconds": float("inf")},
        {"additional_directories": ("../outside",)},
    ],
)
async def test_spawn_rechecks_all_process_boundary_values(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    spec = make_spec(tmp_path)
    values = {
        "command": spec.command,
        "cwd": spec.cwd,
        "stderr_log_path": spec.stderr_log_path,
        "args": spec.args,
        "env": spec.env,
        "handshake_timeout_seconds": spec.handshake_timeout_seconds,
        "idle_timeout_seconds": spec.idle_timeout_seconds,
    }
    values.update(overrides)
    invalid = type(spec)(**values)

    from chrys.service.acp_client.spawn import spawn_acp_process

    with pytest.raises(AcpConfigError):
        await spawn_acp_process(invalid)


async def test_missing_executable_is_deterministic_spawn_error(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    missing = type(spec)(
        command=str(tmp_path / "missing-agent"),
        cwd=spec.cwd,
        stderr_log_path=spec.stderr_log_path,
        env=spec.env,
    )

    from chrys.service.acp_client.spawn import spawn_acp_process

    with pytest.raises(AcpSpawnError):
        await spawn_acp_process(missing)


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX spawn; the Windows spawn branch is stubbed in the process tests"
)
async def test_deleted_cwd_is_deterministic_spawn_error_naming_the_directory(tmp_path: Path) -> None:
    gone = tmp_path / "gone"
    gone.mkdir()
    gone.rmdir()
    deleted = dataclasses.replace(make_spec(tmp_path), cwd=str(gone))

    from chrys.service.acp_client.spawn import spawn_acp_process

    with pytest.raises(AcpSpawnError) as raised:
        await spawn_acp_process(deleted)

    assert raised.value.detail == f"The ACP agent's working directory no longer exists: {gone}"
    assert isinstance(raised.value.cause, MissingWorkingDirectoryError)


async def test_stderr_setup_failure_closes_transports_even_when_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.service.acp_client import spawn as spawn_mod

    spec = make_spec(tmp_path)
    closed: list[None] = []
    original_close = ManagedStdioProcess.close_transports

    def recording_close(self: ManagedStdioProcess) -> None:
        closed.append(None)
        original_close(self)

    async def cancelled_wait(self: ManagedStdioProcess) -> int:
        raise asyncio.CancelledError

    monkeypatch.setattr(ManagedStdioProcess, "close_transports", recording_close)
    monkeypatch.setattr(ManagedStdioProcess, "wait", cancelled_wait)
    monkeypatch.setattr(spawn_mod.StderrCapture, "start", Mock(side_effect=ValueError("secure log failed")))

    with pytest.raises(asyncio.CancelledError):
        await spawn_mod.spawn_acp_process(spec)

    assert closed


async def test_stderr_capture_has_quota_owner_only_log_and_tail(tmp_path: Path) -> None:
    client, _callbacks = await connected_client(tmp_path, scenario="stderr_flood")
    try:
        outcome = await client.prompt("emit stderr")
        assert outcome.stop_reason == "end_turn"
    finally:
        await client.force_close()

    log_path = make_spec(tmp_path).stderr_log_path
    assert log_path.stat().st_size == STDERR_LOG_QUOTA_BYTES
    assert len(client.stderr_tail.encode("utf-8")) <= STDERR_TAIL_BYTES
    assert client.stderr_tail == "x" * STDERR_TAIL_BYTES
    assert client._spawn is not None
    assert client._spawn.stderr.dropped_bytes == 1024 * 1024
    if os.name != "nt":
        assert log_path.stat().st_mode & 0o077 == 0


async def test_stderr_capture_securely_reopens_existing_attempt_log(tmp_path: Path) -> None:
    first, _callbacks = await connected_client(tmp_path)
    await first.force_close()

    second, _callbacks = await connected_client(tmp_path)
    await second.force_close()

    assert make_spec(tmp_path).stderr_log_path.exists()


async def test_stderr_stop_drains_buffered_bytes_before_cancelling(tmp_path: Path) -> None:
    from chrys.service.acp_client import spawn as spawn_mod

    reader = asyncio.StreamReader()
    reader.feed_data(b"last words\n")
    reader.feed_eof()
    capture = spawn_mod.StderrCapture(reader, tmp_path / "stderr.log")
    capture.start()

    # stop() immediately after start(): a cancel-first stop would kill the
    # reader task before it ever ran, discarding the child's last diagnostics.
    await capture.stop()

    assert "last words" in capture.tail
    assert (tmp_path / "stderr.log").read_bytes() == b"last words\n"
