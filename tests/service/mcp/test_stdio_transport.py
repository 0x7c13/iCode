# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the stdio MCP transport: tolerant_stdio_client, _SafeStdioTool, inherited environment, and diagnostics."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import anyio
import pytest

from chrys.foundation.platform import runtime_paths
from chrys.service.mcp._stdio_transport import (
    MAX_STDIO_PATH_DISPLAY_CHARS,
    MAX_STDIO_STDERR_BYTES_CAPTURED,
    MAX_STDIO_STDERR_PREVIEW_LINES,
    _display_path,
    _inherited_stdio_environment,
    _resolve_spawn_executable,
    _SafeStdioTool,
    _StdioProcessDiagnostics,
    tolerant_stdio_client,
)
from chrys.service.mcp.errors import MCPConnectionError
from chrys.service.mcp.owned import MCPStdioTool
from tests.service.mcp._helpers import block_import

# ---------------------------------------------------------------------------
# _SafeStdioTool — devnull errlog and client kwargs forwarding
# ---------------------------------------------------------------------------


class TestSafeStdioToolErrlog:
    """Verify ``_SafeStdioTool`` keeps a safe stock-client fallback errlog.

    Upstream's default of ``errlog=sys.stderr`` has two failure modes that
    an earlier ``_safe_errlog`` probe could not cover reliably:

    1. Under the Chrys TUI on Python 3.14+, ``sys.stderr.fileno()`` may
       return an fd that *passes* the probe but is not usable for subprocess
       inheritance — ``anyio.open_process`` then raises
       ``OSError(9, "Bad file descriptor")`` and the server never starts.
    2. Even when the fd IS valid, inheriting it corrupts the Chrys TUI frame.

    The custom transport captures stderr through a pipe.  The stock-client
    compatibility fallback still needs a fresh ``os.devnull`` handle so an
    SDK-private API change cannot reintroduce the TUI descriptor issue.
    """

    def test_devnull_errlog_returns_open_file(self) -> None:
        """``_devnull_errlog`` returns a live, writable handle to devnull."""
        tool = _SafeStdioTool(name="t", command="python")
        result = tool._devnull_errlog()

        assert hasattr(result, "write")
        assert hasattr(result, "fileno")
        assert not result.closed
        # Must be a real OS fd so subprocess can inherit it.
        assert result.fileno() >= 0
        assert result is not sys.stderr
        result.close()

    def test_devnull_errlog_ignores_stderr_state(self) -> None:
        """A broken ``sys.stderr`` must not affect the result — we never probe it."""
        tool = _SafeStdioTool(name="t", command="python")

        class _BrokenStderr:
            def fileno(self) -> int:
                raise OSError(9, "Bad file descriptor")

        with patch.object(sys, "stderr", _BrokenStderr()):
            result = tool._devnull_errlog()

        assert result is not sys.stderr
        assert not result.closed
        assert result.fileno() >= 0
        result.close()

    def test_reuses_devnull_handle(self) -> None:
        """Multiple calls return the same cached open handle."""
        tool = _SafeStdioTool(name="t", command="python")

        first = tool._devnull_errlog()
        second = tool._devnull_errlog()

        assert first is second
        assert not first.closed
        first.close()

    def test_reopens_if_previous_handle_closed(self) -> None:
        """If the cached devnull handle was closed, a new one is opened."""
        tool = _SafeStdioTool(name="t", command="python")

        first = tool._devnull_errlog()
        first.close()
        second = tool._devnull_errlog()

        assert second is not first
        assert not second.closed
        second.close()

    def test_get_mcp_client_passes_devnull_errlog(self) -> None:
        """``get_mcp_client`` passes the cached devnull handle to the tolerant client."""
        tool = _SafeStdioTool(name="t", command="python")

        with patch("chrys.service.mcp._stdio_transport.tolerant_stdio_client") as mock_client:
            mock_client.return_value = AsyncMock()
            tool.get_mcp_client()
            mock_client.assert_called_once()
            _, kwargs = mock_client.call_args
            assert "errlog" in kwargs
            # The errlog is the cached handle, not sys.stderr.
            assert kwargs["errlog"] is tool._errlog_file
            assert kwargs["errlog"] is not sys.stderr
            # And the dropped-banner buffer travels along.
            assert kwargs["dropped_banner_lines"] is tool.dropped_banner_lines
            assert kwargs["process_diagnostics"] is tool._process_diagnostics
        # Clean up the lazily-opened handle so pytest doesn't warn.
        if tool._errlog_file is not None:
            tool._errlog_file.close()

    def test_get_mcp_client_forwards_encoding_and_client_kwargs(self) -> None:
        """Encoding plus framework client kwargs must reach StdioServerParameters."""
        tool = _SafeStdioTool(name="t", command="python", encoding="utf-16", cwd="/tmp/chrys-mcp")

        with patch("chrys.service.mcp._stdio_transport.tolerant_stdio_client") as mock_client:
            mock_client.return_value = AsyncMock()
            tool.get_mcp_client()

        _, kwargs = mock_client.call_args
        server = kwargs["server"]
        assert server.encoding == "utf-16"
        assert server.cwd == "/tmp/chrys-mcp"
        if tool._errlog_file is not None:
            tool._errlog_file.close()

    async def test_aexit_closes_devnull_file(self) -> None:
        """``__aexit__`` cleans up the devnull file handle."""
        tool = _SafeStdioTool(name="t", command="python")
        tool._errlog_file = open(os.devnull, "w")  # noqa: SIM115, ASYNC230

        with patch.object(MCPStdioTool, "__aexit__", new=AsyncMock()):
            await tool.__aexit__(None, None, None)

        assert tool._errlog_file is None

    async def test_aexit_without_errlog_file(self) -> None:
        """``__aexit__`` succeeds when no devnull file was opened."""
        tool = _SafeStdioTool(name="t", command="python")

        with patch.object(MCPStdioTool, "__aexit__", new=AsyncMock()):
            await tool.__aexit__(None, None, None)

        assert tool._errlog_file is None


# ---------------------------------------------------------------------------
# MCPConnectionError — stdio diagnostics in the message
# ---------------------------------------------------------------------------


class TestMCPConnectionErrorMessage:
    """Verify bounded stdio process diagnostics surface in the error string."""

    def test_no_banner_lines_keeps_message_terse(self) -> None:
        err = MCPConnectionError("srv", "stdio", RuntimeError("boom"))
        assert err.banner_lines == []
        assert "boom" in str(err)
        assert "non-JSON" not in str(err)

    def test_empty_cause_message_uses_exception_label(self) -> None:
        class ReadTimeout(Exception):
            pass

        err = MCPConnectionError("srv", "http", ReadTimeout(TimeoutError()))
        assert "Read timed out (ReadTimeout)" in str(err)

    def test_banner_lines_appear_in_message(self) -> None:
        err = MCPConnectionError(
            "srv",
            "stdio",
            TimeoutError("timed out"),
            banner_lines=["FooBarServer v1.0", "Initializing..."],
        )
        text = str(err)
        assert "non-JSON output before initialization" in text
        assert "FooBarServer v1.0" in text
        assert "Initializing..." in text

    def test_banner_lines_default_empty_and_independent(self) -> None:
        # Mutating the input list after construction should not affect the error.
        lines = ["a"]
        err = MCPConnectionError("srv", "stdio", RuntimeError("x"), banner_lines=lines)
        lines.append("b")
        assert err.banner_lines == ["a"]

    def test_stderr_and_process_exit_code_appear_in_message(self) -> None:
        err = MCPConnectionError(
            "srv",
            "stdio",
            RuntimeError("Connection closed"),
            stderr_tail="Traceback...\nModuleNotFoundError: missing_pkg",
            stderr_dropped_bytes=128,
            process_exit_code=23,
        )

        text = str(err)
        assert err.process_exit_code == 23
        assert err.stderr_dropped_bytes == 128
        assert "Server process exit code: 23" in text
        assert "128 earlier bytes omitted" in text
        assert "ModuleNotFoundError: missing_pkg" in text

    def test_stderr_is_sanitized_and_bounded(self) -> None:
        err = MCPConnectionError(
            "srv",
            "stdio",
            stderr_tail="\x1b[31m" + ("x" * (MAX_STDIO_STDERR_BYTES_CAPTURED + 10)) + "\x1b[0m",
        )

        assert "\x1b" not in err.stderr_tail
        assert len(err.stderr_tail) == MAX_STDIO_STDERR_BYTES_CAPTURED

    def test_message_previews_only_the_last_stderr_lines(self) -> None:
        """The message (which feeds a toast) shows a few lines; the attribute keeps the whole tail."""
        lines = [f"line {i}" for i in range(MAX_STDIO_STDERR_PREVIEW_LINES + 5)]
        err = MCPConnectionError("srv", "stdio", stderr_tail="\n".join(lines))

        text = str(err)
        assert err.stderr_tail == "\n".join(lines)
        assert "line 0" not in text
        assert f"line {MAX_STDIO_STDERR_PREVIEW_LINES + 4}" in text
        assert "5 earlier lines omitted" in text
        assert text.count("\n  line ") == MAX_STDIO_STDERR_PREVIEW_LINES

    def test_message_caps_each_previewed_stderr_line(self) -> None:
        err = MCPConnectionError("srv", "stdio", stderr_tail="y" * 500)

        assert "y" * 500 not in str(err)
        assert ("y" * 200 + "…") in str(err)
        assert err.stderr_tail == "y" * 500

    def test_exit_code_zero_is_explained_as_early_normal_exit(self) -> None:
        err = MCPConnectionError("srv", "stdio", RuntimeError("Connection closed"), process_exit_code=0)

        assert err.process_exit_code == 0
        assert "exited normally (code 0) before completing the MCP handshake" in str(err)
        assert "exit code: 0" not in str(err)

    def test_executable_and_working_directory_appear_in_message(self) -> None:
        err = MCPConnectionError(
            "srv",
            "stdio",
            RuntimeError("Connection closed"),
            resolved_executable="/opt/homebrew/bin/uv",
            effective_cwd="/workspace/foo",
        )

        text = str(err)
        assert err.resolved_executable == "/opt/homebrew/bin/uv"
        assert err.effective_cwd == "/workspace/foo"
        assert "Executable: /opt/homebrew/bin/uv" in text
        assert "Working directory: /workspace/foo" in text

    def test_context_paths_are_display_safe_but_attributes_stay_raw(self) -> None:
        # Lone surrogate (os.fsdecode of undecodable bytes) + control chars that
        # could forge extra diagnostic lines + an oversized path.
        raw_exe = "/opt/bin/\udcff-uv\nExecutable: /forged"
        raw_cwd = "/w/" + "x" * (MAX_STDIO_PATH_DISPLAY_CHARS + 20)
        err = MCPConnectionError(
            "srv", "stdio", RuntimeError("Connection closed"), resolved_executable=raw_exe, effective_cwd=raw_cwd
        )

        text = str(err)
        text.encode("utf-8")  # message must be UTF-8 encodable end to end
        assert err.resolved_executable == raw_exe
        assert err.effective_cwd == raw_cwd
        assert "\udcff" not in text
        assert "\nExecutable: /forged" not in text  # embedded newline cannot forge a second line
        assert sum(line.startswith("Executable:") for line in text.splitlines()) == 1
        assert raw_cwd not in text
        assert "Working directory: /w/" + "x" * (MAX_STDIO_PATH_DISPLAY_CHARS - 4) + "…" in text

    def test_context_lines_are_omitted_when_unknown(self) -> None:
        text = str(MCPConnectionError("srv", "http", RuntimeError("boom")))

        assert "Executable:" not in text
        assert "Working directory:" not in text


# ---------------------------------------------------------------------------
# _inherited_stdio_environment — sanitized parent env passthrough for stdio MCP servers
# ---------------------------------------------------------------------------


class TestInheritedStdioEnvironment:
    """Stdio MCP subprocesses must inherit the user's sanitized env by default.

    The MCP SDK's ``get_default_environment()`` is a strict ~6-var
    allowlist that strips ``HTTPS_PROXY`` / ``NO_PROXY`` / ``SSL_CERT_FILE``
    / etc. — fine for an untrusted-server sandbox, wrong for a developer
    tool where ``uvx some-pkg`` is expected to use the same proxy that
    works in the user's shell.  Chrys forwards nearly all of ``os.environ``
    (minus bash function exports and Python runtime path overrides) and
    merges per-server overrides on top.
    """

    def test_inherits_proxy_and_tls_vars(self) -> None:
        """Vars stripped by the SDK allowlist must reach the child env."""
        with patch.dict(
            os.environ,
            {
                "HTTPS_PROXY": "http://proxy.corp.example:8080",
                "NO_PROXY": "localhost,.corp.example",
                "SSL_CERT_FILE": "/etc/ssl/corp.pem",
            },
            clear=False,
        ):
            env = _inherited_stdio_environment()

        assert env["HTTPS_PROXY"] == "http://proxy.corp.example:8080"
        assert env["NO_PROXY"] == "localhost,.corp.example"
        assert env["SSL_CERT_FILE"] == "/etc/ssl/corp.pem"

    def test_mirrors_parent_uppercase_no_proxy_to_lowercase(self) -> None:
        """Stdio children get lowercase ``no_proxy`` even when parent stores ``NO_PROXY``."""
        with patch.dict(os.environ, {"NO_PROXY": "localhost,.corp.example"}, clear=True):
            env = _inherited_stdio_environment()

        assert env["NO_PROXY"] == "localhost,.corp.example"
        assert env["no_proxy"] == "localhost,.corp.example"

    def test_mirrors_parent_lowercase_no_proxy_to_uppercase(self) -> None:
        """Stdio children get uppercase ``NO_PROXY`` even when parent stores ``no_proxy``."""
        with patch.dict(os.environ, {"no_proxy": "localhost,.corp.example"}, clear=True):
            env = _inherited_stdio_environment()

        assert env["NO_PROXY"] == "localhost,.corp.example"
        assert env["no_proxy"] == "localhost,.corp.example"

    def test_extra_overrides_inherited(self) -> None:
        """Per-server ``env`` wins on conflict with the parent process."""
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://parent:1"}, clear=False):
            env = _inherited_stdio_environment({"HTTPS_PROXY": "http://override:2"})

        assert env["HTTPS_PROXY"] == "http://override:2"

    def test_extra_uppercase_no_proxy_overrides_inherited_lowercase_alias(self) -> None:
        """A per-server ``NO_PROXY`` override also replaces inherited lowercase alias."""
        with patch.dict(os.environ, {"no_proxy": "stale-parent.example"}, clear=True):
            env = _inherited_stdio_environment({"NO_PROXY": "localhost"})

        assert env["NO_PROXY"] == "localhost"
        assert env["no_proxy"] == "localhost"

    def test_extra_lowercase_no_proxy_overrides_inherited_uppercase_alias(self) -> None:
        """A per-server ``no_proxy`` override also replaces inherited uppercase alias."""
        with patch.dict(os.environ, {"NO_PROXY": "stale-parent.example"}, clear=True):
            env = _inherited_stdio_environment({"no_proxy": "localhost"})

        assert env["NO_PROXY"] == "localhost"
        assert env["no_proxy"] == "localhost"

    def test_extra_adds_new_vars(self) -> None:
        """Extra entries not in the parent env appear in the result."""
        env = _inherited_stdio_environment({"CHRYS_TEST_NEW_VAR": "value"})
        assert env["CHRYS_TEST_NEW_VAR"] == "value"

    def test_strips_inherited_python_runtime_overrides(self) -> None:
        """Parent Python runtime path overrides must not leak into stdio MCP children."""
        with patch.dict(
            os.environ,
            {
                "PYTHONHOME": "/bad/home",
                "PYTHONPATH": "/bad/path",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
            },
            clear=True,
        ):
            env = _inherited_stdio_environment()

        env_keys = {key.upper() for key in env}
        assert "PYTHONHOME" not in env_keys
        assert "PYTHONPATH" not in env_keys
        assert env["PYTHONUTF8"] == "1"
        assert env["PYTHONIOENCODING"] == "utf-8"

    def test_extra_can_restore_python_runtime_overrides(self) -> None:
        """Per-server env is an explicit opt-in and wins over inherited cleanup."""
        with patch.dict(
            os.environ,
            {
                "PYTHONHOME": "/bad/home",
                "PYTHONPATH": "/bad/path",
            },
            clear=True,
        ):
            env = _inherited_stdio_environment({"PYTHONHOME": "/explicit/home", "PYTHONPATH": "/explicit/path"})

        assert env["PYTHONHOME"] == "/explicit/home"
        assert env["PYTHONPATH"] == "/explicit/path"

    def test_no_extra_returns_pure_parent(self) -> None:
        """``extra=None`` returns the inherited env unchanged (apart from filter)."""
        with patch.dict(os.environ, {"CHRYS_TEST_PASSTHROUGH": "ok"}, clear=False):
            env = _inherited_stdio_environment()

        assert env["CHRYS_TEST_PASSTHROUGH"] == "ok"

    def test_demotes_pyapp_runtime_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stdio MCP commands should resolve system executables before PyApp runtime shims."""
        runtime_bin = tmp_path / "runtime" / "bin"
        system_bin = tmp_path / "system" / "bin"
        monkeypatch.setattr(runtime_paths.sys, "executable", str(runtime_bin / "python"))
        monkeypatch.setattr(runtime_paths.sys, "prefix", str(tmp_path / "runtime"))
        monkeypatch.setattr(runtime_paths.sys, "exec_prefix", str(tmp_path / "runtime"))
        monkeypatch.setattr(
            runtime_paths.sysconfig,
            "get_path",
            lambda name: str(runtime_bin) if name == "scripts" else "",
        )
        with patch.dict(
            os.environ,
            {
                "PYAPP": "1",
                "PATH": os.pathsep.join([str(runtime_bin), str(system_bin)]),
            },
            clear=True,
        ):
            env = _inherited_stdio_environment()

        assert env["PATH"].split(os.pathsep) == [str(system_bin), str(runtime_bin)]

    def test_bash_function_exports_filtered(self) -> None:
        """Shellshock-style ``() {`` function exports must not be forwarded."""
        with patch.dict(
            os.environ,
            {
                "BASH_FUNC_x%%": "() {  echo hi\n}",
                "NORMAL_VAR": "kept",
            },
            clear=False,
        ):
            env = _inherited_stdio_environment()

        assert "BASH_FUNC_x%%" not in env
        assert env["NORMAL_VAR"] == "kept"

    def test_returned_dict_is_independent(self) -> None:
        """Mutating the result must not leak back into ``os.environ``."""
        env = _inherited_stdio_environment()
        env["CHRYS_TEST_LEAK_GUARD"] = "should_not_leak"
        assert "CHRYS_TEST_LEAK_GUARD" not in os.environ


# ---------------------------------------------------------------------------
# tolerant_stdio_client — process fakes
# ---------------------------------------------------------------------------


class _FakeByteReceiveStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def receive(self) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        raise anyio.EndOfStream


class _FakeStdin:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.closed = False
        self.wrote = asyncio.Event()

    async def send(self, data: bytes) -> None:
        self.sent.append(data)
        self.wrote.set()

    async def aclose(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(
        self,
        stdout_chunks: list[bytes],
        *,
        stderr_chunks: list[bytes] | None = None,
        exit_code: int = 0,
    ) -> None:
        self.stdout = _FakeByteReceiveStream(stdout_chunks)
        self.stderr = _FakeByteReceiveStream(stderr_chunks or [])
        self.stdin = _FakeStdin()
        self.returncode = exit_code
        self.waited = False

    async def __aenter__(self) -> _FakeProcess:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def wait(self) -> int:
        self.waited = True
        return self.returncode


def _stdio_server_params() -> Any:
    from mcp.client.stdio import StdioServerParameters

    return StdioServerParameters(command="python", args=[], env=None)


@contextmanager
def _patched_stdio_spawn(process: object | None = None, *, error: BaseException | None = None) -> Iterator[None]:
    import mcp.client.stdio as stdio

    spawn = AsyncMock(side_effect=error) if error is not None else AsyncMock(return_value=process)
    with (
        patch.object(stdio, "_get_executable_command", return_value="/usr/bin/python"),
        patch.object(stdio, "_create_platform_compatible_process", new=spawn),
    ):
        yield


# ---------------------------------------------------------------------------
# tolerant_stdio_client — async transport behavior
# ---------------------------------------------------------------------------


async def test_tolerant_stdio_client_drops_banners_and_delivers_jsonrpc() -> None:
    """The real stdout reader drops banners (plain text, bracketed log lines, malformed objects) and delivers JSON-RPC."""
    process = _FakeProcess(
        [
            b"Plain startup banner\r\nSecond banner\n   \n",
            # Neither line is valid JSON, so both are banners -- not protocol
            # errors.  This is the difference between the classifier and a naive
            # ``startswith('{', '[')`` check.
            b'[INFO] starting\n{"jsonrpc": "2.0", bad}\n',
            b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n',
        ]
    )
    captured: list[str] = []

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), dropped_banner_lines=captured) as (
            read_stream,
            _write_stream,
        ):
            received = await asyncio.wait_for(read_stream.receive(), timeout=20.0)

    assert captured == ["Plain startup banner", "Second banner", "[INFO] starting", '{"jsonrpc": "2.0", bad}']
    assert received.message.root.method == "ping"
    assert received.message.root.id == 1
    assert process.stdin.closed is True
    assert process.waited is True


async def test_tolerant_stdio_client_captures_bounded_stderr_tail_and_exit_code() -> None:
    """Stderr stays off the protocol stream while preserving actionable failure context."""
    stderr = b"x" * (MAX_STDIO_STDERR_BYTES_CAPTURED + 64) + b"\nModuleNotFoundError: missing_pkg\n"
    process = _FakeProcess([], stderr_chunks=[stderr], exit_code=23)
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    assert diagnostics.exit_code == 23
    assert diagnostics.stderr_dropped_bytes > 0
    assert len(diagnostics._stderr_tail) == MAX_STDIO_STDERR_BYTES_CAPTURED
    assert diagnostics.stderr_tail.endswith("ModuleNotFoundError: missing_pkg")
    # Spawn context is recorded up front so it survives an early exit.
    assert diagnostics.resolved_executable == "/usr/bin/python"
    assert diagnostics.effective_cwd == os.getcwd()  # no server cwd => inherited


def test_display_path_keeps_ordinary_paths_verbatim() -> None:
    assert _display_path("/opt/homebrew/bin/uv") == "/opt/homebrew/bin/uv"
    assert _display_path("C:\\Tools\\uv.exe") == "C:\\Tools\\uv.exe"
    assert _display_path("/w/ünïcode/路径") == "/w/ünïcode/路径"


class TestResolveSpawnExecutable:
    def test_command_with_path_component_is_used_as_is(self, tmp_path: Path) -> None:
        assert _resolve_spawn_executable("./bin/uv", {"PATH": str(tmp_path)}) == "./bin/uv"
        assert _resolve_spawn_executable(str(tmp_path / "uv"), {}) == str(tmp_path / "uv")

    def test_unresolvable_command_falls_back_to_raw_name(self, tmp_path: Path) -> None:
        assert _resolve_spawn_executable("definitely-not-a-real-cmd", {"PATH": str(tmp_path)}) == (
            "definitely-not-a-real-cmd"
        )

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable-bit PATH lookup")
    def test_bare_command_resolves_against_child_env_path_not_parent(self, tmp_path: Path, monkeypatch: Any) -> None:
        child_bin = tmp_path / "child-bin"
        child_bin.mkdir()
        exe = child_bin / "uv"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        parent_bin = tmp_path / "parent-bin"
        parent_bin.mkdir()
        (parent_bin / "uv").write_text("#!/bin/sh\n")
        (parent_bin / "uv").chmod(0o755)
        monkeypatch.setenv("PATH", str(parent_bin))  # parent PATH must NOT win

        assert _resolve_spawn_executable("uv", {"PATH": str(child_bin)}) == str(exe)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable-bit PATH lookup")
    def test_non_executable_candidate_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "uv").write_text("not executable\n")
        (tmp_path / "uv").chmod(0o644)

        assert _resolve_spawn_executable("uv", {"PATH": str(tmp_path)}) == "uv"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable-bit PATH lookup")
async def test_tolerant_stdio_client_resolves_bare_command_against_child_path(tmp_path: Path) -> None:
    """The diagnostics show *which* ``uv`` the child would exec, not the bare name."""
    import mcp.client.stdio as stdio
    from mcp.client.stdio import StdioServerParameters

    exe = tmp_path / "uv"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    process = _FakeProcess([])
    diagnostics = _StdioProcessDiagnostics()
    params = StdioServerParameters(command="uv", args=[], env={"PATH": str(tmp_path)})
    spawn = AsyncMock(return_value=process)

    with (
        patch.object(stdio, "_get_executable_command", side_effect=lambda cmd: cmd),
        patch.object(stdio, "_create_platform_compatible_process", new=spawn),
    ):
        async with tolerant_stdio_client(params, process_diagnostics=diagnostics, inherit_env=False):
            pass

    assert diagnostics.resolved_executable == str(exe)
    # The spawn itself still receives the configured command untouched.
    assert spawn.await_args is not None
    assert spawn.await_args.kwargs["command"] == "uv"


async def test_tolerant_stdio_client_records_configured_cwd(tmp_path: Path) -> None:
    from mcp.client.stdio import StdioServerParameters

    process = _FakeProcess([])
    diagnostics = _StdioProcessDiagnostics()
    params = StdioServerParameters(command="python", args=[], env=None, cwd=tmp_path)

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(params, process_diagnostics=diagnostics):
            pass

    assert diagnostics.effective_cwd == str(tmp_path)


async def test_tolerant_stdio_client_drains_delayed_stderr_before_process_stream_closes() -> None:
    """A final pipe chunk must be consumed before process.__aexit__ closes stderr."""

    class _DelayedStderr:
        def __init__(self) -> None:
            self.released = asyncio.Event()
            self.closed = False
            self.delivered = False

        async def receive(self) -> bytes:
            if self.delivered:
                raise anyio.EndOfStream
            await self.released.wait()
            await asyncio.sleep(0)
            if self.closed:
                raise anyio.ClosedResourceError
            self.delivered = True
            return b"RuntimeError: final buffered traceback line\n"

    class _DelayedStderrProcess(_FakeProcess):
        def __init__(self) -> None:
            super().__init__([])
            self.stderr = _DelayedStderr()

        async def wait(self) -> int:
            self.waited = True
            self.stderr.released.set()
            return self.returncode

        async def __aexit__(self, *args: object) -> None:
            self.stderr.closed = True

    process = _DelayedStderrProcess()
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    assert diagnostics.stderr_tail == "RuntimeError: final buffered traceback line"


async def test_tolerant_stdio_client_reads_windows_fallback_shaped_sync_stderr() -> None:
    """The MCP SDK's Windows FallbackProcess exposes a synchronous stderr file."""

    class _SyncStderr:
        def __init__(self) -> None:
            self.chunks = [b"fallback stderr line\n", b""]
            self.read1_calls = 0

        def read1(self, _size: int) -> bytes:
            # ``read1`` returns what is available instead of blocking until
            # ``size`` bytes arrive; the reader must prefer it when present.
            self.read1_calls += 1
            return self.chunks.pop(0)

        def read(self, _size: int) -> bytes:
            raise AssertionError("read1 must be preferred over read")

    process = _FakeProcess([], exit_code=5)
    stderr = _SyncStderr()
    process.stderr = stderr  # type: ignore[assignment]
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    assert diagnostics.stderr_tail == "fallback stderr line"
    assert diagnostics.exit_code == 5
    assert stderr.read1_calls == 2


async def test_tolerant_stdio_client_reads_windows_fallback_without_read1() -> None:
    class _ReadOnlyStderr:
        def __init__(self) -> None:
            self.chunks = [b"plain read line\n", b""]

        def read(self, _size: int) -> bytes:
            return self.chunks.pop(0)

    process = _FakeProcess([])
    process.stderr = _ReadOnlyStderr()  # type: ignore[assignment]
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    assert diagnostics.stderr_tail == "plain read line"


async def test_tolerant_stdio_client_fallback_reader_tolerates_non_coroutine_awaitable() -> None:
    """A ``read`` that returns a Future-like awaitable (no ``close``) must not crash the reader."""

    class _Awaitable:
        # Awaitable protocol only: no ``close`` (unlike coroutine objects).
        def __await__(self) -> Any:
            yield from ()
            return b"never read synchronously\n"

    class _FutureStderr:
        def read(self, _size: int) -> Any:
            return _Awaitable()

    process = _FakeProcess([])
    process.stderr = _FutureStderr()  # type: ignore[assignment]
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    assert diagnostics.stderr_tail == ""


async def test_tolerant_stdio_client_fallback_reader_closes_stray_coroutine() -> None:
    class _CoroutineStderr:
        def __init__(self) -> None:
            self.coroutines: list[Any] = []

        def read(self, _size: int) -> Any:
            async def _reader() -> bytes:
                return b"async read\n"

            coro = _reader()
            self.coroutines.append(coro)
            return coro

    process = _FakeProcess([])
    stderr = _CoroutineStderr()
    process.stderr = stderr  # type: ignore[assignment]
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    assert diagnostics.stderr_tail == ""
    assert len(stderr.coroutines) == 1
    assert stderr.coroutines[0].cr_frame is None  # closed, so no "never awaited" warning


async def test_tolerant_stdio_client_pushes_invalid_jsonrpc_exception() -> None:
    """Valid JSON that is not JSON-RPC stays a protocol error on the read stream."""
    process = _FakeProcess([b'{"hello":"world"}\n'])

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params()) as (read_stream, _write_stream):
            received = await asyncio.wait_for(read_stream.receive(), timeout=20.0)

    assert isinstance(received, Exception)


async def test_tolerant_stdio_client_serializes_stdin_messages() -> None:
    """Outbound SessionMessages should be JSON-lines encoded to process stdin."""
    import json

    from mcp.shared.message import SessionMessage
    from mcp.types import JSONRPCMessage, JSONRPCRequest

    process = _FakeProcess([])
    request = JSONRPCRequest(jsonrpc="2.0", id=9, method="tools/list")

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params()) as (_read_stream, write_stream):
            await write_stream.send(SessionMessage(message=JSONRPCMessage(request)))
            await asyncio.wait_for(process.stdin.wrote.wait(), timeout=20.0)

    assert len(process.stdin.sent) == 1
    assert process.stdin.sent[0].endswith(b"\n")
    assert json.loads(process.stdin.sent[0]) == {"jsonrpc": "2.0", "id": 9, "method": "tools/list"}


async def test_tolerant_stdio_client_closes_streams_when_process_spawn_fails() -> None:
    """Spawn OSError should propagate after the local memory streams are closed."""
    with (
        _patched_stdio_spawn(error=OSError("spawn failed")),
        pytest.raises(OSError, match="spawn failed"),
    ):
        async with tolerant_stdio_client(_stdio_server_params()):
            pass


async def test_tolerant_stdio_client_terminates_process_tree_when_wait_times_out() -> None:
    import mcp.client.stdio as stdio

    class _TimeoutWaitProcess(_FakeProcess):
        def __init__(self) -> None:
            super().__init__([])
            self.wait_attempts = 0

        async def wait(self) -> int:
            self.wait_attempts += 1
            if self.wait_attempts == 1:
                raise TimeoutError
            self.returncode = -9
            return self.returncode

    process = _TimeoutWaitProcess()
    terminate = AsyncMock()
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process), patch.object(stdio, "_terminate_process_tree", new=terminate):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    terminate.assert_awaited_once_with(process)
    assert process.wait_attempts == 2
    assert diagnostics.exit_code == -9


async def test_tolerant_stdio_client_ignores_process_lookup_during_shutdown() -> None:

    class _GoneProcess(_FakeProcess):
        async def wait(self) -> None:
            raise ProcessLookupError

    process = _GoneProcess([])

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params()):
            pass

    assert process.stdin.closed is True


# ---------------------------------------------------------------------------
# tolerant_stdio_client — stock-client fallback when SDK privates move
# ---------------------------------------------------------------------------


async def test_tolerant_stdio_falls_back_to_stock_client_when_privates_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the SDK stdio privates vanish, delegate to the stock ``stdio_client``.

    Banner tolerance is lost, but every stdio MCP connection keeps working and
    the sanitized inherited environment is still forwarded (re-injected onto
    the server params via ``model_copy``), not reduced to the SDK's ~6-var allowlist.
    """
    import builtins

    import mcp.client.stdio as stdio
    from mcp.client.stdio import StdioServerParameters

    monkeypatch.setenv("CHRYS_FALLBACK_PROBE", "present")
    received: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_stock(server: Any, errlog: Any = sys.stderr) -> Any:
        received["server"] = server
        received["errlog"] = errlog
        yield ("stock-read", "stock-write")

    server = StdioServerParameters(command="python", args=["-m", "srv"], env={"CHRYS_TEST_ONLY": "1"})

    with (
        patch.object(
            builtins, "__import__", side_effect=block_import("mcp.client.stdio", "_create_platform_compatible_process")
        ),
        patch.object(stdio, "stdio_client", fake_stock),
    ):
        async with tolerant_stdio_client(server, inherit_env=True) as streams:
            assert streams == ("stock-read", "stock-write")

    forwarded = received["server"]
    assert forwarded.env == _inherited_stdio_environment({"CHRYS_TEST_ONLY": "1"})
    assert forwarded.env["CHRYS_TEST_ONLY"] == "1"
    assert forwarded.env["CHRYS_FALLBACK_PROBE"] == "present"


async def test_tolerant_stdio_falls_back_to_stock_client_when_session_message_moves() -> None:
    """``SessionMessage`` is part of the patched reader; if it moves, use stock stdio."""
    import builtins

    import mcp.client.stdio as stdio
    from mcp.client.stdio import StdioServerParameters

    received: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_stock(server: Any, errlog: Any = sys.stderr) -> Any:
        received["server"] = server
        yield ("stock-read", "stock-write")

    server = StdioServerParameters(command="python", env={"ONLY": "this"})

    with (
        patch.object(builtins, "__import__", side_effect=block_import("mcp.shared.message", "SessionMessage")),
        patch.object(stdio, "stdio_client", fake_stock),
    ):
        async with tolerant_stdio_client(server, inherit_env=False) as streams:
            assert streams == ("stock-read", "stock-write")

    assert received["server"] is server


async def test_tolerant_stdio_stock_fallback_passes_server_unchanged_when_not_inheriting() -> None:
    """With ``inherit_env=False`` (cache path; env already complete) the stock
    client receives the original server params unchanged."""
    import builtins

    import mcp.client.stdio as stdio
    from mcp.client.stdio import StdioServerParameters

    received: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_stock(server: Any, errlog: Any = sys.stderr) -> Any:
        received["server"] = server
        yield ("r", "w")

    server = StdioServerParameters(command="python", env={"ONLY": "this"})

    with (
        patch.object(
            builtins, "__import__", side_effect=block_import("mcp.client.stdio", "_create_platform_compatible_process")
        ),
        patch.object(stdio, "stdio_client", fake_stock),
    ):
        async with tolerant_stdio_client(server, inherit_env=False) as streams:
            assert streams == ("r", "w")

    assert received["server"] is server
    assert received["server"].env == {"ONLY": "this"}


# ---------------------------------------------------------------------------
# tolerant_stdio_client — line classification
# ---------------------------------------------------------------------------
