# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ACP session requests whose working directory was deleted or never existed."""

from __future__ import annotations

import sys

import pytest

from chrys.app.acp import session_manager as session_manager_module
from chrys.app.acp.session_manager import AcpSessionError
from chrys.foundation.events.types import WorkspaceChange
from chrys.foundation.util.session_ids import session_short_id
from chrys.kernel import Message
from chrys.service.state.store import JsonFileStateStore
from tests.app.acp._session_manager_fakes import _manager, _StartedHost
from tests.support.symlinks import symlink_or_skip


@pytest.fixture
def started_hosts(monkeypatch: pytest.MonkeyPatch) -> list[_StartedHost]:
    _StartedHost.instances = []
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _StartedHost)
    return _StartedHost.instances


async def test_new_session_without_cwd_refuses_a_deleted_default_workdir(
    tmp_path, started_hosts: list[_StartedHost]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = _manager(str(project), JsonFileStateStore(tmp_path / "sessions"))
    default_cwd = manager.process_cwd
    project.rmdir()

    try:
        with pytest.raises(AcpSessionError) as error:
            await manager.new_session(cwd=None, mcp_servers=None)
    finally:
        await manager.shutdown()

    assert str(error.value) == f"working directory no longer exists: {default_cwd}"
    assert started_hosts == []


async def test_new_session_with_a_missing_cwd_is_an_acp_error(tmp_path, started_hosts: list[_StartedHost]) -> None:
    missing = tmp_path / "missing"
    manager = _manager(None, JsonFileStateStore(tmp_path / "sessions"))

    with pytest.raises(AcpSessionError) as error:
        await manager.new_session(cwd=str(missing), mcp_servers=None)

    # The client gets its own path back, not a raw FileNotFoundError.
    assert str(error.value) == f"cwd does not exist: {missing}"
    assert error.value.__cause__ is None
    assert started_hosts == []
    assert manager.process_cwd is None


async def test_new_session_with_an_unreachable_cwd_says_it_is_not_accessible(
    tmp_path, started_hosts: list[_StartedHost]
) -> None:
    """A cwd that exists but cannot be resolved (here a symlink loop) is not reported as missing."""
    loop = tmp_path / "loop"
    symlink_or_skip(loop, loop)
    manager = _manager(None, JsonFileStateStore(tmp_path / "sessions"))

    with pytest.raises(AcpSessionError) as error:
        await manager.new_session(cwd=str(loop), mcp_servers=None)

    assert str(error.value) == f"cwd is not accessible: {loop}"
    assert started_hosts == []


@pytest.mark.skipif(sys.platform == "win32", reason="surrogate-escaped path bytes are a POSIX filesystem concept")
def test_missing_cwd_error_text_is_safe_for_strict_utf8(tmp_path) -> None:
    raw = f"{tmp_path}/missing-\udcff"

    with pytest.raises(AcpSessionError) as error:
        session_manager_module._resolve_dir(raw)

    message = str(error.value)
    message.encode("utf-8")
    assert message.startswith(f"cwd does not exist: {tmp_path}/missing-")


async def test_set_workspace_to_a_missing_directory_keeps_the_session_where_it_was(
    tmp_path, started_hosts: list[_StartedHost]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    missing = tmp_path / "missing"
    manager = _manager(None, JsonFileStateStore(tmp_path / "sessions"))

    try:
        session = await manager.new_session(cwd=str(project), mcp_servers=None)
        published: list[WorkspaceChange] = []

        async def record(event: WorkspaceChange) -> None:
            published.append(event)

        await session.host.event_bus.subscribe(WorkspaceChange, record)
        with pytest.raises(AcpSessionError, match="cwd does not exist"):
            await manager.set_workspace(session.session_id, str(missing))
        assert session.cwd == str(project.resolve())
        # Rejected before any WorkspaceChange reaches the engine.
        assert published == []
    finally:
        await manager.shutdown()


async def _save_session_in(store: JsonFileStateStore, session_id: str, cwd: str) -> None:
    await store.save_session(
        session_id,
        {"messages": [Message("user", ["hello"])]},
        agent_profile="Code",
        primary_cwd=cwd,
    )


async def test_load_session_whose_saved_directory_is_gone_says_so(tmp_path, started_hosts: list[_StartedHost]) -> None:
    gone = tmp_path / "gone"
    gone.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await _save_session_in(store, "gone-session", str(gone))
    gone.rmdir()
    manager = _manager(None, store)

    with pytest.raises(AcpSessionError) as error:
        await manager.load_session(cwd=str(other), session_id="gone-session", mcp_servers=None)

    # Not "belongs to a different workspace": the client learns the directory itself is gone.
    assert str(error.value) == (
        f"working directory of session '{session_short_id('gone-session')}' no longer exists: {gone}"
    )
    assert started_hosts == []
    assert manager.process_cwd is None


async def test_session_history_of_a_session_whose_saved_directory_is_gone_says_so(tmp_path) -> None:
    gone = tmp_path / "gone"
    gone.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await _save_session_in(store, "gone-session", str(gone))
    gone.rmdir()
    manager = _manager(str(other), store)

    with pytest.raises(AcpSessionError, match="no longer exists"):
        await manager.session_history(cwd=None, session_id="gone-session")


async def test_load_session_from_a_deleted_default_workdir_is_refused_before_lookup(
    tmp_path, started_hosts: list[_StartedHost]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await _save_session_in(store, "project-session", str(project))
    manager = _manager(str(project), store)
    default_cwd = manager.process_cwd
    project.rmdir()

    with pytest.raises(AcpSessionError) as error:
        await manager.load_session(cwd=None, session_id="project-session", mcp_servers=None)

    assert str(error.value) == f"working directory no longer exists: {default_cwd}"
    assert started_hosts == []
