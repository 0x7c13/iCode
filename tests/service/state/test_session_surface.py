# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The recorded session surface: written by a marked save, carried verbatim by every other write."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from chrys.foundation.models.session_surface import SessionSurface, parse_session_surface
from chrys.kernel import Message
from chrys.service.state.store import JsonFileStateStore
from chrys.service.state.workflow import WorkflowSessionState
from tests.support.workflow_history import workflow_state


def _state(*texts: str) -> dict[str, Any]:
    return {"messages": [Message("user", [text]) for text in texts or ("hi",)], "compressed_msgs": []}


def _recorded(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))["meta"].get("last_surface", "<absent>")


def _primary(store: JsonFileStateStore, session_id: str) -> Path:
    return store.session_dir(session_id) / "session.json"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("tui", SessionSurface.TUI), ("cli", SessionSurface.CLI), ("acp", SessionSurface.ACP)]
    + [(value, None) for value in (None, "", "TUI", "web", 1, ["tui"])],
)
def test_parse_session_surface_is_lenient(value: object, expected: SessionSurface | None) -> None:
    assert parse_session_surface(value) is expected


async def test_marked_saves_record_and_unmarked_saves_carry(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s", _state())
    assert _recorded(_primary(store, "s")) == "<absent>"
    assert (await store.load_session_meta("s")).last_surface is None

    await store.save_session("s", _state(), last_surface=SessionSurface.CLI)
    assert _recorded(_primary(store, "s")) == "cli"
    await store.save_session("s", _state("hi", "more"))
    assert _recorded(_primary(store, "s")) == "cli"
    await store.update_session_titles("s", custom_title="Named")
    assert _recorded(_primary(store, "s")) == "cli"

    await store.save_session("s", _state("hi", "more"), last_surface=SessionSurface.TUI)
    assert (await store.load_session_meta("s")).last_surface is SessionSurface.TUI


async def test_an_unknown_surface_is_carried_verbatim_and_reads_as_unrecorded(tmp_path: Path) -> None:
    """A value written by a newer version survives this version's saves."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s", _state())
    envelope = json.loads(_primary(store, "s").read_text(encoding="utf-8"))
    envelope["meta"]["last_surface"] = "web"
    _primary(store, "s").write_text(json.dumps(envelope), encoding="utf-8")

    await store.save_session("s", _state("hi", "more"))

    assert _recorded(_primary(store, "s")) == "web"
    assert (await store.load_session_meta("s")).last_surface is None


async def test_a_newer_recovery_sidecar_surface_outranks_the_primary(tmp_path: Path) -> None:
    """The turn a crash cut short was the session's last one, even if the next save is unmarked."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s", _state(), last_surface=SessionSurface.CLI)
    store.save_recovery_session("s", _state("hi", "cut short"), last_surface=SessionSurface.ACP)
    assert _recorded(store.session_dir("s") / "session.recovery.json") == "acp"

    await store.save_session("s", _state("hi", "cut short"))

    assert _recorded(_primary(store, "s")) == "acp"


async def test_carrying_parses_a_foreign_sidecar_once_and_never_this_stores_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checkpoints are frequent and a sidecar can be megabytes: only one another process left is read."""
    await JsonFileStateStore(tmp_path).save_session("s", _state(), last_surface=SessionSurface.CLI)
    JsonFileStateStore(tmp_path).save_recovery_session("s", _state("hi", "cut short"), last_surface=SessionSurface.ACP)
    store = JsonFileStateStore(tmp_path)
    reads: list[str] = []
    read_recovery_meta = store._read_recovery_meta_unlocked

    def counted(session_id: str) -> dict[str, Any]:
        reads.append(session_id)
        return read_recovery_meta(session_id)

    monkeypatch.setattr(store, "_read_recovery_meta_unlocked", counted)
    store.save_recovery_session("s", _state("hi", "cut short"))
    # A title patch rewrites the sidecar in place; it is still this store's own.
    await store.update_session_titles("s", custom_title="Named")
    store.save_recovery_session("s", _state("hi", "cut short", "more"))
    assert _recorded(store.session_dir("s") / "session.recovery.json") == "acp"
    await store.save_session("s", _state("hi", "cut short", "more"))

    assert reads == ["s"]
    assert _recorded(_primary(store, "s")) == "acp"


async def test_the_recovery_memo_is_dropped_whichever_id_form_removes_the_sidecar(tmp_path: Path) -> None:
    """Checkpoints name a session by its full id; the listing's stale-sidecar cleanup by its folder."""
    store = JsonFileStateStore(tmp_path)
    session_id = str(uuid4())
    await store.save_session(session_id, _state(), last_surface=SessionSurface.CLI)
    store.save_recovery_session(session_id, _state("hi", "cut short"), last_surface=SessionSurface.ACP)
    recovery = store.session_dir(session_id) / "session.recovery.json"
    stale = json.loads(recovery.read_text(encoding="utf-8"))
    stale["meta"]["updated_at"] = datetime(2000, 1, 1, tzinfo=UTC).isoformat()
    recovery.write_text(json.dumps(stale), encoding="utf-8")
    assert len(store._written_recovery_meta) == 1

    await store.open_session_listing(kind="chat")

    assert not recovery.exists()
    assert store._written_recovery_meta == {}


async def test_a_stale_recovery_sidecar_surface_does_not_outrank_the_primary(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    store.save_recovery_session("s", _state(), last_surface=SessionSurface.ACP)
    recovery = store.session_dir("s") / "session.recovery.json"
    stale = json.loads(recovery.read_text(encoding="utf-8"))
    await store.save_session("s", _state(), last_surface=SessionSurface.CLI)
    stale["meta"]["updated_at"] = datetime(2000, 1, 1, tzinfo=UTC).isoformat()
    recovery.write_text(json.dumps(stale), encoding="utf-8")

    await store.save_session("s", _state("hi", "more"))

    assert _recorded(_primary(store, "s")) == "cli"


async def test_while_the_primary_is_unreadable_the_sidecar_outranks_the_backup(tmp_path: Path) -> None:
    """As for titles: meta read back from the backup never outranks the sidecar, whatever their timestamps."""
    store = JsonFileStateStore(tmp_path)
    store.save_recovery_session("s", _state(), last_surface=SessionSurface.ACP)
    recovery = store.session_dir("s") / "session.recovery.json"
    sidecar = json.loads(recovery.read_text(encoding="utf-8"))
    await store.save_session("s", _state(), last_surface=SessionSurface.CLI)
    sidecar["meta"]["updated_at"] = datetime(2000, 1, 1, tzinfo=UTC).isoformat()
    recovery.write_text(json.dumps(sidecar), encoding="utf-8")
    backup = store.session_dir("s") / "session.json.bak"
    backup.write_bytes(_primary(store, "s").read_bytes())
    _primary(store, "s").write_text("{", encoding="utf-8")
    assert _recorded(backup) == "cli"

    await store.save_session("s", _state("hi", "more"))

    assert _recorded(_primary(store, "s")) == "acp"


async def test_a_fork_records_the_forking_surface_or_keeps_the_parents(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("parent", _state(), last_surface=SessionSurface.CLI)

    forked = store.fork_session("parent", last_surface=SessionSurface.TUI)
    kept = store.fork_session("parent")

    assert (await store.load_session_meta(forked)).last_surface is SessionSurface.TUI
    assert (await store.load_session_meta(kept)).last_surface is SessionSurface.CLI
    assert (await store.load_session_meta("parent")).last_surface is SessionSurface.CLI


def test_workflow_state_surface_round_trips_and_decodes_leniently(tmp_path: Path) -> None:
    base = workflow_state(tmp_path)
    assert "last_surface" not in base
    assert WorkflowSessionState.decode(base).surface is None
    assert "last_surface" not in WorkflowSessionState.decode(base).encode()

    recorded = WorkflowSessionState.decode({**base, "last_surface": "acp"})
    assert recorded.surface is SessionSurface.ACP
    assert recorded.encode()["last_surface"] == "acp"
    # A value written by a newer version survives this version's saves, like chat's.
    unknown = WorkflowSessionState.decode({**base, "last_surface": "web"})
    assert unknown.surface is None and unknown.encode()["last_surface"] == "web"
    malformed = WorkflowSessionState.decode({**base, "last_surface": 7})
    assert malformed.surface is None and "last_surface" not in malformed.encode()


async def test_workflow_session_meta_reads_the_state_surface(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    state = WorkflowSessionState.decode({**workflow_state(tmp_path), "last_surface": "cli"})
    await store.save_workflow_session("w", state)

    meta = await store.load_session_meta("w")

    assert meta is not None and meta.kind == "workflow" and meta.last_surface is SessionSurface.CLI
    assert _recorded(_primary(store, "w")) == "<absent>"
