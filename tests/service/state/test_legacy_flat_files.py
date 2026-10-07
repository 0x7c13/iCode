# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for pre-versioned session files — flat <id>.json layouts, their migration, and legacy meta keys."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chrys.kernel import Message
from chrys.service.state.store import (
    JsonFileStateStore,
)
from tests.service.state._store_helpers import legacy_envelope, write_legacy_envelope


def _make_state(msgs: list[Message] | None = None) -> dict[str, Any]:
    return {"messages": msgs or [], "compressed_msgs": []}


def _write_legacy_flat_session(tmp_path: Path, session_id: str) -> Path:
    """Write *session_id* as a flat ``<id>.json`` carrying one user message, and return its path.

    ``write_legacy_envelope`` writes an empty state; the migration tests need a
    message in the payload to prove it survives the move into ``<id>/session.json``.
    """
    envelope = legacy_envelope(session_id, agent_profile="legacy", display_name="Legacy Agent", message_count=1)
    envelope["state"] = {"messages": [{"role": "user", "contents": [{"type": "text", "text": "hello"}]}]}
    path = tmp_path / f"{session_id}.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return path


def test_resolve_session_file_migrates_a_flat_session_and_says_when_there_is_none(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    legacy = _write_legacy_flat_session(tmp_path, "flat")
    write_legacy_envelope(tmp_path / "folder" / "session.json", "folder")

    assert store.resolve_session_file("flat") == tmp_path / "flat" / "session.json"
    assert not legacy.exists()
    assert store.resolve_session_file("folder") == tmp_path / "folder" / "session.json"
    assert store.resolve_session_file("missing") is None


async def test_legacy_flat_files_with_duplicate_ids_list_once(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    write_legacy_envelope(tmp_path / "aaa.json", "dupe")
    write_legacy_envelope(tmp_path / "bbb.json", "dupe")

    sessions = await store.list_sessions()

    assert [s.session_id for s in sessions] == ["dupe"]


async def test_list_sessions_legacy_files_default_to_zero(tmp_path: Path) -> None:
    """Sessions written before the version fields existed read back with
    ``schema_version=0`` and empty ``app_version`` — those defaults let
    readers treat them as pre-versioned without special-casing key
    presence."""
    write_legacy_envelope(tmp_path / "legacy" / "session.json", "legacy", agent_profile="code", display_name="")

    store = JsonFileStateStore(tmp_path)
    sessions = await store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].schema_version == 0
    # Legacy file used the v1 key name ``display_name`` — the reader
    # must still surface it via the new ``agent_display_name`` attr.
    assert sessions[0].agent_display_name == ""
    assert sessions[0].app_version == ""
    # Platform + model fields also default to empty on legacy files.
    assert sessions[0].os_name == ""
    assert sessions[0].arch == ""
    assert sessions[0].model_provider == ""
    assert sessions[0].model_api_style == ""
    assert sessions[0].model_id == ""
    assert sessions[0].model_profile_id == ""
    assert sessions[0].model_base_url == ""
    assert sessions[0].service_session_id == ""


async def test_list_sessions_reads_legacy_key_names(tmp_path: Path) -> None:
    """A pre-versioned file on disk uses the old key names
    (``display_name`` / ``profile_history``).  Readers must surface
    them via the current attributes so TUI code never has to branch
    on whether the file predates the rename."""
    # No ``schema_version`` — this file was written by pre-versioned code and uses the old key names.
    write_legacy_envelope(
        tmp_path / "legacy" / "session.json",
        "legacy",
        agent_profile="code",
        display_name="Code Agent",
        profile_history=["Code Agent", "Explore Agent"],
        message_count=3,
    )

    store = JsonFileStateStore(tmp_path)
    sessions = await store.list_sessions()

    assert len(sessions) == 1
    assert sessions[0].schema_version == 0  # pre-versioned
    # Rename is invisible to readers — old keys surface on new attrs.
    assert sessions[0].agent_display_name == "Code Agent"
    assert sessions[0].agent_profile_history == ["Code Agent", "Explore Agent"]


async def test_resave_migrates_legacy_keys_without_blanking(tmp_path: Path) -> None:
    """A pre-versioned file re-saved by the current writer should keep
    its display/history values — the writer falls back to the old keys
    on ``existing_meta`` lookups so the second save doesn't clobber
    them just because the caller didn't explicitly repeat them."""
    from chrys.service.state.store import SESSION_SCHEMA_VERSION

    # No ``schema_version`` — pre-versioned shape.
    write_legacy_envelope(
        tmp_path / "legacy" / "session.json",
        "legacy",
        agent_profile="code",
        display_name="Code Agent",
        profile_history=["Code Agent", "Explore Agent"],
    )

    store = JsonFileStateStore(tmp_path)
    # Re-save omitting both renamed kwargs — writer must pick them up
    # off the existing legacy meta rather than writing empty strings.
    await store.save_session("legacy", {"messages": [], "compressed_msgs": []})

    meta = json.loads((tmp_path / "legacy" / "session.json").read_text(encoding="utf-8"))["meta"]
    # File now carries the current schema version — new keys present
    # with preserved values, old keys gone.
    assert meta["schema_version"] == SESSION_SCHEMA_VERSION
    assert meta["agent_display_name"] == "Code Agent"
    assert meta["agent_profile_history"] == ["Code Agent", "Explore Agent"]
    assert "display_name" not in meta
    assert "profile_history" not in meta


# ===========================================================================
# Store — folder-based sessions
# ===========================================================================


class TestFolderBasedStore:
    """JsonFileStateStore stores sessions as {id}/session.json folders."""

    async def test_save_creates_folder_structure(self, tmp_path: Path) -> None:
        store = JsonFileStateStore(tmp_path)
        await store.save_session("abc123", _make_state([Message("user", ["hi"])]), agent_profile="code")

        session_json = tmp_path / "abc123" / "session.json"
        assert session_json.exists()
        # Old flat-file should NOT exist
        assert not (tmp_path / "abc123.json").exists()


# ===========================================================================
# Store — legacy migration
# ===========================================================================


class TestLegacyMigration:
    """Old {id}.json flat files are migrated into {id}/session.json on access."""

    async def test_load_migrates_legacy_file(self, tmp_path: Path) -> None:
        legacy_path = _write_legacy_flat_session(tmp_path, "old_sess")
        assert legacy_path.exists()

        store = JsonFileStateStore(tmp_path)
        loaded = await store.load_session("old_sess")
        assert loaded is not None

        # Legacy file should be gone, folder should exist
        assert not legacy_path.exists()
        assert (tmp_path / "old_sess" / "session.json").exists()

    async def test_load_raw_migrates_legacy_file(self, tmp_path: Path) -> None:
        _write_legacy_flat_session(tmp_path, "old_raw")
        store = JsonFileStateStore(tmp_path)
        raw = await store.load_session_raw("old_raw")
        assert raw is not None
        assert len(raw) == 1
        assert raw[0]["role"] == "user"

    async def test_list_includes_legacy_sessions(self, tmp_path: Path) -> None:
        """list_sessions finds both folder-based and legacy flat-file sessions."""
        store = JsonFileStateStore(tmp_path)
        await store.save_session("new_sess", _make_state(), agent_profile="new")
        _write_legacy_flat_session(tmp_path, "legacy_sess")

        sessions = await store.list_sessions()
        ids = {s.session_id for s in sessions}
        assert "new_sess" in ids
        assert "legacy_sess" in ids

    async def test_save_migrates_then_updates(self, tmp_path: Path) -> None:
        """save_session on a legacy session migrates it first, then saves."""
        _write_legacy_flat_session(tmp_path, "migrating")
        store = JsonFileStateStore(tmp_path)
        await store.save_session("migrating", _make_state([Message("user", ["new"])]), agent_profile="updated")

        # Legacy gone, folder exists
        assert not (tmp_path / "migrating.json").exists()
        session_json = tmp_path / "migrating" / "session.json"
        assert session_json.exists()
        raw = json.loads(session_json.read_text(encoding="utf-8"))
        assert raw["meta"]["agent_profile"] == "updated"

    async def test_delete_cleans_up_legacy_file(self, tmp_path: Path) -> None:
        _write_legacy_flat_session(tmp_path, "legacy_del")
        store = JsonFileStateStore(tmp_path)
        await store.delete_session("legacy_del")
        assert not (tmp_path / "legacy_del.json").exists()
        assert not (tmp_path / "legacy_del").exists()
