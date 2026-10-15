# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Distinguish configured session stores from unrelated trajectory-shaped trees."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.foundation.config.settings import SESSION_ROOT_DIR_ENV_VAR, resolve_sessions_dir
from chrys.foundation.platform import get_platform
from chrys.service.analytics.session_paths import session_directory_for_events
from tests.support.symlinks import junction_or_skip, symlink_or_skip


@pytest.mark.parametrize("link_kind", ["symlink", "junction"])
def test_default_sessions_link_keeps_ownership_under_a_different_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, link_kind: str
) -> None:
    monkeypatch.setenv(SESSION_ROOT_DIR_ENV_VAR, "")
    root = get_platform().config_dir
    root.mkdir(parents=True, exist_ok=True)
    destination = tmp_path / "icode-history"
    destination.mkdir()
    if link_kind == "junction":
        junction_or_skip(root / "sessions", destination)
    else:
        symlink_or_skip(root / "sessions", destination, target_is_directory=True)
    session_dir = resolve_sessions_dir() / "abc123"
    events = session_dir / "trajectory" / "events.jsonl"

    assert session_dir.parent == destination
    assert session_directory_for_events(events) == session_dir
    # An alternate spelling of the configured store has the same identity.
    alias = tmp_path / "history-alias"
    if link_kind == "junction":
        junction_or_skip(alias, destination)
    else:
        symlink_or_skip(alias, destination, target_is_directory=True)
    assert session_directory_for_events(alias / "abc123" / "trajectory" / "events.jsonl") == alias / "abc123"


def test_unrelated_log_does_not_create_the_configured_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "not-created"
    monkeypatch.setenv(SESSION_ROOT_DIR_ENV_VAR, str(root))
    session = tmp_path / "unrelated" / "abc123"
    (session / "trajectory").mkdir(parents=True)
    (session / "session.json").write_text('{"state": {"messages": []}}', encoding="utf-8")

    assert session_directory_for_events(session / "trajectory" / "events.jsonl") is None
    assert not root.exists()


@pytest.mark.parametrize(
    ("path", "expected_parent"),
    [
        (Path("root/sessions/abc/trajectory/events.jsonl"), True),
        (Path("root/sessions/abc/trajectory/other.jsonl"), False),
        (Path("root/elsewhere/abc/trajectory/events.jsonl"), False),
        (Path("root/sessions/abc/other/events.jsonl"), False),
        (Path("events.jsonl"), False),
    ],
)
def test_session_directory_requires_the_store_layout(path: Path, expected_parent: bool) -> None:
    resolved = session_directory_for_events(path)
    if expected_parent:
        assert resolved == Path("root/sessions/abc")
    else:
        assert resolved is None
