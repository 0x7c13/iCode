# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session-directory storage summary used by the trajectory dashboard."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from threading import Event

import pytest

from chrys.app.tui.widgets.trajectory.overview import _fit_path_tail
from chrys.app.tui.widgets.trajectory.panel import _session_directory
from chrys.app.tui.widgets.trajectory.session_info import SessionStorage, collect_session_storage
from chrys.service.analytics import TrajectoryScanCancelled


def _make_session(tmp_path: Path) -> Path:
    session_dir = tmp_path / "sessions" / "abcdef123456"
    (session_dir / "trajectory").mkdir(parents=True)
    (session_dir / "session.json").write_bytes(b"a" * 100)
    (session_dir / "trajectory" / "events.jsonl").write_bytes(b"b" * 200)
    (session_dir / "mutations").mkdir()
    (session_dir / "mutations" / "0001.diff").write_bytes(b"c" * 300)
    (session_dir / "snapshots" / "deep").mkdir(parents=True)
    (session_dir / "snapshots" / "deep" / "blob").write_bytes(b"d" * 400)
    (session_dir / "sub_agents" / "child").mkdir(parents=True)
    (session_dir / "sub_agents" / "child" / "session.json").write_bytes(b"e" * 500)
    return session_dir


def test_collect_session_storage_sums_tree_and_notable_members(tmp_path: Path) -> None:
    session_dir = _make_session(tmp_path)

    storage = collect_session_storage(session_dir)

    assert storage == SessionStorage(
        session_dir=session_dir,
        total_bytes=1500,
        session_json_bytes=100,
        events_bytes=200,
        mutations_bytes=300,
        snapshots_bytes=400,
        sub_agents_bytes=500,
        file_count=5,
    )


def test_collect_session_storage_skips_symlinks_and_survives_a_missing_directory(tmp_path: Path) -> None:
    session_dir = _make_session(tmp_path)
    if hasattr(os, "symlink"):
        # Symlink-restricted platforms simply skip the extra entry.
        with contextlib.suppress(OSError):
            os.symlink(tmp_path, session_dir / "loop")

    storage = collect_session_storage(session_dir)
    assert storage.total_bytes == 1500  # the symlinked tree is not followed

    missing = collect_session_storage(tmp_path / "absent")
    assert missing == SessionStorage(session_dir=tmp_path / "absent")


def test_collect_session_storage_stops_when_cancelled(tmp_path: Path) -> None:
    """A hidden dashboard must not keep an executor thread walking the tree."""
    session_dir = _make_session(tmp_path)
    cancelled = Event()
    cancelled.set()

    with pytest.raises(TrajectoryScanCancelled):
        collect_session_storage(session_dir, cancel_event=cancelled)


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
    resolved = _session_directory(path)
    if expected_parent:
        assert resolved == Path("root/sessions/abc")
    else:
        assert resolved is None


def test_fit_path_tail_keeps_the_end_of_a_cropped_path() -> None:
    assert _fit_path_tail("short", 10) == "short"
    fitted = _fit_path_tail("/very/long/session/path", 10)
    assert fitted.startswith("…")
    assert fitted.endswith("path")
    assert len(fitted) <= 10
