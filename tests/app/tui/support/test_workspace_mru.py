# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the persisted workspace MRU index."""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chrys.app.tui.support import workspace_mru
from chrys.app.tui.support.workspace_mru import (
    ensure_workspace_mru_index,
    load_workspace_mru,
    record_workspace_mru_uses,
    schedule_workspace_mru_touches,
    session_root_key,
    touch_workspace_mru,
    workspace_mru_backfilled_for,
    workspace_mru_exists,
    workspace_mru_path,
)

ROOT_A = "sha256:root-a"
ROOT_B = "sha256:root-b"


def test_default_mru_path_uses_platform_isolation_without_a_local_patch(tmp_path: Path) -> None:
    from chrys.foundation.platform import get_platform

    assert workspace_mru_path().is_relative_to(tmp_path)
    assert workspace_mru_path().parent == get_platform().config_dir
    get_platform.cache_clear()
    assert workspace_mru_path().is_relative_to(tmp_path)


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "config"
    fake_platform = type("P", (), {"config_dir": target})()
    monkeypatch.setattr(workspace_mru, "get_platform", lambda: fake_platform)
    return target


@pytest.fixture
def workdirs(tmp_path: Path) -> list[str]:
    dirs = []
    for name in ("alpha", "beta", "gamma", "delta"):
        d = tmp_path / "work" / name
        d.mkdir(parents=True)
        dirs.append(str(d))
    return dirs


def _ts(offset_seconds: float = 0) -> datetime:
    return datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC) + timedelta(seconds=offset_seconds)


def _read_index_file(config_dir: Path) -> dict:
    return json.loads((config_dir / "workspace_mru.json").read_text(encoding="utf-8"))


# ── ensure_workspace_mru_index ─────────────────────────────────────────


def test_ensure_creates_index_from_seed(config_dir: Path, workdirs: list[str]) -> None:
    a, b, c, _ = workdirs

    result = ensure_workspace_mru_index(
        [(a, _ts(30)), (b, _ts(20)), (c, _ts(10))],
        max_entries=20,
        root_key=ROOT_A,
    )

    assert result == [a, b, c]
    assert workspace_mru_exists()
    assert workspace_mru_backfilled_for(ROOT_A)
    assert load_workspace_mru(20) == [a, b, c]


def test_ensure_disabled_creates_nothing(config_dir: Path, workdirs: list[str]) -> None:
    assert ensure_workspace_mru_index([(workdirs[0], _ts())], max_entries=0, root_key=ROOT_A) == []
    assert not workspace_mru_exists()
    assert load_workspace_mru(0) == []


def test_ensure_same_root_ignores_new_seed(config_dir: Path, workdirs: list[str]) -> None:
    a, b, *_ = workdirs
    ensure_workspace_mru_index([(a, _ts(10))], max_entries=20, root_key=ROOT_A)

    result = ensure_workspace_mru_index([(b, _ts(20))], max_entries=20, root_key=ROOT_A)

    assert result == [a]
    assert load_workspace_mru(20) == [a]


def test_ensure_new_root_merges_seed_once(config_dir: Path, workdirs: list[str]) -> None:
    a, b, *_ = workdirs
    ensure_workspace_mru_index([(a, _ts(10))], max_entries=20, root_key=ROOT_A)

    merged = ensure_workspace_mru_index([(b, _ts(20))], max_entries=20, root_key=ROOT_B)

    assert merged == [b, a]
    assert workspace_mru_backfilled_for(ROOT_A)
    assert workspace_mru_backfilled_for(ROOT_B)
    roots = _read_index_file(config_dir)["backfilled_session_roots"]
    assert roots == [ROOT_A, ROOT_B]

    # A second ensure for root B must not re-merge its (changed) seed.
    again = ensure_workspace_mru_index([(workdirs[2], _ts(30))], max_entries=20, root_key=ROOT_B)
    assert again == [b, a]


def test_ensure_empty_seed_still_marks_root(config_dir: Path) -> None:
    result = ensure_workspace_mru_index([], max_entries=20, root_key=ROOT_A)

    assert result == []
    assert workspace_mru_exists()
    assert workspace_mru_backfilled_for(ROOT_A)


def test_ensure_seed_filters_favorites_missing_and_relative(
    config_dir: Path, workdirs: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b, *_ = workdirs
    favorite = tmp_path / "favorite"
    favorite.mkdir()
    monkeypatch.setattr(
        workspace_mru,
        "get_default_favorite_paths",
        lambda: {os.path.normcase(os.path.normpath(str(favorite)))},
    )
    missing = str(tmp_path / "nope")

    result = ensure_workspace_mru_index(
        [(str(favorite), _ts(40)), (missing, _ts(30)), ("relative/path", _ts(20)), (a, _ts(10)), (b, _ts(5))],
        max_entries=2,
        root_key=ROOT_A,
    )

    # Filtered entries must not occupy max_entries slots.
    assert result == [a, b]


def test_concurrent_ensure_produces_valid_index(config_dir: Path, workdirs: list[str]) -> None:
    a, b, *_ = workdirs

    def _run(root_key: str, path: str, offset: int) -> list[str]:
        return ensure_workspace_mru_index([(path, _ts(offset))], max_entries=20, root_key=root_key)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(_run, root, path, offset) for root, path, offset in [(ROOT_A, a, 10), (ROOT_B, b, 20)] * 2
        ]
        for future in futures:
            future.result()

    data = _read_index_file(config_dir)
    assert set(data["backfilled_session_roots"]) == {ROOT_A, ROOT_B}
    assert load_workspace_mru(20) == [b, a]


# ── touch_workspace_mru / record_workspace_mru_uses ────────────────────


def test_touch_missing_index_is_noop(config_dir: Path, workdirs: list[str]) -> None:
    assert touch_workspace_mru(workdirs[0], max_entries=20, used_at=_ts()) is False
    assert not workspace_mru_exists()


def test_touch_new_path_goes_first(config_dir: Path, workdirs: list[str]) -> None:
    a, b, *_ = workdirs
    ensure_workspace_mru_index([(a, _ts(10))], max_entries=20, root_key=ROOT_A)

    assert touch_workspace_mru(b, max_entries=20, used_at=_ts(20)) is True

    assert load_workspace_mru(20) == [b, a]


def test_touch_duplicate_path_moves_to_front_keeping_one_entry(config_dir: Path, workdirs: list[str]) -> None:
    a, b, *_ = workdirs
    ensure_workspace_mru_index([(a, _ts(20)), (b, _ts(10))], max_entries=20, root_key=ROOT_A)

    assert touch_workspace_mru(b, max_entries=20, used_at=_ts(30)) is True

    assert load_workspace_mru(20) == [b, a]
    assert len(_read_index_file(config_dir)["paths"]) == 2


def test_out_of_order_touches_keep_newer_first(config_dir: Path, workdirs: list[str]) -> None:
    a, b, *_ = workdirs
    ensure_workspace_mru_index([], max_entries=20, root_key=ROOT_A)

    # Newer event B lands first; the late, older event A must not jump ahead.
    touch_workspace_mru(b, max_entries=20, used_at=_ts(20))
    touch_workspace_mru(a, max_entries=20, used_at=_ts(10))

    assert load_workspace_mru(20) == [b, a]


def test_stale_touch_cannot_downgrade_same_path(config_dir: Path, workdirs: list[str]) -> None:
    a, b, *_ = workdirs
    ensure_workspace_mru_index([], max_entries=20, root_key=ROOT_A)
    touch_workspace_mru(a, max_entries=20, used_at=_ts(30))
    touch_workspace_mru(b, max_entries=20, used_at=_ts(20))

    # A stale event for path A (older than its recorded use) arrives late.
    touch_workspace_mru(a, max_entries=20, used_at=_ts(10))

    assert load_workspace_mru(20) == [a, b]
    entry = _read_index_file(config_dir)["paths"][0]
    assert entry["path"] == a
    assert entry["last_used_at"] == "2026-07-01T12:00:30Z"


def test_touch_relative_path_dropped_not_cwd_resolved(config_dir: Path, workdirs: list[str]) -> None:
    ensure_workspace_mru_index([], max_entries=20, root_key=ROOT_A)

    assert touch_workspace_mru("relative/dir", max_entries=20, used_at=_ts()) is False

    assert load_workspace_mru(20) == []


def test_touch_filters_favorites_and_missing_dirs(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    favorite = tmp_path / "favorite"
    favorite.mkdir()
    monkeypatch.setattr(
        workspace_mru,
        "get_default_favorite_paths",
        lambda: {os.path.normcase(os.path.normpath(str(favorite)))},
    )
    ensure_workspace_mru_index([], max_entries=20, root_key=ROOT_A)

    assert touch_workspace_mru(str(favorite), max_entries=20, used_at=_ts()) is False
    assert touch_workspace_mru(str(tmp_path / "gone"), max_entries=20, used_at=_ts()) is False

    assert load_workspace_mru(20) == []


def test_touch_trims_to_max_entries(config_dir: Path, workdirs: list[str]) -> None:
    a, b, c, d = workdirs
    ensure_workspace_mru_index([(a, _ts(1)), (b, _ts(2)), (c, _ts(3))], max_entries=3, root_key=ROOT_A)

    touch_workspace_mru(d, max_entries=3, used_at=_ts(10))

    assert load_workspace_mru(3) == [d, c, b]
    assert len(_read_index_file(config_dir)["paths"]) == 3


def test_touch_disabled_is_noop(config_dir: Path, workdirs: list[str]) -> None:
    ensure_workspace_mru_index([(workdirs[0], _ts())], max_entries=20, root_key=ROOT_A)
    before = _read_index_file(config_dir)

    assert touch_workspace_mru(workdirs[1], max_entries=0, used_at=_ts(10)) is False

    assert _read_index_file(config_dir) == before


def test_record_uses_keeps_primary_before_secondary_on_shared_timestamp(config_dir: Path, workdirs: list[str]) -> None:
    a, b, c, _ = workdirs
    ensure_workspace_mru_index([], max_entries=20, root_key=ROOT_A)

    assert record_workspace_mru_uses([a, b, c], max_entries=20, used_at=_ts(10)) is True

    assert load_workspace_mru(20) == [a, b, c]


def test_record_uses_rewrites_corrupt_index(config_dir: Path, workdirs: list[str]) -> None:
    ensure_workspace_mru_index([], max_entries=20, root_key=ROOT_A)
    workspace_mru_path().write_text("{not json", encoding="utf-8")

    assert record_workspace_mru_uses([workdirs[0]], max_entries=20, used_at=_ts()) is True

    assert load_workspace_mru(20) == [workdirs[0]]
    # The repair must keep the "corruption counts as backfilled" promise for
    # EVERY root — otherwise the now-valid file re-triggers the session scan.
    assert _read_index_file(config_dir)["backfilled_session_roots"] == ["*"]
    assert workspace_mru_backfilled_for(ROOT_A) is True
    assert workspace_mru_backfilled_for(ROOT_B) is True


def test_ensure_repairs_corrupt_index_without_seeding(config_dir: Path, workdirs: list[str]) -> None:
    """A corrupt file reaching ensure (check/ensure race) is repaired, not reseeded."""
    ensure_workspace_mru_index([(workdirs[0], _ts())], max_entries=20, root_key=ROOT_A)
    workspace_mru_path().write_text("{not json", encoding="utf-8")

    result = ensure_workspace_mru_index([(workdirs[1], _ts(10))], max_entries=20, root_key=ROOT_B)

    assert result == []
    data = _read_index_file(config_dir)
    assert data["backfilled_session_roots"] == ["*"]
    assert data["paths"] == []
    assert workspace_mru_backfilled_for(ROOT_B) is True


# ── load_workspace_mru ─────────────────────────────────────────────────


def test_load_missing_index_returns_empty(config_dir: Path) -> None:
    assert load_workspace_mru(20) == []


def test_load_trims_immediately_when_config_shrinks(config_dir: Path, workdirs: list[str]) -> None:
    a, b, c, d = workdirs
    ensure_workspace_mru_index([(a, _ts(4)), (b, _ts(3)), (c, _ts(2)), (d, _ts(1))], max_entries=20, root_key=ROOT_A)

    assert load_workspace_mru(2) == [a, b]


def test_load_filters_dirs_that_vanished_after_write(config_dir: Path, workdirs: list[str]) -> None:
    a, b, *_ = workdirs
    ensure_workspace_mru_index([(a, _ts(2)), (b, _ts(1))], max_entries=20, root_key=ROOT_A)

    Path(a).rmdir()

    assert load_workspace_mru(20) == [b]


def test_load_corrupt_json_returns_empty_without_raising(config_dir: Path) -> None:
    ensure_workspace_mru_index([], max_entries=20, root_key=ROOT_A)
    workspace_mru_path().write_text("{definitely not json", encoding="utf-8")

    assert load_workspace_mru(20) == []


def test_load_acquires_file_lock(config_dir: Path, workdirs: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_workspace_mru_index([(workdirs[0], _ts())], max_entries=20, root_key=ROOT_A)
    lock_paths: list[Path] = []
    real_lock = workspace_mru.FileLock

    class _SpyLock(real_lock):
        def __init__(self, path, timeout=None) -> None:
            lock_paths.append(Path(path))
            super().__init__(path, timeout)

    monkeypatch.setattr(workspace_mru, "FileLock", _SpyLock)

    assert load_workspace_mru(20) == [workdirs[0]]
    assert lock_paths == [config_dir / "workspace_mru.json.lock"]


# ── backfill bookkeeping ───────────────────────────────────────────────


def test_backfilled_for_missing_index_is_false(config_dir: Path) -> None:
    assert workspace_mru_backfilled_for(ROOT_A) is False


def test_backfilled_for_corrupt_index_reports_handled(config_dir: Path) -> None:
    ensure_workspace_mru_index([], max_entries=20, root_key=ROOT_A)
    workspace_mru_path().write_text("[]", encoding="utf-8")

    # Corruption must never re-trigger the expensive session scan.
    assert workspace_mru_backfilled_for(ROOT_A) is True
    assert workspace_mru_backfilled_for(ROOT_B) is True


def test_session_root_key_is_stable_and_spelling_insensitive(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()

    key = session_root_key(root)

    assert key.startswith("sha256:")
    assert session_root_key(str(root)) == key
    assert session_root_key(tmp_path / "other") != key


# ── schedule_workspace_mru_touches ─────────────────────────────────────


async def _drain_mru_tasks() -> None:
    while workspace_mru._BACKGROUND_TASKS:
        await asyncio.gather(*list(workspace_mru._BACKGROUND_TASKS), return_exceptions=True)


def test_schedule_skips_when_index_missing(config_dir: Path, workdirs: list[str]) -> None:
    async def run() -> None:
        schedule_workspace_mru_touches([workdirs[0]], max_entries=20, used_at=_ts())
        await _drain_mru_tasks()

    asyncio.run(run())

    assert not workspace_mru_exists()


def test_schedule_records_paths_in_background(config_dir: Path, workdirs: list[str]) -> None:
    a, b, *_ = workdirs
    ensure_workspace_mru_index([], max_entries=20, root_key=ROOT_A)

    async def run() -> None:
        schedule_workspace_mru_touches([a, b, "", a], max_entries=20, session_id="s-1", used_at=_ts(10))
        await _drain_mru_tasks()

    asyncio.run(run())

    assert load_workspace_mru(20) == [a, b]
    assert _read_index_file(config_dir)["paths"][0]["session_id"] == "s-1"


def test_schedule_failure_is_swallowed(config_dir: Path, workdirs: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_workspace_mru_index([], max_entries=20, root_key=ROOT_A)

    def _boom(*_args: object, **_kwargs: object) -> bool:
        raise OSError("disk full")

    monkeypatch.setattr(workspace_mru, "record_workspace_mru_uses", _boom)

    async def run() -> None:
        schedule_workspace_mru_touches([workdirs[0]], max_entries=20, used_at=_ts())
        await _drain_mru_tasks()

    asyncio.run(run())  # must not raise

    assert load_workspace_mru(20) == []
