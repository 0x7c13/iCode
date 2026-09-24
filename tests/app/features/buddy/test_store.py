# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The buddy save file: what is on disk, what is refused, and what survives concurrent writers."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from chrys.app.features.buddy import store as store_module
from chrys.app.features.buddy.store import SAVE_VERSION, BuddyStore
from chrys.foundation.platform import get_platform
from tests.support.buddies import a_record
from tests.support.threads import run_to_the_end

if TYPE_CHECKING:
    from pathlib import Path

    from chrys.app.features.buddy.model import BuddyRecord


# Every wait below is bounded by this: far more than the work needs, and well inside the per-test timeout.
_WAIT_SECONDS = 20


@pytest.fixture
def store(tmp_path: Path) -> BuddyStore:
    return BuddyStore(tmp_path / "buddy" / "buddy.json")


def _pet(record: BuddyRecord | None) -> BuddyRecord | None:
    return replace(record, pets=record.pets + 1) if record is not None else None


def _never_written(path: Path, payload: str) -> bytes:
    raise AssertionError(f"nothing changed, but {path.name} was written")


def _saved(store: BuddyStore, record: BuddyRecord) -> BuddyRecord:
    saved = store.update(lambda _current: record)
    assert saved == record
    return record


def test_the_default_store_lives_under_the_config_directory() -> None:
    assert BuddyStore().path == get_platform().config_dir / "extras" / "buddy" / "buddy.json"


def test_nothing_saved_reads_as_no_buddy_and_creates_no_file(store: BuddyStore) -> None:
    assert store.load() is None
    assert not store.path.parent.exists()


def test_a_saved_buddy_reads_back_from_a_signed_plain_envelope(store: BuddyStore) -> None:
    record = _saved(store, a_record(turns=4, pets=2, name="Señor 🦉"))
    assert store.load() == record

    envelope = json.loads(store.path.read_text(encoding="utf-8"))
    assert set(envelope) == {"v", "buddy", "sig"}
    assert envelope["v"] == SAVE_VERSION
    assert envelope["buddy"] == record.to_json()
    assert store.backup_path.read_bytes() == store.path.read_bytes()


def test_a_change_that_changes_nothing_writes_nothing(store: BuddyStore, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _saved(store, a_record())
    store.path.unlink()
    store.backup_path.unlink()

    assert store.update(lambda current: current) is None
    assert store.update(lambda _current: None) is None
    assert not store.path.exists()

    _saved(store, record)
    with monkeypatch.context() as patched:
        patched.setattr(store_module, "atomic_write_text", _never_written)
        assert store.update(lambda current: replace(current, pets=current.pets)) == record


@pytest.mark.parametrize(
    "edit",
    [
        lambda envelope: envelope["buddy"].update(rarity="SSR"),
        lambda envelope: envelope["buddy"].update(turns=99_999),
        lambda envelope: envelope.update(sig="0" * 64),
        lambda envelope: envelope.update(sig="caf\udce9"),
        lambda envelope: envelope.update(sig=None),
        lambda envelope: envelope.pop("sig"),
        lambda envelope: envelope.update(v=SAVE_VERSION + 1),
        lambda envelope: envelope.update(v=1),
    ],
)
def test_an_edited_file_is_refused(store: BuddyStore, edit) -> None:
    _saved(store, a_record())
    envelope = json.loads(store.path.read_text(encoding="utf-8"))
    edit(envelope)
    for path in (store.path, store.backup_path):
        path.write_text(json.dumps(envelope), encoding="utf-8")

    assert store.load() is None


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"{ not json", id="not-json"),
        pytest.param(b"[]", id="not-an-envelope"),
        pytest.param(b"\xff\xfe\x00 not text", id="not-text"),
        # The id matters: pytest would otherwise spell all of it into the test's name.
        pytest.param(b"[" * 200_000, id="nested-too-deep"),
        # Anything an earlier release wrote: opaque bytes, or a document with no envelope around it.
        pytest.param(b"AAECAwQFBgcICQoLDA0ODw==", id="opaque-bytes"),
        pytest.param(json.dumps({"name": "Old", "level": 3}).encode(), id="bare-document"),
        pytest.param(json.dumps(a_record().to_json()).encode(), id="record-without-envelope"),
    ],
)
def test_anything_that_is_not_a_current_save_reads_as_no_buddy(store: BuddyStore, content: bytes) -> None:
    store.path.parent.mkdir(parents=True)
    store.path.write_bytes(content)

    assert store.load() is None
    # The next hatch simply takes the file over.
    assert _saved(store, a_record()) == store.load()


def test_an_unreadable_primary_falls_back_to_the_second_copy_until_the_next_change(store: BuddyStore) -> None:
    record = _saved(store, a_record(pets=5))
    store.path.write_text("{ torn", encoding="utf-8")
    assert store.load() == record

    store.path.unlink()
    assert store.load() == record

    assert store.update(_pet) == replace(record, pets=6)
    assert json.loads(store.path.read_text(encoding="utf-8"))["buddy"]["pets"] == 6


def test_a_failed_second_copy_does_not_fail_the_change(store: BuddyStore, monkeypatch: pytest.MonkeyPatch) -> None:
    _saved(store, a_record())
    real_write = store_module.atomic_write_text

    def write_primary_only(path: Path, payload: str) -> bytes:
        if path == store.backup_path:
            raise PermissionError("second copy is read-only")
        return real_write(path, payload)

    monkeypatch.setattr(store_module, "atomic_write_text", write_primary_only)

    assert store.update(_pet) == a_record(pets=1)
    assert store.load() == a_record(pets=1)


class _RefusingWriter:
    """The real write, except that the first *refusals* replaces of the primary are refused as Windows refuses them."""

    def __init__(self, store: BuddyStore, refusals: int) -> None:
        self._primary = store.path
        self._refusals = refusals
        self.attempts = 0

    def __call__(self, path: Path, payload: str) -> bytes:
        if path == self._primary:
            self.attempts += 1
            if self.attempts <= self._refusals:
                raise PermissionError("The process cannot access the file because it is being used by another process")
        return _real_write(path, payload)


_real_write = store_module.atomic_write_text


def _on(monkeypatch: pytest.MonkeyPatch, os_name: str, writer: _RefusingWriter) -> None:
    monkeypatch.setattr(store_module, "get_platform", lambda: replace(get_platform(), os_name=os_name))
    monkeypatch.setattr(store_module, "_REPLACE_RETRY_SECONDS", 0)
    monkeypatch.setattr(store_module, "atomic_write_text", writer)


def test_on_windows_a_replace_refused_while_a_reader_has_the_file_is_tried_again(
    store: BuddyStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _saved(store, a_record())
    writer = _RefusingWriter(store, refusals=store_module._REPLACE_ATTEMPTS - 1)
    _on(monkeypatch, "windows", writer)

    assert store.update(_pet) == a_record(pets=1)

    assert writer.attempts == store_module._REPLACE_ATTEMPTS
    assert store.load() == a_record(pets=1)
    assert store.backup_path.read_bytes() == store.path.read_bytes()


def test_on_windows_a_replace_refused_every_time_gives_the_change_up(
    store: BuddyStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _saved(store, a_record())
    writer = _RefusingWriter(store, refusals=store_module._REPLACE_ATTEMPTS)
    _on(monkeypatch, "windows", writer)

    with pytest.raises(PermissionError):
        store.update(_pet)

    assert writer.attempts == store_module._REPLACE_ATTEMPTS
    assert store.load() == record


@pytest.mark.parametrize("os_name", ["macos", "linux"])
def test_elsewhere_a_refused_replace_is_final_at_once(
    store: BuddyStore, monkeypatch: pytest.MonkeyPatch, os_name: str
) -> None:
    record = _saved(store, a_record())
    writer = _RefusingWriter(store, refusals=1)
    _on(monkeypatch, os_name, writer)

    with pytest.raises(PermissionError):
        store.update(_pet)

    assert writer.attempts == 1
    assert store.load() == record


def test_a_failed_write_leaves_the_saved_buddy_as_it_was(store: BuddyStore, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _saved(store, a_record(pets=7))

    def fail(path: Path, payload: str) -> bytes:
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(store_module, "atomic_write_text", fail)
        with pytest.raises(OSError, match="disk full"):
            store.update(_pet)

    assert store.load() == record


def test_a_wedged_writer_times_the_next_one_out(store: BuddyStore, monkeypatch: pytest.MonkeyPatch) -> None:
    _saved(store, a_record())
    monkeypatch.setattr(store_module, "_LOCK_TIMEOUT_SECONDS", 0.2)
    holding, release = threading.Event(), threading.Event()

    def wedge(record: BuddyRecord | None) -> BuddyRecord | None:
        holding.set()
        assert release.wait(_WAIT_SECONDS)
        return record

    holder = threading.Thread(target=store.update, args=(wedge,))
    holder.start()
    try:
        assert holding.wait(_WAIT_SECONDS)
        with pytest.raises(TimeoutError):
            store.update(_pet)
    finally:
        release.set()
        holder.join(_WAIT_SECONDS)
    assert not holder.is_alive()
    assert store.load() == a_record()


def test_threads_petting_at_once_lose_no_pets(store: BuddyStore) -> None:
    _saved(store, a_record())
    threads, pets_each = 8, 25
    start = threading.Barrier(threads)

    def worker() -> None:
        start.wait(_WAIT_SECONDS)
        for _ in range(pets_each):
            store.update(_pet)

    run_to_the_end([threading.Thread(target=worker) for _ in range(threads)], within=_WAIT_SECONDS)

    assert store.load() == a_record(pets=threads * pets_each)


def test_a_change_of_one_field_keeps_a_concurrent_change_of_another(store: BuddyStore) -> None:
    _saved(store, a_record())

    def petter() -> None:
        for _ in range(50):
            store.update(_pet)

    def muter() -> None:
        for turn in range(51):
            store.update(lambda record, muted=turn % 2 == 0: replace(record, muted=muted) if record else None)

    run_to_the_end([threading.Thread(target=petter), threading.Thread(target=muter)], within=_WAIT_SECONDS)

    assert store.load() == a_record(pets=50, muted=True)


_PETTING_PROCESS = textwrap.dedent(
    """
    import sys
    from dataclasses import replace
    from pathlib import Path

    from chrys.app.features.buddy.store import BuddyStore

    store = BuddyStore(Path(sys.argv[1]))
    for _ in range(int(sys.argv[2])):
        store.update(lambda record: replace(record, pets=record.pets + 1))
    """
)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Interpreter startup plus msvcrt lock polling under a loaded Windows runner outlasts the store's "
    "lock timeout. The threaded tests take the same FileLock path; the lock's cross-process behaviour on "
    "Windows is tested with the lock itself.",
)
def test_processes_petting_at_once_lose_no_pets(store: BuddyStore) -> None:
    _saved(store, a_record())
    processes, pets_each = 4, 10

    workers = [
        subprocess.Popen(
            [sys.executable, "-c", _PETTING_PROCESS, str(store.path), str(pets_each)],
            stdin=subprocess.DEVNULL,
        )
        for _ in range(processes)
    ]
    deadline = time.monotonic() + _WAIT_SECONDS
    try:
        for worker in workers:
            assert worker.wait(timeout=max(0.0, deadline - time.monotonic())) == 0
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()

    assert store.load() == a_record(pets=processes * pets_each)
