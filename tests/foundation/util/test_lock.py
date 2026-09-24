# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the cross-platform :class:`FileLock` primitive."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from chrys.foundation.util.lock import FileLock


def test_filelock_serialises_critical_sections(tmp_path: Path) -> None:
    """Two threads racing on FileLock can't both be inside at once."""
    lock_path = tmp_path / "test.lock"
    inside = 0
    max_seen = 0
    barrier = threading.Barrier(4)
    state_lock = threading.Lock()

    def worker() -> None:
        nonlocal inside, max_seen
        barrier.wait()
        for _ in range(20):
            with FileLock(lock_path):
                with state_lock:
                    inside += 1
                    if inside > max_seen:
                        max_seen = inside
                time.sleep(0.001)
                with state_lock:
                    inside -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_seen == 1


def test_filelock_timeout(tmp_path: Path) -> None:
    """A second waiter with a tight timeout raises TimeoutError."""
    lock_path = tmp_path / "test.lock"
    holder = FileLock(lock_path)
    holder.acquire()
    try:
        with pytest.raises(TimeoutError), FileLock(lock_path, timeout=0.1):
            pass
    finally:
        holder.release()


def test_filelock_reentry_after_release(tmp_path: Path) -> None:
    """A second acquire after release succeeds without contention."""
    lock_path = tmp_path / "test.lock"
    lock = FileLock(lock_path)

    with lock:
        pass

    with lock:
        # If we got here, re-acquisition worked.
        assert lock_path.exists()


def test_filelock_release_is_idempotent(tmp_path: Path) -> None:
    """Calling release twice is safe."""
    lock_path = tmp_path / "test.lock"
    lock = FileLock(lock_path)
    lock.acquire()
    lock.release()
    lock.release()  # second call must not raise


def test_filelock_zero_timeout_fails_fast(tmp_path: Path) -> None:
    """``timeout=0`` raises immediately when the lock is contended."""
    lock_path = tmp_path / "test.lock"
    holder = FileLock(lock_path)
    holder.acquire()
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError), FileLock(lock_path, timeout=0):
            pass
        elapsed = time.monotonic() - start
        # Generous bound — we mainly want to confirm we didn't block.
        assert elapsed < 0.5
    finally:
        holder.release()


def test_filelock_short_positive_timeout_acquires_free_lock(tmp_path: Path) -> None:
    """Small positive timeouts are safe for fail-fast uncontended probes."""
    lock_path = tmp_path / "test.lock"

    with FileLock(lock_path, timeout=0.01):
        assert lock_path.exists()


# ── cross-process locking ─────────────────────────────────────────────────


_HOLDER_SCRIPT = textwrap.dedent(
    """
    import sys
    import time
    from pathlib import Path

    from chrys.foundation.util.lock import FileLock

    lock_path = Path(sys.argv[1])
    ready_path = Path(sys.argv[2])
    hold_seconds = float(sys.argv[3])

    with FileLock(lock_path):
        ready_path.write_text("held", encoding="utf-8")
        time.sleep(hold_seconds)
    """
)


def _spawn_holder(lock_path: Path, ready_path: Path, hold_seconds: float) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SCRIPT, str(lock_path), str(ready_path), str(hold_seconds)],
    )
    deadline = time.monotonic() + 10.0
    while not ready_path.exists():
        if time.monotonic() > deadline:
            proc.kill()
            proc.wait(timeout=5)
            msg = "holder subprocess never signalled readiness"
            raise RuntimeError(msg)
        time.sleep(0.02)
    return proc


def test_filelock_blocks_across_processes(tmp_path: Path) -> None:
    """A child process holding the lock blocks a sibling acquirer until released."""
    lock_path = tmp_path / "test.lock"
    ready_path = tmp_path / "holder.ready"

    proc = _spawn_holder(lock_path, ready_path, hold_seconds=1.0)
    try:
        # While the child holds the lock, a tight-timeout acquire must fail.
        with pytest.raises(TimeoutError), FileLock(lock_path, timeout=0.1):
            pass
        # And a generous-timeout acquire must succeed once the child exits.
        proc.wait(timeout=10)
        assert proc.returncode == 0
        with FileLock(lock_path, timeout=2.0):
            pass
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL semantics are POSIX-specific")
def test_filelock_released_when_holder_killed(tmp_path: Path) -> None:
    """A SIGKILL'd holder must let the next waiter acquire — OS reclaims the lock."""
    lock_path = tmp_path / "test.lock"
    ready_path = tmp_path / "holder.ready"

    # Holder sleeps long enough that we definitely kill it mid-hold.
    proc = _spawn_holder(lock_path, ready_path, hold_seconds=30.0)
    try:
        # Confirm the lock is genuinely held right now.
        with pytest.raises(TimeoutError), FileLock(lock_path, timeout=0.1):
            pass
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
        # OS should have dropped the flock when the fd was reaped.
        with FileLock(lock_path, timeout=2.0):
            pass
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.skipif(sys.platform == "win32", reason="mkfifo/O_NONBLOCK are POSIX-specific")
def test_filelock_refuses_fifo_without_hanging(tmp_path: Path) -> None:
    """A FIFO planted at the lock path must be rejected, never block open()."""
    lock_path = tmp_path / "planted.lock"
    os.mkfifo(lock_path)

    # A same-uid attacker can pre-plant a FIFO at the predictable ``.lock``
    # path. ``open("ab")`` would block *inside open* waiting for a reader,
    # before the acquire timeout begins — freezing the caller (and the event
    # loop, for synchronous locks). Run in a thread with a bounded join so a
    # regression that hangs fails loudly instead of stalling the suite.
    result: dict[str, object] = {"error": None, "done": False}

    def attempt() -> None:
        try:
            with FileLock(lock_path, timeout=5.0):
                pass
        except BaseException as exc:  # record for the assertion, never re-raise
            result["error"] = exc
        finally:
            result["done"] = True

    worker = threading.Thread(target=attempt, daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert result["done"], "FileLock hung on a FIFO lock path instead of refusing it"
    assert isinstance(result["error"], OSError)
    assert not isinstance(result["error"], TimeoutError)


@pytest.mark.skipif(sys.platform == "win32", reason="O_NOFOLLOW/symlink semantics are POSIX-specific")
def test_filelock_refuses_symlinked_lock_path(tmp_path: Path) -> None:
    """A symlink at the lock path must not be followed (O_NOFOLLOW)."""
    target = tmp_path / "real_target"
    target.write_bytes(b"")
    lock_path = tmp_path / "planted.lock"
    lock_path.symlink_to(target)

    with pytest.raises(OSError) as excinfo, FileLock(lock_path, timeout=1.0):
        pass
    assert not isinstance(excinfo.value, TimeoutError)
