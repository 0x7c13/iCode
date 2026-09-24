# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cross-session mutation coordination registry.

Resolves the chrys-vs-chrys attribution case: N chrys sessions working in
the same working tree, each publishing what it *provably* wrote, each
subtracting peers' proven writes from its own window inferences.

Storage layout::

    {session_root}/attribution/{repo_key}/{session_id}.json

- ``repo_key`` = first 16 hex chars of sha256 over the canonicalized
  working-tree root (``.git`` directory *or* file — linked worktrees and
  submodules resolve to their own tree).  Outside any repo, a path keys
  by its own directory — a pure function of the path, so independent
  sessions land in the same tree.  Keying by tree root is for
  *discovery* only; matching itself is path-exact.
- One file per (tree, session), **single writer** (the owning session),
  replaced atomically.  Readers tolerate missing/partial/undecodable
  files by skipping them — no cross-process locks.  Within the session
  the registry is in-memory state behind one lock and the file is its
  serialization (no read-modify-write of the file itself).
- Files outlive their session: clean shutdown stamps ``closed_at`` but
  does not delete (late rollbacks in other sessions still need the
  claims).  Removal happens only via opportunistic GC — on write,
  peer files whose mtime is older than :data:`REGISTRY_GC_AGE_SECONDS`
  are deleted.

Reclassification (read side) is **retroactive, idempotent, and runs at
every opportunity** — own finalize, pre-summary display refresh, and
rollback-plan construction.  Matching uses write *intervals* (never
point timestamps) on *canonical* path keys (``realpath`` +
``normcase``).  Conflict rules per overlapping path:

    ========== ============================== ===========================
    local row  evidence                       result
    ========== ============================== ===========================
    ASSUMED    content identity               demote to FOREIGN
    ASSUMED    overlap without identity       stays ASSUMED, contested
    PROVEN     any overlap                    stays PROVEN, contested
    ========== ============================== ===========================

Content identity = hashes present and equal, or both sides gone.
Skip-marked sides and MOVEs never demote (they fall to ``contested``).
Coordination must never break tool execution: every filesystem failure
is swallowed and logged at debug level.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import atomic_write_text
from chrys.service.mutations.types import (
    FileMutation,
    MutationProvenance,
    RollbackExclusionReason,
    RollbackPlan,
    RollbackWarning,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from chrys.service.mutations.tracker import MutationTracker
    from chrys.service.mutations.types import MutationLog

logger = logging.getLogger(__name__)

REGISTRY_SCHEMA = 1
MAX_CLAIMS = 1000
REGISTRY_GC_AGE_SECONDS = 7 * 24 * 60 * 60
ATTRIBUTION_DIR_NAME = "attribution"

# In-flight peer warning (RollbackWarning.code).
PEER_ACTIVE_WINDOW_WARNING = "peer_active_window"


# ---------------------------------------------------------------------------
# Path canonicalization and working-tree discovery
# ---------------------------------------------------------------------------


def fold_case(text: str) -> str:
    """The case fold ``canonical_key`` applies — usable on path fragments.

    ``normcase`` folds on Windows ONLY — darwin needs the explicit fold,
    since default APFS/HFS+ volumes are case-insensitive too.  (On a
    case-SENSITIVE mac volume case twins then merely share a registry
    namespace; matching still demands interval overlap plus content
    identity — the same trade Windows makes for case-sensitive NTFS
    directories.)
    """
    folded = os.path.normcase(text)
    if get_platform().is_macos:
        folded = folded.lower()
    return folded


def canonical_key(path: str) -> str:
    """Canonical matching key: ``realpath`` + case folding.

    The codebase's usual ``abspath``+``normpath`` misses case-variant
    launches (Windows, default macOS volumes) and symlinked routes; a
    miss only costs a demotion, but it is cheap to avoid.
    """
    try:
        real = os.path.realpath(path)
    except OSError:
        real = os.path.abspath(path)
    return fold_case(real)


def resolve_tree_root(path: str) -> str | None:
    """Nearest ancestor whose ``.git`` exists — directory *or* file.

    File paths are normalized to their containing directory before the
    ancestor walk.  Linked worktrees/submodules pair up by their own
    ``.git`` file, so they stay separate from their parent repository.
    """
    current = os.path.abspath(path)
    if not os.path.isdir(current):
        current = os.path.dirname(current)
    while True:
        if os.path.exists(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def repo_key_for_root(tree_root: str) -> str:
    """First 16 hex chars of sha256 over the canonicalized tree root."""
    return hashlib.sha256(canonical_key(tree_root).encode("utf-8")).hexdigest()[:16]


def _is_pid_alive(pid: int) -> bool:
    """Cross-platform liveness probe (psutil, else a native fallback).

    A peer whose recorded pid is no longer alive counts as closed —
    crash leftovers must not warn until GC.  psutil ships only with the
    ``tui`` extra, but coordination is service-level and default-on, so
    headless installs fall back to ``kill(pid, 0)`` on POSIX and a
    ctypes ``OpenProcess`` probe on Windows.
    """
    if pid <= 0:
        return False
    try:
        import psutil
    except ImportError:
        if get_platform().is_windows:
            return _windows_pid_alive(pid)
        return _posix_pid_alive(pid)
    try:
        return psutil.pid_exists(pid)
    except Exception:  # pragma: no cover - psutil failure → assume alive (conservative)
        return True


def _posix_pid_alive(pid: int) -> bool:
    """psutil-free liveness probe for POSIX.

    ``os.kill(pid, 0)`` is a pure existence probe on POSIX only — on
    Windows any signal outside the CTRL_*_EVENT pair TERMINATES the
    target process, so this must never run there (the platform guard
    is defense in depth on top of the caller's dispatch).
    """
    if get_platform().is_windows:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # PermissionError et al.: pid exists (or unknowable)
        return True
    return True


def _windows_pid_alive(pid: int) -> bool:  # pragma: no cover - exercised on Windows CI only
    """psutil-free liveness probe for Windows (ctypes, no signals).

    ``OpenProcess`` failing with ERROR_ACCESS_DENIED means the pid
    exists under another user; any other open failure means no such
    process.  An open handle can still name an already-exited process,
    so the exit code must read STILL_ACTIVE.  Unexpected ctypes
    failures stay conservative (alive) — a false "peer crashed"
    demotion is worse than a lingering warning.
    """
    if not get_platform().is_windows:
        return True
    try:
        import ctypes
        from ctypes import wintypes

        ctypes_module = cast(Any, ctypes)

        process_query_limited_information = 0x1000
        still_active = 259
        error_access_denied = 5

        kernel32 = ctypes_module.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return ctypes_module.get_last_error() == error_access_denied
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return True


def _strict_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    """Strict interval overlap — no slack, touching endpoints don't pair.

    Both intervals are conservative supersets of the true write instant
    taken from the same machine's wall clock, so a genuinely in-window
    peer write always overlaps strictly; slack could only manufacture
    pairings that never happened.
    """
    return a0 < b1 and b0 < a1


# ---------------------------------------------------------------------------
# Registry records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationClaim:
    """One published proven write: path identity, after-state, write interval.

    ``key``/``old_key`` are canonical (:func:`canonical_key`) forms
    carried alongside the display paths.  ``after_exists`` is skip-aware,
    mirroring ``FileMutation``: it distinguishes deletion from a
    policy-skipped backup — both read ``after_hash: null``.
    ``[t0, t1]`` bounds when the disk write actually occurred.
    """

    path: str
    key: str
    after_hash: str | None
    after_exists: bool
    t0: float
    t1: float
    source: str = ""
    old_path: str | None = None
    old_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "path": self.path,
            "key": self.key,
            "after_hash": self.after_hash,
            "after_exists": self.after_exists,
            "t0": self.t0,
            "t1": self.t1,
            "source": self.source,
        }
        if self.old_path is not None:
            d["old_path"] = self.old_path
        if self.old_key is not None:
            d["old_key"] = self.old_key
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MutationClaim:
        return cls(
            path=str(d["path"]),
            key=str(d.get("key") or canonical_key(str(d["path"]))),
            after_hash=d.get("after_hash"),
            after_exists=bool(d.get("after_exists", d.get("after_hash") is not None)),
            t0=float(d["t0"]),
            t1=float(d["t1"]),
            source=str(d.get("source", "")),
            old_path=d.get("old_path"),
            old_key=d.get("old_key"),
        )


@dataclass
class ObservationWindow:
    """One advertised observation window (shell / implicit tool call).

    Opened in ``prepare_mutation_tracking`` (``end: None``) *before* any
    observation (git capture, pre-scan, snapshots) so the advertised
    window is a superset of the observation; closed in finalize.  Peers
    use still-open windows for the in-flight rollback warning.
    """

    id: str
    scope: list[str]
    start: float
    end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "scope": list(self.scope), "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObservationWindow:
        return cls(
            id=str(d.get("id", "")),
            scope=[str(s) for s in d.get("scope", [])],
            start=float(d["start"]),
            end=None if d.get("end") is None else float(d["end"]),
        )


@dataclass
class PeerRegistry:
    """Parsed peer registry file (tolerant read)."""

    session_id: str
    pid: int
    closed_at: float | None
    windows: list[ObservationWindow]
    claims: list[MutationClaim]

    @property
    def is_active(self) -> bool:
        """Open for business: not cleanly closed AND its process is alive."""
        return self.closed_at is None and _is_pid_alive(self.pid)


def parse_registry_payload(payload: str) -> PeerRegistry | None:
    """Parse one registry file's text; ``None`` on any structural problem.

    Readers must tolerate missing/partial/undecodable files by skipping
    them — a peer may be mid-replace or of a future schema.
    """
    try:
        data = json.loads(payload)
        if not isinstance(data, dict) or int(data.get("schema", 0)) != REGISTRY_SCHEMA:
            return None
        closed_raw = data.get("closed_at")
        windows = [ObservationWindow.from_dict(w) for w in data.get("windows", [])]
        claims = [MutationClaim.from_dict(c) for c in data.get("claims", [])]
        return PeerRegistry(
            session_id=str(data.get("session_id", "")),
            pid=int(data.get("pid", 0)),
            closed_at=None if closed_raw is None else float(closed_raw),
            windows=windows,
            claims=claims,
        )
    except Exception:
        return None


@dataclass
class _TreeState:
    """In-memory single-writer state for one (tree, session) file."""

    windows: dict[str, ObservationWindow] = field(default_factory=dict)
    claims: list[MutationClaim] = field(default_factory=list)
    loaded_existing: bool = False


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class MutationCoordinator:
    """Owns cross-session attribution for one session.

    Built alongside the :class:`MutationTracker` at engine construction
    (it needs session identity and settings the tracker never sees);
    the tracker itself stays coordination-unaware.  All methods are
    thread-safe (pipeline calls arrive from executor threads) and all
    registry I/O failures degrade to a no-op with a debug log.
    """

    def __init__(
        self,
        *,
        registry_root: str | Path,
        session_id: str,
        enabled: bool = True,
        pid: int | None = None,
        now: Callable[[], float] = time.time,
        max_claims: int = MAX_CLAIMS,
        gc_age_seconds: float = REGISTRY_GC_AGE_SECONDS,
    ) -> None:
        self._root = Path(registry_root)
        self._session_id = session_id
        self._enabled = enabled
        self._pid = os.getpid() if pid is None else pid
        self._now = now
        self._max_claims = max_claims
        self._gc_age = gc_age_seconds
        self._lock = threading.Lock()
        # repo_key -> single-writer state (this session's own file).
        self._states: dict[str, _TreeState] = {}
        # window id -> repo_keys the window was published under.
        self._window_trees: dict[str, list[str]] = {}
        # Directory (canonical) -> resolved tree root; cheap memo for the
        # hot path (every mutation path resolves its tree on publish and
        # reclassify).
        self._tree_cache: dict[str, str | None] = {}
        # Short-circuit signatures for reclassify: peer-dir signature per
        # repo_key + local log signature at the last completed pass.
        self._last_peer_sig: dict[str, tuple] = {}
        self._last_local_sig: tuple | None = None
        self._unsaved_reclassification = False
        self._closed_at: float | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # -- Tree routing -------------------------------------------------

    def _tree_root_for(self, path: str, fallback_root: str | None) -> str | None:
        # Window scope paths are directories; claim paths are files.
        absolute = os.path.abspath(path)
        directory = absolute if os.path.isdir(absolute) else (os.path.dirname(absolute) or absolute)
        cache_key = os.path.normcase(directory)
        if cache_key not in self._tree_cache:
            self._tree_cache[cache_key] = resolve_tree_root(directory)
        root = self._tree_cache[cache_key]
        if root is not None:
            return root
        # Non-git paths key by their OWN directory, never the caller's
        # fallback root: the registry key must be a pure function of
        # the path — sessions with different primary workspaces touching
        # the same untracked directory must land in the same tree or
        # they never see each other's claims.  ``fallback_root`` only
        # gates WHETHER untracked paths participate (None → the caller
        # has no workspace context → they don't).
        if fallback_root:
            return directory
        return None

    def _repo_keys_for_paths(self, paths: Iterable[str], fallback_root: str | None) -> dict[str, str]:
        """Map repo_key -> tree root for every resolvable path."""
        keys: dict[str, str] = {}
        for p in paths:
            root = self._tree_root_for(p, fallback_root)
            if root is not None:
                keys.setdefault(repo_key_for_root(root), root)
        return keys

    # -- Own-file serialization ----------------------------------------

    def _tree_dir(self, repo_key: str) -> Path:
        return self._root / repo_key

    def _own_file(self, repo_key: str) -> Path:
        return self._tree_dir(repo_key) / f"{self._session_id}.json"

    def _state_for(self, repo_key: str) -> _TreeState:
        """Get (lazily creating) the in-memory state for one tree.

        On first touch, merge any existing file left by a previous
        process of this session (resume): claims are kept — peers'
        late rollbacks still need them — and stale open windows are
        closed at the file's mtime (the last instant that process
        provably lived); the dead pid would already read as closed, but
        the file is about to be re-stamped with *our* live pid.
        """
        state = self._states.get(repo_key)
        if state is not None:
            return state
        state = _TreeState()
        own = self._own_file(repo_key)
        try:
            if own.exists():
                stamp = own.stat().st_mtime
                prior = parse_registry_payload(own.read_text(encoding="utf-8"))
                if prior is not None:
                    state.claims = prior.claims[-self._max_claims :]
                    for w in prior.windows:
                        if w.end is None:
                            w.end = max(w.start, stamp)
                        state.windows[w.id] = w
                    state.loaded_existing = True
        except OSError:
            logger.debug("Coordination: could not merge prior registry file %s", own, exc_info=True)
        self._states[repo_key] = state
        return state

    def _write_state(self, repo_key: str) -> None:
        state = self._state_for(repo_key)
        payload = {
            "session_id": self._session_id,
            "pid": self._pid,
            "schema": REGISTRY_SCHEMA,
            # Sticky close stamp: a straggler racing close() (an
            # in-flight finalize's close_window / publish_claims) must
            # not revive the file as open — close() itself never
            # re-stamps (idempotence guard), so the stamp would be lost
            # for good and peers would read a live pid as "active".
            "closed_at": self._closed_at,
            "windows": [w.to_dict() for w in state.windows.values()],
            "claims": [c.to_dict() for c in state.claims],
        }
        try:
            self._tree_dir(repo_key).mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._own_file(repo_key), json.dumps(payload))
        except OSError:
            logger.debug("Coordination: registry write failed for %s", repo_key, exc_info=True)
            return
        self._gc_tree(repo_key)

    def _gc_tree(self, repo_key: str) -> None:
        """Opportunistic on-write GC: drop peer files older than the age cap."""
        cutoff = self._now() - self._gc_age
        own_name = f"{self._session_id}.json"
        try:
            with os.scandir(self._tree_dir(repo_key)) as it:
                for entry in it:
                    if entry.name == own_name or not entry.name.endswith(".json"):
                        continue
                    with contextlib.suppress(OSError):
                        if entry.stat().st_mtime < cutoff:
                            os.unlink(entry.path)
        except OSError:
            pass

    # -- Write side ------------------------------------------------------

    def open_window(self, scope_paths: list[str], *, fallback_root: str | None = None) -> str | None:
        """Advertise an observation window; returns its id (None when disabled).

        Must be called as the *first* operation of an observation
        (before git capture, shell pre-scan, or any snapshot) so the
        advertised window is a superset — a peer write in a pre-scan
        gap would otherwise be attributed by the diff yet never overlap
        our interval.
        """
        if not self._enabled or not scope_paths:
            return None
        window = ObservationWindow(
            id=uuid.uuid4().hex, scope=[os.path.abspath(p) for p in scope_paths], start=self._now()
        )
        with self._lock:
            keys = self._repo_keys_for_paths(scope_paths, fallback_root)
            if not keys:
                return None
            self._window_trees[window.id] = list(keys)
            for repo_key in keys:
                self._state_for(repo_key).windows[window.id] = window
                self._write_state(repo_key)
        return window.id

    def close_window(self, window_id: str | None) -> None:
        """Stamp the window's end and republish the affected tree files."""
        if not self._enabled or not window_id:
            return
        with self._lock:
            end = self._now()
            for repo_key in self._window_trees.pop(window_id, []):
                window = self._states[repo_key].windows.get(window_id)
                if window is not None:
                    window.end = end
                self._write_state(repo_key)

    def publish_claims(self, mutations: list[FileMutation], *, fallback_root: str | None = None) -> int:
        """Publish claims for the eligible subset of ``mutations``.

        Eligibility is decided here for the *evidence* conditions
        (proven provenance, provable net change); the *authorship*
        condition — did the tool actually execute and succeed for
        write/edit — is the caller's to enforce by not passing such
        mutations at all (the pipeline sees the tool result; we don't).
        Over-publishing falsely poisons peers' rollbacks via contested
        marks, while under-publishing only costs missed demotions.
        """
        if not self._enabled or not mutations:
            return 0
        eligible = [m for m in mutations if self._claim_eligible(m)]
        if not eligible:
            return 0
        published = 0
        with self._lock:
            touched: set[str] = set()
            for m in eligible:
                t0 = m.t_start if m.t_start is not None else m.timestamp
                t1 = m.t_end if m.t_end is not None else self._now()
                claim = MutationClaim(
                    path=m.path,
                    key=canonical_key(m.path),
                    after_hash=m.after_hash,
                    after_exists=m.after_hash is not None or m.after_skip is not None,
                    t0=t0,
                    t1=t1,
                    source=m.source.value,
                    old_path=m.old_path,
                    old_key=canonical_key(m.old_path) if m.old_path else None,
                )
                # A MOVE names two paths, but a same-repo rename (the
                # common case) resolves both to one tree — dedupe or the
                # claim lands twice and burns MAX_CLAIMS double.
                repo_keys: set[str] = set()
                for path in (m.path, m.old_path) if m.old_path else (m.path,):
                    root = self._tree_root_for(path, fallback_root)
                    if root is not None:
                        repo_keys.add(repo_key_for_root(root))
                for repo_key in sorted(repo_keys):
                    state = self._state_for(repo_key)
                    state.claims.append(claim)
                    if len(state.claims) > self._max_claims:
                        # Bounded by count, not age — FIFO.
                        del state.claims[: len(state.claims) - self._max_claims]
                    touched.add(repo_key)
                    published += 1
            for repo_key in touched:
                self._write_state(repo_key)
        return published

    @staticmethod
    def _claim_eligible(m: FileMutation) -> bool:
        """Evidence test: proven provenance + provable net change.

        The change test is the same skip-aware rule as ``is_net_zero``:
        hash inequality or existence flip.  A fully-skipped MODIFY
        proves nothing and publishes nothing.  ``ASSUMED`` inferences
        are never published.
        """
        if m.provenance is not MutationProvenance.PROVEN:
            return False
        before_exists = m.before_hash is not None or m.before_skip is not None
        after_exists = m.after_hash is not None or m.after_skip is not None
        return m.before_hash != m.after_hash or before_exists != after_exists

    def close(self) -> None:
        """Stamp ``closed_at`` on every tree file this session wrote.

        Idempotent; files are never deleted here (peers' late rollbacks
        still need the claims — removal is GC's job).
        """
        if not self._enabled or self._closed_at is not None:
            return
        with self._lock:
            if self._closed_at is not None:
                return
            end = self._now()
            self._closed_at = end
            for repo_key, state in self._states.items():
                for window in state.windows.values():
                    if window.end is None:
                        window.end = end
                self._write_state(repo_key)

    # -- Read side -------------------------------------------------------

    def _peer_dir_signature(self, repo_key: str) -> tuple:
        """Cheap change signature over peer files in one tree dir."""
        own_name = f"{self._session_id}.json"
        entries: list[tuple[str, int, int]] = []
        try:
            with os.scandir(self._tree_dir(repo_key)) as it:
                for entry in it:
                    if entry.name == own_name or not entry.name.endswith(".json"):
                        continue
                    with contextlib.suppress(OSError):
                        st = entry.stat()
                        entries.append((entry.name, st.st_mtime_ns, st.st_size))
        except OSError:
            return ()
        entries.sort()
        return tuple(entries)

    def _read_peers(self, repo_key: str) -> list[PeerRegistry]:
        own_name = f"{self._session_id}.json"
        peers: list[PeerRegistry] = []
        try:
            with os.scandir(self._tree_dir(repo_key)) as it:
                names = [e.name for e in it if e.name.endswith(".json") and e.name != own_name]
        except OSError:
            return []
        for name in names:
            try:
                payload = (self._tree_dir(repo_key) / name).read_text(encoding="utf-8")
            except OSError:
                continue
            peer = parse_registry_payload(payload)
            if peer is not None and peer.session_id != self._session_id:
                peers.append(peer)
        return peers

    @staticmethod
    def _local_log_signature(log: MutationLog) -> tuple:
        # The completed-interval count is load-bearing: file-tool rows
        # exist from record() time with an OPEN interval (t_end unset)
        # and only become matchable when finalize stamps t_end.  A read
        # refresh (/diff, ACP) landing mid-call would otherwise cache a
        # signature the t_end stamp cannot change — turn/row counts stay
        # identical — and the finalize reclassify would short-circuit
        # past the completed row forever.
        rows = 0
        complete = 0
        for turn in log.periods:
            rows += len(turn.mutations)
            for m in turn.mutations:
                if m.t_end is not None:
                    complete += 1
        return (len(log.periods), rows, complete)

    def reclassify(self, tracker: MutationTracker, *, force: bool = False, fallback_root: str | None = None) -> bool:
        """Retroactively re-attribute the tracker's log against peer claims.

        Idempotent; safe to run at every opportunity.  Returns True when
        any row changed (caller persists via the normal session-state
        save).  ``force=True`` bypasses the no-change short-circuit —
        used at rollback-plan construction, the authoritative last check
        before the only destructive consumer acts.
        """
        if not self._enabled:
            return False

        # Collect the candidate rows and their trees under the tracker
        # lock; read peer files outside it; apply matches under it again.
        def _collect(log: MutationLog) -> tuple[tuple, list[str]]:
            paths: list[str] = []
            for turn in log.periods:
                for m in turn.mutations:
                    if m.provenance is MutationProvenance.FOREIGN:
                        continue
                    if m.t_start is None or m.t_end is None:
                        continue  # legacy row — no interval, never matches
                    paths.append(m.path)
                    if m.old_path:
                        paths.append(m.old_path)
            return self._local_log_signature(log), paths

        local_sig, paths = tracker.with_locked_log(_collect)
        if not paths:
            return False

        with self._lock:
            repo_keys = self._repo_keys_for_paths(paths, fallback_root)
            peer_sigs = {k: self._peer_dir_signature(k) for k in repo_keys}
            if (
                not force
                and self._last_local_sig == local_sig
                and all(self._last_peer_sig.get(k) == sig for k, sig in peer_sigs.items())
            ):
                return False

        claims_by_key: dict[str, list[MutationClaim]] = {}
        for repo_key in repo_keys:
            for peer in self._read_peers(repo_key):
                for claim in peer.claims:
                    claims_by_key.setdefault(claim.key, []).append(claim)
                    if claim.old_key:
                        claims_by_key.setdefault(claim.old_key, []).append(claim)

        changed = False
        if claims_by_key:
            key_cache: dict[str, str] = {}

            def _key(p: str) -> str:
                if p not in key_cache:
                    key_cache[p] = canonical_key(p)
                return key_cache[p]

            def _apply(log: MutationLog) -> bool:
                any_change = False
                for turn in log.periods:
                    for m in turn.mutations:
                        if m.provenance is MutationProvenance.FOREIGN:
                            continue
                        if m.t_start is None or m.t_end is None:
                            continue
                        candidates = list(claims_by_key.get(_key(m.path), ()))
                        if m.old_path:
                            candidates += claims_by_key.get(_key(m.old_path), ())
                        for claim in candidates:
                            if not _strict_overlap(m.t_start, m.t_end, claim.t0, claim.t1):
                                continue
                            if (
                                m.provenance is MutationProvenance.ASSUMED
                                and m.old_path is None
                                # MOVEs never demote — EITHER side.  A peer
                                # MOVE claim is indexed under both key and
                                # old_key, so a local simple row on the move
                                # destination can carry identical bytes; the
                                # rename still isn't proof the peer authored
                                # our row's content.  Fall to contested.
                                and claim.old_path is None
                                and self._content_identity(m, claim)
                            ):
                                # Foreign by policy: identity proves the peer
                                # wrote these bytes; preferring under-claiming
                                # our own work over clobbering theirs.
                                m.provenance = MutationProvenance.FOREIGN
                                any_change = True
                                break
                            if not m.contested:
                                m.contested = True
                                any_change = True
                return any_change

            changed = tracker.with_locked_log(_apply)

        with self._lock:
            self._last_local_sig = tracker.with_locked_log(self._local_log_signature)
            self._last_peer_sig.update(peer_sigs)
            if changed:
                # The signature update above makes this change invisible
                # to the next non-forced call — remember that no one has
                # persisted it yet (see consume_unsaved_reclassification).
                self._unsaved_reclassification = True
        return changed

    def consume_unsaved_reclassification(self) -> bool:
        """Return-and-clear the "rows changed, nobody saved yet" flag.

        ``reclassify`` updates its short-circuit signatures as it
        applies changes, so a later refresh that itself changes nothing
        cannot re-discover that the serialized session is behind — a
        finalize-time reclassification (which has no saver) would leave
        detached readers of ``session.json`` stale forever.  The
        engine's refresh entry point (the persister) consumes this
        instead of trusting only its own call's return value.
        """
        with self._lock:
            dirty = self._unsaved_reclassification
            self._unsaved_reclassification = False
            return dirty

    @staticmethod
    def _content_identity(m: FileMutation, claim: MutationClaim) -> bool:
        """Positive evidence the resulting bytes are the peer's own write.

        Hashes present and equal, or both sides gone.  Skip-marked sides
        (no bytes to compare) never qualify.
        """
        if m.after_hash is not None and claim.after_hash is not None:
            return m.after_hash == claim.after_hash
        local_gone = m.after_hash is None and m.after_skip is None
        return local_gone and not claim.after_exists

    # -- Rollback augmentation --------------------------------------------

    def augment_rollback_plan(
        self,
        tracker: MutationTracker,
        plan: RollbackPlan,
        *,
        scope_paths: list[str] | None = None,
        fallback_root: str | None = None,
    ) -> RollbackPlan:
        """Final peer check on a freshly built plan.

        - A path carrying a peer claim strictly *newer* than our last
          write on it is excluded (``PEER_MODIFIED_SINCE``): restoring
          our older snapshot would clobber the peer's newer write, and
          warning-only would let non-modal rollback paths proceed
          straight through the data loss.  (Claims *overlapping* our
          writes were already handled by reclassification — the caller
          runs ``reclassify(force=True)`` before building the plan.)
        - An active peer (not ``closed_at``, live pid) with a still-open
          window whose scope overlaps ours produces a repo-level
          :class:`RollbackWarning` — covering the one case attribution
          cannot: writes on disk whose claims don't exist yet.
        """
        if not self._enabled or (not plan.entries and not plan.exclusions):
            return plan

        # Last local write end per plan path (from the log, under lock).
        entry_paths = [path for path, _snap in plan.entries]
        wanted = set(entry_paths)

        def _last_writes(log: MutationLog) -> dict[str, float]:
            last: dict[str, float] = {}
            for turn in log.periods:
                for m in turn.mutations:
                    if m.provenance is MutationProvenance.FOREIGN:
                        continue
                    end = m.t_end if m.t_end is not None else m.timestamp
                    for p in (m.path, m.old_path) if m.old_path else (m.path,):
                        if p in wanted and end > last.get(p, float("-inf")):
                            last[p] = end
            return last

        last_write = tracker.with_locked_log(_last_writes)

        reference_paths = list(entry_paths) + [path for path, _reason in plan.exclusions] + list(scope_paths or [])
        with self._lock:
            repo_keys = self._repo_keys_for_paths(reference_paths, fallback_root)
        peers = [peer for repo_key in repo_keys for peer in self._read_peers(repo_key)]

        claims_by_key: dict[str, list[MutationClaim]] = {}
        for peer in peers:
            for claim in peer.claims:
                claims_by_key.setdefault(claim.key, []).append(claim)
                if claim.old_key:
                    claims_by_key.setdefault(claim.old_key, []).append(claim)

        kept: list[tuple[str, Any]] = []
        added_exclusions: list[tuple[str, RollbackExclusionReason]] = []
        for path, snap in plan.entries:
            claims = claims_by_key.get(canonical_key(path), ())
            threshold = last_write.get(path, float("-inf"))
            if any(c.t0 > threshold for c in claims):
                added_exclusions.append((path, RollbackExclusionReason.PEER_MODIFIED_SINCE))
            else:
                kept.append((path, snap))

        warnings = list(plan.warnings)
        our_scope = [canonical_key(p) for p in reference_paths]
        warned_sessions: set[str] = set()
        for peer in peers:
            if not peer.is_active or peer.session_id in warned_sessions:
                continue
            for window in peer.windows:
                if window.end is None and self._scopes_overlap(window.scope, our_scope):
                    warned_sessions.add(peer.session_id)
                    warnings.append(
                        RollbackWarning(
                            code=PEER_ACTIVE_WINDOW_WARNING,
                            message=(
                                f"Another {APP_DISPLAY_NAME} session has an active command in this "
                                "workspace; its changes may not be attributed yet."
                            ),
                        )
                    )
                    break

        if not added_exclusions and len(warnings) == len(plan.warnings):
            return plan
        return RollbackPlan(
            entries=kept,
            exclusions=sorted([*plan.exclusions, *added_exclusions], key=lambda item: item[0]),
            warnings=warnings,
        )

    @staticmethod
    def _scopes_overlap(peer_scope: list[str], our_scope_keys: list[str]) -> bool:
        """Ancestor-or-equal containment in either direction, canonically."""
        for raw in peer_scope:
            p = canonical_key(raw)
            for q in our_scope_keys:
                if p == q:
                    return True
                if q.startswith(p.rstrip(os.sep) + os.sep) or p.startswith(q.rstrip(os.sep) + os.sep):
                    return True
        return False
