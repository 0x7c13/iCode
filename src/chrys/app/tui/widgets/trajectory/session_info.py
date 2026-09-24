# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session directory facts shown in the dashboard's session info section."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from chrys.service.analytics import TrajectoryScanCancelled

SESSION_JSON_NAME = "session.json"
TRAJECTORY_DIR_NAME = "trajectory"
EVENTS_LOG_NAME = "events.jsonl"
MUTATIONS_DIR_NAME = "mutations"
SNAPSHOTS_DIR_NAME = "snapshots"
SUB_AGENTS_DIR_NAME = "sub_agents"


@dataclass(frozen=True, slots=True)
class SessionStorage:
    """Apparent on-disk sizes of one session directory and its notable members."""

    session_dir: Path
    total_bytes: int = 0
    session_json_bytes: int = 0
    events_bytes: int = 0
    mutations_bytes: int = 0
    snapshots_bytes: int = 0
    sub_agents_bytes: int = 0
    file_count: int = 0


def collect_session_storage(session_dir: Path, *, cancel_event: Event | None = None) -> SessionStorage:
    """Sum regular-file sizes under *session_dir* without following links.

    Neither symlinks nor Windows junctions are descended into — a junction is a
    directory to ``is_dir(follow_symlinks=False)``, so it needs its own check
    or a link to an ancestor would cycle the walk. Unreadable entries are
    skipped rather than failing the whole summary: the section is informational
    and a partially readable tree still answers "how big is this session".

    Raises :class:`TrajectoryScanCancelled` when *cancel_event* is set so a
    hidden dashboard does not keep an executor thread walking the tree.
    """
    total = 0
    count = 0
    session_json = 0
    events = 0
    members = {MUTATIONS_DIR_NAME: 0, SNAPSHOTS_DIR_NAME: 0, SUB_AGENTS_DIR_NAME: 0}
    try:
        with os.scandir(session_dir) as entries:
            top_level = list(entries)
    except OSError:
        return SessionStorage(session_dir=session_dir)
    for entry in top_level:
        _check_cancelled(cancel_event)
        try:
            if entry.is_file(follow_symlinks=False):
                size = entry.stat(follow_symlinks=False).st_size
                total += size
                count += 1
                if entry.name == SESSION_JSON_NAME:
                    session_json = size
                continue
            if not entry.is_dir(follow_symlinks=False) or os.path.isjunction(entry.path):
                continue
        except OSError:
            continue
        subtree_bytes, subtree_files = _tree_size(entry.path, cancel_event)
        total += subtree_bytes
        count += subtree_files
        if entry.name in members:
            members[entry.name] = subtree_bytes
        elif entry.name == TRAJECTORY_DIR_NAME:
            events = _file_size(os.path.join(entry.path, EVENTS_LOG_NAME))
    return SessionStorage(
        session_dir=session_dir,
        total_bytes=total,
        session_json_bytes=session_json,
        events_bytes=events,
        mutations_bytes=members[MUTATIONS_DIR_NAME],
        snapshots_bytes=members[SNAPSHOTS_DIR_NAME],
        sub_agents_bytes=members[SUB_AGENTS_DIR_NAME],
        file_count=count,
    )


def _check_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise TrajectoryScanCancelled


def _tree_size(root: str, cancel_event: Event | None) -> tuple[int, int]:
    total = 0
    count = 0
    pending = [root]
    while pending:
        _check_cancelled(cancel_event)
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if not os.path.isjunction(entry.path):
                                pending.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            count += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total, count


def _file_size(path: str) -> int:
    try:
        return os.stat(path, follow_symlinks=False).st_size
    except OSError:
        return 0
