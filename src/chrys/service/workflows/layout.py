# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Paths for immutable run metadata, full artifacts and the lifecycle journal."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

WORKFLOWS_DIR: Final = "workflows"
HEADER_FILE: Final = "run.json"
SPEC_FILE: Final = "spec.json"
INPUT_FILE: Final = "input.txt"
OUTPUT_INDEX_FILE: Final = "outputs.json"
RUN_OUTPUT_FILE: Final = "output.json"
SOURCE_FILE: Final = "source.py"
EVENTS_FILE: Final = "events.jsonl"
NODES_DIR: Final = "nodes"

MAX_SCANNED_RUN_DIRS: Final = 4096
"""How many entries of ``workflows/`` a scan examines before it stops (a same-uid flood must not spin it)."""


def workflows_root(session_dir: Path) -> Path:
    return session_dir / WORKFLOWS_DIR


def run_dir(session_dir: Path, run_id: str) -> Path:
    return workflows_root(session_dir) / run_id


def iter_run_dirs(session_dir: Path) -> tuple[list[Path], bool]:
    """Run directories (those holding a header) under the session, sorted, and whether the scan was cut short.

    ``os.scandir`` is lazy where ``Path.iterdir`` is eager, so the cap bounds
    enumeration itself. Entries that are not directories, dot-prefixed, or
    without a header are aborted creates or foreign files and are skipped.
    """
    root = workflows_root(session_dir)
    found: list[Path] = []
    truncated = False
    try:
        with os.scandir(root) as entries:
            for examined, entry in enumerate(entries, start=1):
                if examined > MAX_SCANNED_RUN_DIRS:
                    truncated = True
                    break
                if entry.name.startswith(".") or not entry.is_dir(follow_symlinks=False):
                    continue
                if os.path.isfile(os.path.join(entry.path, HEADER_FILE)):
                    found.append(Path(entry.path))
    except FileNotFoundError:
        return [], False
    found.sort()
    return found, truncated
