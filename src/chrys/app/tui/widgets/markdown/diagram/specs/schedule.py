# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Immutable schedule, commit-history, and bit-field diagram data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class GanttTask:
    """A calendar-day task; the end boundary is exclusive."""

    task_id: str
    label: str
    section: str
    start: date
    end: date
    statuses: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GanttChart:
    """Resolved, ordered task intervals on a shared date scale."""

    title: str
    tasks: tuple[GanttTask, ...]


@dataclass(frozen=True, slots=True)
class GitCommit:
    """One commit with indices into its chart's earlier commits."""

    commit_id: str
    branch: str
    parents: tuple[int, ...]
    tags: tuple[str, ...] = ()
    kind: str = "NORMAL"


@dataclass(frozen=True, slots=True)
class GitChart:
    """Branch declaration order, explicit ancestry, and time orientation."""

    branches: tuple[str, ...]
    commits: tuple[GitCommit, ...]
    orientation: str = "LR"


@dataclass(frozen=True, slots=True)
class PacketField:
    """One named inclusive absolute bit interval."""

    start: int
    end: int
    label: str


@dataclass(frozen=True, slots=True)
class PacketChart:
    """Ordered packet fields displayed in fixed 32-bit rows."""

    title: str
    fields: tuple[PacketField, ...]
