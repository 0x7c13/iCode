# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Immutable presentation data for planning and organization diagrams."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JourneyTask:
    """One ordered journey step, with satisfaction and participating actors."""

    section: str
    label: str
    score: int
    actors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JourneyChart:
    """A sequence of user journey steps."""

    title: str
    tasks: tuple[JourneyTask, ...]


@dataclass(frozen=True, slots=True)
class TimelinePeriod:
    """One source-ordered period and its ordered events."""

    section: str
    label: str
    events: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TimelineChart:
    """An ordinal timeline; period labels need not be dates."""

    title: str
    vertical: bool
    periods: tuple[TimelinePeriod, ...]


@dataclass(frozen=True, slots=True)
class KanbanTask:
    """A read-only task card with the source's metadata."""

    task_id: str
    label: str
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class KanbanColumn:
    """One board column, including empty columns."""

    column_id: str
    label: str
    tasks: tuple[KanbanTask, ...]


@dataclass(frozen=True, slots=True)
class KanbanChart:
    """Ordered read-only board columns."""

    columns: tuple[KanbanColumn, ...]
