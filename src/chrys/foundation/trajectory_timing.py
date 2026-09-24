# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Portable timing metadata for persisted conversation trajectories.

The adapter-facing contract is a ``_chrys_timing`` object containing UTC
ISO-8601 ``started_at`` / ``finished_at`` values and an integer
``duration_ms``. It is stored on user/assistant message
``additional_properties`` and on both sides of each tool exchange (the call
and its paired result content). User input is an event, so its timestamps are
equal and its duration is zero. Assistant and tool spans use monotonic clocks
for duration while retaining wall-clock timestamps for dashboard placement.

Older session files remain readable and may omit the object; adapters should
treat absence as unknown legacy timing instead of inventing a duration.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from typing import Any

from chrys.foundation.util.time import parse_created_at, utc_iso

TRAJECTORY_TIMING_KEY = "_chrys_timing"
"""Additional-properties key shared by messages and tool call/result contents."""


def build_trajectory_timing(
    *,
    started_at: datetime | str,
    finished_at: datetime | str,
    duration_ms: int,
) -> dict[str, Any]:
    """Return one JSON-safe timing envelope with canonical UTC timestamps."""
    if not isinstance(duration_ms, int) or isinstance(duration_ms, bool) or duration_ms < 0:
        raise ValueError("Trajectory timing duration_ms must be a non-negative integer.")
    parsed_started_at = parse_created_at(started_at)
    parsed_finished_at = parse_created_at(finished_at)
    if parsed_started_at is None or parsed_finished_at is None:
        raise ValueError("Trajectory timing timestamps must be valid ISO-8601 values.")
    return {
        "started_at": utc_iso(parsed_started_at),
        "finished_at": utc_iso(parsed_finished_at),
        "duration_ms": duration_ms,
    }


def build_instant_trajectory_timing(at: datetime | str | None = None) -> dict[str, Any]:
    """Return a zero-duration envelope for an event-like trajectory item."""
    timestamp = parse_created_at(at) if at is not None else datetime.now(UTC)
    if timestamp is None:
        raise ValueError("Trajectory timing timestamp must be a valid ISO-8601 value.")
    return build_trajectory_timing(started_at=timestamp, finished_at=timestamp, duration_ms=0)


def trajectory_timing_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a normalized copy of a stored timing envelope, or ``None``."""
    value = metadata.get(TRAJECTORY_TIMING_KEY)
    if not isinstance(value, Mapping):
        return None
    started_at = value.get("started_at")
    finished_at = value.get("finished_at")
    duration_ms = value.get("duration_ms")
    if (
        not isinstance(started_at, datetime | str)
        or not isinstance(finished_at, datetime | str)
        or not isinstance(duration_ms, int)
        or isinstance(duration_ms, bool)
    ):
        return None
    try:
        return build_trajectory_timing(
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
        )
    except ValueError:
        return None


def stamp_trajectory_timing(
    metadata: MutableMapping[str, Any],
    timing: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> bool:
    """Persist a validated timing envelope, returning whether metadata changed."""
    if TRAJECTORY_TIMING_KEY in metadata and not overwrite:
        return False
    normalized = trajectory_timing_from_metadata({TRAJECTORY_TIMING_KEY: timing})
    if normalized is None:
        return False
    if metadata.get(TRAJECTORY_TIMING_KEY) == normalized:
        return False
    metadata[TRAJECTORY_TIMING_KEY] = normalized
    return True
