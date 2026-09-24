# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Time helpers shared across Chrys."""

from __future__ import annotations

from datetime import UTC, datetime


def parse_created_at(value: object) -> datetime | None:
    """Parse Chrys timestamp metadata into a datetime, or None when invalid."""
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith(("Z", "z")):
        raw = f"{raw[:-1]}+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def utc_iso(value: datetime | None = None) -> str:
    """Return a canonical aware UTC ISO-8601 timestamp."""
    if value is None:
        value = datetime.now(UTC)
    elif value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value.isoformat(timespec="microseconds")
