# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared helpers for the state-store test files."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from chrys.kernel import Message
from chrys.service.state.store import JsonFileStateStore


async def _save(store: JsonFileStateStore, session_id: str, *texts: str) -> None:
    """Save *session_id* holding one user message per entry of *texts*."""
    await store.save_session(
        session_id,
        {"messages": [Message("user", [text]) for text in texts], "compressed_msgs": []},
    )


def legacy_envelope(session_id: str, **meta: object) -> dict[str, object]:
    """A session envelope in the pre-versioned on-disk shape: no ``schema_version``, empty state.

    Keyword arguments override or extend the meta block — legacy key names such as
    ``display_name`` included.
    """
    fields: dict[str, object] = {
        "session_id": session_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "message_count": 0,
    }
    fields.update(meta)
    return {"meta": fields, "state": {"messages": [], "compressed_msgs": []}}


def write_legacy_envelope(path: Path, session_id: str, **meta: object) -> Path:
    """Write ``legacy_envelope`` to *path* — a flat ``<name>.json`` or a ``<dir>/session.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(legacy_envelope(session_id, **meta)), encoding="utf-8")
    return path


class SkewedDateTime(datetime):
    """A ``datetime`` whose ``now()`` answers the class attribute ``current``.

    Tests move the clock by assigning ``current`` after installing the class over
    ``store_module.datetime``; pin the start value through ``monkeypatch.setattr``
    so the moves are undone at teardown and never leak into the next test.
    """

    current = datetime(2026, 1, 1, 12, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls.current.replace(tzinfo=None)
        return cls.current.astimezone(tz)
