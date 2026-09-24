# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chrys-owned metadata stored on persisted framework messages."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.tool_call_context import (
    TOOL_CALL_CONTEXT_METADATA_KEY as TOOL_CALL_CONTEXT_METADATA_KEY,
)
from chrys.foundation.tool_kinds import (
    TOOL_CALL_KIND_METADATA_KEY as TOOL_CALL_KIND_METADATA_KEY,
)
from chrys.foundation.tool_result_metadata import (
    TOOL_RESULT_METADATA_KEY as TOOL_RESULT_METADATA_KEY,
)
from chrys.foundation.trajectory_timing import (
    build_instant_trajectory_timing,
    build_trajectory_timing,
    stamp_trajectory_timing,
)
from chrys.foundation.util.time import parse_created_at, utc_iso
from chrys.kernel import Message

MESSAGE_CREATED_AT_KEY = "_chrys_created_at"
LAST_ASSISTANT_CREATED_AT_STATE_KEY = "_chrys_last_assistant_created_at"

# TOOL_CALL_KIND_METADATA_KEY / TOOL_CALL_CONTEXT_METADATA_KEY /
# TOOL_RESULT_METADATA_KEY live at the foundation layer so the kernel tool
# loop can stamp call provenance pre-pipeline and fold result metadata at
# result construction; re-exported here so session-layer importers keep one
# import site.


def persisted_tool_call_kind(additional_properties: object) -> str:
    """Return a persisted tool kind from message content metadata, if present."""
    if not isinstance(additional_properties, Mapping):
        return ""
    value = additional_properties.get(TOOL_CALL_KIND_METADATA_KEY)
    return value if isinstance(value, str) else ""


def persisted_tool_call_context(additional_properties: object) -> dict[str, Any]:
    """Return persisted tool-call provenance context from content metadata, if present."""
    if not isinstance(additional_properties, Mapping):
        return {}
    value = additional_properties.get(TOOL_CALL_CONTEXT_METADATA_KEY)
    # Kernel-stamped tool-call context metadata always uses string keys.
    return dict(cast("Mapping[str, Any]", value)) if isinstance(value, Mapping) else {}


def is_structural_marker(message: Message) -> bool:
    """Return True when *message* is a Chrys structural/status marker."""
    return HistoryMarkerKind.KEY in message.additional_properties


def should_stamp_message_created_at(message: Message) -> bool:
    """Return True for persisted conversation messages that should carry a timestamp."""
    return message.role in ("user", "assistant") and not is_structural_marker(message)


def try_normalize_created_at(value: datetime | str | None = None) -> str | None:
    """Return a canonical UTC timestamp string, or None when unparsable."""
    parsed = parse_created_at(value)
    return utc_iso(parsed) if parsed is not None else None


def normalize_created_at(value: datetime | str | None = None) -> str:
    """Return a canonical UTC timestamp string, falling back to now."""
    return try_normalize_created_at(value) or utc_iso()


def stamp_message_created_at(
    message: Message,
    created_at: datetime | str | None = None,
    *,
    overwrite: bool = False,
) -> bool:
    """Stamp ``_chrys_created_at`` on a persisted message if appropriate.

    Returns True when the message was mutated.
    """
    if not should_stamp_message_created_at(message):
        return False
    existing: Any = message.additional_properties.get(MESSAGE_CREATED_AT_KEY)
    changed = False
    if existing and not overwrite:
        timestamp = try_normalize_created_at(existing) if isinstance(existing, datetime | str) else None
    else:
        timestamp = normalize_created_at(created_at)
        message.additional_properties[MESSAGE_CREATED_AT_KEY] = timestamp
        changed = True
    if timestamp is not None:
        changed = (
            stamp_trajectory_timing(
                message.additional_properties,
                build_instant_trajectory_timing(timestamp),
                overwrite=overwrite,
            )
            or changed
        )
    return changed


def stamp_message_response_timing(
    message: Message,
    *,
    started_at: datetime | str,
    finished_at: datetime | str,
    duration_ms: int,
) -> bool:
    """Stamp measured model-response timing on a persisted assistant message."""
    if message.role != "assistant" or not should_stamp_message_created_at(message):
        return False
    timing = build_trajectory_timing(
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
    )
    created_at_changed = message.additional_properties.get(MESSAGE_CREATED_AT_KEY) != timing["finished_at"]
    message.additional_properties[MESSAGE_CREATED_AT_KEY] = timing["finished_at"]
    timing_changed = stamp_trajectory_timing(message.additional_properties, timing, overwrite=True)
    return created_at_changed or timing_changed
