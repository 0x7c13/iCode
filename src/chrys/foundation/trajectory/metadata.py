# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Stable analytics identifiers stamped on persisted conversation items.

``session.json`` stays the conversation truth; the trajectory log refers to
its items by two additional-properties keys rather than by position:

* ``_chrys_analytics_item_id`` — one per persisted message / tool-call
  content / tool-result content, minted when the item is created (or, as a
  backstop, right before the save that first persists it) and never changed
  afterwards. Save/restore, compaction, fork and rollback copy the property
  verbatim, which is what makes the id stable.
* ``_chrys_operation_id`` — the trajectory operation the item belongs to:
  the model exchange that produced an assistant message, or the tool
  operation a call content and its paired result content share.

Both are 32 lowercase hex characters (see :mod:`chrys.foundation.trajectory.ids`).
Values that fail that shape are treated as absent.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any, Final

from chrys.foundation.trajectory.ids import is_valid_analytics_id, new_analytics_id

ANALYTICS_ITEM_ID_KEY: Final = "_chrys_analytics_item_id"
OPERATION_ID_KEY: Final = "_chrys_operation_id"

TOOL_PARENT_OPERATION_METADATA_KEY: Final = "_chrys_trajectory_parent_operation_id"
"""Invocation-context key: the model exchange operation the tool call belongs to (never persisted)."""
TOOL_RESULT_ITEM_ID_METADATA_KEY: Final = "_chrys_trajectory_result_item_id"
"""Invocation-context key: the item id pre-minted for the call's result content (never persisted)."""
TOOL_RESULT_CARRIER_ITEM_ID_METADATA_KEY: Final = "_chrys_trajectory_result_carrier_item_id"
"""Invocation-context key: the item id pre-minted for the result batch's carrier message (never persisted)."""


def read_analytics_item_id(additional_properties: object) -> str | None:
    """Return the stamped item id, or ``None`` when absent or malformed."""
    return _read(additional_properties, ANALYTICS_ITEM_ID_KEY)


def read_operation_id(additional_properties: object) -> str | None:
    """Return the stamped operation id, or ``None`` when absent or malformed."""
    return _read(additional_properties, OPERATION_ID_KEY)


def ensure_analytics_item_id(additional_properties: MutableMapping[str, Any], *, item_id: str | None = None) -> str:
    """Return the item id, stamping *item_id* (or a fresh id) when none is set.

    An existing valid id always wins: ids are minted once and never rewritten.
    """
    existing = read_analytics_item_id(additional_properties)
    if existing is not None:
        return existing
    value = item_id if item_id is not None and is_valid_analytics_id(item_id) else new_analytics_id()
    additional_properties[ANALYTICS_ITEM_ID_KEY] = value
    return value


def stamp_operation_id(additional_properties: MutableMapping[str, Any], operation_id: str) -> None:
    """Bind the item to *operation_id*; malformed ids are ignored."""
    if is_valid_analytics_id(operation_id):
        additional_properties[OPERATION_ID_KEY] = operation_id


def _read(additional_properties: object, key: str) -> str | None:
    if not isinstance(additional_properties, Mapping):
        return None
    value = additional_properties.get(key)
    return value if isinstance(value, str) and is_valid_analytics_id(value) else None
