# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Backstop stamping of analytics item ids on persisted history.

Items normally get their ``_chrys_analytics_item_id`` when they are created
(the executor stamps user messages, the kernel loop stamps what it lands).
Anything that reached history another way — recovery re-creation, a
restored legacy session, a marker the history manager inserted — is
stamped here right before the save that persists it, so every saved item
has an id before any trajectory event could refer to it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from chrys.foundation.trajectory.metadata import ensure_analytics_item_id
from chrys.kernel import Message

_CONTENT_TYPES_WITH_IDENTITY = frozenset({"function_call", "function_result"})


def ensure_history_item_ids(messages: Iterable[Any]) -> int:
    """Stamp missing item ids on *messages* and their tool contents; return the count stamped."""
    stamped = 0
    for message in messages:
        if not isinstance(message, Message):
            continue
        props = message.additional_properties
        before = len(props)
        ensure_analytics_item_id(props)
        stamped += int(len(props) != before)
        for content in message.contents:
            if content.type not in _CONTENT_TYPES_WITH_IDENTITY:
                continue
            content_props = content.additional_properties
            before = len(content_props)
            ensure_analytics_item_id(content_props)
            stamped += int(len(content_props) != before)
    return stamped
