# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Single-producer tool invocation ordinal.

The kernel tool loop is the sole authoritative producer: it stamps every
non-informational ``function_call`` content of a run attempt with a 0-based
ordinal in model emission order, at response arrival — before the dispatch
filter, tool lookup, and argument validation. Calls that never reach the
middleware pipeline (invalid arguments, unknown tools, duplicate ids, no-tools
batches) still consume ordinals, so persisted numbering and pipeline numbering
can never diverge.

The same key carries the ordinal in two mappings with one meaning:

- ``Content.additional_properties`` on the assistant ``function_call``
  (persisted to session.json; consumers such as approval-decision
  persistence read it back instead of counting positions), and
- ``FunctionInvocationContext.metadata`` (runtime carriage; event/approval
  middleware read it via their existing accessors).

Consumers must treat a missing stamp as "never order-match" (fail closed),
not as "count for yourself".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

TOOL_INVOCATION_ORDER_KEY: Final[str] = "_chrys_tool_invocation_order"
"""Current-run tool invocation ordinal (see module docstring)."""


def read_tool_invocation_order(mapping: object) -> int | None:
    """Return the stamped invocation ordinal from a metadata mapping, if any."""
    if not isinstance(mapping, Mapping):
        return None
    raw = mapping.get(TOOL_INVOCATION_ORDER_KEY)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return None
    return None
