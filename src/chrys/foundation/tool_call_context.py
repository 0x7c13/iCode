# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Out-of-band tool-call provenance context channel.

Mirrors the ``chrys_kind`` attribute channel (:mod:`chrys.foundation.tool_kinds`):
producers attach a small JSON-safe provenance dict to a tool object, and the
persistence pipeline copies it onto the assistant ``function_call`` content as
``additional_properties["_chrys_tool_context"]`` so offline session.json
readers can answer "which skill? which MCP server?" without name guessing.

Two segments with different lifecycles:

- **static** (:func:`set_tool_context`) — fixed at tool creation
  (e.g. MCP server/remote identity); sanitized once at registration and read
  back as-is.
- **builder** (:func:`set_tool_context_builder`) — computed per invocation
  from the final call args (e.g. canonical skill names); sanitized at
  resolve time by the same segment sanitizer.

Sanitization is exact-or-absent per segment: values are identifiers, so a
non-conforming or over-cap value drops its key (never truncated — a mangled
identifier is worse than a missing one), and an over-budget segment drops
whole. Provenance must never break tool creation or execution: nothing in
this module raises on bad producer input; problems are debug-logged.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_CONTEXT_ATTR: Final[str] = "chrys_tool_context"
_CONTEXT_BUILDER_ATTR: Final[str] = "chrys_tool_context_builder"

TOOL_CALL_CONTEXT_METADATA_KEY: Final[str] = "_chrys_tool_context"
"""Kind-specific provenance persisted on assistant ``function_call`` contents."""

MAX_CONTEXT_VALUE_CHARS: Final[int] = 512
"""Per-value cap; a longer string value drops its key (exact-or-absent)."""

MAX_CONTEXT_SEGMENT_KEYS: Final[int] = 16
"""Per-segment key-count bound; exceeding it drops the whole segment."""

MAX_CONTEXT_SEGMENT_SERIALIZED_CHARS: Final[int] = 1024
"""Per-segment serialized budget; exceeding it drops the whole segment."""

_SCALAR_TYPES: Final[tuple[type, ...]] = (str, int, float, bool)


def sanitize_context_segment(segment: Mapping[str, Any], *, source: str = "") -> dict[str, Any]:
    """Return the sanitized form of one context segment (exact-or-absent).

    The single sanitization authority for both write points: the static
    segment at :func:`set_tool_context` registration and the builder segment
    inside :func:`resolve_tool_call_context`. Never raises.
    """
    kept: dict[str, Any] = {}
    for key, value in segment.items():
        if not isinstance(key, str) or not key:
            logger.debug("Dropping tool-context key %r from %s: keys must be non-empty strings.", key, source)
            continue
        if not isinstance(value, _SCALAR_TYPES):
            logger.debug("Dropping tool-context key %r from %s: non-scalar value %r.", key, source, type(value))
            continue
        if isinstance(value, float) and not math.isfinite(value):
            # json.dumps would emit non-standard JSON (NaN/Infinity) for these,
            # and the session store serializes with the same default.
            logger.debug("Dropping tool-context key %r from %s: non-finite float %r.", key, source, value)
            continue
        if isinstance(value, str) and len(value) > MAX_CONTEXT_VALUE_CHARS:
            # Never truncate: context values are identifiers, and a mangled
            # identifier is worse than a missing one.
            logger.debug(
                "Dropping tool-context key %r from %s: value length %d exceeds %d.",
                key,
                source,
                len(value),
                MAX_CONTEXT_VALUE_CHARS,
            )
            continue
        kept[key] = value
    if len(kept) > MAX_CONTEXT_SEGMENT_KEYS:
        logger.debug(
            "Dropping tool-context segment from %s: %d keys exceed the %d-key bound.",
            source,
            len(kept),
            MAX_CONTEXT_SEGMENT_KEYS,
        )
        return {}
    if kept:
        try:
            # allow_nan=False backstops the per-value filter above: if a
            # non-finite float ever slips through, the segment drops whole
            # here instead of writing non-standard JSON into session.json.
            serialized_len = len(json.dumps(kept, ensure_ascii=False, allow_nan=False))
        except TypeError, ValueError:  # pragma: no cover - filtered scalars always serialize
            logger.debug("Dropping tool-context segment from %s: not JSON-serializable.", source)
            return {}
        if serialized_len > MAX_CONTEXT_SEGMENT_SERIALIZED_CHARS:
            logger.debug(
                "Dropping tool-context segment from %s: serialized size %d exceeds %d.",
                source,
                serialized_len,
                MAX_CONTEXT_SEGMENT_SERIALIZED_CHARS,
            )
            return {}
    return kept


def set_tool_context(tool: Any, context: Mapping[str, Any]) -> None:
    """Sanitize the static context segment and store it on *tool*. Never raises.

    Non-conforming keys are dropped and an over-budget segment is stored
    empty — each with a debug log. Provenance must not break tool creation
    any more than tool execution: MCP server/remote names are externally
    controlled, and a provenance cap must never prevent a tool from loading.
    """
    setattr(tool, _CONTEXT_ATTR, sanitize_context_segment(context, source=f"static context of {_tool_label(tool)}"))


def get_tool_context(tool: Any) -> Mapping[str, Any] | None:
    """Return the stored, already-sanitized static context of *tool*, if any."""
    context = getattr(tool, _CONTEXT_ATTR, None)
    if isinstance(context, Mapping) and context:
        return context
    return None


def set_tool_context_builder(tool: Any, builder: Callable[[Mapping[str, Any]], Mapping[str, Any] | None]) -> None:
    """Register a per-call context builder invoked with the final call args."""
    setattr(tool, _CONTEXT_BUILDER_ATTR, builder)


def resolve_tool_call_context(tool: Any, args: Mapping[str, Any]) -> dict[str, Any]:
    """Merge the static context with the per-call builder output (static wins).

    A builder key that collides with a static key is dropped and debug-logged
    — collisions are producer bugs (static and builder key sets must be
    disjoint), and static-wins keeps this resolve-time output identical to
    what the kernel-stamp + subkey-backfill persistence sequence produces.
    Builder exceptions are swallowed (debug log): provenance must never break
    tool execution. The static segment is used as-is — already sanitized at
    :func:`set_tool_context`, never re-sanitized.
    """
    static = get_tool_context(tool)
    merged: dict[str, Any] = dict(static) if static else {}

    builder = getattr(tool, _CONTEXT_BUILDER_ATTR, None)
    if not callable(builder):
        return merged
    label = _tool_label(tool)
    try:
        built = builder(args)
    except Exception:
        logger.debug("Tool-context builder for %s raised; keeping static context only.", label, exc_info=True)
        return merged
    if not isinstance(built, Mapping) or not built:
        return merged
    for key, value in sanitize_context_segment(built, source=f"context builder of {label}").items():
        if key in merged:
            logger.debug(
                "Dropping builder tool-context key %r for %s: collides with static context (producer bug).",
                key,
                label,
            )
            continue
        merged[key] = value
    return merged


def merge_tool_call_context_property(props: Any, tool_context: Mapping[str, Any]) -> None:
    """Backfill call provenance context at subkey granularity.

    Unlike the kind (an atomic string, whole-key first-write-wins), the
    context merges per subkey: a pre-execution static context gets the
    builder subkeys it couldn't know filled in, while existing subkeys are
    never overwritten (``setdefault``).
    """
    if not tool_context:
        return
    existing = props.get(TOOL_CALL_CONTEXT_METADATA_KEY)
    if TOOL_CALL_CONTEXT_METADATA_KEY not in props:
        props[TOOL_CALL_CONTEXT_METADATA_KEY] = dict(tool_context)
        return
    if not isinstance(existing, dict):
        # Corrupt/foreign value — first-write-wins, leave it alone.
        return
    for key, value in tool_context.items():
        existing.setdefault(key, value)


def _tool_label(tool: Any) -> str:
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) and name else repr(tool)
