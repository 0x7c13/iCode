# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Draft builders for ``tool.operation.*`` events.

The kernel loop and the service tool middleware both describe tool operations
— the kernel for calls that never enter the middleware pipeline (unknown tool,
invalid arguments), the middleware for everything that runs — so the payload
shape lives here, once, below both of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chrys.foundation.trajectory.context import TrajectoryContext
from chrys.foundation.trajectory.envelope import EventDraft, Link, LinkRelation, MeasurementSource, measurement
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.fingerprint import DOMAIN_TOOL_ARGUMENTS, fingerprint_json, fingerprint_text


def tool_argument_fingerprint(context: TrajectoryContext, arguments: object) -> str | None:
    """Keyed fingerprint of *arguments* (mapping or raw text); ``None`` without a key or arguments."""
    key = context.sink.fingerprint_key
    if not key or arguments is None:
        return None
    if isinstance(arguments, Mapping):
        try:
            return fingerprint_json(key, DOMAIN_TOOL_ARGUMENTS, dict(arguments))
        except TypeError, ValueError:
            return fingerprint_text(key, DOMAIN_TOOL_ARGUMENTS, repr(arguments))
    return fingerprint_text(key, DOMAIN_TOOL_ARGUMENTS, arguments if isinstance(arguments, str) else repr(arguments))


def tool_operation_started_draft(
    context: TrajectoryContext,
    *,
    operation_id: str,
    tool_name: str,
    tool_kind: str,
    invocation_order: int | None,
    batch_index: int | None,
    arguments: object,
    call_item_id: str | None = None,
    tool_context: Mapping[str, Any] | None = None,
    caused_by_operation_id: str | None = None,
) -> EventDraft:
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_kind": tool_kind,
        "parent_model_operation_id": context.innermost_model_operation_id,
    }
    if tool_context:
        # Kind-specific identity the tool name alone cannot carry: which MCP
        # server served the call, which skill was loaded. Sanitized and
        # bounded by its own channel — names of things, never URLs, headers
        # or command lines — with one documented exception: a skill lookup
        # that misses records the name the model asked for, exact or absent,
        # capped like every other segment. That ceiling also keeps the whole
        # line inside the writer's budget.
        payload["tool_context"] = dict(tool_context)
    if invocation_order is not None:
        payload["invocation_order"] = invocation_order
    if batch_index is not None:
        payload["batch_index"] = batch_index
    fingerprint = tool_argument_fingerprint(context, arguments)
    if fingerprint is not None:
        payload["argument_fingerprint"] = fingerprint
    if call_item_id is not None:
        payload["call_item_id"] = call_item_id
    links = (
        (Link(relation=LinkRelation.CAUSED_BY, target_operation_id=caused_by_operation_id),)
        if caused_by_operation_id is not None
        else ()
    )
    return context.draft(
        EventType.TOOL_OPERATION_STARTED,
        operation_id=operation_id,
        parent_operation_id=context.innermost_model_operation_id,
        payload=payload,
        links=links,
    )


def tool_operation_finished_draft(
    context: TrajectoryContext,
    *,
    operation_id: str,
    outcome: str,
    duration_ms: int,
    result_item_id: str | None,
    result_carrier_item_id: str | None,
    exit_code: int | None = None,
    timed_out: bool | None = None,
    error_kind: str | None = None,
    abandoned: bool = False,
) -> EventDraft:
    payload: dict[str, Any] = {"outcome": outcome, "duration_ms": max(0, duration_ms)}
    if result_item_id is not None:
        payload["result_item_id"] = result_item_id
    if result_carrier_item_id is not None:
        payload["result_carrier_item_id"] = result_carrier_item_id
    if exit_code is not None:
        payload["exit_code"] = exit_code
    if timed_out is not None:
        payload["timed_out"] = timed_out
    if error_kind:
        payload["error_kind"] = error_kind
    if abandoned:
        payload["abandoned"] = True
    return context.draft(
        EventType.TOOL_OPERATION_FINISHED,
        operation_id=operation_id,
        parent_operation_id=context.innermost_model_operation_id,
        payload=payload,
        measurements={"/payload/duration_ms": measurement(MeasurementSource.MONOTONIC_CLOCK, method_version=1)},
    )
