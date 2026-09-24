# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tool-operation recording for the tool event middleware.

One :class:`ToolOperationTrace` spans one tool call through the middleware
pipeline: ``tool.operation.started`` when the call is dispatched,
``tool.payload.observed`` once the model-visible result is known, and
``tool.operation.finished`` with the terminal outcome. The kernel loop stamps
the operation id and the result item id on the invocation context before the
pipeline runs (and records the calls that never reach it), so call, result
and events all name the same operation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from chrys.foundation.text.tokenizer import MixedLanguageTokenizer
from chrys.foundation.tool_result_metadata import (
    PROCESS_EXIT_CODE_METADATA_KEY,
    PROCESS_TIMED_OUT_METADATA_KEY,
    SHELL_EXIT_CODE_METADATA_KEY,
    SHELL_TIMED_OUT_METADATA_KEY,
    TOOL_ERROR_KIND_METADATA_KEY,
    tool_result_metadata_failure_state,
)
from chrys.foundation.trajectory.context import TrajectoryContext, current_trajectory
from chrys.foundation.trajectory.envelope import MeasurementSource, measurement
from chrys.foundation.trajectory.event_types import EventType, ToolOutcome
from chrys.foundation.trajectory.fingerprint import DOMAIN_TOOL_CONTENT, fingerprint_text
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory.metadata import (
    ANALYTICS_ITEM_ID_KEY,
    OPERATION_ID_KEY,
    TOOL_RESULT_CARRIER_ITEM_ID_METADATA_KEY,
    TOOL_RESULT_ITEM_ID_METADATA_KEY,
)
from chrys.foundation.trajectory.tools import tool_operation_finished_draft, tool_operation_started_draft

logger = logging.getLogger(__name__)

TOKENIZER_FINGERPRINT = "mixed_language_ratio/1"
"""Identifies the local estimator behind ``local_token_estimate`` (never a provider count)."""

_tokenizer = MixedLanguageTokenizer()


def _metadata_id(metadata: Mapping[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def tool_operation_id(invocation_metadata: Mapping[str, Any] | None) -> str | None:
    """The kernel-minted tool operation id stamped on an invocation context, if any."""
    if not isinstance(invocation_metadata, Mapping):
        return None
    return _metadata_id(invocation_metadata, OPERATION_ID_KEY)


def tool_outcome(
    *,
    cancelled: bool,
    rejected: bool,
    errored: bool,
    result_metadata: Mapping[str, Any],
) -> str:
    """Map the middleware's terminal facts onto the ``tool.operation.finished`` outcome enum."""
    if cancelled:
        return ToolOutcome.INTERRUPTED
    if rejected:
        return ToolOutcome.REJECTED
    if errored:
        return ToolOutcome.ERRORED
    if result_metadata.get(PROCESS_TIMED_OUT_METADATA_KEY) is True or result_metadata.get(SHELL_TIMED_OUT_METADATA_KEY):
        return ToolOutcome.TIMED_OUT
    failed = tool_result_metadata_failure_state(result_metadata)
    if failed is True:
        return ToolOutcome.FAILED
    if failed is False:
        return ToolOutcome.SUCCESS
    # No structured verdict: the tool neither raised nor recorded failure.
    return ToolOutcome.SUCCESS


def _exit_code(result_metadata: Mapping[str, Any]) -> int | None:
    for key in (PROCESS_EXIT_CODE_METADATA_KEY, SHELL_EXIT_CODE_METADATA_KEY):
        value = result_metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


class ToolOperationTrace:
    """Record one tool operation under the ambient trajectory context."""

    __slots__ = (
        "_context",
        "_finished",
        "_operation_id",
        "_result_carrier_item_id",
        "_result_item_id",
        "_started",
    )

    def __init__(
        self,
        context: TrajectoryContext,
        *,
        operation_id: str,
        result_item_id: str | None,
        result_carrier_item_id: str | None,
    ) -> None:
        self._context = context
        self._operation_id = operation_id
        self._result_item_id = result_item_id
        self._result_carrier_item_id = result_carrier_item_id
        self._started = False
        self._finished = False

    @classmethod
    def open(cls, invocation_metadata: Mapping[str, Any]) -> ToolOperationTrace | None:
        """Bind to the ambient context, or ``None`` when recording is off."""
        context = current_trajectory()
        if context is None:
            return None
        operation_id = _metadata_id(invocation_metadata, OPERATION_ID_KEY) or new_analytics_id()
        return cls(
            context,
            operation_id=operation_id,
            result_item_id=_metadata_id(invocation_metadata, TOOL_RESULT_ITEM_ID_METADATA_KEY),
            result_carrier_item_id=_metadata_id(invocation_metadata, TOOL_RESULT_CARRIER_ITEM_ID_METADATA_KEY),
        )

    @property
    def operation_id(self) -> str:
        return self._operation_id

    @property
    def context(self) -> TrajectoryContext:
        return self._context

    async def started(
        self,
        *,
        tool_name: str,
        tool_kind: str,
        batch_index: int | None,
        invocation_order: int | None,
        arguments: object,
        invocation_metadata: Mapping[str, Any],
        tool_context: Mapping[str, Any] | None = None,
        preamble_operation_id: str | None = None,
    ) -> None:
        # Marked before the wait, not after it: once the draft is handed to
        # the sink its line may be committed even if this ack is cancelled,
        # so the terminal has to follow it either way.
        self._started = True
        draft = tool_operation_started_draft(
            self._context,
            operation_id=self._operation_id,
            tool_name=tool_name,
            tool_kind=tool_kind,
            invocation_order=invocation_order,
            batch_index=batch_index,
            arguments=arguments,
            call_item_id=_metadata_id(invocation_metadata, ANALYTICS_ITEM_ID_KEY),
            tool_context=tool_context,
            caused_by_operation_id=preamble_operation_id,
        )
        try:
            await self._context.sink.emit(draft)
        except Exception:
            logger.debug("Trajectory tool.operation.started emit failed", exc_info=True)

    async def payload_observed(
        self,
        *,
        result_text: str,
        image_count: int,
        observation: Mapping[str, Any] | None,
    ) -> None:
        """Record the model-visible result shape (sizes and fingerprints, never content)."""
        payload: dict[str, Any] = {
            "model_visible_bytes": len(result_text.encode("utf-8", errors="backslashreplace")),
            "local_token_estimate": _tokenizer.count_tokens(result_text),
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "truncated": False,
            "content_type": "text+image" if image_count else "text",
        }
        if image_count:
            payload["image_count"] = image_count
        if self._result_item_id is not None:
            payload["result_item_id"] = self._result_item_id
        if observation:
            for key in ("original_bytes", "truncated", "artifact_id"):
                if key in observation:
                    payload[key] = observation[key]
        key = self._context.sink.fingerprint_key
        if key:
            payload["content_fingerprint"] = fingerprint_text(key, DOMAIN_TOOL_CONTENT, result_text)
        draft = self._context.draft(
            EventType.TOOL_PAYLOAD_OBSERVED,
            operation_id=self._operation_id,
            parent_operation_id=self._context.innermost_model_operation_id,
            payload=payload,
            measurements={
                "/payload/local_token_estimate": measurement(MeasurementSource.LOCAL_TOKENIZER, method_version=1)
            },
        )
        try:
            await self._context.sink.emit(draft)
        except Exception:
            logger.debug("Trajectory tool.payload.observed emit failed", exc_info=True)

    def _finished_draft(
        self,
        *,
        outcome: str,
        duration_ms: int,
        result_metadata: Mapping[str, Any],
        error_kind: str | None,
        abandoned: bool = False,
    ) -> Any:
        timed_out = result_metadata.get(PROCESS_TIMED_OUT_METADATA_KEY) is True or bool(
            result_metadata.get(SHELL_TIMED_OUT_METADATA_KEY)
        )
        structured_kind = result_metadata.get(TOOL_ERROR_KIND_METADATA_KEY)
        return tool_operation_finished_draft(
            self._context,
            operation_id=self._operation_id,
            outcome=outcome,
            duration_ms=duration_ms,
            result_item_id=self._result_item_id,
            result_carrier_item_id=self._result_carrier_item_id,
            exit_code=_exit_code(result_metadata),
            timed_out=timed_out or None,
            error_kind=error_kind or (structured_kind if isinstance(structured_kind, str) else None),
            abandoned=abandoned,
        )

    async def finished(
        self,
        *,
        outcome: str,
        duration_ms: int,
        result_metadata: Mapping[str, Any],
        error_kind: str | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        draft = self._finished_draft(
            outcome=outcome, duration_ms=duration_ms, result_metadata=result_metadata, error_kind=error_kind
        )
        try:
            await self._context.sink.emit(draft)
        except Exception:
            logger.debug("Trajectory tool.operation.finished emit failed", exc_info=True)

    def abandoned_soon(
        self,
        *,
        tool_name: str,
        tool_kind: str,
        batch_index: int | None,
        invocation_order: int | None,
        arguments: object,
        invocation_metadata: Mapping[str, Any],
        tool_context: Mapping[str, Any] | None,
        outcome: str,
        preamble_operation_id: str | None = None,
    ) -> None:
        """Write the whole pair for an operation that never got to open.

        The kernel hands the operation over the moment it dispatches the
        batch, but the middleware still has preprocessing to do — draining
        intermediate text, before-tool hooks, a mutation lock, the start event
        on the bus — and a cancellation or failure landing in there would
        otherwise leave a minted operation id with no event at all. Both lines
        are queued without awaiting: a task unwinding from a cancellation has
        no await left to spend.
        """
        if self._started or self._finished:
            return
        self._started = True
        self._finished = True
        try:
            started = tool_operation_started_draft(
                self._context,
                operation_id=self._operation_id,
                tool_name=tool_name,
                tool_kind=tool_kind,
                invocation_order=invocation_order,
                batch_index=batch_index,
                arguments=arguments,
                call_item_id=_metadata_id(invocation_metadata, ANALYTICS_ITEM_ID_KEY),
                tool_context=tool_context,
                caused_by_operation_id=preamble_operation_id,
            )
            finished = self._finished_draft(
                outcome=outcome,
                duration_ms=0,
                result_metadata={},
                error_kind=outcome,
                abandoned=True,
            )
            self._context.sink.emit_soon(started)
            self._context.sink.emit_soon(finished)
        except Exception:
            logger.debug("Trajectory abandoned tool.operation emit failed", exc_info=True)

    def finished_soon(self, *, outcome: str, duration_ms: int) -> None:
        """Synchronous close for the cancellation path (queued in order, ack not awaited)."""
        if self._finished:
            return
        self._finished = True
        draft = self._finished_draft(outcome=outcome, duration_ms=duration_ms, result_metadata={}, error_kind=None)
        try:
            self._context.sink.emit_soon(draft)
        except Exception:
            logger.debug("Trajectory tool.operation.finished emit failed", exc_info=True)
