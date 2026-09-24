# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Runtime helpers for recording structured tool-result metadata."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar

from chrys.foundation.platform.files import surrogate_safe_text
from chrys.foundation.tool_result_metadata import (
    PROCESS_EXIT_CODE_METADATA_KEY,
    PROCESS_TIMED_OUT_METADATA_KEY,
    PROCESS_TIMEOUT_SECONDS_METADATA_KEY,
    TOOL_ERROR_CODE_METADATA_KEY,
    TOOL_ERROR_DETAILS_METADATA_KEY,
    TOOL_ERROR_KIND_METADATA_KEY,
    TOOL_ERROR_MESSAGE_METADATA_KEY,
    TOOL_ERROR_RETRYABLE_METADATA_KEY,
    TOOL_FAILED_METADATA_KEY,
)
from chrys.kernel._serialization import make_json_safe

tool_result_metadata: ContextVar[dict[str, object] | None] = ContextVar("tool_result_metadata", default=None)
"""Mutable metadata buffer set by tool event middleware for the active tool call."""


def record_tool_failure(
    kind: str,
    message: str,
    *,
    code: str | int | None = None,
    retryable: bool | None = None,
    details: Mapping[str, object] | None = None,
) -> None:
    """Record structured metadata for a tool failure in the active tool context."""
    metadata = tool_result_metadata.get(None)
    if metadata is None:
        return
    metadata[TOOL_FAILED_METADATA_KEY] = True
    metadata[TOOL_ERROR_KIND_METADATA_KEY] = kind
    metadata[TOOL_ERROR_MESSAGE_METADATA_KEY] = _unprefixed_error(message)
    if code is not None:
        metadata[TOOL_ERROR_CODE_METADATA_KEY] = code
    if retryable is not None:
        metadata[TOOL_ERROR_RETRYABLE_METADATA_KEY] = retryable
    if details is not None:
        metadata[TOOL_ERROR_DETAILS_METADATA_KEY] = make_json_safe(details)


def tool_error(
    kind: str,
    message: str,
    *,
    code: str | int | None = None,
    retryable: bool | None = None,
    details: Mapping[str, object] | None = None,
) -> str:
    """Return model-visible ``Error:`` text and record structured failure metadata."""
    safe_message = surrogate_safe_text(message)
    record_tool_failure(kind, safe_message, code=code, retryable=retryable, details=details)
    if safe_message.startswith("Error:"):
        return safe_message
    return f"Error: {safe_message}"


def record_tool_success() -> None:
    """Record structured success metadata in the active tool context."""
    metadata = tool_result_metadata.get(None)
    if metadata is None or metadata.get(TOOL_FAILED_METADATA_KEY) is True:
        return
    metadata[TOOL_FAILED_METADATA_KEY] = False


def record_process_result(exit_code: int) -> None:
    """Record generic process completion metadata for the active tool context."""
    metadata = tool_result_metadata.get(None)
    if metadata is None:
        return
    metadata[PROCESS_EXIT_CODE_METADATA_KEY] = exit_code
    if exit_code != 0:
        metadata[TOOL_FAILED_METADATA_KEY] = True
    elif metadata.get(TOOL_FAILED_METADATA_KEY) is not True:
        metadata[TOOL_FAILED_METADATA_KEY] = False


def record_process_timeout(timeout_seconds: int | float | None = None) -> None:
    """Record generic process timeout metadata for the active tool context."""
    metadata = tool_result_metadata.get(None)
    if metadata is None:
        return
    metadata[PROCESS_TIMED_OUT_METADATA_KEY] = True
    metadata[TOOL_FAILED_METADATA_KEY] = True
    if timeout_seconds is not None:
        metadata[PROCESS_TIMEOUT_SECONDS_METADATA_KEY] = make_json_safe(timeout_seconds)


def _unprefixed_error(message: str) -> str:
    if message.startswith("Error: "):
        return message[len("Error: ") :]
    if message.startswith("Error:"):
        return message[len("Error:") :].lstrip()
    return message
