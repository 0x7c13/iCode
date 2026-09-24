# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for result-metadata persistence: the persistable subset and provenance carriage into the context."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
from chrys.foundation.tool_result_metadata import (
    PROCESS_EXIT_CODE_METADATA_KEY,
    TOOL_ERROR_CODE_METADATA_KEY,
    TOOL_ERROR_RETRYABLE_METADATA_KEY,
    TOOL_FAILED_METADATA_KEY,
    TOOL_RESULT_METADATA_KEY,
)
from chrys.service.agent_middleware.events.result_persistence import (
    persistable_result_metadata,
    write_result_carriage,
)


def _context(metadata: Any = None) -> Any:
    return SimpleNamespace(metadata={} if metadata is None else metadata)


def test_no_persistable_metadata_and_no_context_writes_nothing() -> None:
    context = _context()
    write_result_carriage(context, metadata={"file_snapshot": "sn-1"})
    assert context.metadata == {}


def test_allowlist_filters_non_persistable_keys() -> None:
    context = _context()
    write_result_carriage(
        context,
        metadata={"shell_exit_code": 1, "file_snapshot": "sn-1", "shell_file_snapshots": ["a"]},
    )
    assert context.metadata[TOOL_RESULT_METADATA_KEY] == {"shell_exit_code": 1}


def test_context_only_carriage_is_kept() -> None:
    """A successful call of an unclassified-but-contextualized tool must not vanish."""
    context = _context()
    write_result_carriage(context, metadata={}, tool_context={"skill_name": "pdf"})
    assert TOOL_RESULT_METADATA_KEY not in context.metadata
    assert context.metadata[TOOL_CALL_CONTEXT_METADATA_KEY] == {"skill_name": "pdf"}


def test_context_passthrough_is_a_copy() -> None:
    tool_context = {"skill_name": "pdf", "skill_revision": "a3f9c02b71de"}
    context = _context()
    write_result_carriage(context, metadata={}, tool_context=tool_context)
    assert context.metadata[TOOL_CALL_CONTEXT_METADATA_KEY] == tool_context
    assert context.metadata[TOOL_CALL_CONTEXT_METADATA_KEY] is not tool_context


def test_empty_context_does_not_emit_key() -> None:
    context = _context()
    write_result_carriage(context, metadata={"errored": True}, tool_context={})
    assert TOOL_CALL_CONTEXT_METADATA_KEY not in context.metadata
    assert context.metadata[TOOL_RESULT_METADATA_KEY] == {"errored": True}


def test_non_dict_context_metadata_is_replaced_before_write() -> None:
    context = _context(metadata=None)
    context.metadata = None
    write_result_carriage(context, metadata={"errored": True})
    assert context.metadata[TOOL_RESULT_METADATA_KEY] == {"errored": True}


def test_falsey_error_fields_survive_carriage() -> None:
    context = _context()
    write_result_carriage(context, metadata={"failed": False, "tool_error_retryable": False})
    assert context.metadata[TOOL_RESULT_METADATA_KEY] == {"failed": False, "tool_error_retryable": False}


def test_persistable_result_metadata_preserves_explicit_falsey_error_fields() -> None:
    metadata = {
        PROCESS_EXIT_CODE_METADATA_KEY: 0,
        TOOL_ERROR_RETRYABLE_METADATA_KEY: False,
        TOOL_ERROR_CODE_METADATA_KEY: 0,
        TOOL_FAILED_METADATA_KEY: False,
        "unrelated": False,
    }

    assert persistable_result_metadata(metadata) == {
        PROCESS_EXIT_CODE_METADATA_KEY: 0,
        TOOL_ERROR_RETRYABLE_METADATA_KEY: False,
        TOOL_ERROR_CODE_METADATA_KEY: 0,
        TOOL_FAILED_METADATA_KEY: False,
    }
