# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for shared tool-result metadata status helpers."""

from __future__ import annotations

import pytest

from chrys.foundation.tool_result_metadata import (
    PROCESS_EXIT_CODE_METADATA_KEY,
    PROCESS_TIMED_OUT_METADATA_KEY,
    SHELL_EXIT_CODE_METADATA_KEY,
    SHELL_TIMED_OUT_METADATA_KEY,
    TOOL_ERROR_KIND_METADATA_KEY,
    TOOL_ERRORED_METADATA_KEY,
    TOOL_FAILED_METADATA_KEY,
    canonical_tool_result_status,
    legacy_error_text_applies,
    tool_result_metadata_failure_state,
)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({TOOL_FAILED_METADATA_KEY: True}, True),
        ({TOOL_FAILED_METADATA_KEY: False}, False),
        ({TOOL_FAILED_METADATA_KEY: "true"}, None),
        ({TOOL_FAILED_METADATA_KEY: False, TOOL_ERRORED_METADATA_KEY: True}, True),
        ({TOOL_FAILED_METADATA_KEY: False, PROCESS_EXIT_CODE_METADATA_KEY: 1}, True),
        ({TOOL_FAILED_METADATA_KEY: False, PROCESS_EXIT_CODE_METADATA_KEY: 0}, False),
        ({PROCESS_TIMED_OUT_METADATA_KEY: True}, True),
        ({PROCESS_TIMED_OUT_METADATA_KEY: "true"}, None),
        ({SHELL_EXIT_CODE_METADATA_KEY: 1}, True),
        ({TOOL_FAILED_METADATA_KEY: False, SHELL_EXIT_CODE_METADATA_KEY: 0}, False),
        ({SHELL_TIMED_OUT_METADATA_KEY: True}, True),
        ({TOOL_ERROR_KIND_METADATA_KEY: "validation"}, None),
        ({TOOL_ERROR_KIND_METADATA_KEY: "hook_denied"}, True),
        ({TOOL_ERROR_KIND_METADATA_KEY: "approval_rejected"}, True),
        ({"approval": "user_rejected", TOOL_FAILED_METADATA_KEY: False}, True),
    ],
)
def test_tool_result_metadata_failure_state(metadata: dict[str, object], expected: bool | None) -> None:
    assert tool_result_metadata_failure_state(metadata) is expected


def test_legacy_error_text_applies_to_known_unkinded_chrys_tool_names() -> None:
    assert legacy_error_text_applies("", "run_skill_script") is True
    assert legacy_error_text_applies("", "compress_context") is True
    assert legacy_error_text_applies("", "read_file") is True
    assert legacy_error_text_applies("", "remote_lookup") is False


def test_legacy_error_text_parity_for_skill_and_context_kinds() -> None:
    """Stamping the new kinds must not regress replay failure detection.

    The name fallback only fires when the kind is empty, so KIND_SKILL /
    KIND_CONTEXT must be in ERROR_TEXT_TOOL_KINDS or every legacy-listed
    skill/context tool would silently lose ``Error:`` text detection the
    moment its calls persist a kind.
    """
    from chrys.foundation.tool_kinds import KIND_CONTEXT, KIND_SKILL

    for name in ("load_skill", "read_skill_resource", "run_skill_script"):
        assert legacy_error_text_applies(KIND_SKILL, name) is True
    for name in ("compress_context", "recall_context"):
        assert legacy_error_text_applies(KIND_CONTEXT, name) is True


def test_legacy_error_text_list_compressed_contexts_broadening_is_forward_only() -> None:
    """`list_compressed_contexts` gains coverage only through the kind.

    It was never in LEGACY_ERROR_TEXT_TOOL_NAMES, so old sessions (empty
    kind) keep today's behavior; new sessions treat the whole
    context-management family uniformly.
    """
    from chrys.foundation.tool_kinds import KIND_CONTEXT

    assert legacy_error_text_applies("", "list_compressed_contexts") is False
    assert legacy_error_text_applies(KIND_CONTEXT, "list_compressed_contexts") is True


@pytest.mark.parametrize(
    ("provider_status", "metadata", "expected"),
    [
        ("completed", {"failed": True}, "completed"),
        ("failed", {"failed": False}, "failed"),
        ("interrupted", {"failed": False}, "interrupted"),
        ("running", {"failed": True}, "failed"),
        ("pending", {"failed": False}, "completed"),
        (None, {"shell_exit_code": 1}, "failed"),
        ("unknown", {}, "completed"),
    ],
)
def test_terminal_provider_status_precedes_result_metadata(
    provider_status: str | None, metadata: dict[str, object], expected: str
) -> None:
    assert canonical_tool_result_status(provider_status, metadata) == expected
