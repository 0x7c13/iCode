# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import pytest

from chrys.foundation.trajectory.ids import (
    ARTIFACT_ID_MAX_LENGTH,
    IdClass,
    classify_id_field,
    is_id_field,
    is_valid_analytics_id,
    is_valid_artifact_id,
    is_valid_id,
    is_valid_invocation_id,
    is_valid_opaque_id,
    is_valid_session_id,
    new_analytics_id,
)


def test_new_analytics_id_is_32_lower_hex_and_unique() -> None:
    ids = {new_analytics_id() for _ in range(64)}
    assert len(ids) == 64
    for value in ids:
        assert is_valid_analytics_id(value)
        assert len(value) == 32


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        ("event_id", IdClass.ANALYTICS),
        ("operation_id", IdClass.ANALYTICS),
        ("item_id", IdClass.ANALYTICS),
        ("some_brand_new_id", IdClass.ANALYTICS),
        ("session_id", IdClass.SESSION),
        ("origin_session_id", IdClass.SESSION),
        ("invocation_id", IdClass.INVOCATION),
        ("sub_agent_log_id", IdClass.INVOCATION),
        ("agent_profile_id", IdClass.PROFILE),
        ("model_profile_id", IdClass.PROFILE),
        ("compressed_context_id", IdClass.COMPRESSED_CONTEXT),
        ("artifact_id", IdClass.ARTIFACT),
        ("response_id", IdClass.OPAQUE),
        ("hook_id", IdClass.OPAQUE),
        ("child_session_id", IdClass.OPAQUE),
        ("provider_request_id", IdClass.OPAQUE),
        ("transport_exchange_id", IdClass.OPAQUE),
    ],
)
def test_classify_id_field(field_name: str, expected: IdClass) -> None:
    assert classify_id_field(field_name) == expected


def test_is_id_field_matches_suffix_and_item_id() -> None:
    assert is_id_field("event_id")
    assert is_id_field("item_id")
    assert not is_id_field("hook_key")
    assert not is_id_field("identity")
    assert not is_id_field("idle")
    assert not is_id_field("count")


def test_session_id_accepts_uuid_and_legacy_short_form() -> None:
    assert is_valid_session_id("12345678-1234-1234-1234-123456789abc")
    assert is_valid_session_id("0123456789ab")
    assert not is_valid_session_id("0123456789AB")
    assert not is_valid_session_id("not-a-session")
    assert not is_valid_session_id(12)


def test_invocation_and_opaque_validation() -> None:
    assert is_valid_invocation_id("0123456789ab")
    assert not is_valid_invocation_id("0123456789abc")
    assert is_valid_opaque_id("resp_abc-123/xyz")
    assert not is_valid_opaque_id("")
    assert not is_valid_opaque_id("x" * 513)
    assert not is_valid_opaque_id("has\nnewline")


def test_is_valid_id_dispatches_by_field_class() -> None:
    analytics = new_analytics_id()
    assert is_valid_id("operation_id", analytics)
    assert not is_valid_id("operation_id", "0123456789ab")
    assert is_valid_id("invocation_id", "0123456789ab")
    assert is_valid_id("response_id", "resp_anything")
    assert is_valid_id("model_profile_id", "my profile")
    assert not is_valid_id("model_profile_id", "")
    assert is_valid_id("compressed_context_id", "ctx_0123abcd")
    assert not is_valid_id("compressed_context_id", "ctx_0123ABCD")
    assert not is_valid_id("compressed_context_id", analytics)


def test_artifact_id_is_a_bare_file_name() -> None:
    # The shape spill files really have: prefix, short suffix, extension.
    assert is_valid_artifact_id("shell_1a2b3c4d.txt")
    assert is_valid_artifact_id("结果_1a2b3c4d.txt")
    assert is_valid_artifact_id("x" * ARTIFACT_ID_MAX_LENGTH)
    # A name is never a path, and never climbs out of its directory.
    assert not is_valid_artifact_id("tool_results/shell_1a2b3c4d.txt")
    assert not is_valid_artifact_id("tool_results\\shell_1a2b3c4d.txt")
    assert not is_valid_artifact_id("..")
    assert not is_valid_artifact_id(".")
    assert not is_valid_artifact_id("nul\x00byte.txt")
    assert not is_valid_artifact_id("two\nlines.txt")
    assert not is_valid_artifact_id("x" * (ARTIFACT_ID_MAX_LENGTH + 1))
    assert not is_valid_artifact_id("")
    assert not is_valid_artifact_id(None)


def test_a_spill_artifact_name_is_a_valid_id_but_not_an_analytics_one() -> None:
    name = "shell_1a2b3c4d.txt"
    assert is_valid_id("artifact_id", name)
    assert not is_valid_analytics_id(name)
    assert not is_valid_id("operation_id", name)
    assert not is_valid_id("artifact_id", "tool_results/" + name)
