# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for tool base types."""

from __future__ import annotations

from chrys.service.approval.level import ApprovalLevel


def test_approval_level_values() -> None:
    assert ApprovalLevel.AUTO.value == "auto"
    assert ApprovalLevel.REQUIRE.value == "require"
    assert ApprovalLevel.SKIP.value == "skip"


def test_approval_level_from_string() -> None:
    assert ApprovalLevel("auto") is ApprovalLevel.AUTO
    assert ApprovalLevel("require") is ApprovalLevel.REQUIRE
    assert ApprovalLevel("skip") is ApprovalLevel.SKIP


def test_approval_level_members() -> None:
    assert len(ApprovalLevel) == 3
