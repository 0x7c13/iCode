# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Approval concerns carry stable fields and render in the selected language."""

from __future__ import annotations

import pytest

from chrys.app.tui.screens.dialogs.approval.review_reason import review_reason
from chrys.app.tui.screens.main.view_adapter import _judge_verdict
from chrys.foundation.events.types import ApprovalReviewed
from chrys.foundation.i18n import Localizer
from chrys.service.approval.judge import JudgeVerdict


def test_triggered_conditions_reach_chinese_dialog():
    event = ApprovalReviewed(
        approved=False, reason="Requires human review: destructive_action.", triggered=("destructive_action",)
    )
    verdict = _judge_verdict(event)
    text = review_reason(verdict, Localizer("zh-Hans"))
    assert "删除" in text
    assert "destructive_action" not in text
    assert "Requires human review" not in text


@pytest.mark.parametrize(
    "kind",
    ["invalid_input", "asset_unavailable", "timeout", "cancelled", "invalid_shape", "invalid_verdict", "rate_limited"],
)
def test_fixed_failure_reasons_are_localized(kind):
    event = ApprovalReviewed(approved=False, reason="English failure", failure_reason=kind)
    text = review_reason(_judge_verdict(event), Localizer("zh-Hans"))
    assert "English failure" not in text
    assert any("\u4e00" <= character <= "\u9fff" for character in text)


def test_direct_reason_keeps_model_language():
    reason = "模型说明为什么需要人工确认"
    assert review_reason(JudgeVerdict(False, reason), Localizer("en")) == reason
