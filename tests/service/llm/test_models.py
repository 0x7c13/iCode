# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the model registry."""

from __future__ import annotations

import pytest

from chrys.service.llm.models import KNOWN_MODELS, ModelInfo


def test_model_info_frozen() -> None:
    m = ModelInfo("anthropic", "claude-sonnet-4-5", "Claude Sonnet 4.5", 200_000)
    assert m.provider == "anthropic"
    assert m.supports_tools is True


def test_model_info_supports_tools_override() -> None:
    m = ModelInfo("test", "test-model", "Test", 1000, supports_tools=False)
    assert m.supports_tools is False


def test_known_models_not_empty() -> None:
    assert len(KNOWN_MODELS) > 0


def test_known_models_have_required_fields() -> None:
    for m in KNOWN_MODELS:
        assert m.provider, f"{m.model_id} missing provider"
        assert m.model_id, f"{m.display_name} missing model_id"
        assert m.display_name, f"{m.model_id} missing display_name"
        assert m.context_window > 0, f"{m.model_id} has invalid context_window"


def test_known_models_are_frozen() -> None:
    m = KNOWN_MODELS[0]
    with pytest.raises(AttributeError):
        m.provider = "changed"  # type: ignore[misc]
