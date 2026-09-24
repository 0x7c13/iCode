# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for model profile schema helpers."""

from __future__ import annotations

from typing import Any

import pytest

from chrys.service.profiles.models.schema import (
    UNCONFIGURED_MODEL_ID,
    VALID_PROVIDERS,
    ModelProfile,
    is_model_profile_selectable,
)


def _profile(**overrides: Any) -> ModelProfile:
    values: dict[str, Any] = {
        "id": "profile-id",
        "name": "Profile",
        "provider": "openai",
        "api_style": "chat_completions",
        "model_id": "gpt-test",
        "max_context_tokens": 200_000,
        "max_output_tokens": 32_000,
    }
    values.update(overrides)
    return ModelProfile(**values)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, True, id="valid-profile"),
        pytest.param({"model_id": ""}, False, id="empty-model-id"),
        pytest.param({"model_id": "   "}, False, id="blank-model-id"),
        pytest.param({"model_id": UNCONFIGURED_MODEL_ID}, False, id="unconfigured-model-id"),
        pytest.param({"provider": "mock", "model_id": ""}, True, id="mock-empty-model-id"),
        pytest.param({"provider": "unknown"}, False, id="unknown-provider"),
        pytest.param({"api_style": "completions"}, False, id="invalid-api-style"),
        pytest.param({"max_context_tokens": 0}, False, id="zero-context-tokens"),
        pytest.param({"max_context_tokens": -1}, False, id="negative-context-tokens"),
        pytest.param({"max_context_tokens": "200000"}, False, id="string-context-tokens"),
        pytest.param({"max_context_tokens": True}, False, id="bool-context-tokens"),
        pytest.param({"max_output_tokens": 0}, False, id="zero-output-tokens"),
        pytest.param({"max_output_tokens": -1}, False, id="negative-output-tokens"),
        pytest.param({"max_output_tokens": "32000"}, False, id="string-output-tokens"),
        pytest.param({"max_output_tokens": True}, False, id="bool-output-tokens"),
        pytest.param({"model_id": 123}, False, id="numeric-model-id"),
        pytest.param({"id": "   "}, False, id="blank-id"),
        pytest.param({"name": "   "}, False, id="blank-name"),
        pytest.param({"id": 123}, False, id="numeric-id"),
        pytest.param({"name": 123}, False, id="numeric-name"),
        pytest.param({"provider": 123}, False, id="numeric-provider"),
        pytest.param({"api_style": 123}, False, id="numeric-api-style"),
    ],
)
def test_is_model_profile_selectable(overrides: dict[str, Any], expected: bool) -> None:
    assert is_model_profile_selectable(_profile(**overrides)) is expected


def test_valid_providers_match_client_factory_branches() -> None:
    assert frozenset({"openai", "anthropic", "deepseek-openai", "glm-openai", "mock"}) == VALID_PROVIDERS
