# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for LLM route-session id derivation."""

from __future__ import annotations

from chrys.service.llm.route_sessions import derive_llm_route_session_id
from chrys.service.profiles.models.schema import ApiStyle, ModelProfile


def _profile(
    *,
    id: str = "model-a",
    provider: str = "openai",
    api_style: ApiStyle = "chat_completions",
    model_id: str = "gpt-test",
    base_url: str = "https://llm-a.example/v1",
) -> ModelProfile:
    return ModelProfile(
        id=id,
        name=id,
        provider=provider,
        api_style=api_style,
        model_id=model_id,
        base_url=base_url,
    )


def _derive(
    *,
    route_kind: str = "sub-agent",
    route_parts: tuple[str, ...] = ("Explore", "Explore"),
    model_profile: ModelProfile | None = None,
) -> str:
    return derive_llm_route_session_id(
        "parent-session",
        route_kind=route_kind,
        route_parts=route_parts,
        model_profile=model_profile or _profile(),
    )


def test_derive_llm_route_session_id_is_stable_for_same_inputs() -> None:
    assert _derive() == _derive()


def test_derive_llm_route_session_id_changes_for_route_identity_changes() -> None:
    base = _derive()

    assert _derive(route_kind="approval-judge") != base
    assert _derive(route_parts=("Plan", "Plan")) != base
    assert _derive(route_parts=("Explore", "DifferentProfile")) != base


def test_derive_llm_route_session_id_changes_for_model_identity_changes() -> None:
    base = _derive()

    assert _derive(model_profile=_profile(id="model-b")) != base
    assert _derive(model_profile=_profile(api_style="responses")) != base
    assert _derive(model_profile=_profile(model_id="gpt-other")) != base
    assert _derive(model_profile=_profile(provider="anthropic", model_id="claude-test")) != base
    assert _derive(model_profile=_profile(base_url="https://llm-b.example/v1")) != base


def test_derive_llm_route_session_id_without_parent_is_random() -> None:
    profile = _profile()

    first = derive_llm_route_session_id(None, route_kind="sub-agent", model_profile=profile)
    second = derive_llm_route_session_id(None, route_kind="sub-agent", model_profile=profile)

    assert first
    assert second
    assert first != second
