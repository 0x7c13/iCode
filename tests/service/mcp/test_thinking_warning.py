# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""When on-demand MCP tool loading warns that it can unbind an agent's preserved thinking."""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Sequence
from typing import Any

import pytest

from chrys.service.mcp.thinking_warning import tool_loading_thinking_warning, warn_if_tool_loading_unbinds_thinking
from chrys.service.profiles.agents.schema import AgentProfile, MCPServerConfig, ToolsConfig
from chrys.service.profiles.models.options import effective_chat_options
from chrys.service.profiles.models.schema import ModelProfile, ThinkingBlockBinding

_ANTHROPIC = "https://api.anthropic.com"
_GATEWAY = "https://gateway.example"
_ADAPTIVE = {"thinking": {"type": "adaptive"}}
_BUDGETED = {"thinking": {"type": "enabled", "budget_tokens": 2048}}
_LOGGER = "chrys.service.mcp.thinking_warning"


def _model(
    *,
    model_id: str = "claude-opus-5-5",
    provider: str = "anthropic",
    base_url: str = _ANTHROPIC,
    options: dict[str, Any] | None = None,
    binding: ThinkingBlockBinding = "auto",
) -> ModelProfile:
    return ModelProfile(
        id="model",
        name="model",
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        chat_options=json.dumps(options) if options is not None else "",
        thinking_block_binding=binding,
    )


def _server(name: str = "docs", *, progressive: bool = True, enabled: bool = True) -> MCPServerConfig:
    return MCPServerConfig(
        name=name, transport="stdio", command="unused", use_progressive_disclosure=progressive, enabled=enabled
    )


def _agent(servers: Sequence[MCPServerConfig]) -> AgentProfile:
    return AgentProfile(name="Docs", tools=ToolsConfig(mcp=list(servers)))


def _warning(
    model: ModelProfile,
    servers: Sequence[MCPServerConfig] = (_server(),),
    connected: Collection[str] = ("docs",),
) -> str | None:
    return tool_loading_thinking_warning(_agent(servers), model, effective_chat_options(model), connected)


@pytest.mark.parametrize(
    "model",
    [
        _model(),
        _model(options=_BUDGETED),
        _model(options=_ADAPTIVE, base_url=_GATEWAY),
        _model(options=_ADAPTIVE, binding="off"),
        _model(binding="drop_block"),
        _model(model_id="claude-opus-4-7", options={"model": "claude-fable-5-1"}),
        _model(model_id="claude-opus-4-7", options={"extra_body": {"model": "claude-sonnet-5-5"}}),
    ],
    ids=[
        "no-thinking-configured",
        "budgeted-thinking",
        "gateway",
        "binding-off",
        "drop-block-without-thinking",
        "final-model-in-options",
        "final-model-in-extra-body",
    ],
)
def test_warns_when_tool_loading_can_get_a_request_refused(model: ModelProfile) -> None:
    message = _warning(model)

    assert message is not None
    assert "'docs'" in message
    assert "resent once without the earlier thinking" in message
    assert "thinking_block_binding: drop_block" in message


@pytest.mark.parametrize(
    ("block_binding", "binding"),
    [({}, "auto"), ({}, "drop_block"), ("drop_block", "drop_block")],
    ids=["empty", "empty-with-setting", "not-a-mapping"],
)
def test_the_warning_for_a_written_block_binding_names_the_field_to_write(
    block_binding: object, binding: ThinkingBlockBinding
) -> None:
    thinking = {"type": "adaptive", "block_binding": block_binding}
    message = _warning(_model(options={"thinking": thinking}, binding=binding))

    assert message is not None
    assert "resent once without the earlier thinking" in message
    assert 'Writing the block_binding in the thinking as {"prefix_mismatch_behavior": "drop_block"}' in message
    assert "thinking_block_binding" not in message


@pytest.mark.parametrize(
    "model",
    [_model(options=_ADAPTIVE, binding="error"), _model(binding="error")],
    ids=["sent", "setting-only"],
)
def test_the_warning_under_error_names_no_recovery(model: ModelProfile) -> None:
    message = _warning(model)

    assert message is not None
    assert "'docs'" in message
    assert "the prefix mismatch behavior is error" in message
    assert "resent" not in message
    assert "drop_block" not in message


@pytest.mark.parametrize(
    "model",
    [
        _model(options=_ADAPTIVE),
        _model(options=_BUDGETED, base_url=_GATEWAY, binding="drop_block"),
        _model(
            options={"thinking": {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}},
            base_url=_GATEWAY,
            binding="off",
        ),
        _model(options={"thinking": {"type": "disabled"}}),
        _model(model_id="claude-opus-4-7"),
        _model(options={"model": "claude-opus-4-7"}),
        _model(provider="openai", base_url=""),
    ],
    ids=[
        "auto-drops-blocks",
        "drop-block-setting",
        "drop-block-written",
        "thinking-disabled",
        "older-model",
        "final-model-older",
        "not-anthropic",
    ],
)
def test_no_warning_when_the_thinking_cannot_get_a_request_refused(model: ModelProfile) -> None:
    assert _warning(model) is None


@pytest.mark.parametrize(
    ("servers", "connected"),
    [
        ([], ()),
        ([_server(progressive=False)], ("docs",)),
        ([_server(enabled=False)], ("docs",)),
        ([_server()], ()),
    ],
    ids=["no-servers", "loads-every-tool", "disabled", "not-connected"],
)
def test_no_warning_without_a_connected_server_loading_tools_on_demand(
    servers: list[MCPServerConfig], connected: tuple[str, ...]
) -> None:
    assert _warning(_model(), servers, connected) is None


def test_the_default_endpoint_comes_from_the_environment_when_the_profile_names_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(options=_ADAPTIVE, base_url="")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert _warning(model) is None

    monkeypatch.setenv("ANTHROPIC_BASE_URL", _GATEWAY)
    assert _warning(model) is not None


def test_one_warning_names_every_connected_server_loading_tools_on_demand(caplog: pytest.LogCaptureFixture) -> None:
    servers = [
        _server("alpha"),
        _server("idle", enabled=False),
        _server("down"),
        _server("plain", progressive=False),
        _server("beta"),
    ]
    model = _model()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        warn_if_tool_loading_unbinds_thinking(
            _agent(servers), model, effective_chat_options(model), ("alpha", "plain", "beta")
        )

    [record] = [record for record in caplog.records if record.name == _LOGGER]
    assert record.levelno == logging.WARNING
    assert record.getMessage().startswith(
        "Agent 'Docs' on model profile 'model': MCP server(s) 'alpha', 'beta' load tools on demand"
    )


def test_nothing_is_logged_without_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    model = _model(options=_ADAPTIVE)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        warn_if_tool_loading_unbinds_thinking(_agent([_server()]), model, effective_chat_options(model), ("docs",))

    assert [record for record in caplog.records if record.name == _LOGGER] == []
