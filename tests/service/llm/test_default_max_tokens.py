# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for provider output token defaults."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from chrys.kernel import CompactionCallContext, Message, ResponseStream
from chrys.service.llm.anthropic_chat import RawAnthropicClient
from chrys.service.llm.deepseek import DeepSeekChatCompletionClient
from chrys.service.llm.defaults import FALLBACK_MAX_OUTPUT_TOKENS
from chrys.service.llm.glm import GLMChatCompletionClient
from chrys.service.llm.mock import MockChatClient
from chrys.service.llm.openai_chat_completion import RawOpenAIChatCompletionClient
from chrys.service.llm.openai_responses import RawOpenAIChatClient
from chrys.service.llm.token_limit_params import CHAT_COMPLETIONS_TOKEN_LIMIT_PARAMS


class _UnusedCompletions:
    async def create(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("default token tests should not call the SDK")


class _UnusedChat:
    def __init__(self) -> None:
        self.completions = _UnusedCompletions()


class _UnusedAsyncOpenAI:
    base_url = "https://api.test"

    def __init__(self) -> None:
        self.chat = _UnusedChat()


def _messages() -> list[Message]:
    return [Message("user", ["hi"])]


class _AdmissionStrategy:
    def __init__(self, *, last_included_tokens: int, max_context_tokens: int = 100) -> None:
        self.last_included_tokens = last_included_tokens
        self.max_context_tokens = max_context_tokens
        self.system_overhead_tokens = 0
        self.calibration_ratio = 1.0

    async def __call__(self, _messages: list[Message], _context: CompactionCallContext | None = None) -> bool:
        return False


async def _admitted_options(client: Any, options: dict[str, Any]) -> dict[str, Any]:
    strategy = _AdmissionStrategy(last_included_tokens=99)
    _prepared, wire_options = await client._prepare_wire_call(
        _messages(),
        call_context=CompactionCallContext(),
        options_snapshot=options,
        sanitized_client_kwargs={},
        compaction_overrides={"compaction_strategy": strategy},
    )
    return wire_options


def test_anthropic_defaults_max_tokens_and_preserves_explicit_value() -> None:
    client = RawAnthropicClient(model="claude-test", anthropic_client=SimpleNamespace())

    assert client._prepare_options(_messages(), {})["max_tokens"] == FALLBACK_MAX_OUTPUT_TOKENS
    assert client._prepare_options(_messages(), {"max_tokens": 0})["max_tokens"] == FALLBACK_MAX_OUTPUT_TOKENS
    assert client._prepare_options(_messages(), {"max_tokens": 4096})["max_tokens"] == 4096


def test_openai_chat_omits_default_max_tokens_and_preserves_explicit_values() -> None:
    client = RawOpenAIChatCompletionClient(model="gpt-test", async_client=_UnusedAsyncOpenAI())

    default_options = client._prepare_options(_messages(), {})
    assert "max_tokens" not in default_options
    assert "max_completion_tokens" not in default_options

    standard_options = client._prepare_options(_messages(), {"max_tokens": 4096})
    assert standard_options["max_completion_tokens"] == 4096

    native_options = client._prepare_options(_messages(), {"max_completion_tokens": 8192})
    assert native_options["max_completion_tokens"] == 8192


def test_deepseek_omits_default_max_tokens_and_uses_deepseek_request_field() -> None:
    client = DeepSeekChatCompletionClient(model="deepseek-reasoner", async_client=_UnusedAsyncOpenAI())

    default_options = client._prepare_options(_messages(), {})
    assert "max_tokens" not in default_options
    assert "max_completion_tokens" not in default_options

    standard_options = client._prepare_options(_messages(), {"max_tokens": 4096})
    assert standard_options["max_tokens"] == 4096
    assert "max_completion_tokens" not in standard_options

    openai_native_options = client._prepare_options(_messages(), {"max_completion_tokens": 8192})
    assert openai_native_options["max_tokens"] == 8192
    assert "max_completion_tokens" not in openai_native_options


def test_glm_omits_default_max_tokens_and_uses_glm_request_field() -> None:
    client = GLMChatCompletionClient(model="glm-5.2", async_client=_UnusedAsyncOpenAI())

    default_options = client._prepare_options(_messages(), {})
    assert "max_tokens" not in default_options
    assert "max_completion_tokens" not in default_options

    standard_options = client._prepare_options(_messages(), {"max_tokens": 4096})
    assert standard_options["max_tokens"] == 4096
    assert "max_completion_tokens" not in standard_options

    openai_native_options = client._prepare_options(_messages(), {"max_completion_tokens": 8192})
    assert openai_native_options["max_tokens"] == 8192
    assert "max_completion_tokens" not in openai_native_options


@pytest.mark.parametrize(
    ("provider", "client_type"),
    [
        ("openai", RawOpenAIChatCompletionClient),
        ("deepseek-openai", DeepSeekChatCompletionClient),
        ("glm-openai", GLMChatCompletionClient),
    ],
)
def test_chat_completions_clients_send_the_token_limit_param_of_their_provider(
    provider: str, client_type: type[RawOpenAIChatCompletionClient]
) -> None:
    """Each client sends the output cap under the parameter the Models screen names in its label."""
    client = client_type(model="model-test", async_client=_UnusedAsyncOpenAI())

    options = client._prepare_options(_messages(), {"max_tokens": 4096})

    assert {key: value for key, value in options.items() if value == 4096} == {
        CHAT_COMPLETIONS_TOKEN_LIMIT_PARAMS[provider]: 4096
    }


def test_instrumented_subclass_preserves_provider_token_limit_param() -> None:
    """The instrumented dynamic subclass must keep the provider's output-cap field."""
    from chrys.service.llm.instrumented import create_instrumented_openai_client

    stack = create_instrumented_openai_client(
        model_id="glm-5.2",
        client=_UnusedAsyncOpenAI(),
        chat_client_cls=GLMChatCompletionClient,
    )
    instrumented_client = stack.inner.inner

    options = instrumented_client._prepare_options(_messages(), {"max_tokens": 4096})
    assert options["max_tokens"] == 4096
    assert "max_completion_tokens" not in options


@pytest.mark.asyncio
async def test_openai_responses_omits_default_max_tokens_and_preserves_explicit_values() -> None:
    client = RawOpenAIChatClient(model="gpt-test", async_client=_UnusedAsyncOpenAI())

    default_options = await client._prepare_options(_messages(), {})
    assert "max_tokens" not in default_options
    assert "max_output_tokens" not in default_options

    standard_options = await client._prepare_options(_messages(), {"max_tokens": 4096})
    assert standard_options["max_output_tokens"] == 4096

    native_options = await client._prepare_options(_messages(), {"max_output_tokens": 8192})
    assert native_options["max_output_tokens"] == 8192


@pytest.mark.asyncio
async def test_provider_preparation_receives_admitted_output_caps() -> None:
    anthropic = RawAnthropicClient(model="claude-test", anthropic_client=SimpleNamespace())
    anthropic_options = await _admitted_options(anthropic, {"max_tokens": 4096})
    assert anthropic._prepare_options(_messages(), anthropic_options)["max_tokens"] == 1

    openai = RawOpenAIChatCompletionClient(model="gpt-test", async_client=_UnusedAsyncOpenAI())
    openai_options = await _admitted_options(openai, {"max_tokens": 4096})
    assert openai._prepare_options(_messages(), openai_options)["max_completion_tokens"] == 1

    deepseek = DeepSeekChatCompletionClient(model="deepseek-reasoner", async_client=_UnusedAsyncOpenAI())
    deepseek_options = await _admitted_options(deepseek, {"max_completion_tokens": 4096})
    assert deepseek._prepare_options(_messages(), deepseek_options)["max_tokens"] == 1

    responses = RawOpenAIChatClient(model="gpt-test", async_client=_UnusedAsyncOpenAI())
    responses_options = await _admitted_options(responses, {"max_tokens": 4096, "store": False})
    assert (await responses._prepare_options(_messages(), responses_options))["max_output_tokens"] == 16


@pytest.mark.asyncio
async def test_anthropic_admission_respects_thinking_budget() -> None:
    client = RawAnthropicClient(model="claude-test", anthropic_client=SimpleNamespace())
    strategy = _AdmissionStrategy(last_included_tokens=99)
    options = {"max_tokens": 4096, "thinking": {"type": "enabled", "budget_tokens": 20}}

    _prepared, admitted = await client._prepare_wire_call(
        _messages(),
        call_context=CompactionCallContext(),
        options_snapshot=options,
        sanitized_client_kwargs={},
        compaction_overrides={"compaction_strategy": strategy},
    )

    assert client._prepare_options(_messages(), admitted)["max_tokens"] == 21


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_mock_loop_adapter_uses_shared_context_admission(stream: bool) -> None:
    client = MockChatClient()
    client.compaction_strategy = _AdmissionStrategy(last_included_tokens=95)

    result = client.get_response(_messages(), stream=stream, options={"max_tokens": 50})
    if isinstance(result, ResponseStream):
        await result.get_final_response()
    else:
        await result

    assert client.call_history[0][1]["max_tokens"] == 5
