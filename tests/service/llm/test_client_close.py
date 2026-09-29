# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Closing a client stack closes its provider SDK client and HTTP pool, once."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

import chrys.service.llm.clients as clients_module
from chrys.kernel import ToolLoopLayer
from chrys.service.llm.anthropic_chat import RawAnthropicClient
from chrys.service.llm.clients import create_client, scoped_client
from chrys.service.llm.openai_chat_completion import RawOpenAIChatCompletionClient
from chrys.service.llm.openai_responses import RawOpenAIChatClient
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.close_races import ReleaseGate

if TYPE_CHECKING:
    from tests.support.llm_http_clients import HttpClientLedger


def _profile(provider: str, api_style: str = "chat_completions") -> ModelProfile:
    return ModelProfile(
        id="p",
        name="p",
        provider=provider,
        api_style=api_style,  # type: ignore[arg-type]
        model_id="test-model",
        api_key="sk-test",
    )


def _sdk_client(stack: ToolLoopLayer) -> Any:
    raw = stack.inner.inner
    if isinstance(raw, RawAnthropicClient):
        return raw.anthropic_client
    if not isinstance(raw, RawOpenAIChatCompletionClient | RawOpenAIChatClient):
        raise TypeError(f"unexpected raw client {type(raw).__name__}")
    return raw.client


@pytest.mark.parametrize(
    ("provider", "api_style"),
    [
        ("openai", "chat_completions"),
        ("openai", "responses"),
        ("anthropic", "chat_completions"),
        ("deepseek-openai", "chat_completions"),
        ("glm-openai", "chat_completions"),
    ],
)
async def test_closing_the_stack_closes_sdk_and_http_client(
    provider: str, api_style: str, http_client_ledger: HttpClientLedger
) -> None:
    stack = await create_client(_profile(provider, api_style))
    assert isinstance(stack, ToolLoopLayer)
    sdk = _sdk_client(stack)
    [http_client] = http_client_ledger.clients
    assert sdk._client is http_client
    assert not sdk.is_closed()

    await stack.aclose()

    assert sdk.is_closed()
    assert http_client_ledger.open() == []


async def test_a_provider_with_a_key_entry_but_no_stack_is_refused_and_its_pool_closed(
    monkeypatch: pytest.MonkeyPatch, http_client_ledger: HttpClientLedger
) -> None:
    # Never build another provider's stack (the last branch used to be GLM's) for it.
    monkeypatch.setitem(clients_module._PROVIDER_API_KEY_ENVS, "azure", "AZURE_API_KEY")

    with pytest.raises(ValueError, match="Unknown provider: 'azure'"):
        await create_client(_profile("azure"))

    assert len(http_client_ledger.clients) == 1
    assert http_client_ledger.open() == []


async def test_closing_the_mock_stack_does_nothing(http_client_ledger: HttpClientLedger) -> None:
    stack = await create_client(_profile("mock"))

    await stack.aclose()
    await stack.aclose()

    assert http_client_ledger.clients == []


async def test_close_is_one_shared_task_that_survives_waiter_cancellation(
    http_client_ledger: HttpClientLedger,
) -> None:
    stack = await create_client(_profile("openai"))
    [http_client] = http_client_ledger.clients
    release = ReleaseGate()
    close = http_client.aclose
    calls = 0

    async def gated_close() -> None:
        nonlocal calls
        calls += 1
        await release()
        await close()

    http_client.aclose = gated_close  # type: ignore[method-assign]
    first = asyncio.create_task(stack.aclose())
    second: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(release.entered.wait(), 5)
        first.cancel()
        second = asyncio.create_task(stack.aclose())
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()
        assert not http_client.is_closed

        release.proceed.set()
        await asyncio.wait_for(second, 5)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 5)
        assert calls == 1
        assert http_client.is_closed
    finally:
        release.proceed.set()
        await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)


async def test_scoped_client_closes_on_success_and_on_error(http_client_ledger: HttpClientLedger) -> None:
    async with scoped_client(_profile("openai")) as client:
        assert isinstance(client, ToolLoopLayer)
        assert len(http_client_ledger.open()) == 1
    assert http_client_ledger.open() == []

    error = RuntimeError("call failed")
    with pytest.raises(RuntimeError) as info:
        async with scoped_client(_profile("anthropic")):
            assert len(http_client_ledger.open()) == 1
            raise error
    assert info.value is error
    assert len(http_client_ledger.clients) == 2
    assert http_client_ledger.open() == []
