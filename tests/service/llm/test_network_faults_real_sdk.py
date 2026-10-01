# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real SDK clients from ``create_client`` against injected DNS and connect faults.

Checks the kind and retry verdict, how many attempts the SDK's own retry loop
made under Chrys's deterministic-error guard, and that the displayed message
names the socket or resolver error rather than a generic wrapper, without the
address the socket dialed.
"""

from __future__ import annotations

import errno
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anthropic
import httpx
import openai
import pytest

from chrys.foundation.errors import ErrorKind, classify_error, clean_error_message
from chrys.service.llm.clients import create_client
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.network_faults import (
    INJECTED_V4,
    INJECTED_V6,
    NetworkFaults,
    gaierror,
    network_faults,
    os_error,
    socket_error_text,
)
from tests.support.provider_errors import API_HOST

_MAX_RETRIES = 2
_ATTEMPTS = _MAX_RETRIES + 1
_CHAT = [{"role": "user", "content": "hi"}]


@dataclass(frozen=True, slots=True)
class _Fault:
    arrange: Callable[[NetworkFaults], None]
    kind: ErrorKind
    retryable: bool
    # (resolver calls, connect attempts) the SDK made in all.
    calls: tuple[int, int]
    # Text of the failing leaf the displayed message must include.
    leaf_text: str


def _resolve_v4(then: Callable[[NetworkFaults], None]) -> Callable[[NetworkFaults], None]:
    def arrange(faults: NetworkFaults) -> None:
        faults.resolve_to(API_HOST, INJECTED_V4)
        then(faults)

    return arrange


def _dual_stack(v6: int, v4: int | None) -> Callable[[NetworkFaults], None]:
    def arrange(faults: NetworkFaults) -> None:
        faults.resolve_to(API_HOST, INJECTED_V6, INJECTED_V4)
        faults.fail_connect(INJECTED_V6, os_error(v6))
        if v4 is None:
            faults.refuse(INJECTED_V4)
        else:
            faults.fail_connect(INJECTED_V4, os_error(v4))

    return arrange


_FAULTS = {
    # The first hop never answered in this process: a typo, so one attempt.
    "dns-noname": _Fault(
        lambda faults: faults.fail_resolve(API_HOST, socket.EAI_NONAME),
        ErrorKind.DNS_FAILED,
        False,
        (1, 0),
        str(gaierror(socket.EAI_NONAME)),
    ),
    "dns-again": _Fault(
        lambda faults: faults.fail_resolve(API_HOST, socket.EAI_AGAIN),
        ErrorKind.DNS_FAILED,
        True,
        (_ATTEMPTS, 0),
        str(gaierror(socket.EAI_AGAIN)),
    ),
    "refused": _Fault(
        _resolve_v4(lambda faults: faults.refuse(INJECTED_V4)),
        ErrorKind.CONNECTION_REFUSED,
        True,
        (_ATTEMPTS, _ATTEMPTS),
        socket_error_text(errno.ECONNREFUSED),
    ),
    "unreachable": _Fault(
        _resolve_v4(lambda faults: faults.fail_connect(INJECTED_V4, os_error(errno.EHOSTUNREACH))),
        ErrorKind.HOST_UNREACHABLE,
        True,
        (_ATTEMPTS, _ATTEMPTS),
        socket_error_text(errno.EHOSTUNREACH),
    ),
    "dual-stack-mixed": _Fault(
        _dual_stack(errno.ENETUNREACH, None),
        ErrorKind.CONNECTION_REFUSED,
        True,
        (_ATTEMPTS, 2 * _ATTEMPTS),
        # The refused attempt got furthest, so it names the group.
        socket_error_text(errno.ECONNREFUSED),
    ),
    "dual-stack-no-route": _Fault(
        _dual_stack(errno.ENETUNREACH, errno.ENETUNREACH),
        ErrorKind.NO_ROUTE,
        True,
        (_ATTEMPTS, 2 * _ATTEMPTS),
        socket_error_text(errno.ENETUNREACH),
    ),
    "connect-timeout": _Fault(
        _resolve_v4(lambda faults: faults.hang_connect(INJECTED_V4)),
        ErrorKind.CONNECT_TIMEOUT,
        True,
        (_ATTEMPTS, _ATTEMPTS),
        "Connection timed out (ConnectTimeout)",
    ),
}


def _profile(provider: str) -> ModelProfile:
    base_url = f"http://{API_HOST}"
    return ModelProfile(
        id="p",
        name="p",
        provider=provider,
        model_id="test-model",
        api_key="sk-test",
        base_url=base_url if provider == "anthropic" else f"{base_url}/v1",
        http_max_retries=_MAX_RETRIES,
        http_connect_timeout=0.05,
    )


async def _fail(provider: str, sdk: Any) -> BaseException:
    try:
        if provider == "anthropic":
            await sdk.messages.create(model="test-model", max_tokens=8, messages=_CHAT)
        else:
            await sdk.chat.completions.create(model="test-model", messages=_CHAT)
    # A deterministic failure leaves the SDK as the transport error itself:
    # Chrys's retry guard re-raises it instead of sleeping.
    except (openai.APIError, anthropic.APIError, httpx.HTTPError) as exc:
        return exc
    raise AssertionError("the request did not fail")


@pytest.fixture
def direct_env(monkeypatch: pytest.MonkeyPatch, clear_proxy_env: Callable[[], None]) -> None:
    # With no proxy env at all, httpx falls back to the OS proxy settings.
    clear_proxy_env()
    monkeypatch.setenv("NO_PROXY", "*")


@pytest.mark.usefixtures("direct_env")
@pytest.mark.parametrize("fault", sorted(_FAULTS))
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_injected_fault_through_the_real_sdk(provider: str, fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _FAULTS[fault]
    stack = await create_client(_profile(provider))
    raw = stack.inner.inner
    sdk = raw.sdk_client
    # The SDK's own backoff, at the instance boundary: retries happen, instantly.
    monkeypatch.setattr(sdk, "_calculate_retry_timeout", lambda *_args, **_kwargs: 0.0)
    try:
        with network_faults() as faults:
            expected.arrange(faults)
            exc = await _fail(provider, sdk)
    finally:
        await stack.aclose()

    result = classify_error(exc)
    assert (result.kind, result.retryable) == (expected.kind, expected.retryable)
    assert (len(faults.resolve_calls), len(faults.connect_attempts)) == expected.calls
    message = clean_error_message(exc)
    assert expected.leaf_text in message
    # The text reaches the model: never the address the socket dialed.
    assert INJECTED_V4 not in message
    assert INJECTED_V6 not in message
