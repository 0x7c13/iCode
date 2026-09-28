# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Quota exhaustion: the SDK's own retries stay, Chrys's outer lanes stop."""

from __future__ import annotations

import httpx
import pytest

from chrys.foundation.events.types import Error
from tests.support.mock_provider_turns import mock_provider_profile, run_mock_provider_turn


@pytest.mark.parametrize("stream", [False, True])
async def test_insufficient_quota_is_not_retried_beyond_the_sdk(
    stream: bool, agent_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        body = {"error": {"type": "insufficient_quota", "code": "insufficient_quota", "message": "Quota exceeded"}}
        # ``retry-after-ms`` keeps the SDK's own backoff at a millisecond.
        return httpx.Response(429, headers={"retry-after-ms": "1"}, json=body, request=request)

    profile = mock_provider_profile("openai", stream=stream, http_max_retries=2)
    turn = await run_mock_provider_turn(agent_engine, monkeypatch, profile, respond)

    # The SDK's two retries run (it can't read the quota code from a streamed
    # 429 before retrying); no Chrys lane retries on top of them.
    assert len(turn.requests) == 3
    assert turn.retries == []
    assert isinstance(turn.terminal, Error)
