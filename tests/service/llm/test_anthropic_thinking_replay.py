# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Thinking an Anthropic response returns is replayed, signed, in the next request of the tool loop."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from tests.support.mock_provider_turns import mock_provider_profile, run_mock_provider_turn
from tests.support.wire_cases._kit import anth_message, anth_replies, anth_text

MODES = pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
_CALL = {"type": "tool_use", "id": "toolu_1", "name": "nope", "input": {}}


async def _replayed(
    thinking: list[dict[str, Any]], *, stream: bool, agent_engine, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """The assistant message the tool loop's second request replays after a response of *thinking* and a call."""
    call = anth_message(message_id="msg_call", content=[*thinking, _CALL], stop_reason="tool_use")
    responses = [
        httpx.Response(reply.status, headers=list(reply.headers), content=reply.body)
        for reply in anth_replies([call, anth_text("done", message_id="msg_done")], stream=stream)
    ]

    turn = await run_mock_provider_turn(
        agent_engine, monkeypatch, mock_provider_profile("anthropic", stream=stream), lambda _: responses.pop(0)
    )

    assert len(turn.requests) == 2
    return json.loads(turn.requests[1].content)["messages"][1]


@MODES
@pytest.mark.parametrize("thought", ["", "Check the file."], ids=["omitted", "visible"])
async def test_thinking_replays_with_its_signature(
    thought: str, stream: bool, agent_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thinking replays with its signature; omitted thinking, an empty delta in a stream, as an empty block."""
    thinking = {"type": "thinking", "thinking": thought, "signature": "sig-1"}

    replayed = await _replayed([thinking], stream=stream, agent_engine=agent_engine, monkeypatch=monkeypatch)

    assert replayed == {"role": "assistant", "content": [thinking, _CALL]}


@MODES
async def test_each_omitted_thinking_block_replays_with_its_own_signature(
    stream: bool, agent_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitted thinking blocks in a row never merge into one."""
    thinking = [{"type": "thinking", "thinking": "", "signature": f"sig-{n}"} for n in (1, 2)]

    replayed = await _replayed(thinking, stream=stream, agent_engine=agent_engine, monkeypatch=monkeypatch)

    assert replayed == {"role": "assistant", "content": [*thinking, _CALL]}
