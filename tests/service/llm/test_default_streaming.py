# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A model profile file that never chose a streaming mode streams its main turn on the wire.

The engine, the OpenAI SDK and Chrys's client stack are real; only the HTTP
answers are scripted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from chrys.foundation.events.types import InvocationMessage
from chrys.service.profiles.models.loader import load_profile_from_yaml
from tests.support.mock_provider_turns import run_mock_provider_turn
from tests.support.wire_cases._kit import cc_replies, cc_text

_PROFILE_FILE = """\
name: file-profile
provider: openai
model_id: test-model
base_url: https://provider.example/v1
api_key: test-key
http_max_retries: 0
"""


@pytest.mark.parametrize(
    ("stream_line", "streams"),
    [("", True), ("stream: null\n", True), ("stream: false\n", False)],
    ids=["missing", "null", "false"],
)
async def test_a_profile_file_streams_unless_it_says_stream_false(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stream_line: str, streams: bool
) -> None:
    path = tmp_path / "file-profile.yaml"
    path.write_text(_PROFILE_FILE + stream_line, encoding="utf-8")
    (reply,) = cc_replies([cc_text("Sunny.", response_id="chatcmpl-1")], stream=streams)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(reply.status, headers=list(reply.headers), content=reply.body, request=request)

    turn = await run_mock_provider_turn(agent_engine, monkeypatch, load_profile_from_yaml(path), respond)

    assert [json.loads(request.content).get("stream", False) for request in turn.requests] == [streams]
    assert isinstance(turn.terminal, InvocationMessage)
    assert turn.terminal.text == "Sunny."
