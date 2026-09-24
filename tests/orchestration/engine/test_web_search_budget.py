# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The per-pass web search call budget is reset by the engine for every pass.

The budget lives in a ContextVar, so the reset only reaches tool execution when
the builder's run-cycle start hook runs in an ancestor context of the tool
calls. This drives real passes through the engine and the builder's registry
instead of calling ``reset_budget()`` directly.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import create_autospec

import httpx
import pytest

import chrys.orchestration.engine.build.builder as builder_module
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationToolCallResult, UserMessage
from chrys.foundation.models.workspace import Workspace
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.profiles.agents.schema import AgentProfile, ApprovalConfig, CompactionConfig, ToolsConfig
from chrys.service.state.store import JsonFileStateStore
from tests.service.tools.web._support import Chunks, socket_network
from tests.support.engines import AgentEngineFactory
from tests.support.event_capture import capture_events
from tests.support.pipeline_helpers import make_mock_settings_and_registry
from tests.support.waiting import ENGINE_TURN_TIMEOUT

_PASS_LIMIT = 20
_BUDGET_ERROR = "Error: Web search call limit reached for this pass"


async def _exa_hit(request: httpx.Request) -> httpx.Response:
    # At the socket seam the pinned request still carries its body as a stream.
    query = json.loads(await request.aread())["params"]["arguments"]["query"]
    text = f"Title: Result for {query}\nURL: https://example.com/{len(query)}\nHighlights:\nSnippet."
    reply = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": text}]}}
    return httpx.Response(200, headers={"Content-Type": "application/json"}, stream=Chunks(json.dumps(reply).encode()))


async def test_each_pass_gets_a_fresh_web_search_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory
) -> None:
    settings, models = make_mock_settings_and_registry(stream=False)
    profile = AgentProfile(
        name="WebSearcher",
        instructions="Search the web.",
        # Enabling web_search without configuration selects the keyless Exa default.
        tools=ToolsConfig(builtins=["web_search"]),
        approval=ApprovalConfig(default="auto", overrides={"web_search": "auto"}),
        compaction=CompactionConfig(enabled=False),
    )
    first_pass = [("web_search", f"first-{index}", {"query": f"query {index}"}) for index in range(_PASS_LIMIT + 1)]
    client = MockChatClient(
        responses=[
            MockResponse(tool_calls=first_pass),
            MockResponse(text="Done"),
            MockResponse(tool_calls=[("web_search", "second-0", {"query": "the next pass"})]),
            MockResponse(text="Done"),
        ]
    )
    monkeypatch.setattr(
        builder_module, "create_client", create_autospec(builder_module.create_client, return_value=client)
    )
    bus = EventBus()
    results = await capture_events(bus, InvocationToolCallResult)
    engine = agent_engine(
        bus,
        settings=settings,
        model_registry=models,
        state_store=JsonFileStateStore(tmp_path / "sessions"),
        initial_workspace=Workspace.from_cwd(str(tmp_path)),
    )
    engine.pin_model_profile()
    await engine.start(profile)

    with socket_network(_exa_hit, {"mcp.exa.ai": "93.184.215.14"}) as network:
        await bus.publish(UserMessage(text="Search a lot"), raise_handler_errors=True)
        await asyncio.wait_for(engine.wait_for_run_task(), ENGINE_TURN_TIMEOUT)
        first = [event for event in results if event.tool_name == "web_search"]
        assert len(first) == _PASS_LIMIT + 1
        refused = [event for event in first if event.result == _BUDGET_ERROR]
        assert len(refused) == 1, [event.result[:80] for event in first]
        assert refused[0].metadata["tool_error_code"] == "call_budget_exceeded"
        assert all('"provider": "exa"' in event.result for event in first if event is not refused[0])
        # The refused call never reached the network.
        assert network.send.call_count == _PASS_LIMIT

        await bus.publish(UserMessage(text="Search again"), raise_handler_errors=True)
        await asyncio.wait_for(engine.wait_for_run_task(), ENGINE_TURN_TIMEOUT)
    second = [event for event in results if event.tool_name == "web_search"][len(first) :]
    assert len(second) == 1
    assert '"provider": "exa"' in second[0].result, second[0].result
    assert network.send.call_count == _PASS_LIMIT + 1
