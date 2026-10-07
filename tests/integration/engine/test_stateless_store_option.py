# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A ``store`` option keeps the turn on the local retry lane where the protocol keeps no conversation.

Chat Completions and the Messages API keep no conversation state, so a
profile that asks for ``store: true`` (a Chat Completions option that only
keeps the completion for evals, or one copied into an Anthropic profile's
``extra_body``) still retries a failed wire call in place, as a profile
without the option does.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Error, InvocationMessage, InvocationRetryAttempt, UserMessage
from chrys.foundation.models.workspace import Workspace
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.event_capture import capture_events
from tests.support.llm_client_engines import parent_profile
from tests.support.scripted_wire import ScriptedWire, pin_wire_inputs, route_clients_to
from tests.support.waiting import ENGINE_TURN_TIMEOUT, await_run_task_chain, wait_for
from tests.support.wire_cases import Reply
from tests.support.wire_cases._kit import anth_replies, anth_text, cc_replies, cc_text

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support.engines import AgentEngineFactory

_UNAVAILABLE = Reply(
    503,
    json.dumps({"type": "error", "error": {"type": "api_error", "message": "unavailable"}}).encode("utf-8"),
    (("content-type", "application/json"),),
)


def _answers(provider: str, *, stream: bool) -> tuple[Reply, ...]:
    if provider == "anthropic":
        return anth_replies([anth_text("answer", message_id="msg_1")], stream=stream)
    return cc_replies([cc_text("answer", response_id="chatcmpl_1")], stream=stream)


@pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
@pytest.mark.parametrize(
    ("provider", "chat_options"),
    [("openai", {"store": True}), ("anthropic", {"extra_body": {"store": True}})],
    ids=["chat-completions", "anthropic"],
)
async def test_a_stored_stateless_profile_retries_a_failed_call_in_place(
    agent_engine: AgentEngineFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    chat_options: dict[str, Any],
    stream: bool,
) -> None:
    pin_wire_inputs(monkeypatch)
    monkeypatch.setattr(TurnBindings, "_BACKOFF_SCHEDULE", (0,))
    wire = ScriptedWire([_UNAVAILABLE, *_answers(provider, stream=stream)])
    route_clients_to(wire.transport, monkeypatch)
    models = ModelProfileRegistry()
    models.register(
        ModelProfile(
            id="model",
            name="model",
            provider=provider,
            model_id="model-test",
            api_key="sk-test",
            base_url="https://model.example.test",
            http_max_retries=0,
            stream=stream,
            chat_options=json.dumps(chat_options),
        )
    )
    profile = parent_profile(sub_agent=False)
    agents = AgentProfileRegistry()
    agents.register(profile)
    bus = EventBus()
    engine = agent_engine(
        bus,
        settings=Settings(model_profile="model", workspace_change_notice=False),
        agent_registry=agents,
        model_registry=models,
        initial_workspace=Workspace.from_cwd(str(tmp_path)),
    )
    await engine.start(profile)
    messages = await capture_events(bus, InvocationMessage)
    retries = await capture_events(bus, InvocationRetryAttempt)
    errors = await capture_events(bus, Error)

    await bus.publish(UserMessage(text="hello"))
    await wait_for(
        lambda: errors or any(message.is_final and message.origin.kind == "turn" for message in messages),
        timeout=ENGINE_TURN_TIMEOUT,
        description="turn to end",
    )
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    assert errors == []
    assert [(retry.origin.kind, retry.scope) for retry in retries] == [("turn", "wire")]
    assert len(wire.requests) == 2
