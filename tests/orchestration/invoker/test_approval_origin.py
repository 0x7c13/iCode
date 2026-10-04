# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real main/child approval edits retain their invocation across the TUI boundary."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from chrys.app.tui.screens.main.state import MainScreenState, RunState
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    ApprovalRequest,
    ApprovalResponse,
    InvocationRetryAttempt,
    InvocationToolCallArgsUpdated,
    UserMessage,
)
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.foundation.retry import RetryAttemptInfo
from chrys.orchestration.invoker.origin import BoundEmitter, invocation_routing_key
from chrys.orchestration.sub_agents.tools import SubAgentTools
from chrys.service.agent_middleware.control.approval import ApprovalMiddleware
from chrys.service.approval.policy import ApprovalPolicy
from chrys.service.llm.mock import MockResponse
from chrys.service.profiles.agents.schema import ApprovalConfig
from tests.orchestration.invoker._build_fixtures import build_recipe_engine
from tests.support.event_capture import collect_events
from tests.support.tui_helpers import make_backend_handler


async def test_main_and_child_approval_edits_reach_their_own_cards(tmp_path, monkeypatch, agent_engine):
    target = tmp_path / "edited.txt"
    target.write_text("approved child contents")
    engine, main, child = await build_recipe_engine(
        agent_engine,
        monkeypatch,
        tmp_path,
        main=[MockResponse(tool_calls=[("Explore", "same-call", {"prompt": "old prompt"})]), MockResponse(text="done")],
        child=[MockResponse(tool_calls=[("read_file", "same-call", {"path": "old.txt"})]), MockResponse(text="read")],
    )
    engine.current.loaded.bindings.approval._policy = ApprovalPolicy(ApprovalConfig(default="require"))
    events, cards = [], []

    class Panel:
        def update_tool_args(self, call_id, args):
            cards.append(("turn", call_id, args))

        def update_sub_agent_tool_args(self, invocation_id, call_id, args):
            cards.append((invocation_id, call_id, args))

    panel = Panel()
    handler = make_backend_handler(
        SimpleNamespace(_state=MainScreenState(run=RunState(agent_running=True)), query_one=lambda _: panel)
    )

    async def approve(event: ApprovalRequest):
        args = {"prompt": "edited prompt"} if event.tool_name == "Explore" else {"path": str(target)}
        await engine.event_bus.publish(ApprovalResponse(request_id=event.request_id, approved=True, modified_args=args))

    async def updated(event: InvocationToolCallArgsUpdated):
        events.append(event)
        await handler.on_tool_args_updated(event)

    await engine.event_bus.subscribe(ApprovalRequest, approve)
    await engine.event_bus.subscribe(InvocationToolCallArgsUpdated, updated)
    try:
        await engine.event_bus.publish(UserMessage(text="delegate"))
        await engine.wait_for_run_task()
        assert not engine.current.loaded.bindings.state.run_failed
        assert [event.origin.kind for event in events] == ["turn", "sub_agent"]
        assert events[1].origin.parent == events[0].origin
        assert events[1].origin.invocation_id != events[0].origin.invocation_id
        assert cards == [
            ("turn", events[0].call_id, {"prompt": "edited prompt"}),
            (events[1].origin.invocation_id, events[1].call_id, {"path": str(target), "max_tokens": 5000}),
        ]
        assert "edited prompt" in child.call_history[0][0][-1].text
        assert any(
            "approved child contents" in str(content.result)
            for message in child.call_history[1][0]
            for content in message.contents
            if content.type == "function_result"
        )
        assert main.call_count == child.call_count == 2
    finally:
        await engine.event_bus.unsubscribe(ApprovalRequest, approve)
        await engine.event_bus.unsubscribe(InvocationToolCallArgsUpdated, updated)
        await engine.shutdown()


@pytest.mark.parametrize("kind", ["turn", "sub_agent"])
@pytest.mark.parametrize("replacement", ["none", "successor"])
async def test_waiting_approval_keeps_captured_origin_after_unbind(
    tmp_path, monkeypatch, agent_engine, kind, replacement
):
    from chrys.service.agent_middleware.events import sub_agent_events, tool_events

    # Deliberately collide presentation IDs as well as the provider call IDs.
    for module in (sub_agent_events, tool_events):
        monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="same-call"))
    target = tmp_path / "edited.txt"
    target.write_text("approved contents")
    engine, main, child = await build_recipe_engine(
        agent_engine,
        monkeypatch,
        tmp_path,
        main=[MockResponse(tool_calls=[("Explore", "same-call", {"prompt": "old"})]), MockResponse(text="done")],
        child=[MockResponse(tool_calls=[("read_file", "same-call", {"path": "old.txt"})]), MockResponse(text="read")],
    )
    engine.current.loaded.bindings.approval._policy = ApprovalPolicy(ApprovalConfig(default="require"))
    entered, release = asyncio.Event(), asyncio.Event()
    captured = []
    events, cards, successor_events = [], [], []
    original_process = ApprovalMiddleware.process

    async def process(middleware, context, call_next):
        if context.function.name == ("Explore" if kind == "turn" else "read_file"):
            captured.append((middleware, middleware._publisher))
        return await original_process(middleware, context, call_next)

    monkeypatch.setattr(ApprovalMiddleware, "process", create_autospec(original_process, side_effect=process))

    class Panel:
        def update_tool_args(self, call_id, args):
            cards.append(("turn", call_id, args))

        def update_sub_agent_tool_args(self, invocation_id, call_id, args):
            cards.append((invocation_id, call_id, args))

    panel = Panel()
    handler = make_backend_handler(
        SimpleNamespace(_state=MainScreenState(run=RunState(agent_running=True)), query_one=lambda _: panel)
    )

    async def approve(event):
        if event.tool_name == ("Explore" if kind == "turn" else "read_file"):
            entered.set()
            await release.wait()
        args = {"prompt": "edited prompt"} if event.tool_name == "Explore" else {"path": str(target)}
        await engine.event_bus.publish(ApprovalResponse(request_id=event.request_id, approved=True, modified_args=args))

    async def updated(event):
        events.append(event)
        await handler.on_tool_args_updated(event)

    successor_bus = EventBus()
    await successor_bus.subscribe(InvocationToolCallArgsUpdated, lambda event: collect_events(successor_events, event))
    await engine.event_bus.subscribe(ApprovalRequest, approve)
    await engine.event_bus.subscribe(InvocationToolCallArgsUpdated, updated)
    try:
        await engine.event_bus.publish(UserMessage(text="delegate"))
        await asyncio.wait_for(entered.wait(), 5)
        [(middleware, emitter)] = captured
        assert emitter is not None
        origin = emitter.origin
        if kind == "turn":
            registry = engine.current.loaded.bindings._invocation_publishers
            registry.unbind(origin)
            assert origin.invocation_id not in registry._emitters
        else:
            assert engine.current.loaded.sub_agent_tools._invocation_emitters.pop(origin.invocation_id) is emitter
        successor = BoundEmitter(successor_bus, InvocationOrigin(kind, origin.session_id, "successor", origin.parent))
        middleware.bind_publisher(None if replacement == "none" else successor)
        release.set()
        await engine.wait_for_run_task()
        assert engine.current.loaded.bindings.state.run_failed is False
        owned = [event for event in events if event.origin.kind == kind]
        assert len(owned) == 1
        assert owned[0].origin is origin
        assert owned[0].call_id == "same-call"
        key = "turn" if kind == "turn" else origin.invocation_id
        assert [card for card in cards if card[0] == key] == [(key, "same-call", owned[0].args)]
        assert not [card for card in cards if card[0] == "successor"]
        assert successor_events == []
        assert main.call_count == child.call_count == 2
    finally:
        release.set()
        await engine.event_bus.unsubscribe(ApprovalRequest, approve)
        await engine.event_bus.unsubscribe(InvocationToolCallArgsUpdated, updated)
        await engine.shutdown()


@pytest.mark.parametrize("key", ["", "missing"])
async def test_emitter_for_rejects_unbound_key_without_publication(key):
    bus = EventBus()
    tools = SubAgentTools(event_bus=bus)
    published = []
    await bus.subscribe(InvocationRetryAttempt, lambda event: collect_events(published, event))
    with pytest.raises(ValueError, match=r"^Cannot route an unbound invocation callback$"):
        tools._emitter_for(key)
    assert published == []
    origin = InvocationOrigin("sub_agent", "parent", "bound", None)
    emitter = BoundEmitter(bus, origin)
    tools._invocation_emitters[origin.invocation_id] = emitter
    assert tools._emitter_for("bound") is emitter
    await tools._emitter_for("bound").publish(InvocationRetryAttempt(origin=origin, scope="wire"))
    assert len(published) == 1
    assert published[0].origin is origin


async def test_real_retry_callback_rejects_unbound_registry_key(tmp_path, monkeypatch, agent_engine):
    from chrys.orchestration.sub_agents import tools as tools_module

    recipes = []
    original = tools_module.LastWordsInputs

    def inputs(*args, **kwargs):
        result = original(*args, **kwargs)
        recipes.append(result)
        return result

    monkeypatch.setattr(tools_module, "LastWordsInputs", create_autospec(original, side_effect=inputs))
    engine, _, _ = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, main=[], child=[])
    published = []
    await engine.event_bus.subscribe(InvocationRetryAttempt, lambda event: collect_events(published, event))
    token = invocation_routing_key.set("unbound-real-callback")
    try:
        assert recipes
        with pytest.raises(ValueError, match=r"^Cannot route an unbound invocation callback$"):
            await recipes[0]["publish_retry"](
                RetryAttemptInfo(attempt=1, max_attempts=2, delay_seconds=0, reason="retry")
            )
        assert published == []
    finally:
        invocation_routing_key.reset(token)
        await engine.shutdown()
