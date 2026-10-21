# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Standard and extension ACP projections for every invocation origin."""

from __future__ import annotations

import pytest

from chrys.app.acp.bridge import AcpEventBridge
from chrys.app.acp.server import ChrysAcpServer
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationStarted
from chrys.foundation.models.invocations import InvocationOrigin
from tests.app.acp._server_fakes import _FakeClient, _FakeHost, _FakeManager
from tests.support.invocation_events import ORIGIN_IDS, ORIGINS, PHASES, PROSE_PHASES, has_chat_card, projection_event


@pytest.mark.parametrize("origin", ORIGINS, ids=ORIGIN_IDS)
@pytest.mark.parametrize("phase", PHASES)
async def test_standard_and_extension_projection_matrix(origin: InvocationOrigin, phase: str) -> None:
    bridge = AcpEventBridge()
    bridge.updates_for_event(InvocationStarted(origin=ORIGINS[1], parent_call_id="parent"))
    event = projection_event(origin, phase)
    updates = bridge.updates_for_event(event)
    chat_child = has_chat_card(origin)
    standard_allowed = (origin.kind == "turn" and phase in ("answer", "reasoning", "tool", "args")) or (
        chat_child and phase not in PROSE_PHASES and phase != "pressure"
    )
    assert bool(updates) is standard_allowed
    if chat_child:
        assert all(update.session_update == "tool_call_update" for update in updates)
        assert all(update.tool_call_id == "parent" for update in updates)

    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(_FakeHost(event_bus=EventBus())), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    handled = await server._handle_chrys_extension_event("s1", event)
    if phase == "pressure":
        # A workflow run's pressure, from its node or a node's child, is not the chat session's.
        assert bool(client.ext_notifications) is (origin.root.kind != "workflow_node")
        assert handled is (origin.root.kind != "workflow_node")
        if client.ext_notifications:
            method, payload = client.ext_notifications[0]
            assert method == "chrys/context_pressure"
            assert payload["invocationId"] == (origin.invocation_id if origin.kind == "sub_agent" else "")
        return
    extension_allowed = chat_child and phase not in PROSE_PHASES and phase not in {"pressure", "args"}
    assert bool(client.ext_notifications) is extension_allowed
    # Every child extension also takes the standard bridge path.
    assert handled is False
    if extension_allowed:
        assert len(client.ext_notifications) == 1
        method, payload = client.ext_notifications[0]
        assert method.startswith("chrys/sub_agent_")
        assert payload["invocationId"] == origin.invocation_id
        assert payload["sessionId"] == "s1"
        assert "origin" not in payload
        assert "scope" not in payload


async def test_child_prose_never_touches_the_parent_segment() -> None:
    bridge = AcpEventBridge()
    first = bridge.updates_for_event(projection_event(ORIGINS[0], "answer"))
    # The child's segment shares the parent's id, so only its origin keeps them apart.
    for phase in [*sorted(PROSE_PHASES), "wire"]:
        assert bridge.updates_for_event(projection_event(ORIGINS[1], phase)) == []
    second = bridge.updates_for_event(projection_event(ORIGINS[0], "answer"))
    assert [update.content.text for update in first + second] == ["main answer", "main answer"]
    assert first[0].message_id == second[0].message_id
