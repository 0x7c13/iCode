# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Child history repair preserves and backfill occurrence identities."""

from __future__ import annotations

from chrys.foundation.events.bus import EventBus
from chrys.foundation.trajectory.metadata import read_analytics_item_id
from chrys.kernel import AgentSession, Message
from chrys.service.session.history import stamp_history_item_ids
from tests.orchestration.sub_agents._controller_fixtures import _make_controller, _ScriptedAgent


def test_child_history_repair_stamps_legacy_items_and_retains_existing_ids() -> None:
    retained = Message("user", ["retained"], message_id="msg_1")
    legacy = Message("user", ["legacy"], message_id="msg_1")
    stamp_history_item_ids({"messages": [retained]})
    retained_id = read_analytics_item_id(retained.additional_properties)
    session = AgentSession()
    state = {"messages": [retained, legacy]}
    session.state["chrys_history"] = state
    shell = _make_controller(_ScriptedAgent(), EventBus(), session=session)
    shell.policy._repair_paused_history()
    assert session.state["chrys_history"] is state
    assert read_analytics_item_id(retained.additional_properties) == retained_id
    legacy_id = read_analytics_item_id(legacy.additional_properties)
    assert legacy_id
    assert legacy_id != retained_id
    shell.policy._repair_paused_history()
    assert session.state["chrys_history"] is state
    assert read_analytics_item_id(legacy.additional_properties) == legacy_id
    assert [message.message_id for message in state["messages"]] == ["msg_1", "msg_1"]
