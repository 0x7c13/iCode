# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""History preparation stamps occurrence identities before binding live state."""

from __future__ import annotations

import pytest

from chrys.foundation.trajectory.metadata import read_analytics_item_id
from chrys.kernel import Content, Message
from chrys.service.session.history import SessionHistoryManager, stamp_history_item_ids


def test_history_bind_only_replaces_reference_without_stamping() -> None:
    message = Message("user", ["legacy"], message_id="msg_3")
    state = {"messages": [message]}
    history = SessionHistoryManager()
    history.bind(state)
    assert history.state is state
    assert message.additional_properties == {}
    assert message.message_id == "msg_3"


def test_explicit_history_stamping_preserves_existing_occurrence_and_message_ids() -> None:
    call = Content.from_function_call("call", "tool", arguments={})
    message = Message("assistant", [call], message_id="msg_3")
    state = {"messages": [message]}
    stamp_history_item_ids(state)
    message_id = read_analytics_item_id(message.additional_properties)
    call_id = read_analytics_item_id(call.additional_properties)
    assert message_id
    assert call_id
    stamp_history_item_ids(state)
    history = SessionHistoryManager()
    history.bind(state)
    assert read_analytics_item_id(message.additional_properties) == message_id
    assert read_analytics_item_id(call.additional_properties) == call_id
    assert message.message_id == "msg_3"


@pytest.mark.parametrize("state", [{}, {"messages": []}, {"messages": None}, {"messages": ()}])
def test_history_stamping_keeps_non_message_states_unchanged(state) -> None:
    before = state.copy()
    stamp_history_item_ids(state)
    assert state == before
