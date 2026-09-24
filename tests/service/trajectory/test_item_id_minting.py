# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Every persisted history item carries its analytics id from the moment it is minted.

A membership can only name items that already have an id.  Stamping at the
turn-ending save is too late for anything a request sends before that save —
tool-result carriers, compaction summaries and restored history all reach the
wire first — so each mint site stamps its own item and the revision counts
whatever it still could not name.
"""

from __future__ import annotations

from typing import Any

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.trajectory.ids import is_valid_analytics_id
from chrys.foundation.trajectory.metadata import ANALYTICS_ITEM_ID_KEY, read_analytics_item_id
from chrys.kernel import Message
from chrys.service.context.compaction.summaries import _make_summary_message
from chrys.service.context.providers.history import _compress_state
from chrys.service.session.history import SessionHistoryManager, stamp_history_item_ids
from chrys.service.trajectory.revisions import record_context_revision
from tests.service.trajectory._fakes import FakeSink, make_context


def _stamped(message: Message) -> str:
    item_id = read_analytics_item_id(message.additional_properties)
    assert item_id is not None, "message reached a request without an analytics id"
    assert is_valid_analytics_id(item_id)
    return item_id


def test_compaction_group_summary_is_stamped_where_it_is_minted() -> None:
    summary = _make_summary_message("summary text", "msg-1", ["msg-a", "msg-b"], "group-1")

    # The summary stands in for the whole group in every later request.
    _stamped(summary)


def test_folded_block_summary_is_stamped_where_it_is_minted() -> None:
    marker = Message("user", ["question"])
    marker.additional_properties["_turn_id"] = "turn-1"
    state: dict[str, Any] = {"messages": [marker, Message("assistant", ["answer"])], "compressed_msgs": []}

    _compress_state(state, "turn-1", "folded summary")

    summaries = [
        message
        for message in state["messages"]
        if message.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.SUMMARY
    ]
    assert len(summaries) == 1
    _stamped(summaries[0])


def test_restored_history_is_stamped_before_the_manager_binds() -> None:
    manager = SessionHistoryManager()
    # A session saved before analytics ids existed: nothing carries one.
    state: dict[str, Any] = {"messages": [Message("user", ["hi"]), Message("assistant", ["hello"])]}

    stamp_history_item_ids(state)
    manager.bind(state)

    ids = [_stamped(message) for message in state["messages"]]
    assert len(set(ids)) == 2

    # Preparing and binding again is idempotent: the restored identities must survive.
    stamp_history_item_ids(state)
    manager.bind(state)
    assert [_stamped(message) for message in state["messages"]] == ids


def test_binding_empty_or_absent_history_is_a_no_op() -> None:
    manager = SessionHistoryManager()
    manager.bind({})
    manager.bind({"messages": []})
    manager.bind({"messages": "not a list"})


def test_revision_counts_the_items_it_could_not_name() -> None:
    sink = FakeSink()
    context = make_context(sink)
    stamped = Message("user", ["named"])
    stamped.additional_properties[ANALYTICS_ITEM_ID_KEY] = "0" * 32
    unstamped = Message("assistant", ["unnamed"])

    revision_id = record_context_revision(context, [stamped, unstamped])

    assert revision_id is not None
    payload = sink.drafts[0].payload
    # Both items went on the wire; only one can enter the membership, and the
    # reader is told so rather than shown a request one item short.
    assert payload["item_count"] == 1
    assert payload["unidentified_item_count"] == 1


def test_a_fully_stamped_request_reports_no_unidentified_items() -> None:
    sink = FakeSink()
    context = make_context(sink)
    messages = [Message("user", ["one"]), Message("assistant", ["two"])]
    for index, message in enumerate(messages):
        message.additional_properties[ANALYTICS_ITEM_ID_KEY] = f"{index:032x}"

    assert record_context_revision(context, messages) is not None
    payload = sink.drafts[0].payload
    assert payload["item_count"] == 2
    assert payload["unidentified_item_count"] == 0
