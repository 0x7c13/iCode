# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``context.revision.recorded``: exact request membership, recorded per actor without a tokenizer."""

from __future__ import annotations

from typing import Any

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.trajectory.context import side_call_actor
from chrys.foundation.trajectory.envelope import SEGMENT_EVENT_TYPE, MeasurementSource
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory.metadata import ANALYTICS_ITEM_ID_KEY
from chrys.foundation.trajectory.revisions import (
    ACTION_ADD,
    ACTION_REMOVE,
    CHECKPOINT_INTERVAL,
    membership_hash,
    membership_of,
)
from chrys.foundation.trajectory.segments import reassemble_array_slice
from chrys.kernel import Content, Message
from chrys.kernel.compaction import (
    GROUP_ANNOTATION_KEY,
    GROUP_TOKEN_COUNT_KEY,
    GROUP_TOKEN_ESTIMATOR_VERSION_KEY,
    TOKEN_ESTIMATOR_VERSION,
)
from chrys.service.trajectory.revisions import (
    BUCKET_COMPRESSED_SUMMARIES,
    BUCKET_CURRENT_USER,
    BUCKET_LIVE_HISTORY,
    BUCKET_SYSTEM,
    BUCKET_TOOL_RESULTS,
    REFS_POINTER,
    TOKENIZER_FINGERPRINT,
    record_context_revision,
)
from tests.service.trajectory._fakes import SESSION_ID, FakeSink, make_context


def _annotate(message: Message, tokens: int | None) -> Message:
    annotation: dict[str, Any] = {"id": new_analytics_id(), "kind": "user", "index": 0, "has_reasoning": False}
    if tokens is not None:
        annotation[GROUP_TOKEN_COUNT_KEY] = tokens
        annotation[GROUP_TOKEN_ESTIMATOR_VERSION_KEY] = TOKEN_ESTIMATOR_VERSION
    message.additional_properties[GROUP_ANNOTATION_KEY] = annotation
    return message


def _message(role: str, *, item_id: str | None, tokens: int | None = 10, contents: list[Content] | None = None):
    message = Message(role=role, contents=contents if contents is not None else [Content.from_text("hi")])
    if item_id is not None:
        message.additional_properties[ANALYTICS_ITEM_ID_KEY] = item_id
    return _annotate(message, tokens)


def _item_ids(count: int) -> list[str]:
    return [new_analytics_id() for _ in range(count)]


def _refs(sink: FakeSink) -> list[dict[str, Any]]:
    segments = [
        draft.payload for draft in sink.of_type(SEGMENT_EVENT_TYPE) if draft.payload["field_pointer"] == REFS_POINTER
    ]
    return reassemble_array_slice(segments)


# ---------------------------------------------------------------- recording


def test_first_request_records_a_parentless_checkpoint_of_every_item() -> None:
    sink = FakeSink()
    context = make_context(sink)
    ids = _item_ids(3)
    messages = [
        _message("system", item_id=ids[0]),
        _message("assistant", item_id=ids[1]),
        _message("user", item_id=ids[2]),
    ]

    revision_id = record_context_revision(context, messages)

    event = sink.only(EventType.CONTEXT_REVISION_RECORDED)
    assert revision_id == event.payload["revision_id"]
    assert event.operation_id == revision_id
    assert event.parent_operation_id == context.run_operation_id
    assert "parent_revision_id" not in event.payload
    assert event.payload["is_checkpoint"] is True
    assert event.payload["item_count"] == 3
    assert [ref["item_id"] for ref in _refs(sink)] == ids
    assert {ref["action"] for ref in _refs(sink)} == {ACTION_ADD}


def test_membership_hash_matches_the_ordered_item_ids() -> None:
    sink = FakeSink()
    ids = _item_ids(2)

    record_context_revision(make_context(sink), [_message("user", item_id=item_id) for item_id in ids])

    payload = sink.only(EventType.CONTEXT_REVISION_RECORDED).payload
    assert payload["membership_hash"] == membership_hash(sink.fingerprint_key, membership_of(ids))


def test_without_a_fingerprint_key_the_membership_hash_is_absent_not_bare() -> None:
    sink = FakeSink(fingerprint_key=None)

    record_context_revision(make_context(sink), [_message("user", item_id=new_analytics_id())])

    # A bare digest of item ids would be a privacy regression; absence is the contract.
    assert "membership_hash" not in sink.only(EventType.CONTEXT_REVISION_RECORDED).payload


def test_a_second_request_records_only_what_changed() -> None:
    sink = FakeSink()
    context = make_context(sink)
    ids = _item_ids(4)
    first = [_message("user", item_id=item_id) for item_id in ids[:3]]
    record_context_revision(context, first)
    first_id = sink.only(EventType.CONTEXT_REVISION_RECORDED).payload["revision_id"]

    # The middle item was compacted away and a new answer was appended.
    second = [_message("user", item_id=ids[0]), _message("user", item_id=ids[2]), _message("user", item_id=ids[3])]
    record_context_revision(context, second)

    event = sink.of_type(EventType.CONTEXT_REVISION_RECORDED)[1]
    assert event.payload["parent_revision_id"] == first_id
    assert event.payload["is_checkpoint"] is False
    assert event.payload["item_count"] == 3
    refs = _refs(sink)[3:]
    assert [(ref["item_id"], ref["action"], ref["position"]) for ref in refs] == [
        (ids[1], ACTION_REMOVE, 1),
        (ids[3], ACTION_ADD, 2),
    ]


def test_an_item_carried_twice_is_two_slots() -> None:
    sink = FakeSink()
    item_id = new_analytics_id()

    record_context_revision(make_context(sink), [_message("user", item_id=item_id) for _ in range(2)])

    assert [(ref["item_id"], ref["occurrence"], ref["position"]) for ref in _refs(sink)] == [
        (item_id, 0, 0),
        (item_id, 1, 1),
    ]


def test_items_without_a_stamped_id_are_left_out_of_membership() -> None:
    sink = FakeSink()
    stamped = new_analytics_id()

    record_context_revision(
        make_context(sink),
        [_message("system", item_id=None), _message("user", item_id=stamped)],
    )

    payload = sink.only(EventType.CONTEXT_REVISION_RECORDED).payload
    assert payload["item_count"] == 1
    assert [ref["item_id"] for ref in _refs(sink)] == [stamped]


def test_a_checkpoint_returns_on_schedule_within_one_chain() -> None:
    sink = FakeSink()
    context = make_context(sink)
    messages = [_message("user", item_id=item_id) for item_id in _item_ids(3)]

    for _ in range(CHECKPOINT_INTERVAL + 1):
        record_context_revision(context, messages)

    checkpoints = [
        index
        for index, draft in enumerate(sink.of_type(EventType.CONTEXT_REVISION_RECORDED))
        if draft.payload["is_checkpoint"]
    ]
    assert checkpoints == [0, CHECKPOINT_INTERVAL]


def test_each_actor_keeps_its_own_chain() -> None:
    sink = FakeSink()
    context = make_context(sink)
    messages = [_message("user", item_id=item_id) for item_id in _item_ids(2)]
    record_context_revision(context, messages)

    side = context.with_actor(side_call_actor(SESSION_ID, "title_gen"))
    record_context_revision(side, messages)
    record_context_revision(context, messages)

    events = sink.of_type(EventType.CONTEXT_REVISION_RECORDED)
    # The side call's first revision is a checkpoint of its own, not a delta of the main chain.
    assert "parent_revision_id" not in events[1].payload
    assert events[1].payload["is_checkpoint"] is True
    assert events[2].payload["parent_revision_id"] == events[0].payload["revision_id"]


def test_a_failing_sink_returns_none_and_does_not_advance_the_chain() -> None:
    sink = FakeSink()
    context = make_context(sink)
    messages = [_message("user", item_id=item_id) for item_id in _item_ids(2)]
    sink.fail_next = True

    assert record_context_revision(context, messages) is None

    # The dropped revision never became a parent: the next one is still the chain's first.
    assert record_context_revision(context, messages) is not None
    assert "parent_revision_id" not in sink.only(EventType.CONTEXT_REVISION_RECORDED).payload


def test_an_actorless_context_records_nothing() -> None:
    sink = FakeSink()
    context = make_context(sink)
    anonymous = context.with_actor(type(context.actor)(kind=context.actor.kind, role=context.actor.role))

    assert record_context_revision(anonymous, [_message("user", item_id=new_analytics_id())]) is None
    assert sink.drafts == []


# ------------------------------------------------------------ token buckets


def test_token_buckets_split_the_request_by_role_without_running_a_tokenizer() -> None:
    sink = FakeSink()
    summary = _message("assistant", item_id=new_analytics_id(), tokens=40)
    summary.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.SUMMARY
    tool_result = _message(
        "tool",
        item_id=new_analytics_id(),
        tokens=30,
        contents=[Content.from_function_result("call-1", result="ok")],
    )
    messages = [
        _message("system", item_id=new_analytics_id(), tokens=10),
        summary,
        _message("assistant", item_id=new_analytics_id(), tokens=20),
        tool_result,
        _message("user", item_id=new_analytics_id(), tokens=50),
    ]

    record_context_revision(make_context(sink), messages)

    payload = sink.only(EventType.CONTEXT_REVISION_RECORDED).payload
    assert payload["token_buckets"] == {
        BUCKET_SYSTEM: 10,
        BUCKET_LIVE_HISTORY: 20,
        BUCKET_COMPRESSED_SUMMARIES: 40,
        BUCKET_TOOL_RESULTS: 30,
        BUCKET_CURRENT_USER: 50,
    }
    assert payload["untokenized_item_count"] == 0
    assert payload["tokenizer_fingerprint"] == TOKENIZER_FINGERPRINT
    event = sink.only(EventType.CONTEXT_REVISION_RECORDED)
    assert event.measurements["/payload/token_buckets"]["source"] == MeasurementSource.LOCAL_TOKENIZER


def test_only_the_turns_own_user_message_counts_as_current_user() -> None:
    sink = FakeSink()
    messages = [
        _message("user", item_id=new_analytics_id(), tokens=5),
        _message("assistant", item_id=new_analytics_id(), tokens=5),
        _message("user", item_id=new_analytics_id(), tokens=7),
    ]

    record_context_revision(make_context(sink), messages)

    buckets = sink.only(EventType.CONTEXT_REVISION_RECORDED).payload["token_buckets"]
    # The earlier user message is history; only the last one is the live input.
    assert buckets[BUCKET_CURRENT_USER] == 7
    assert buckets[BUCKET_LIVE_HISTORY] == 10


def test_a_user_message_carrying_tool_results_is_not_the_current_input() -> None:
    sink = FakeSink()
    carrier = _message(
        "user",
        item_id=new_analytics_id(),
        tokens=9,
        contents=[Content.from_function_result("call-1", result="ok")],
    )
    messages = [_message("user", item_id=new_analytics_id(), tokens=4), carrier]

    record_context_revision(make_context(sink), messages)

    buckets = sink.only(EventType.CONTEXT_REVISION_RECORDED).payload["token_buckets"]
    assert buckets[BUCKET_TOOL_RESULTS] == 9
    assert buckets[BUCKET_CURRENT_USER] == 4


def test_unannotated_items_are_counted_rather_than_guessed() -> None:
    sink = FakeSink()
    messages = [
        _message("user", item_id=new_analytics_id(), tokens=None),
        _message("assistant", item_id=new_analytics_id(), tokens=6),
    ]

    record_context_revision(make_context(sink), messages)

    payload = sink.only(EventType.CONTEXT_REVISION_RECORDED).payload
    # An unmeasured item is reported as unmeasured, never as zero tokens.
    assert payload["untokenized_item_count"] == 1
    assert payload["token_buckets"][BUCKET_LIVE_HISTORY] == 6
    assert payload["item_count"] == 2


def test_recorded_payload_carries_no_message_text() -> None:
    sink = FakeSink()
    secret = "SENTINEL-do-not-log"
    messages = [_message("user", item_id=new_analytics_id(), contents=[Content.from_text(secret)])]

    record_context_revision(make_context(sink), messages)

    assert all(secret not in repr(draft.payload) for draft in sink.drafts)
