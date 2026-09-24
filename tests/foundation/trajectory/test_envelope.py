# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from chrys.foundation.trajectory.envelope import (
    INT64_MAX,
    LINE_BUDGET_BYTES,
    SCHEMA_VERSION,
    SYSTEM_ACTOR,
    Actor,
    ActorKind,
    ActorRole,
    EnvelopeError,
    EventDraft,
    Link,
    LinkRelation,
    MeasurementSource,
    SegmentedField,
    build_event,
    check_int64_range,
    decode_event_dict,
    decode_event_line,
    encode_event_line,
    encode_json_value,
    iter_complete_lines,
    malformed_id_pointers,
    measurement,
    rfc3339_from_datetime,
    utc_now_rfc3339,
)
from chrys.foundation.trajectory.ids import new_analytics_id

SESSION_ID = "12345678-1234-1234-1234-123456789abc"


def _addressed(draft: EventDraft, *, sequence: int = 1):
    return build_event(
        draft,
        sequence=sequence,
        runtime_id=new_analytics_id(),
        coverage_id=new_analytics_id(),
        session_id=SESSION_ID,
        branch_id=new_analytics_id(),
    )


def test_draft_defaults_capture_identity_and_clocks() -> None:
    before = datetime.now(UTC)
    draft = EventDraft(event_type="turn.started")
    after = datetime.now(UTC)
    assert len(draft.event_id) == 32
    assert draft.actor == SYSTEM_ACTOR
    assert draft.occurred_at.endswith("Z")
    parsed = datetime.fromisoformat(draft.occurred_at)
    assert before - timedelta(seconds=1) <= parsed <= after + timedelta(seconds=1)
    assert isinstance(draft.monotonic_ns, int)
    # Two drafts never share an event id (the writer relies on this).
    assert EventDraft(event_type="x").event_id != draft.event_id


def test_build_event_carries_the_draft_event_id() -> None:
    draft = EventDraft(event_type="turn.started", payload={"turn_number": 1})
    event = _addressed(draft, sequence=7)
    assert event.event_id == draft.event_id
    assert event.sequence == 7
    assert event.schema_version == SCHEMA_VERSION
    assert event.payload == {"turn_number": 1}


def test_to_dict_omits_optional_keys_and_keeps_required_ones() -> None:
    """Acceptance 12: optional keys are omitted rather than written as null."""
    event = _addressed(EventDraft(event_type="turn.started"))
    data = event.to_dict()
    for key in (
        "schema_version",
        "event_id",
        "sequence",
        "runtime_id",
        "coverage_id",
        "session_id",
        "branch_id",
        "actor",
        "event_type",
        "occurred_at",
        "monotonic_ns",
        "payload",
    ):
        assert key in data
    for key in ("turn_id", "operation_id", "parent_operation_id", "links", "segmented_fields", "measurements"):
        assert key not in data
    assert "null" not in encode_event_line(event).decode()


def test_to_dict_includes_optional_keys_when_present() -> None:
    op = new_analytics_id()
    parent = new_analytics_id()
    turn = new_analytics_id()
    link = Link(relation=LinkRelation.RETRY_OF, target_operation_id=new_analytics_id())
    seg = SegmentedField(field_pointer="/payload/entries", segment_group_id=new_analytics_id(), segment_count=2)
    draft = EventDraft(
        event_type="model.exchange.finished",
        actor=Actor(kind=ActorKind.AGENT, role=ActorRole.SUB_AGENT, invocation_id="0123456789ab"),
        turn_id=turn,
        operation_id=op,
        parent_operation_id=parent,
        links=(link,),
        segmented_fields=(seg,),
        measurements={"/payload/duration_ms": measurement(MeasurementSource.MONOTONIC_CLOCK)},
        payload={"duration_ms": 12},
    )
    data = _addressed(draft).to_dict()
    assert data["turn_id"] == turn
    assert data["operation_id"] == op
    assert data["parent_operation_id"] == parent
    assert data["links"] == [link.to_dict()]
    assert data["segmented_fields"] == [seg.to_dict()]
    assert data["measurements"] == {"/payload/duration_ms": {"source": "monotonic_clock"}}
    assert data["actor"] == {"kind": "agent", "role": "sub_agent", "invocation_id": "0123456789ab"}


def test_round_trip_through_line_encoding() -> None:
    draft = EventDraft(
        event_type="tool.operation.finished",
        operation_id=new_analytics_id(),
        payload={"outcome": "ok", "nested": {"ids": [1, 2, 3]}, "text": "héllo ✓"},
    )
    event = _addressed(draft, sequence=3)
    line = encode_event_line(event)
    assert line.endswith(b"\n")
    assert b"\r\n" not in line
    decoded = decode_event_line(line)
    assert decoded == event


def test_encode_json_value_is_compact_utf8_and_never_raises_on_surrogates() -> None:
    assert encode_json_value({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'
    assert encode_json_value({"s": "é"}) == '{"s":"é"}'.encode()
    surrogate = "bad\udcff"
    encoded = encode_json_value({"s": surrogate})
    assert b"\\udcff" in encoded


def test_encode_rejects_nan() -> None:
    with pytest.raises(ValueError):
        encode_json_value({"x": float("nan")})


def test_decode_ignores_unknown_keys() -> None:
    """Acceptance 13: unknown fields are ignored."""
    event = _addressed(EventDraft(event_type="turn.started"))
    data = event.to_dict()
    data["future_top_level_key"] = {"anything": True}
    data["actor"]["future_actor_key"] = 1
    decoded = decode_event_dict(data)
    assert decoded.event_type == "turn.started"
    assert decoded.actor.kind == "system"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("sequence"),
        lambda d: d.pop("payload"),
        lambda d: d.update(sequence=0),
        lambda d: d.update(sequence="1"),
        lambda d: d.update(sequence=True),
        lambda d: d.update(monotonic_ns=-1),
        lambda d: d.update(schema_version=0),
        lambda d: d.update(actor="system"),
        lambda d: d.update(payload=[]),
        lambda d: d.update(event_id="short"),
        lambda d: d.update(session_id="nope"),
        lambda d: d.update(operation_id=""),
        lambda d: d.update(links=[{"relation": "parent"}]),
        lambda d: d.update(segmented_fields=[{"field_pointer": "/p", "segment_group_id": "x", "segment_count": 1}]),
        lambda d: d.update(measurements={"/p": "not-an-object"}),
        lambda d: d["payload"].update(operation_id="0123456789ab"),
        lambda d: d["payload"].update(nested={"items": [{"item_id": "bad"}]}),
        lambda d: d["payload"].update(big=INT64_MAX + 1),
        lambda d: d["payload"].update(big=[INT64_MAX + 1]),
    ],
)
def test_decode_rejects_structural_violations(mutate) -> None:
    data = _addressed(EventDraft(event_type="turn.started")).to_dict()
    mutate(data)
    with pytest.raises(EnvelopeError):
        decode_event_dict(data)


def test_decode_allows_nulls_and_foreign_ids_inside_opaque_subtrees() -> None:
    data = _addressed(
        EventDraft(
            event_type="model.exchange.finished",
            payload={
                "response_id": "resp_abc",
                "usage": {"provider_reported": {"request_id": "anything-goes", "weird_id": 42, "cached": None}},
                "entries": [{"key": "k", "value": {"some_id": "!!!"}}],
            },
        )
    ).to_dict()
    decoded = decode_event_dict(data)
    assert decoded.payload["usage"]["provider_reported"]["request_id"] == "anything-goes"
    # Mirrored verbatim: a provider that reported a null still reported it.
    assert decoded.payload["usage"]["provider_reported"]["cached"] is None


def test_a_draft_omits_the_payload_fields_it_has_nothing_to_say_about() -> None:
    draft = EventDraft(
        event_type="context.revision.recorded",
        payload={
            "revision_id": "a" * 32,
            "parent_revision_id": None,
            "membership_hash": None,
            "usage": {"total_tokens": 12, "provider_reported": None},
            "refs": [{"item_id": "b" * 32, "role": None}],
        },
    )

    assert draft.payload == {
        "revision_id": "a" * 32,
        "usage": {"total_tokens": 12},
        "refs": [{"item_id": "b" * 32}],
    }
    assert "parent_revision_id" not in _addressed(draft).to_dict()["payload"]


def test_decode_line_rejects_non_json_and_non_object() -> None:
    with pytest.raises(EnvelopeError):
        decode_event_line(b"\xff\xfe not utf8")
    with pytest.raises(EnvelopeError):
        decode_event_line(b"[1, 2, 3]\n")
    with pytest.raises(EnvelopeError):
        decode_event_line(b'{"truncated": ')


def test_check_int64_range_reports_pointers() -> None:
    offenders = check_int64_range({"ok": 1, "flag": True, "bad": 2**63, "deep": {"list": [1, 2**64]}})
    assert offenders == ["/bad", "/deep/list/1"]
    # The rule is about the integers a line carries, not how deeply a producer
    # happened to wrap them.
    assert check_int64_range({"nested": [[2**70]]}) == ["/nested/0/0"]


def test_malformed_ids_are_reported_at_any_depth() -> None:
    assert malformed_id_pointers({"payload": {"entries": [[{"item_id": "not-an-id"}]]}}) == [
        "/payload/entries/0/0/item_id"
    ]


def test_iter_complete_lines_splits_torn_tail() -> None:
    lines, tail = iter_complete_lines(b"a\nb\nc")
    assert list(lines) == [b"a", b"b"]
    assert tail == b"c"
    lines, tail = iter_complete_lines(b"a\n")
    assert list(lines) == [b"a"]
    assert tail == b""
    assert iter_complete_lines(b"") == ([], b"") or iter_complete_lines(b"")[1] == b""


def test_rfc3339_from_datetime_normalises_to_utc_z() -> None:
    naive = datetime(2026, 8, 19, 10, 0, 0)  # noqa: DTZ001 - naive input is the case under test
    assert rfc3339_from_datetime(naive) == "2026-08-19T10:00:00.000000Z"
    offset = datetime(2026, 8, 19, 12, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    assert rfc3339_from_datetime(offset) == "2026-08-19T10:00:00.000000Z"
    assert utc_now_rfc3339().endswith("Z")
    assert "+00:00" not in utc_now_rfc3339()


def test_measurement_includes_only_supplied_provenance() -> None:
    assert measurement(MeasurementSource.PROVIDER) == {"source": "provider"}
    assert measurement(MeasurementSource.LOCAL_TOKENIZER, method_version="4", adapter_version="a1") == {
        "source": "local_tokenizer",
        "method_version": "4",
        "adapter_version": "a1",
    }


def test_typical_event_fits_well_under_the_line_budget() -> None:
    """Acceptance 22: a representative event stays far below 4 KiB."""
    draft = EventDraft(
        event_type="model.exchange.finished",
        operation_id=new_analytics_id(),
        parent_operation_id=new_analytics_id(),
        turn_id=new_analytics_id(),
        payload={
            "outcome": "ok",
            "duration_ms": 1234,
            "usage": {"input_tokens": 10, "output_tokens": 20, "cache_read_tokens": 0},
            "response_id": "resp_" + "x" * 40,
        },
        measurements={"/payload/usage": measurement(MeasurementSource.PROVIDER)},
    )
    line = encode_event_line(_addressed(draft))
    assert len(line) < LINE_BUDGET_BYTES // 4
    assert json.loads(line)["payload"]["duration_ms"] == 1234
