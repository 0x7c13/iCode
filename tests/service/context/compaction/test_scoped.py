# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the physically scoped Phase-4 timeline and completer slice."""

from __future__ import annotations

from copy import deepcopy
from io import BytesIO

from PIL import Image

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.kernel import Content, Message, annotate_message_groups, estimate_message_tokens
from chrys.service.context.compaction.scoped import (
    DEGRADED_SCOPED_PREAMBLE,
    SLICE_TRUNCATION_MARKER,
    ScopedGroup,
    build_scoped_group_timeline,
    clone_for_slice,
    prepare_scoped_slice,
)


class _LenTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text)


_tokenizer = _LenTokenizer()


def _user(text: str, *, properties: dict | None = None) -> Message:
    return Message("user", [text], additional_properties=properties)


def _call(call_id: str, name: str = "tool", *, properties: dict | None = None) -> Content:
    return Content.from_function_call(call_id, name, arguments={"value": name}, additional_properties=properties)


def _result(call_id: str, text: str) -> Content:
    return Content.from_function_result(call_id, result=text)


def _timeline(messages: list[Message], *, degraded: bool = False):  # type: ignore[no-untyped-def]
    annotate_message_groups(messages, force_reannotate=True)
    return build_scoped_group_timeline(messages, span_start=0, span_end=len(messages), degraded=degraded)


def _group_by_kind(timeline, kind: str):  # type: ignore[no-untyped-def]
    return [group for group in timeline.groups if group.kind == kind]


def _tool_pair(call_id: str, *, name: str = "tool", result: str = "ok") -> list[Message]:
    return [Message("assistant", [_call(call_id, name)]), Message("tool", [_result(call_id, result)])]


def test_strict_timeline_sanitizes_user_clone_and_keeps_previous_last_words() -> None:
    runtime = "<system-reminder>\n[Runtime Environment] cwd=/tmp\n</system-reminder>"
    previous = "<system-reminder>\n[LAST_WORDS] previous state\n</system-reminder>"
    opener = Message("user", ["do X", runtime, previous])
    injection = _user("also Y", properties={HistoryMarkerKind.INJECTED_KEY: True})
    nudge = _user("continue", properties={HistoryMarkerKind.CONTINUATION_KEY: True})
    messages = [opener, *_tool_pair("c1"), injection, nudge, Message("assistant", ["done"])]

    timeline = _timeline(messages)

    assert timeline.degraded_opener is False
    assert [group.kind for group in timeline.groups] == ["user", "tool_call", "user", "user", "assistant_text"]
    scoped_opener = timeline.groups[0].messages[0]
    assert scoped_opener is not opener
    assert [content.text for content in scoped_opener.contents] == ["do X", previous]
    assert [message.text for group in _group_by_kind(timeline, "user") for message in group.messages] == [
        "do X " + previous,
        "also Y",
        "continue",
    ]


def test_degraded_and_empty_sanitized_opener_use_synthetic_preamble() -> None:
    reminder_only = _user("<system-reminder>\n[Runtime Environment] chrome\n</system-reminder>")
    empty_text = _user("")
    strict = _timeline([reminder_only, *_tool_pair("c1")])
    empty = _timeline([empty_text, *_tool_pair("c-empty")])
    degraded = _timeline([*_tool_pair("c2"), _user("continue")], degraded=True)

    assert strict.degraded_opener is True
    assert strict.groups[0].messages[0].text == DEGRADED_SCOPED_PREAMBLE
    assert all(message is not reminder_only for group in strict.groups for message in group.messages)
    assert empty.groups[0].messages[0].text == DEGRADED_SCOPED_PREAMBLE
    assert degraded.groups[0].messages[0].text == DEGRADED_SCOPED_PREAMBLE
    assert degraded.groups[1].kind == "tool_call"


def test_structural_and_excluded_groups_do_not_enter_timeline() -> None:
    structural = Message("assistant", ["compressed"])
    structural.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.SUMMARY
    excluded = Message("assistant", ["hidden"])
    excluded.additional_properties["_excluded"] = True
    messages = [_user("do X"), structural, excluded, *_tool_pair("c1")]

    timeline = _timeline(messages)

    texts = [message.text for group in timeline.groups for message in group.messages]
    assert "compressed" not in texts
    assert "hidden" not in texts
    assert _group_by_kind(timeline, "tool_call")[0].integrity is True


def test_falsy_kind_marker_group_stays_out_of_timeline() -> None:
    # KEY presence makes the message structural chrome even when the kind
    # value is falsy — the grammar already bounds exchanges on presence.
    chrome = Message("assistant", ["chrome"])
    chrome.additional_properties[HistoryMarkerKind.KEY] = ""
    messages = [_user("do X"), chrome, *_tool_pair("c1")]

    timeline = _timeline(messages)

    texts = [message.text for group in timeline.groups for message in group.messages]
    assert "chrome" not in texts


def test_incomplete_and_malformed_tool_groups_demote_the_slice() -> None:
    cases = [
        [Message("assistant", [_call("")])],
        [Message("assistant", [_call("dup"), _call("dup")]), Message("tool", [_result("dup", "one")])],
        [Message("assistant", [_call("missing")])],
        [Message("tool", [_result("orphan", "extra")])],
        [
            Message("assistant", [_call("intervened")]),
            Message("assistant", [Content.from_text_reasoning(text="standalone reasoning")]),
            Message("tool", [_result("intervened", "late")]),
        ],
    ]
    for malformed in cases:
        timeline = _timeline([_user("do X"), *malformed])
        tool_group = _group_by_kind(timeline, "tool_call")[0]
        assert tool_group.integrity is False
        prepared = prepare_scoped_slice(timeline.groups, tokenizer=_tokenizer, slice_budget=100_000)
        assert prepared is None


def test_provider_hosted_call_and_result_in_one_message_demotes_the_slice() -> None:
    hosted = Message(
        "assistant",
        [
            Content.from_search_tool_call("search-1", tool_name="web_search", arguments={"query": "x"}),
            Content.from_search_tool_result(
                "search-1",
                tool_name="web_search",
                result={"answer": "critical hosted result"},
            ),
        ],
    )
    timeline = _timeline([_user("research this"), hosted])

    tool_group = _group_by_kind(timeline, "tool_call")[0]
    assert tool_group.integrity is False
    assert prepare_scoped_slice(timeline.groups, tokenizer=_tokenizer, slice_budget=100_000) is None


def test_parallel_calls_and_reused_ids_are_valid_group_locally() -> None:
    messages = [
        _user("do X"),
        Message("assistant", [_call("same", "first"), _call("other", "parallel")]),
        Message("tool", [_result("same", "a"), _result("other", "b")]),
        Message("assistant", [_call("same", "second")]),
        Message("tool", [_result("same", "c")]),
    ]
    timeline = _timeline(messages)
    tool_groups = _group_by_kind(timeline, "tool_call")
    assert len(tool_groups) == 2
    assert all(group.integrity for group in tool_groups)
    prepared = prepare_scoped_slice(timeline.groups, tokenizer=_tokenizer, slice_budget=100_000)
    assert prepared is not None
    assert sum(content.type == "function_call" for message in prepared.messages for content in message.contents) == 3


def test_clone_for_slice_preserves_provider_fields_and_deep_copies_annotations() -> None:
    raw = object()
    call = _call("c1", properties={"fc_id": "fc_provider", "nested": {"values": [1]}})
    call.raw_representation = raw
    message = Message(
        "assistant",
        [call],
        author_name="agent",
        message_id="message-1",
        additional_properties={
            "provider": {"nested": ["live"]},
            "_group": {"id": "g", "kind": "tool_call", "index": 1, "token_count": 999},
        },
        raw_representation=raw,
    )

    cloned = clone_for_slice(message)
    cloned.additional_properties["provider"]["nested"].append("clone")
    cloned.contents[0].additional_properties["nested"]["values"].append(2)

    assert cloned.role == message.role
    assert cloned.author_name == message.author_name
    assert cloned.message_id == message.message_id
    assert cloned.raw_representation is raw
    assert cloned.contents[0].raw_representation is raw
    assert cloned.contents[0].additional_properties["fc_id"] == "fc_provider"
    assert "token_count" not in cloned.additional_properties["_group"]
    assert message.additional_properties["_group"]["token_count"] == 999
    assert message.additional_properties["provider"] == {"nested": ["live"]}
    assert message.contents[0].additional_properties["nested"] == {"values": [1]}


def test_non_mutating_estimator_writes_neither_annotations_nor_image_dimensions() -> None:
    output = BytesIO()
    Image.new("RGB", (8, 8), color="red").save(output, format="PNG")
    image = Content.from_data(output.getvalue(), "image/png")
    message = Message("user", [image], additional_properties={"_group": {"token_count": 7}})
    before_message = deepcopy(message.additional_properties)
    before_content = deepcopy(image.additional_properties)

    assert estimate_message_tokens(message, _tokenizer) > 0
    assert message.additional_properties == before_message
    assert image.additional_properties == before_content


def test_result_clipping_changes_only_clone_and_preserves_fc_id() -> None:
    live_result = "start-" + "x" * 12_000 + "-finish"
    messages = [
        _user("do X"),
        Message("assistant", [_call("c1", properties={"fc_id": "fc_123"})]),
        Message("tool", [_result("c1", live_result)]),
    ]
    timeline = _timeline(messages)
    generous = prepare_scoped_slice(timeline.groups, tokenizer=_tokenizer, slice_budget=100_000)
    assert generous is not None
    clipped = prepare_scoped_slice(
        timeline.groups,
        tokenizer=_tokenizer,
        slice_budget=generous.estimated_tokens - 5_000,
    )

    assert clipped is not None
    clipped_call = next(
        content for message in clipped.messages for content in message.contents if content.type == "function_call"
    )
    clipped_result = next(
        content for message in clipped.messages for content in message.contents if content.type == "function_result"
    )
    assert clipped_call.additional_properties["fc_id"] == "fc_123"
    assert SLICE_TRUNCATION_MARKER in (clipped_result.result or "")
    assert "full content" not in SLICE_TRUNCATION_MARKER
    assert messages[2].contents[0].result == live_result


def test_many_results_reach_floor_then_assistant_caps_or_demotes() -> None:
    calls = [_call(f"c{index}") for index in range(5)]
    results = [_result(f"c{index}", str(index) * 8_000) for index in range(5)]
    messages = [
        _user("do X"),
        Message("assistant", calls),
        Message("tool", results),
        Message("assistant", ["agent " * 5_000]),
    ]
    timeline = _timeline(messages)
    full = prepare_scoped_slice(timeline.groups, tokenizer=_tokenizer, slice_budget=1_000_000)
    assert full is not None
    clipped = prepare_scoped_slice(timeline.groups, tokenizer=_tokenizer, slice_budget=22_000)
    assert clipped is not None
    assert clipped.clipped_results >= 5
    assert clipped.clipped_assistant_texts == 1
    assert SLICE_TRUNCATION_MARKER in clipped.messages[-1].text
    assert prepare_scoped_slice(timeline.groups, tokenizer=_tokenizer, slice_budget=500) is None


def test_runtime_shape_check_demotes_first_assistant_or_empty_message() -> None:
    invalid_first = [ScopedGroup("assistant", "assistant_text", (Message("assistant", ["work"]),), True)]
    empty = [
        ScopedGroup("user", "user", (_user("do X"),), True),
        ScopedGroup("assistant", "assistant_text", (Message("assistant", []),), False),
    ]
    assert prepare_scoped_slice(invalid_first, tokenizer=_tokenizer, slice_budget=10_000) is None
    assert prepare_scoped_slice(empty, tokenizer=_tokenizer, slice_budget=10_000) is None


def test_marker_carried_result_leaves_call_group_in_timeline_as_incomplete() -> None:
    """A history marker riding the call's result is chrome between exchanges,
    not a member of the call's group: the group must not turn structural (and
    silently vanish from the timeline, retained untouched) — it enters the
    timeline call-only with integrity False and demotes the slice to the
    reconstruction fallback, where Phase 4 may spill or exclude it."""
    marker_result = Message("assistant", [_result("c1", "late")])
    marker_result.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    messages = [_user("do X"), Message("assistant", [_call("c1")]), marker_result]

    timeline = _timeline(messages)

    tool_groups = _group_by_kind(timeline, "tool_call")
    assert len(tool_groups) == 1
    assert tool_groups[0].integrity is False
    assert marker_result not in tool_groups[0].messages
    assert prepare_scoped_slice(timeline.groups, tokenizer=_tokenizer, slice_budget=100_000) is None
