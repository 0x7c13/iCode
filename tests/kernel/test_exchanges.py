# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Differential grammar corpus for the exchange iterator.

Every corpus case runs three ways — the production iterator over live
kernel objects, the production iterator over the same transcript's
``to_dict`` serialization, and the independent reference state machine
(``tests.support.exchange_reference``, which imports nothing from the
production module) — and all three must report identical exchange
boundaries and identical canonical pairing. A corpus case therefore cannot
pass because production and reference share a bug, and a serialization
drift between the two accessors is caught by construction.

The differential runs under the canonical policy row (all types,
informational calls included, positional falsy queues, stringify).
Consumer-specific policy rows are exercised by the targeted unit tests
below the corpus, and the type-set parity pins hold the test-owned literal
sets to the production constants.
"""

from __future__ import annotations

import pytest

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.tool_result_metadata import TOOL_ERROR_KIND_METADATA_KEY
from chrys.kernel import exchanges
from chrys.kernel.exchanges import (
    DictAccessor,
    EmptyIdPolicy,
    LiveAccessor,
    NoneIdPolicy,
    PairingPolicy,
    is_result_only_message,
    iter_exchanges,
    pair_results,
)
from chrys.kernel.types import Content, Message
from tests.support import transcript_invariants as oracle
from tests.support.exchange_reference import (
    REFERENCE_CALL_TYPES,
    REFERENCE_RESULT_TYPES,
    reference_exchange_trace,
)

# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def _a(*contents: Content) -> Message:
    return Message("assistant", list(contents))


def _tool(*contents: Content) -> Message:
    return Message("tool", list(contents))


def _user(*contents: Content) -> Message:
    return Message("user", list(contents))


def _marker(*contents: Content) -> Message:
    message = Message("assistant", list(contents))
    message.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    return message


def _text(text: str = "note") -> Content:
    return Content.from_text(text)


def _call(call_id: str, *, informational: bool = False) -> Content:
    return Content.from_function_call(call_id, "echo", arguments={"text": call_id}, informational_only=informational)


def _result(call_id: str) -> Content:
    return Content.from_function_result(call_id, result="ok")


def _failure_result(call_id: str | None, kind: str) -> Content:
    content = _result(call_id or "x")
    content.call_id = call_id
    content.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] = kind
    return content


def _mcp_call(call_id: str) -> Content:
    return Content.from_mcp_server_tool_call(call_id, "search")


def _mcp_result(call_id: str) -> Content:
    return Content.from_mcp_server_tool_result(call_id, output="ok")


def _hosted_call(call_id: str | None) -> Content:
    return Content.from_hosted_tool_call(call_id, tool_name="tool_search", arguments={"query": "docs"})


def _hosted_result(call_id: str | None) -> Content:
    return Content.from_hosted_tool_result(call_id, tool_name="tool_search", result={"tools": ["search_docs"]})


def _image_call(image_id: str) -> Content:
    return Content.from_image_generation_tool_call(image_id=image_id)


def _image_result(image_id: str) -> Content:
    return Content.from_image_generation_tool_result(image_id=image_id)


def _raw_call(raw_id: object) -> Content:
    # Restored sessions rebuild live contents from serialized dicts without
    # validation, so any raw id shape can reach live consumers; mutating the
    # attribute reproduces that path without a serialization detour.
    content = _call("placeholder")
    content.call_id = raw_id
    return content


def _raw_result(raw_id: object) -> Content:
    content = _result("placeholder")
    content.call_id = raw_id
    return content


# ---------------------------------------------------------------------------
# Differential harness
# ---------------------------------------------------------------------------

Coordinate = tuple[int, int]
ExchangeView = tuple[tuple[int, ...], tuple[int, ...], dict[Coordinate, Coordinate | None], list[Coordinate]]


def _canonical_policy(accessor: exchanges.MessageAccessor) -> PairingPolicy:
    return PairingPolicy(
        call_types=accessor.call_types(),
        include_informational_calls=True,
        result_types=accessor.result_types(),
        none_id=NoneIdPolicy.POSITIONAL,
        empty_id=EmptyIdPolicy.POSITIONAL,
        malformed_id="stringify",
    )


def _production_view(messages: list, accessor: exchanges.MessageAccessor) -> list[ExchangeView]:
    views: list[ExchangeView] = []
    for exchange in iter_exchanges(messages, accessor):
        pairing = pair_results(messages, exchange, accessor, _canonical_policy(accessor))
        mapping: dict[Coordinate, Coordinate | None] = {}
        for assignments in pairing.truthy_assignments.values():
            for call, result in assignments:
                coordinate = None if result is None else (result.message_index, result.content_index)
                mapping[(call.message_index, call.content_index)] = coordinate
        for assignments in pairing.falsy_assignments.values():
            for call, result in assignments:
                coordinate = None if result is None else (result.message_index, result.content_index)
                mapping[(call.message_index, call.content_index)] = coordinate
        orphans = sorted(
            [(o.message_index, o.content_index) for group in pairing.unconsumed_results.values() for o in group]
            + [(o.message_index, o.content_index) for group in pairing.falsy_unconsumed_results.values() for o in group]
        )
        views.append((exchange.response_indices, exchange.output_indices, mapping, orphans))
    return views


def _reference_view(messages: list) -> list[ExchangeView]:
    views: list[ExchangeView] = []
    for trace in reference_exchange_trace(messages):
        mapping = dict(trace.assignments)
        views.append((trace.response_indices, trace.output_indices, mapping, sorted(trace.orphan_results)))
    return views


def _assert_three_way_agreement(messages: list[Message]) -> None:
    live_view = _production_view(messages, LiveAccessor())
    assert live_view == _reference_view(messages)
    serialized = [message.to_dict() for message in messages]
    dict_view = _production_view(serialized, DictAccessor())
    assert dict_view == _reference_view(serialized)
    assert dict_view == live_view


# ---------------------------------------------------------------------------
# Grammar corpus
# ---------------------------------------------------------------------------


def _case_single_call_answered() -> list[Message]:
    return [_user(_text("go")), _a(_call("c1")), _tool(_result("c1"))]


def _case_sibling_run_shares_one_output_block() -> list[Message]:
    return [_a(_call("c1")), _a(_call("c2")), _tool(_result("c1"), _result("c2"))]


def _case_text_riders_stay_in_the_response_run() -> list[Message]:
    return [_a(_text()), _a(_call("c1")), _a(_text()), _tool(_result("c1"))]


def _case_marker_between_siblings_splits_the_run() -> list[Message]:
    return [_a(_call("c1")), _marker(), _a(_call("c2")), _tool(_result("c2"))]


def _case_marker_carried_result_leaves_call_unanswered() -> list[Message]:
    # The output walk stops AT the marker, so the result it carries is never
    # scanned: the call stays unanswered instead of pairing across a
    # compaction boundary.
    return [_user(_text("go")), _a(_call("c1")), _marker(_result("c1"))]


def _case_marker_after_output_closes_the_exchange() -> list[Message]:
    return [_a(_call("c1")), _tool(_result("c1")), _marker(), _tool(_result("c1"))]


def _case_result_only_assistant_continues_output() -> list[Message]:
    return [_a(_mcp_call("m1")), _a(_mcp_result("m1"))]


def _case_tool_and_result_only_assistant_share_the_output_run() -> list[Message]:
    return [_a(_call("c1"), _mcp_call("m1")), _tool(_result("c1")), _a(_mcp_result("m1"))]


def _case_call_after_output_starts_a_new_exchange() -> list[Message]:
    return [_a(_call("c1")), _tool(_result("c1")), _a(_call("c2")), _tool(_result("c2"))]


def _case_embedded_same_message_pair() -> list[Message]:
    return [_a(_mcp_call("m1"), _mcp_result("m1"))]


def _case_embedded_reversed_within_message_still_pairs() -> list[Message]:
    return [_a(_mcp_result("m1"), _mcp_call("m1"))]


def _case_embedded_forward_reference_is_an_orphan() -> list[Message]:
    # Message 0's embedded result consumes from registrations SO FAR, so it
    # cannot pair with the call sibling 1 introduces.
    return [_a(_call("c1"), _mcp_result("m2")), _a(_mcp_call("m2")), _tool(_result("c1"))]


def _case_user_role_result_stops_the_output_run() -> list[Message]:
    return [_a(_call("c1")), _user(_result("c1"))]


def _case_duplicate_id_sibling_calls_share_one_cursor() -> list[Message]:
    return [_a(_call("c1")), _a(_call("c1")), _tool(_result("c1"))]


def _case_second_same_id_result_is_an_orphan() -> list[Message]:
    return [_a(_call("c1")), _tool(_result("c1"), _result("c1"))]


def _case_duplicate_hosted_ids_share_one_cursor() -> list[Message]:
    return [_a(_hosted_call("h1"), _hosted_call("h1")), _a(_hosted_result("h1"))]


def _case_missing_hosted_ids_pair_positionally() -> list[Message]:
    return [_a(_hosted_call(None)), _a(_hosted_result(None))]


def _case_image_and_call_id_namespaces_never_collide() -> list[Message]:
    return [_a(_call("X"), _image_call("X")), _tool(_result("X")), _a(_image_result("X"))]


def _case_none_id_pair_is_positional() -> list[Message]:
    return [_a(_raw_call(None)), _tool(_raw_result(None))]


def _case_empty_id_pair_is_positional() -> list[Message]:
    return [_a(_raw_call("")), _tool(_raw_result(""))]


def _case_none_and_empty_queues_stay_separate() -> list[Message]:
    return [_a(_raw_call(None), _raw_call("")), _tool(_raw_result(""), _raw_result(None))]


def _case_m1_int_id_collides_with_its_string() -> list[Message]:
    return [_a(_raw_call(7)), _tool(_raw_result("7"))]


def _case_m1_nan_ids_stringify_and_pair() -> list[Message]:
    return [_a(_raw_call(float("nan"))), _tool(_raw_result(float("nan")))]


def _case_m2_falsy_non_string_rides_the_empty_stream() -> list[Message]:
    return [_a(_raw_call(0)), _tool(_raw_result(False))]


def _case_m3_python_equal_ids_split_by_string_form() -> list[Message]:
    return [_a(_raw_call(1), _raw_call(True)), _tool(_raw_result(1.0), _raw_result(1), _raw_result(True))]


def _case_m4_truthy_unhashable_id_is_processed() -> list[Message]:
    return [_a(_raw_call(["x"])), _tool(_raw_result(["x"]))]


def _case_m5_falsy_unhashable_id_rides_the_empty_stream() -> list[Message]:
    return [_a(_raw_call([])), _tool(_raw_result({}))]


def _case_informational_call_participates_under_canonical_row() -> list[Message]:
    return [_a(_call("c1", informational=True)), _tool(_result("c1"))]


def _case_assistant_run_without_a_call_forms_no_exchange() -> list[Message]:
    return [_user(_text("hi")), _a(_text()), _a(_text()), _user(_text("ok"))]


def _case_orphan_leading_tool_output_belongs_to_no_exchange() -> list[Message]:
    return [_tool(_result("c9")), _a(_call("c1")), _tool(_result("c1"))]


def _case_empty_contents_assistant_rides_the_run() -> list[Message]:
    return [_a(_call("c1")), _a(), _tool(_result("c1"))]


def _case_marker_at_transcript_start_is_skipped() -> list[Message]:
    return [_marker(), _a(_call("c1")), _tool(_result("c1"))]


_LIVE_CORPUS = [
    _case_single_call_answered,
    _case_sibling_run_shares_one_output_block,
    _case_text_riders_stay_in_the_response_run,
    _case_marker_between_siblings_splits_the_run,
    _case_marker_carried_result_leaves_call_unanswered,
    _case_marker_after_output_closes_the_exchange,
    _case_result_only_assistant_continues_output,
    _case_tool_and_result_only_assistant_share_the_output_run,
    _case_call_after_output_starts_a_new_exchange,
    _case_embedded_same_message_pair,
    _case_embedded_reversed_within_message_still_pairs,
    _case_embedded_forward_reference_is_an_orphan,
    _case_user_role_result_stops_the_output_run,
    _case_duplicate_id_sibling_calls_share_one_cursor,
    _case_second_same_id_result_is_an_orphan,
    _case_duplicate_hosted_ids_share_one_cursor,
    _case_missing_hosted_ids_pair_positionally,
    _case_image_and_call_id_namespaces_never_collide,
    _case_none_id_pair_is_positional,
    _case_empty_id_pair_is_positional,
    _case_none_and_empty_queues_stay_separate,
    _case_m1_int_id_collides_with_its_string,
    _case_m1_nan_ids_stringify_and_pair,
    _case_m2_falsy_non_string_rides_the_empty_stream,
    _case_m3_python_equal_ids_split_by_string_form,
    _case_m4_truthy_unhashable_id_is_processed,
    _case_m5_falsy_unhashable_id_rides_the_empty_stream,
    _case_informational_call_participates_under_canonical_row,
    _case_assistant_run_without_a_call_forms_no_exchange,
    _case_orphan_leading_tool_output_belongs_to_no_exchange,
    _case_empty_contents_assistant_rides_the_run,
    _case_marker_at_transcript_start_is_skipped,
]


class TestGrammarCorpusDifferential:
    @pytest.mark.parametrize("build", _LIVE_CORPUS, ids=lambda build: build.__name__.removeprefix("_case_"))
    def test_live_dict_and_reference_views_agree(self, build) -> None:
        _assert_three_way_agreement(build())

    def test_falsy_valued_marker_key_is_still_a_boundary(self) -> None:
        # Marker detection is KEY presence: the value is the marker kind,
        # and a corrupted or falsy kind must not un-mark the message and
        # let results pair across the boundary.
        stamped = _a(_text())
        stamped.additional_properties[HistoryMarkerKind.KEY] = ""
        messages = [_a(_call("c1")), stamped, _tool(_result("c1"))]
        accessor = LiveAccessor()
        exchange = next(iter(iter_exchanges(messages, accessor)))
        assert exchange.response_indices == (0,)
        assert exchange.output_indices == ()
        pairing = pair_results(messages, exchange, accessor, _canonical_policy(accessor))
        assert pairing.answered_keys == set()

    def test_marker_carried_result_reports_the_call_unanswered(self) -> None:
        # The exact declared delta the trim migration relies on: today's
        # helper collects the marker-carried result into its answered set,
        # the grammar never scans past the marker.
        messages = _case_marker_carried_result_leaves_call_unanswered()
        accessor = LiveAccessor()
        (exchange,) = iter_exchanges(messages, accessor)
        assert exchange.output_indices == ()
        pairing = pair_results(messages, exchange, accessor, _canonical_policy(accessor))
        assert pairing.answered_keys == set()
        assert pairing.truthy_assignments[("call_id", "c1")][0][1] is None

    def test_hosted_pair_serializes_deserializes_and_pairs(self) -> None:
        messages = [
            _a(
                Content.from_hosted_tool_call(
                    "hosted-1",
                    tool_name="tool_search",
                    hosted_family="tool_discovery",
                    hosted_provider="openai",
                    provider_item_type="tool_search_call",
                    provider_item_id="item-1",
                    provider_phase="start",
                    provider_status="in_progress",
                    retry_safety="read_only",
                )
            ),
            _a(
                Content.from_hosted_tool_result(
                    "hosted-1",
                    tool_name="tool_search",
                    result={"tools": ["search_docs"]},
                    hosted_family="tool_discovery",
                    hosted_provider="openai",
                    provider_item_type="tool_search_call",
                    provider_item_id="item-1",
                    provider_phase="terminal",
                    provider_status="completed",
                    retry_safety="read_only",
                )
            ),
        ]
        restored = [Message.from_dict(message.to_dict()) for message in messages]

        _assert_three_way_agreement(restored)
        assert _production_view(restored, LiveAccessor()) == [((0,), (1,), {(0, 0): (1, 0)}, [])]

    def test_duplicate_and_missing_hosted_ids_follow_canonical_cursors(self) -> None:
        duplicate_view = _production_view(_case_duplicate_hosted_ids_share_one_cursor(), LiveAccessor())
        missing_view = _production_view(_case_missing_hosted_ids_pair_positionally(), LiveAccessor())

        assert duplicate_view == [((0,), (1,), {(0, 0): (1, 0), (0, 1): None}, [])]
        assert missing_view == [((0,), (1,), {(0, 0): (1, 0)}, [])]


class TestDictOnlyShapes:
    def test_legacy_call_opens_an_exchange(self) -> None:
        serialized = [
            {"role": "assistant", "contents": [{"type": "legacy", "call_id": "L1"}]},
            {"role": "tool", "contents": [{"type": "function_result", "call_id": "L1"}]},
        ]
        dict_view = _production_view(serialized, DictAccessor())
        assert dict_view == _reference_view(serialized)
        assert dict_view == [((0,), (1,), {(0, 0): (1, 0)}, [])]

    def test_legacy_type_is_not_a_call_on_live_objects(self) -> None:
        assert "legacy" not in LiveAccessor().call_types()
        assert "legacy" in DictAccessor().call_types()

    def test_garbage_shapes_fail_closed(self) -> None:
        serialized = [
            {"role": "assistant", "contents": ["junk", {"type": "function_call", "call_id": "c1"}, 42]},
            {"role": "tool", "contents": [{"type": "function_result", "call_id": "c1"}]},
            {"contents": None},
            {"role": "tool", "contents": {"not": "a list"}},
        ]
        dict_view = _production_view(serialized, DictAccessor())
        assert dict_view == _reference_view(serialized)
        assert dict_view == [((0,), (1,), {(0, 1): (1, 0)}, [])]

    def test_falsy_valued_marker_key_is_a_boundary_on_dicts(self) -> None:
        serialized = [
            {"role": "assistant", "contents": [{"type": "function_call", "call_id": "c1"}]},
            {"role": "assistant", "contents": ["pause"], "additional_properties": {HistoryMarkerKind.KEY: None}},
            {"role": "tool", "contents": [{"type": "function_result", "call_id": "c1"}]},
        ]
        dict_view = _production_view(serialized, DictAccessor())
        assert dict_view == _reference_view(serialized)
        assert dict_view[0][:2] == ((0,), ())

    def test_non_mapping_additional_properties_is_no_marker(self) -> None:
        serialized = [
            {
                "role": "assistant",
                "contents": [{"type": "function_call", "call_id": "c1"}],
                "additional_properties": "turn",
            },
            {"role": "tool", "contents": [{"type": "function_result", "call_id": "c1"}]},
        ]
        dict_view = _production_view(serialized, DictAccessor())
        assert dict_view == _reference_view(serialized)
        assert dict_view == [((0,), (1,), {(0, 0): (1, 0)}, [])]


class TestTypeSetParity:
    """The test-owned literal sets vs the production constants.

    The reference machine and the invariant oracle both carry literal
    copies so a production edit cannot silently retune them; these pins
    make any divergence a visible, two-sided change.
    """

    def test_reference_literals_match_production(self) -> None:
        assert REFERENCE_CALL_TYPES == exchanges.TOOL_CALL_CONTENT_TYPES
        assert REFERENCE_RESULT_TYPES == exchanges.TOOL_RESULT_CONTENT_TYPES

    def test_oracle_literals_match_production(self) -> None:
        assert oracle.TOOL_CALL_CONTENT_TYPES == exchanges.TOOL_CALL_CONTENT_TYPES
        assert oracle.TOOL_RESULT_CONTENT_TYPES == exchanges.TOOL_RESULT_CONTENT_TYPES

    def test_dict_accessor_adds_exactly_the_legacy_type(self) -> None:
        assert DictAccessor().call_types() == exchanges.TOOL_CALL_CONTENT_TYPES | {exchanges.LEGACY_CALL_CONTENT_TYPE}
        assert DictAccessor().result_types() == exchanges.TOOL_RESULT_CONTENT_TYPES

    def test_compaction_reexports_the_same_objects(self) -> None:
        from chrys.kernel import compaction

        assert compaction.TOOL_CALL_CONTENT_TYPES is exchanges.TOOL_CALL_CONTENT_TYPES
        assert compaction.TOOL_RESULT_CONTENT_TYPES is exchanges.TOOL_RESULT_CONTENT_TYPES


# ---------------------------------------------------------------------------
# Policy rows beyond the canonical one
# ---------------------------------------------------------------------------


def _policy(**overrides) -> PairingPolicy:
    settings = {
        "call_types": exchanges.TOOL_CALL_CONTENT_TYPES,
        "include_informational_calls": True,
        "result_types": exchanges.TOOL_RESULT_CONTENT_TYPES,
        "none_id": NoneIdPolicy.POSITIONAL,
        "empty_id": EmptyIdPolicy.POSITIONAL,
        "malformed_id": "stringify",
    }
    settings.update(overrides)
    return PairingPolicy(**settings)


def _pair(messages: list[Message], policy: PairingPolicy) -> exchanges.ExchangePairing:
    accessor = LiveAccessor()
    (exchange,) = iter_exchanges(messages, accessor)
    return pair_results(messages, exchange, accessor, policy)


class TestPolicyRows:
    def test_positional_failures_consumes_only_prevalidation_kinds(self) -> None:
        # An arbitrary id-less result must not eat the queue slot ahead of a
        # later qualifying failure result.
        messages = [
            _a(_raw_call(None)),
            _tool(_raw_result(None), _failure_result(None, "argument_parsing")),
        ]
        pairing = _pair(messages, _policy(none_id=NoneIdPolicy.POSITIONAL_FAILURES))
        (call, consumed) = pairing.falsy_assignments["none"][0]
        assert call.message_index == 0
        assert consumed is not None and (consumed.message_index, consumed.content_index) == (1, 1)
        assert [(o.message_index, o.content_index) for o in pairing.falsy_unconsumed_results["none"]] == [(1, 0)]

    def test_unpairable_occurrence_surfaces_without_pairing(self) -> None:
        messages = [_a(_raw_call(None)), _tool(_raw_result(None))]
        pairing = _pair(
            messages,
            _policy(none_id=NoneIdPolicy.UNPAIRABLE_OCCURRENCE, empty_id=EmptyIdPolicy.UNPAIRABLE_OCCURRENCE),
        )
        assert [(o.message_index, o.content_index) for o in pairing.unpairable_calls] == [(0, 0)]
        assert [(o.message_index, o.content_index) for o in pairing.unpairable_results] == [(1, 0)]
        assert pairing.falsy_assignments == {"none": [], "empty": []}

    def test_coerced_single_key_merges_empty_ids_into_the_none_queue(self) -> None:
        # The legacy conflation: None and "" shared one coerced bucket, so a
        # None-id result may answer an empty-id call through the merged queue.
        messages = [_a(_raw_call("")), _tool(_raw_result(None))]
        pairing = _pair(messages, _policy(empty_id=EmptyIdPolicy.COERCED_SINGLE_KEY))
        (call, consumed) = pairing.falsy_assignments["none"][0]
        assert (call.message_index, call.content_index) == (0, 0)
        assert consumed is not None and (consumed.message_index, consumed.content_index) == (1, 0)

    def test_ignore_skips_calls_and_results_symmetrically(self) -> None:
        messages = [_a(_raw_call(None), _call("k")), _tool(_raw_result(None), _result("k"))]
        pairing = _pair(messages, _policy(none_id=NoneIdPolicy.IGNORE))
        assert pairing.falsy_assignments == {"none": [], "empty": []}
        assert pairing.falsy_unconsumed_results == {"none": [], "empty": []}
        assert pairing.answered_keys == {("call_id", "k")}

    def test_treat_as_none_routes_malformed_ids_to_the_none_stream(self) -> None:
        messages = [_a(_raw_call(7)), _tool(_raw_result(7))]
        pairing = _pair(messages, _policy(malformed_id="treat_as_none"))
        assert pairing.truthy_assignments == {}
        (call, consumed) = pairing.falsy_assignments["none"][0]
        assert call.raw_id is None and consumed is not None

    def test_informational_calls_excluded_when_the_row_says_so(self) -> None:
        messages = [_a(_mcp_call("m1")), _a(_mcp_result("m1"))]
        pairing = _pair(messages, _policy(include_informational_calls=False))
        assert pairing.truthy_assignments == {}
        assert [(o.message_index, o.content_index) for o in pairing.unconsumed_results[("call_id", "m1")]] == [(1, 0)]

    def test_type_scope_filters_pairing_but_not_the_grammar(self) -> None:
        messages = [_a(_call("c1"), _image_call("i1")), _tool(_result("c1")), _a(_image_result("i1"))]
        accessor = LiveAccessor()
        (exchange,) = iter_exchanges(messages, accessor)
        assert exchange.output_indices == (1, 2)
        pairing = pair_results(
            messages,
            exchange,
            accessor,
            _policy(call_types=frozenset({"function_call"}), result_types=frozenset({"function_result"})),
        )
        assert set(pairing.truthy_assignments) == {("call_id", "c1")}
        assert pairing.unconsumed_results == {}


class TestPairingViews:
    def test_answered_keys_excludes_forward_orphans(self) -> None:
        messages = _case_embedded_forward_reference_is_an_orphan()
        pairing = _pair(messages, _policy())
        assert pairing.answered_keys == {("call_id", "c1")}
        assert set(pairing.result_keys) == {("call_id", "c1"), ("call_id", "m2")}
        assert [(o.message_index, o.content_index) for o in pairing.results_by_key[("call_id", "m2")]] == [(0, 1)]

    def test_eligible_results_excludes_the_forward_orphan(self) -> None:
        messages = _case_embedded_forward_reference_is_an_orphan()
        pairing = _pair(messages, _policy())
        assert ("call_id", "m2") not in pairing.eligible_results.truthy
        assert [(o.message_index, o.content_index) for o in pairing.eligible_results.truthy[("call_id", "c1")]] == [
            (2, 0)
        ]

    def test_eligible_falsy_is_stream_blind(self) -> None:
        # A prior falsy call in EITHER stream qualifies a falsy result: the
        # block-wide boolean consumers derive from this view ignores which
        # queue the call actually rode.
        messages = [_a(_raw_call(None)), _tool(_raw_result(""))]
        pairing = _pair(messages, _policy())
        assert [(o.message_index, o.content_index) for o in pairing.eligible_results.falsy["empty"]] == [(1, 0)]
        assert [(o.message_index, o.content_index) for o in pairing.falsy_unconsumed_results["empty"]] == [(1, 0)]

    def test_embedded_results_carry_the_structural_id_view(self) -> None:
        content = _mcp_result("x")
        content.call_id = 7
        messages = [_a(_call("c1"), content)]
        accessor = LiveAccessor()
        (exchange,) = iter_exchanges(messages, accessor)
        (embedded,) = exchange.embedded_results
        assert embedded.raw_id == "7"
        assert embedded.content_type == "mcp_server_tool_result"


class TestResultOnlyPredicate:
    def test_role_gates_and_content_shapes(self) -> None:
        accessor = LiveAccessor()
        assert is_result_only_message(_a(_mcp_result("m1")), accessor)
        assert is_result_only_message(_tool(_result("c1")), accessor) is False
        assert is_result_only_message(_user(_result("c1")), accessor) is False
        assert is_result_only_message(_a(_mcp_call("m1"), _mcp_result("m1")), accessor) is False
        assert is_result_only_message(_a(), accessor) is False
        assert is_result_only_message(_a(_text()), accessor) is False
