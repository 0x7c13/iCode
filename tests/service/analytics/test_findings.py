# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Positive, negative, and identity tests for the frozen P1 finding rules."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chrys.foundation.trajectory.envelope import Actor, SegmentedField, measurement
from chrys.foundation.trajectory.event_types import EventType
from chrys.service.analytics import analyze_trajectory
from chrys.service.analytics.findings import ContextCarryingLoad, RetriedExchange, SequenceSpan, evaluate_findings
from chrys.service.analytics.model import (
    ActionClass,
    ActionFunnel,
    ActionOperation,
    ChangeVerification,
    ChangeVerificationRow,
    ChangeVerificationState,
    FindingRow,
    Metric,
    Precision,
    TimeSlice,
    TurnAnalysis,
    TurnAttemptRef,
    ValidationMetrics,
    WallBucket,
    least_precision,
)
from tests.service.analytics._events import NS, EventLog, caused_by


def test_unverified_change_rule_has_positive_and_negative_paths() -> None:
    edit = _action(1, classification=ActionClass.EDIT)
    verify = _action(2, classification=ActionClass.VERIFY)

    assert "unverified-change" in _rule_ids(_evaluate(actions=(edit,)))
    assert "unverified-change" not in _rule_ids(_evaluate(actions=(edit, verify)))


def test_unverified_change_rule_requires_the_verify_to_start_after_the_edit_terminal() -> None:
    edit = _action(1, classification=ActionClass.EDIT, end_sequence=4)
    overlapped_verify = _action(2, classification=ActionClass.VERIFY)
    ordered_verify = _action(5, classification=ActionClass.VERIFY)

    assert "unverified-change" in _rule_ids(_evaluate(actions=(edit, overlapped_verify)))
    assert "unverified-change" not in _rule_ids(_evaluate(actions=(edit, ordered_verify)))


def test_unverified_change_details_name_the_verifications_seen() -> None:
    verify = replace(_action(1, classification=ActionClass.VERIFY), turn_number=3)
    edit = _action(2, classification=ActionClass.EDIT)

    failed_verify = replace(verify, outcome="failed")
    unreadable_shell = replace(
        _action(3, classification=ActionClass.OTHER, tool_kind="shell"), classification_precision=Precision.UNRESOLVED
    )
    earlier_unreadable_shell = replace(unreadable_shell, start_sequence=1)
    failed_unreadable_shell = replace(unreadable_shell, outcome="failed")
    long_edit = replace(edit, end_sequence=4)
    open_edit = replace(edit, end_sequence=None)

    never_verified = next(f for f in _evaluate(actions=(edit,)) if f.rule_id == "unverified-change")
    edited_after = next(f for f in _evaluate(actions=(verify, edit)) if f.rule_id == "unverified-change")
    never_passed = next(f for f in _evaluate(actions=(edit, failed_verify)) if f.rule_id == "unverified-change")
    unknown = next(f for f in _evaluate(actions=(edit, unreadable_shell)) if f.rule_id == "unverified-change")
    unknown_before = next(
        f for f in _evaluate(actions=(earlier_unreadable_shell, edit)) if f.rule_id == "unverified-change"
    )
    unknown_during = next(
        f for f in _evaluate(actions=(long_edit, unreadable_shell)) if f.rule_id == "unverified-change"
    )
    unknown_failed = next(
        f for f in _evaluate(actions=(edit, failed_unreadable_shell)) if f.rule_id == "unverified-change"
    )
    unknown_after_open = next(
        f for f in _evaluate(actions=(open_edit, unreadable_shell)) if f.rule_id == "unverified-change"
    )

    assert never_verified.detail_args == (
        ("count", 1),
        ("verifies", 0),
        ("verify_attempts", 0),
        ("unclassified_shell", 0),
    )
    assert edited_after.detail_args == (
        ("count", 1),
        ("verifies", 1),
        ("verify_attempts", 1),
        ("unclassified_shell", 0),
        ("last_verify_turn", 3),
    )
    # A test that ran and failed is a recognized verification, not a missing one.
    assert never_passed.detail_args == (
        ("count", 1),
        ("verifies", 0),
        ("verify_attempts", 1),
        ("unclassified_shell", 0),
    )
    # A later command whose text is unavailable may have been the test; one
    # that started before the edit ended could not test it.
    assert ("unclassified_shell", 1) in unknown.detail_args
    assert ("unclassified_shell", 0) in unknown_before.detail_args
    assert ("unclassified_shell", 0) in unknown_during.detail_args
    # One that failed was no passing test either.
    assert ("unclassified_shell", 0) in unknown_failed.detail_args
    # An edit in a turn still running, or cut off by a crash, has no end yet:
    # a command that started after it did may have tested it.
    assert ("unclassified_shell", 1) in unknown_after_open.detail_args


@pytest.mark.parametrize(
    ("outcome", "reported"),
    [("success", True), ("interrupted", True), ("failed", False), ("rejected", False), ("invalid_arguments", False)],
)
def test_an_edit_call_that_failed_is_not_an_unverified_change(outcome: str, reported: bool) -> None:
    # A failed or refused edit changed nothing; an interrupted one may have.
    verify = _action(1, classification=ActionClass.VERIFY)
    edit = _action(2, classification=ActionClass.EDIT, outcome=outcome)

    assert ("unverified-change" in _rule_ids(_evaluate(actions=(verify, edit)))) is reported


@pytest.mark.parametrize(
    ("states", "net_zero_precision", "change_fields", "reported"),
    [
        ((ChangeVerificationState.VERIFIED, ChangeVerificationState.NET_ZERO), Precision.EXACT, {}, False),
        ((ChangeVerificationState.NET_ZERO, ChangeVerificationState.AFTER_VERIFY), Precision.EXACT, {}, True),
        ((ChangeVerificationState.VERIFIED,), Precision.EXACT, {}, True),
        # A net-zero row whose contents could not be compared proves nothing.
        ((ChangeVerificationState.VERIFIED, ChangeVerificationState.NET_ZERO), Precision.UNRESOLVED, {}, True),
        # Without complete per-file detail the rows cannot vouch for every edit.
        (
            (ChangeVerificationState.VERIFIED, ChangeVerificationState.NET_ZERO),
            Precision.EXACT,
            {"detail_available": False},
            True,
        ),
        (
            (ChangeVerificationState.VERIFIED, ChangeVerificationState.NET_ZERO),
            Precision.EXACT,
            {"detection_truncated": True},
            True,
        ),
    ],
)
def test_unverified_change_is_not_reported_when_the_later_edits_cancelled_out(
    states: tuple[ChangeVerificationState, ...],
    net_zero_precision: Precision,
    change_fields: dict[str, bool],
    reported: bool,
) -> None:
    verify = _action(1, classification=ActionClass.VERIFY)
    edit = _action(2, classification=ActionClass.EDIT)
    rows = tuple(
        ChangeVerificationRow(
            f"file-{index}.py",
            state,
            1,
            net_zero_precision if state is ChangeVerificationState.NET_ZERO else Precision.EXACT,
        )
        for index, state in enumerate(states)
    )
    change = replace(_change(), rows=rows, **change_fields)

    assert ("unverified-change" in _rule_ids(_evaluate(actions=(verify, edit), change=change))) is reported


@pytest.mark.parametrize(("saved_through", "reported"), [(3, False), (2, True), (None, True)])
def test_unverified_change_stays_reported_for_an_edit_the_saved_detail_does_not_cover(
    saved_through: int | None, reported: bool
) -> None:
    # A turn's per-file detail is saved after the turn ends, and its mutation
    # summary follows the save; until then, rows from earlier turns say
    # nothing about the edit.
    verify = _action(1, classification=ActionClass.VERIFY)
    edit = _action(2, classification=ActionClass.EDIT)
    change = replace(
        _change(),
        rows=(
            ChangeVerificationRow("a.py", ChangeVerificationState.VERIFIED, 1, Precision.EXACT),
            ChangeVerificationRow("b.py", ChangeVerificationState.NET_ZERO, 1, Precision.EXACT),
        ),
        saved_through_sequence=saved_through,
    )

    assert ("unverified-change" in _rule_ids(_evaluate(actions=(verify, edit), change=change))) is reported


def test_repeated_read_rule_reports_an_identical_read_with_nothing_changed_between() -> None:
    first = _action(1, fingerprint="same", tool_name="read_file", tool_kind="filesystem.read")
    repeated = replace(
        _action(2, fingerprint="same", tool_name="read_file", tool_kind="filesystem.read"),
        payload_token_estimate=40,
    )

    findings = [finding for finding in _evaluate(actions=(first, repeated)) if finding.rule_id == "repeated-read"]

    assert len(findings) == 1
    assert findings[0].operation_id == repeated.operation_id
    assert findings[0].detail_args == (("count", 1), ("duration_ms", 0), ("result_tokens", 40))


@pytest.mark.parametrize(
    "between_kind",
    ["filesystem.write", "shell", "sleep", "sub_agent", "ask_user", "mcp"],
)
def test_repeated_read_rule_skips_reads_after_work_that_may_change_state(between_kind: str) -> None:
    first = _action(1, fingerprint="same", tool_name="read_file", tool_kind="filesystem.read")
    between = _action(2, tool_kind=between_kind)
    repeated = _action(3, fingerprint="same", tool_name="read_file", tool_kind="filesystem.read")

    assert "repeated-read" not in _rule_ids(_evaluate(actions=(first, between, repeated)))


def test_repeated_read_rule_ignores_reads_and_todos_between() -> None:
    first = _action(1, fingerprint="same", tool_name="grep")
    other_read = _action(2, tool_kind="filesystem.read")
    todo = _action(3, tool_kind="todo")
    repeated = _action(4, fingerprint="same", tool_name="grep")

    assert "repeated-read" in _rule_ids(_evaluate(actions=(first, other_read, todo, repeated)))


def test_repeated_read_rule_counts_a_concurrent_change_overlapping_the_gap() -> None:
    # A shell command another agent started before the first read finished,
    # and which ended after it, may have changed what the repeat reads.
    overlapping_shell = _action(1, tool_kind="shell", end_sequence=5)
    first = _action(2, fingerprint="same", tool_name="grep", end_sequence=3)
    repeated = _action(6, fingerprint="same", tool_name="grep")
    # One that ended before the first read finished cannot have.
    earlier_shell = _action(1, tool_kind="shell", end_sequence=2)

    assert "repeated-read" not in _rule_ids(_evaluate(actions=(overlapping_shell, first, repeated)))
    assert "repeated-read" in _rule_ids(_evaluate(actions=(earlier_shell, first, repeated)))


def test_repeated_read_rule_counts_a_change_that_ran_inside_the_first_read() -> None:
    # The first read may have run before or after that change.
    first = _action(2, fingerprint="same", tool_name="grep", end_sequence=10)
    shell = _action(3, tool_kind="shell", end_sequence=5)
    repeated = _action(11, fingerprint="same", tool_name="grep")

    assert "repeated-read" not in _rule_ids(_evaluate(actions=(first, shell, repeated)))


@pytest.mark.parametrize(
    ("task", "reported"),
    [
        (SequenceSpan(1, None), False),  # still running: its log, or files, may have changed
        (SequenceSpan(1, 5), False),  # finished between the reads
        (SequenceSpan(1, 2), True),  # finished before the first read
    ],
)
def test_repeated_read_rule_counts_a_running_background_task_as_a_change(task: SequenceSpan, reported: bool) -> None:
    first = _action(3, fingerprint="same", tool_name="read_file", tool_kind="filesystem.read")
    repeated = _action(6, fingerprint="same", tool_name="read_file", tool_kind="filesystem.read")

    findings = _evaluate(actions=(first, repeated), outside_changes=(task,))

    assert ("repeated-read" in _rule_ids(findings)) is reported


@pytest.mark.parametrize(("open_call_turn", "reported"), [("turn", False), ("earlier", True)])
def test_a_call_left_without_an_end_hides_repeats_only_in_its_own_turn(open_call_turn: str, reported: bool) -> None:
    # A crash leaves a turn's calls without an end; turns never overlap, so
    # such a call cannot have changed what a later turn read.
    open_call = replace(_action(1, tool_name="run_shell", tool_kind="shell"), turn_id=open_call_turn, end_sequence=None)
    first = _action(3, fingerprint="same", tool_name="read_file", tool_kind="filesystem.read")
    repeated = _action(6, fingerprint="same", tool_name="read_file", tool_kind="filesystem.read")

    findings = _evaluate(actions=(open_call, first, repeated))

    assert ("repeated-read" in _rule_ids(findings)) is reported


def test_repeated_read_rule_reports_repeats_inside_a_sub_agent() -> None:
    # The call that started the sub-agent spans its reads; it does not count
    # against them, while another sub-agent running beside them does.
    parent_call = _action(1, tool_kind="sub_agent", end_sequence=20)
    sibling_call = _action(2, tool_kind="sub_agent", end_sequence=19)
    first, repeated = (
        replace(
            _action(index, fingerprint="same", tool_name="read_file", tool_kind="filesystem.read"),
            actor_id="child",
            enclosing_operation_ids=(parent_call.operation_id,),
        )
        for index in (10, 12)
    )

    assert "repeated-read" in _rule_ids(_evaluate(actions=(parent_call, first, repeated)))
    assert "repeated-read" not in _rule_ids(_evaluate(actions=(parent_call, sibling_call, first, repeated)))


def test_repeated_read_rule_needs_same_agent_turn_tool_arguments_and_successful_calls() -> None:
    first = _action(1, fingerprint="same", tool_name="grep")
    repeat = _action(2, fingerprint="same", tool_name="grep")

    assert "repeated-read" in _rule_ids(_evaluate(actions=(first, repeat)))
    assert "repeated-read" not in _rule_ids(_evaluate(actions=(first, replace(repeat, argument_fingerprint="other"))))
    assert "repeated-read" not in _rule_ids(_evaluate(actions=(first, replace(repeat, tool_name="glob"))))
    assert "repeated-read" not in _rule_ids(_evaluate(actions=(first, replace(repeat, tool_name=None))))
    assert "repeated-read" not in _rule_ids(_evaluate(actions=(first, replace(repeat, actor_id="sub-agent"))))
    assert "repeated-read" not in _rule_ids(_evaluate(actions=(first, replace(repeat, turn_id="other-turn"))))
    assert "repeated-read" not in _rule_ids(_evaluate(actions=(replace(first, outcome="failed"), repeat)))
    # A declined or failed repeat fetched nothing, and the next one still
    # repeats the successful first call.
    assert "repeated-read" not in _rule_ids(_evaluate(actions=(first, replace(repeat, outcome="rejected"))))
    assert "repeated-read" not in _rule_ids(_evaluate(actions=(first, replace(repeat, outcome="errored"))))
    third = _action(3, fingerprint="same", tool_name="grep")
    assert "repeated-read" in _rule_ids(_evaluate(actions=(first, replace(repeat, outcome="errored"), third)))
    # Shell commands are not read-only calls, even when repeated verbatim.
    shell_first = _action(1, fingerprint="same", tool_name="shell", tool_kind="shell")
    shell_repeat = _action(2, fingerprint="same", tool_name="shell", tool_kind="shell")
    assert "repeated-read" not in _rule_ids(_evaluate(actions=(shell_first, shell_repeat)))


def test_repeated_read_evidence_keys_distinguish_tools_without_call_item_ids() -> None:
    actions = tuple(
        replace(_action(index, fingerprint="same", tool_name=tool_name), call_item_id=None)
        for index, tool_name in enumerate(("first", "first", "second", "second"), start=1)
    )

    repeated = [finding for finding in _evaluate(actions=actions) if finding.rule_id == "repeated-read"]

    assert len(repeated) == 2
    assert len({finding.evidence_key for finding in repeated}) == 2


def test_failed_attempt_critical_path_rule_has_positive_and_negative_paths() -> None:
    failed = _action(1, outcome="errored")
    dominating = _turn(critical={failed.operation_id: 30})
    minor = _turn(critical={failed.operation_id: 24})

    assert "failed-attempt-critical-path" in _rule_ids(_evaluate(actions=(failed,), turn=dominating))
    assert "failed-attempt-critical-path" not in _rule_ids(_evaluate(actions=(failed,), turn=minor))


@pytest.mark.parametrize(("retry_ns", "reported"), [(0, True), (900, False)])
def test_failed_attempt_share_is_measured_against_every_attempt_of_its_turn(retry_ns: int, reported: bool) -> None:
    # The finding names a share of the turn; a retried turn's response time
    # spans every attempt, and the failed call ran in one of them.
    failed = replace(_action(1, outcome="errored"), turn_id="retry")
    turn = _retried(_turn(critical={failed.operation_id: 30}), retry_ns)

    assert ("failed-attempt-critical-path" in _rule_ids(_evaluate(actions=(failed,), turn=turn))) is reported


@pytest.mark.parametrize(("retry_ns", "reported"), [(0, True), (900, False)])
def test_approval_share_is_measured_against_every_attempt_of_its_turn(retry_ns: int, reported: bool) -> None:
    action = replace(_action(1), turn_id="retry")
    turn = _retried(_turn(slices=(_approval_slice(30),)), retry_ns)

    assert ("approval-blocking-share" in _rule_ids(_evaluate(actions=(action,), turn=turn))) is reported


@pytest.mark.parametrize(("retried", "reported"), [(False, True), (True, False)])
def test_analysis_measures_approval_share_against_a_retried_turn(tmp_path: Path, retried: bool, reported: bool) -> None:
    # 4 s of approval is 40% of a 10 s attempt, but under 4% of a turn whose
    # retry then ran another 100 s.
    def attempt(turn_id: str, start: int, duration: int, *, wait: int = 0) -> None:
        log.add("turn.started", start * NS, turn_id=turn_id, payload={"turn_number": 1, "is_retry": start > 0})
        if wait:
            tool_id = "1" * 32
            log.span(
                "tool.operation",
                tool_id,
                start * NS,
                (start + wait) * NS,
                turn_id=turn_id,
                start_payload={"tool_name": "run_shell", "tool_kind": "shell", "call_item_id": "2" * 32},
            )
            request = {"approval_request_id": "3" * 32}
            log.add("approval.requested", start * NS, turn_id=turn_id, operation_id=tool_id, payload=request)
            log.add(
                "approval.resolved",
                (start + wait) * NS,
                turn_id=turn_id,
                operation_id=tool_id,
                payload={**request, "wait_ms": wait * 1000},
                measurements={"/payload/wait_ms": measurement("monotonic_clock", method_version=1)},
            )
        log.add(
            "turn.finished",
            (start + duration) * NS,
            turn_id=turn_id,
            payload={"end_reason": "cancelled", "duration_ms": duration * 1000},
        )

    log = EventLog()
    log.coverage()
    attempt("a" * 32, 0, 10, wait=4)
    if retried:
        attempt("b" * 32, 20, 100)
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert [len(turn.attempts) for turn in analysis.turns] == [2 if retried else 1]
    assert ("approval-blocking-share" in _rule_ids(analysis.findings)) is reported


def test_unknown_tool_outcome_does_not_emit_failed_attempt_finding() -> None:
    unknown = _action(1, outcome="unknown")
    dominating = _turn(critical={unknown.operation_id: 100})

    assert "failed-attempt-critical-path" not in _rule_ids(_evaluate(actions=(unknown,), turn=dominating))


@pytest.mark.parametrize(("outcome", "reported"), [("errored", True), ("rejected", False)])
def test_a_declined_call_is_not_reported_as_a_failed_call(outcome: str, reported: bool) -> None:
    # A declined call never ran: its time is the approval wait the approval
    # rule reports, not work that failed.
    call = _action(1, outcome=outcome)
    dominating = _turn(critical={call.operation_id: 30})

    assert ("failed-attempt-critical-path" in _rule_ids(_evaluate(actions=(call,), turn=dominating))) is reported


@pytest.mark.parametrize("waits_on", ["approval", "question"])
def test_waiting_on_the_user_inside_a_failed_call_is_not_reported_as_failed_work(tmp_path: Path, waits_on: str) -> None:
    # 25 s waiting on the user (an approval of a command that then failed, or
    # an ask_user question nobody answered) inside a 28 s failed call: only
    # the 3 s the call itself ran, under 10% of the turn, is failed work.
    # The approval rule reports an approval wait.
    tool_id = "f" * 32
    call_item_id = "9" * 32
    result_item_id = "a" * 32
    revision_id = "6" * 32
    segment_group_id = "5" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 40 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        30 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 2 * NS, parent_operation_id="c" * 32)
    log.span(
        "preparation",
        "e" * 32,
        2 * NS,
        2 * NS,
        parent_operation_id="c" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": tool_id},
    )
    log.add(
        "tool.operation.started",
        2 * NS,
        operation_id=tool_id,
        parent_operation_id="c" * 32,
        payload={
            "tool_name": "run_shell" if waits_on == "approval" else "ask_user",
            "tool_kind": "shell" if waits_on == "approval" else "ask_user",
            "parent_model_operation_id": "d" * 32,
            "call_item_id": call_item_id,
        },
        links=caused_by("e" * 32),
    )
    if waits_on == "approval":
        request = {"approval_request_id": "4" * 32, "target_tool_operation_id": tool_id}
        log.add("approval.requested", 2 * NS, operation_id=tool_id, parent_operation_id="d" * 32, payload=request)
        log.add(
            "approval.resolved",
            27 * NS,
            operation_id=tool_id,
            parent_operation_id="d" * 32,
            payload={**request, "decision": "approved", "wait_ms": 25_000},
            measurements={"/payload/wait_ms": measurement("monotonic_clock", method_version=1)},
        )
    else:
        question = {"category": "user_input", "target_operation_id": tool_id}
        log.span(
            "wait",
            "4" * 32,
            2 * NS,
            27 * NS,
            parent_operation_id=tool_id,
            start_payload=question,
            finish_payload={**question, "outcome": "timed_out"},
        )
    log.add(
        "tool.operation.finished",
        30 * NS,
        operation_id=tool_id,
        parent_operation_id="c" * 32,
        payload={"outcome": "failed", "duration_ms": 28_000, "result_item_id": result_item_id},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    revision = log.add(
        "context.revision.recorded",
        30 * NS,
        operation_id=revision_id,
        parent_operation_id="8" * 32,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 2, "unidentified_item_count": 0},
        segmented_fields=(
            SegmentedField(field_pointer="/payload/refs", segment_group_id=segment_group_id, segment_count=1),
        ),
    )
    log.add(
        "event.segment",
        30 * NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": segment_group_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": call_item_id, "occurrence": 0, "position": 0, "action": "add"},
                {"item_id": result_item_id, "occurrence": 0, "position": 1, "action": "add"},
            ],
        },
    )
    log.span(
        "model.cycle",
        "7" * 32,
        30 * NS,
        40 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "8" * 32},
    )
    log.span(
        "model.exchange",
        "8" * 32,
        30 * NS,
        40 * NS,
        parent_operation_id="7" * 32,
        start_payload={"context_revision_id": revision_id},
    )
    log.add("turn.finished", 40 * NS, payload={"end_reason": "cancelled", "duration_ms": 40_000})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    turn = analysis.turns[0]
    assert turn.response_cp_ns == Metric(40 * NS, Precision.EXACT)
    assert turn.critical_tool_contributions_ns == {tool_id: 3 * NS}
    assert ("approval-blocking-share" in _rule_ids(analysis.findings)) is (waits_on == "approval")
    assert "failed-attempt-critical-path" not in _rule_ids(analysis.findings)


def test_retry_token_amplification_rule_has_positive_and_negative_paths() -> None:
    action = _action(1)
    exchange = RetriedExchange(action.turn_id, action.operation_id, finish_sequence=7)
    positive = _evaluate(
        actions=(action,),
        retry=Metric(500, Precision.EXACT),
        retry_evidence=("item:stable",),
        retried_exchange=exchange,
    )
    negative = _evaluate(
        actions=(action,), retry=Metric(0, Precision.EXACT), retry_evidence=("item:stable",), retried_exchange=exchange
    )

    assert "retry-token-amplification" in _rule_ids(positive)
    assert "retry-token-amplification" not in _rule_ids(negative)


def test_net_zero_churn_rule_has_positive_and_negative_paths() -> None:
    edit = _action(1, classification=ActionClass.EDIT)
    evidence = (f"call:{edit.call_item_id}",)
    positive_change = _change(
        ChangeVerificationRow("a.py", ChangeVerificationState.NET_ZERO, 1, Precision.EXACT, evidence)
    )
    negative_change = _change(
        ChangeVerificationRow("a.py", ChangeVerificationState.UNVERIFIED, 1, Precision.EXACT, evidence)
    )
    # Withheld content backups leave the net change unprovable: the row only
    # believes the file returned to its original state, so the churn warning
    # must not present it as proven wasted work.
    unprovable_change = _change(
        ChangeVerificationRow("a.py", ChangeVerificationState.NET_ZERO, 1, Precision.UNRESOLVED, evidence)
    )

    assert "net-zero-churn" in _rule_ids(_evaluate(actions=(edit,), change=positive_change))
    assert "net-zero-churn" not in _rule_ids(_evaluate(actions=(edit,), change=negative_change))
    assert "net-zero-churn" not in _rule_ids(_evaluate(actions=(edit,), change=unprovable_change))


def test_approval_blocking_share_rule_has_positive_and_negative_paths() -> None:
    action = _action(1)
    high = _turn(slices=(_approval_slice(30),))
    low = _turn(slices=(_approval_slice(24),))

    assert "approval-blocking-share" in _rule_ids(_evaluate(actions=(action,), turn=high))
    assert "approval-blocking-share" not in _rule_ids(_evaluate(actions=(action,), turn=low))


def test_approval_blocking_share_unions_overlapping_waits() -> None:
    action = _action(1)
    # Two concurrent waits: the summed durations (20 + 14) would cross the
    # 25% threshold, but together they cover only 24ns of the 100ns turn.
    overlapping = _turn(slices=(_approval_slice(20), _approval_slice(14, start=10, index=1)))
    blocking = _turn(slices=(_approval_slice(20), _approval_slice(20, start=10, index=1)))

    assert "approval-blocking-share" not in _rule_ids(_evaluate(actions=(action,), turn=overlapping))
    assert "approval-blocking-share" in _rule_ids(_evaluate(actions=(action,), turn=blocking))


def test_context_carrying_load_rule_has_positive_and_negative_paths() -> None:
    carrying = (
        ContextCarryingLoad(
            120_000, "stable-item", "context:1", "turn", 1, token_count=12_000, carry_count=10, origin_turn_number=1
        ),
        ContextCarryingLoad(
            99_999, "light-item", "context:2", "turn", 1, token_count=33_333, carry_count=3, origin_turn_number=1
        ),
    )

    findings = [finding for finding in _evaluate(context=carrying) if finding.rule_id == "context-carrying-load"]

    # Only the item past the token floor becomes a finding; the lighter one
    # stays in the Insights ranking.
    assert [finding.detail_args for finding in findings] == [
        (("load", 120_000), ("tokens", 12_000), ("requests", 10), ("origin_turn", 1))
    ]
    assert "context-carrying-load" not in _rule_ids(_evaluate())


def test_finding_evidence_key_is_stable_across_projection_local_identity_changes() -> None:
    original = _action(1, classification=ActionClass.EDIT, operation_id="a" * 32, sequence=10)
    replayed = _action(1, classification=ActionClass.EDIT, operation_id="b" * 32, sequence=900)

    first = next(finding for finding in _evaluate(actions=(original,)) if finding.rule_id == "unverified-change")
    second = next(finding for finding in _evaluate(actions=(replayed,)) if finding.rule_id == "unverified-change")

    assert first.evidence_key == second.evidence_key
    assert first.occurrence_id != second.occurrence_id


def test_net_zero_finding_identity_depends_on_stable_evidence_not_file_path() -> None:
    edit = _action(1, classification=ActionClass.EDIT)
    evidence = (f"call:{edit.call_item_id}",)
    first_change = _change(
        ChangeVerificationRow("old.py", ChangeVerificationState.NET_ZERO, 1, Precision.EXACT, evidence)
    )
    second_change = _change(
        ChangeVerificationRow("renamed.py", ChangeVerificationState.NET_ZERO, 1, Precision.EXACT, evidence)
    )

    first = next(
        finding for finding in _evaluate(actions=(edit,), change=first_change) if finding.rule_id == "net-zero-churn"
    )
    second = next(
        finding for finding in _evaluate(actions=(edit,), change=second_change) if finding.rule_id == "net-zero-churn"
    )

    assert first.evidence_key == second.evidence_key


def test_deterministic_describes_every_rule() -> None:
    edit = _action(1, classification=ActionClass.EDIT)
    active = _evaluate(actions=(edit,))

    assert all(finding.deterministic for finding in active)


def _evaluate(
    *,
    actions: tuple[ActionOperation, ...] = (),
    turn: TurnAnalysis | None = None,
    retry: Metric | None = None,
    retry_evidence: tuple[str, ...] = (),
    retried_exchange: RetriedExchange | None = None,
    change: ChangeVerification | None = None,
    context: tuple[ContextCarryingLoad, ...] = (),
    outside_changes: tuple[SequenceSpan, ...] = (),
) -> tuple[FindingRow, ...]:
    selected_turn = turn or _turn()
    return evaluate_findings(
        actions=actions,
        turns=(selected_turn,),
        validation=_validation(actions, retry or Metric(0, Precision.EXACT)),
        change_verification=change or _change(),
        retry_amplification_evidence=retry_evidence,
        retried_exchange=retried_exchange,
        context_carrying_load=context,
        outside_changes=outside_changes,
    )


def _action(
    index: int,
    *,
    classification: ActionClass = ActionClass.SEARCH,
    outcome: str = "success",
    fingerprint: str | None = None,
    operation_id: str | None = None,
    sequence: int | None = None,
    end_sequence: int | None = None,
    tool_name: str | None = None,
    tool_kind: str = "search",
) -> ActionOperation:
    call_item_id = f"{index:032x}"
    argument_fingerprint = fingerprint or f"fingerprint-{index}"
    operation = operation_id or f"{index + 100:032x}"
    start_sequence = sequence or index
    return ActionOperation(
        evidence_key=f"classification:{call_item_id}",
        occurrence_id=f"tool:{start_sequence}:{operation}",
        operation_id=operation,
        turn_id="turn",
        turn_number=1,
        tool_name=tool_name or f"tool-{index}",
        tool_kind=tool_kind,
        call_item_id=call_item_id,
        argument_fingerprint=argument_fingerprint,
        classification=classification,
        classification_precision=Precision.EXACT,
        classification_reason=None,
        outcome=outcome,
        outcome_precision=Precision.EXACT,
        start_sequence=start_sequence,
        start_ns=index * 10,
        end_ns=index * 10 + 5,
        end_sequence=end_sequence if end_sequence is not None else start_sequence,
        actor_id="main",
    )


def _turn(
    *,
    critical: dict[str, int] | None = None,
    slices: tuple[TimeSlice, ...] = (),
) -> TurnAnalysis:
    exact_100 = Metric(100, Precision.EXACT)
    zero = Metric(0, Precision.EXACT)
    return TurnAnalysis(
        turn_id="turn",
        turn_number=1,
        runtime_id="runtime",
        start_sequence=1,
        end_sequence=99,
        elapsed_ns=exact_100,
        compute_cp_ns=exact_100,
        response_cp_ns=exact_100,
        exclusive_work_ns=exact_100,
        parallelism=Metric(1.0, Precision.EXACT),
        overlap_gain_ns=zero,
        wall_time_ns=dict.fromkeys(WallBucket, zero),
        utilization=dict.fromkeys(WallBucket, zero),
        usage_tokens=zero,
        attempts=(
            TurnAttemptRef(
                turn_id="turn",
                runtime_id="runtime",
                is_retry=False,
                physical_axis_start_ns=0,
                physical_axis_end_ns=100,
                logical_axis_start_ns=0,
                operation_start_index=0,
                operation_end_index=0,
                slice_start_index=0,
                slice_end_index=len(slices),
            ),
        ),
        axis_start_ns=0,
        axis_end_ns=100,
        slices=slices,
        critical_tool_contributions_ns=critical or {},
    )


def _retried(turn: TurnAnalysis, retry_ns: int) -> TurnAnalysis:
    """*turn* folded with a retry attempt ``"retry"`` that added *retry_ns* to its times."""
    added = Metric(100 + retry_ns, Precision.EXACT)
    retry = replace(turn.attempts[0], turn_id="retry", is_retry=True)
    return replace(turn, elapsed_ns=added, response_cp_ns=added, attempts=(*turn.attempts, retry))


def _approval_slice(duration: int, *, start: int = 0, index: int = 0) -> TimeSlice:
    return TimeSlice(
        family="approval",
        slice_index=index,
        turn_id="turn",
        runtime_id="runtime",
        operation_id="approval",
        owner="approval",
        start_ns=start,
        end_ns=start + duration,
        wall_bucket=WallBucket.WAIT,
        counts_as_work=False,
        compute_weight=False,
        response_weight=True,
    )


def _validation(actions: tuple[ActionOperation, ...], retry: Metric) -> ValidationMetrics:
    exact_zero = Metric(0, Precision.EXACT)
    return ValidationMetrics(
        funnel=ActionFunnel(exact_zero, exact_zero, exact_zero, exact_zero),
        time_to_first_edit_ns=exact_zero,
        first_edit_to_first_verify_ns=exact_zero,
        edit_verify_cycle_count=exact_zero,
        edit_verify_cycle_median_ns=exact_zero,
        unverified_change_count=Metric(
            sum(action.classification is ActionClass.EDIT for action in actions),
            Precision.EXACT,
        ),
        net_zero_churn_count=exact_zero,
        repeated_failure_signature_count=exact_zero,
        failure_recovery_median_ns=exact_zero,
        tool_failure_count=exact_zero,
        tool_count=Metric(len(actions), Precision.EXACT),
        retry_amplification_tokens=retry,
    )


def _change(row: ChangeVerificationRow | None = None) -> ChangeVerification:
    exact_zero = Metric(0, Precision.EXACT)
    return ChangeVerification(
        detail_available=True,
        detection_truncated=False,
        files_touched=Metric(1 if row is not None else 0, Precision.EXACT),
        created=exact_zero,
        modified=exact_zero,
        deleted=exact_zero,
        net_zero=Metric(row is not None and row.state is ChangeVerificationState.NET_ZERO, Precision.EXACT),
        rows=(row,) if row is not None else (),
        saved_through_sequence=100,
    )


def _rule_ids(findings: tuple[FindingRow, ...]) -> set[str]:
    return {finding.rule_id for finding in findings}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((), Precision.EXACT),
        ((Precision.EXACT, Precision.ESTIMATED), Precision.ESTIMATED),
        ((Precision.MISSING, Precision.ESTIMATED), Precision.MISSING),
        ((Precision.UNRESOLVED, Precision.MISSING, Precision.EXACT), Precision.UNRESOLVED),
    ],
)
def test_least_precision_takes_the_least_precise_value(values: tuple[Precision, ...], expected: Precision) -> None:
    assert least_precision(values) is expected
    assert least_precision(value for value in values) is expected


def test_a_sub_agent_s_reads_record_the_call_that_started_it(tmp_path: Path) -> None:
    # The recorded parent chain runs read → child cycle → sub-agent → the
    # parent's tool call, so the analyzer can tell that call encloses the
    # reads and does not count it as a change between them.
    child = Actor(kind="agent", role="sub_agent", actor_id="c" * 32)
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("model.run", "b" * 32, 0, 30 * NS)
    log.span("model.cycle", "d" * 32, 0, 30 * NS, parent_operation_id="b" * 32)
    log.span(
        "tool.operation",
        "e" * 32,
        2 * NS,
        20 * NS,
        parent_operation_id="d" * 32,
        start_payload={"tool_name": "general_agent", "tool_kind": "sub_agent", "call_item_id": "1" * 32},
    )
    log.span("sub_agent", "6" * 32, 3 * NS, 19 * NS, parent_operation_id="e" * 32, actor=child)
    log.span("model.cycle", "7" * 32, 4 * NS, 18 * NS, parent_operation_id="6" * 32, actor=child)
    for operation_id, call_item_id, start in (("8" * 32, "2" * 32, 5), ("9" * 32, "3" * 32, 8)):
        log.span(
            "tool.operation",
            operation_id,
            start * NS,
            (start + 1) * NS,
            parent_operation_id="7" * 32,
            start_payload={
                "tool_name": "read_file",
                "tool_kind": "filesystem.read",
                "call_item_id": call_item_id,
                "argument_fingerprint": "f" * 64,
            },
            actor=child,
        )
    log.add("turn.finished", 30 * NS, payload={"end_reason": "completed", "duration_ms": 30_000})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    reads = [action for action in analysis.actions if action.tool_name == "read_file"]
    assert [action.enclosing_operation_ids for action in reads] == [("e" * 32,), ("e" * 32,)]
    assert "repeated-read" in _rule_ids(analysis.findings)


@pytest.mark.parametrize(("phase", "reported"), [("phase4", False), ("phase1", True), (None, True)])
def test_compaction_that_drops_the_turns_results_keeps_a_re_read_from_counting_as_a_repeat(
    tmp_path: Path, phase: str | None, reported: bool
) -> None:
    # Only phase 4 drops the running turn's tool results; the other phases
    # rewrite earlier turns, which a repeat in this turn does not read from.
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    reads = (("8" * 32, "2" * 32, 1), ("9" * 32, "3" * 32, 7))
    for operation_id, call_item_id, start in reads[:1]:
        log.span(
            "tool.operation",
            operation_id,
            start * NS,
            (start + 1) * NS,
            start_payload={
                "tool_name": "read_file",
                "tool_kind": "filesystem.read",
                "call_item_id": call_item_id,
                "argument_fingerprint": "f" * 64,
            },
        )
    cycle_id = "c" * 32
    log.add("model.cycle.started", 3 * NS, operation_id=cycle_id)
    if phase is not None:
        log.span(
            "compaction",
            "d" * 32,
            4 * NS,
            5 * NS,
            parent_operation_id=cycle_id,
            start_payload={"compaction_run_id": "d" * 32, "trigger": "usage_threshold"},
        )
        log.add(
            EventType.COMPACTION_PHASE_FINISHED,
            5 * NS,
            operation_id="e" * 32,
            parent_operation_id="d" * 32,
            payload={
                "compaction_run_id": "d" * 32,
                "phase": phase,
                "groups_compacted": 1,
                "duration_ms": 500,
                "last_words_generated": phase == "phase4",
            },
            measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
        )
    log.add("model.cycle.finished", 6 * NS, operation_id=cycle_id, payload={"outcome": "success"})
    for operation_id, call_item_id, start in reads[1:]:
        log.span(
            "tool.operation",
            operation_id,
            start * NS,
            (start + 1) * NS,
            start_payload={
                "tool_name": "read_file",
                "tool_kind": "filesystem.read",
                "call_item_id": call_item_id,
                "argument_fingerprint": "f" * 64,
            },
        )
    log.add("turn.finished", 10 * NS, payload={"end_reason": "completed", "duration_ms": 10_000})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len([action for action in analysis.actions if action.tool_name == "read_file"]) == 2
    assert ("repeated-read" in _rule_ids(analysis.findings)) is reported


@pytest.mark.parametrize(("finished", "reported"), [(False, False), (True, True)])
def test_background_task_events_keep_a_log_re_read_from_counting_as_a_repeat(
    tmp_path: Path, finished: bool, reported: bool
) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(EventType.BACKGROUND_TASK_STARTED, NS, payload={"task": "t1", "pid": 1, "trigger": "timeout"})
    if finished:
        log.add(EventType.BACKGROUND_TASK_FINISHED, 2 * NS, payload={"task": "t1", "exit_code": 0})
    for operation_id, call_item_id, start in (("8" * 32, "2" * 32, 3), ("9" * 32, "3" * 32, 6)):
        log.span(
            "tool.operation",
            operation_id,
            start * NS,
            (start + 1) * NS,
            start_payload={
                "tool_name": "read_file",
                "tool_kind": "filesystem.read",
                "call_item_id": call_item_id,
                "argument_fingerprint": "f" * 64,
            },
        )
    log.add("turn.finished", 10 * NS, payload={"end_reason": "completed", "duration_ms": 10_000})
    path = tmp_path / "events.jsonl"
    log.write(path)

    assert ("repeated-read" in _rule_ids(analyze_trajectory(path).findings)) is reported


@pytest.mark.parametrize(("closed", "reported"), [(False, True), (True, False)])
def test_a_background_task_left_open_by_a_crashed_runtime_no_longer_counts_as_a_change(
    tmp_path: Path, closed: bool, reported: bool
) -> None:
    # A runtime that ended without closing its log took the task with it, or
    # no longer tracks it; one that closed its log in place (a rollback) keeps
    # the task running into the next.
    next_runtime_id = "7" * 32
    log = EventLog()
    log.coverage()
    log.add(EventType.RUNTIME_STARTED, 0, turn_id=None)
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(EventType.BACKGROUND_TASK_STARTED, NS, payload={"task": "t1", "pid": 1, "trigger": "timeout"})
    log.add("turn.finished", 2 * NS, payload={"end_reason": "completed", "duration_ms": 2_000})
    if closed:
        log.add(EventType.RUNTIME_FINISHED, 2 * NS, turn_id=None, payload={"reason": "session_switch"})
    log.add(
        EventType.COVERAGE_STARTED,
        3 * NS,
        turn_id=None,
        runtime_id=next_runtime_id,
        payload={"coverage_reason": "runtime_resumed"},
    )
    log.add(EventType.RUNTIME_STARTED, 3 * NS, turn_id=None, runtime_id=next_runtime_id)
    log.add("turn.started", 4 * NS, turn_id="5" * 32, runtime_id=next_runtime_id, payload={"turn_number": 2})
    for operation_id, call_item_id, start in (("8" * 32, "2" * 32, 5), ("9" * 32, "3" * 32, 8)):
        log.span(
            "tool.operation",
            operation_id,
            start * NS,
            (start + 1) * NS,
            turn_id="5" * 32,
            runtime_id=next_runtime_id,
            start_payload={
                "tool_name": "read_file",
                "tool_kind": "filesystem.read",
                "call_item_id": call_item_id,
                "argument_fingerprint": "f" * 64,
            },
        )
    log.add(
        "turn.finished",
        10 * NS,
        turn_id="5" * 32,
        runtime_id=next_runtime_id,
        payload={"end_reason": "completed", "duration_ms": 6_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len([action for action in analysis.actions if action.tool_name == "read_file"]) == 2
    assert ("repeated-read" in _rule_ids(analysis.findings)) is reported
