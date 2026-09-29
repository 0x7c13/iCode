# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Each attempt of a workflow agent node has its own live transcript: a retry resumes one or starts over."""

from __future__ import annotations

from chrys.app.tui.widgets.chat.agent_transcript_surface import (
    AgentTranscriptJournal,
    AgentTranscriptOp,
    TranscriptAssistantOp,
    TranscriptErrorOp,
    TranscriptResumedOp,
    TranscriptToolResultOp,
    TranscriptToolStartOp,
    TranscriptUserOp,
)
from chrys.app.tui.widgets.workflow.projector import WorkflowProjector
from chrys.foundation.events import types as events
from chrys.foundation.models.invocations import InvocationOrigin


def _origin(attempt: int) -> InvocationOrigin:
    return InvocationOrigin("workflow_node", "", "review", None, attempt=attempt)


def _state(attempt: int, state: str, error: str = "") -> events.WorkflowNodeStateChanged:
    return events.WorkflowNodeStateChanged(
        run_id="run",
        node_id="review",
        activation_id="review@iter#1",
        attempt=attempt,
        state=state,
        invocation_id="review",
        error=error,
    )


def _first_attempt(projector: WorkflowProjector) -> None:
    projector.record(events.WorkflowRunStarted(run_id="run", manifest={"nodes": []}))
    projector.record(_state(1, "running"))
    projector.record_invocation(events.InvocationStarted(origin=_origin(1), opening_prompt="Review the change."))
    projector.record_invocation(events.InvocationToolCallStart(origin=_origin(1), call_id="a", tool_name="read"))
    projector.record_invocation(
        events.InvocationToolCallResult(origin=_origin(1), call_id="a", tool_name="read", result="text")
    )
    projector.record(_state(1, "retrying", error="no response within 30s"))


def _kinds(operations: tuple[AgentTranscriptOp, ...]) -> list[type]:
    return [type(operation) for operation in operations]


def test_a_resumed_attempt_carries_the_previous_transcript_on_and_drops_its_late_facts() -> None:
    projector = WorkflowProjector()
    _first_attempt(projector)
    projector.record(_state(2, "running"))
    projector.record_invocation(events.InvocationResumed(origin=_origin(2)))
    # The first attempt's pass drained after its failure was reported; its late fact stays out.
    projector.record_invocation(events.InvocationToolCallStart(origin=_origin(1), call_id="late", tool_name="read"))
    projector.record_invocation(events.InvocationMessage(origin=_origin(2), text="Done.", is_final=True))

    run = projector.current
    assert run is not None
    assert list(run.journals) == [("review", 2)]
    assert _kinds(run.journals["review", 2].operations) == [
        TranscriptUserOp,
        TranscriptToolStartOp,
        TranscriptToolResultOp,
        TranscriptErrorOp,
        TranscriptResumedOp,
        TranscriptAssistantOp,
    ]


def test_an_attempt_that_starts_over_shows_only_the_prompt_and_its_own_facts() -> None:
    projector = WorkflowProjector()
    _first_attempt(projector)
    projector.record(_state(2, "running"))
    projector.record_invocation(events.InvocationToolCallStart(origin=_origin(2), call_id="b", tool_name="read"))
    projector.record_invocation(events.InvocationToolCallStart(origin=_origin(1), call_id="late", tool_name="read"))

    run = projector.current
    assert run is not None
    operations = run.journals["review", 2].operations
    assert _kinds(operations) == [TranscriptUserOp, TranscriptToolStartOp]
    first, tool = operations
    assert isinstance(first, TranscriptUserOp) and first.text == "Review the change."
    assert isinstance(tool, TranscriptToolStartOp) and tool.call_id == "b"


def test_an_attempt_that_fails_before_any_fact_shows_the_prompt_and_its_error() -> None:
    projector = WorkflowProjector()
    _first_attempt(projector)
    projector.record(_state(2, "running"))
    projector.record(_state(2, "failed", error="The ACP agent requires authentication."))

    run = projector.current
    assert run is not None
    operations = run.journals["review", 2].operations
    assert _kinds(operations) == [TranscriptUserOp, TranscriptErrorOp]
    assert isinstance(operations[1], TranscriptErrorOp)
    assert operations[1].reason == "The ACP agent requires authentication."


class _Subscriber:
    def __init__(self) -> None:
        self.received: list[AgentTranscriptOp] = []

    def enqueue(self, operation: AgentTranscriptOp) -> None:
        self.received.append(operation)


def test_a_continued_journal_sends_the_carried_transcript_to_its_live_subscribers() -> None:
    earlier = AgentTranscriptJournal()
    earlier.record(TranscriptUserOp("Review the change."))
    earlier.record(TranscriptErrorOp("failed"))
    journal = AgentTranscriptJournal()
    subscriber = _Subscriber()
    assert journal.subscribe(subscriber) == ()  # type: ignore[arg-type]  # a surface only enqueues

    journal.continue_from(earlier)
    journal.record(TranscriptResumedOp())
    journal.record(TranscriptAssistantOp("Done.", final=True))

    assert _kinds(tuple(subscriber.received)) == [
        TranscriptUserOp,
        TranscriptErrorOp,
        TranscriptResumedOp,
        TranscriptAssistantOp,
    ]
    assert journal.operations == tuple(subscriber.received)
