# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Recent workflow facts and transcript journals for the two most recent observed runs.

Recording never walks a manifest or touches a widget. Frontends coalesce only
visual projection; activation/attempt and process records retain their order.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime

from chrys.app.tui.util.context_pressure import context_pressure_message
from chrys.app.tui.widgets.chat.agent_transcript_surface import (
    AgentTranscriptJournal,
    AgentTranscriptOp,
    TranscriptAssistantOp,
    TranscriptCompactionFinishedOp,
    TranscriptCompactionStartOp,
    TranscriptErrorOp,
    TranscriptInterruptedOp,
    TranscriptPresentationAcceptedOp,
    TranscriptPresentationRejectedOp,
    TranscriptResumedOp,
    TranscriptRetryOp,
    TranscriptToolArgsOp,
    TranscriptToolProgressOp,
    TranscriptToolResultOp,
    TranscriptToolStartOp,
    TranscriptToolStatusOp,
    TranscriptUserOp,
    TranscriptWarningOp,
)
from chrys.app.tui.widgets.workflow import text
from chrys.foundation.events import types as events
from chrys.foundation.events.workflow import WORKFLOW_RUN_EVENTS, WorkflowRunEvent
from chrys.foundation.tool_result_metadata import canonical_tool_result_status
from chrys.service.workflows.transcript import NodeUsage

WORKFLOW_EVENTS = (events.WorkflowRunAccepted, events.WorkflowRunRejected, *WORKFLOW_RUN_EVENTS)
INVOCATION_EVENTS = (
    events.InvocationStarted,
    events.InvocationMessage,
    events.InvocationPresentationAttemptAccepted,
    events.InvocationPresentationAttemptRejected,
    events.InvocationToolCallStart,
    events.InvocationToolCallArgsUpdated,
    events.InvocationToolCallStatusUpdated,
    events.InvocationToolCallProgress,
    events.InvocationToolCallResult,
    events.InvocationProgress,
    events.InvocationCompactionStarted,
    events.InvocationCompactionFinished,
    events.InvocationCompactionCommitted,
    events.InvocationRetryAttempt,
    events.InvocationPaused,
    events.InvocationResumed,
    events.InvocationCascadeAborted,
    events.InvocationAborted,
    events.InvocationContextPressure,
)


RECENT_FACT_CAPACITY = 200


@dataclass
class NodeTiming:
    """Active time across attempts; manual retry waits do not advance the clock."""

    seconds: float = 0.0
    running_since: datetime | None = None

    def record(self, state: str, timestamp: datetime) -> None:
        if state == "running":
            if self.running_since is None:
                self.running_since = timestamp
        elif self.running_since is not None:
            self.seconds += max(0.0, (timestamp - self.running_since).total_seconds())
            self.running_since = None


@dataclass
class ObservedRun:
    started: events.WorkflowRunStarted
    facts: deque[WorkflowRunEvent] = field(default_factory=lambda: deque(maxlen=RECENT_FACT_CAPACITY))
    fact_count: int = 0
    nodes: dict[str, events.WorkflowNodeStateChanged] = field(default_factory=dict)
    attempts: dict[tuple[str, int], events.WorkflowNodeStateChanged] = field(default_factory=dict)
    iterations: dict[str, events.WorkflowLoopIteration] = field(default_factory=dict)
    journals: dict[str, AgentTranscriptJournal] = field(default_factory=dict)
    questions: dict[str, events.WorkflowNodeAskUser] = field(default_factory=dict)
    question_history: dict[str, events.WorkflowNodeAskUser] = field(default_factory=dict)
    question_states: dict[str, str] = field(default_factory=dict)
    answers: dict[str, events.WorkflowNodeAnswered] = field(default_factory=dict)
    notices: dict[str, events.WorkflowRunNotice] = field(default_factory=dict)
    revisions: dict[str, int] = field(default_factory=dict)
    usage: dict[str, NodeUsage] = field(default_factory=dict)
    timings: dict[str, NodeTiming] = field(default_factory=dict)
    finished: events.WorkflowRunFinished | None = None

    @property
    def loop_iterations(self) -> dict[str, tuple[int, int]]:
        """Current iterations come from entry activations; verdicts arrive only after an iteration ends."""
        iterations = {}
        for node in self.started.manifest.get("nodes", []):
            if node["kind"] != "loop":
                continue
            loop_id, loop = node["id"], node["loop"]
            verdict = self.iterations.get(loop_id)
            current = verdict.iteration if verdict is not None else 0
            entry = self.nodes.get(loop["entry"])
            if entry is not None:
                current = max(current, entry.iteration)
            activation = self.nodes.get(loop_id)
            if activation is not None and activation.state not in {"pending", "skipped"}:
                current = max(current, 1)
            if current:
                iterations[loop_id] = current, loop["max_iterations"]
        return iterations

    @property
    def status(self) -> str:
        """One run-level status shared by the graph header and main-screen chrome."""
        if self.finished is not None:
            return self.finished.outcome
        if any(node.state == "awaiting_retry" for node in self.nodes.values()):
            return "awaiting_retry"
        return "running"


class WorkflowProjector:
    def __init__(self) -> None:
        self.current: ObservedRun | None = None
        self.previous: ObservedRun | None = None

    def run(self, run_id: str) -> ObservedRun | None:
        for run in (self.current, self.previous):
            if run is not None and run.started.run_id == run_id:
                return run
        return None

    def record(self, event: WorkflowRunEvent) -> None:
        if isinstance(event, events.WorkflowRunStarted):
            self.previous, self.current = self.current, ObservedRun(event)
        run = self.run(event.run_id)
        if run is None:
            return
        run.facts.append(event)
        run.fact_count += 1
        if isinstance(event, events.WorkflowNodeStateChanged):
            previous_node = run.nodes.get(event.node_id)
            previous = run.attempts.get((event.activation_id, event.attempt))
            if previous is not None and not event.invocation_id:
                event = replace(event, invocation_id=previous.invocation_id)
            run.nodes[event.node_id] = event
            run.attempts[event.activation_id, event.attempt] = event
            if event.state == "running" or event.activation_id in run.timings:
                run.timings.setdefault(event.activation_id, NodeTiming()).record(event.state, event.timestamp)
            if event.state != "running":
                for key, question in list(run.questions.items()):
                    if (question.activation_id, question.attempt) == (event.activation_id, event.attempt):
                        run.question_states[key] = "cancelled" if event.state == "cancelled" else "abandoned"
                        del run.questions[key]
            if event.invocation_id:
                run.usage.setdefault(event.invocation_id, NodeUsage())
                journal = run.journals.setdefault(event.invocation_id, AgentTranscriptJournal())
                if event.state == "cancelled":
                    journal.record(TranscriptInterruptedOp(text.CANCELLED.bind()))
                elif event.state in {"failed", "retrying", "awaiting_retry"}:
                    journal.record(TranscriptErrorOp(event.error or text.FAILED.bind()))
                elif (
                    event.state == "running"
                    and previous_node is not None
                    and previous_node.invocation_id == event.invocation_id
                    and previous_node.state in {"failed", "retrying", "awaiting_retry"}
                ):
                    journal.record(TranscriptResumedOp())
        elif isinstance(event, events.WorkflowLoopIteration):
            run.iterations[event.loop_id] = event
        elif isinstance(event, events.WorkflowRunFinished):
            run.finished = event
            for key in run.questions:
                run.question_states[key] = "cancelled" if event.outcome == "cancelled" else "abandoned"
            run.questions.clear()
            for timing in run.timings.values():
                timing.record(event.outcome, event.timestamp)
        elif isinstance(event, events.WorkflowNodeAskUser):
            run.questions[event.request_id] = event
            run.question_history[event.request_id] = event
            run.question_states[event.request_id] = "pending"
        elif isinstance(event, events.WorkflowNodeAnswered):
            run.questions.pop(event.request_id, None)
            run.answers[event.request_id] = event
            run.question_states[event.request_id] = "answered"
        elif isinstance(event, events.WorkflowRunNotice):
            run.notices[event.code] = event
        if isinstance(event, (events.WorkflowNodeStateChanged, events.WorkflowNodeOutput)):
            run.revisions[event.node_id] = run.revisions.get(event.node_id, 0) + 1

    def record_invocation(self, event: events.InvocationEvent) -> None:
        if event.origin.kind != "workflow_node":
            return
        invocation_id = event.origin.invocation_id
        for run in (self.current, self.previous):
            if run is not None and invocation_id in run.journals:
                if isinstance(event, events.InvocationProgress):
                    run.usage[invocation_id] = NodeUsage(
                        event.tool_call_count, event.total_usage_tokens, event.usage_unreported_attempts
                    )
                operation = transcript_operation(event)
                if operation is not None:
                    run.journals[invocation_id].record(operation)
                return


def transcript_operation(event: events.InvocationEvent) -> AgentTranscriptOp | None:
    if isinstance(event, events.InvocationStarted) and event.opening_prompt:
        return TranscriptUserOp(event.opening_prompt)
    if isinstance(event, events.InvocationMessage):
        return TranscriptAssistantOp(event.text, final=event.is_final, presentation=event.presentation)
    if isinstance(event, events.InvocationPresentationAttemptAccepted):
        return TranscriptPresentationAcceptedOp(event.attempt_id, event.segment_ids)
    if isinstance(event, events.InvocationPresentationAttemptRejected):
        return TranscriptPresentationRejectedOp(event.attempt_id)
    if isinstance(event, events.InvocationToolCallStart):
        return TranscriptToolStartOp(
            event.call_id,
            event.tool_name,
            event.tool_kind,
            event.args,
            event.provider_hosted,
            event.hosted_family,
            event.provider,
            event.provider_item_type,
            event.provider_status,
            event.provider_call_id,
        )
    if isinstance(event, events.InvocationToolCallArgsUpdated):
        return TranscriptToolArgsOp(event.call_id, event.args)
    if isinstance(event, events.InvocationToolCallStatusUpdated):
        return TranscriptToolStatusOp(event.call_id, event.status, event.provider_status, event.metadata)
    if isinstance(event, events.InvocationToolCallProgress):
        return TranscriptToolProgressOp(
            event.call_id, event.lines, event.image_contents, event.snapshot_metadata, event.provider_status
        )
    if isinstance(event, events.InvocationToolCallResult):
        canonical_status = canonical_tool_result_status(event.provider_status, event.metadata)
        return TranscriptToolResultOp(
            event.call_id,
            event.tool_name,
            event.result,
            event.duration_ms,
            event.image_contents,
            event.artifacts,
            event.metadata.get("approval"),
            event.metadata,
            event.provider_status,
            canonical_status,
        )
    if isinstance(event, events.InvocationCompactionStarted):
        return TranscriptCompactionStartOp(event.compaction_id)
    if isinstance(event, events.InvocationCompactionFinished):
        return TranscriptCompactionFinishedOp(
            event.compaction_id, event.outcome, event.duration_ms, event.format_violation, event.failure_reason
        )
    if isinstance(event, events.InvocationRetryAttempt):
        if event.scope == "compaction":
            return TranscriptRetryOp(
                event.detail or event.message,
                event.attempt,
                event.max_attempts,
                event.delay_seconds,
                compaction=True,
            )
        return TranscriptRetryOp(
            event.message if event.display_message is None else event.display_message,
            event.attempt,
            event.max_attempts,
            event.delay_seconds,
            hint=None if event.display_message is None else event.display_hint,
        )
    if isinstance(event, events.InvocationPaused):
        return TranscriptErrorOp(event.last_error or text.AWAITING_RETRY.bind())
    if isinstance(event, events.InvocationResumed):
        return TranscriptResumedOp()
    if isinstance(event, (events.InvocationAborted, events.InvocationCascadeAborted)):
        return TranscriptInterruptedOp(text.CANCELLED.bind())
    if isinstance(event, events.InvocationContextPressure):
        return TranscriptWarningOp(context_pressure_message(event.reason, source=event.source))
    return None
