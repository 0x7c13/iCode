# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Read historical workflow presentation and node values only when a view requests them."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from chrys.app.tui.widgets.workflow.projector import ObservedRun, WorkflowProjector
from chrys.foundation.events import types as events
from chrys.service.workflows.outcomes import RunOutcome
from chrys.service.workflows.records import decode_run_event, read_run_started
from chrys.service.workflows.store import read_run_events
from chrys.service.workflows.transcript import read_node_usage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from chrys.service.workflows.artifacts import WorkflowRunRecord


def read_observed_run(record: WorkflowRunRecord) -> ObservedRun:
    """Restore graph/attempt facts; node session archives are loaded lazily by details."""
    projector = WorkflowProjector()
    started = read_run_started(record.directory)
    projector.record(started)
    for record_event in read_run_events(record.directory).events:
        event = decode_run_event(record_event, directory=record.directory)
        if event is not None:
            projector.record(event)
    run = projector.current
    if run is None:
        raise RuntimeError("Replaying a workflow record did not establish a run.")
    terminal = run.finished
    if terminal is None:
        terminal = events.WorkflowRunFinished(
            run_id=started.run_id, outcome=RunOutcome.ORPHANED.value, timestamp=started.timestamp
        )
        projector.record(terminal)
    if terminal.outcome == RunOutcome.ORPHANED.value:
        run.attempts = {
            key: replace(node, state="cancelled") if node.state in {"running", "retrying", "awaiting_retry"} else node
            for key, node in run.attempts.items()
        }
        run.nodes = {node_id: run.attempts[node.activation_id, node.attempt] for node_id, node in run.nodes.items()}
    run.journals.clear()
    for node in run.nodes.values():
        if not node.invocation_id:
            continue
        try:
            usage = read_node_usage(record.directory, node.activation_id, node.attempt)
        except OSError, ValueError:
            logger.debug("Unable to load workflow node usage for %s", node.activation_id, exc_info=True)
            continue
        if usage is not None:
            run.usage[node.invocation_id] = usage
    return run
