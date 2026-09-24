# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Service-owned conversion from immutable run artifacts and log records to lifecycle facts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chrys.foundation.events import types as events
from chrys.foundation.events.workflow import WorkflowRunEvent
from chrys.foundation.trajectory.envelope import TrajectoryEvent
from chrys.foundation.util.time import parse_created_at
from chrys.service.state.workflow import decode_workflow_model
from chrys.service.workflows.layout import OUTPUT_INDEX_FILE
from chrys.service.workflows.store import (
    MAX_OUTPUT_INDEX_BYTES,
    RunRecord,
    read_json_object,
    read_run_header,
    read_run_input,
    read_run_spec,
)


def record_timestamp(value: object) -> datetime:
    parsed = parse_created_at(value)
    return parsed if parsed is not None and parsed.tzinfo is not None else datetime(1970, 1, 1, tzinfo=UTC)


def run_started_event(
    header: Mapping[str, Any], spec: Mapping[str, Any], input_text: str, *, seq: int = 0
) -> events.WorkflowRunStarted:
    """The same start fact for live publication and historical replay."""
    return events.WorkflowRunStarted(
        seq=seq,
        run_id=header["run_id"],
        session_id=header["session_id"],
        input_text=input_text,
        workflow_id=header["workflow_id"],
        source_kind=header["source_kind"],
        canonical_path=header["canonical_path"],
        title=header["title"],
        spec_digest=header["spec_digest"],
        manifest=dict(spec["manifest"]),
        resolved_nodes=[dict(node) for node in spec["resolved_nodes"]],
        model=decode_workflow_model(header["model"]),
        timestamp=record_timestamp(header["started_at"]),
    )


def read_run_started(directory: Path, *, seq: int = 0) -> events.WorkflowRunStarted:
    return run_started_event(read_run_header(directory), read_run_spec(directory), read_run_input(directory), seq=seq)


def decode_run_event(record: TrajectoryEvent, *, directory: Path) -> WorkflowRunEvent | None:
    """Decode the run-log namespace; trajectory runtime markers have no workflow projection."""
    kind, payload = record.event_type, record.payload
    if kind == RunRecord.RUN_STARTED:
        return read_run_started(directory, seq=record.sequence)
    common: dict[str, Any] = {
        "run_id": record.runtime_id,
        "session_id": record.session_id,
        "seq": record.sequence,
        "timestamp": record_timestamp(record.occurred_at),
    }
    if kind in {
        RunRecord.NODE_STATE,
        RunRecord.NODE_OUTPUT,
        RunRecord.NODE_ASK,
        RunRecord.NODE_ANSWER,
        RunRecord.LOOP_ITERATION,
        RunRecord.RUN_NOTICE,
    }:
        ref = {"activation_id": payload["activation"], "attempt": payload["attempt"]}
        if kind == RunRecord.LOOP_ITERATION:
            return events.WorkflowLoopIteration(
                **common, **ref, loop_id=payload["node"], iteration=payload["iteration"], verdict=payload["verdict"]
            )
        ref["node_id"] = payload["node"]
        if kind == RunRecord.NODE_STATE:
            return events.WorkflowNodeStateChanged(
                **common,
                **ref,
                state=payload["state"],
                iteration=payload["iteration"],
                failure_phase=payload["failure_phase"],
                invocation_id=payload.get("invocation", ""),
                error=payload.get("error", ""),
                error_class=payload.get("error_class", ""),
            )
        if kind == RunRecord.NODE_OUTPUT:
            return events.WorkflowNodeOutput(
                **common, **ref, kind=payload["kind"], ordinal=payload["ordinal"], summary_text=payload["summary"]
            )
        if kind == RunRecord.NODE_ASK:
            return events.WorkflowNodeAskUser(**common, **ref, request_id=payload["request"], prompt=payload["prompt"])
        if kind == RunRecord.NODE_ANSWER:
            return events.WorkflowNodeAnswered(**common, **ref, request_id=payload["request"], answer=payload["answer"])
        return events.WorkflowRunNotice(**common, **ref, code=payload["code"], message=payload["message"])
    if kind == RunRecord.RUN_FINISHED:
        outputs = payload.get("outputs", [])
        omitted = payload.get("outputs_omitted", 0)
        if omitted:
            try:
                outputs = read_json_object(directory / OUTPUT_INDEX_FILE, MAX_OUTPUT_INDEX_BYTES)["outputs"]
                if len(outputs) != omitted:
                    raise ValueError("Output count does not match the terminal.")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Cannot read workflow output index: {exc}") from exc
        if not isinstance(outputs, list):
            raise ValueError("Workflow outputs must be an array.")
        summaries = []
        for output in outputs:
            if (
                not isinstance(output, dict)
                or not isinstance(output.get("node"), str)
                or not isinstance(output.get("activation"), str)
                or type(output.get("attempt")) is not int
                or output["attempt"] < 0
            ):
                raise ValueError("Invalid workflow output identity in terminal or output index.")
            summaries.append(
                events.WorkflowOutputSummary(
                    node_id=output["node"], activation_id=output["activation"], attempt=output["attempt"]
                )
            )
        return events.WorkflowRunFinished(
            **common,
            outcome=payload["outcome"],
            outputs=summaries,
            error=payload.get("error", ""),
            duration=payload.get("duration", 0),
            reason=payload.get("reason", ""),
            node_id=payload.get("node", ""),
        )
    return None
