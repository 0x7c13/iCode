# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session run listings and narrow, path-based workflow artifact readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chrys.foundation.events import types as events
from chrys.foundation.platform.files import read_owner_verified_bounded
from chrys.service.workflows.discovery import MAX_SOURCE_BYTES
from chrys.service.workflows.layout import SOURCE_FILE, iter_run_dirs
from chrys.service.workflows.orphans import read_run_terminal
from chrys.service.workflows.records import RunRecord, decode_run_event
from chrys.service.workflows.store import (
    NODE_RECORD_INPUT,
    NODE_RECORD_OUTPUT,
    RunTerminal,
    node_stderr_path,
    read_node_diagnostics,
    read_node_emits,
    read_node_value,
    read_run_events,
    read_run_header,
)


@dataclass(frozen=True, slots=True)
class NodeRecords:
    input: dict[str, Any] | None
    output: dict[str, Any] | None
    emits: list[tuple[int, str]]
    diagnostics: dict[str, Any] | None
    stderr_path: Path | None = None


@dataclass(frozen=True, slots=True)
class WorkflowRunRecord:
    """A header-only history entry; larger artifacts are read only when requested."""

    directory: Path
    header: dict[str, Any]
    terminal: RunTerminal | None = None


def read_node_records(directory: Path, activation_id: str, attempt: int, *, node_kind: str = "") -> NodeRecords:
    stderr = node_stderr_path(directory, activation_id)
    return NodeRecords(
        input=read_node_value(directory, activation_id, attempt, NODE_RECORD_INPUT, node_kind=node_kind),
        output=read_node_value(directory, activation_id, attempt, NODE_RECORD_OUTPUT, node_kind=node_kind),
        emits=read_node_emits(directory, activation_id, attempt),
        diagnostics=read_node_diagnostics(directory, activation_id, attempt),
        stderr_path=stderr if stderr.is_file() else None,
    )


def read_node_output(
    directory: Path, activation_id: str, attempt: int, *, node_kind: str = ""
) -> dict[str, Any] | None:
    """Read a result without loading its input, emit log or diagnostics."""
    return read_node_value(directory, activation_id, attempt, NODE_RECORD_OUTPUT, node_kind=node_kind)


def read_run_source(directory: Path) -> bytes:
    return read_owner_verified_bounded(directory / SOURCE_FILE, max_bytes=MAX_SOURCE_BYTES)


def session_runs(session_dir: Path) -> list[WorkflowRunRecord]:
    """Read and order headers once when a caller needs history for multiple sources."""
    try:
        directories, _truncated = iter_run_dirs(session_dir)
    except OSError:
        return []
    records: list[WorkflowRunRecord] = []
    for directory in directories:
        try:
            header = read_run_header(directory)
        except OSError, ValueError:
            continue
        try:
            terminal = read_run_terminal(directory)
        except OSError, ValueError:
            terminal = None
        records.append(WorkflowRunRecord(directory, header, terminal))
    records.sort(key=lambda record: str(record.header.get("started_at", "")), reverse=True)
    return records


def latest_node_output(directory: Path, node_id: str, *, node_kind: str) -> dict[str, Any] | None:
    """Locate one node's latest stored result without projecting a run or reading usage archives."""
    candidates: dict[tuple[str, int], None] = {}
    for record in read_run_events(directory).events:
        if (
            record.event_type not in (RunRecord.NODE_STATE, RunRecord.NODE_OUTPUT)
            or record.payload.get("node") != node_id
        ):
            continue
        event = decode_run_event(record, directory=directory)
        if isinstance(event, events.WorkflowNodeStateChanged) or (
            isinstance(event, events.WorkflowNodeOutput) and event.kind == events.WORKFLOW_OUTPUT_FINAL
        ):
            candidates[event.activation_id, event.attempt] = None
    for activation_id, attempt in reversed(candidates):
        output = read_node_output(directory, activation_id, attempt, node_kind=node_kind)
        if output is not None:
            return output
    return None
