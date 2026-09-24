# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Per-attempt agent archives, using the same state and replay contract as sub-agents."""

from __future__ import annotations

import errno
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from chrys.service.session.sub_agent_logs import MAX_SUB_AGENT_AUDIT_BYTES
from chrys.service.session.sub_agent_transcript import PersistedSubAgentTranscript, project_agent_transcript
from chrys.service.workflows.store import NODE_RECORD_SESSION, NODE_RECORD_USAGE, node_value_path, read_json_object


@dataclass(frozen=True, slots=True)
class NodeUsage:
    """Cumulative agent usage, independent of the much larger transcript."""

    tool_calls: int = 0
    usage_tokens: int = 0
    unreported_attempts: int = 0


def read_node_usage(run_dir: Path, activation_id: str, attempt: int) -> NodeUsage | None:
    """Read the small graph summary without deserializing the agent history."""
    path = node_value_path(run_dir, activation_id, attempt, NODE_RECORD_USAGE)
    try:
        payload = read_json_object(path, 4096)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR):
            return None
        raise
    return NodeUsage(
        _counter(payload, "tool_call_count"),
        _counter(payload, "total_usage_tokens"),
        _counter(payload, "usage_unreported_attempts"),
    )


@dataclass(frozen=True, slots=True)
class NodeTranscript:
    """Replay messages and presentation counters from one immutable attempt boundary."""

    replay: PersistedSubAgentTranscript
    status: str
    error: str
    usage: NodeUsage
    context_tokens: int


def read_node_transcript(run_dir: Path, activation_id: str, attempt: int) -> NodeTranscript | None:
    """Load only the selected attempt, with the audit reader's ownership and size bounds."""
    path = node_value_path(run_dir, activation_id, attempt, NODE_RECORD_SESSION)
    try:
        envelope = read_json_object(path, MAX_SUB_AGENT_AUDIT_BYTES)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR):
            return None
        raise
    meta = envelope.get("meta")
    if (
        not isinstance(meta, dict)
        or meta.get("record_type") != "workflow_node_session"
        or meta.get("activation_id") != activation_id
        or meta.get("attempt") != attempt
    ):
        raise ValueError("Workflow node session identity does not match its attempt")
    # A running snapshot has no live owner when loaded from history. Replay
    # its available prefix, settling any unfinished ACP tools as interrupted.
    if meta.get("status") == "running":
        meta = {**meta, "status": "orphaned"}
        envelope = {**envelope, "meta": meta}
    replay = project_agent_transcript(envelope)
    if replay is None:
        raise ValueError("Invalid workflow agent transcript")
    if meta.get("runner") == "acp":
        # ACP updates contain agent/tool output only. Preserve the exact prompt
        # sent for this attempt, including its archived instructions suffix.
        prompt = envelope["acp_state"]["prompt"]
        if prompt:
            replay = replace(
                replay,
                messages=[{"role": "user", "contents": [{"type": "text", "text": prompt}]}, *replay.messages],
            )
    return NodeTranscript(
        replay=replay,
        status=str(meta.get("status", "")),
        error=str(meta.get("last_error", "")),
        usage=NodeUsage(
            _counter(meta, "tool_call_count"),
            _counter(meta, "total_usage_tokens"),
            _counter(meta, "usage_unreported_attempts"),
        ),
        context_tokens=_counter(meta, "total_tokens"),
    )


def _counter(meta: dict[str, Any], key: str) -> int:
    value = meta.get(key)
    return value if type(value) is int and value >= 0 else 0
