# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Context variables shared between sub-agent event middleware and tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

SUB_AGENT_RESULT_COMMIT_CALLBACK_KEY = "_chrys_sub_agent_result_commit_callback"
SUB_AGENT_TRANSCRIPT_FINAL_TEXT_METADATA_KEY = "sub_agent_transcript_final_text"

# Carries the parent tool call id into the sub-agent tool body. Tool event
# middleware sets it before invoking KIND_SUB_AGENT tools; the sub-agent tool
# reads it when emitting InvocationStarted.
sub_agent_parent_call_id: ContextVar[str] = ContextVar("sub_agent_parent_call_id", default="")


@dataclass(slots=True)
class SubAgentParentResultMetadata:
    """Mutable parent-call metadata populated by sub-agent tool invocations."""

    parent_provider_call_id: str = ""
    parent_event_call_id: str = ""
    sub_agent_invocation_id: str = ""
    sub_agent_log_file: str = ""
    sub_agent_audit_complete: bool = False
    transcript_final_text: str | None = None
    interrupted_result_committed: bool = False
    commit_interrupted_result: Callable[[Mapping[str, Any]], None] | None = None


sub_agent_parent_result_metadata: ContextVar[SubAgentParentResultMetadata | None] = ContextVar(
    "sub_agent_parent_result_metadata",
    default=None,
)
