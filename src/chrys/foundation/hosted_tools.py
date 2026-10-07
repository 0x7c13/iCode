# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Provider-neutral hosted-tool vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from chrys.foundation.tool_kinds import KIND_MCP, KIND_SEARCH, KIND_SHELL

PRESENTATION_TEXT_SEGMENT_ID_KEY: Final[str] = "_chrys_presentation_text_segment_id"
"""Private Content metadata key identifying one streamed output-text item."""

ANTHROPIC_HOSTED_WIRE_BLOCK_KEY: Final[str] = "anthropic.hosted_block"
OPENAI_HOSTED_WIRE_ITEM_KEY: Final[str] = "openai.responses.hosted_item"
HOSTED_WIRE_REPLAY_PROPERTY_KEYS: Final[frozenset[str]] = frozenset(
    {
        ANTHROPIC_HOSTED_WIRE_BLOCK_KEY,
        OPENAI_HOSTED_WIRE_ITEM_KEY,
    }
)
"""Exact provider-wire mirrors that must not be counted beside canonical content."""


class HostedToolFamily(StrEnum):
    """Canonical hosted-tool presentation families."""

    SEARCH = "search"
    FETCH = "fetch"
    MCP = "mcp"
    CODE = "code"
    IMAGE = "image"
    SHELL = "shell"
    TOOL_DISCOVERY = "tool_discovery"
    FILE_OPERATION = "file_operation"
    GENERIC = "generic"


class HostedToolPhase(StrEnum):
    """Canonical phases of a hosted-tool lifecycle update."""

    START = "start"
    DELTA = "delta"
    SNAPSHOT = "snapshot"
    TERMINAL = "terminal"


class HostedToolStatus(StrEnum):
    """Canonical hosted-tool lifecycle statuses."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class HostedRetrySafety(StrEnum):
    """Canonical retry-safety classes for provider-hosted work."""

    READ_ONLY = "read_only"
    SANDBOXED = "sandboxed"
    SIDE_EFFECTFUL = "side_effectful"
    UNKNOWN = "unknown"


HOSTED_TOOL_DEFAULT_KIND_BY_FAMILY: Final[dict[str, str]] = {
    HostedToolFamily.SEARCH: KIND_SEARCH,
    HostedToolFamily.FETCH: KIND_SEARCH,
    HostedToolFamily.MCP: KIND_MCP,
    HostedToolFamily.CODE: "",
    HostedToolFamily.IMAGE: "",
    HostedToolFamily.SHELL: KIND_SHELL,
    HostedToolFamily.TOOL_DISCOVERY: "",
    HostedToolFamily.FILE_OPERATION: "",
    HostedToolFamily.GENERIC: "",
}

HOSTED_TOOL_DEFAULT_TITLE_BY_FAMILY: Final[dict[str, str]] = {
    HostedToolFamily.SEARCH: "Hosted Search",
    HostedToolFamily.FETCH: "Hosted Fetch",
    HostedToolFamily.MCP: "Hosted MCP",
    HostedToolFamily.CODE: "Hosted Code",
    HostedToolFamily.IMAGE: "Hosted Image",
    HostedToolFamily.SHELL: "Hosted Shell",
    HostedToolFamily.TOOL_DISCOVERY: "Hosted Tool Discovery",
    HostedToolFamily.FILE_OPERATION: "Hosted File Operation",
    HostedToolFamily.GENERIC: "Hosted Tool",
}

HOSTED_TOOL_DEFAULT_RETRY_SAFETY_BY_FAMILY: Final[dict[str, HostedRetrySafety]] = {
    HostedToolFamily.SEARCH: HostedRetrySafety.READ_ONLY,
    HostedToolFamily.FETCH: HostedRetrySafety.READ_ONLY,
    HostedToolFamily.MCP: HostedRetrySafety.SIDE_EFFECTFUL,
    HostedToolFamily.CODE: HostedRetrySafety.SANDBOXED,
    HostedToolFamily.IMAGE: HostedRetrySafety.SANDBOXED,
    HostedToolFamily.SHELL: HostedRetrySafety.SIDE_EFFECTFUL,
    HostedToolFamily.TOOL_DISCOVERY: HostedRetrySafety.READ_ONLY,
    HostedToolFamily.FILE_OPERATION: HostedRetrySafety.SIDE_EFFECTFUL,
    HostedToolFamily.GENERIC: HostedRetrySafety.UNKNOWN,
}

_STATUS_ALIASES: Final[dict[str, HostedToolStatus]] = {
    "pending": HostedToolStatus.PENDING,
    "queued": HostedToolStatus.PENDING,
    "running": HostedToolStatus.RUNNING,
    "in_progress": HostedToolStatus.RUNNING,
    "searching": HostedToolStatus.RUNNING,
    "interpreting": HostedToolStatus.RUNNING,
    "generating": HostedToolStatus.RUNNING,
    "completed": HostedToolStatus.COMPLETED,
    "succeeded": HostedToolStatus.COMPLETED,
    "success": HostedToolStatus.COMPLETED,
    "failed": HostedToolStatus.FAILED,
    "error": HostedToolStatus.FAILED,
    "incomplete": HostedToolStatus.FAILED,
    "interrupted": HostedToolStatus.INTERRUPTED,
    "cancelled": HostedToolStatus.INTERRUPTED,
    "canceled": HostedToolStatus.INTERRUPTED,
}


def normalize_hosted_tool_status(status: str | None) -> HostedToolStatus:
    """Normalize a provider status to the canonical hosted-tool vocabulary."""
    if status is None:
        return HostedToolStatus.UNKNOWN
    return _STATUS_ALIASES.get(status.strip().lower(), HostedToolStatus.UNKNOWN)


@dataclass(frozen=True, slots=True)
class HeldHostedEvidence:
    """Hosted work a stream received but holds back to keep its place behind an unfinished call.

    It rides a contentless update as its ``raw_representation``, so the retry
    gates count the work as soon as it arrives. The contents themselves are
    sent, presented and replayed only when the adapter releases them in order.
    """

    contents: tuple[object, ...]
