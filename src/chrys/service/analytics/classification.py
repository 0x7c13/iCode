# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Deterministic classification of recorded tool evidence: action classes and tool outcomes."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Final

from chrys.foundation.trajectory.event_types import ToolOutcome
from chrys.service.analytics.model import ActionClass, Precision

_DIRECT_CLASSES = {
    "search": ActionClass.SEARCH,
    "filesystem.read": ActionClass.READ,
    "filesystem.write": ActionClass.EDIT,
}
_WORD_CHARACTER = r"A-Za-z0-9_"


def evidence_key(namespace: str, version: int, evidence: list[str] | tuple[str, ...]) -> str:
    """Hash one sorted rollback-stable evidence set into a persisted identity."""
    canonical = json.dumps(sorted(evidence), ensure_ascii=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    return f"{namespace}:v{version}:{digest}"


def classification_evidence_key(
    *,
    call_item_id: str | None,
    argument_fingerprint: str | None,
) -> str | None:
    """Build the stable per-action identity from rollback-stable call evidence."""
    stable = [
        value
        for value in (
            f"call:{call_item_id}" if call_item_id is not None else None,
            f"arguments:{argument_fingerprint}" if argument_fingerprint is not None else None,
        )
        if value is not None
    ]
    return evidence_key("action-classification", 1, stable) if stable else None


def parse_verify_commands(value: str) -> tuple[str, ...]:
    """Normalize a comma-separated verify word list without changing phrase order."""
    return tuple(dict.fromkeys(entry.strip().casefold() for entry in value.split(",") if entry.strip()))


def command_matches_verify(command: str, verify_commands: str | tuple[str, ...]) -> bool:
    """Match command words and multi-word phrases on shell-token boundaries."""
    entries = parse_verify_commands(verify_commands) if isinstance(verify_commands, str) else verify_commands
    for entry in entries:
        words = entry.split()
        if not words:
            continue
        phrase = r"\s+".join(re.escape(word) for word in words)
        pattern = rf"(?<![{_WORD_CHARACTER}]){phrase}(?![{_WORD_CHARACTER}])"
        if re.search(pattern, command, flags=re.IGNORECASE) is not None:
            return True
    return False


def classify_action(
    tool_kind: str | None,
    *,
    command: str | None,
    verify_commands: str | tuple[str, ...],
) -> tuple[ActionClass, Precision, str | None]:
    """Classify one tool without using names, timing, or outcome as evidence."""
    direct = _DIRECT_CLASSES.get(tool_kind or "")
    if direct is not None:
        return direct, Precision.EXACT, None
    if tool_kind != "shell":
        return ActionClass.OTHER, Precision.EXACT, None
    if command is None:
        return (
            ActionClass.OTHER,
            Precision.UNRESOLVED,
            "shell command carrier is unavailable; classification is heuristic-degraded",
        )
    if command_matches_verify(command, verify_commands):
        return ActionClass.VERIFY, Precision.ESTIMATED, "matched the configured verification word list"
    return ActionClass.OTHER, Precision.ESTIMATED, "did not match the configured verification word list"


KNOWN_TOOL_OUTCOMES: Final = frozenset(
    {
        ToolOutcome.SUCCESS,
        ToolOutcome.FAILED,
        ToolOutcome.ERRORED,
        ToolOutcome.INTERRUPTED,
        ToolOutcome.TIMED_OUT,
        ToolOutcome.REJECTED,
        ToolOutcome.INVALID_ARGUMENTS,
        ToolOutcome.UNKNOWN_TOOL,
        ToolOutcome.FILTERED,
        ToolOutcome.UNKNOWN,
    }
)
FAILED_TOOL_OUTCOMES: Final = KNOWN_TOOL_OUTCOMES - {
    ToolOutcome.SUCCESS,
    ToolOutcome.INTERRUPTED,
    ToolOutcome.UNKNOWN,
}


def tool_failed(outcome: str | None) -> bool:
    return outcome in FAILED_TOOL_OUTCOMES


def tool_succeeded(outcome: str | None) -> bool:
    return outcome == ToolOutcome.SUCCESS


__all__ = [
    "FAILED_TOOL_OUTCOMES",
    "KNOWN_TOOL_OUTCOMES",
    "classification_evidence_key",
    "classify_action",
    "command_matches_verify",
    "evidence_key",
    "parse_verify_commands",
    "tool_failed",
    "tool_succeeded",
]
