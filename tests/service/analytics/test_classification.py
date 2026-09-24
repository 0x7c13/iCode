# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Action-classification evidence and session-store join tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chrys.foundation.config.settings import DEFAULT_TRAJECTORY_VERIFY_COMMANDS
from chrys.foundation.trajectory.metadata import ANALYTICS_ITEM_ID_KEY
from chrys.service.analytics import ActionClass, Precision, analyze_trajectory
from chrys.service.analytics.classification import (
    classification_evidence_key,
    classify_action,
    command_matches_verify,
    parse_verify_commands,
)
from tests.service.analytics._events import EventLog

_NS = 1_000_000_000
_CALL_ITEM_ID = "7" * 32


@pytest.mark.parametrize(
    ("tool_kind", "expected"),
    [
        ("search", ActionClass.SEARCH),
        ("filesystem.read", ActionClass.READ),
        ("filesystem.write", ActionClass.EDIT),
        ("mcp", ActionClass.OTHER),
        (None, ActionClass.OTHER),
    ],
)
def test_structured_tool_kinds_map_directly_and_exactly(tool_kind: str | None, expected: ActionClass) -> None:
    action, precision, reason = classify_action(
        tool_kind,
        command=None,
        verify_commands=DEFAULT_TRAJECTORY_VERIFY_COMMANDS,
    )

    assert action is expected
    assert precision is Precision.EXACT
    assert reason is None


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("uv run pytest tests/service", True),
        ("npm test -- --runInBand", True),
        ("cargo test", True),
        ("ruff check src", True),
        ("cargo clippy --all-targets", True),
        ("./gradlew test", True),
        ("golangci-lint run ./...", True),
        ("npx vitest run", True),
        ("node --test test/", True),
        ("shellcheck install.sh", True),
        ("pytester --help", False),
        ("echo npm then test", False),
        ("cat Makefile", False),
        # Build/deploy verbs behind a launcher word are not verification even
        # though the launcher appears in verify phrases.
        ("cargo build --release", False),
        ("mvn deploy", False),
        ("gradle build", False),
        # Bare launcher/package-manager invocations are not verification; a
        # verify verb run through them still matches on its own word.
        ("uv sync --extra all", False),
        ("uv add requests", False),
    ],
)
def test_verify_word_list_uses_token_boundaries_and_multi_word_phrases(command: str, expected: bool) -> None:
    assert command_matches_verify(command, parse_verify_commands(DEFAULT_TRAJECTORY_VERIFY_COMMANDS)) is expected


def test_shell_classification_is_heuristic_and_missing_text_degrades_honestly() -> None:
    matched = classify_action("shell", command="pytest -q", verify_commands="pytest,npm test")
    unmatched = classify_action("shell", command="git status", verify_commands="pytest,npm test")
    absent = classify_action("shell", command=None, verify_commands="pytest,npm test")

    assert matched[:2] == (ActionClass.VERIFY, Precision.ESTIMATED)
    assert unmatched[:2] == (ActionClass.OTHER, Precision.ESTIMATED)
    assert absent[:2] == (ActionClass.OTHER, Precision.UNRESOLVED)
    assert "carrier" in (absent[2] or "")


def test_session_store_joins_content_level_call_identity_without_requiring_a_message_carrier(tmp_path: Path) -> None:
    path = _installed_log(tmp_path)
    _write_shell_log(path)
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [
                        {
                            "contents": [
                                {
                                    "type": "function_call",
                                    "arguments": json.dumps({"command": "npm test"}),
                                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: _CALL_ITEM_ID},
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    (action,) = analyze_trajectory(path, verify_commands="npm test").actions

    assert action.classification is ActionClass.VERIFY
    assert action.classification_precision is Precision.ESTIMATED


def test_session_store_join_accepts_mapping_valued_arguments(tmp_path: Path) -> None:
    """Anthropic blocking responses persist the tool input as the decoded
    mapping rather than a JSON string; the join must read both shapes."""
    path = _installed_log(tmp_path)
    _write_shell_log(path)
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [
                        {
                            "contents": [
                                {
                                    "type": "function_call",
                                    "arguments": {"command": "npm test"},
                                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: _CALL_ITEM_ID},
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    (action,) = analyze_trajectory(path, verify_commands="npm test").actions

    assert action.classification is ActionClass.VERIFY
    assert action.classification_precision is Precision.ESTIMATED


def test_compacted_messages_still_back_the_session_store_join(tmp_path: Path) -> None:
    """Compaction moves original messages into ``compressed_msgs[*].messages``
    and leaves only a summary live; the join must scan those blocks too."""
    path = _installed_log(tmp_path)
    _write_shell_log(path)
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [{"role": "user", "contents": [{"type": "text", "text": "summary"}]}],
                    "compressed_msgs": [
                        {
                            "compressed_context_id": "ctx-1",
                            "summary_text": "summary",
                            "turn_range": [1, 1],
                            "messages": [
                                {
                                    "contents": [
                                        {
                                            "type": "function_call",
                                            "arguments": json.dumps({"command": "npm test"}),
                                            "additional_properties": {ANALYTICS_ITEM_ID_KEY: _CALL_ITEM_ID},
                                        }
                                    ]
                                }
                            ],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    (action,) = analyze_trajectory(path, verify_commands="npm test").actions

    assert action.classification is ActionClass.VERIFY
    assert action.classification_precision is Precision.ESTIMATED


def test_absent_session_store_never_guesses_that_shell_work_was_verification(tmp_path: Path) -> None:
    path = _installed_log(tmp_path)
    _write_shell_log(path)

    analysis = analyze_trajectory(path, verify_commands="pytest")
    (action,) = analysis.actions

    assert action.classification is ActionClass.OTHER
    assert action.classification_precision is Precision.UNRESOLVED
    assert analysis.validation is not None
    assert analysis.validation.funnel.verify.precision is Precision.UNRESOLVED


def test_estimated_shell_classification_degrades_the_empty_verify_bucket(tmp_path: Path) -> None:
    """The word-list heuristic decides between verify and other, so a zero
    verify count is only as precise as the heuristic that produced it."""
    path = _installed_log(tmp_path)
    _write_shell_log(path)
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [
                        {
                            "contents": [
                                {
                                    "type": "function_call",
                                    "arguments": json.dumps({"command": "git status"}),
                                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: _CALL_ITEM_ID},
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_trajectory(path, verify_commands="pytest")
    (action,) = analysis.actions

    assert action.classification is ActionClass.OTHER
    assert action.classification_precision is Precision.ESTIMATED
    assert analysis.validation is not None
    verify = analysis.validation.funnel.verify
    assert (verify.value, verify.precision) == (0, Precision.ESTIMATED)


def test_nested_past_decoder_recursion_arguments_degrade_to_absent_evidence(tmp_path: Path) -> None:
    """An argument string can out-nest the decoder even when the outer session
    document decoded fine; it must degrade like undecodable arguments."""
    path = _installed_log(tmp_path)
    _write_shell_log(path)
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [
                        {
                            "contents": [
                                {
                                    "type": "function_call",
                                    "arguments": "[" * 100_000 + "]" * 100_000,
                                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: _CALL_ITEM_ID},
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    (action,) = analyze_trajectory(path, verify_commands="npm test").actions

    assert action.classification is ActionClass.OTHER
    assert action.classification_precision is Precision.UNRESOLVED


def test_oversized_integer_arguments_degrade_to_absent_evidence(tmp_path: Path) -> None:
    """An integer token past the digit limit raises a bare ValueError even
    when the outer session document decoded fine; it must degrade like
    undecodable arguments."""
    path = _installed_log(tmp_path)
    _write_shell_log(path)
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [
                        {
                            "contents": [
                                {
                                    "type": "function_call",
                                    "arguments": '{"n": ' + "1" * 5000 + "}",
                                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: _CALL_ITEM_ID},
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    (action,) = analyze_trajectory(path, verify_commands="npm test").actions

    assert action.classification is ActionClass.OTHER
    assert action.classification_precision is Precision.UNRESOLVED


def test_classification_identity_requires_stable_content_evidence() -> None:
    assert classification_evidence_key(call_item_id=None, argument_fingerprint=None) is None


def _installed_log(tmp_path: Path) -> Path:
    path = tmp_path / ".chrys" / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    return path


def _write_shell_log(path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span(
        "tool.operation",
        "8" * 32,
        _NS,
        2 * _NS,
        start_payload={
            "tool_name": "Bash",
            "tool_kind": "shell",
            "call_item_id": _CALL_ITEM_ID,
            "argument_fingerprint": "shell-fingerprint",
        },
    )
    log.add("turn.finished", 3 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)
