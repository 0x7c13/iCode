# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the session state serializers — message, compressed-block, and state round-trips."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chrys.kernel import Message
from chrys.service.agent_middleware.reminders.archive_pointer import CATALOG_POINTER_RECORD_COUNT_STATE_KEY
from chrys.service.context.providers.history import CompressedBlock
from chrys.service.session.message_metadata import stamp_message_created_at
from chrys.service.state.serializers import (
    deserialize_compressed_block,
    deserialize_message,
    deserialize_state,
    serialize_compressed_block,
    serialize_message,
    serialize_state,
)
from chrys.service.state.store import (
    _earliest_history_created_at,
    parse_snapshot_turn,
)


def test_parse_snapshot_turn_accepts_prefixed_and_legacy_numeric_names(tmp_path: Path) -> None:
    assert parse_snapshot_turn(tmp_path / "turn_12.json") == 12
    assert parse_snapshot_turn(tmp_path / "12.json") == 12
    assert parse_snapshot_turn(tmp_path / "turn_nope.json") == -1


def test_serialize_message_roundtrip() -> None:
    msg = Message("user", ["hello", "world"])
    msg.additional_properties["_key"] = "val"
    data = serialize_message(msg)
    restored = deserialize_message(data)
    assert restored.role == "user"
    assert [str(c) for c in restored.contents] == ["hello", "world"]
    assert restored.additional_properties["_key"] == "val"


def test_serialize_compressed_block_roundtrip() -> None:
    block = CompressedBlock(
        compressed_context_id="ctx_abc12345",
        messages=[Message("user", ["msg1"]), Message("assistant", ["msg2"])],
        summary_text="Did some work",
        marker_id="turn_2",
        turn_range=(1, 2),
        created_at="2026-03-17T00:00:00+00:00",
    )
    data = serialize_compressed_block(block)
    restored = deserialize_compressed_block(data)
    assert restored.compressed_context_id == "ctx_abc12345"
    assert restored.summary_text == "Did some work"
    assert restored.marker_id == "turn_2"
    assert restored.turn_range == (1, 2)
    assert restored.created_at == "2026-03-17T00:00:00+00:00"
    assert len(restored.messages) == 2


def test_serialize_state_roundtrip() -> None:
    state = {
        "messages": [Message("user", ["hi"]), Message("assistant", ["hello"])],
        "compressed_msgs": [
            CompressedBlock(
                compressed_context_id="ctx_001",
                messages=[Message("user", ["old"])],
                summary_text="summary",
                marker_id="turn_1",
                turn_range=(1, 1),
                created_at="2026-03-17T00:00:00+00:00",
            )
        ],
        "turn_counter": 3,
    }
    data = serialize_state(state)
    restored = deserialize_state(data)
    assert len(restored["messages"]) == 2
    assert len(restored["compressed_msgs"]) == 1
    assert restored["compressed_msgs"][0].compressed_context_id == "ctx_001"
    assert restored["turn_counter"] == 3


def test_earliest_history_created_at_supports_live_and_serialized_compressed_blocks() -> None:
    legacy = Message("user", ["legacy unstamped"])
    compressed = Message("user", ["old"])
    live = Message("user", ["new"])
    stamp_message_created_at(compressed, "2026-01-01T00:00:00+00:00")
    stamp_message_created_at(live, "2026-02-01T00:00:00+00:00")
    state = {
        "messages": [live],
        "compressed_msgs": [
            CompressedBlock(
                compressed_context_id="ctx_legacy",
                messages=[legacy],
                summary_text="legacy summary",
            ),
            CompressedBlock(
                compressed_context_id="ctx_001",
                messages=[compressed],
                summary_text="summary",
            ),
        ],
    }

    expected = datetime(2026, 1, 1, tzinfo=UTC)
    assert _earliest_history_created_at(state) == expected
    assert _earliest_history_created_at(serialize_state(state)) == expected


def test_serialize_state_roundtrip_with_chrys_mutations() -> None:
    """chrys_mutations data survives serialize/deserialize round-trip."""
    mutations_data = {
        "turns": [
            {
                "turn_id": 1,
                "mutations": [
                    {
                        "path": "/tmp/test.py",
                        "operation": "modify",
                        "source": "edit_file",
                        "tool_call_id": "call_001",
                        "timestamp": 1234567890.0,
                        "before_hash": "abc123",
                        "after_hash": "def456",
                    }
                ],
            }
        ],
        "snapshots": {
            "/tmp/test.py::1": {
                "path": "/tmp/test.py",
                "turn_id": 1,
                "existed": True,
                "content_hash": "abc123",
                "size": 42,
            }
        },
    }
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
        "chrys_mutations": mutations_data,
    }
    data = serialize_state(state)
    assert "chrys_mutations" in data

    restored = deserialize_state(data)
    assert "chrys_mutations" in restored
    assert restored["chrys_mutations"]["turns"][0]["turn_id"] == 1
    assert restored["chrys_mutations"]["turns"][0]["mutations"][0]["path"] == "/tmp/test.py"
    assert restored["chrys_mutations"]["snapshots"]["/tmp/test.py::1"]["content_hash"] == "abc123"


def test_serialize_state_round_trips_context_calibration() -> None:
    """The calibration record is allowlisted and copied (not aliased) both ways."""
    record = {
        "v": 2,
        "system_overhead_tokens": 7,
        "calibration_ratio": 1.2,
        "model_profile_fingerprint": "mfp",
        "agent_profile_fingerprint": "afp",
    }
    state = {"messages": [], "compressed_msgs": [], "context_calibration": record}

    data = serialize_state(state)
    assert data["context_calibration"] == record
    record["v"] = 999
    assert data["context_calibration"]["v"] == 2

    restored = deserialize_state(json.loads(json.dumps(data)))
    assert restored["context_calibration"]["calibration_ratio"] == 1.2


@pytest.mark.parametrize("malformed", ["not-a-dict", 42, ["v", 1], True])
@pytest.mark.parametrize("key", ["context_calibration", "last_usage"])
def test_serialize_state_drops_malformed_dict_valued_keys(key: str, malformed: object) -> None:
    """A corrupted dict-valued key degrades to "absent" instead of crashing the load."""
    state = {"messages": [], "compressed_msgs": [], key: malformed}
    data = serialize_state(state)
    assert key not in data

    restored = deserialize_state({"messages": [], "compressed_msgs": [], key: malformed})
    assert key not in restored


def test_serialize_state_without_chrys_mutations() -> None:
    """State without chrys_mutations serializes cleanly (no key added)."""
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
    }
    data = serialize_state(state)
    assert "chrys_mutations" not in data
    restored = deserialize_state(data)
    assert "chrys_mutations" not in restored


def test_serialize_state_roundtrip_with_last_words() -> None:
    """The Phase 4 LAST_WORDS note survives serialize/deserialize round-trip."""
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
        "last_words": "[LAST_WORDS] progress note for the interrupted turn",
    }
    data = serialize_state(state)
    assert data["last_words"] == "[LAST_WORDS] progress note for the interrupted turn"
    restored = deserialize_state(data)
    assert restored["last_words"] == "[LAST_WORDS] progress note for the interrupted turn"


def test_serialize_state_roundtrip_with_last_words_manifest_and_breaker() -> None:
    manifest = [{"record_id": "r1", "relative_path": "compactions/dropped/turn001/a.md"}]
    breaker = {
        "version": 1,
        "attempts": 3,
        "consecutive_no_progress": 1,
        "tail_override": True,
        "disabled": False,
        "side_call_tokens": 900,
    }
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
        "last_words_manifest": manifest,
        "last_words_breaker": breaker,
        CATALOG_POINTER_RECORD_COUNT_STATE_KEY: 0,
    }

    restored = deserialize_state(serialize_state(state))

    assert restored["last_words_manifest"] == manifest
    assert restored["last_words_breaker"] == breaker
    assert restored[CATALOG_POINTER_RECORD_COUNT_STATE_KEY] == 0


def test_serialize_state_without_last_words() -> None:
    """State without a LAST_WORDS note serializes cleanly (no key added)."""
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
    }
    data = serialize_state(state)
    assert "last_words" not in data
    restored = deserialize_state(data)
    assert "last_words" not in restored


def test_serialize_state_roundtrip_with_chrys_todos() -> None:
    """The session todo list survives serialize/deserialize round-trip."""
    todos = [
        {"content": "Read the plan", "status": "completed", "active_form": "Reading the plan"},
        {"content": "Implement", "status": "in_progress", "active_form": "Implementing"},
        {"content": "Test", "status": "pending", "active_form": ""},
    ]
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
        "chrys_todos": todos,
    }
    data = serialize_state(state)
    assert data["chrys_todos"] == todos
    restored = deserialize_state(data)
    assert restored["chrys_todos"] == todos


def test_serialize_state_drops_empty_chrys_todos() -> None:
    """An empty todo list is dropped (empty ≡ absent), and no key is invented."""
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
        "chrys_todos": [],
    }
    data = serialize_state(state)
    assert "chrys_todos" not in data
    restored = deserialize_state(data)
    assert "chrys_todos" not in restored

    without_key = serialize_state({"messages": [], "compressed_msgs": []})
    assert "chrys_todos" not in without_key


def test_serialize_state_optional_keys_preserve_truthy_only_behavior() -> None:
    """Optional metadata keys are copied from one declarative key table."""
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
        "last_usage": {},
        "agent_profile_switches": [],
        "chrys_mutations": {},
        "total_session_tokens": 0,
        "total_session_input_tokens": 0,
        "total_session_output_tokens": 0,
    }

    data = serialize_state(state)

    assert "last_usage" not in data
    assert "agent_profile_switches" not in data
    assert "chrys_mutations" not in data
    assert "total_session_tokens" not in data
    assert "total_session_input_tokens" not in data
    assert "total_session_output_tokens" not in data

    clean_baseline = {
        "version": 1,
        "turn_id": 4,
        "roots": {},
        "degraded": {},
        "root_limit_omitted": 0,
    }
    state["chrys_workspace_baseline"] = clean_baseline
    data = serialize_state(state)
    restored = deserialize_state(data)
    assert restored["chrys_workspace_baseline"] == clean_baseline

    state.pop("chrys_workspace_baseline")
    assert "chrys_workspace_baseline" not in serialize_state(state)

    state.update(
        {
            "last_usage": {"total_token_count": 11, "calibration_ratio": 1.2},
            "agent_profile_switches": [{"from": "Code", "to": "Explore"}],
            "chrys_mutations": {"turns": []},
            "total_session_tokens": 11,
            "total_session_input_tokens": 5,
            "total_session_output_tokens": 6,
        }
    )
    data = serialize_state(state)
    restored = deserialize_state(data)

    assert restored["last_usage"] == {"total_token_count": 11, "calibration_ratio": 1.2}
    assert restored["agent_profile_switches"] == [{"from": "Code", "to": "Explore"}]
    assert restored["chrys_mutations"] == {"turns": []}
    assert restored["total_session_tokens"] == 11
    assert restored["total_session_input_tokens"] == 5
    assert restored["total_session_output_tokens"] == 6
