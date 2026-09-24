# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Lazy session-document projection: conflicting joins, undecodable documents, twin messages, and cache reuse."""

from __future__ import annotations

import json

import pytest

from chrys.foundation.trajectory.metadata import ANALYTICS_ITEM_ID_KEY
from chrys.service.analytics import _session_projection as session_projection_module
from tests.service.analytics._events import installed_events_path


@pytest.mark.parametrize("legacy", [False, True])
def test_chat_move_projection_reads_current_and_legacy_snapshot_indices(tmp_path, legacy: bool) -> None:
    from chrys.service.mutations.store import SnapshotStore
    from chrys.service.mutations.tracker import MutationTracker
    from chrys.service.mutations.types import MutationOp, MutationSource

    events = installed_events_path(tmp_path, session_id="fixture")
    tracker = MutationTracker(SnapshotStore(events.parents[1]))
    old, new = tmp_path / "old.txt", tmp_path / "new.txt"
    old.write_text("contents")
    tracker.start_turn(1)
    tracker.pre_snapshot([str(old), str(new)])
    old.rename(new)
    tracker.record(str(new), MutationOp.MOVE, MutationSource.SHELL, "mv", old_path=str(old))
    state = tracker.serialize()
    if legacy:
        for snapshot in state["snapshots"].values():
            snapshot["turn_id"] = snapshot.pop("period_index")
    events.parents[1].joinpath("session.json").write_text(json.dumps({"state": {"chrys_mutations": state}}))
    projection = session_projection_module._read_session_projection(events)
    assert projection.mutation_detail_available


def test_conflicting_session_carrier_mappings_are_not_treated_as_proof(tmp_path) -> None:
    content_id = "2" * 32
    path = installed_events_path(tmp_path, session_id="fixture")
    messages = [
        {
            "role": "tool",
            "additional_properties": {ANALYTICS_ITEM_ID_KEY: carrier_id},
            "contents": [
                {
                    "type": "function_result",
                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: content_id},
                }
            ],
        }
        for carrier_id in ("7" * 32, "8" * 32)
    ]
    path.parents[1].joinpath("session.json").write_text(
        json.dumps({"state": {"messages": messages}}),
        encoding="utf-8",
    )

    projection = session_projection_module._read_session_projection(path)

    assert content_id not in projection.carriers


def test_conflicting_session_token_counts_are_not_used_for_context_load(tmp_path) -> None:
    item_id = "7" * 32
    content_id = "2" * 32
    path = installed_events_path(tmp_path, session_id="fixture")
    messages = [
        {
            "role": "assistant",
            "additional_properties": {ANALYTICS_ITEM_ID_KEY: item_id, "_group": {"token_count": token_count}},
            "contents": [
                {
                    "type": "function_call",
                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: content_id},
                    "arguments": json.dumps(
                        {"command": f"command-{token_count}", "skill_name": f"skill-{token_count}"}
                    ),
                }
            ],
        }
        for token_count in (10, 20)
    ]
    path.parents[1].joinpath("session.json").write_text(
        json.dumps({"state": {"messages": messages}}),
        encoding="utf-8",
    )

    projection = session_projection_module._read_session_projection(path)

    assert item_id not in projection.item_tokens
    assert content_id not in projection.commands
    assert content_id not in projection.skill_names


@pytest.mark.parametrize(
    "document",
    [
        # json.load decodes the raw bytes itself, so invalid UTF-8 must degrade
        # exactly like invalid JSON instead of escaping as UnicodeDecodeError.
        pytest.param(b'{"state": {"messages": ["\xff"]}}', id="invalid_utf8"),
        # Valid JSON nested past the decoder's recursion budget raises
        # RecursionError rather than JSONDecodeError; it must degrade the same way.
        pytest.param(b"[" * 100_000 + b"]" * 100_000, id="recursion"),
        # An integer token past the digit limit raises a bare ValueError rather
        # than JSONDecodeError; it must degrade the same way.
        pytest.param(b'{"state": {"n": ' + b"1" * 5000 + b"}}", id="oversized_int"),
    ],
)
def test_undecodable_session_document_degrades_to_an_unavailable_projection(tmp_path, document: bytes) -> None:
    path = installed_events_path(tmp_path, session_id="fixture")
    path.parents[1].joinpath("session.json").write_bytes(document)

    projection = session_projection_module._read_session_projection(path)

    assert projection.available is False


def test_twin_messages_in_block_and_live_list_keep_single_tool_names_and_roles(tmp_path) -> None:
    """The session store may hold one message both in a compressed block and
    in the live list; the projection must not double its tool names."""
    path = installed_events_path(tmp_path)
    call_message = {
        "role": "assistant",
        "additional_properties": {ANALYTICS_ITEM_ID_KEY: "7" * 32},
        "contents": [
            {"type": "function_call", "call_id": "call-1", "name": "zsh", "arguments": "{}"},
            {"type": "function_call", "call_id": "call-2", "name": "zsh", "arguments": "{}"},
        ],
    }
    result_message = {
        "role": "tool",
        "additional_properties": {ANALYTICS_ITEM_ID_KEY: "6" * 32},
        "contents": [{"type": "function_result", "call_id": "call-1", "result": "ok"}],
    }
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [call_message, result_message],
                    "compressed_msgs": [
                        {
                            "compressed_context_id": "ctx-1",
                            "summary_text": "summary",
                            "turn_range": [1, 1],
                            "messages": [call_message, result_message],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    projection = session_projection_module._read_session_projection(path)

    # Genuine duplicate calls inside one message survive; the twin copies of
    # that message across transcripts collapse to one.
    assert projection.item_tool_names["7" * 32] == ("zsh", "zsh")
    assert projection.item_tool_names["6" * 32] == ("zsh",)
    assert projection.item_roles["7" * 32] == "assistant"
    assert projection.item_roles["6" * 32] == "tool"


def test_conflicting_carrier_tool_names_or_roles_across_transcripts_are_dropped(tmp_path) -> None:
    path = installed_events_path(tmp_path)

    def message(carrier: str, *, role: str, name: str | None) -> dict:
        contents = [] if name is None else [{"type": "function_call", "call_id": "c", "name": name, "arguments": "{}"}]
        return {"role": role, "additional_properties": {ANALYTICS_ITEM_ID_KEY: carrier}, "contents": contents}

    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [
                        message("7" * 32, role="assistant", name="python"),
                        message("6" * 32, role="tool", name="zsh"),
                        message("5" * 32, role="assistant", name=None),
                    ],
                    "compressed_msgs": [
                        {
                            "compressed_context_id": "ctx-1",
                            "summary_text": "summary",
                            "turn_range": [1, 1],
                            "messages": [
                                message("7" * 32, role="assistant", name="zsh"),
                                message("6" * 32, role="assistant", name="zsh"),
                                message("5" * 32, role="assistant", name="zsh"),
                            ],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    projection = session_projection_module._read_session_projection(path)

    # Same carrier with different names, roles, or call counts across
    # transcripts is not the same message; every conflicting join drops.
    assert "7" * 32 not in projection.item_tool_names
    assert "5" * 32 not in projection.item_tool_names
    assert "6" * 32 not in projection.item_roles
    assert projection.item_tool_names["6" * 32] == ("zsh",)
    assert projection.item_roles["7" * 32] == "assistant"


def test_session_projection_cache_reuses_the_unchanged_document(tmp_path) -> None:
    """The live dashboard refreshes twice a second; an unchanged session
    document must come back as the same parsed object, and a rewritten one
    must be reparsed."""
    path = installed_events_path(tmp_path / ".chrys")
    document = path.parents[1] / "session.json"
    document.write_text(json.dumps({"state": {"messages": []}}), encoding="utf-8")

    cache = session_projection_module._SessionProjectionCache()
    first = cache.lazy(path)()
    assert cache.lazy(path)() is first
    assert first.available and not first.mutation_detail_available

    document.write_text(
        json.dumps({"state": {"messages": [], "chrys_mutations": {"turns": []}}}),
        encoding="utf-8",
    )
    refreshed = cache.lazy(path)()
    assert refreshed is not first
    assert refreshed.mutation_detail_available
