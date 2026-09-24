# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the reload-recovery path: drain paused records + inject error tool-results.

On session reload, any sub-agent that was paused in the previous session
has a serialized record under ``{session_dir}/sub_agents/pending/*.json``
or the legacy ``{session_dir}/sub_agents/*.json`` path, but no live
controller to resume.  The engine's strategy is to:

1. :meth:`SubAgentSessionArtifactService.drain_paused_records` — load every valid record
2. :meth:`SessionHistoryManager.inject_error_results_for_sub_agents` —
   append a synthetic ``Error:`` tool-result for each dangling sub-agent
   function_call, matched to records by ``tool_name`` in appearance order
3. Publish a ``Warning`` so the user sees what was discarded (engine-level,
   not tested here)

These tests verify (1) and (2) in isolation — the engine integration is
exercised via existing session-restore tests.
"""

from __future__ import annotations

import json

import pytest

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.foundation.tool_kinds import KIND_SUB_AGENT, TOOL_CALL_KIND_METADATA_KEY
from chrys.kernel import ChatResponse, Content, Message
from chrys.service.session.history import SessionHistoryManager
from chrys.service.session.message_metadata import TOOL_RESULT_METADATA_KEY
from chrys.service.session.sub_agent_logs import (
    SUB_AGENT_RESTORE_CONSUMED_KEY,
    SubAgentSessionArtifactService,
)
from tests.support.transcript_invariants import assert_transcript_invariants

# --- SubAgentSessionArtifactService.drain_paused_records ----------------


def _write_record(root, invocation_id: str, *, tool_name: str, last_error: str = "boom") -> None:
    sub_dir = root / "sub_agents"
    sub_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_owner_only_text(
        sub_dir / f"{invocation_id}.json",
        json.dumps(
            {
                "schema_version": 1,
                "invocation_id": invocation_id,
                "tool_name": tool_name,
                "agent_name": tool_name,
                "prompt": "whatever",
                "session_id": "s-1",
                "failure_reason": "framework_exc",
                "last_error": last_error,
                "retry_attempts_total": 1,
            }
        ),
    )


def _write_pending_record(root, invocation_id: str, *, tool_name: str, created_at: str = "") -> None:
    sub_dir = root / "sub_agents" / "pending"
    sub_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "record_type": "sub_agent_pending",
        "invocation_id": invocation_id,
        "tool_name": tool_name,
        "agent_name": tool_name,
        "prompt_preview": "whatever",
        "session_id": "s-1",
        "parent_call_id": f"provider-{invocation_id}",
        "parent_provider_call_id": f"provider-{invocation_id}",
        "parent_event_call_id": f"event-{invocation_id}",
        "failure_reason": "framework_exc",
        "last_error": "boom",
        "retry_attempts_total": 1,
    }
    if created_at:
        data["created_at"] = created_at
        data["paused_at"] = created_at
    atomic_write_owner_only_text(sub_dir / f"{invocation_id}.json", json.dumps(data))


def _write_audit_log(root, invocation_id: str, *, status: str = "running", filename: str | None = None) -> str:
    log_file = filename or f"Explore_{invocation_id}.json"
    sub_dir = root / "sub_agents" / "sessions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_owner_only_text(
        sub_dir / log_file,
        json.dumps(
            {
                "meta": {
                    "record_type": "sub_agent_session",
                    "invocation_id": invocation_id,
                    "tool_name": "Explore",
                    "status": status,
                    "parent_session_id": "s-1",
                },
                "state": {"messages": []},
            }
        ),
    )
    return log_file


def test_drain_empty_dir_returns_empty_list(tmp_path):
    tools = SubAgentSessionArtifactService(tmp_path)
    assert tools.drain_paused_records() == []


def test_drain_loads_records_without_deleting_until_finalize(tmp_path):
    _write_record(tmp_path, "a", tool_name="Explore")
    _write_record(tmp_path, "b", tool_name="Plan", last_error="context overflow")

    tools = SubAgentSessionArtifactService(tmp_path)
    records = tools.drain_paused_records()

    assert {r["invocation_id"] for r in records} == {"a", "b"}
    assert sorted(path.name for path in (tmp_path / "sub_agents").glob("*.json")) == ["a.json", "b.json"]

    for record in records:
        record[SUB_AGENT_RESTORE_CONSUMED_KEY] = True
    tools.finalize_restored_paused_records(records)

    assert list((tmp_path / "sub_agents").glob("*.json")) == []


def test_finalize_restored_records_archives_unconsumed_ambiguous_records(tmp_path):
    pending_dir = tmp_path / "sub_agents" / "pending"
    pending_dir.mkdir(parents=True)
    for invocation_id in ("first", "second"):
        data = {
            "schema_version": 1,
            "record_type": "sub_agent_pending",
            "invocation_id": invocation_id,
            "tool_name": "Explore",
            "parent_provider_call_id": "provider-call",
            "last_error": f"{invocation_id}-error",
        }
        atomic_write_owner_only_text(pending_dir / f"{invocation_id}.json", json.dumps(data))

    tools = SubAgentSessionArtifactService(tmp_path)
    records = tools.drain_paused_records()
    h = _history_with_dangling_call("Explore", "provider-call")

    injected = h.inject_error_results_for_sub_agents(records, {"Explore"})
    tools.finalize_restored_paused_records(records)

    assert injected == 1
    assert list(pending_dir.glob("*.json")) == []
    assert sorted(path.name for path in (pending_dir / "unmatched").glob("*.json")) == ["first.json", "second.json"]
    assert SUB_AGENT_RESTORE_CONSUMED_KEY not in records[0]
    assert SUB_AGENT_RESTORE_CONSUMED_KEY not in records[1]


def test_archive_unconsumed_restored_records_moves_stale_without_save(tmp_path):
    _write_pending_record(tmp_path, "stale", tool_name="Explore")
    tools = SubAgentSessionArtifactService(tmp_path)
    records = tools.drain_paused_records()
    h = _history_with_dangling_call("Explore", "provider-stale")
    h.messages.append(Message("tool", [Content.from_function_result(call_id="provider-stale", result="done")]))

    injected = h.inject_error_results_for_sub_agents(records, {"Explore"})
    tools.archive_unconsumed_restored_paused_records(records)

    assert injected == 0
    assert not (tmp_path / "sub_agents" / "pending" / "stale.json").exists()
    assert (tmp_path / "sub_agents" / "pending" / "unmatched" / "stale.json").exists()


def test_reconcile_orphaned_running_logs_marks_unowned_running_log(tmp_path):
    _write_audit_log(tmp_path, "a1b2c3d4e5f6")
    tools = SubAgentSessionArtifactService(tmp_path)

    reconciled = tools.reconcile_orphaned_running_logs([Message("user", ["go"])], [])

    assert reconciled == 1
    data = json.loads((tmp_path / "sub_agents" / "sessions" / "Explore_a1b2c3d4e5f6.json").read_text(encoding="utf-8"))
    assert data["meta"]["status"] == "orphaned"
    assert data["meta"]["failure_reason"] == "process_terminated"


def test_reconcile_orphaned_running_logs_skips_parent_backlink(tmp_path):
    log_file = _write_audit_log(tmp_path, "a1b2c3d4e5f6")
    result = Content.from_function_result(call_id="parent-call", result="done")
    result.additional_properties[TOOL_RESULT_METADATA_KEY] = {
        "sub_agent_invocation_id": "a1b2c3d4e5f6",
        "sub_agent_log_file": log_file,
    }
    tools = SubAgentSessionArtifactService(tmp_path)

    reconciled = tools.reconcile_orphaned_running_logs([Message("tool", [result])], [])

    assert reconciled == 0
    data = json.loads((tmp_path / "sub_agents" / "sessions" / log_file).read_text(encoding="utf-8"))
    assert data["meta"]["status"] == "running"


def test_drain_reads_new_pending_and_legacy_records_without_sessions(tmp_path):
    _write_record(tmp_path, "legacy", tool_name="Explore")
    _write_pending_record(tmp_path, "pending", tool_name="Plan", created_at="2026-06-25T00:00:00+00:00")
    sessions_dir = tmp_path / "sub_agents" / "sessions"
    sessions_dir.mkdir(parents=True)
    atomic_write_owner_only_text(
        sessions_dir / "Explore_ignored.json",
        json.dumps({"meta": {"invocation_id": "ignored"}}),
    )

    tools = SubAgentSessionArtifactService(tmp_path)
    records = tools.drain_paused_records()

    assert [r["invocation_id"] for r in records] == ["legacy", "pending"]
    assert (sessions_dir / "Explore_ignored.json").exists()
    assert (tmp_path / "sub_agents" / "legacy.json").exists()
    assert (tmp_path / "sub_agents" / "pending" / "pending.json").exists()


def test_drain_enriches_missing_sub_agent_log_file_from_audit_log(tmp_path):
    _write_pending_record(tmp_path, "a1b2c3d4e5f6", tool_name="Explore")
    sessions_dir = tmp_path / "sub_agents" / "sessions"
    sessions_dir.mkdir(parents=True)
    log_file = "Explore_a1b2c3d4e5f6.json"
    atomic_write_owner_only_text(
        sessions_dir / log_file,
        json.dumps({"meta": {"record_type": "sub_agent_session", "invocation_id": "a1b2c3d4e5f6"}}),
    )
    tools = SubAgentSessionArtifactService(tmp_path)

    records = tools.drain_paused_records()
    h = _history_with_dangling_call("Explore", "provider-a1b2c3d4e5f6")
    injected = h.inject_error_results_for_sub_agents(records, {"Explore"})

    assert injected == 1
    metadata = h.messages[2].contents[0].additional_properties["_chrys_tool_result_metadata"]
    assert metadata == {
        "sub_agent_invocation_id": "a1b2c3d4e5f6",
        "sub_agent_log_file": log_file,
    }


def test_drain_skips_and_quarantines_corrupt_files(tmp_path):
    sub_dir = tmp_path / "sub_agents"
    sub_dir.mkdir()
    atomic_write_owner_only_text(sub_dir / "broken.json", "{not valid")
    _write_record(tmp_path, "ok", tool_name="Explore")

    tools = SubAgentSessionArtifactService(tmp_path)
    records = tools.drain_paused_records()

    assert [r["invocation_id"] for r in records] == ["ok"]
    assert sorted(path.name for path in sub_dir.glob("*.json")) == ["ok.json"]
    quarantined = list((tmp_path / "sub_agents" / "pending" / "corrupt").glob("broken*.json"))
    assert len(quarantined) == 1


def test_drain_quarantines_non_object_json_records(tmp_path):
    sub_dir = tmp_path / "sub_agents" / "pending"
    sub_dir.mkdir(parents=True)
    atomic_write_owner_only_text(sub_dir / "array.json", "[]")
    _write_pending_record(tmp_path, "ok", tool_name="Explore")

    tools = SubAgentSessionArtifactService(tmp_path)
    records = tools.drain_paused_records()

    assert [r["invocation_id"] for r in records] == ["ok"]
    assert not (sub_dir / "array.json").exists()
    quarantined = list((tmp_path / "sub_agents" / "pending" / "corrupt").glob("array*.json"))
    assert len(quarantined) == 1


# --- SessionHistoryManager.inject_error_results_for_sub_agents ----------


def _history_with_dangling_call(tool_name: str, call_id: str) -> SessionHistoryManager:
    h = SessionHistoryManager()
    assistant = Message(
        "assistant",
        [Content.from_function_call(call_id, tool_name, arguments={})],
    )
    h.bind({"messages": [Message("user", ["go"]), assistant]})
    return h


def test_inject_appends_error_result_matched_by_tool_name(tmp_path):
    h = _history_with_dangling_call("Explore", "call-1")
    records = [
        {"tool_name": "Explore", "last_error": "whoops"},
    ]
    injected = h.inject_error_results_for_sub_agents(records, {"Explore"})

    assert injected == 1
    assert len(h.messages) == 3
    tool_msg = h.messages[2]
    assert tool_msg.role == "tool"
    result = tool_msg.contents[0]
    assert result.type == "function_result"
    assert result.call_id == "call-1"
    assert "Error: sub-agent 'Explore'" in result.result
    assert "whoops" in result.result


def test_inject_neutralizes_surrogate_last_error_for_provider_encode(tmp_path):
    """A restored pending record can carry a lone surrogate (round-tripped from an
    owner-valid audit whose `\\udcXX` escape json.loads decodes back to a surrogate).
    The injected result is provider-visible repaired history, so it must stay
    strict-UTF-8 encodable — the LLM request serializer's strict encode would
    otherwise crash every subsequent turn on the restored session."""
    h = _history_with_dangling_call("Explore", "call-1")
    records = [
        {"tool_name": "Explore", "last_error": "byte \udcff at /x/\udcfe"},
    ]
    injected = h.inject_error_results_for_sub_agents(records, {"Explore"})

    assert injected == 1
    result = h.messages[2].contents[0].result
    # Discriminator: a strict encode (what the provider request serializer does)
    # must not raise, and no lone surrogate survives into the provider-visible string.
    result.encode("utf-8")
    assert "\udcff" not in result
    assert "\udcfe" not in result
    assert "Error: sub-agent 'Explore'" in result


def test_inject_handles_dangling_call_with_no_matching_record(tmp_path):
    """Dangling sub-agent call with no persisted record still gets a generic error."""
    h = _history_with_dangling_call("Explore", "call-1")
    injected = h.inject_error_results_for_sub_agents([], {"Explore"})
    assert injected == 1
    result = h.messages[2].contents[0]
    assert "state discarded on reload" in result.result
    assert "—" not in result.result  # no last_error suffix


def test_inject_skips_already_resolved_calls(tmp_path):
    """If a function_call already has a tool_result (even an earlier error),
    we must not duplicate."""
    h = _history_with_dangling_call("Explore", "call-1")
    # Append an existing result so the call isn't dangling.
    h.messages.append(Message("tool", [Content.from_function_result(call_id="call-1", result="done")]))

    injected = h.inject_error_results_for_sub_agents([{"tool_name": "Explore", "last_error": "stale"}], {"Explore"})
    assert injected == 0


def test_inject_previous_duplicate_call_id_does_not_resolve_current_call(tmp_path):
    """An older tool result with a reused call_id must not hide a dangling current call."""
    h = SessionHistoryManager()
    old_call = Message(
        "assistant",
        [Content.from_function_call("shared", "Explore", arguments={})],
    )
    old_result = Message("tool", [Content.from_function_result(call_id="shared", result="old done")])
    current_call = Message(
        "assistant",
        [Content.from_function_call("shared", "Explore", arguments={})],
    )
    h.bind(
        {
            "messages": [
                Message("user", ["old"]),
                old_call,
                old_result,
                Message("user", ["current"]),
                current_call,
            ]
        }
    )

    injected = h.inject_error_results_for_sub_agents([{"tool_name": "Explore", "last_error": "stale"}], {"Explore"})

    assert injected == 1
    assert h.messages[5].role == "tool"
    result = h.messages[5].contents[0]
    assert result.call_id == "shared"
    assert "stale" in result.result


def test_inject_skips_non_sub_agent_tools(tmp_path):
    """Dangling calls for non-sub-agent tools are left alone (trim path handles them)."""
    h = _history_with_dangling_call("write_file", "call-1")
    injected = h.inject_error_results_for_sub_agents([{"tool_name": "write_file", "last_error": "x"}], {"Explore"})
    assert injected == 0
    # History unchanged.
    assert len(h.messages) == 2


def test_inject_pairs_multiple_records_by_order(tmp_path):
    """Two dangling Explore calls + two records → paired in appearance order."""
    h = SessionHistoryManager()
    first = Message(
        "assistant",
        [Content.from_function_call("c1", "Explore", arguments={})],
    )
    second = Message(
        "assistant",
        [Content.from_function_call("c2", "Explore", arguments={})],
    )
    h.bind({"messages": [Message("user", ["go"]), first, second]})

    records = [
        {"tool_name": "Explore", "last_error": "first-error"},
        {"tool_name": "Explore", "last_error": "second-error"},
    ]
    injected = h.inject_error_results_for_sub_agents(records, {"Explore"})

    assert injected == 2
    # Look up the injected tool messages by their resolved call_id.
    tool_msgs = [m for m in h.messages if m.role == "tool"]
    assert len(tool_msgs) == 2
    by_id = {m.contents[0].call_id: m.contents[0].result for m in tool_msgs}
    assert "first-error" in by_id["c1"]
    assert "second-error" in by_id["c2"]


def test_inject_handles_multiple_calls_on_one_assistant_message(tmp_path):
    """Two parallel sub-agent calls in one assistant message → two tool results
    inserted consecutively right after the assistant message."""
    h = SessionHistoryManager()
    parallel = Message(
        "assistant",
        [
            Content.from_function_call("p1", "Explore", arguments={}),
            Content.from_function_call("p2", "Explore", arguments={}),
        ],
    )
    h.bind({"messages": [Message("user", ["go"]), parallel]})

    injected = h.inject_error_results_for_sub_agents(
        [
            {"tool_name": "Explore", "last_error": "err-1"},
            {"tool_name": "Explore", "last_error": "err-2"},
        ],
        {"Explore"},
    )
    assert injected == 2
    # Assistant at idx 1, tool results at idx 2 and 3.
    assert h.messages[2].role == "tool"
    assert h.messages[3].role == "tool"
    assert h.messages[2].contents[0].call_id == "p1"
    assert h.messages[3].contents[0].call_id == "p2"


def test_inject_pairs_by_parent_call_id_when_present(tmp_path):
    """Records carrying ``parent_call_id`` pair by id, not by order.

    Two concurrent Explore calls (p1, p2) paused in the order (p2, p1)
    — i.e. the filesystem/drain order differs from the function_call
    appearance order.  Without parent_call_id the name+order fallback
    would cross-match; with parent_call_id each record lands on its
    true call.
    """
    h = SessionHistoryManager()
    parallel = Message(
        "assistant",
        [
            Content.from_function_call("p1", "Explore", arguments={}),
            Content.from_function_call("p2", "Explore", arguments={}),
        ],
    )
    h.bind({"messages": [Message("user", ["go"]), parallel]})

    # Records arrive in REVERSE order (p2 first, p1 second).  The v1
    # policy would give p1→p2-error and p2→p1-error.
    records = [
        {"tool_name": "Explore", "parent_call_id": "p2", "last_error": "err-for-p2"},
        {"tool_name": "Explore", "parent_call_id": "p1", "last_error": "err-for-p1"},
    ]
    injected = h.inject_error_results_for_sub_agents(records, {"Explore"})

    assert injected == 2
    by_id = {m.contents[0].call_id: m.contents[0].result for m in h.messages if m.role == "tool"}
    assert "err-for-p1" in by_id["p1"]
    assert "err-for-p2" in by_id["p2"]


def test_inject_prefers_parent_provider_call_id_and_adds_backlink_metadata(tmp_path):
    h = _history_with_dangling_call("Explore", "provider-call")
    records = [
        {
            "tool_name": "Explore",
            "parent_call_id": "event-call",
            "parent_provider_call_id": "provider-call",
            "invocation_id": "a1b2c3d4e5f6",
            "sub_agent_log_file": "Explore_a1b2c3d4e5f6.json",
            "last_error": "paused",
        }
    ]

    injected = h.inject_error_results_for_sub_agents(records, {"Explore"})

    assert injected == 1
    metadata = h.messages[2].contents[0].additional_properties["_chrys_tool_result_metadata"]
    assert metadata == {
        "sub_agent_invocation_id": "a1b2c3d4e5f6",
        "sub_agent_log_file": "Explore_a1b2c3d4e5f6.json",
    }
    assert records[0][SUB_AGENT_RESTORE_CONSUMED_KEY] is True


def test_inject_duplicate_parent_provider_call_id_is_ambiguous_and_generic(tmp_path):
    h = _history_with_dangling_call("Explore", "provider-call")
    records = [
        {
            "tool_name": "Explore",
            "parent_provider_call_id": "provider-call",
            "invocation_id": "first",
            "last_error": "first-error",
        },
        {
            "tool_name": "Explore",
            "parent_provider_call_id": "provider-call",
            "invocation_id": "second",
            "last_error": "second-error",
        },
    ]

    injected = h.inject_error_results_for_sub_agents(records, {"Explore"})

    assert injected == 1
    result = h.messages[2].contents[0]
    assert (
        result.result == "Error: sub-agent 'Explore' was paused when the session was saved; state discarded on reload."
    )
    assert "_chrys_tool_result_metadata" not in result.additional_properties
    assert SUB_AGENT_RESTORE_CONSUMED_KEY not in records[0]
    assert SUB_AGENT_RESTORE_CONSUMED_KEY not in records[1]


def test_inject_uses_record_tool_name_for_removed_sub_agent(tmp_path):
    h = _history_with_dangling_call("OldExplore", "call-1")
    records = [{"invocation_id": "old-inv", "tool_name": "OldExplore", "last_error": "profile removed"}]

    injected = h.inject_error_results_for_sub_agents(records, set())

    assert injected == 1
    result = h.messages[2].contents[0]
    assert "profile removed" in result.result
    assert records[0][SUB_AGENT_RESTORE_CONSUMED_KEY] is True


def test_inject_mixed_records_with_and_without_parent_call_id(tmp_path):
    """A record with ``parent_call_id`` and one without share history —
    the former claims its specific call, the latter falls back to
    name+order for whatever remains."""
    h = SessionHistoryManager()
    parallel = Message(
        "assistant",
        [
            Content.from_function_call("c-alpha", "Explore", arguments={}),
            Content.from_function_call("c-beta", "Explore", arguments={}),
        ],
    )
    h.bind({"messages": [Message("user", ["go"]), parallel]})

    records = [
        # No parent_call_id → name+order fallback pool
        {"tool_name": "Explore", "last_error": "legacy-err"},
        # parent_call_id targets c-beta explicitly
        {"tool_name": "Explore", "parent_call_id": "c-beta", "last_error": "targeted-err"},
    ]
    injected = h.inject_error_results_for_sub_agents(records, {"Explore"})

    assert injected == 2
    by_id = {m.contents[0].call_id: m.contents[0].result for m in h.messages if m.role == "tool"}
    # Targeted record pinned c-beta; fallback record flows to remaining c-alpha.
    assert "targeted-err" in by_id["c-beta"]
    assert "legacy-err" in by_id["c-alpha"]


def test_inject_unmatched_parent_call_id_falls_through(tmp_path):
    """If a record's parent_call_id doesn't match any dangling call,
    it should NOT be silently dropped — the name-order fallback can't
    claim it either (distinct pools), so it's discarded.  The dangling
    call gets the generic error.  This keeps behavior safe when a
    profile is deleted and the dangling call has no matching record."""
    h = _history_with_dangling_call("Explore", "real-call")
    records = [
        # parent_call_id points at a call that no longer exists in history
        {"tool_name": "Explore", "parent_call_id": "ghost-call", "last_error": "ignored"},
    ]
    injected = h.inject_error_results_for_sub_agents(records, {"Explore"})

    assert injected == 1
    # Generic error — the ghost record didn't match and didn't fall
    # through to the name pool.
    result = h.messages[2].contents[0].result
    assert "state discarded on reload" in result
    assert "ignored" not in result


def test_inject_no_op_when_empty_tool_name_set(tmp_path):
    h = _history_with_dangling_call("Explore", "call-1")
    injected = h.inject_error_results_for_sub_agents([], set())
    assert injected == 0


def test_inject_repairs_dangling_call_from_registry_with_no_records(tmp_path):
    """A dangling sub-agent call is repaired from the live tool registry even
    when drain returned ZERO records (the over-cap/corrupt drop path).

    This is the mechanism the restore handler relies on: it now passes the
    registry tool names and runs injection even when ``paused_records`` is
    empty, so a dropped pending record can no longer leave the parent call
    without a tool_result (which the provider would reject on the next turn).
    """
    h = _history_with_dangling_call("Explore", "call-1")
    injected = h.inject_error_results_for_sub_agents([], {"Explore"})
    assert injected == 1
    result = h.messages[2].contents[0].result
    assert "state discarded on reload" in result


def test_inject_repairs_kind_marked_call_with_empty_records_and_registry(tmp_path):
    """A persisted sub-agent call is repaired by its OWN _chrys_tool_kind marker
    even when BOTH the drained records AND the live registry are empty.

    Failure mode this pins: drain lost every record (over-cap/corrupt) AND the
    restored profile no longer registers the tool (removed/disabled/depth-skip),
    so name-based matching finds nothing. Without kind-based detection the
    dangling call survives and the next provider request rejects the history.
    """
    h = SessionHistoryManager()
    call = Content.from_function_call("call-1", "RetiredAgent", arguments={})
    call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] = KIND_SUB_AGENT
    h.bind({"messages": [Message("user", ["go"]), Message("assistant", [call])]})

    # Empty records AND empty registry — only the kind marker identifies it.
    injected = h.inject_error_results_for_sub_agents([], set())

    assert injected == 1
    result = h.messages[2].contents[0]
    assert result.type == "function_result"
    assert result.call_id == "call-1"
    assert "state discarded on reload" in result.result


def test_inject_ignores_non_sub_agent_call_with_empty_candidates(tmp_path):
    """A plain (non-sub-agent) dangling call is NOT repaired when records and
    registry are empty — kind-based detection must not over-inject."""
    h = SessionHistoryManager()
    call = Content.from_function_call("call-1", "write_file", arguments={})
    call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] = "filesystem.write"
    h.bind({"messages": [Message("user", ["go"]), Message("assistant", [call])]})

    injected = h.inject_error_results_for_sub_agents([], set())

    assert injected == 0
    assert len(h.messages) == 2  # no tool result appended


class _SliceSpyList(list):
    """A list that counts slice reads, to prove the inject scan is O(N) not Θ(N²)."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.slice_reads = 0

    def __getitem__(self, item: object) -> object:
        if isinstance(item, slice):
            self.slice_reads += 1
        return super().__getitem__(item)


def test_inject_scan_never_slices_history_tail(tmp_path):
    """The dangling-call scan must never copy the message tail.

    ``_immediate_tool_result_call_ids`` used to do ``messages[assistant_idx+1:]``
    once per assistant across two passes — Θ(N²) copying. Since r7 lifecycle
    calls inject on EVERY restore (even sessions with no dangling sub-agent
    calls), that quadratic runs every time. Pin it: zero slice reads over the
    whole scan of a fully-resolved alternating history.
    """
    msgs = _SliceSpyList()
    for k in range(50):
        msgs.append(Message("user", ["hi"]))
        msgs.append(Message("assistant", [Content.from_function_call(f"c{k}", "Explore", arguments={})]))
        msgs.append(Message("tool", [Content.from_function_result(call_id=f"c{k}", result="ok")]))
    h = SessionHistoryManager()
    h.bind({"messages": msgs})
    msgs.slice_reads = 0  # measure only the inject scan, not any binding-time access

    injected = h.inject_error_results_for_sub_agents([], {"Explore"})

    assert injected == 0  # every call is already resolved
    assert msgs.slice_reads == 0  # no tail-slice copies anywhere in the scan


def test_inject_idempotent_on_second_call(tmp_path):
    """Calling ``inject_error_results_for_sub_agents`` twice with the
    same records must not double-inject — the second pass should see
    the first-pass results as already-resolved and skip."""
    h = _history_with_dangling_call("Explore", "call-1")
    records = [{"tool_name": "Explore", "last_error": "err"}]

    first = h.inject_error_results_for_sub_agents(records, {"Explore"})
    assert first == 1
    before = len(h.messages)

    # Second call with the SAME records — call_id is now in resolved set.
    second = h.inject_error_results_for_sub_agents(records, {"Explore"})
    assert second == 0
    assert len(h.messages) == before


def test_inject_skips_call_resolved_in_combined_block_behind_consecutive_assistant(tmp_path):
    """A call answered in a shared result block behind another assistant is NOT dangling.

    The kernel folds all of a response's tool results into ONE tool message
    appended after the response's complete message list, so a multi-message
    response persists as ``[assistant, assistant, tool]``. An
    immediate-successor scan breaks at the second assistant message, counts
    the first message's answered call as dangling, and injects a DUPLICATE
    result for it — the resolved-call scan must read the block after the
    whole consecutive assistant run.
    """
    h = SessionHistoryManager()
    sub_call = Content.from_function_call("call-1", "Explore", arguments={})
    other_call = Content.from_function_call("call-2", "zsh", arguments={})
    h.bind(
        {
            "messages": [
                Message("user", ["go"]),
                Message("assistant", [sub_call]),
                Message("assistant", [other_call]),
                Message(
                    "tool",
                    [
                        Content.from_function_result(call_id="call-1", result="explored"),
                        Content.from_function_result(call_id="call-2", result="ok"),
                    ],
                ),
            ]
        }
    )

    injected = h.inject_error_results_for_sub_agents([], {"Explore"})

    assert injected == 0
    assert len(h.messages) == 4  # no duplicate result appended


def test_inject_places_repair_after_consecutive_assistant_siblings() -> None:
    """A synthetic result must not split a multi-message response."""
    h = SessionHistoryManager()
    call_message = Message(
        "assistant",
        [Content.from_function_call("call-1", "Explore", arguments={})],
    )
    text_message = Message("assistant", [Content.from_text("pre-tool text")])
    user_message = Message("user", ["go"])
    h.bind({"messages": [user_message, call_message, text_message]})

    injected = h.inject_error_results_for_sub_agents([], {"Explore"})

    assert injected == 1
    assert h.messages[:3] == [user_message, call_message, text_message]
    assert h.messages[3].role == "tool"
    assert h.messages[3].contents[0].call_id == "call-1"


def test_inject_deduplicates_same_call_id_across_assistant_siblings() -> None:
    """One shared-block result answers every shadowed same-id call occurrence."""
    h = SessionHistoryManager()
    first_call = Content.from_function_call("dup", "Explore", arguments={})
    first_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    second_call = Content.from_function_call("dup", "Plan", arguments={})
    second_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 1
    messages = [
        Message("user", ["delegate"]),
        Message("assistant", [first_call]),
        Message("assistant", [second_call]),
    ]
    h.bind({"messages": messages})
    record = {
        "tool_name": "Explore",
        "parent_provider_call_id": "dup",
        "invocation_id": "inv-1",
        "last_error": "paused",
    }
    assert_transcript_invariants(ChatResponse(messages=list(messages)))

    injected = h.inject_error_results_for_sub_agents([record], {"Explore", "Plan"})

    assert injected == 1
    results = [
        content
        for message in h.messages
        if message.role == "tool"
        for content in message.contents
        if content.type == "function_result"
    ]
    assert [result.call_id for result in results] == ["dup"]
    assert results[0].additional_properties[TOOL_RESULT_METADATA_KEY] == {"sub_agent_invocation_id": "inv-1"}
    assert record[SUB_AGENT_RESTORE_CONSUMED_KEY] is True
    assert_transcript_invariants(ChatResponse(messages=list(h.messages)))


def test_inject_places_repair_before_history_markers() -> None:
    """A synthetic result must remain inside the call's turn region."""
    h = SessionHistoryManager()
    user_message = Message("user", ["go"])
    call_message = Message(
        "assistant",
        [Content.from_function_call("call-1", "Explore", arguments={})],
    )
    awaiting_marker = Message("assistant", ["Awaiting 1 sub-agent"])
    awaiting_marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.AWAITING_SUB_AGENTS
    turn_marker = Message("assistant", [""])
    turn_marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    h.bind({"messages": [user_message, call_message, awaiting_marker, turn_marker]})

    injected = h.inject_error_results_for_sub_agents([], {"Explore"})

    assert injected == 1
    assert h.messages[:2] == [user_message, call_message]
    assert h.messages[2].role == "tool"
    assert h.messages[2].contents[0].call_id == "call-1"
    assert h.messages[3:] == [awaiting_marker, turn_marker]


def test_drain_mixed_corrupt_and_valid_records_preserves_valid(tmp_path):
    """One corrupt + two valid records — valid ones drained, corrupt
    quarantined, neither leaks into the returned list."""
    sub_dir = tmp_path / "sub_agents"
    sub_dir.mkdir()
    atomic_write_owner_only_text(sub_dir / "corrupt.json", "{partial")
    _write_record(tmp_path, "alpha", tool_name="Explore", last_error="a")
    _write_record(tmp_path, "beta", tool_name="Plan", last_error="b")

    tools = SubAgentSessionArtifactService(tmp_path)
    records = tools.drain_paused_records()

    ids = sorted(r["invocation_id"] for r in records)
    assert ids == ["alpha", "beta"]
    assert sorted(path.name for path in sub_dir.glob("*.json")) == ["alpha.json", "beta.json"]


@pytest.mark.asyncio
async def test_drain_plus_inject_end_to_end(tmp_path):
    """Happy-path integration: drain records from disk, inject into history."""
    _write_record(tmp_path, "inv-A", tool_name="Explore", last_error="context-full")
    tools = SubAgentSessionArtifactService(tmp_path)

    h = _history_with_dangling_call("Explore", "call-X")

    records = tools.drain_paused_records()
    injected = h.inject_error_results_for_sub_agents(records, {"Explore"})

    assert injected == 1
    assert "context-full" in h.messages[2].contents[0].result
    tools.finalize_restored_paused_records(records)
    assert list((tmp_path / "sub_agents").glob("*.json")) == []


# --- inject_error_results_for_sub_agents: exchange-shape preservation ----


def _bound_history(messages: list[Message]) -> SessionHistoryManager:
    h = SessionHistoryManager()
    h.bind({"messages": messages})
    return h


def _sub_call(call_id: object, name: str = "Explore") -> Content:
    return Content.from_function_call(call_id, name, arguments={})


def _tool_result(call_id: object) -> Content:
    return Content.from_function_result(call_id, result=f"result_{call_id!r}")


def test_inject_repairs_dangling_call_in_shared_block_sibling():
    """A dangling sibling call is repaired at the response's shared result boundary."""
    messages = [
        Message("user", ["go"]),
        Message("assistant", [_sub_call("c1")]),
        Message("assistant", [_sub_call("c2")]),
        Message("tool", [_tool_result("c2")]),
    ]
    h = _bound_history(messages)
    injected = h.inject_error_results_for_sub_agents([], {"Explore"})
    assert injected == 1
    assert h.messages[3].role == "tool"
    assert h.messages[3].contents[0].call_id == "c1"
    assert h.messages[4].contents[0].call_id == "c2"


def test_inject_ignores_none_id_sub_agent_call():
    """A None-id call never enters repair pairing."""
    messages = [Message("user", ["go"]), Message("assistant", [_sub_call(None)])]
    h = _bound_history(messages)
    assert h.inject_error_results_for_sub_agents([], {"Explore"}) == 0
    assert len(h.messages) == 2


def test_inject_falsy_numeric_call_with_falsy_result_untouched():
    """A falsy non-string call id answered by a falsy non-string result needs no repair."""
    messages = [
        Message("user", ["go"]),
        Message("assistant", [_sub_call(0)]),
        Message("tool", [_tool_result(False)]),
    ]
    h = _bound_history(messages)
    assert h.inject_error_results_for_sub_agents([], {"Explore"}) == 0
    assert len(h.messages) == 3


def test_inject_falsy_unhashable_call_with_falsy_unhashable_result_untouched():
    """A falsy unhashable call id answered by a falsy unhashable result needs no repair."""
    messages = [
        Message("user", ["go"]),
        Message("assistant", [_sub_call([])]),
        Message("tool", [_tool_result({})]),
    ]
    h = _bound_history(messages)
    assert h.inject_error_results_for_sub_agents([], {"Explore"}) == 0
    assert len(h.messages) == 3


# --- inject_error_results_for_sub_agents: exchange-grammar migration -----


def test_inject_user_role_result_does_not_resolve_call():
    """A user-role message carrying only results is not part of the output block."""
    messages = [
        Message("user", ["go"]),
        Message("assistant", [_sub_call("c1")]),
        Message("user", [_tool_result("c1")]),
    ]
    h = _bound_history(messages)
    injected = h.inject_error_results_for_sub_agents([], {"Explore"})
    assert injected == 1


def test_inject_marker_message_result_does_not_resolve_call():
    """A history-marker message never carries answers for a preceding call."""
    marker = Message("assistant", [_tool_result("c1")])
    marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    messages = [Message("user", ["go"]), Message("assistant", [_sub_call("c1")]), marker]
    h = _bound_history(messages)
    injected = h.inject_error_results_for_sub_agents([], {"Explore"})
    assert injected == 1


def test_inject_embedded_result_resolves_call_without_duplicate():
    """A result embedded in the call's own message answers it; no second result."""
    messages = [
        Message("user", ["go"]),
        Message("assistant", [_sub_call("c1"), _tool_result("c1")]),
    ]
    h = _bound_history(messages)
    assert h.inject_error_results_for_sub_agents([], {"Explore"}) == 0
    assert len(h.messages) == 2


def test_inject_empty_id_sub_agent_call_repaired():
    """An empty-id dangling sub-agent call is visible to recovery."""
    messages = [Message("user", ["go"]), Message("assistant", [_sub_call("")])]
    h = _bound_history(messages)
    injected = h.inject_error_results_for_sub_agents([], {"Explore"})
    assert injected == 1
    assert h.messages[2].contents[0].call_id == ""


def test_inject_numeric_call_resolved_by_digit_string_result():
    """Malformed ids pair by their string forms: int 7 is answered by "7"."""
    messages = [
        Message("user", ["go"]),
        Message("assistant", [_sub_call(7)]),
        Message("tool", [_tool_result("7")]),
    ]
    h = _bound_history(messages)
    assert h.inject_error_results_for_sub_agents([], {"Explore"}) == 0
    assert len(h.messages) == 3


def test_inject_separately_constructed_nan_pair_resolved():
    """Two distinct NaN id objects pair through their identical string forms."""
    messages = [
        Message("user", ["go"]),
        Message("assistant", [_sub_call(float("nan"))]),
        Message("tool", [_tool_result(float("nan"))]),
    ]
    h = _bound_history(messages)
    assert h.inject_error_results_for_sub_agents([], {"Explore"}) == 0
    assert len(h.messages) == 3


def test_inject_falsy_numeric_call_repaired():
    """A falsy non-string call id joins the empty-id stream and is repairable."""
    messages = [Message("user", ["go"]), Message("assistant", [_sub_call(0)])]
    h = _bound_history(messages)
    injected = h.inject_error_results_for_sub_agents([], {"Explore"})
    assert injected == 1
    assert h.messages[2].contents[0].call_id == 0


def test_inject_int_call_not_resolved_by_bool_result():
    """Python-equal ids of different types split by string form: 1 is not True."""
    messages = [
        Message("user", ["go"]),
        Message("assistant", [_sub_call(1)]),
        Message("tool", [_tool_result(True)]),
    ]
    h = _bound_history(messages)
    injected = h.inject_error_results_for_sub_agents([], {"Explore"})
    assert injected == 1


def test_inject_truthy_unhashable_call_processed():
    """An unhashable call id is processed without a TypeError; the synthetic result
    copies the original id value verbatim."""
    original_id = ["x"]
    messages = [Message("user", ["go"]), Message("assistant", [_sub_call(original_id)])]
    h = _bound_history(messages)
    injected = h.inject_error_results_for_sub_agents([], {"Explore"})
    assert injected == 1
    assert h.messages[2].contents[0].call_id == original_id


def test_inject_falsy_unhashable_call_repaired():
    """A falsy unhashable call id joins the empty-id stream and is repairable."""
    messages = [Message("user", ["go"]), Message("assistant", [_sub_call([])])]
    h = _bound_history(messages)
    injected = h.inject_error_results_for_sub_agents([], {"Explore"})
    assert injected == 1
    assert h.messages[2].contents[0].call_id == []
