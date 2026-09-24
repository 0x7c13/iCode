# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Backstop item stamping, the forked session's own prelude, and usage normalization."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from chrys.foundation.trajectory.envelope import MeasurementSource
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import is_valid_analytics_id
from chrys.foundation.trajectory.lease import WriterLease
from chrys.foundation.trajectory.metadata import ANALYTICS_ITEM_ID_KEY
from chrys.foundation.trajectory.reader import read_trajectory
from chrys.foundation.trajectory.usage import (
    USAGE_ADAPTER_VERSION,
    normalized_usage,
    provider_reported_usage,
    usage_measurements,
)
from chrys.kernel import Content, Message
from chrys.service.trajectory.fork import record_fork
from chrys.service.trajectory.items import ensure_history_item_ids
from chrys.service.trajectory.session import SessionStartInfo, trajectory_events_path, trajectory_lease_path
from tests.support.trajectory_invariants import assert_trajectory_accounted
from tests.support.waiting import wait_for

SESSION_ID = "12345678-1234-1234-1234-123456789abc"
ORIGIN_SESSION_ID = "87654321-4321-4321-4321-cba987654321"


# ------------------------------------------------------------------- items


def test_every_message_and_tool_content_is_stamped_once() -> None:
    messages = [
        Message(role="user", contents=[Content.from_text("hi")]),
        Message(role="assistant", contents=[Content.from_function_call("call-1", "echo", arguments={})]),
        Message(role="tool", contents=[Content.from_function_result("call-1", result="ok")]),
    ]

    assert ensure_history_item_ids(messages) == 5

    ids = [message.additional_properties[ANALYTICS_ITEM_ID_KEY] for message in messages]
    ids += [
        content.additional_properties[ANALYTICS_ITEM_ID_KEY]
        for message in messages
        for content in message.contents
        if content.type in {"function_call", "function_result"}
    ]
    assert all(is_valid_analytics_id(item_id) for item_id in ids)
    assert len(set(ids)) == len(ids)
    # A second pass finds nothing left to stamp and never rewrites an id.
    assert ensure_history_item_ids(messages) == 0
    assert [message.additional_properties[ANALYTICS_ITEM_ID_KEY] for message in messages] == ids[:3]


def test_plain_text_contents_carry_no_identity_of_their_own() -> None:
    message = Message(role="assistant", contents=[Content.from_text("a"), Content.from_text("b")])

    assert ensure_history_item_ids([message]) == 1
    assert all(ANALYTICS_ITEM_ID_KEY not in content.additional_properties for content in message.contents)


def test_non_messages_are_skipped_rather_than_failing() -> None:
    # History can hold foreign objects during recovery; stamping must not raise.
    assert ensure_history_item_ids([SimpleNamespace(role="user"), None, "text"]) == 0


# -------------------------------------------------------------------- fork


@pytest.mark.asyncio
async def test_a_fork_opens_its_own_closed_runtime_pointing_at_its_origin(tmp_path: Path) -> None:
    fork_dir = tmp_path / "fork"
    fork_dir.mkdir()
    (fork_dir / "session.json").write_text("{}", encoding="utf-8")

    recorded = await record_fork(
        fork_session_id=SESSION_ID,
        fork_session_dir=fork_dir,
        fork_write_lock_path=None,
        origin_session_id=ORIGIN_SESSION_ID,
        forked_at_sequence=42,
        session_start_info=SessionStartInfo(
            primary_cwd=str(tmp_path), agent_profile_fingerprint="afp", model_profile_fingerprint="mfp"
        ),
    )

    assert recorded is True
    result = read_trajectory(trajectory_events_path(fork_dir))
    assert result.corrupt_lines == []
    assert_trajectory_accounted(result)
    types = [event.event_type for event in result.events]
    # The fork's log starts at sequence 1 with its own runtime and closes cleanly.
    assert result.events[0].sequence == 1
    assert types[0] == EventType.COVERAGE_STARTED
    assert EventType.SESSION_STARTED in types
    assert types[-1] == EventType.RUNTIME_FINISHED
    forked = next(event for event in result.events if event.event_type == EventType.SESSION_FORKED)
    assert forked.payload == {"origin_session_id": ORIGIN_SESSION_ID, "forked_at_sequence": 42}
    assert forked.session_id == SESSION_ID
    # The parent's directory path never appears in the child's log.
    assert str(tmp_path) not in trajectory_events_path(fork_dir).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_fork_that_cannot_open_a_log_reports_failure(tmp_path: Path) -> None:
    # A file where the trajectory directory must go: activation cannot succeed.
    fork_dir = tmp_path / "fork"
    fork_dir.mkdir()
    (fork_dir / "trajectory").write_text("not a directory", encoding="utf-8")

    assert (
        await record_fork(
            fork_session_id=SESSION_ID,
            fork_session_dir=fork_dir,
            fork_write_lock_path=None,
            origin_session_id=ORIGIN_SESSION_ID,
            forked_at_sequence=1,
            session_start_info=None,
        )
        is False
    )


@pytest.mark.asyncio
async def test_a_cancelled_fork_still_closes_the_log_its_prelude_opened(tmp_path: Path) -> None:
    """Cancellation detaches the caller from the prelude; it never abandons an open log."""
    fork_dir = tmp_path / "fork"
    fork_dir.mkdir()
    (fork_dir / "session.json").write_text("{}", encoding="utf-8")

    task = asyncio.ensure_future(
        record_fork(
            fork_session_id=SESSION_ID,
            fork_session_dir=fork_dir,
            fork_write_lock_path=trajectory_lease_path(fork_dir),
            origin_session_id=ORIGIN_SESSION_ID,
            forked_at_sequence=7,
            session_start_info=None,
        )
    )
    await asyncio.sleep(0)  # far enough in that the prelude exists to be orphaned
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The prelude outlives its caller, so the test waits for the same thing the
    # process would: the close it still owes, arriving after the caller is gone.
    events_path = trajectory_events_path(fork_dir)
    await wait_for(
        lambda: (
            events_path.is_file()
            and any(event.event_type == EventType.RUNTIME_FINISHED for event in read_trajectory(events_path).events)
        ),
        description="the orphaned prelude closing the fork's log",
    )

    result = read_trajectory(events_path)
    assert result.corrupt_lines == []
    assert_trajectory_accounted(result)
    types = [event.event_type for event in result.events]
    assert EventType.SESSION_FORKED in types
    assert types[-1] == EventType.RUNTIME_FINISHED  # closed, not left open on a dead worker
    # The lease goes back after the line it was held for, so the log showing
    # the close is not yet the worker having let go of the file.
    await wait_for(
        lambda: not WriterLease.is_held_elsewhere(trajectory_lease_path(fork_dir)),
        description="the orphaned prelude handing its writer lease back",
    )


# ------------------------------------------------------------------- usage


def test_provider_usage_keeps_the_breakdowns_the_buckets_cannot_rebuild() -> None:
    reported = provider_reported_usage(
        {
            "input_token_count": 10,
            "vendor_extra": "x",
            "ratio": 1.5,
            "flag": True,
            "cache_creation": {"ephemeral_5m_input_tokens": 7, "ephemeral_1h_input_tokens": 0},
            "tiers": [1, 2],
            "handle": object(),
        }
    )

    assert reported is not None
    assert reported.values == {
        "input_token_count": 10,
        "vendor_extra": "x",
        "ratio": 1.5,
        "flag": True,
        "cache_creation": {"ephemeral_5m_input_tokens": 7, "ephemeral_1h_input_tokens": 0},
        "tiers": [1, 2],
    }
    # What cannot be written as JSON is named, not silently dropped.
    assert reported.omitted == ("handle",)


def test_a_non_finite_usage_value_is_named_not_raised() -> None:
    """The mirror is built on the caller's thread: a value it cannot encode must not raise there."""
    reported = provider_reported_usage(
        {"input_token_count": 10, "rate": float("nan"), "ceiling": float("inf"), "detail": {"x": float("-inf")}}
    )

    assert reported is not None
    assert reported.values == {"input_token_count": 10}
    assert sorted(reported.omitted) == ["ceiling", "detail", "rate"]


def test_a_provider_mapping_too_large_to_mirror_keeps_its_scalars() -> None:
    reported = provider_reported_usage(
        {"input_token_count": 10, "detail": {f"bucket_{index}": index for index in range(200)}}
    )

    assert reported is not None
    assert reported.values == {"input_token_count": 10}
    assert reported.omitted == ("detail",)


def test_absent_usage_is_reported_as_unavailable_not_zero() -> None:
    assert provider_reported_usage(None) is None
    assert normalized_usage(None) == {"adapter_version": USAGE_ADAPTER_VERSION, "normalization_unavailable": True}


def test_normalization_copies_known_buckets_and_derives_visible_output() -> None:
    normalized = normalized_usage(
        {
            "input_token_count": 100,
            "cache_read_input_token_count": 60,
            "cache_creation_input_token_count": 5,
            "output_token_count": 40,
            "reasoning_output_token_count": 15,
        }
    )

    assert normalized == {
        "adapter_version": USAGE_ADAPTER_VERSION,
        "input_total": 100,
        "cache_read": 60,
        "cache_creation": 5,
        "output_total": 40,
        "reasoning": 15,
        "output_visible": 25,
    }


def test_a_missing_bucket_stays_missing() -> None:
    normalized = normalized_usage({"input_token_count": 7})

    assert normalized == {"adapter_version": USAGE_ADAPTER_VERSION, "input_total": 7}
    # No output count means no derived visible-output bucket, not zero.
    assert "output_visible" not in normalized


def test_implausible_counts_are_dropped_rather_than_recorded() -> None:
    normalized = normalized_usage({"input_token_count": -1, "output_token_count": 5, "reasoning_output_token_count": 9})

    assert "input_total" not in normalized
    # Reasoning larger than the total it belongs to cannot be subtracted.
    assert "output_visible" not in normalized
    assert normalized["reasoning"] == 9


def test_usage_that_carries_nothing_countable_is_unavailable() -> None:
    assert normalized_usage({"cache_read_input_token_count": 3})["normalization_unavailable"] is True


def test_every_normalized_bucket_is_labelled_provider_sourced() -> None:
    normalized = normalized_usage({"input_token_count": 3, "output_token_count": 4})

    measurements = usage_measurements(normalized)

    assert set(measurements) == {
        "/payload/usage/normalized/input_total",
        "/payload/usage/normalized/output_total",
    }
    for entry in measurements.values():
        assert entry["source"] == MeasurementSource.PROVIDER
        assert entry["adapter_version"] == USAGE_ADAPTER_VERSION
