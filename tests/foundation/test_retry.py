# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for :mod:`chrys.foundation.retry`.

``StreamRetryLoop`` is driven by the main ``TurnBindings`` and (after Tier 2)
by sub-agent controllers, so it must be well-covered in isolation. These
tests exercise the loop against synthetic attempt callables without
touching a real ``Agent`` or event bus.
"""

from __future__ import annotations

import asyncio

import pytest

from chrys.foundation.retry import (
    HistorySnapshot,
    RetryAttemptInfo,
    StreamRetryLoop,
    StreamStall,
    StreamStallExhausted,
    restore_message_properties,
    snapshot_message_properties,
)
from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.kernel import Content, LoopRecorder, Message


class _Harness:
    """Collects loop callbacks so tests can assert on them."""

    def __init__(self, *, retryable: bool = True, interrupted: bool = False):
        self.retryable = retryable
        self.interrupted = interrupted
        self.snapshot_calls = 0
        self.restore_calls = 0
        self.publish_events: list[tuple[str, int, int, int]] = []
        self.sleep_calls: list[int] = []
        self.base_snapshot = HistorySnapshot(messages=[], compressed_count=0)

    def snapshot(self) -> HistorySnapshot:
        self.snapshot_calls += 1
        return self.base_snapshot

    def restore(self, snap: HistorySnapshot) -> None:
        self.restore_calls += 1
        assert snap is self.base_snapshot

    def is_retryable(self, _exc: BaseException) -> bool:
        return self.retryable

    async def publish(
        self,
        message: str,
        attempt: int,
        max_attempts: int,
        delay_seconds: int,
        _exc: BaseException,
    ) -> None:
        self.publish_events.append((message, attempt, max_attempts, delay_seconds))

    def is_interrupted(self) -> bool:
        return self.interrupted

    async def sleep(self, seconds: int) -> bool:
        self.sleep_calls.append(seconds)
        return self.interrupted

    def clean_err(self, exc: BaseException) -> str:
        return str(exc) or type(exc).__name__

    def build(
        self,
        *,
        max_retries: int = 3,
        backoff: tuple[int, ...] = (0, 0, 0, 0),
        restore_on_stall_exhaustion: bool = False,
        after_restore=None,
        retry_exemption=None,
    ):
        return StreamRetryLoop(
            max_retries=max_retries,
            backoff_schedule=backoff,
            is_retryable=self.is_retryable,
            snapshot_history=self.snapshot,
            restore_history=self.restore,
            publish_retry_attempt=self.publish,
            is_interrupted=self.is_interrupted,
            interruptible_sleep=self.sleep,
            clean_error_message=self.clean_err,
            restore_on_stall_exhaustion=restore_on_stall_exhaustion,
            after_restore=after_restore,
            retry_exemption=retry_exemption,
        )


@pytest.mark.asyncio
async def test_success_on_first_attempt() -> None:
    """Happy path: attempt succeeds, no retry/publish/restore fires."""
    h = _Harness()
    loop = h.build()
    calls = 0

    async def attempt() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await loop.run(attempt)

    assert result == "ok"
    assert calls == 1
    assert h.snapshot_calls == 1
    assert h.restore_calls == 0
    assert h.publish_events == []
    assert h.sleep_calls == []


@pytest.mark.asyncio
async def test_stall_then_success_retries_and_restores() -> None:
    """StreamStall on first attempt must trigger restore + retry, then succeed."""
    h = _Harness()
    loop = h.build(max_retries=3)
    attempts = 0

    async def attempt() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise StreamStall
        return "done"

    result = await loop.run(attempt)

    assert result == "done"
    assert attempts == 2
    assert h.restore_calls == 1
    assert len(h.publish_events) == 1
    msg, attempt_num, max_attempts, delay = h.publish_events[0]
    assert msg == "Stream stalled"
    assert attempt_num == 1
    assert max_attempts == 3
    assert delay == 0


@pytest.mark.asyncio
async def test_stall_exhaustion_raises_stream_stall_exhausted() -> None:
    """Every attempt stalls → StreamStallExhausted (caller falls back)."""
    h = _Harness()
    loop = h.build(max_retries=2)

    async def attempt() -> str:
        raise StreamStall

    with pytest.raises(StreamStallExhausted) as exc_info:
        await loop.run(attempt)

    assert isinstance(exc_info.value.__cause__, StreamStall)
    # Two retries published (attempts 1, 2 of 2 max_retries); the third
    # attempt's stall raises StreamStallExhausted without publishing.
    assert len(h.publish_events) == 2
    assert h.restore_calls == 2


@pytest.mark.asyncio
async def test_stall_exhaustion_restores_state_when_opted_in() -> None:
    """restore_on_stall_exhaustion=True → the final stall rolls back history
    AND caller state (after_restore) before StreamStallExhausted surfaces,
    so the blocking fallback starts from a clean slate."""
    h = _Harness()
    after_restore_seen: list[int] = []

    def after_restore() -> None:
        # History must already be restored when caller state is replayed.
        after_restore_seen.append(h.restore_calls)

    loop = h.build(
        max_retries=2,
        restore_on_stall_exhaustion=True,
        after_restore=after_restore,
    )

    async def attempt() -> str:
        raise StreamStall

    with pytest.raises(StreamStallExhausted):
        await loop.run(attempt)

    # Two mid-loop retry restores plus the exhaustion rollback.
    assert h.restore_calls == 3
    # after_restore fired synchronously after EVERY history restore.
    assert after_restore_seen == [1, 2, 3]


@pytest.mark.asyncio
async def test_stall_exhaustion_without_optin_skips_final_rollback() -> None:
    """Default semantics (sub-agent pause/retry flow): exhaustion must NOT
    discard the failed attempt's state — only mid-loop retries restore."""
    h = _Harness()
    after_restore_calls = 0

    def after_restore() -> None:
        nonlocal after_restore_calls
        after_restore_calls += 1

    loop = h.build(max_retries=1, after_restore=after_restore)

    async def attempt() -> str:
        raise StreamStall

    with pytest.raises(StreamStallExhausted):
        await loop.run(attempt)

    assert h.restore_calls == 1
    assert after_restore_calls == 1


@pytest.mark.asyncio
async def test_after_restore_runs_before_publish_and_backoff() -> None:
    """Caller-state replay must happen in the same synchronous phase as the
    history restore — before the retry event publish and before backoff, so
    an interrupt landing in either await gap cannot observe half-restored
    state."""
    order: list[str] = []
    snap = HistorySnapshot(messages=[], compressed_count=0)

    async def publish(message, attempt, max_attempts, delay, exc) -> None:
        order.append("publish")

    async def sleep(seconds: int) -> bool:
        order.append("sleep")
        return False

    loop = StreamRetryLoop(
        max_retries=1,
        backoff_schedule=(0,),
        is_retryable=lambda _e: True,
        snapshot_history=lambda: snap,
        restore_history=lambda _s: order.append("restore"),
        publish_retry_attempt=publish,
        is_interrupted=lambda: False,
        interruptible_sleep=sleep,
        clean_error_message=str,
        after_restore=lambda: order.append("after_restore"),
    )

    attempts = 0

    async def attempt() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise StreamStall
        return "ok"

    assert await loop.run(attempt) == "ok"
    assert order == ["restore", "after_restore", "publish", "sleep"]


@pytest.mark.asyncio
async def test_zero_commit_retry_discards_staging_before_backoff_interrupt() -> None:
    recorder = LoopRecorder()
    recorder_snapshot = recorder.snapshot()
    history_snapshot = HistorySnapshot(messages=[], compressed_count=0)
    attempts = 0
    restores = 0
    sleeps = 0

    def restore(_snapshot: HistorySnapshot) -> None:
        nonlocal restores
        restores += 1

    async def sleep(_seconds: int) -> bool:
        nonlocal sleeps
        sleeps += 1
        return sleeps == 2

    loop = StreamRetryLoop(
        max_retries=3,
        backoff_schedule=(0, 0, 0),
        is_retryable=lambda _exc: True,
        snapshot_history=lambda: history_snapshot,
        restore_history=restore,
        publish_retry_attempt=lambda *_args: asyncio.sleep(0),
        is_interrupted=lambda: False,
        interruptible_sleep=sleep,
        clean_error_message=str,
        after_restore=lambda: recorder.restore(recorder_snapshot),
        may_retry=lambda _exc: recorder.committed_count == 0,
    )

    async def attempt() -> str:
        nonlocal attempts
        attempts += 1
        call = Content.from_function_call(f"call-{attempts}", "tool", arguments={})
        call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = attempts - 1
        recorder.stage_exchange([Message("assistant", [call])], [call], result_carrier_item_id="a" * 32)
        raise ConnectionError("failed before any tool returned")

    with pytest.raises(asyncio.CancelledError):
        await loop.run(attempt)

    assert attempts == 2
    assert restores == 2
    assert recorder.committed_count == 0
    assert recorder.loop_messages is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (ConnectionError("retryable"), ConnectionError),
        (StreamStall("stalled"), RuntimeError),
    ],
)
async def test_retry_gate_preserves_attempt_state_without_restore(
    exception: BaseException, expected: type[BaseException]
) -> None:
    h = _Harness()
    committed = True
    loop = StreamRetryLoop(
        max_retries=2,
        backoff_schedule=(0, 0),
        is_retryable=lambda _exc: True,
        snapshot_history=h.snapshot,
        restore_history=h.restore,
        publish_retry_attempt=h.publish,
        is_interrupted=h.is_interrupted,
        interruptible_sleep=h.sleep,
        clean_error_message=h.clean_err,
        restore_on_stall_exhaustion=True,
        may_retry=lambda _exc: not committed,
    )

    async def attempt() -> str:
        raise exception

    with pytest.raises(expected) as exc_info:
        await loop.run(attempt)

    if expected is RuntimeError:
        assert "preserve the completed work" in str(exc_info.value)
        assert exc_info.value.__cause__ is exception
    assert h.restore_calls == 0
    assert h.publish_events == []


@pytest.mark.asyncio
async def test_non_retryable_exception_propagates_immediately() -> None:
    """is_retryable=False → exception re-raises on first occurrence."""
    h = _Harness(retryable=False)
    loop = h.build(max_retries=3)

    class NonRetryable(Exception):
        pass

    async def attempt() -> str:
        raise NonRetryable("boom")

    with pytest.raises(NonRetryable, match="boom"):
        await loop.run(attempt)

    assert h.publish_events == []
    assert h.restore_calls == 0


@pytest.mark.asyncio
async def test_retryable_exception_exhaustion_reraises_original() -> None:
    """Retryable exception exhaustion re-raises the *last* exception (not StreamStallExhausted)."""
    h = _Harness(retryable=True)
    loop = h.build(max_retries=2)

    class Transient(Exception):
        pass

    async def attempt() -> str:
        raise Transient("network blip")

    with pytest.raises(Transient, match="network blip"):
        await loop.run(attempt)

    assert len(h.publish_events) == 2


@pytest.mark.asyncio
async def test_zero_budget_retryable_exception_runs_single_attempt() -> None:
    h = _Harness(retryable=True)
    loop = h.build(max_retries=0)
    calls = 0

    async def attempt() -> str:
        nonlocal calls
        calls += 1
        raise ConnectionError("network blip")

    with pytest.raises(ConnectionError, match="network blip"):
        await loop.run(attempt)

    assert calls == 1
    assert h.publish_events == []


@pytest.mark.asyncio
async def test_interrupt_before_attempt_raises_cancelled() -> None:
    """is_interrupted() True at loop entry → CancelledError."""
    h = _Harness(interrupted=True)
    loop = h.build()

    async def attempt() -> str:
        raise AssertionError("should not be called")

    with pytest.raises(asyncio.CancelledError):
        await loop.run(attempt)


@pytest.mark.asyncio
async def test_interrupt_during_backoff_raises_cancelled() -> None:
    """Sleep callback returning True (interrupted) cancels the loop."""
    h = _Harness()

    interrupted_holder = {"val": False}

    async def sleep(seconds: int) -> bool:
        interrupted_holder["val"] = True
        return True

    loop = StreamRetryLoop(
        max_retries=3,
        backoff_schedule=(0,),
        is_retryable=lambda _e: True,
        snapshot_history=h.snapshot,
        restore_history=h.restore,
        publish_retry_attempt=h.publish,
        is_interrupted=lambda: interrupted_holder["val"],
        interruptible_sleep=sleep,
        clean_error_message=h.clean_err,
    )

    async def attempt() -> str:
        raise StreamStall

    with pytest.raises(asyncio.CancelledError):
        await loop.run(attempt)

    # First attempt stalled, published + restored, slept → sleep signalled
    # interrupt → loop raises CancelledError instead of trying again.
    assert len(h.publish_events) == 1
    assert h.restore_calls == 1


@pytest.mark.asyncio
async def test_backoff_schedule_is_indexed_by_attempt() -> None:
    """Delay per attempt must come from backoff_schedule[attempt] (clamped)."""
    h = _Harness()
    loop = h.build(max_retries=5, backoff=(1, 3, 7))  # shorter than max_retries

    async def attempt() -> str:
        raise StreamStall

    with pytest.raises(StreamStallExhausted):
        await loop.run(attempt)

    # attempts 0..4 all stalled before the final one raised Exhausted.
    # Five retries published with delays = [1, 3, 7, 7, 7] (clamped).
    assert [e[3] for e in h.publish_events] == [1, 3, 7, 7, 7]
    assert h.sleep_calls == [1, 3, 7, 7, 7]


@pytest.mark.asyncio
async def test_empty_backoff_schedule_yields_zero_delay() -> None:
    """Edge case: no schedule entries → delay 0 (defensive; matches prod use)."""
    h = _Harness()
    loop = h.build(max_retries=1, backoff=())

    attempts = 0

    async def attempt() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise StreamStall
        return "ok"

    result = await loop.run(attempt)

    assert result == "ok"
    assert h.sleep_calls == [0]


@pytest.mark.asyncio
async def test_snapshot_captured_once_and_reused_across_retries() -> None:
    """The loop must snapshot once on entry, not re-snapshot per attempt."""
    h = _Harness()
    loop = h.build(max_retries=3)
    attempts = 0

    async def attempt() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise StreamStall
        return "ok"

    await loop.run(attempt)

    assert h.snapshot_calls == 1
    assert h.restore_calls == 2  # restored before retry 2 and retry 3


@pytest.mark.asyncio
async def test_publish_receives_original_exception() -> None:
    """publish_retry_attempt must receive the triggering exception so
    callers (e.g. sub-agents) can classify failures downstream."""
    seen: list[BaseException] = []

    async def publish(message, attempt, max_attempts, delay, exc):
        seen.append(exc)

    h = _Harness()
    loop = StreamRetryLoop(
        max_retries=2,
        backoff_schedule=(0,),
        is_retryable=lambda _e: True,
        snapshot_history=h.snapshot,
        restore_history=h.restore,
        publish_retry_attempt=publish,
        is_interrupted=lambda: False,
        interruptible_sleep=h.sleep,
        clean_error_message=h.clean_err,
    )

    class Flake(Exception):
        pass

    attempts = 0

    async def attempt() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise Flake("first")
        raise StreamStall

    with pytest.raises(StreamStallExhausted):
        await loop.run(attempt)

    assert len(seen) == 2
    assert isinstance(seen[0], Flake)
    assert isinstance(seen[1], StreamStall)


@pytest.mark.asyncio
async def test_exempt_retry_has_independent_accounting_and_integer_delay() -> None:
    h = _Harness()

    class ExemptFailure(ConnectionError):
        pass

    loop = h.build(
        max_retries=1,
        backoff=(7,),
        retry_exemption=lambda exc: (
            RetryAttemptInfo(reason="validation", attempt=2, max_attempts=3, delay_seconds=1.2)
            if isinstance(exc, ExemptFailure)
            else None
        ),
    )
    failures: list[BaseException] = [ExemptFailure("invalid"), ConnectionError("transport")]

    async def attempt() -> str:
        if failures:
            raise failures.pop(0)
        return "ok"

    assert await loop.run(attempt) == "ok"
    assert h.publish_events == [
        ("validation", 2, 3, 2),
        ("transport", 1, 1, 7),
    ]
    assert h.sleep_calls == [2, 7]
    assert h.restore_calls == 2


@pytest.mark.asyncio
async def test_zero_budget_allows_exempt_retry_then_stall_exhausts() -> None:
    h = _Harness()
    loop = h.build(
        max_retries=0,
        restore_on_stall_exhaustion=True,
        retry_exemption=lambda exc: (
            RetryAttemptInfo(reason="validation", attempt=1, max_attempts=3, delay_seconds=0)
            if isinstance(exc, ConnectionError)
            else None
        ),
    )
    failures: list[BaseException] = [ConnectionError("invalid"), StreamStall("stalled")]

    async def attempt() -> str:
        raise failures.pop(0)

    with pytest.raises(StreamStallExhausted):
        await loop.run(attempt)

    assert h.publish_events == [("validation", 1, 3, 0)]
    assert h.restore_calls == 2


@pytest.mark.asyncio
async def test_exempt_then_transient_then_stall_keeps_per_lane_counters() -> None:
    """Stall and transport failures share the transient lane; exempt retries touch neither."""
    h = _Harness()

    class ExemptFailure(ConnectionError):
        pass

    loop = h.build(
        max_retries=2,
        backoff=(5, 9),
        retry_exemption=lambda exc: (
            RetryAttemptInfo(reason="validation", attempt=1, max_attempts=3, delay_seconds=0)
            if isinstance(exc, ExemptFailure)
            else None
        ),
    )
    failures: list[BaseException] = [
        ExemptFailure("invalid"),
        ConnectionError("transport"),
        StreamStall("stalled"),
    ]

    async def attempt() -> str:
        if failures:
            raise failures.pop(0)
        return "ok"

    assert await loop.run(attempt) == "ok"
    assert h.publish_events == [
        ("validation", 1, 3, 0),
        ("transport", 1, 2, 5),
        ("Stream stalled", 2, 2, 9),
    ]
    assert h.sleep_calls == [0, 5, 9]
    assert h.restore_calls == 3


@pytest.mark.asyncio
async def test_retry_gate_vetoes_exempt_retry_before_restore_or_event() -> None:
    h = _Harness()
    loop = StreamRetryLoop(
        max_retries=0,
        backoff_schedule=(0,),
        is_retryable=lambda _exc: True,
        snapshot_history=h.snapshot,
        restore_history=h.restore,
        publish_retry_attempt=h.publish,
        is_interrupted=h.is_interrupted,
        interruptible_sleep=h.sleep,
        clean_error_message=h.clean_err,
        may_retry=lambda _exc: False,
        retry_exemption=lambda _exc: RetryAttemptInfo("validation", 1, 3, 0),
    )

    async def attempt() -> str:
        raise ConnectionError("invalid")

    with pytest.raises(ConnectionError, match="invalid"):
        await loop.run(attempt)

    assert h.restore_calls == 0
    assert h.publish_events == []


@pytest.mark.asyncio
async def test_interrupt_during_exempt_backoff_raises_cancelled_after_event() -> None:
    h = _Harness()

    async def interrupted_sleep(seconds: int) -> bool:
        h.sleep_calls.append(seconds)
        return True

    loop = StreamRetryLoop(
        max_retries=0,
        backoff_schedule=(0,),
        is_retryable=lambda _exc: True,
        snapshot_history=h.snapshot,
        restore_history=h.restore,
        publish_retry_attempt=h.publish,
        is_interrupted=h.is_interrupted,
        interruptible_sleep=interrupted_sleep,
        clean_error_message=h.clean_err,
        retry_exemption=lambda _exc: RetryAttemptInfo("validation", 1, 3, 1),
    )

    async def attempt() -> str:
        raise ConnectionError("invalid")

    with pytest.raises(asyncio.CancelledError):
        await loop.run(attempt)

    assert h.restore_calls == 1
    assert h.publish_events == [("validation", 1, 3, 1)]
    assert h.sleep_calls == [1]


# ---------------------------------------------------------------------------
# Exact-restore property snapshots
# ---------------------------------------------------------------------------


class _Msg:
    """Duck-typed stand-in for Message — the helpers only touch additional_properties."""

    def __init__(self, props: dict | None = None):
        self.additional_properties: dict = props if props is not None else {}


def test_snapshot_message_properties_is_isolated_from_later_mutations():
    msg = _Msg({"_excluded": True, "_group": {"group_id": "g1"}})

    snap = snapshot_message_properties([msg])

    msg.additional_properties["_excluded"] = False
    msg.additional_properties["new_key"] = 1
    msg.additional_properties["_group"]["_summarized_by_summary_id"] = "s1"

    assert snap == ({"_excluded": True, "_group": {"group_id": "g1"}},)


def test_restore_message_properties_reverts_and_survives_double_restore():
    msg = _Msg({"_group": {"group_id": "g1"}})
    snap = snapshot_message_properties([msg])

    msg.additional_properties["_excluded"] = True
    msg.additional_properties["_group"]["extra"] = "x"
    restore_message_properties([msg], snap)

    assert msg.additional_properties == {"_group": {"group_id": "g1"}}

    # Copy-on-restore: the live message must not hold the snapshot dicts, so
    # a second mutate/restore cycle from the SAME snapshot stays clean.
    assert msg.additional_properties is not snap[0]
    assert msg.additional_properties["_group"] is not snap[0]["_group"]
    msg.additional_properties["_excluded"] = True
    msg.additional_properties["_group"]["extra"] = "y"
    restore_message_properties([msg], snap)
    assert msg.additional_properties == {"_group": {"group_id": "g1"}}


def test_restore_message_properties_rejects_length_mismatch():
    msg = _Msg()
    with pytest.raises(ValueError):
        restore_message_properties([msg, _Msg()], snapshot_message_properties([msg]))
