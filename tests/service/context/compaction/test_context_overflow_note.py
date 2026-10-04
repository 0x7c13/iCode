# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A provider's context-overflow verdict forces the next compaction pass.

The local estimate can sit below the trigger while the provider's real window
is already full. After :meth:`UnifiedContextStrategy.note_context_overflow`
the next pass runs on the real occupancy, once, instead of letting the same
input go out again.
"""

from __future__ import annotations

import asyncio

import pytest

from chrys.foundation.errors import is_context_overflow
from chrys.foundation.trajectory.context import trajectory_scope
from chrys.foundation.trajectory.event_types import EventType
from chrys.kernel import ContextOverflowSink, Message
from chrys.service.context.compaction import PreCompactInfo, UnifiedContextStrategy
from chrys.service.context.compaction.last_words import LastWordsGenerationError
from tests.service.context.compaction._compaction_helpers import (
    _async_appender,
    _build_multi_turn,
    _estimate_tokens,
    _make_strategy,
)
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support.provider_errors import openai_context_overflow, openai_status, raised_from

_USAGE = 0.7  # between the default target (0.50) and trigger (0.85)


async def _overflow() -> BaseException:
    return await openai_context_overflow()


def _messages() -> list[Message]:
    """Three finished turns of tool work and a fourth in progress; a pass mutates them, so each call gets fresh ones."""
    return _build_multi_turn(4, groups_per_turn=3, result_size=2000)


def _window() -> int:
    return round(_estimate_tokens(_messages()) / _USAGE)


def _usage(strategy: UnifiedContextStrategy) -> float:
    return strategy.estimated_context_input_tokens / strategy.max_context_tokens


def _started_triggers(sink: FakeSink) -> list[str]:
    return [event.payload["trigger"] for event in sink.of_type(EventType.COMPACTION_STARTED)]


async def test_a_noted_overflow_compacts_once_below_the_trigger() -> None:
    pre_compact: list[PreCompactInfo] = []
    strategy = _make_strategy(max_context_tokens=_window(), on_pre_compact=_async_appender(pre_compact))
    assert isinstance(strategy, ContextOverflowSink)
    sink = FakeSink()

    with trajectory_scope(make_context(sink)):
        assert await strategy(_messages()) is False
        usage = _usage(strategy)
        assert strategy.target_pct < usage < strategy.trigger_pct

        assert strategy.note_context_overflow(await _overflow()) is True
        assert await strategy(_messages()) is True
        assert _usage(strategy) <= strategy.target_pct

        # The note forced one pass; the next call is back to the trigger.
        assert await strategy(_messages()) is False

    assert _started_triggers(sink) == ["context_overflow"]
    assert pre_compact[0].trigger == "phase1"
    assert pre_compact[0].usage_pct == pytest.approx(usage)
    assert (strategy.max_context_tokens, strategy.trigger_pct, strategy.target_pct) == (_window(), 0.85, 0.50)


async def test_a_note_without_the_error_still_forces_the_pass() -> None:
    """The output-truncated verdict carries no exception."""
    strategy = _make_strategy(max_context_tokens=_window())

    assert strategy.note_context_overflow() is True
    assert await strategy(_messages()) is True


@pytest.mark.parametrize(
    ("limit_offset", "resend_helps"), [(-1, False), (0, True)], ids=["server_limit_below_profile", "equal"]
)
async def test_a_server_limit_below_the_profiles_window_rules_out_the_resend_but_still_compacts(
    limit_offset: int, resend_helps: bool
) -> None:
    window = _window()
    message = f"This model's maximum context length is {window + limit_offset} tokens."
    error = await openai_status(400, {"error": {"type": "invalid_request_error", "message": message}})
    strategy = _make_strategy(max_context_tokens=window)

    assert strategy.note_context_overflow(error) is resend_helps
    assert await strategy(_messages()) is True


async def test_past_the_trigger_the_noted_pass_is_labelled_by_the_overflow() -> None:
    strategy = _make_strategy(max_context_tokens=round(_window() * _USAGE / 0.9))
    sink = FakeSink()

    with trajectory_scope(make_context(sink)):
        strategy.note_context_overflow(await _overflow())
        assert await strategy(_messages()) is True
        assert await strategy(_messages()) is True

    assert _started_triggers(sink) == ["context_overflow", "usage_threshold"]


async def test_with_compaction_off_a_note_changes_nothing() -> None:
    strategy = UnifiedContextStrategy(max_context_tokens=_window(), compaction_enabled=False)
    sink = FakeSink()

    with trajectory_scope(make_context(sink)):
        assert strategy.note_context_overflow(await _overflow()) is False
        assert await strategy(_messages()) is False

    assert sink.of_type(EventType.COMPACTION_STARTED) == []
    assert sink.of_type(EventType.COMPACTION_SKIPPED) == []


async def _last_words_failure() -> BaseException:
    return raised_from(
        LastWordsGenerationError(
            "Fallback request cannot fit the model context window without dropping current-turn work"
        ),
        await _overflow(),
    )


@pytest.mark.parametrize("wrapped", [False, True], ids=["bare", "behind_a_wrapper"])
async def test_compactions_own_last_words_failure_is_not_noted(wrapped: bool) -> None:
    error = await _last_words_failure()
    if wrapped:
        error = raised_from(RuntimeError("compaction failed"), error)
    # The loop would hand it over: it reads as an overflow.
    assert is_context_overflow(error)
    strategy = _make_strategy(max_context_tokens=_window())
    sink = FakeSink()

    with trajectory_scope(make_context(sink)):
        assert strategy.note_context_overflow(error) is False
        assert await strategy(_messages()) is False

    assert sink.of_type(EventType.COMPACTION_STARTED) == []


def _without_turns(messages: list[Message]) -> list[Message]:
    return [message for message in messages if message.role != "user"]


async def test_a_noted_pass_with_no_turn_waits_for_the_next_call() -> None:
    strategy = _make_strategy(max_context_tokens=_window())
    sink = FakeSink()

    with trajectory_scope(make_context(sink)):
        strategy.note_context_overflow(await _overflow())
        assert await strategy(_without_turns(_messages())) is False
        assert strategy.target_pct < _usage(strategy) < strategy.trigger_pct
        # ``compaction.skipped`` is about usage past the trigger, which this is not.
        assert sink.of_type(EventType.COMPACTION_SKIPPED) == []
        assert await strategy(_messages()) is True

    assert _started_triggers(sink) == ["context_overflow"]


async def test_a_noted_pass_with_no_turn_ends_a_stretch_past_the_trigger() -> None:
    """Like any call below the trigger, it lets the next stretch past it report its skip again."""
    past_trigger = _without_turns(_build_multi_turn(4, groups_per_turn=3, result_size=4000))
    strategy = _make_strategy(max_context_tokens=_window())
    sink = FakeSink()

    with trajectory_scope(make_context(sink)):
        assert await strategy(past_trigger) is False
        assert _usage(strategy) > strategy.trigger_pct
        strategy.note_context_overflow(await _overflow())
        assert await strategy(_without_turns(_messages())) is False
        assert await strategy(past_trigger) is False

    assert len(sink.of_type(EventType.COMPACTION_SKIPPED)) == 2


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError], ids=["error", "cancelled"])
async def test_a_noted_pass_that_does_not_finish_keeps_the_note(failure: type[BaseException]) -> None:
    calls = 0

    async def fail_first_phase(_info: PreCompactInfo) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure("pass stopped")

    strategy = _make_strategy(max_context_tokens=_window(), on_pre_compact=fail_first_phase)
    strategy.note_context_overflow(await _overflow())

    with pytest.raises(failure):
        await strategy(_messages())

    assert await strategy(_messages()) is True


async def test_rolling_back_a_retry_attempt_keeps_the_note() -> None:
    strategy = _make_strategy(max_context_tokens=_window())
    snapshot = strategy.snapshot_retry_state()
    strategy.note_context_overflow(await _overflow())

    strategy.restore_retry_state(snapshot)

    assert await strategy(_messages()) is True
