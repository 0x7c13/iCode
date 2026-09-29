# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Resource close/acquire races without scheduler sleeps or backend semantics."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from chrys.orchestration.invoker import resources
from chrys.orchestration.invoker.contracts import AbortCause, PreparedClosed
from chrys.orchestration.invoker.resources import Conversation, PreparedAgent
from tests.support.close_races import ReleaseGate, assert_cancel_during_rollback, assert_entered_before_completion


def _release(order: list[str], name: str) -> Callable[[], Awaitable[None]]:
    async def close() -> None:
        order.append(name)

    return close


async def test_failed_open_rolls_back_private_resources_before_shared_release() -> None:
    prepared = PreparedAgent()
    order: list[str] = []
    prepared.own(_release(order, "shared-first"))
    prepared.own(_release(order, "shared-last"))

    async def acquire(owner: Conversation) -> None:
        owner.own(_release(order, "private-first"))
        owner.own(_release(order, "private-last"))
        raise ValueError("open failed")

    with pytest.raises(ValueError, match="open failed"):
        await prepared.open(acquire)
    assert order == ["private-last", "private-first"]
    await prepared.aclose()
    await prepared.aclose()
    assert order == ["private-last", "private-first", "shared-last", "shared-first"]


async def test_concurrent_close_waits_and_repeated_cancel_cannot_orphan_releases() -> None:
    prepared = PreparedAgent()
    entered, release, second_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    order: list[str] = []

    async def blocked_release() -> None:
        entered.set()
        await release.wait()
        order.append("last")

    prepared.own(_release(order, "first"))
    prepared.own(blocked_release)
    first = asyncio.create_task(prepared.aclose())

    async def close_again() -> None:
        second_entered.set()
        await prepared.aclose()

    second = asyncio.create_task(close_again())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(second_entered.wait(), 5)
        assert not first.done() and not second.done()
        first.cancel()
        first.cancel()

        async def forbidden(owner: Conversation) -> None:
            pytest.fail("closing Prepared invoked an open factory")

        with pytest.raises(PreparedClosed):
            await prepared.open(forbidden)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        await prepared.aclose()
        assert order == ["last", "first"]
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)


async def test_close_cancels_partial_open_and_waits_for_its_reverse_rollback() -> None:
    prepared = PreparedAgent()
    acquired, rollback_entered, release_rollback = asyncio.Event(), asyncio.Event(), asyncio.Event()
    order: list[str] = []
    prepared.own(_release(order, "shared"))

    async def rollback() -> None:
        rollback_entered.set()
        await release_rollback.wait()
        order.append("private-last")

    async def acquire(owner: Conversation) -> None:
        owner.own(_release(order, "private-first"))
        owner.own(rollback)
        acquired.set()
        await asyncio.Event().wait()

    opening = asyncio.create_task(prepared.open(acquire))
    await asyncio.wait_for(acquired.wait(), 5)
    closing = asyncio.create_task(prepared.aclose())
    try:
        await asyncio.wait_for(rollback_entered.wait(), 5)
        assert not closing.done()
        release_rollback.set()
        with pytest.raises(asyncio.CancelledError):
            await opening
        await closing
        assert order == ["private-last", "private-first", "shared"]
    finally:
        release_rollback.set()
        await asyncio.gather(opening, closing, return_exceptions=True)


async def test_close_without_operation_drains_bound_pass_before_resources() -> None:
    prepared = PreparedAgent()
    requested, drained = asyncio.Event(), asyncio.Event()
    order: list[str] = []

    class Pass:
        def request_close(self, cause: AbortCause) -> None:
            assert cause is AbortCause.OWNER_CLOSE
            order.append("abort-pass")
            requested.set()

        @property
        def drained(self) -> Awaitable[None]:
            return drained.wait()

    async def acquire(owner: Conversation) -> Conversation:
        owner.bind_pass(Pass())
        owner.own(_release(order, "conversation"))
        return owner

    await prepared.open(acquire)
    prepared.own(_release(order, "shared"))
    closing = asyncio.create_task(prepared.aclose())
    try:
        await asyncio.wait_for(requested.wait(), 5)
        assert order == ["abort-pass"]
        assert not closing.done()
        drained.set()
        await closing
        assert order == ["abort-pass", "conversation", "shared"]
    finally:
        drained.set()
        await closing


async def test_close_after_unbind_waits_for_in_progress_resource_release(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = PreparedAgent()
    entered, allow_release = asyncio.Event(), asyncio.Event()
    order: list[str] = []

    async def release() -> None:
        entered.set()
        await allow_release.wait()
        order.append("approval")

    async def acquire(owner: Conversation) -> Conversation:
        owner.own(release)
        return owner

    conversation = await prepared.open(acquire)

    shared_observations: list[bool] = []

    async def shared_release() -> None:
        shared_observations.append("approval" in order)
        order.append("shared")

    prepared.own(shared_release)
    releasing = asyncio.create_task(conversation.release(release))
    await asyncio.wait_for(entered.wait(), 5)
    drain_entered = asyncio.Event()
    original_finish = resources.finish_close

    async def finish(task: asyncio.Task[None]) -> None:
        if asyncio.current_task() is conversation._close_task and task in conversation._releasing:
            drain_entered.set()
        await original_finish(task)

    monkeypatch.setattr(resources, "finish_close", finish)
    closing = asyncio.create_task(prepared.aclose())
    try:
        await assert_entered_before_completion(drain_entered, closing)
        assert not closing.done()
        assert order == []
        allow_release.set()
        await asyncio.gather(releasing, closing)
        assert order == ["approval", "shared"]
        assert shared_observations == [True]
    finally:
        allow_release.set()
        await asyncio.gather(releasing, closing, return_exceptions=True)


async def test_failed_open_propagates_cancel_received_during_rollback() -> None:
    prepared = PreparedAgent()
    release = ReleaseGate()

    async def acquire(owner: Conversation) -> None:
        owner.own(release)
        raise ValueError("factory failed")

    opening = asyncio.create_task(prepared.open(acquire))
    try:
        await assert_cancel_during_rollback(opening, release)
        assert prepared._opening == []
        assert prepared._conversations == []
    finally:
        await prepared.aclose()


async def test_create_approval_on_closing_owner_closes_constructed_middleware(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import create_autospec

    from chrys.foundation.events.bus import EventBus
    from chrys.orchestration.invoker.runtime import ApprovalInputs, create_approval
    from chrys.service.agent_middleware import ApprovalMiddleware
    from chrys.service.approval.policy import ApprovalPolicy
    from chrys.service.profiles.agents.schema import ApprovalConfig

    owner = Conversation()
    await owner.aclose()
    closed: list[ApprovalMiddleware] = []
    original_close = ApprovalMiddleware.close

    async def close(self: ApprovalMiddleware) -> None:
        await original_close(self)
        closed.append(self)

    monkeypatch.setattr(ApprovalMiddleware, "close", create_autospec(original_close, side_effect=close))
    with pytest.raises(PreparedClosed):
        await create_approval(
            owner, ApprovalInputs(approval_policy=ApprovalPolicy(ApprovalConfig()), event_bus=EventBus())
        )
    assert len(closed) == 1
    assert owner._releases == []


async def test_own_or_release_registers_while_the_owner_is_open() -> None:
    owner = Conversation()
    order: list[str] = []

    await owner.own_or_release(_release(order, "client"))

    assert order == []
    await owner.aclose()
    assert order == ["client"]


async def test_a_resource_acquired_while_its_owner_closed_is_released_at_once() -> None:
    owner = Conversation()
    acquisition = asyncio.Event()
    order: list[str] = []

    async def acquire() -> None:
        await acquisition.wait()
        await owner.own_or_release(_release(order, "client"))

    acquiring = asyncio.create_task(acquire())
    await owner.aclose()
    acquisition.set()

    with pytest.raises(PreparedClosed):
        await acquiring
    assert order == ["client"]
    assert owner._releases == []


async def test_a_refused_resource_whose_release_fails_still_reports_the_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    owner = Conversation()
    await owner.aclose()

    async def broken() -> None:
        raise OSError("pool close failed")

    with pytest.raises(PreparedClosed):
        await owner.own_or_release(broken)
    assert "Error releasing a resource its closing owner refused" in caplog.text


async def test_a_refused_resource_is_released_even_when_its_caller_is_cancelled() -> None:
    owner = Conversation()
    await owner.aclose()
    release = ReleaseGate()

    await assert_cancel_during_rollback(asyncio.create_task(owner.own_or_release(release)), release)


async def test_cancelled_open_caller_can_join_same_prepared_close() -> None:
    prepared = PreparedAgent()
    entered = asyncio.Event()
    order: list[str] = []
    prepared.own(_release(order, "shared"))

    async def acquire(owner: Conversation) -> None:
        owner.own(_release(order, "private"))
        entered.set()
        await asyncio.Event().wait()

    async def build() -> None:
        try:
            await prepared.open(acquire)
        finally:
            await prepared.aclose()

    building = asyncio.create_task(build())
    await asyncio.wait_for(entered.wait(), 5)
    closing = asyncio.create_task(prepared.aclose())
    try:
        await asyncio.wait_for(asyncio.shield(closing), 5)
        with pytest.raises(asyncio.CancelledError):
            await building
        assert order == ["private", "shared"]
    finally:
        await asyncio.gather(building, closing, return_exceptions=True)
