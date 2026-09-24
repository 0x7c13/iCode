# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Policy configuration and callback contracts, independent of retry algorithms."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass, field
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.kernel import Agent, AgentSession, StallExhaustedAction, WireRetryPolicy
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.orchestration.invoker.attempts import WireRetryPolicyAdapter
from chrys.orchestration.invoker.resources import Conversation
from chrys.service.agent_middleware.control.approval import ApprovalMiddleware
from chrys.service.agent_middleware.control.ask_user import AskUserMiddleware
from chrys.service.agent_middleware.injection import InjectionMiddleware
from chrys.service.approval.policy import ApprovalPolicy
from chrys.service.llm.mock import MockChatClient
from chrys.service.profiles.agents.schema import ApprovalConfig
from tests.support.waiting import wait_for


@dataclass
class _RetryPorts:
    """Explicit callback signatures with independently mutable caller state."""

    budget: int = 2
    stall_budget: int = 1
    backoff: tuple[int, ...] = (3, 7)
    interrupted: bool = False
    hosted: tuple[str, ...] = ()
    preparations: int = 0
    sleeps: list[int] = field(default_factory=list)
    notices: list[tuple[str, int, int, int, BaseException]] = field(default_factory=list)

    def is_interrupted(self) -> bool:
        return self.interrupted

    async def sleep(self, seconds: int) -> bool:
        self.sleeps.append(seconds)
        return self.interrupted

    async def publish(self, message: str, attempt: int, maximum: int, delay: int, exc: BaseException) -> None:
        self.notices.append((message, attempt, maximum, delay, exc))

    def prepare(self) -> None:
        self.preparations += 1

    def probe(self) -> tuple[str, ...]:
        return self.hosted

    def policy(self, *, live: bool = False) -> WireRetryPolicyAdapter:
        return WireRetryPolicyAdapter(
            max_retries=(lambda: self.budget) if live else self.budget,
            stall_max_retries=(lambda: self.stall_budget) if live else self.stall_budget,
            stall_timeout_seconds=None,
            stall_exhausted_action=StallExhaustedAction.RAISE,
            backoff_schedule=(lambda: self.backoff) if live else self.backoff,
            interrupted=self.is_interrupted,
            interruptible_sleep=self.sleep,
            publish_retry=self.publish,
        )


@pytest.mark.parametrize("live", [False, True], ids=["snapshot", "live-readers"])
def test_configuration_preserves_snapshot_or_live_reads(live: bool) -> None:
    ports = _RetryPorts()
    policy = ports.policy(live=live)
    contract: WireRetryPolicy = policy
    assert (contract.max_retries, contract.stall_max_retries) == (2, 1)
    assert contract.stall_timeout_seconds is None
    assert contract.stall_exhausted_action is StallExhaustedAction.RAISE
    assert [contract.backoff_seconds(n) for n in (0, 1, 9)] == [3, 7, 7]
    assert policy.hosted_commits_in_flight is None
    assert contract.before_retry() is None
    assert ports.preparations == 0

    ports.budget, ports.stall_budget, ports.backoff = 0, 4, ()
    assert (contract.max_retries, contract.stall_max_retries) == ((0, 4) if live else (2, 1))
    assert contract.backoff_seconds(9) == (0 if live else 7)
    # The kernel Protocol exposes writable limits as well as readable limits.
    contract.max_retries, contract.stall_max_retries = 5, 6
    assert (contract.max_retries, contract.stall_max_retries) == (5, 6)


@pytest.mark.parametrize(
    "error,expected",
    [(ConnectionError("reset"), True), (TimeoutError("timeout"), True), (ValueError("invalid"), False)],
    ids=["connection", "timeout", "terminal"],
)
def test_retryability_uses_existing_foundation_classifier(error: BaseException, expected: bool) -> None:
    assert _RetryPorts().policy().is_retryable(error) is expected


async def test_callbacks_forward_values_and_observe_live_state() -> None:
    ports = _RetryPorts()
    policy = ports.policy()
    policy.prepare_retry = ports.prepare
    probe = ports.probe
    policy.hosted_commits_in_flight = probe
    assert policy.hosted_commits_in_flight is probe
    assert policy.hosted_commits_in_flight() == ()
    assert not policy.is_interrupted()
    assert await policy.sleep(3) is False
    ports.interrupted = True
    ports.hosted = ("shell", "shell_tool_result")
    assert policy.is_interrupted()
    assert await policy.sleep(7) is True
    assert ports.sleeps == [3, 7]
    assert policy.hosted_commits_in_flight() == ("shell", "shell_tool_result")
    policy.before_retry()
    assert ports.preparations == 1
    error = ConnectionError("original")
    await policy.on_retry("provider detail", 2, 4, 7, error)
    assert ports.notices == [("provider detail", 2, 4, 7, error)]
    assert ports.notices[0][-1] is error


@pytest.mark.parametrize("callback", ["sleep", "on_retry"])
@pytest.mark.parametrize("cancel", [False, True], ids=["completion", "cancellation"])
async def test_async_callbacks_await_caller_owned_future(callback: str, cancel: bool) -> None:
    policy = _RetryPorts().policy()
    entered = asyncio.Event()
    sleep_result = asyncio.get_running_loop().create_future()
    notice_result = asyncio.get_running_loop().create_future()

    def sleep(seconds: int) -> Awaitable[bool]:
        assert seconds == 7
        entered.set()
        return sleep_result

    error = ConnectionError("original")

    def publish(message: str, attempt: int, maximum: int, delay: int, exc: BaseException) -> Awaitable[None]:
        assert (message, attempt, maximum, delay) == ("detail", 2, 4, 7)
        assert exc is error
        entered.set()
        return notice_result

    policy.interruptible_sleep = sleep
    policy.publish_retry = publish
    operation = policy.sleep(7) if callback == "sleep" else policy.on_retry("detail", 2, 4, 7, error)
    task = asyncio.create_task(operation)
    try:
        await wait_for(entered.is_set, description="policy has entered caller callback")
        assert not task.done()
        result = sleep_result if callback == "sleep" else notice_result
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert result.cancelled()
        else:
            result.set_result(True if callback == "sleep" else None)
            assert await task is (True if callback == "sleep" else None)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for future in (sleep_result, notice_result):
            if not future.done():
                future.cancel()


@pytest.mark.parametrize("callback", ["sleep", "on_retry", "before_retry"])
@pytest.mark.parametrize("error", [RuntimeError("callback failed"), asyncio.CancelledError()])
async def test_callback_failures_propagate_unchanged(callback: str, error: BaseException) -> None:
    ports = _RetryPorts()
    policy = ports.policy()
    policy.interruptible_sleep = create_autospec(ports.sleep, side_effect=error)
    policy.publish_retry = create_autospec(ports.publish, side_effect=error)
    policy.prepare_retry = create_autospec(ports.prepare, side_effect=error)
    with pytest.raises(type(error)) as raised:
        if callback == "sleep":
            await policy.sleep(3)
        elif callback == "on_retry":
            await policy.on_retry("retry", 1, 2, 3, ConnectionError())
        else:
            policy.before_retry()
    assert raised.value is error


@pytest.fixture
async def policy_executor() -> AsyncIterator[TurnBindings]:
    bus = EventBus()
    executor = TurnBindings(
        conversation=Conversation(),
        agent=Agent(client=MockChatClient()),
        session=AgentSession(),
        event_bus=bus,
        approval_middleware=ApprovalMiddleware(ApprovalPolicy(ApprovalConfig()), bus),
        ask_user_middleware=AskUserMiddleware(bus),
        injection_middleware=InjectionMiddleware(),
    )
    executor.resource_scope.own(executor.approval.close)
    try:
        yield executor
    finally:
        await executor.resource_scope.aclose()


async def test_executor_defaults_and_configuration_snapshot(policy_executor: TurnBindings) -> None:
    executor = policy_executor
    policy = executor._build_wire_retry_policy()
    assert isinstance(policy, WireRetryPolicyAdapter)
    assert (policy.max_retries, policy.stall_max_retries) == (5, 5)
    assert policy.stall_timeout_seconds == 300.0
    assert policy.stall_exhausted_action is StallExhaustedAction.BLOCKING_FALLBACK
    assert [policy.backoff_seconds(n) for n in range(7)] == [3, 7, 15, 30, 60, 60, 60]
    assert policy.hosted_commits_in_flight is None
    executor._max_retries_override = 9
    executor._BACKOFF_SCHEDULE = ()
    executor._stream_attempt_timeout = 12.0
    assert (policy.max_retries, policy.stall_max_retries, policy.stall_timeout_seconds) == (5, 5, 300.0)
    assert policy.backoff_seconds(0) == 3
    rebuilt = executor._build_wire_retry_policy()
    assert (rebuilt.max_retries, rebuilt.stall_max_retries, rebuilt.stall_timeout_seconds) == (9, 9, 12.0)
    assert rebuilt.backoff_seconds(0) == 0
    assert not policy.is_interrupted()
    await executor.interrupt()
    assert policy.is_interrupted()
    assert await policy.sleep(0) is True


async def test_executor_wires_injection_notice_sleep_and_in_flight_probe(
    policy_executor: TurnBindings, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = policy_executor
    restore = create_autospec(executor._injection.restore_for_retry)
    sleep = create_autospec(executor._interruptible_sleep, return_value=False)
    publish = create_autospec(executor._publish_wire_retry_attempt)
    ports = _RetryPorts(hosted=("shell",))
    probe = ports.probe
    monkeypatch.setattr(executor._injection, "restore_for_retry", restore)
    monkeypatch.setattr(executor, "_interruptible_sleep", sleep)
    monkeypatch.setattr(executor, "_publish_wire_retry_attempt", publish)
    executor._hosted_commits_in_flight_probe = probe
    executor._hosted_commits_probe = lambda: ("pass-only",)
    policy = executor._build_wire_retry_policy()
    assert isinstance(policy, WireRetryPolicyAdapter)
    assert policy.hosted_commits_in_flight is probe
    assert policy.hosted_commits_in_flight() == ("shell",)
    ports.hosted = ()
    assert policy.hosted_commits_in_flight() == ()
    policy.before_retry()
    restore.assert_called_once_with()
    assert await policy.sleep(7) is False
    sleep.assert_awaited_once_with(7)
    error = ConnectionError("reset")
    await policy.on_retry("detail", 1, 5, 7, error)
    publish.assert_awaited_once_with("detail", 1, 5, 7, error)
