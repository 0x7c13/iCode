# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the per-chunk stream stall watchdog.

The stall watchdog in ``AttemptRunner._stream_single_attempt`` wraps each
``__anext__()`` on the stream in ``asyncio.wait_for`` with the stall
timeout, so the timer resets whenever a chunk arrives.  This measures
*stream-idle* time (matching httpx read-timeout semantics) rather than
total wall-clock time.

These tests verify:

1. A healthy stream whose total wall time exceeds the stall timeout but
   whose individual chunk gaps stay under it must NOT stall (regression
   test for the pre-fix behaviour, where a single outer ``wait_for``
   capped the whole attempt and tool/chunk accumulation could blow the
   budget).
2. A stream with an idle gap longer than the stall timeout DOES stall
   and triggers the retry path.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Annotated
from unittest.mock import create_autospec

import pytest
from anthropic.types.beta import (
    BetaInputJSONDelta,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawContentBlockStopEvent,
    BetaRawMessageStopEvent,
    BetaRawMessageStreamEvent,
    BetaTextBlock,
    BetaTextDelta,
    BetaToolUseBlock,
)

from chrys.foundation.errors import ProviderResponseError
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationMessage, InvocationRetryAttempt
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.foundation.retry import HistorySnapshot, StreamStall, StreamStallExhausted
from chrys.kernel import (
    RETRY_BOUNDARY_UPDATE_KEY,
    Agent,
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    ChatMiddlewareLayer,
    Content,
    FunctionTool,
    Message,
    ResponseStream,
)
from chrys.kernel.middleware import ChatContext
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.orchestration.invoker.attempts import continuation_token_observer_for
from chrys.orchestration.invoker.origin import BoundEmitter
from chrys.orchestration.invoker.resources import Conversation
from chrys.service.agent_middleware import ApprovalMiddleware, AskUserMiddleware
from chrys.service.agent_middleware.injection import InjectionMiddleware
from chrys.service.agent_middleware.response_validation import (
    ResponseValidationMiddleware,
    RetryableResponseValidationError,
    TerminalResponseValidationError,
    ValidationRetryExemption,
)
from chrys.service.llm.anthropic_messages import AnthropicMessagesClient
from chrys.service.llm.mock import MockResponse
from tests.support.pipeline_helpers import create_test_engine


def _make_executor(
    *,
    agent: Agent | None = None,
    session: AgentSession | None = None,
    injection: InjectionMiddleware | None = None,
    session_id: str | None = None,
) -> TurnBindings:
    """Construct shared shell/L0 dependencies once, before either captures them."""
    return TurnBindings(
        conversation=Conversation(),
        agent=agent if agent is not None else create_autospec(Agent, instance=True),
        session=session if session is not None else AgentSession(),
        event_bus=EventBus(),
        session_id=session_id,
        approval_middleware=create_autospec(ApprovalMiddleware, instance=True),
        ask_user_middleware=create_autospec(AskUserMiddleware, instance=True),
        injection_middleware=injection if injection is not None else InjectionMiddleware(),
    )


def _finals(events: list) -> list[str]:
    return [e.text for e in events if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final]


class TestStreamStallPerChunk:
    @pytest.mark.asyncio
    async def test_retry_events_bind_only_the_stream_stall_display_message(self) -> None:
        executor = _make_executor(session_id="session-retry")
        executor.trajectory_context = None
        executor._hosted_bridge = None
        executor._bound_emitter = executor._invocation_publishers.bind(executor.inputs.origin)
        retries: list[InvocationRetryAttempt] = []

        async def _capture(event: InvocationRetryAttempt) -> None:
            retries.append(event)

        await executor._bus.subscribe(InvocationRetryAttempt, _capture)

        await executor._attempts._publish_retry_attempt("Stream stalled", 1, 5, 3, StreamStall("stalled"))
        await executor._attempts._publish_retry_attempt("connection reset", 2, 5, 7, ConnectionError("reset"))

        stall, generic = retries
        assert (stall.message, stall.attempt, stall.max_attempts, stall.delay_seconds, stall.session_id) == (
            "Stream stalled",
            1,
            5,
            3,
            "session-retry",
        )
        assert stall.display_message is not None
        assert stall.display_message.definition.key == "retry.stream_stalled"
        assert stall.display_message.args == ()
        assert stall.detail == ""
        assert (generic.message, generic.attempt, generic.max_attempts, generic.delay_seconds) == (
            "connection reset",
            2,
            5,
            7,
        )
        assert generic.display_message is None
        assert generic.detail == ""

    def test_wire_retry_policy_uses_injected_override(self):
        executor = _make_executor(injection=SimpleNamespace(restore_for_retry=lambda: None))
        executor.trajectory_context = None
        executor._max_retries_override = 9
        executor._stream_attempt_timeout = 12.0
        executor._hosted_commits_in_flight_probe = None
        executor._attempts._publish_retry_attempt = lambda *_args: None
        executor._interruptible_sleep = lambda _seconds: None

        policy = executor._build_wire_retry_policy()

        assert policy.max_retries == 9
        assert policy.stall_max_retries == 9

    def test_no_override_lazily_honors_instance_max_retries_shadow(self):
        executor = _make_executor()
        executor.trajectory_context = None
        executor._MAX_RETRIES = 2

        assert executor._effective_max_retries() == 2

    def test_no_override_lazily_honors_class_max_retries_shadow(self, monkeypatch):
        executor = _make_executor()
        executor.trajectory_context = None
        monkeypatch.setattr(TurnBindings, "_MAX_RETRIES", 4)

        assert executor._effective_max_retries() == 4

    @pytest.mark.asyncio
    async def test_retry_boundary_resets_private_streaming_text_buffer(self):
        updates = [
            AgentResponseUpdate(contents=[Content.from_text("failed partial")], role="assistant"),
            AgentResponseUpdate(
                contents=[],
                additional_properties={RETRY_BOUNDARY_UPDATE_KEY: True},
            ),
            AgentResponseUpdate(contents=[Content.from_text("fresh answer")], role="assistant"),
        ]

        async def _updates():
            for update in updates:
                yield update

        class _Agent:
            def run(self, *_args, **_kwargs):
                return ResponseStream(_updates(), finalizer=AgentResponse.from_updates)

        published: list[object] = []

        async def _publish(event: object) -> None:
            published.append(event)

        executor = _make_executor(agent=_Agent())
        executor.trajectory_context = None
        executor._intermediate_buffer = None
        executor._stream_attempt_timeout = 1.0
        executor._attempt_handle.task = None
        executor._session_id = "session"
        executor._bus = SimpleNamespace(publish=_publish)
        executor._bound_emitter = BoundEmitter(
            executor._bus, InvocationOrigin("turn", executor._session_id or "", "test", None)
        )

        await executor._attempts._stream_single_attempt([], {}, watchdog=False)

        streamed = [
            event.text for event in published if (isinstance(event, InvocationMessage) and event.origin.kind == "turn")
        ]
        assert streamed == ["fresh answer"]

    def test_forced_stateless_executor_never_enables_service_storage(self):
        executor = _make_executor(
            agent=SimpleNamespace(client=SimpleNamespace(STORES_BY_DEFAULT=True, FORCES_STATELESS=True))
        )
        executor.trajectory_context = None
        executor._chat_options = {
            "store": True,
            "previous_response_id": "resp_1",
            "extra_body": {"store": True},
        }

        assert executor.backend.service_session_storage_enabled is False

    def test_forced_stateless_executor_restore_site_receives_capability_flag(self):
        executor = _make_executor(
            agent=SimpleNamespace(client=SimpleNamespace(FORCES_STATELESS=True)),
            injection=SimpleNamespace(restore_for_retry=lambda: None),
            session=SimpleNamespace(service_session_id="session-handle"),
        )
        executor.trajectory_context = None
        run_kwargs = {
            "options": {
                "store": True,
                "continuation_token": {"response_id": "pending"},
                "background": True,
                "extra_body": {"previous_response_id": "nested", "background": True},
            },
            "client_kwargs": {
                "conversation_id": "kwarg",
                "continuation_token": {"response_id": "kwarg-pending"},
            },
        }

        executor._attempts._restore_service_retry_inputs(run_kwargs)

        assert run_kwargs["options"] == {"store": True, "extra_body": {}}
        assert run_kwargs["client_kwargs"] == {}

    @pytest.mark.asyncio
    async def test_function_result_rearms_service_watchdog_for_next_provider_pull(self):
        async def _updates():
            yield AgentResponseUpdate(
                contents=[Content.from_function_call("c1", "echo", arguments={})],
                role="assistant",
            )
            yield AgentResponseUpdate(
                contents=[Content.from_function_result("c1", result="done")],
                role="tool",
            )
            await asyncio.sleep(30)

        class _Agent:
            def run(self, *_args, **_kwargs):
                return ResponseStream(_updates(), finalizer=AgentResponse.from_updates)

        executor = _make_executor(agent=_Agent())
        executor.trajectory_context = None
        executor._intermediate_buffer = None
        executor._stream_attempt_timeout = 0.01
        executor._attempt_handle.task = None
        executor._session_id = "session"
        executor._bus = SimpleNamespace(publish=lambda _event: asyncio.sleep(0))
        executor._bound_emitter = BoundEmitter(
            executor._bus, InvocationOrigin("turn", executor._session_id or "", "test", None)
        )

        with pytest.raises(StreamStall):
            await asyncio.wait_for(
                executor._attempts._stream_single_attempt([], {}, watchdog=True),
                timeout=1,
            )

    @pytest.mark.asyncio
    async def test_informational_function_call_keeps_stream_watchdog_armed(self):
        """Hosted calls have no local result await, so the next idle gap must still time out."""
        import asyncio as _asyncio

        hosted_update = AgentResponseUpdate(
            contents=[
                Content.from_function_call(
                    "hosted-1",
                    "web_search",
                    arguments={"query": "chrys"},
                    informational_only=True,
                )
            ],
            role="assistant",
        )

        async def _updates():
            yield hosted_update
            await _asyncio.sleep(30)

        class _Agent:
            def run(self, *_args, **_kwargs):
                return ResponseStream(_updates(), finalizer=AgentResponse.from_updates)

        executor = _make_executor(agent=_Agent())
        executor.trajectory_context = None
        executor._intermediate_buffer = None
        executor._stream_attempt_timeout = 0.01
        executor._attempt_handle.task = None
        executor._session_id = "session"

        async def _publish(event: object) -> None:
            raise AssertionError(f"unexpected publish for hosted-call stall path: {event!r}")

        executor._bus = SimpleNamespace(publish=_publish)
        executor._bound_emitter = BoundEmitter(
            executor._bus, InvocationOrigin("turn", executor._session_id or "", "test", None)
        )

        with pytest.raises(StreamStall):
            await _asyncio.wait_for(executor._attempts._stream_single_attempt([], {}), timeout=5.0)

    @pytest.mark.asyncio
    async def test_function_call_with_trailing_text_keeps_watchdog_suspended(self):
        """One tool-response-boundary update must not re-arm before the tool await."""
        import asyncio as _asyncio

        boundary_update = AgentResponseUpdate(
            contents=[
                Content.from_function_call("call-1", "slow_tool", arguments={}),
                Content.from_text("trailing intermediate text"),
            ],
            role="assistant",
        )

        async def _updates():
            yield boundary_update
            # Simulate the tool loop performing a legitimate long-running tool inside
            # the next __anext__() after it exposes the model's tool-response boundary.
            await _asyncio.sleep(0.25)
            yield AgentResponseUpdate(contents=[Content.from_text("done")], role="assistant")

        class _Agent:
            def run(self, *_args, **_kwargs):
                return ResponseStream(_updates(), finalizer=AgentResponse.from_updates)

        published: list[object] = []

        async def _publish(event: object) -> None:
            published.append(event)

        executor = _make_executor(agent=_Agent())
        executor.trajectory_context = None
        executor._intermediate_buffer = None
        executor._stream_attempt_timeout = 0.05
        executor._attempt_handle.task = None
        executor._session_id = "session"
        executor._bus = SimpleNamespace(publish=_publish)
        executor._bound_emitter = BoundEmitter(
            executor._bus, InvocationOrigin("turn", executor._session_id or "", "test", None)
        )

        result = await _asyncio.wait_for(executor._attempts._stream_single_attempt([], {}), timeout=5.0)

        assert result.messages
        assert any(
            (isinstance(event, InvocationMessage) and event.origin.kind == "turn") and event.text == "done"
            for event in published
        )

    @pytest.mark.asyncio
    async def test_anthropic_argument_heartbeats_keep_stream_watchdog_armed(self):
        """Buffered input JSON deltas still reset the real TurnBindings watchdog."""
        import asyncio as _asyncio

        # Timing margins (Windows CI runs under -n 8 and can starve a worker
        # for hundreds of ms): 15 events x 0.12 s >= 1.8 s total, which is
        # guaranteed to exceed the 1.5 s per-chunk budget (asyncio.sleep never
        # undershoots), so a heartbeat-less drain still stalls.  Meanwhile each
        # healthy gap is 0.12 s, leaving ~1.4 s of scheduler-jitter headroom
        # before the watchdog could fire spuriously.
        argument_deltas = ['{"payload":"', *["a"] * 10, '"}']
        events: list[BetaRawMessageStreamEvent] = [
            BetaRawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=BetaToolUseBlock(type="tool_use", id="call-a", name="slow_tool", input={}),
            ),
            *[
                BetaRawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=0,
                    delta=BetaInputJSONDelta(type="input_json_delta", partial_json=partial_json),
                )
                for partial_json in argument_deltas
            ],
            BetaRawContentBlockStopEvent(type="content_block_stop", index=0),
            BetaRawMessageStopEvent(type="message_stop"),
        ]

        class _DelayedMessages:
            async def create(self, **_kwargs):
                async def _stream():
                    for event in events:
                        await _asyncio.sleep(0.12)
                        yield event

                return _stream()

        anthropic_client = SimpleNamespace(
            base_url="https://api.anthropic.com", default_headers={}, beta=SimpleNamespace(messages=_DelayedMessages())
        )
        client = AnthropicMessagesClient(model="kimi-k3", sdk_client=anthropic_client)  # type: ignore[arg-type]
        validated_client = ChatMiddlewareLayer(
            client,
            middleware=[ResponseValidationMiddleware(max_retries=0)],
        )

        class _Agent:
            def run(self, *_args, **_kwargs):
                return validated_client.get_response([], options={}, stream=True)

        executor = _make_executor(agent=_Agent())
        executor.trajectory_context = None
        executor._intermediate_buffer = None
        executor._stream_attempt_timeout = 1.5
        executor._attempt_handle.task = None
        executor._session_id = "session"
        executor._bus = SimpleNamespace()
        executor._bound_emitter = BoundEmitter(
            executor._bus, InvocationOrigin("turn", executor._session_id or "", "test", None)
        )

        result = await _asyncio.wait_for(executor._attempts._stream_single_attempt([], {}), timeout=10.0)

        assert result.messages[0].contents[0].parse_arguments() == {"payload": "a" * 10}

    @pytest.mark.asyncio
    async def test_anthropic_text_heartbeats_keep_stream_watchdog_armed(self):
        """Buffered text deltas still reset the watchdog for every provider chunk."""
        import asyncio as _asyncio

        # Same timing margins as the argument-delta test above: 15 events x
        # 0.12 s >= 1.8 s total > 1.5 s budget, 0.12 s healthy gaps.
        text_deltas = ["a"] * 12
        events: list[BetaRawMessageStreamEvent] = [
            BetaRawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=BetaTextBlock(type="text", text="", citations=None),
            ),
            *[
                BetaRawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=0,
                    delta=BetaTextDelta(type="text_delta", text=text),
                )
                for text in text_deltas
            ],
            BetaRawContentBlockStopEvent(type="content_block_stop", index=0),
            BetaRawMessageStopEvent(type="message_stop"),
        ]

        class _DelayedMessages:
            async def create(self, **_kwargs):
                async def _stream():
                    for event in events:
                        await _asyncio.sleep(0.12)
                        yield event

                return _stream()

        anthropic_client = SimpleNamespace(
            base_url="https://api.anthropic.com", default_headers={}, beta=SimpleNamespace(messages=_DelayedMessages())
        )
        client = AnthropicMessagesClient(model="kimi-k3", sdk_client=anthropic_client)  # type: ignore[arg-type]
        validated_client = ChatMiddlewareLayer(
            client,
            middleware=[ResponseValidationMiddleware(max_retries=0)],
        )

        class _Agent:
            def run(self, *_args, **_kwargs):
                return validated_client.get_response([], options={}, stream=True)

        executor = _make_executor(agent=_Agent())
        executor.trajectory_context = None
        executor._intermediate_buffer = None
        executor._stream_attempt_timeout = 1.5
        executor._attempt_handle.task = None
        executor._session_id = "session"

        async def _publish(_event: object) -> None:
            return None

        executor._bus = SimpleNamespace(publish=_publish)
        executor._bound_emitter = BoundEmitter(
            executor._bus, InvocationOrigin("turn", executor._session_id or "", "test", None)
        )

        result = await _asyncio.wait_for(executor._attempts._stream_single_attempt([], {}), timeout=10.0)

        assert result.text == "".join(text_deltas)

    @pytest.mark.asyncio
    async def test_stall_fallback_uses_interruptible_agent_task(self, monkeypatch):
        """The non-stream fallback must go through AttemptRunner._run_agent."""
        executor = _make_executor(injection=InjectionMiddleware())
        executor.trajectory_context = None
        executor._loop_recorder = None
        executor._BACKOFF_SCHEDULE = (0, 0)
        executor._MAX_RETRIES = 1
        executor._rollback.snapshot = lambda: HistorySnapshot(messages=[], compressed_count=0)
        executor._rollback.restore = lambda _snapshot: None

        async def _publish_retry_attempt(*_args) -> None:
            return None

        async def _interruptible_sleep(_seconds: int) -> bool:
            return False

        async def _stream_single_attempt(_current_input, _run_kwargs):
            raise AssertionError("StreamRetryLoop.run is monkeypatched and should not call attempt")

        sentinel = object()
        fallback_calls: list[tuple[list, dict]] = []

        async def _run_agent(current_input, run_kwargs):
            fallback_calls.append((current_input, run_kwargs))
            return sentinel

        async def _exhausted(_loop, _attempt):
            raise StreamStallExhausted

        executor._attempts._publish_retry_attempt = _publish_retry_attempt
        executor._attempts._interruptible_sleep = _interruptible_sleep
        executor._attempts._stream_single_attempt = _stream_single_attempt
        executor._attempts._run_agent = _run_agent
        monkeypatch.setattr("chrys.orchestration.invoker.attempts.StreamRetryLoop.run", _exhausted)

        result = await executor._attempts._stream_with_retry([], {})

        assert result is sentinel
        assert fallback_calls == [([], {})]

    @pytest.mark.asyncio
    async def test_stored_blocking_loop_uses_injected_budget_and_validation_exemption(self, monkeypatch):
        captured: dict[str, object] = {}

        class _Loop:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            async def run(self, _attempt) -> object:
                return object()

        executor = _make_executor(
            agent=SimpleNamespace(client=SimpleNamespace(STORES_BY_DEFAULT=True)), injection=InjectionMiddleware()
        )
        executor.trajectory_context = None
        executor._chat_options = None
        executor._max_retries_override = 6
        executor._BACKOFF_SCHEDULE = (0,)
        executor._build_run_kwargs = lambda _policy: {}
        monkeypatch.setattr("chrys.orchestration.invoker.attempts.StreamRetryLoop", _Loop)

        await executor._attempts._run_blocking([], {}, service_side=True)

        assert captured["max_retries"] == 6
        assert callable(captured["retry_exemption"])

    @pytest.mark.asyncio
    async def test_stored_streaming_loop_uses_injected_budget_and_validation_exemption(self, monkeypatch):
        captured: dict[str, object] = {}
        sentinel = object()

        class _Loop:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            async def run(self, _attempt) -> object:
                return sentinel

        executor = _make_executor(injection=InjectionMiddleware())
        executor.trajectory_context = None
        executor._max_retries_override = 6
        executor._BACKOFF_SCHEDULE = (0,)
        monkeypatch.setattr("chrys.orchestration.invoker.attempts.StreamRetryLoop", _Loop)

        result = await executor._attempts._stream_with_retry([], {})

        assert result is sentinel
        assert captured["max_retries"] == 6
        assert callable(captured["retry_exemption"])

    @pytest.mark.parametrize("transient_budget", [0, 1])
    @pytest.mark.asyncio
    async def test_stored_validation_uses_full_exempt_budget(self, transient_budget: int):
        executor = _make_executor(injection=InjectionMiddleware())
        executor.trajectory_context = None
        executor._max_retries_override = transient_budget
        executor._BACKOFF_SCHEDULE = (0,)
        executor._loop_recorder = SimpleNamespace(committed_count=0)
        executor._hosted_commits_probe = None
        executor._rollback.snapshot = lambda: HistorySnapshot(messages=[], compressed_count=0)
        executor._rollback.restore = lambda _snapshot: None
        executor._attempts._restore_service_retry_inputs = lambda _kwargs: None
        retry_events: list[tuple[int, int, int]] = []

        async def _publish_retry_attempt(
            _message: str,
            attempt: int,
            max_attempts: int,
            delay: int,
            _exc: BaseException,
        ) -> None:
            retry_events.append((attempt, max_attempts, delay))

        async def _interruptible_sleep(_seconds: int) -> bool:
            return False

        validation_errors = [
            RetryableResponseValidationError(
                reason,
                exemption=ValidationRetryExemption(attempt=attempt, max_attempts=3, delay_seconds=0),
            )
            for attempt, reason in enumerate(("empty contents", "whitespace", "leaked marker"), start=1)
        ]
        sentinel = object()
        outcomes: list[object] = [*validation_errors, sentinel]

        async def _stream_single_attempt(_current_input, _run_kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        executor._attempts._publish_retry_attempt = _publish_retry_attempt
        executor._attempts._interruptible_sleep = _interruptible_sleep
        executor._attempts._stream_single_attempt = _stream_single_attempt

        result = await executor._attempts._stream_with_retry([], {})

        assert result is sentinel
        assert retry_events == [(1, 3, 0), (2, 3, 0), (3, 3, 0)]

    @pytest.mark.asyncio
    async def test_stored_zero_budget_stall_makes_one_blocking_fallback(self):
        executor = _make_executor(injection=InjectionMiddleware())
        executor.trajectory_context = None
        executor._max_retries_override = 0
        executor._BACKOFF_SCHEDULE = (0,)
        executor._loop_recorder = SimpleNamespace(committed_count=0)
        executor._hosted_commits_probe = None
        executor._rollback.snapshot = lambda: HistorySnapshot(messages=[], compressed_count=0)
        executor._rollback.restore = lambda _snapshot: None
        executor._attempts._restore_service_retry_inputs = lambda _kwargs: None
        stream_attempts = 0
        fallback_calls = 0
        sentinel = object()

        async def _stream_single_attempt(_current_input, _run_kwargs):
            nonlocal stream_attempts
            stream_attempts += 1
            raise StreamStall("stalled")

        async def _run_agent(_current_input, _run_kwargs):
            nonlocal fallback_calls
            fallback_calls += 1
            return sentinel

        async def _publish_retry_attempt(*_args) -> None:
            return None

        async def _interruptible_sleep(_seconds: int) -> bool:
            return False

        executor._attempts._stream_single_attempt = _stream_single_attempt
        executor._attempts._run_agent = _run_agent
        executor._attempts._publish_retry_attempt = _publish_retry_attempt
        executor._attempts._interruptible_sleep = _interruptible_sleep

        result = await executor._attempts._stream_with_retry([], {})

        assert result is sentinel
        assert stream_attempts == 1
        assert fallback_calls == 1

    @pytest.mark.asyncio
    async def test_stored_zero_budget_exempt_retry_then_stall_falls_back_to_blocking(self):
        """An exempt validation retry at budget 0 still leaves the stall→blocking degrade intact."""
        executor = _make_executor(injection=InjectionMiddleware())
        executor.trajectory_context = None
        executor._max_retries_override = 0
        executor._BACKOFF_SCHEDULE = (0,)
        executor._loop_recorder = SimpleNamespace(committed_count=0)
        executor._hosted_commits_probe = None
        executor._rollback.snapshot = lambda: HistorySnapshot(messages=[], compressed_count=0)
        executor._rollback.restore = lambda _snapshot: None
        executor._attempts._restore_service_retry_inputs = lambda _kwargs: None
        retry_events: list[tuple[int, int, int]] = []
        fallback_calls = 0
        sentinel = object()
        outcomes: list[BaseException] = [
            RetryableResponseValidationError(
                "empty contents",
                exemption=ValidationRetryExemption(attempt=1, max_attempts=3, delay_seconds=0),
            ),
            StreamStall("stalled"),
        ]

        async def _stream_single_attempt(_current_input, _run_kwargs):
            raise outcomes.pop(0)

        async def _run_agent(_current_input, _run_kwargs):
            nonlocal fallback_calls
            fallback_calls += 1
            return sentinel

        async def _publish_retry_attempt(
            _message: str,
            attempt: int,
            max_attempts: int,
            delay: int,
            _exc: BaseException,
        ) -> None:
            retry_events.append((attempt, max_attempts, delay))

        async def _interruptible_sleep(_seconds: int) -> bool:
            return False

        executor._attempts._stream_single_attempt = _stream_single_attempt
        executor._attempts._run_agent = _run_agent
        executor._attempts._publish_retry_attempt = _publish_retry_attempt
        executor._attempts._interruptible_sleep = _interruptible_sleep

        result = await executor._attempts._stream_with_retry([], {})

        assert result is sentinel
        assert not outcomes
        assert fallback_calls == 1
        assert retry_events == [(1, 3, 0)]

    @pytest.mark.asyncio
    async def test_stored_identical_reason_gives_terminal_after_one_exempt_event(self):
        """The middleware's identical-reason give-up escapes the loop as a terminal raise."""
        executor = _make_executor(injection=InjectionMiddleware())
        executor.trajectory_context = None
        executor._max_retries_override = 5
        executor._BACKOFF_SCHEDULE = (0,)
        executor._loop_recorder = SimpleNamespace(committed_count=0)
        executor._hosted_commits_probe = None
        executor._rollback.snapshot = lambda: HistorySnapshot(messages=[], compressed_count=0)
        executor._rollback.restore = lambda _snapshot: None
        executor._attempts._restore_service_retry_inputs = lambda _kwargs: None
        retry_events: list[tuple[int, int, int]] = []
        outcomes: list[BaseException] = [
            RetryableResponseValidationError(
                "empty contents",
                exemption=ValidationRetryExemption(attempt=1, max_attempts=3, delay_seconds=0),
            ),
            TerminalResponseValidationError("empty contents"),
        ]

        async def _stream_single_attempt(_current_input, _run_kwargs):
            raise outcomes.pop(0)

        async def _publish_retry_attempt(
            _message: str,
            attempt: int,
            max_attempts: int,
            delay: int,
            _exc: BaseException,
        ) -> None:
            retry_events.append((attempt, max_attempts, delay))

        async def _interruptible_sleep(_seconds: int) -> bool:
            return False

        executor._attempts._stream_single_attempt = _stream_single_attempt
        executor._attempts._publish_retry_attempt = _publish_retry_attempt
        executor._attempts._interruptible_sleep = _interruptible_sleep

        with pytest.raises(TerminalResponseValidationError):
            await executor._attempts._stream_with_retry([], {})

        assert not outcomes
        assert retry_events == [(1, 3, 0)]

    @pytest.mark.asyncio
    async def test_stall_exhaustion_rolls_back_state_before_blocking_fallback(self):
        """Drive the REAL StreamRetryLoop to exhaustion: every stalled attempt
        (including the final, exhausted one) must restore history and then
        replay injection state before the blocking fallback issues its
        request, all inside one begin/end retry window."""

        order: list[str] = []
        snapshot = HistorySnapshot(messages=[], compressed_count=0)

        class _RecordingInjection:
            def begin_retry(self) -> None:
                order.append("begin_retry")

            def end_retry(self) -> None:
                order.append("end_retry")

            def restore_for_retry(self) -> None:
                order.append("restore_for_retry")

        executor = _make_executor(
            session=SimpleNamespace(service_session_id="failed-service-id"), injection=_RecordingInjection()
        )
        executor.trajectory_context = None
        executor._loop_recorder = None
        executor._BACKOFF_SCHEDULE = (0, 0)
        executor._MAX_RETRIES = 1
        executor._rollback.snapshot = lambda: snapshot
        executor._rollback.restore = lambda _snap: order.append("restore_history")

        async def _publish_retry_attempt(*_args) -> None:
            return None

        async def _interruptible_sleep(_seconds: int) -> bool:
            return False

        async def _stream_single_attempt(_current_input, _run_kwargs):
            order.append("attempt")
            raise StreamStall

        sentinel = object()

        async def _run_agent(_current_input, _run_kwargs):
            order.append("fallback")
            return sentinel

        executor._attempts._publish_retry_attempt = _publish_retry_attempt
        executor._attempts._interruptible_sleep = _interruptible_sleep
        executor._attempts._stream_single_attempt = _stream_single_attempt
        executor._attempts._run_agent = _run_agent

        result = await executor._attempts._stream_with_retry([], {})

        assert result is sentinel
        assert executor._session.service_session_id is None
        # MAX_RETRIES=1 → two stalled attempts.  The mid-loop retry AND the
        # exhaustion branch each restore history then injection replay
        # (restore_on_stall_exhaustion=True wiring), before the fallback.
        assert order == [
            "begin_retry",
            "attempt",
            "restore_history",
            "restore_for_retry",
            "attempt",
            "restore_history",
            "restore_for_retry",
            "fallback",
            "end_retry",
        ]

    def test_service_retry_restore_drops_all_failed_attempt_handles(self):
        replayed: list[bool] = []
        executor = _make_executor(
            injection=SimpleNamespace(restore_for_retry=lambda: replayed.append(True)),
            session=SimpleNamespace(service_session_id="session-handle"),
        )
        executor.trajectory_context = None
        run_kwargs = {
            "options": {
                "store": True,
                "conversation_id": "option-handle",
                "continuation_token": {"response_id": "pending"},
                "extra_body": {
                    "previous_response_id": "nested-option-handle",
                    "continuation_token": "nested-pending",
                },
            },
            "client_kwargs": {
                "conversation": "kwarg-handle",
                "extra_body": {"conversation_id": "nested-kwarg-handle"},
            },
        }

        executor._attempts._restore_service_retry_inputs(run_kwargs)

        # A live continuation token means this restore precedes a poll
        # resume, which never consumes a replay — the transaction is kept.
        assert replayed == []
        assert executor._session.service_session_id is None
        assert run_kwargs["options"] == {
            "store": True,
            "continuation_token": {"response_id": "pending"},
            "extra_body": {"continuation_token": "nested-pending"},
        }
        assert run_kwargs["client_kwargs"] == {"extra_body": {}}

    def test_continuation_token_observer_survives_handle_stripping_restore(self):
        executor = _make_executor(
            injection=SimpleNamespace(restore_for_retry=lambda: None),
            session=SimpleNamespace(service_session_id="session-handle"),
        )
        executor.trajectory_context = None
        run_kwargs: dict[str, object] = {"options": {"store": True, "conversation_id": "handle"}}
        observe = continuation_token_observer_for(run_kwargs)

        observe({"response_id": "pending"})
        assert run_kwargs["options"]["continuation_token"] == {"response_id": "pending"}

        executor._attempts._restore_service_retry_inputs(run_kwargs)
        assert run_kwargs["options"]["continuation_token"] == {"response_id": "pending"}
        assert "conversation_id" not in run_kwargs["options"]

        # The restore replaced the options dict; the observer must keep
        # targeting the live one.
        observe({"response_id": "renewed"})
        assert run_kwargs["options"]["continuation_token"] == {"response_id": "renewed"}
        observe(None)
        assert "continuation_token" not in run_kwargs["options"]

    def test_pending_token_seeds_rebuilt_run_kwargs_and_clears_on_terminal(self):
        """A user retry rebuilds run kwargs — the announced token must ride along.

        Without the carry-over, a retry after a gate-blocked poll failure
        would CREATE a second background response while the announced one
        keeps running remotely.
        """
        executor = _make_executor(
            injection=SimpleNamespace(
                drain_consumed_injection_messages=list,
                commit_logical_call=lambda: None,
            ),
            session=SimpleNamespace(),
        )
        executor.trajectory_context = None
        executor._loop_recorder = None
        executor.tool_events = SimpleNamespace()
        executor._ask_user = SimpleNamespace()
        executor.approval = SimpleNamespace()
        executor._extra_function_middleware = ()
        executor._sleep = SimpleNamespace()
        executor._compaction_strategy = None
        executor._chat_options = {"store": True}
        executor.inputs.pending_continuation_token = None

        first = TurnBindings._build_run_kwargs(executor)
        assert "continuation_token" not in first["options"]

        token = {"response_id": "bg-pending"}
        first["client_kwargs"]["continuation_token_observer"](token)
        assert executor.inputs.pending_continuation_token == token
        assert first["options"]["continuation_token"] == token

        rebuilt = TurnBindings._build_run_kwargs(executor)
        assert rebuilt["options"]["continuation_token"] == token

        rebuilt["client_kwargs"]["continuation_token_observer"](None)
        assert executor.inputs.pending_continuation_token is None
        assert "continuation_token" not in rebuilt["options"]
        assert "continuation_token" not in TurnBindings._build_run_kwargs(executor)["options"]

    def test_restore_skips_injection_replay_while_continuation_token_lives(self):
        replays: list[str] = []
        executor = _make_executor(
            injection=SimpleNamespace(restore_for_retry=lambda: replays.append("replayed")),
            session=SimpleNamespace(service_session_id="session-handle"),
        )
        executor.trajectory_context = None
        run_kwargs: dict[str, object] = {"options": {"store": True, "continuation_token": {"response_id": "pending"}}}

        executor._attempts._restore_service_retry_inputs(run_kwargs)
        # The retry resumes the already-created background response: the
        # create that consumed the injections succeeded, so the consumed
        # transaction must survive for the terminal weave.
        assert replays == []
        assert run_kwargs["options"]["continuation_token"] == {"response_id": "pending"}

        run_kwargs["options"].pop("continuation_token")
        executor._attempts._restore_service_retry_inputs(run_kwargs)
        assert replays == ["replayed"]

    @pytest.mark.asyncio
    async def test_long_stream_with_fast_chunks_does_not_stall(self, tmp_path, monkeypatch):
        """Regression: cumulative wall time > stall timeout, but each chunk
        arrives well within the per-chunk budget — must NOT stall.

        With the previous outer ``asyncio.wait_for`` wrapping the entire
        attempt, this case failed: 40 chunks at 0.05 s each = 2.0 s total, which
        blew the 0.5 s budget even though the stream was actively sending
        data the whole time.

        NOTE: ``ResponseValidationMiddleware`` (innermost chat middleware)
        drains the full inner stream before handing a replay to the
        outer iterator, so from chrys's ``_stream_single_attempt``
        perspective all 40 chunks arrive in a single burst after the
        middleware finishes draining.  The per-chunk watchdog therefore
        now wraps the entire drain.  Use a generous budget so the real
        invariant (healthy stream → no spurious retry) still holds.
        """
        # Flatten backoff so any (unexpected) retries don't slow the test.
        monkeypatch.setattr(TurnBindings, "_BACKOFF_SCHEDULE", (0, 0, 0, 0, 0))

        text = "x" * 40  # 40 one-char chunks, 0.05 s apart → ~2.0 s total
        ctx = await create_test_engine(
            [MockResponse(text=text, chunk_size=1, chunk_delay=0.05)],
            tmp_path,
            stream=True,
        )

        retries: list[InvocationRetryAttempt] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)

        # 10s budget comfortably exceeds the ~2s total drain time so the
        # watchdog never fires on this healthy stream.
        ctx.engine.current.loaded.bindings._stream_attempt_timeout = 10.0

        try:
            await ctx.send_message("go")

            assert retries == [], (
                "Per-chunk watchdog must not fire on a healthy stream: "
                f"unexpected retries {[r.message for r in retries]}"
            )
            assert _finals(ctx.events) == [text]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_idle_gap_triggers_stall(self, tmp_path, monkeypatch):
        """A single chunk gap exceeding the per-chunk timeout fires _StreamStall,
        which the retry loop surfaces as a RetryAttempt with "Stream stalled"."""
        # Zero backoff + cap retries so the test converges quickly.  After
        # retries exhaust, the executor falls back to non-streaming, which
        # the mock client satisfies instantly (no chunk delay on that path).
        monkeypatch.setattr(TurnBindings, "_BACKOFF_SCHEDULE", (0, 0, 0, 0, 0))
        # Single chunk, but chunk_delay is large relative to the per-chunk
        # budget we'll install below.  With 2 one-char chunks 1.0 s apart,
        # the second __anext__() will never arrive within 0.1 s → stall.
        ctx = await create_test_engine(
            [MockResponse(text="ab", chunk_size=1, chunk_delay=1.0)],
            tmp_path,
            stream=True,
            max_transient_retries=1,
        )

        retries: list[InvocationRetryAttempt] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)

        ctx.engine.current.loaded.bindings._stream_attempt_timeout = 0.1

        try:
            await ctx.send_message("go")

            assert len(retries) >= 1, "Expected at least one RetryAttempt from stream stall"
            assert any("stall" in r.message.lower() for r in retries), (
                f"Expected a stall retry, got messages: {[r.message for r in retries]}"
            )
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_local_zero_budget_stall_makes_one_blocking_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(TurnBindings, "_BACKOFF_SCHEDULE", (0,))
        ctx = await create_test_engine(
            [MockResponse(text="ab", chunk_size=1, chunk_delay=0.2)],
            tmp_path,
            stream=True,
            max_transient_retries=0,
        )
        ctx.engine.current.loaded.bindings._stream_attempt_timeout = 0.01

        try:
            await ctx.send_message("go")

            assert ctx.mock_client.call_count == 2
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_single_fast_chunk_completes(self, tmp_path, monkeypatch):
        """Sanity: a single-chunk response well inside the budget streams
        normally with no stall and no retry."""
        monkeypatch.setattr(TurnBindings, "_BACKOFF_SCHEDULE", (0, 0, 0, 0, 0))

        ctx = await create_test_engine(
            [MockResponse(text="hello")],  # chunk_size=0 → one chunk
            tmp_path,
            stream=True,
        )

        retries: list[InvocationRetryAttempt] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)

        ctx.engine.current.loaded.bindings._stream_attempt_timeout = 1.0

        try:
            await ctx.send_message("go")

            assert retries == []
            assert _finals(ctx.events) == ["hello"]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_long_tool_call_does_not_trigger_stream_stall(self, tmp_path, monkeypatch):
        """Regression for the reported bug: a tool that runs longer than the
        per-chunk stall timeout must NOT be mis-detected as a stream stall.

        When the LLM streams a ``function_call`` chunk, FunctionInvocationLayer
        awaits tool execution inside the NEXT ``__anext__()``
        call — so the per-chunk watchdog window would otherwise tick
        through the entire tool duration and falsely stall the stream
        (especially for long-running sub-agents in production).  The
        watchdog must be suspended across tool execution.
        """
        import asyncio as _asyncio

        monkeypatch.setattr(TurnBindings, "_BACKOFF_SCHEDULE", (0, 0, 0, 0, 0))

        tool_entered = _asyncio.Event()
        release_tool = _asyncio.Event()

        # Hold the tool open until the test has observed a complete watchdog
        # window.  This proves the watchdog is suspended *during* tool
        # execution without making the pre-tool chunk race a 100ms CI clock.
        async def _slow_echo(message: Annotated[str, "msg"]) -> str:
            tool_entered.set()
            await release_tool.wait()
            return f"slow: {message}"

        slow_tool = FunctionTool(
            func=_slow_echo,
            name="slow_echo",
            description="A deliberately slow echo",
        )

        ctx = await create_test_engine(
            [
                MockResponse(tool_calls=[("slow_echo", "call_1", {"message": "hi"})]),
                MockResponse(text="done"),
            ],
            tmp_path,
            stream=True,
            tools=[slow_tool],
        )

        retries: list[InvocationRetryAttempt] = []

        async def _on_retry(ev: InvocationRetryAttempt) -> None:
            retries.append(ev)

        await ctx.bus.subscribe(InvocationRetryAttempt, _on_retry)

        # The watchdog value must be in place BEFORE the tool-carrying
        # ``__anext__`` is issued: ``asyncio.wait_for`` captures its timeout
        # when the await starts, so lowering it after the tool is observed
        # in flight can never affect the already-armed wait.  Pick a value
        # with generous headroom over mock-chunk scheduling jitter, then
        # hold the tool open past a full watchdog window.
        watchdog_timeout = 1.5
        ctx.engine.current.loaded.bindings._stream_attempt_timeout = watchdog_timeout
        run_task = _asyncio.create_task(ctx.send_message("go"))

        try:
            await _asyncio.wait_for(tool_entered.wait(), timeout=10)
            await _asyncio.sleep(watchdog_timeout + 0.5)

            assert not any("stall" in r.message.lower() for r in retries), (
                f"Long tool call must not trigger stream stall; got retries: {[r.message for r in retries]}"
            )
            assert not run_task.done(), "the tool must remain in flight for the watchdog observation window"
            release_tool.set()
            await run_task
            assert not any("stall" in r.message.lower() for r in retries), (
                f"Long tool call must not trigger stream stall; got retries: {[r.message for r in retries]}"
            )
            assert _finals(ctx.events) == ["done"]
        finally:
            release_tool.set()
            try:
                await _asyncio.wait_for(run_task, timeout=10)
            finally:
                await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_interrupt_cancels_streaming_run_with_inflight_tool(self, tmp_path, monkeypatch):
        """Regression: ``executor.interrupt()`` must actually cancel the
        streaming agent.run, even when a tool is in flight.

        Pre-fix the streaming path had no ``_current_agent_task`` handle,
        so ``interrupt()`` set the flag but never cancelled the live
        run — letting long-running tools (and their approval/tool-call
        events) keep firing after the user hit Stop.  Post-fix the
        streaming path wraps iteration in a task that ``interrupt()``
        can cancel directly.
        """
        import asyncio as _asyncio

        monkeypatch.setattr(TurnBindings, "_BACKOFF_SCHEDULE", (0, 0, 0, 0, 0))

        tool_entered = _asyncio.Event()
        tool_cancelled = _asyncio.Event()
        hold_tool_open = _asyncio.Event()

        async def _very_slow(message: Annotated[str, "msg"]) -> str:
            tool_entered.set()
            try:
                # Keep the tool in flight until the streaming task is
                # cancelled.  An event avoids coupling this regression to an
                # arbitrary sleep duration.
                await hold_tool_open.wait()
            except _asyncio.CancelledError:
                tool_cancelled.set()
                raise
            return "too late"

        slow_tool = FunctionTool(
            func=_very_slow,
            name="very_slow",
            description="Blocks until cancelled",
        )

        ctx = await create_test_engine(
            [
                MockResponse(tool_calls=[("very_slow", "call_1", {"message": "hi"})]),
                MockResponse(text="should not arrive"),
            ],
            tmp_path,
            stream=True,
            tools=[slow_tool],
        )

        # Large stall budget so we're testing interrupt, not stall.
        ctx.engine.current.loaded.bindings._stream_attempt_timeout = 60.0

        from chrys.foundation.events.types import UserMessage

        try:
            # Fire the user message in the background; the engine starts
            # streaming, enters the tool, and blocks.
            await ctx.bus.publish(UserMessage(text="go"))
            # Wait until the tool is actually running.
            await _asyncio.wait_for(tool_entered.wait(), timeout=5.0)

            # Interrupt — the live tool must observe cancellation directly.
            # Do not time the subsequent wait_for_idle(): it also includes
            # turn finalization and filesystem persistence, whose latency on
            # a contended Windows runner says nothing about cancellation.
            await ctx.send_interrupt()
            await _asyncio.wait_for(tool_cancelled.wait(), timeout=10.0)

            # Give the engine a chance to unwind.
            from tests.support.pipeline_helpers import wait_for_idle

            await wait_for_idle(ctx)

            assert ctx.engine.current.loaded.bindings.state.was_interrupted is True
        finally:
            hold_tool_open.set()
            await ctx.cleanup()


class TestWholeRunRetryGate:
    """``AttemptRunner._may_retry_attempt`` blocks whole-run retries once any
    answered tool work exists: locally answered results via the loop
    recorder's commit count, provider-hosted calls via the evidence the
    validation middleware attaches to the raised error or via its
    run-scoped observation probe (the rejected/aborted exchange never
    reaches the recorder).  A live continuation token exempts the gate:
    the retry then resumes the background response instead of
    re-creating the request."""

    @staticmethod
    def _turn_bindings(
        committed_count: int | None,
        hosted_observed: tuple[str, ...] = (),
    ) -> TurnBindings:
        executor = _make_executor()
        executor.trajectory_context = None
        executor._loop_recorder = None if committed_count is None else SimpleNamespace(committed_count=committed_count)
        executor._hosted_commits_probe = lambda: hosted_observed
        return executor

    def test_transient_error_with_no_commits_retries(self):
        assert self._turn_bindings(0)._attempts._may_retry_attempt(ConnectionError("transient")) is True
        assert self._turn_bindings(None)._attempts._may_retry_attempt(ConnectionError("transient")) is True

    def test_local_commits_block_retry(self):
        assert self._turn_bindings(1)._attempts._may_retry_attempt(ConnectionError("transient")) is False

    def test_hosted_commits_block_retry_despite_empty_recorder(self):
        err = RetryableResponseValidationError(
            "empty or whitespace-only text response",
            exemption=ValidationRetryExemption(attempt=1, max_attempts=3, delay_seconds=1),
            hosted_commits=("create_issue",),
        )
        assert self._turn_bindings(0)._attempts._may_retry_attempt(err) is False
        assert self._turn_bindings(None)._attempts._may_retry_attempt(err) is False

    def test_hosted_free_validation_failure_still_retries(self):
        err = RetryableResponseValidationError(
            "empty contents",
            exemption=ValidationRetryExemption(attempt=1, max_attempts=3, delay_seconds=1),
        )
        assert self._turn_bindings(0)._attempts._may_retry_attempt(err) is True

    def test_observed_hosted_work_blocks_stall_and_transport_retries(self):
        # A stall/transport drop after hosted updates streamed carries no
        # evidence on the exception — the middleware's probe must gate it.
        executor = self._turn_bindings(0, hosted_observed=("create_issue",))
        assert executor._attempts._may_retry_attempt(StreamStall("stalled")) is False
        assert executor._attempts._may_retry_attempt(ConnectionError("dropped")) is False

    def test_live_continuation_token_exempts_hosted_gate(self):
        # With a live background token the retry polls the SAME response —
        # hosted work is not re-created, so the retry must stay allowed.
        executor = self._turn_bindings(0, hosted_observed=("create_issue",))
        run_kwargs = {"options": {"continuation_token": {"response_id": "pending"}}}
        assert executor._attempts._may_retry_attempt(StreamStall("stalled"), run_kwargs) is True
        assert executor._attempts._may_retry_attempt(StreamStall("stalled"), {"options": {}}) is False

    async def test_hosted_work_on_a_provider_failure_blocks_whole_run_retry(self):
        # The adapter raised before the response landed: the hosted work it
        # carried reaches the gate only through the validation middleware.
        middleware = ResponseValidationMiddleware(backoff_schedule=[0.0])
        error = ProviderResponseError(
            "stream_truncated",
            "the response failed",
            retryable=True,
            observed_contents=(
                Content.from_mcp_server_tool_call("mc1", "create_issue", server_name="github"),
                Content.from_mcp_server_tool_result("mc1", output=[Content.from_text("created #42")]),
            ),
        )
        context = ChatContext(client=None, messages=[Message("user", ["go"])], options=None, stream=False)

        async def _call_next() -> None:
            raise error

        with pytest.raises(ProviderResponseError):
            await middleware.process(context, _call_next)
        executor = self._turn_bindings(0)
        executor._hosted_commits_probe = middleware.hosted_commits_observed

        live = {"options": {"continuation_token": {"response_id": "pending"}}}
        assert executor._attempts._may_retry_attempt(error) is False
        assert executor._attempts._may_retry_attempt(error, live) is True
