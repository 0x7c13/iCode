# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Service-storage mode of the response-validation middleware: raised errors, hosted commits, retry budget."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from chrys.foundation.hosted_tools import HostedRetrySafety, HostedToolPhase
from chrys.foundation.retry import RetryAttemptInfo
from chrys.foundation.trajectory.context import ExchangeTrace
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.kernel import (
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)
from chrys.kernel.middleware import ChatContext
from chrys.service.agent_middleware.response_validation import (
    ResponseValidationMiddleware,
    RetryableResponseValidationError,
    TerminalResponseValidationError,
    ValidationRetryExemption,
    hosted_commits_from_error,
)
from tests.service.agent_middleware._response_validation_fakes import (
    _assistant,
    _bad_empty,
    _bad_hosted_mcp,
    _bad_leaked,
    _bad_whitespace,
    _FakeCallNext,
    _make_context,
    _service_context,
)
from tests.service.trajectory._fakes import FakeSink, make_context


@pytest.mark.parametrize("stream", [False, True])
async def test_service_storage_validation_failure_surfaces_without_internal_replay(stream: bool) -> None:
    bad = _bad_empty()
    fake = _FakeCallNext([bad, _assistant([Content.from_text("must stay unused")])], stream=stream)
    ctx = _service_context(stream)
    fake.bind(ctx)
    retries: list[RetryAttemptInfo] = []

    async def _on_retry(info: RetryAttemptInfo) -> None:
        retries.append(info)

    middleware = ResponseValidationMiddleware(
        publish_retry=_on_retry,
        backoff_schedule=[0.0],
    )

    if stream:
        await middleware.process(ctx, fake)
        assert isinstance(ctx.result, ResponseStream)
        with pytest.raises(RetryableResponseValidationError, match="empty contents"):
            _ = [update async for update in ctx.result]
    else:
        with pytest.raises(RetryableResponseValidationError, match="empty contents"):
            await middleware.process(ctx, fake)

    assert fake.call_count == 1
    assert retries == []


@pytest.mark.parametrize("stream", [False, True])
async def test_forced_stateless_store_true_uses_client_side_validation_retry(stream: bool) -> None:
    fake = _FakeCallNext([_bad_empty(), _assistant([Content.from_text("recovered")])], stream=stream)
    ctx = ChatContext(
        client=SimpleNamespace(STORES_BY_DEFAULT=True, FORCES_STATELESS=True),
        messages=[Message("user", ["hi"])],
        options={"store": True, "previous_response_id": "resp_1"},
        stream=stream,
        kwargs={"client_kwargs": {"store": True}},
    )
    fake.bind(ctx)
    middleware = ResponseValidationMiddleware(backoff_schedule=[0.0])

    await middleware.process(ctx, fake)
    if stream:
        assert isinstance(ctx.result, ResponseStream)
        final = await ctx.result.get_final_response()
    else:
        assert isinstance(ctx.result, ChatResponse)
        final = ctx.result

    assert final.text == "recovered"
    assert fake.call_count == 2


class TestHostedCommitEvidence:
    """Service-storage failures carry the hosted tool calls the rejected
    response already executed server-side.  The invalid response never
    reaches the kernel loop recorder, so the error is the only carrier by
    which the whole-run retry owner can honour hosted work as commit
    points instead of re-creating the request and re-running the side
    effects."""

    async def _failing_attempt(
        self, response: ChatResponse, *, stream: bool = False
    ) -> RetryableResponseValidationError:
        fake = _FakeCallNext([response], stream=stream)
        ctx = _service_context(stream)
        fake.bind(ctx)
        middleware = ResponseValidationMiddleware(backoff_schedule=[0.0])
        if stream:
            await middleware.process(ctx, fake)
            assert isinstance(ctx.result, ResponseStream)
            with pytest.raises(RetryableResponseValidationError) as exc_info:
                _ = [update async for update in ctx.result]
        else:
            with pytest.raises(RetryableResponseValidationError) as exc_info:
                await middleware.process(ctx, fake)
        return exc_info.value

    @pytest.mark.parametrize("stream", [False, True])
    async def test_hosted_mcp_exchange_rides_on_the_raised_error(self, stream: bool) -> None:
        response = _assistant(
            [
                Content.from_mcp_server_tool_call("mc1", "create_issue", server_name="github"),
                Content.from_mcp_server_tool_result("mc1", output=[Content.from_text("created #42")]),
                Content.from_text("<tool_use>malformed</tool_use>"),
            ]
        )
        err = await self._failing_attempt(response, stream=stream)
        assert err.hosted_commits == ("create_issue",)
        assert hosted_commits_from_error(err) == ("create_issue",)

    async def test_hosted_shell_counts_as_commit(self) -> None:
        response = _assistant(
            [
                Content.from_shell_tool_call(call_id="sh1", commands=["curl -X POST https://x.test"]),
                Content.from_shell_tool_result(call_id="sh1", outputs=[]),
                Content.from_text("<tool_use>malformed</tool_use>"),
            ]
        )
        err = await self._failing_attempt(response)
        assert err.hosted_commits == ("shell",)

    async def test_hosted_free_failure_carries_no_commits(self) -> None:
        err = await self._failing_attempt(_bad_empty())
        assert err.hosted_commits == ()

    async def test_sandboxed_hosted_work_is_not_a_commit(self) -> None:
        # Web/file search and code interpreter cannot reach outside the
        # provider sandbox — re-running them is wasteful but safe, so they
        # must not turn a recoverable blank response into a hard failure.
        response = _assistant(
            [
                Content.from_search_tool_call(call_id="ws1", tool_name="web_search", arguments={}),
                Content.from_search_tool_result(call_id="ws1", tool_name="web_search", result={}),
                Content.from_code_interpreter_tool_call(call_id="ci1", inputs=[Content.from_text("1+1")]),
                Content.from_code_interpreter_tool_result(call_id="ci1", outputs=[]),
                Content.from_text("<tool_use>malformed</tool_use>"),
            ]
        )
        err = await self._failing_attempt(response)
        assert err.hosted_commits == ()

    async def test_retry_safety_side_effectful_start_is_a_commit(self) -> None:
        response = _assistant(
            [
                Content.from_hosted_tool_call(
                    "write_1",
                    tool_name="write_workspace",
                    status="running",
                    provider_phase=HostedToolPhase.START,
                    provider_status="running",
                    retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
                ),
                Content.from_text("<tool_use>malformed</tool_use>"),
            ]
        )

        err = await self._failing_attempt(response)

        assert err.hosted_commits == ("write_workspace",)

    async def test_retry_safety_read_only_start_is_exempt(self) -> None:
        response = _assistant(
            [
                Content.from_hosted_tool_call(
                    "read_1",
                    tool_name="read_workspace",
                    status="running",
                    provider_phase=HostedToolPhase.START,
                    provider_status="running",
                    retry_safety=HostedRetrySafety.READ_ONLY,
                ),
                Content.from_text("<tool_use>malformed</tool_use>"),
            ]
        )

        err = await self._failing_attempt(response)

        assert err.hosted_commits == ()

    def test_unrelated_errors_have_no_commits(self) -> None:
        assert hosted_commits_from_error(ConnectionError("transient")) == ()


class TestHostedCommitLocalReplayVeto:
    """Client-mode (in-place) validation retries re-send the request, so an
    invalid response that already executed hosted tool calls must give up
    immediately — the scrubbed response keeps the hosted transcript in
    history instead of re-rolling the side effects."""

    async def test_blocking_hosted_failure_is_not_replayed(self) -> None:
        fake = _FakeCallNext([_bad_hosted_mcp(), _assistant([Content.from_text("must stay unused")])], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 1
        assert isinstance(ctx.result, ChatResponse)
        types = [c.type for m in ctx.result.messages for c in m.contents]
        assert "mcp_server_tool_call" in types
        assert "mcp_server_tool_result" in types

    async def test_streaming_hosted_failure_is_not_replayed(self) -> None:
        fake = _FakeCallNext([_bad_hosted_mcp(), _assistant([Content.from_text("must stay unused")])], stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)
        assert isinstance(ctx.result, ResponseStream)
        updates = [update async for update in ctx.result]

        assert fake.call_count == 1
        types = [c.type for u in updates for c in u.contents or []]
        assert "mcp_server_tool_call" in types

    async def test_hosted_free_failure_still_replays_in_place(self) -> None:
        fake = _FakeCallNext([_bad_empty(), _assistant([Content.from_text("recovered")])], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages[-1].contents[0].text == "recovered"


class TestHostedCommitObservationProbes:
    """The middleware records hosted executions as they land so retry owners
    can consult them even when the failure (stall, transport drop) carries
    no evidence of its own."""

    async def test_valid_hosted_response_registers_in_both_scopes(self) -> None:
        response = _assistant(
            [
                Content.from_mcp_server_tool_call("mc1", "create_issue", server_name="github"),
                Content.from_mcp_server_tool_result("mc1", output=[Content.from_text("done")]),
                Content.from_text("issue filed"),
            ]
        )
        fake = _FakeCallNext([response], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert mw.hosted_commits_observed() == ("create_issue",)
        assert mw.hosted_commits_in_flight() == ("create_issue",)

    async def test_streaming_updates_register_before_validation_concludes(self) -> None:
        # Evidence must exist the moment the update lands — a stall after the
        # hosted result but before stream completion consults these probes.
        hosted_update = ChatResponseUpdate(
            contents=[Content.from_mcp_server_tool_result("mc1", output=[Content.from_text("done")])],
            role="assistant",
        )
        seen_during_stream: list[tuple[str, ...]] = []

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield ChatResponseUpdate(
                contents=[Content.from_mcp_server_tool_call("mc1", "create_issue")],
                role="assistant",
            )
            yield hosted_update
            seen_during_stream.append(mw.hosted_commits_in_flight())
            yield ChatResponseUpdate(contents=[Content.from_text("answer")], role="assistant")

        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, _call_next)
        assert isinstance(ctx.result, ResponseStream)
        _ = [update async for update in ctx.result]

        assert seen_during_stream == [("create_issue",)]

    async def test_new_wire_attempt_resets_in_flight_but_not_run_scope(self) -> None:
        hosted = _assistant(
            [
                Content.from_mcp_server_tool_call("mc1", "create_issue", server_name="github"),
                Content.from_text("issue filed"),
            ]
        )
        plain = _assistant([Content.from_text("plain answer")])
        fake = _FakeCallNext([hosted, plain], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)
        assert mw.hosted_commits_in_flight() == ("create_issue",)

        fake.bind(ctx)
        await mw.process(ctx, fake)
        # The second wire call carried no hosted work: an in-place replay of
        # THAT request is safe, but a whole-run retry (which would restore
        # pre-run history and re-create the hosted exchange) is not.
        assert mw.hosted_commits_in_flight() == ()
        assert mw.hosted_commits_observed() == ("create_issue",)

    def test_reset_clears_both_scopes(self) -> None:
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        mw._observe_hosted_contents([Content.from_mcp_server_tool_call("mc1", "create_issue")])
        assert mw.hosted_commits_observed() == ("create_issue",)
        mw.reset_hosted_commit_observations()
        assert mw.hosted_commits_observed() == ()
        assert mw.hosted_commits_in_flight() == ()

    def test_a_pass_polling_a_background_response_keeps_its_hosted_work(self) -> None:
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        mw._observe_hosted_contents([Content.from_mcp_server_tool_call("mc1", "create_issue")])
        mw.begin_pass_hosted_baseline(resumes_background_response=True)
        assert mw.hosted_commits_observed() == ("create_issue",)
        assert mw.hosted_commits_in_flight() == ("create_issue",)
        mw.begin_pass_hosted_baseline(resumes_background_response=False)
        assert mw.hosted_commits_observed() == ()
        assert mw.hosted_commits_in_flight() == ()


class TestServiceStorageRetryBudget:
    """Service-storage failures retry at the whole-run boundary, so the
    middleware carries its budget and last failure reason on the instance
    across outer attempts.  Each ``process()`` call below simulates one
    outer whole-run attempt against the same middleware instance."""

    async def _outer_attempt(
        self,
        mw: ResponseValidationMiddleware,
        response: ChatResponse,
        *,
        stream: bool = False,
        exchange: ExchangeTrace | None = None,
    ) -> ChatContext:
        fake = _FakeCallNext([response], stream=stream)
        ctx = _service_context(stream, exchange)
        fake.bind(ctx)
        await mw.process(ctx, fake)
        if stream:
            assert isinstance(ctx.result, ResponseStream)
            _ = [update async for update in ctx.result]
        return ctx

    async def test_identical_reason_second_outer_attempt_is_terminal(self) -> None:
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError, match="empty contents"):
            await self._outer_attempt(mw, _bad_empty())
        with pytest.raises(TerminalResponseValidationError, match="empty contents"):
            await self._outer_attempt(mw, _bad_empty())

    async def test_identical_reason_second_outer_attempt_is_terminal_stream(self) -> None:
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError, match="empty contents"):
            await self._outer_attempt(mw, _bad_empty(), stream=True)
        with pytest.raises(TerminalResponseValidationError, match="empty contents"):
            await self._outer_attempt(mw, _bad_empty(), stream=True)

    async def test_budget_exhaustion_with_distinct_reasons_is_terminal(self) -> None:
        """Distinct reasons dodge the short-circuit; the count budget still trips."""
        mw = ResponseValidationMiddleware(max_retries=2, backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_whitespace())
        with pytest.raises(TerminalResponseValidationError):
            await self._outer_attempt(mw, _bad_leaked())

    async def test_retry_errors_stamp_actual_budget_attempt_and_ceiled_delay(self) -> None:
        mw = ResponseValidationMiddleware(max_retries=2, backoff_schedule=[0.2, 1.2])

        with pytest.raises(RetryableResponseValidationError) as first:
            await self._outer_attempt(mw, _bad_empty())
        with pytest.raises(RetryableResponseValidationError) as second:
            await self._outer_attempt(mw, _bad_whitespace())

        assert first.value.exemption == ValidationRetryExemption(attempt=1, max_attempts=2, delay_seconds=1)
        assert second.value.exemption == ValidationRetryExemption(attempt=2, max_attempts=2, delay_seconds=2)

    async def test_success_resets_carried_budget(self) -> None:
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())
        ctx = await self._outer_attempt(mw, _assistant([Content.from_text("recovered")]))
        assert isinstance(ctx.result, ChatResponse)
        # The next identical failure starts a fresh cycle, not a short-circuit.
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())

    async def test_cycle_start_reset_gives_fresh_budget(self) -> None:
        """An aborted outer cycle (interrupt during backoff, unrelated
        exception) concludes through none of the internal reset points; the
        executor calls ``reset_service_retry_state`` at the next cycle start
        so the leftover count/reason cannot judge an independent run."""
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())
        mw.reset_service_retry_state()
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())

    async def test_client_mode_giveup_resets_carried_state(self) -> None:
        """A concluded client-mode cycle must not leave stale carry-over
        behind a later storage-mode flip back to service storage."""
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())
        # Client-mode cycle: identical failures short-circuit to scrub-accept.
        fake = _FakeCallNext([_bad_empty(), _bad_empty()], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)
        await mw.process(ctx, fake)
        assert isinstance(ctx.result, ChatResponse)
        # Back in service mode, the same reason starts a fresh cycle.
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())

    @pytest.mark.parametrize("stream", [False, True])
    async def test_the_recorded_verdict_states_the_service_side_give_up(self, stream: bool) -> None:
        """The give-up decision belongs in the event that reports the failure.

        Service-storage mode takes it against the carried count/reason instead
        of a loop attempt index, and the terminal raise below is what an
        analyst reading the log has to see as the cycle's give-up."""
        sink = FakeSink()
        exchange = ExchangeTrace(make_context(sink).with_cycle(new_analytics_id()).with_exchange(new_analytics_id()))
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty(), stream=stream, exchange=exchange)
        with pytest.raises(TerminalResponseValidationError):
            await self._outer_attempt(mw, _bad_empty(), stream=stream, exchange=exchange)

        retried, gave_up = sink.of_type(EventType.MODEL_VALIDATION_FINISHED)
        assert "gave_up" not in retried.payload
        assert gave_up.payload["gave_up"] is True

    async def test_terminal_giveup_resets_carried_budget(self) -> None:
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())
        with pytest.raises(TerminalResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())
        # A later user-initiated run gets a fresh budget after the terminal raise.
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())
