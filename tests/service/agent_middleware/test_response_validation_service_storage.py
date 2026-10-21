# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Service-storage rejections and the hosted work the response-validation middleware observes.

A rejection is terminal in either storage mode. Hosted tool calls a response
shows are recorded as they land, accepted or rejected, so the retry owners'
commit-point gates can consult them whatever ends the attempt.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import pytest

from chrys.foundation.hosted_tools import HostedRetrySafety, HostedToolPhase
from chrys.foundation.trajectory.context import ExchangeTrace
from chrys.foundation.trajectory.event_types import EventType, ValidationReason
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
    TerminalResponseValidationError,
)
from tests.service.agent_middleware._response_validation_fakes import (
    _assistant,
    _bad_empty,
    _FakeCallNext,
    _final_response,
    _make_context,
    _service_context,
)
from tests.service.trajectory._fakes import FakeSink, make_context

_MODES = pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])


@_MODES
async def test_a_service_storage_rejection_is_terminal_without_internal_replay(stream: bool) -> None:
    fake = _FakeCallNext([_bad_empty(), _assistant([Content.from_text("must stay unused")])], stream=stream)
    context = _service_context(stream)
    fake.bind(context)

    with pytest.raises(TerminalResponseValidationError, match="empty contents"):
        await ResponseValidationMiddleware().process(context, fake)
        await _final_response(context, stream=stream)

    assert fake.call_count == 1


@_MODES
async def test_the_rejection_records_one_give_up_verdict(stream: bool) -> None:
    sink = FakeSink()
    exchange = ExchangeTrace(make_context(sink).with_cycle(new_analytics_id()).with_exchange(new_analytics_id()))
    fake = _FakeCallNext([_bad_empty()], stream=stream)
    context = _service_context(stream, exchange)
    fake.bind(context)

    with pytest.raises(TerminalResponseValidationError):
        await ResponseValidationMiddleware().process(context, fake)
        await _final_response(context, stream=stream)

    [finished] = sink.of_type(EventType.MODEL_VALIDATION_FINISHED)
    assert finished.payload == {
        "outcome": "rejected",
        "exchange_operation_id": exchange.operation_id,
        "attempt_index": 0,
        "reason_code": ValidationReason.EMPTY_CONTENTS,
    }


def _unnamed_call() -> Content:
    """A call no tool can run: it rejects the response whatever hosted work came before it."""
    return Content.from_function_call("call_1", "", arguments="{}")


_HOSTED_WORK: list[object] = [
    pytest.param(
        lambda: [
            Content.from_mcp_server_tool_call("mc1", "create_issue", server_name="github"),
            Content.from_mcp_server_tool_result("mc1", output=[Content.from_text("created #42")]),
        ],
        ("create_issue",),
        id="hosted_mcp",
    ),
    pytest.param(
        lambda: [Content.from_shell_tool_call(call_id="sh1", commands=["curl -X POST https://x.test"])],
        ("shell",),
        id="hosted_shell",
    ),
    # Web search and code interpreter cannot reach outside the provider
    # sandbox: running them again is wasteful but safe.
    pytest.param(
        lambda: [
            Content.from_search_tool_call(call_id="ws1", tool_name="web_search", arguments={}),
            Content.from_search_tool_result(call_id="ws1", tool_name="web_search", result={}),
            Content.from_code_interpreter_tool_call(call_id="ci1", inputs=[Content.from_text("1+1")]),
            Content.from_code_interpreter_tool_result(call_id="ci1", outputs=[]),
        ],
        (),
        id="sandboxed",
    ),
    pytest.param(
        lambda: [
            Content.from_hosted_tool_call(
                "write_1",
                tool_name="write_workspace",
                status="running",
                provider_phase=HostedToolPhase.START,
                provider_status="running",
                retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
            )
        ],
        ("write_workspace",),
        id="side_effectful_start",
    ),
    pytest.param(
        lambda: [
            Content.from_hosted_tool_call(
                "read_1",
                tool_name="read_workspace",
                status="running",
                provider_phase=HostedToolPhase.START,
                provider_status="running",
                retry_safety=HostedRetrySafety.READ_ONLY,
            )
        ],
        (),
        id="read_only_start",
    ),
    pytest.param(list, (), id="no_hosted_work"),
]


class TestHostedWorkOfARejectedResponse:
    """The rejected response never reaches the loop recorder, so these probes are what the retry owners read."""

    @pytest.mark.parametrize(("hosted", "commits"), _HOSTED_WORK)
    @_MODES
    async def test_hosted_work_counts_as_executed_although_the_response_was_rejected(
        self, stream: bool, hosted: Callable[[], list[Content]], commits: tuple[str, ...]
    ) -> None:
        fake = _FakeCallNext([_assistant([*hosted(), _unnamed_call()])], stream=stream)
        context = _service_context(stream)
        fake.bind(context)
        middleware = ResponseValidationMiddleware()

        with pytest.raises(TerminalResponseValidationError, match="without a function name"):
            await middleware.process(context, fake)
            await _final_response(context, stream=stream)

        assert middleware.hosted_commits_observed() == commits
        assert middleware.hosted_commits_in_flight() == commits


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

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, fake)

        assert mw.hosted_commits_observed() == ("create_issue",)
        assert mw.hosted_commits_in_flight() == ("create_issue",)

    async def test_streaming_updates_register_before_validation_concludes(self) -> None:
        # Evidence must exist the moment the update lands — a stall after the
        # hosted result but before stream completion consults these probes.
        seen_during_stream: list[tuple[str, ...]] = []

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield ChatResponseUpdate(
                contents=[Content.from_mcp_server_tool_call("mc1", "create_issue")],
                role="assistant",
            )
            yield ChatResponseUpdate(
                contents=[Content.from_mcp_server_tool_result("mc1", output=[Content.from_text("done")])],
                role="assistant",
            )
            seen_during_stream.append(mw.hosted_commits_in_flight())
            yield ChatResponseUpdate(contents=[Content.from_text("answer")], role="assistant")

        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        mw = ResponseValidationMiddleware()
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

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, fake)
        assert mw.hosted_commits_in_flight() == ("create_issue",)

        await mw.process(ctx, fake)
        # The second wire call carried no hosted work: an in-place replay of
        # THAT request is safe, but a whole-run retry (which would restore
        # pre-run history and re-create the hosted exchange) is not.
        assert mw.hosted_commits_in_flight() == ()
        assert mw.hosted_commits_observed() == ("create_issue",)

    async def test_a_poll_keeps_the_hosted_work_of_the_response_it_resumes(self) -> None:
        running = ChatResponse(
            messages=[
                Message(
                    role="assistant",
                    contents=[Content.from_mcp_server_tool_call("mc1", "create_issue", server_name="github")],
                )
            ],
            continuation_token={"response_id": "resp_1"},
        )
        fake = _FakeCallNext([running, _assistant([Content.from_text("issue filed")])], stream=False)
        mw = ResponseValidationMiddleware()
        fake.bind(_make_context(stream=False))
        assert fake.context is not None
        await mw.process(fake.context, fake)

        poll = ChatContext(
            client=None,  # type: ignore[arg-type]
            messages=[Message("user", ["hi"])],
            options={"continuation_token": {"response_id": "resp_1"}},
            stream=False,
        )
        fake.bind(poll)
        await mw.process(poll, fake)

        # Until the response ends, a fresh request would run its hosted work again.
        assert mw.hosted_commits_in_flight() == ("create_issue",)

    def test_reset_clears_both_scopes(self) -> None:
        mw = ResponseValidationMiddleware()
        mw._observe_hosted_contents([Content.from_mcp_server_tool_call("mc1", "create_issue")])
        assert mw.hosted_commits_observed() == ("create_issue",)
        mw.reset_hosted_commit_observations()
        assert mw.hosted_commits_observed() == ()
        assert mw.hosted_commits_in_flight() == ()

    def test_a_pass_polling_a_background_response_keeps_its_hosted_work(self) -> None:
        mw = ResponseValidationMiddleware()
        mw._observe_hosted_contents([Content.from_mcp_server_tool_call("mc1", "create_issue")])
        mw.begin_pass_hosted_baseline(resumes_background_response=True)
        assert mw.hosted_commits_observed() == ("create_issue",)
        assert mw.hosted_commits_in_flight() == ("create_issue",)
        mw.begin_pass_hosted_baseline(resumes_background_response=False)
        assert mw.hosted_commits_observed() == ()
        assert mw.hosted_commits_in_flight() == ()
