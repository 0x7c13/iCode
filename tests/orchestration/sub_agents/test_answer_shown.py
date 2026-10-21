# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Whether a sub-agent's answer text was shown decides whether its call may be sent again.

The TUI chat's sub-agent cards show each answer segment whole once it completes, so there a completed segment
stops every retry. ``icode run`` and ACP show none of a sub-agent's text, so there the text it wrote never stops
one. Either way a pass the retry gave up never leaves its text in the child's history.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from chrys.foundation.events.types import (
    InvocationAborted,
    InvocationAbortRequested,
    InvocationContentFinished,
    InvocationPaused,
    InvocationRetryAttempt,
    InvocationToolCallStart,
    UserInterrupt,
    UserMessage,
)
from chrys.foundation.models.session_surface import SessionSurface
from chrys.kernel import ChatResponseUpdate, Content
from chrys.service.llm.mock import MockResponse
from tests.orchestration.sub_agents._pipeline import (
    _child_answer_segments,
    _explore_result,
    _make_ctx,
    _sub_tool_call,
    _SubAgentPipelineCtx,
    use_fast_sub_agent_policy,
)
from tests.support.scripted_clients import FrameworkBoom, hosted_image_result
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from chrys.kernel import FinishReason, FinishReasonLiteral


@pytest.fixture
def fast_sub_agent_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fast_sub_agent_policy(monkeypatch)


def _cut_stream(
    ctx: _SubAgentPipelineCtx,
    monkeypatch: pytest.MonkeyPatch,
    *,
    after_first: list[Content],
    retry_entered: asyncio.Event | None = None,
) -> None:
    """Drop the sub-agent's first stream after its first two text chunks, streaming *after_first* between them.

    With *retry_entered*, a later stream sets it and then hangs without writing anything.
    """
    original = ctx.sub_client._stream_updates
    calls = 0

    async def cut(
        resp: MockResponse, model_id: str, finish_reason: FinishReasonLiteral | FinishReason
    ) -> AsyncIterator[ChatResponseUpdate]:
        nonlocal calls
        calls += 1
        if calls > 1 and retry_entered is not None:
            retry_entered.set()
            await asyncio.Event().wait()
        texts = 0
        async for update in original(resp, model_id, finish_reason):
            yield update
            if calls == 1 and update.text:
                texts += 1
                if texts == 2:
                    raise ConnectionError("dropped")
                if after_first:
                    yield ChatResponseUpdate(role="assistant", contents=list(after_first))

    monkeypatch.setattr(ctx.sub_client, "_stream_updates", cut)


async def _child_segment_ends(ctx: _SubAgentPipelineCtx) -> list[str]:
    ends: list[str] = []

    async def record(event: InvocationContentFinished) -> None:
        if event.origin.kind == "sub_agent" and event.channel == "answer":
            ends.append(event.status)

    await ctx.bus.subscribe(InvocationContentFinished, record)
    return ends


def _audit_record(ctx: _SubAgentPipelineCtx) -> str:
    session_dir = ctx.engine.session.session_dir
    assert session_dir is not None
    [audit] = (session_dir / "sub_agents" / "sessions").glob("*.json")
    return audit.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_sub_agent_answer_cut_off_before_its_segment_completes_is_retried_in_place(
    tmp_path: Path, fast_sub_agent_controller: None, agent_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sub-agent views show an answer segment once it completes: one cut off before then was never shown."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[_sub_tool_call("explore"), MockResponse(text="Parent done.")],
        sub_outcomes=[
            MockResponse(text="Cut off here.", chunk_size=4, chunk_delay=0),
            MockResponse(text="Looking it up."),
        ],
        sub_stream=True,
        agent_engine=agent_engine,
    )
    _cut_stream(ctx, monkeypatch, after_first=[])
    ends = await _child_segment_ends(ctx)
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        await ctx.wait_for_idle()

        assert _explore_result(ctx).result == "Looking it up."
        assert [e.origin.kind for e in ctx.events if isinstance(e, InvocationRetryAttempt)] == ["sub_agent"]
        assert not [e for e in ctx.events if isinstance(e, InvocationPaused)]
        assert [text for _index, text in _child_answer_segments(ctx)] == ["Cut off ", "Looking it up."]
        # The views drop the failed segment: only the retried answer shows.
        assert ends == ["failed", "completed"]
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", [None, SessionSurface.TUI], ids=["no_surface", "tui"])
async def test_a_paused_sub_agent_keeps_only_the_answer_segments_that_completed(
    tmp_path: Path,
    fast_sub_agent_controller: None,
    agent_engine,
    monkeypatch: pytest.MonkeyPatch,
    surface: SessionSurface | None,
) -> None:
    """A completed segment was shown, so the call is never sent again; Abort keeps what the card showed."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[_sub_tool_call("explore"), MockResponse(text="Acknowledged failure.")],
        sub_outcomes=[MockResponse(text="Shown part.Cut off", chunk_size=11, chunk_delay=0)],
        sub_stream=True,
        surface=surface,
        agent_engine=agent_engine,
    )
    # Reasoning after the first chunk completes its segment.
    _cut_stream(ctx, monkeypatch, after_first=[Content.from_text_reasoning(text="hmm")])
    ends = await _child_segment_ends(ctx)
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        (paused,) = await ctx.wait_for_event(InvocationPaused)
        assert not [e for e in ctx.events if isinstance(e, InvocationRetryAttempt)]
        assert [text for _index, text in _child_answer_segments(ctx)] == ["Shown part.", "Cut off"]
        assert ends == ["completed", "failed"]

        await ctx.bus.publish(InvocationAbortRequested(invocation_id=paused.origin.invocation_id))
        await ctx.wait_for_event(InvocationAborted)
        await ctx.wait_for_idle()

        record = _audit_record(ctx)
        assert "Shown part." in record
        assert "Cut off" not in record
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_a_hosted_card_a_retry_releases_never_shows_the_answer_cut_off_before_it(
    tmp_path: Path, fast_sub_agent_controller: None, agent_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hosted card held behind a local call shows when the retry gives the attempt up, after its cut-off answer ended."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[_sub_tool_call("explore"), MockResponse(text="Parent done.")],
        sub_outcomes=[MockResponse(text="unused"), MockResponse(text="Looking it up.")],
        sub_stream=True,
        sub_builtins=["sleep"],
        agent_engine=agent_engine,
    )
    original = ctx.sub_client._stream_updates
    calls = 0

    async def cut(
        resp: MockResponse, model_id: str, finish_reason: FinishReasonLiteral | FinishReason
    ) -> AsyncIterator[ChatResponseUpdate]:
        nonlocal calls
        calls += 1
        if calls > 1:
            async for update in original(resp, model_id, finish_reason):
                yield update
            return
        for content in (
            Content.from_function_call("sleep-1", "sleep", arguments={"seconds": 0}),
            hosted_image_result(),
            Content.from_text("Partial ans"),
        ):
            yield ChatResponseUpdate(role="assistant", contents=[content], model=model_id)
        raise ConnectionError("dropped")

    monkeypatch.setattr(ctx.sub_client, "_stream_updates", cut)
    ends = await _child_segment_ends(ctx)
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        await ctx.wait_for_idle()

        assert _explore_result(ctx).result == "Looking it up."
        assert [e.origin.kind for e in ctx.events if isinstance(e, InvocationRetryAttempt)] == ["sub_agent"]
        assert any(
            isinstance(e, InvocationToolCallStart) and e.origin.kind == "sub_agent" and e.provider_hosted
            for e in ctx.events
        )
        assert [text for _index, text in _child_answer_segments(ctx)] == ["Partial ans", "Looking it up."]
        # The views drop the cut-off answer: only the retried one shows.
        assert ends == ["failed", "completed"]
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("surface", "allow_user_interaction"),
    [(SessionSurface.CLI, False), (SessionSurface.ACP, True)],
    ids=["cli", "acp"],
)
async def test_text_no_frontend_shows_never_stops_a_sub_agent_retry(
    tmp_path: Path,
    fast_sub_agent_controller: None,
    agent_engine,
    monkeypatch: pytest.MonkeyPatch,
    surface: SessionSurface,
    allow_user_interaction: bool,
) -> None:
    """``icode run`` and ACP show none of a sub-agent's text: a completed segment is no reason to give the call up."""
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[_sub_tool_call("explore"), MockResponse(text="Parent done.")],
        sub_outcomes=[
            MockResponse(text="Shown part.Cut off", chunk_size=11, chunk_delay=0),
            MockResponse(text="Looking it up."),
        ],
        sub_stream=True,
        surface=surface,
        allow_user_interaction=allow_user_interaction,
        agent_engine=agent_engine,
    )
    # Reasoning after the first chunk completes its segment.
    _cut_stream(ctx, monkeypatch, after_first=[Content.from_text_reasoning(text="hmm")])
    ends = await _child_segment_ends(ctx)
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        await ctx.wait_for_idle()

        assert _explore_result(ctx).result == "Looking it up."
        assert [e.origin.kind for e in ctx.events if isinstance(e, InvocationRetryAttempt)] == ["sub_agent"]
        assert not [e for e in ctx.events if isinstance(e, InvocationPaused)]
        assert ends == ["completed", "failed", "completed"]
        record = _audit_record(ctx)
        assert "Looking it up." in record
        assert "Shown part." not in record
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("service_side", [False, True], ids=["local", "service_side"])
@pytest.mark.parametrize("ending", ["failed", "cancelled"])
async def test_a_retried_sub_agent_that_ends_keeps_no_answer_from_the_attempt_it_gave_up(
    tmp_path: Path,
    fast_sub_agent_controller: None,
    agent_engine,
    monkeypatch: pytest.MonkeyPatch,
    service_side: bool,
    ending: str,
) -> None:
    """A failed or cancelled call keeps the answer its last attempt wrote, never one a retry replaced."""
    retry_entered = asyncio.Event() if ending == "cancelled" else None
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[_sub_tool_call("explore"), MockResponse(text="Parent done.")],
        sub_outcomes=[
            MockResponse(text="Shown part.Cut off", chunk_size=11, chunk_delay=0),
            FrameworkBoom("retry failed") if retry_entered is None else MockResponse(text="unused"),
        ],
        sub_stream=True,
        sub_responses_store=service_side,
        surface=SessionSurface.CLI,
        allow_user_interaction=False,
        agent_engine=agent_engine,
    )
    _cut_stream(ctx, monkeypatch, after_first=[Content.from_text_reasoning(text="hmm")], retry_entered=retry_entered)
    try:
        await ctx.bus.publish(UserMessage(text="go"))
        if retry_entered is not None:
            await wait_for(retry_entered.is_set, description="the retry's stream to open")
            await ctx.bus.publish(UserInterrupt())
        await ctx.wait_for_idle()

        assert [e.origin.kind for e in ctx.events if isinstance(e, InvocationRetryAttempt)] == ["sub_agent"]
        if retry_entered is None:
            assert _explore_result(ctx).result.startswith("Error: sub-agent")
        assert "Shown part." not in _audit_record(ctx)
    finally:
        await ctx.cleanup()
