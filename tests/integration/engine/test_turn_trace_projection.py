# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
"""End to end: a real engine's log, projected by ``turn_trace``.

The engine, the provider SDK, Chrys's client stack and the writer are all
real; only the HTTP transport is scripted. The projection therefore reads what
production actually writes, which is the only way to tell the retry lanes
apart: an SDK-internal retry, an in-place wire retry and a whole-run retry can
all end in the same answered turn, and only their request attempts differ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Error, InvocationMessage, InvocationRetryAttempt, UserMessage, UserRetry
from chrys.foundation.models.workspace import Workspace
from chrys.foundation.util.chrys_headers import REQUEST_ATTEMPT_ID_HEADER
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.service.analytics.turn_trace import scan_turn_trace, trace_payload_with_registry
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import API_STYLE_CHAT_COMPLETIONS, API_STYLE_RESPONSES, ModelProfile
from chrys.service.state.store import JsonFileStateStore
from chrys.service.trajectory.session import trajectory_events_path
from tests.support.event_capture import capture_events
from tests.support.llm_client_engines import parent_profile
from tests.support.scripted_wire import ScriptedWire, pin_wire_inputs, route_clients_to
from tests.support.waiting import ENGINE_TURN_TIMEOUT, await_run_task_chain, wait_for
from tests.support.wire_cases import Reply
from tests.support.wire_cases._kit import cc_replies, cc_text, resp_message, resp_replies, resp_response

if TYPE_CHECKING:
    from tests.support.engines import AgentEngineFactory

_UNAVAILABLE = Reply(
    503,
    json.dumps({"type": "error", "error": {"type": "api_error", "message": "unavailable"}}).encode("utf-8"),
    (("content-type", "application/json"),),
)
_REJECTED = Reply(
    400,
    json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": "rejected"}}).encode("utf-8"),
    (("content-type", "application/json"),),
)
_MODEL_ID = "model"


@dataclass(frozen=True, slots=True)
class ProjectedTurn:
    """One finished turn, as the real log describes it."""

    payload: dict[str, Any]
    messages: list[dict[str, Any]]
    requests: list[httpx.Request]
    retries: list[InvocationRetryAttempt]

    @property
    def turn(self) -> dict[str, Any]:
        turns = self.payload["turns"]
        assert len(turns) == 1, f"expected one turn, got {[turn['turnId'] for turn in turns]}"
        return turns[0]

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.turn["modelCalls"]

    @property
    def attempt_ids(self) -> list[str]:
        return [attempt["requestAttemptId"] for call in self.calls for attempt in call["attempts"]]

    @property
    def sent_attempt_ids(self) -> list[str]:
        return [request.headers[REQUEST_ATTEMPT_ID_HEADER] for request in self.requests]


def _profile(
    *, stream: bool, http_max_retries: int, api_style: str = API_STYLE_CHAT_COMPLETIONS, chat_options: str = ""
) -> ModelProfile:
    return ModelProfile(
        id=_MODEL_ID,
        name=_MODEL_ID,
        provider="openai",
        api_style=api_style,
        model_id="model-test",
        api_key="sk-test",
        base_url="https://model.example.test",
        http_max_retries=http_max_retries,
        stream=stream,
        chat_options=chat_options,
    )


async def _run_turn(
    agent_engine: AgentEngineFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stream: bool,
    http_max_retries: int,
    replies: list[Reply],
    api_style: str = API_STYLE_CHAT_COMPLETIONS,
    chat_options: str = "",
    retry: bool = False,
) -> ProjectedTurn:
    """Run one turn whose provider traffic is *replies*, then project its closed log.

    With *retry*, the first pass is left failing and a user retry resumes the
    same turn, so the log describes one turn that ran twice.
    """
    pin_wire_inputs(monkeypatch)
    # Every lane backs off 0 s: the log, not the clock, is what this test reads.
    monkeypatch.setattr(TurnBindings, "_BACKOFF_SCHEDULE", (0,))
    wire = ScriptedWire(replies)
    route_clients_to(wire.transport, monkeypatch)
    models = ModelProfileRegistry()
    models.register(
        _profile(stream=stream, http_max_retries=http_max_retries, api_style=api_style, chat_options=chat_options)
    )
    profile = parent_profile(sub_agent=False)
    agents = AgentProfileRegistry()
    agents.register(profile)
    store = JsonFileStateStore(tmp_path)
    bus = EventBus()
    engine = agent_engine(
        bus,
        settings=Settings(model_profile=_MODEL_ID, workspace_change_notice=False),
        agent_registry=agents,
        model_registry=models,
        state_store=store,
        initial_workspace=Workspace.from_cwd(str(tmp_path)),
    )
    await engine.start(profile)
    messages = await capture_events(bus, InvocationMessage)
    retries = await capture_events(bus, InvocationRetryAttempt)
    errors = await capture_events(bus, Error)

    await bus.publish(UserMessage(text="hello"))
    await _await_turn_end(messages, errors)
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)
    if retry:
        # A failed pass leaves the turn retryable; the next pass resumes it.
        assert errors, "the first pass was expected to fail"
        errors.clear()
        await bus.publish(UserRetry())
        await _await_turn_end(messages, errors)
        await await_run_task_chain(engine, turn_state=engine.turns.turn_state)
    assert errors == []
    session_id = engine.session.session_id
    session_dir = engine.session_dir
    assert session_dir is not None
    # ``model.request.*`` rides ``emit_soon``, so the log is only complete once
    # the engine has closed it.
    await engine.shutdown()

    transcript = await store.load_session_transcript(session_id)
    assert transcript is not None
    trace = scan_turn_trace(trajectory_events_path(session_dir))
    payload = trace_payload_with_registry(trace, transcript.turns)
    assert payload is not None
    return ProjectedTurn(
        payload=payload,
        messages=transcript.messages,
        requests=list(wire.requests),
        retries=list(retries),
    )


async def _await_turn_end(messages: list[InvocationMessage], errors: list[Error]) -> None:
    """Wait for the pass in flight to reach its terminal event."""
    await wait_for(
        lambda: errors or any(message.is_final and message.origin.kind == "turn" for message in messages),
        timeout=ENGINE_TURN_TIMEOUT,
        description="turn to end",
    )


def _answer(response_id: str, *, stream: bool) -> tuple[Reply, ...]:
    return cc_replies([cc_text("answer", response_id=response_id)], stream=stream)


# ---------------------------------------------------------------- SDK retry


@pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
async def test_an_sdk_retry_is_two_request_attempts_of_one_model_call(
    agent_engine: AgentEngineFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    """The SDK re-sends inside one provider call: no Chrys retry, two attempts."""
    turn = await _run_turn(
        agent_engine,
        tmp_path,
        monkeypatch,
        stream=stream,
        http_max_retries=1,
        replies=[_UNAVAILABLE, *_answer("chatcmpl_1", stream=stream)],
    )

    assert len(turn.requests) == 2
    assert len(turn.calls) == 1
    assert turn.turn["retryCount"] == 0
    assert turn.turn["requestAttemptCount"] == 2
    # Both attempts belong to the one call: the SDK never told Chrys an exchange
    # had ended, so the retry is invisible in the log's own records.
    assert turn.attempt_ids == turn.sent_attempt_ids
    assert turn.retries == []
    assert turn.calls[0]["attempts"][0]["httpStatus"] == 503
    assert turn.calls[0]["attempts"][1]["httpStatus"] == 200


# --------------------------------------------------------------- wire retry


@pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
async def test_a_wire_retry_is_a_second_model_call_with_its_own_attempt(
    agent_engine: AgentEngineFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    """With the SDK's own retries off, Chrys replays the call and records the retry."""
    turn = await _run_turn(
        agent_engine,
        tmp_path,
        monkeypatch,
        stream=stream,
        http_max_retries=0,
        replies=[_UNAVAILABLE, *_answer("chatcmpl_1", stream=stream)],
    )

    assert len(turn.requests) == 2
    assert len(turn.calls) == 2
    assert [call["outcome"] for call in turn.calls] == ["error", "success"]
    assert turn.turn["retryCount"] == 1
    assert turn.turn["retries"][0]["retryMode"] == "wire"
    assert turn.turn["requestAttemptCount"] == 2
    assert turn.attempt_ids == turn.sent_attempt_ids
    # The frontend was told once, on the turn's own channel.
    assert [(retry.origin.kind, retry.scope) for retry in turn.retries] == [("turn", "wire")]


# ------------------------------------------------------- validation verdict


@pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
async def test_an_empty_reply_is_sent_again_once_inside_its_pass(
    agent_engine: AgentEngineFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    """A reply that showed nothing is sampled once more in place, recorded as a validation retry."""
    turn = await _run_turn(
        agent_engine,
        tmp_path,
        monkeypatch,
        stream=stream,
        http_max_retries=0,
        replies=[
            *cc_replies([cc_text("", response_id="chatcmpl_empty")], stream=stream),
            *_answer("chatcmpl_1", stream=stream),
        ],
    )

    assert len(turn.requests) == 2
    # Both requests got a reply; validation rejected the first one.
    assert [call["outcome"] for call in turn.calls] == ["success", "success"]
    assert turn.turn["retryCount"] == 1
    assert turn.turn["retries"][0]["retryMode"] == "validation"
    assert turn.turn["requestAttemptCount"] == 2
    assert [passed["endReason"] for passed in turn.turn["passes"]] == ["completed"]
    assert turn.attempt_ids == turn.sent_attempt_ids
    assert [(retry.origin.kind, retry.scope) for retry in turn.retries] == [("turn", "wire")]


@pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
async def test_a_second_empty_reply_ends_its_pass(
    agent_engine: AgentEngineFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    """The resend that is also empty fails its pass; the user's retry is the next pass."""
    empty = [
        *cc_replies([cc_text("", response_id="chatcmpl_empty")], stream=stream),
        *cc_replies([cc_text("", response_id="chatcmpl_empty_again")], stream=stream),
    ]
    turn = await _run_turn(
        agent_engine,
        tmp_path,
        monkeypatch,
        stream=stream,
        http_max_retries=0,
        replies=[*empty, *_answer("chatcmpl_1", stream=stream)],
        retry=True,
    )

    assert len(turn.requests) == 3
    assert turn.turn["retryCount"] == 1
    passes = turn.turn["passes"]
    assert [passed["endReason"] for passed in passes] == ["error", "completed"]
    assert [passed["modelCallCount"] for passed in passes] == [2, 1]
    assert turn.attempt_ids == turn.sent_attempt_ids


# ------------------------------------------------------------ whole-run lane


@pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
async def test_a_whole_run_retry_stays_on_the_turn_it_belongs_to(
    agent_engine: AgentEngineFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    """Service-side storage retries the whole run and names its own retry mode.

    Only the Responses wire keeps the provider's conversation, so ``store``
    puts this turn on the whole-run lane: one more pass of the same turn.
    """
    turn = await _run_turn(
        agent_engine,
        tmp_path,
        monkeypatch,
        stream=stream,
        http_max_retries=0,
        replies=[
            _UNAVAILABLE,
            *resp_replies(
                [resp_response(response_id="resp_1", output=[resp_message("msg_1", "answer")])], stream=stream
            ),
        ],
        api_style=API_STYLE_RESPONSES,
        chat_options='{"store": true}',
    )

    assert len(turn.requests) == 2
    assert turn.turn["retryCount"] == 1
    assert turn.turn["retries"][0]["retryMode"] == "run"
    assert turn.turn["requestAttemptCount"] == 2
    assert turn.attempt_ids == turn.sent_attempt_ids


# ---------------------------------------------------- user retry of a turn


@pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
async def test_a_user_retry_adds_a_pass_to_the_turn_that_failed(
    agent_engine: AgentEngineFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    """The failed pass and the pass that answered are one turn, so one turnId.

    A frontend that labeled its first attempt with the failed turn's id has to
    find the answer under the same id afterwards — and the second pass has to
    be visible as a pass, not as a separate turn.
    """
    turn = await _run_turn(
        agent_engine,
        tmp_path,
        monkeypatch,
        stream=stream,
        http_max_retries=0,
        replies=[
            _REJECTED,
            *_answer("chatcmpl_1", stream=stream),
        ],
        retry=True,
    )

    assert len(turn.requests) == 2
    passes = turn.turn["passes"]
    assert len(passes) == 2
    assert passes[0]["passId"] != passes[1]["passId"]
    assert [passed["endReason"] for passed in passes] == ["error", "completed"]
    # One model call per pass: the rejected request and the answer, each filed
    # under the pass that made it.
    assert [passed["modelCallCount"] for passed in passes] == [1, 1]
    assert [call["outcome"] for call in passes[0]["modelCalls"]] == ["error"]
    assert [call["outcome"] for call in passes[1]["modelCalls"]] == ["success"]
    # The turn's status is its last pass's: the retry answered the question.
    assert turn.turn["status"] == "completed"


# ------------------------------------------------------- attempt association


async def test_the_saved_reply_names_the_attempt_the_log_attributes_to_the_turn(
    agent_engine: AgentEngineFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session's reply carries the last attempt's id, and the log knows it."""
    turn = await _run_turn(
        agent_engine,
        tmp_path,
        monkeypatch,
        stream=False,
        http_max_retries=0,
        replies=[_UNAVAILABLE, *_answer("chatcmpl_1", stream=False)],
    )

    replies = [
        message
        for message in turn.messages
        if message["role"] == "assistant"
        and (message.get("additional_properties") or {}).get("_chrys_request_attempt_id")
    ]
    assert len(replies) == 1
    assert replies[0]["additional_properties"]["_chrys_request_attempt_id"] == turn.attempt_ids[-1]
    assert set(turn.attempt_ids) == set(turn.sent_attempt_ids)
