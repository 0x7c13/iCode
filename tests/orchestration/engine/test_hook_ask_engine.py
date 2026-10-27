# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Engine-driven: a hook's ``ask`` stops the main agent's and a sub-agent's tool calls and the user's input the same way.

The hooks file sits in the per-test config dir, as a user's would; the hooks
are real subprocesses. A fake frontend on the engine's bus stands in for the
TUI or the ACP client. Written from the plan (``.plan/hook-ask-confirm.md``
§5, §7): every entry checks through the one service, nothing runs that the
user did not approve, and a headless engine asks nobody.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from chrys.foundation.events.types import (
    ApprovalRequest,
    ApprovalResponse,
    Error,
    Event,
    InvocationToolCallResult,
    SessionNew,
    SessionRestore,
    UserInjectCancel,
    UserInjectResult,
    UserInterrupt,
    UserMessage,
    UserRetry,
)
from chrys.foundation.platform import get_platform
from chrys.service.llm.mock import MockResponse
from tests.orchestration.invoker._build_fixtures import build_recipe_engine
from tests.support.hook_confirmations import FakeConfirmationFrontend, confirmations_settled
from tests.support.waiting import (
    ENGINE_TEST_WAIT_TIMEOUT,
    ENGINE_TURN_TIMEOUT,
    await_run_task_chain,
    wait_for,
    with_wait_deadline,
)


def _result(result: dict[str, Any]) -> list[str]:
    script = textwrap.dedent(
        f"""
        import json, os
        with open(os.environ["CHRYS_HOOK_RESULT"], "w") as f:
            json.dump({result!r}, f)
        """
    )
    return [sys.executable, "-c", script]


def _copy_payload(target: Path) -> list[str]:
    script = textwrap.dedent(
        f"""
        import os, shutil
        shutil.copy(os.environ["CHRYS_HOOK_PAYLOAD_FILE"], {str(target)!r})
        """
    )
    return [sys.executable, "-c", script]


def _entry(hook_id: str, event: str, argv: list[str], *, tool_name: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": hook_id,
        "event": event,
        "run": {"type": "command", "argv": argv},
        "execution": {"mode": "blocking", "timeout_seconds": 30},
    }
    if tool_name is not None:
        entry["match"] = {"tool_name": tool_name}
    return entry


def _write_hooks(*entries: dict[str, Any]) -> None:
    hooks_dir = get_platform().config_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    document = {"version": 1, "hooks": list(entries)}
    (hooks_dir / "hooks.json").write_text(json.dumps(document), encoding="utf-8")


class _Events:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def record(self, event: Event) -> None:
        self.events.append(event)

    async def attach(self, engine: Any) -> None:
        for kind in (InvocationToolCallResult, Error):
            await engine.event_bus.subscribe(kind, self.record)

    def tool_results(self, tool_name: str) -> list[InvocationToolCallResult]:
        return [e for e in self.events if isinstance(e, InvocationToolCallResult) and e.tool_name == tool_name]

    def errors(self) -> list[Error]:
        return [e for e in self.events if isinstance(e, Error)]


async def _frontend(engine: Any, **kwargs: Any) -> FakeConfirmationFrontend:
    frontend = FakeConfirmationFrontend(engine.event_bus, session_id=engine.session_id, **kwargs)
    await frontend.attach()
    return frontend


def _explore_turn() -> dict[str, Any]:
    return {
        "main": [MockResponse(tool_calls=[("Explore", "call-1", {"prompt": "look"})]), MockResponse(text="done")],
        "child": [MockResponse(text="looked")],
    }


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_an_approved_ask_lets_the_main_agent_s_tool_run(tmp_path, monkeypatch, agent_engine) -> None:
    _write_hooks(
        _entry("guard", "before_tool_call", _result({"action": "ask", "reason": "delegating"}), tool_name="Explore")
    )
    engine, _main, child = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, **_explore_turn())
    frontend = await _frontend(engine, answer_inline="approved")
    events = _Events()
    await events.attach(engine)

    await engine.event_bus.publish(UserMessage(text="delegate"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    (request,) = frontend.requests
    assert (request.subject, request.tool_name, request.arguments, request.reason) == (
        "tool",
        "Explore",
        {"prompt": "look"},
        "delegating",
    )
    # The main agent's own calls name no invocation; a sub-agent's do.
    assert request.origin is None
    assert len(child.stream_flags) == 1
    (result,) = events.tool_results("Explore")
    assert "asked for confirmation" not in str(result.result)


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_declined_ask_stops_the_tool_and_tells_the_model_and_the_after_hooks(
    tmp_path, monkeypatch, agent_engine
) -> None:
    after_payload = tmp_path / "after.json"
    _write_hooks(
        _entry("guard", "before_tool_call", _result({"action": "ask", "reason": "delegating"}), tool_name="Explore"),
        _entry("audit", "after_tool_call", _copy_payload(after_payload), tool_name="Explore"),
    )
    engine, main, child = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, **_explore_turn())
    await _frontend(engine, answer_inline="declined")
    events = _Events()
    await events.attach(engine)

    await engine.event_bus.publish(UserMessage(text="delegate"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    assert child.stream_flags == []
    (result,) = events.tool_results("Explore")
    assert "hook 'guard' asked for confirmation (delegating); the user declined" in str(result.result)
    after = json.loads(after_payload.read_text(encoding="utf-8"))["result"]
    assert (after["rejection_source"], after["hook_denied"], after["hook_confirmation"]) == ("hook", True, "declined")
    # The model got the refusal and answered the turn.
    assert len(main.stream_flags) == 2


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_sub_agent_s_tool_asks_through_the_same_service(tmp_path, monkeypatch, agent_engine) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("notes", encoding="utf-8")
    _write_hooks(_entry("read-guard", "before_tool_call", _result({"action": "ask"}), tool_name="read_file"))
    engine, _main, child = await build_recipe_engine(
        agent_engine,
        monkeypatch,
        tmp_path,
        main=[MockResponse(tool_calls=[("Explore", "call-1", {"prompt": "look"})]), MockResponse(text="done")],
        child=[MockResponse(tool_calls=[("read_file", "read-1", {"path": str(target)})]), MockResponse(text="read")],
    )
    frontend = await _frontend(engine, answer_inline="approved")
    approvals: list[ApprovalRequest] = []

    async def approve(event: ApprovalRequest) -> None:
        approvals.append(event)
        await engine.event_bus.publish(ApprovalResponse(request_id=event.request_id, approved=True))

    await engine.event_bus.subscribe(ApprovalRequest, approve)

    await engine.event_bus.publish(UserMessage(text="delegate"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    (request,) = frontend.requests
    assert request.origin is not None and request.origin.kind == "sub_agent"
    assert (request.caller_name, request.tool_name) == ("Explore", "read_file")
    # Confirming the hook's question is not approving the tool: the profile's approval still asks.
    assert [a.tool_name for a in approvals] == ["read_file"]
    assert len(child.stream_flags) == 2


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_headless_engine_asks_nobody_and_refuses_the_call(tmp_path, monkeypatch, agent_engine) -> None:
    _write_hooks(_entry("guard", "before_tool_call", _result({"action": "ask"}), tool_name="Explore"))
    engine, _main, child = await build_recipe_engine(
        agent_engine, monkeypatch, tmp_path, allow_user_interaction=False, **_explore_turn()
    )
    frontend = await _frontend(engine)
    events = _Events()
    await events.attach(engine)

    await engine.event_bus.publish(UserMessage(text="delegate"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    assert frontend.requests == []
    assert child.stream_flags == []
    (result,) = events.tool_results("Explore")
    assert "no one can confirm it in this session" in str(result.result)


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_an_unclaimed_ask_is_refused_at_once(tmp_path, monkeypatch, agent_engine) -> None:
    _write_hooks(_entry("guard", "before_tool_call", _result({"action": "ask"}), tool_name="Explore"))
    engine, _main, child = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, **_explore_turn())
    frontend = await _frontend(engine, claim=False)
    events = _Events()
    await events.attach(engine)

    await engine.event_bus.publish(UserMessage(text="delegate"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    assert len(frontend.requests) == 1
    assert child.stream_flags == []
    assert "no one can confirm it in this session" in str(events.tool_results("Explore")[0].result)


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_declined_ask_on_new_input_keeps_it_from_the_model(tmp_path, monkeypatch, agent_engine) -> None:
    _write_hooks(_entry("secrets", "user_prompt_submit", _result({"action": "ask", "reason": "a key?"})))
    engine, main, _child = await build_recipe_engine(
        agent_engine, monkeypatch, tmp_path, main=[MockResponse(text="hi")], child=[]
    )
    frontend = await _frontend(engine, answer_inline="declined")
    events = _Events()
    await events.attach(engine)

    await engine.event_bus.publish(UserMessage(text="deploy with sk-123"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    (request,) = frontend.requests
    assert (request.subject, request.input_text) == ("input", "deploy with sk-123")
    assert main.stream_flags == []
    assert [(e.code, e.refused_input) for e in events.errors()] == [("hook_blocked", "prompt")]


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_an_approved_ask_on_new_input_starts_the_turn(tmp_path, monkeypatch, agent_engine) -> None:
    _write_hooks(_entry("secrets", "user_prompt_submit", _result({"action": "ask"})))
    engine, main, _child = await build_recipe_engine(
        agent_engine, monkeypatch, tmp_path, main=[MockResponse(text="hi")], child=[]
    )
    frontend = await _frontend(engine)

    message = UserMessage(text="hello")
    task = asyncio.ensure_future(engine.event_bus.publish(message))
    await wait_for(lambda: frontend.claimed, timeout=ENGINE_TURN_TIMEOUT, description="the input's question shown")
    # The input's own publish is still checking it: nothing went to the model yet.
    assert not task.done()
    assert main.stream_flags == []
    await frontend.answer(frontend.claimed[0], "approved")
    await task
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    assert message.started_turn is True
    assert len(main.stream_flags) == 1


@pytest.mark.parametrize("configured", [False, True])
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_without_an_ask_tools_and_input_go_through_the_same_check(
    tmp_path, monkeypatch, agent_engine, configured: bool
) -> None:
    if configured:
        # Hooks for other events only: the gating events check an empty chain.
        _write_hooks(_entry("noop", "after_tool_call", [sys.executable, "-c", "pass"]))
    engine, main, child = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, **_explore_turn())
    events = _Events()
    await events.attach(engine)

    await engine.event_bus.publish(UserMessage(text="delegate"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    assert len(child.stream_flags) == 1
    assert len(main.stream_flags) == 2
    assert "Error:" not in str(events.tool_results("Explore")[0].result)


def _held(release: Path) -> list[str]:
    script = textwrap.dedent(
        f"""
        import os, time
        while not os.path.exists({str(release)!r}):
            time.sleep(0.01)
        """
    )
    return [sys.executable, "-c", script]


async def _inject_while_running(tmp_path: Path, monkeypatch: Any, agent_engine: Any, release: Path):
    """A turn held inside its tool's hook, its opener approved; returns the engine, frontend and inject task."""
    _write_hooks(
        _entry("hold", "before_tool_call", _held(release), tool_name="Explore"),
        _entry("secrets", "user_prompt_submit", _result({"action": "ask"})),
    )
    engine, _main, _child = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, **_explore_turn())
    frontend = await _frontend(engine)
    events = _Events()
    await events.attach(engine)
    await engine.event_bus.subscribe(UserInjectResult, events.record)
    opener = asyncio.ensure_future(engine.event_bus.publish(UserMessage(text="delegate")))
    await wait_for(lambda: frontend.claimed, timeout=ENGINE_TURN_TIMEOUT, description="the opener's question")
    await frontend.answer(frontend.claimed[0], "approved")
    await opener
    await wait_for(lambda: engine.turns.turn_state.lease.run_task is not None, description="the turn running")
    inject = asyncio.ensure_future(engine.event_bus.publish(UserMessage(text="also run tests", injection_id="inj-1")))
    await wait_for(
        lambda: len(frontend.claimed) == 2, timeout=ENGINE_TURN_TIMEOUT, description="the injection's question"
    )
    return engine, frontend, events, inject


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_withdrawing_an_injected_message_withdraws_its_question(tmp_path, monkeypatch, agent_engine) -> None:
    release = tmp_path / "release"
    try:
        engine, frontend, events, inject = await _inject_while_running(tmp_path, monkeypatch, agent_engine, release)
        assert not inject.done()
        await engine.event_bus.publish(UserInjectCancel(injection_id="inj-1"))
        await asyncio.wait_for(inject, timeout=ENGINE_TURN_TIMEOUT)

        assert [(w.request_id, w.outcome) for w in frontend.withdrawn] == [(frontend.claimed[1], "invalidated")]
        assert not any(isinstance(e, Error) and e.code == "hook_blocked" for e in events.events)
    finally:
        release.touch()
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_declined_injected_message_is_refused_as_an_injection(tmp_path, monkeypatch, agent_engine) -> None:
    release = tmp_path / "release"
    try:
        engine, frontend, events, inject = await _inject_while_running(tmp_path, monkeypatch, agent_engine, release)
        await frontend.answer(frontend.claimed[1], "declined")
        await asyncio.wait_for(inject, timeout=ENGINE_TURN_TIMEOUT)
    finally:
        release.touch()
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    # Refused as an injection: only this message, not the turn it was sent into.
    assert [(e.code, e.refused_input) for e in events.errors()] == [("hook_blocked", "injection")]


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_an_injected_message_s_question_ends_with_the_run_it_targets(tmp_path, monkeypatch, agent_engine) -> None:
    release = tmp_path / "release"
    try:
        engine, frontend, events, inject = await _inject_while_running(tmp_path, monkeypatch, agent_engine, release)
    finally:
        release.touch()
    # The model goes on while the user has not answered; its run ends, and the question with it.
    await asyncio.wait_for(inject, timeout=ENGINE_TURN_TIMEOUT)
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    assert [(w.request_id, w.outcome) for w in frontend.withdrawn] == [(frontend.claimed[1], "invalidated")]
    results = [e for e in events.events if isinstance(e, UserInjectResult)]
    assert [(r.consumed, r.text) for r in results] == [(False, "also run tests")]


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_an_approval_edit_is_asked_about_again_through_the_production_chain(
    tmp_path, monkeypatch, agent_engine
) -> None:
    first, edited = tmp_path / "first.txt", tmp_path / "edited.txt"
    first.write_text("first", encoding="utf-8")
    edited.write_text("edited", encoding="utf-8")
    _write_hooks(_entry("read-guard", "before_tool_call", _result({"action": "ask"}), tool_name="read_file"))
    engine, _main, child = await build_recipe_engine(
        agent_engine,
        monkeypatch,
        tmp_path,
        main=[MockResponse(tool_calls=[("Explore", "call-1", {"prompt": "look"})]), MockResponse(text="done")],
        child=[MockResponse(tool_calls=[("read_file", "read-1", {"path": str(first)})]), MockResponse(text="read")],
    )
    frontend = await _frontend(engine, answer_inline="approved")

    async def edit(event: ApprovalRequest) -> None:
        # The user edits the call in the approval dialog: the approval middleware checks the edit again.
        await engine.event_bus.publish(
            ApprovalResponse(request_id=event.request_id, approved=True, modified_args={"path": str(edited)})
        )

    await engine.event_bus.subscribe(ApprovalRequest, edit)
    events = _Events()
    await events.attach(engine)

    await engine.event_bus.publish(UserMessage(text="delegate"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    assert [r.arguments["path"] for r in frontend.requests] == [str(first), str(edited)]
    assert frontend.requests[0].operation_id == frontend.requests[1].operation_id
    assert [r.revision for r in frontend.requests] == [1, 2]
    assert len(child.stream_flags) == 2


def _note_asker() -> list[str]:
    script = textwrap.dedent(
        """
        import json, os
        with open(os.environ["CHRYS_HOOK_PAYLOAD_FILE"]) as f:
            text = json.load(f)["text"]
        with open(os.environ["CHRYS_HOOK_RESULT"], "w") as f:
            json.dump({"action": "ask", "reason": "a note"} if "note" in text else {}, f)
        """
    )
    return [sys.executable, "-c", script]


@pytest.mark.parametrize("answer", ["approved", "declined"])
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_retry_note_is_asked_about_before_the_retry_runs(
    tmp_path, monkeypatch, agent_engine, answer: str
) -> None:
    _write_hooks(_entry("notes", "user_prompt_submit", _note_asker()))
    engine, main, _child = await build_recipe_engine(
        agent_engine,
        monkeypatch,
        tmp_path,
        main=[RuntimeError("model went away"), MockResponse(text="recovered")],
        child=[],
    )
    frontend = await _frontend(engine)
    events = _Events()
    await events.attach(engine)
    await engine.event_bus.publish(UserMessage(text="work"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)
    calls_before = len(main.stream_flags)
    assert frontend.requests == []

    retry = asyncio.ensure_future(engine.event_bus.publish(UserRetry(text="note: try again")))
    await wait_for(lambda: frontend.claimed, timeout=ENGINE_TURN_TIMEOUT, description="the note's question")
    assert frontend.requests[0].input_text == "note: try again"
    assert len(main.stream_flags) == calls_before
    await frontend.answer(frontend.claimed[0], answer)
    await asyncio.wait_for(retry, timeout=ENGINE_TURN_TIMEOUT)
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    if answer == "approved":
        assert len(main.stream_flags) == calls_before + 1
        return
    assert len(main.stream_flags) == calls_before
    assert ("hook_blocked", "retry_note") in [(e.code, e.refused_input) for e in events.errors()]
    # The refused note took nothing with it: a retry without a note still runs, and without that note.
    await engine.event_bus.publish(UserRetry())
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)
    assert len(main.stream_flags) == calls_before + 1
    sent = " ".join(message.text for message in main.call_history[-1][0])
    assert "work" in sent and "note: try again" not in sent


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_shutdown_cancelled_during_the_release_still_closes_the_questions(
    tmp_path, monkeypatch, agent_engine
) -> None:
    engine, _main, _child = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, **_explore_turn())
    confirmations = engine._hook_confirmations
    assert confirmations is not None and confirmations.available()
    entered = asyncio.Event()
    release = asyncio.Event()
    lifecycle_shutdown = engine._lifecycle.shutdown

    async def held_shutdown() -> None:
        entered.set()
        await release.wait()
        await lifecycle_shutdown()

    monkeypatch.setattr(engine._lifecycle, "shutdown", held_shutdown)
    shutdown = asyncio.create_task(engine.shutdown())
    await wait_for(entered.is_set, description="the release is under way")
    shutdown.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    release.set()
    await wait_for(lambda: engine._release_task is not None and engine._release_task.done(), description="released")

    assert not confirmations.available()


@pytest.mark.parametrize("ending", ["interrupt", "new_session", "restore"])
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_tool_question_left_waiting_is_withdrawn_once_when_its_turn_ends(
    tmp_path, monkeypatch, agent_engine, ending: str
) -> None:
    # The user presses Esc, starts a new session or restores one while the main agent's tool call waits
    # for an answer: the question is withdrawn exactly once, the tool never runs, and nothing is left behind.
    _write_hooks(
        _entry("guard", "before_tool_call", _result({"action": "ask", "reason": "delegating"}), tool_name="Explore")
    )
    delegate = MockResponse(tool_calls=[("Explore", "call-1", {"prompt": "look"})])
    engine, _main, child = await build_recipe_engine(
        agent_engine,
        monkeypatch,
        tmp_path,
        main=[delegate, MockResponse(text="done"), delegate, MockResponse(text="unreached")],
        child=[MockResponse(text="looked"), MockResponse(text="unreached")],
    )
    # A first turn, answered, saves the session: a restore has something to load.
    frontend = await _frontend(engine, answer_inline="approved")
    await engine.event_bus.publish(UserMessage(text="delegate"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)
    assert len(child.stream_flags) == 1
    frontend.answer_inline = None
    events = _Events()
    await events.attach(engine)
    await engine.event_bus.publish(UserMessage(text="delegate again"))
    await wait_for(lambda: len(frontend.claimed) == 2, timeout=ENGINE_TURN_TIMEOUT, description="the tool's question")
    turn_state = engine.turns.turn_state
    session_id = engine.session_id

    if ending == "interrupt":
        await engine.event_bus.publish(UserInterrupt())
    elif ending == "new_session":
        await engine.event_bus.publish(SessionNew())
    else:
        await engine.event_bus.publish(SessionRestore(session_id=session_id))
    await await_run_task_chain(engine, turn_state=turn_state)
    await wait_for(lambda: frontend.withdrawn, description="the question withdrawn")

    assert [w.request_id for w in frontend.withdrawn] == frontend.claimed[1:]
    assert len(child.stream_flags) == 1
    assert events.errors() == []
    if ending == "restore":
        assert engine.session_id == session_id
    elif ending == "new_session":
        assert engine.session_id != session_id
    confirmations = engine._hook_confirmations
    assert confirmations is not None and confirmations.available()
    await wait_for(lambda: confirmations_settled(confirmations), description="the question wound down")


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_shutdown_cancelled_before_the_release_still_closes_the_questions(
    tmp_path, monkeypatch, agent_engine
) -> None:
    engine, _main, _child = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, **_explore_turn())
    confirmations = engine._hook_confirmations
    assert confirmations is not None and confirmations.available()
    entered = asyncio.Event()
    workflows_shutdown = engine._workflows.shutdown

    async def held_workflows_shutdown() -> None:
        # Held only the first time: the fixture's own shutdown at teardown goes through.
        monkeypatch.setattr(engine._workflows, "shutdown", workflows_shutdown)
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(engine._workflows, "shutdown", held_workflows_shutdown)
    shutdown = asyncio.create_task(engine.shutdown())
    await wait_for(lambda: entered.is_set() or shutdown.done(), description="the workflow drain under way")
    assert not shutdown.done()
    shutdown.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown

    # The release never started; the questions closed all the same.
    assert engine._release_task is None
    assert not confirmations.available()
