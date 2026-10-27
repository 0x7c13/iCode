# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A session host's first input can ask the user, cold or started: the question never waits on the turn stream.

The host publishes a new input and does not read its turn stream until that
publish returns, while the input's ``user_prompt_submit`` hooks wait for the
answer inside it. Confirmations travel by session-level subscription (as the
ACP server's frontend takes them), so the question still reaches the frontend
and the turn goes on once answered (plan §4, second review P1).
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap

import pytest

from chrys.foundation.events.types import Error, UserInject
from chrys.foundation.platform import get_platform
from chrys.orchestration.session_host import Cancelled, ChrysSessionHost, EndTurn, Errored, HeadlessRunError, Refused
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.state.store import JsonFileStateStore
from tests.orchestration._session_host_helpers import (
    _make_agent_registry,
    _make_profile,
    _make_settings_and_models,
    _make_tool,
    _patch_runtime,
)
from tests.support.hook_confirmations import FakeConfirmationFrontend
from tests.support.scripted_clients import ErrorMockChatClient
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for


def _write_ask_hook() -> None:
    script = textwrap.dedent(
        """
        import json, os
        with open(os.environ["CHRYS_HOOK_RESULT"], "w") as f:
            json.dump({"action": "ask", "reason": "check"}, f)
        """
    )
    hooks_dir = get_platform().config_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "version": 1,
        "hooks": [
            {
                "id": "secrets",
                "event": "user_prompt_submit",
                "run": {"type": "command", "argv": [sys.executable, "-c", script]},
                "execution": {"mode": "blocking", "timeout_seconds": 30},
            }
        ],
    }
    (hooks_dir / "hooks.json").write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize("started", [False, True])
@pytest.mark.asyncio
async def test_the_first_input_s_question_reaches_the_frontend_and_the_turn_goes_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path, started: bool
) -> None:
    _write_ask_hook()
    client = MockChatClient(responses=[MockResponse(text="done")])
    _patch_runtime(monkeypatch, [client], [])
    settings, model_registry = _make_settings_and_models()
    host = ChrysSessionHost(
        profile_name="Headless",
        settings=settings,
        agent_registry=_make_agent_registry(_make_profile(approval_default="skip")),
        model_registry=model_registry,
        state_store=JsonFileStateStore(tmp_path / "sessions"),
        allow_user_interaction=True,
    )
    frontend = FakeConfirmationFrontend(host.event_bus, session_id=None)
    await frontend.attach()
    try:
        if started:
            await host.start()
        run = asyncio.create_task(host.run_until_final("hello", timeout=45))
        await wait_for(lambda: frontend.claimed or run.done(), timeout=ENGINE_TURN_TIMEOUT, description="the question")
        assert not run.done()
        assert frontend.requests[0].input_text == "hello"
        await frontend.answer(frontend.claimed[0], "approved")
        result = await asyncio.wait_for(run, timeout=ENGINE_TURN_TIMEOUT)
    finally:
        await host.shutdown()

    assert result.text == "done"


@pytest.mark.asyncio
async def test_cancelling_while_the_input_s_question_waits_ends_the_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # The input has no run task yet while its hooks wait: the cancel (ACP session/cancel
    # takes this path too) must end the input itself, and a late answer must not revive it.
    _write_ask_hook()
    client = MockChatClient(responses=[MockResponse(text="done")])
    _patch_runtime(monkeypatch, [client], [])
    settings, model_registry = _make_settings_and_models()
    host = ChrysSessionHost(
        profile_name="Headless",
        settings=settings,
        agent_registry=_make_agent_registry(_make_profile(approval_default="skip")),
        model_registry=model_registry,
        state_store=JsonFileStateStore(tmp_path / "sessions"),
        allow_user_interaction=True,
    )
    frontend = FakeConfirmationFrontend(host.event_bus, session_id=None)
    await frontend.attach()
    run = asyncio.create_task(host.run_until_final("hello", timeout=45))
    try:
        await wait_for(lambda: frontend.claimed or run.done(), timeout=ENGINE_TURN_TIMEOUT, description="the question")
        assert not run.done()
        await host.cancel_current_turn()
        await wait_for(lambda: frontend.withdrawn, timeout=ENGINE_TURN_TIMEOUT, description="the question withdrawn")
        await frontend.answer(frontend.claimed[0], "approved")
        await asyncio.wait({run}, timeout=ENGINE_TURN_TIMEOUT)
        assert run.done()
    finally:
        if not run.done():
            run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        await host.shutdown()

    (withdrawn,) = frontend.withdrawn
    assert withdrawn.outcome == "invalidated"
    assert client.call_count == 0
    # A cancel ends the request as cancelled, never as the input's refusal.
    assert isinstance(host.last_turn_outcome, Cancelled)


# ── a refused input is the input's own outcome, not an engine failure ───


def _write_hooks(*entries: dict[str, object]) -> None:
    hooks_dir = get_platform().config_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    (hooks_dir / "hooks.json").write_text(json.dumps({"version": 1, "hooks": list(entries)}), encoding="utf-8")


def _gate_on(word: str, action: str) -> dict[str, object]:
    """A user_prompt_submit hook that returns *action* for a message containing *word*, else allows it."""
    script = textwrap.dedent(
        f"""
        import json, os
        with open(os.environ["CHRYS_HOOK_PAYLOAD_FILE"], encoding="utf-8") as f:
            text = json.load(f)["text"]
        decision = {{"action": {action!r}, "reason": "not this one"}} if {word!r} in text else {{}}
        with open(os.environ["CHRYS_HOOK_RESULT"], "w") as f:
            json.dump(decision, f)
        """
    )
    return {
        "id": "gate",
        "event": "user_prompt_submit",
        "run": {"type": "command", "argv": [sys.executable, "-c", script]},
        "execution": {"mode": "blocking", "timeout_seconds": 30},
    }


def _host(tmp_path, *, interactive: bool) -> ChrysSessionHost:
    settings, model_registry = _make_settings_and_models()
    profile = _make_profile(approval_default="skip")
    return ChrysSessionHost(
        profile_name="Headless",
        settings=settings,
        agent_registry=_make_agent_registry(profile),
        model_registry=model_registry,
        state_store=JsonFileStateStore(tmp_path / "sessions"),
        allow_user_interaction=interactive,
    )


@pytest.mark.parametrize(
    ("action", "interactive", "answer"),
    [("block", True, None), ("ask", True, "declined"), ("ask", False, None)],
    ids=["blocked", "ask-declined", "ask-nobody"],
)
@pytest.mark.asyncio
async def test_a_refused_prompt_is_refused_not_errored_and_the_session_goes_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path, action: str, interactive: bool, answer: str | None
) -> None:
    _write_hooks(_gate_on("secret", action))
    client = MockChatClient(responses=[MockResponse(text="done")])
    _patch_runtime(monkeypatch, [client], [])
    host = _host(tmp_path, interactive=interactive)
    if answer is not None:
        frontend = FakeConfirmationFrontend(host.event_bus, session_id=None, answer_inline=answer)
        await frontend.attach()
    try:
        with pytest.raises(HeadlessRunError):
            await host.run_until_final("the secret plan", timeout=45)
        refused = host.last_turn_outcome
        assert isinstance(refused, Refused)
        assert (refused.error.code, refused.error.refused_input) == ("hook_blocked", "prompt")
        assert client.call_count == 0
        # Nothing about the session failed: the next message runs.
        result = await host.run_until_final("hello", timeout=45)
    finally:
        await host.shutdown()

    assert result.text == "done"
    assert isinstance(host.last_turn_outcome, EndTurn)


@pytest.mark.parametrize("model_fails", [False, True], ids=["turn-completes", "model-fails"])
@pytest.mark.asyncio
async def test_a_refused_injection_leaves_the_turn_its_own_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path, model_fails: bool
) -> None:
    started, release = tmp_path / "started", tmp_path / "release"
    hold = textwrap.dedent(
        f"""
        import os, time
        open({str(started)!r}, "w").close()
        while not os.path.exists({str(release)!r}):
            time.sleep(0.01)
        """
    )
    _write_hooks(
        _gate_on("inject", "block"),
        {
            "id": "hold",
            "event": "before_tool_call",
            "match": {"tool_name": "guarded_echo"},
            "run": {"type": "command", "argv": [sys.executable, "-c", hold]},
            "execution": {"mode": "blocking", "timeout_seconds": 30},
        },
    )
    after_tool: MockResponse | BaseException = (
        RuntimeError("model went away") if model_fails else MockResponse(text="done")
    )
    client = ErrorMockChatClient([MockResponse(tool_calls=[("guarded_echo", "c1", {"message": "hi"})]), after_tool])
    _patch_runtime(monkeypatch, [client], [_make_tool()])
    host = _host(tmp_path, interactive=True)
    errors: list[Error] = []

    async def record(event: Error) -> None:
        errors.append(event)

    await host.event_bus.subscribe(Error, record)
    run = asyncio.create_task(host.run_until_final("work", timeout=45))
    try:
        await wait_for(lambda: started.exists() or run.done(), timeout=ENGINE_TURN_TIMEOUT, description="the tool held")
        assert not run.done()
        await host.event_bus.publish(UserInject(text="inject this", injection_id="inj-1", session_id=host.session_id))
        assert [e.refused_input for e in errors] == ["injection"]
        release.touch()
        outcome = await asyncio.gather(run, return_exceptions=True)
    finally:
        release.touch()
        if not run.done():
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)
        await host.shutdown()

    turn = host.last_turn_outcome
    if model_fails:
        # The turn's own failure stays the outcome; the refused injection neither hides nor replaces it.
        assert isinstance(outcome[0], HeadlessRunError)
        assert isinstance(turn, Errored) and turn.error.refused_input is None
    else:
        assert outcome[0].text == "done"
        assert isinstance(turn, EndTurn)
