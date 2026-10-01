# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for SessionTitleUpdater — scheduling, cancellation, and persistence gates."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.app.features.session_title.updater import (
    _SYSTEM_PROMPT,
    SessionTitleUpdater,
    _build_title_request,
    _conversation_text,
    _resolve_title_profile,
    _sanitize_title,
)
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import SessionTitleUpdated
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.kernel import Content, Message
from chrys.kernel.compaction import EXCLUDED_KEY
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.state.store import JsonFileStateStore
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for, wait_until

pytestmark = pytest.mark.asyncio


class _FakeHistory:
    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages


class _FakeEngine:
    def __init__(self, session_id: str = "sess1", messages: list[Message] | None = None) -> None:
        self.history = _FakeHistory(
            messages
            if messages is not None
            else [Message("user", ["fix the login bug"]), Message("assistant", ["done, patched auth.py"])]
        )
        self._session_id = session_id
        # Isolate from the developer's shell: an exported
        # CHRYS_SESSION_TITLE_AUTO=0 or CHRYS_MODEL_PROFILE_SESSION_TITLE
        # must not fail the suite.
        self.settings = replace(
            Settings.from_env(),
            session_title_auto=True,
            session_title_model_profile="",
        )
        self.active_model_profile = ModelProfile(id="m1", name="M1", model_id="test-model")
        self.model_registry = None
        self.session_dir = None
        self.is_turn_active = False
        self.session_generation = 0
        self.bound_context: object | None = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def trajectory_context(self):
        """The session's recording scope — rebound, like the engine's, on every switch."""
        return self.bound_context

    @property
    def history_messages(self) -> list[Message]:
        return self.history.messages


async def _saved_store(tmp_path: Path, session_id: str = "sess1") -> JsonFileStateStore:
    store = JsonFileStateStore(tmp_path)
    state = {"messages": [Message("user", ["fix the login bug"])], "compressed_msgs": []}
    await store.save_session(session_id, state, agent_profile="code")
    return store


def _updater(store, engine, monkeypatch, *, title: str = "Login bug fix") -> tuple[SessionTitleUpdater, EventBus]:
    bus = EventBus()
    updater = SessionTitleUpdater(bus, store, engine_getter=lambda: engine)

    async def fake_generate_title(self, engine, session_id, transcript, profile, *, existing_title, trajectory):
        return title

    monkeypatch.setattr(SessionTitleUpdater, "_generate_title", fake_generate_title)
    return updater, bus


# ─── happy path ─────────────────────────────────────────────────────


async def test_turn_finished_persists_generated_title_and_publishes(tmp_path: Path, monkeypatch) -> None:
    store = await _saved_store(tmp_path)
    engine = _FakeEngine()
    updater, bus = _updater(store, engine, monkeypatch)

    events: list[SessionTitleUpdated] = []

    async def on_event(event: SessionTitleUpdated) -> None:
        events.append(event)

    await bus.subscribe(SessionTitleUpdated, on_event)

    updater.on_turn_finished()
    assert updater._task is not None
    await updater._task

    meta = await store.load_session_meta("sess1")
    assert meta is not None
    assert meta.generated_title == "Login bug fix"
    assert len(events) == 1
    assert events[0].title == "Login bug fix"
    assert events[0].custom is False
    assert events[0].display_title == "Login bug fix"


async def test_existing_generated_title_is_passed_and_identical_result_is_noop(tmp_path: Path, monkeypatch) -> None:
    store = await _saved_store(tmp_path)
    await store.update_session_titles("sess1", generated_title="Login bug fix")
    engine = _FakeEngine()
    seen_existing_titles: list[str] = []

    async def keep_title(self, engine, session_id, transcript, profile, *, existing_title, trajectory):
        seen_existing_titles.append(existing_title)
        return existing_title

    monkeypatch.setattr(SessionTitleUpdater, "_generate_title", keep_title)
    update_calls: list[str] = []
    original_update = store.update_session_titles

    async def spy_update(session_id, *, custom_title=None, generated_title=None):
        update_calls.append(generated_title)
        return await original_update(session_id, custom_title=custom_title, generated_title=generated_title)

    monkeypatch.setattr(store, "update_session_titles", spy_update)
    bus = EventBus()
    events: list[SessionTitleUpdated] = []

    async def on_event(event: SessionTitleUpdated) -> None:
        events.append(event)

    await bus.subscribe(SessionTitleUpdated, on_event)
    updater = SessionTitleUpdater(bus, store, engine_getter=lambda: engine)

    updater.on_turn_finished()
    assert updater._task is not None
    await updater._task

    assert seen_existing_titles == ["Login bug fix"]
    assert update_calls == []
    assert events == []


async def test_existing_generated_title_is_replaced_when_model_returns_a_new_title(tmp_path: Path, monkeypatch) -> None:
    store = await _saved_store(tmp_path)
    await store.update_session_titles("sess1", generated_title="Fix login bug")
    engine = _FakeEngine()
    updater, bus = _updater(store, engine, monkeypatch, title="Plan database migration")
    events: list[SessionTitleUpdated] = []

    async def on_event(event: SessionTitleUpdated) -> None:
        events.append(event)

    await bus.subscribe(SessionTitleUpdated, on_event)

    updater.on_turn_finished()
    assert updater._task is not None
    await updater._task

    meta = await store.load_session_meta("sess1")
    assert meta is not None
    assert meta.generated_title == "Plan database migration"
    assert [event.title for event in events] == ["Plan database migration"]


# ─── cancellation ───────────────────────────────────────────────────


async def test_new_turn_cancels_inflight_generation(tmp_path: Path, monkeypatch) -> None:
    store = await _saved_store(tmp_path)
    engine = _FakeEngine()
    bus = EventBus()
    updater = SessionTitleUpdater(bus, store, engine_getter=lambda: engine)

    started = asyncio.Event()

    async def slow_generate_title(self, engine, session_id, transcript, profile, *, existing_title, trajectory):
        started.set()
        await asyncio.sleep(30)
        return "too late"

    monkeypatch.setattr(SessionTitleUpdater, "_generate_title", slow_generate_title)

    updater.on_turn_finished()
    task = updater._task
    assert task is not None
    await asyncio.wait_for(started.wait(), timeout=5)

    updater.on_turn_started()
    with pytest.raises(asyncio.CancelledError):
        await task

    meta = await store.load_session_meta("sess1")
    assert meta is not None
    assert meta.generated_title == ""


async def test_next_turn_finish_supersedes_previous_task(tmp_path: Path, monkeypatch) -> None:
    store = await _saved_store(tmp_path)
    engine = _FakeEngine()
    bus = EventBus()
    updater = SessionTitleUpdater(bus, store, engine_getter=lambda: engine)

    started = asyncio.Event()

    async def slow_generate_title(self, engine, session_id, transcript, profile, *, existing_title, trajectory):
        started.set()
        await asyncio.sleep(30)
        return "first"

    monkeypatch.setattr(SessionTitleUpdater, "_generate_title", slow_generate_title)
    updater.on_turn_finished()
    first_task = updater._task
    assert first_task is not None
    await asyncio.wait_for(started.wait(), timeout=5)

    async def fast_generate_title(self, engine, session_id, transcript, profile, *, existing_title, trajectory):
        return "second"

    monkeypatch.setattr(SessionTitleUpdater, "_generate_title", fast_generate_title)
    updater.on_turn_finished()
    second_task = updater._task
    assert second_task is not None and second_task is not first_task
    assert first_task.cancelled() or first_task.cancelling()

    await second_task
    meta = await store.load_session_meta("sess1")
    assert meta is not None
    assert meta.generated_title == "second"


async def test_shutdown_cancels_inflight_task(tmp_path: Path, monkeypatch) -> None:
    store = await _saved_store(tmp_path)
    engine = _FakeEngine()
    bus = EventBus()
    updater = SessionTitleUpdater(bus, store, engine_getter=lambda: engine)

    started = asyncio.Event()

    async def slow_generate_title(self, engine, session_id, transcript, profile, *, existing_title, trajectory):
        started.set()
        await asyncio.sleep(30)
        return "too late"

    monkeypatch.setattr(SessionTitleUpdater, "_generate_title", slow_generate_title)
    updater.on_turn_finished()
    await asyncio.wait_for(started.wait(), timeout=5)

    await updater.shutdown()
    assert updater._task is None


async def test_turn_finished_after_shutdown_schedules_nothing(tmp_path: Path, monkeypatch) -> None:
    """Engine shutdown drains a still-finalizing run whose success callback
    fires after the app already drained the updater; it must not schedule an
    orphan task that nobody cancels or awaits."""
    store = await _saved_store(tmp_path)
    engine = _FakeEngine()
    updater, _bus = _updater(store, engine, monkeypatch)

    await updater.shutdown()
    updater.on_turn_finished()

    assert updater._task is None


# ─── gates ──────────────────────────────────────────────────────────


async def test_custom_title_disables_generation(tmp_path: Path, monkeypatch) -> None:
    store = await _saved_store(tmp_path)
    await store.update_session_titles("sess1", custom_title="Pinned")
    engine = _FakeEngine()

    generate_calls = []

    async def spy_generate_title(self, engine, session_id, transcript, profile, *, existing_title, trajectory):
        generate_calls.append(session_id)
        return "generated"

    monkeypatch.setattr(SessionTitleUpdater, "_generate_title", spy_generate_title)
    updater = SessionTitleUpdater(EventBus(), store, engine_getter=lambda: engine)

    updater.on_turn_finished()
    assert updater._task is not None
    await updater._task

    assert generate_calls == []
    meta = await store.load_session_meta("sess1")
    assert meta is not None
    assert meta.generated_title == ""


async def test_raised_metadata_load_failure_skips_generation(tmp_path: Path, monkeypatch) -> None:
    store = await _saved_store(tmp_path)
    await store.update_session_titles("sess1", generated_title="Stable existing title")
    engine = _FakeEngine()
    generate_calls: list[str] = []

    async def fail_load_session_meta(session_id, *, prefer_recovery=False):
        raise OSError("temporary read failure")

    async def spy_generate_title(self, engine, session_id, transcript, profile, *, existing_title, trajectory):
        generate_calls.append(session_id)
        return "unstable replacement"

    monkeypatch.setattr(store, "load_session_meta", fail_load_session_meta)
    monkeypatch.setattr(SessionTitleUpdater, "_generate_title", spy_generate_title)
    updater = SessionTitleUpdater(EventBus(), store, engine_getter=lambda: engine)

    updater.on_turn_finished()
    assert updater._task is not None
    await updater._task

    assert generate_calls == []


async def test_missing_session_meta_skips_generation(tmp_path: Path, monkeypatch, caplog) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _FakeEngine()
    generate_calls: list[str] = []

    async def spy_generate_title(self, engine, session_id, transcript, profile, *, existing_title, trajectory):
        generate_calls.append(session_id)
        return "title with nowhere to persist"

    monkeypatch.setattr(SessionTitleUpdater, "_generate_title", spy_generate_title)
    updater = SessionTitleUpdater(EventBus(), store, engine_getter=lambda: engine)

    with caplog.at_level("DEBUG", logger="chrys.app.features.session_title.updater"):
        updater.on_turn_finished()
        assert updater._task is not None
        await updater._task

    assert generate_calls == []
    assert "Session metadata unavailable for title refresh: sess1" in caplog.text


async def test_busy_engine_at_persist_time_drops_result(tmp_path: Path, monkeypatch) -> None:
    store = await _saved_store(tmp_path)
    engine = _FakeEngine()
    updater, _bus = _updater(store, engine, monkeypatch)

    engine.is_turn_active = True
    updater.on_turn_finished()
    assert updater._task is not None
    await updater._task

    meta = await store.load_session_meta("sess1")
    assert meta is not None
    assert meta.generated_title == ""


async def test_auto_disabled_by_settings(tmp_path: Path, monkeypatch) -> None:
    store = await _saved_store(tmp_path)
    engine = _FakeEngine()
    engine.settings = replace(engine.settings, session_title_auto=False)
    updater, _bus = _updater(store, engine, monkeypatch)

    updater.on_turn_finished()
    assert updater._task is None


async def test_no_session_id_skips(tmp_path: Path, monkeypatch) -> None:
    store = await _saved_store(tmp_path)
    engine = _FakeEngine(session_id="")
    updater, _bus = _updater(store, engine, monkeypatch)

    updater.on_turn_finished()
    assert updater._task is None


# ─── title request ──────────────────────────────────────────────────


async def test_title_prompt_defaults_to_retaining_a_valid_existing_title() -> None:
    assert "default to keeping it" in _SYSTEM_PROMPT
    assert "Replace it only when the primary topic or goal has clearly changed" in _SYSTEM_PROMPT
    assert "copy the existing title EXACTLY" in _SYSTEM_PROMPT
    assert "Correct: Fix login bug" in _SYSTEM_PROMPT
    assert "Incorrect: <existing-title>Fix login bug</existing-title>" in _SYSTEM_PROMPT

    request = _build_title_request("User: keep fixing auth.py", existing_title="Fix login bug")

    assert "<existing-title>\nFix login bug\n</existing-title>" in request
    assert "Keep it unless the retention policy requires replacing it" in request
    assert "<conversation-transcript>\nUser: keep fixing auth.py\n</conversation-transcript>" in request
    assert "Return only the raw title text" in request
    assert "Do not include <existing-title>" in request


async def test_title_request_distinguishes_initial_generation() -> None:
    request = _build_title_request("User: fix the login bug", existing_title="")

    assert "No existing generated title is available" in request
    assert "<existing-title>\n" not in request


# ─── conversation extraction ────────────────────────────────────────


async def test_conversation_text_excludes_tools_and_markers() -> None:
    tool_msg = Message("tool", [Content.from_function_result("call_1", result="raw output")])
    marker = Message("user", ["turn marker"])
    marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    assistant_with_call = Message(
        "assistant",
        [Content.from_text("let me check"), Content.from_function_call("call_1", "read_file", arguments={})],
    )
    engine = _FakeEngine(
        messages=[
            Message("system", ["system prompt"]),
            Message("user", ["fix the login bug"]),
            marker,
            assistant_with_call,
            tool_msg,
            Message("assistant", ["done, patched auth.py"]),
        ]
    )

    text = _conversation_text(engine)
    assert "User: fix the login bug" in text
    assert "Assistant: done, patched auth.py" in text
    assert "system prompt" not in text
    assert "turn marker" not in text
    assert "raw output" not in text


async def test_conversation_text_skips_compaction_excluded_messages() -> None:
    compacted = Message("user", ["OLD compacted question"])
    compacted.additional_properties[EXCLUDED_KEY] = True
    compacted_reply = Message("assistant", ["OLD compacted answer"])
    compacted_reply.additional_properties[EXCLUDED_KEY] = True
    engine = _FakeEngine(
        messages=[
            compacted,
            compacted_reply,
            Message("user", ["fix the login bug"]),
            Message("assistant", ["done, patched auth.py"]),
        ]
    )

    text = _conversation_text(engine)
    assert "OLD compacted" not in text
    assert "User: fix the login bug" in text


async def test_conversation_text_skips_continuation_nudge_keeps_injected() -> None:
    """§2.4: a synthetic ``continue`` nudge is never a "User:" line, while an
    injected mid-turn message still is — it IS user input."""
    nudge = Message("user", ["continue"])
    nudge.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
    injected = Message("user", ["also update the changelog"])
    injected.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    engine = _FakeEngine(
        messages=[
            Message("user", ["fix the login bug"]),
            nudge,
            injected,
            Message("assistant", ["done, patched auth.py"]),
        ]
    )

    text = _conversation_text(engine)
    assert "User: continue" not in text
    assert "User: also update the changelog" in text
    assert "User: fix the login bug" in text


async def test_conversation_text_nudge_alone_does_not_satisfy_user_gate() -> None:
    """§2.4: a flagged nudge alone must not authorize a title side call."""
    nudge = Message("user", ["continue"])
    nudge.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
    engine = _FakeEngine(messages=[nudge, Message("assistant", ["resumed the task"])])

    assert _conversation_text(engine) == ""


async def test_conversation_text_requires_user_message() -> None:
    engine = _FakeEngine(messages=[Message("assistant", ["hello there"])])
    assert _conversation_text(engine) == ""


async def test_conversation_text_truncates_long_transcripts() -> None:
    messages = [Message("user", ["x" * 2000])]
    messages += [Message("assistant", ["y" * 2000]) for _ in range(20)]
    engine = _FakeEngine(messages=messages)
    text = _conversation_text(engine)
    assert "\n...\n" in text
    assert len(text) < 20_000


async def test_conversation_text_token_cap_keeps_newest_whole_messages() -> None:
    """A small summarizer context window keeps only the newest whole messages."""
    messages = [Message("user", ["first question about login"])]
    for i in range(40):
        messages.append(Message("assistant", [f"answer {i} " + "y " * 100]))
        messages.append(Message("user", [f"question {i} " + "z " * 100]))
    engine = _FakeEngine(messages=messages)

    full = _conversation_text(engine)
    capped = _conversation_text(engine, max_tokens=300)

    assert len(capped) < len(full)
    assert "first question about login" not in capped
    capped_lines = capped.splitlines()
    # Whole messages only, ending with the newest one.
    assert all(line.startswith(("User: ", "Assistant: ")) for line in capped_lines)
    assert capped_lines[-1].startswith("User: question 39")


async def test_conversation_text_under_token_budget_keeps_full_transcript() -> None:
    engine = _FakeEngine()
    assert _conversation_text(engine, max_tokens=10_000) == _conversation_text(engine)


class _ClosingClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


async def test_generate_title_rebuilds_client_on_profile_config_change(tmp_path: Path, monkeypatch) -> None:
    """The route id omits connection config (API key, headers, proxy/SSL,
    timeouts), so a live profile edit must invalidate the cached client —
    including an in-place mutation of the registry's profile object."""
    from types import SimpleNamespace

    from chrys.service.llm import clients as clients_mod
    from chrys.service.llm import one_shot as one_shot_mod
    from chrys.service.profiles.models.schema import ModelProfile

    store = await _saved_store(tmp_path)
    engine = _FakeEngine()
    engine.settings = replace(engine.settings, session_title_model_profile="")
    profile = ModelProfile(id="m1", name="M1", model_id="test-model", api_key="old")
    engine.active_model_profile = profile
    engine.model_registry = None
    engine.session_dir = None

    created: list[str] = []
    clients: list[_ClosingClient] = []

    async def fake_create_client(profile, **kwargs):
        created.append(profile.api_key)
        clients.append(_ClosingClient())
        return clients[-1]

    async def fake_get_final_response(client, messages, **kwargs):
        return SimpleNamespace(text=" <existing-title>\nFix login bug\n</existing-title>\n")

    monkeypatch.setattr(clients_mod, "create_client", fake_create_client)
    monkeypatch.setattr(one_shot_mod, "get_final_response", fake_get_final_response)

    def title_profile():
        return _resolve_title_profile(engine)

    updater = SessionTitleUpdater(EventBus(), store, engine_getter=lambda: engine)
    assert (
        await updater._generate_title(
            engine, "sess1", "User: hi", title_profile(), existing_title="Login bug", trajectory=None
        )
        == "Fix login bug"
    )
    assert (
        await updater._generate_title(
            engine, "sess1", "User: hi", title_profile(), existing_title="Login bug", trajectory=None
        )
        == "Fix login bug"
    )
    assert created == ["old"]

    engine.active_model_profile = replace(profile, api_key="new")
    await updater._generate_title(
        engine, "sess1", "User: hi", title_profile(), existing_title="Login bug", trajectory=None
    )
    assert created == ["old", "new"]

    engine.active_model_profile.api_key = "newer"  # registries may edit profiles in place
    await updater._generate_title(
        engine, "sess1", "User: hi", title_profile(), existing_title="Login bug", trajectory=None
    )
    assert created == ["old", "new", "newer"]
    # A replaced client with no call in flight closes at once; the current one stays open.
    assert [client.closed for client in clients] == [True, True, False]


# ─── client lease ───────────────────────────────────────────────────


class _TitleCall:
    """One title call: answered when the test says so; once cancelled, cleans up until released."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cleaning = asyncio.Event()
        self.cleanup_release = asyncio.Event()
        self.answer = asyncio.Event()
        self.cancels = 0


class _LeaseClient:
    """A title client that records its calls and closes."""

    def __init__(self) -> None:
        self.calls: list[_TitleCall] = []
        self.closes = 0

    @property
    def closed(self) -> bool:
        return self.closes > 0

    async def get_response(self) -> str:
        call = _TitleCall()
        self.calls.append(call)
        call.entered.set()
        try:
            await call.answer.wait()
        except asyncio.CancelledError:
            call.cancels += 1
            call.cleaning.set()
            while not call.cleanup_release.is_set():
                try:
                    await call.cleanup_release.wait()
                except asyncio.CancelledError:
                    call.cancels += 1
            raise
        return "Leased title"

    async def aclose(self) -> None:
        self.closes += 1


def _lease_updater(
    store: JsonFileStateStore, engine: _FakeEngine, monkeypatch, clients: list[_LeaseClient]
) -> SessionTitleUpdater:
    from types import SimpleNamespace
    from unittest.mock import create_autospec

    from chrys.service.llm import clients as clients_mod
    from chrys.service.llm import one_shot as one_shot_mod

    async def get_final_response(
        client: _LeaseClient, messages: list[Message], *, stream: bool, options: dict, timeout: float
    ) -> SimpleNamespace:
        return SimpleNamespace(text=await client.get_response())

    monkeypatch.setattr(
        clients_mod, "create_client", create_autospec(clients_mod.create_client, side_effect=list(clients))
    )
    monkeypatch.setattr(
        one_shot_mod,
        "get_final_response",
        create_autospec(one_shot_mod.get_final_response, side_effect=get_final_response),
    )
    return SessionTitleUpdater(EventBus(), store, engine_getter=lambda: engine)


def _start_title_task(updater: SessionTitleUpdater) -> asyncio.Task[None]:
    updater.on_turn_finished()
    task = updater._task
    assert task is not None
    return task


def _change_title_profile(engine: _FakeEngine) -> None:
    engine.active_model_profile = replace(engine.active_model_profile, api_key="rotated")


async def _entered(client: _LeaseClient, index: int, task: asyncio.Task[None]) -> _TitleCall:
    await wait_for(
        lambda: len(client.calls) > index or task.done(), timeout=ENGINE_TURN_TIMEOUT, description="title call"
    )
    if task.done():
        await task
    return client.calls[index]


async def test_replacing_the_client_waits_for_the_old_call_to_drain(tmp_path: Path, monkeypatch) -> None:
    engine = _FakeEngine()
    old, new = _LeaseClient(), _LeaseClient()
    updater = _lease_updater(await _saved_store(tmp_path), engine, monkeypatch, [old, new])
    first = _start_title_task(updater)
    old_call = await _entered(old, 0, first)

    _change_title_profile(engine)
    second = _start_title_task(updater)
    new_call = await _entered(new, 0, second)
    try:
        assert old_call.cleaning.is_set()
        assert not first.done()
        assert not old.closed

        old_call.cleanup_release.set()
        await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), ENGINE_TURN_TIMEOUT)
        assert old.closes == 1
        assert not new.closed
        assert not second.done()
    finally:
        old_call.cleanup_release.set()
        new_call.cleanup_release.set()
        new_call.answer.set()
        await updater.shutdown()
    assert new.closes == 1


async def test_shutdown_drains_every_unfinished_task_before_closing(tmp_path: Path, monkeypatch) -> None:
    engine = _FakeEngine()
    old, new = _LeaseClient(), _LeaseClient()
    updater = _lease_updater(await _saved_store(tmp_path), engine, monkeypatch, [old, new])
    first = _start_title_task(updater)
    old_call = await _entered(old, 0, first)
    _change_title_profile(engine)
    second = _start_title_task(updater)
    new_call = await _entered(new, 0, second)

    shutting_down = asyncio.create_task(updater.shutdown())
    try:
        await wait_for(new_call.cleaning.is_set, timeout=ENGINE_TURN_TIMEOUT, description="second call cancelled")
        assert not await wait_until(lambda: shutting_down.done() or old.closed or new.closed, timeout=0.2)

        old_call.cleanup_release.set()
        new_call.cleanup_release.set()
        await asyncio.wait_for(shutting_down, ENGINE_TURN_TIMEOUT)
        assert first.done() and second.done()
        assert (old.closes, new.closes) == (1, 1)
    finally:
        old_call.cleanup_release.set()
        new_call.cleanup_release.set()
        await asyncio.gather(shutting_down, return_exceptions=True)


async def test_shutdown_after_cancel_only_waits_for_cleanup(tmp_path: Path, monkeypatch) -> None:
    client = _LeaseClient()
    updater = _lease_updater(await _saved_store(tmp_path), _FakeEngine(), monkeypatch, [client])
    first = _start_title_task(updater)
    first_call = await _entered(client, 0, first)
    second = _start_title_task(updater)
    second_call = await _entered(client, 1, second)
    assert first_call.cleaning.is_set()

    shutting_down = asyncio.create_task(updater.shutdown())
    try:
        await wait_for(second_call.cleaning.is_set, timeout=ENGINE_TURN_TIMEOUT, description="second call cancelled")
        assert not await wait_until(lambda: shutting_down.done() or client.closed, timeout=0.2)
        assert first_call.cancels == 1

        first_call.cleanup_release.set()
        second_call.cleanup_release.set()
        await asyncio.wait_for(shutting_down, ENGINE_TURN_TIMEOUT)
        assert (first_call.cancels, second_call.cancels) == (1, 1)
        assert client.closes == 1
    finally:
        first_call.cleanup_release.set()
        second_call.cleanup_release.set()
        await asyncio.gather(shutting_down, return_exceptions=True)


async def test_cancelled_shutdown_waiter_does_not_interrupt_drain(tmp_path: Path, monkeypatch) -> None:
    client = _LeaseClient()
    updater = _lease_updater(await _saved_store(tmp_path), _FakeEngine(), monkeypatch, [client])
    task = _start_title_task(updater)
    call = await _entered(client, 0, task)

    waiter = asyncio.create_task(updater.shutdown())
    try:
        await wait_for(call.cleaning.is_set, timeout=ENGINE_TURN_TIMEOUT, description="title call cancelled")
        shutdown_task = updater._shutdown_task
        assert shutdown_task is not None
        waiter.cancel()
        # The waiter stays until the shared shutdown finishes; only then does its cancellation surface.
        assert not await wait_until(lambda: waiter.done() or task.done() or client.closed, timeout=0.2)
        assert call.cancels == 1

        call.cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, ENGINE_TURN_TIMEOUT)
        assert shutdown_task.done() and not shutdown_task.cancelled()
        assert task.done()
        assert client.closes == 1

        await updater.shutdown()
        assert updater._shutdown_task is shutdown_task
        assert client.closes == 1
    finally:
        call.cleanup_release.set()
        await asyncio.gather(waiter, return_exceptions=True)


async def test_same_client_reuse_across_turns_is_not_closed(tmp_path: Path, monkeypatch) -> None:
    client = _LeaseClient()
    updater = _lease_updater(await _saved_store(tmp_path), _FakeEngine(), monkeypatch, [client])
    for index in range(2):
        task = _start_title_task(updater)
        call = await _entered(client, index, task)
        call.answer.set()
        await asyncio.wait_for(task, ENGINE_TURN_TIMEOUT)
        assert not client.closed

    await updater.shutdown()
    assert client.closes == 1


# ─── sanitizer ──────────────────────────────────────────────────────


async def test_sanitize_title_strips_quotes_labels_and_punctuation() -> None:
    assert _sanitize_title('"Fix login bug"') == "Fix login bug"
    assert _sanitize_title("Title: Fix login bug.") == "Fix login bug"
    assert _sanitize_title("标题：修复登录问题。") == "修复登录问题"  # noqa: RUF001
    assert _sanitize_title("multi\nline   title") == "multi line title"


async def test_sanitize_title_extracts_a_single_model_return_wrapper() -> None:
    assert _sanitize_title(" <existing-title>new title string</existing-title>\n") == "new title string"
    assert _sanitize_title("Title:\n<TITLE>\nFix login bug\n</TITLE>") == "Fix login bug"
    assert _sanitize_title("<title>Fix login bug</title>.") == "Fix login bug"
    assert _sanitize_title('"<title>Fix login bug</title>"') == "Fix login bug"
    assert _sanitize_title("Here is the title: <title>Fix login bug</title>") == "Fix login bug"
    assert _sanitize_title('"<existing-title>X</existing-title>"') == "X"


async def test_sanitize_title_rejects_malformed_or_ambiguous_prompt_markup() -> None:
    assert _sanitize_title("<existing-title>unterminated title") == ""
    assert _sanitize_title("orphan close</existing-title>") == ""
    assert _sanitize_title("<current-title>unterminated title") == ""
    assert _sanitize_title("<session-title>unterminated title") == ""
    assert _sanitize_title("<final-title>unterminated title") == ""
    assert _sanitize_title("<conversation-transcript>not a title</conversation-transcript>") == ""
    assert _sanitize_title("<existing-title>first</existing-title><existing-title>second</existing-title>") == ""


async def test_sanitize_title_preserves_code_and_markup_fragments() -> None:
    assert _sanitize_title("Implement List<T> cache") == "Implement List<T> cache"
    assert _sanitize_title("Refactor vector<int> usage") == "Refactor vector<int> usage"
    assert _sanitize_title("Fix <Component> render bug") == "Fix <Component> render bug"
    assert _sanitize_title("Cast to <int> type") == "Cast to <int> type"
    assert _sanitize_title("Improve <div> layout") == "Improve <div> layout"
    assert _sanitize_title("Optimize <title> metadata") == "Optimize <title> metadata"
    assert _sanitize_title("Fix <title>foo</title> metadata") == "Fix <title>foo</title> metadata"
    assert _sanitize_title("Escape <title></title> in HTML") == "Escape <title></title> in HTML"
    assert _sanitize_title("<title>first</title><title>second</title>") == "<title>first</title><title>second</title>"


async def test_sanitize_title_caps_length() -> None:
    assert len(_sanitize_title("z" * 500)) <= 80


async def test_history_swap_during_generation_drops_result(tmp_path: Path, monkeypatch) -> None:
    """A restore/rollback that replaces the engine history invalidates the in-flight summary."""
    store = await _saved_store(tmp_path)
    engine = _FakeEngine()
    bus = EventBus()
    updater = SessionTitleUpdater(bus, store, engine_getter=lambda: engine)

    async def swapping_generate_title(self, engine, session_id, transcript, profile, *, existing_title, trajectory):
        engine.history = _FakeHistory([Message("user", ["entirely different transcript"])])
        return "stale summary"

    monkeypatch.setattr(SessionTitleUpdater, "_generate_title", swapping_generate_title)
    updater.on_turn_finished()
    assert updater._task is not None
    await updater._task

    meta = await store.load_session_meta("sess1")
    assert meta is not None
    assert meta.generated_title == ""


async def test_history_growth_during_generation_drops_result(tmp_path: Path, monkeypatch) -> None:
    """New messages landing mid-generation invalidate the in-flight summary."""
    store = await _saved_store(tmp_path)
    engine = _FakeEngine()
    bus = EventBus()
    updater = SessionTitleUpdater(bus, store, engine_getter=lambda: engine)

    async def growing_generate_title(self, engine, session_id, transcript, profile, *, existing_title, trajectory):
        engine.history.messages.append(Message("user", ["another message"]))
        return "stale summary"

    monkeypatch.setattr(SessionTitleUpdater, "_generate_title", growing_generate_title)
    updater.on_turn_finished()
    assert updater._task is not None
    await updater._task

    meta = await store.load_session_meta("sess1")
    assert meta is not None
    assert meta.generated_title == ""


async def test_session_generation_bump_during_generation_drops_result(tmp_path: Path, monkeypatch) -> None:
    """A same-session restore (same id, same message count) still invalidates via session_generation."""
    store = await _saved_store(tmp_path)
    engine = _FakeEngine()
    bus = EventBus()
    updater = SessionTitleUpdater(bus, store, engine_getter=lambda: engine)

    async def restoring_generate_title(self, engine, session_id, transcript, profile, *, existing_title, trajectory):
        # Simulate lifecycle restore: same engine, same session id, the
        # history manager is rebound in place (same object, same count),
        # only the session generation moves.
        engine.session_generation += 1
        return "stale summary"

    monkeypatch.setattr(SessionTitleUpdater, "_generate_title", restoring_generate_title)
    updater.on_turn_finished()
    assert updater._task is not None
    await updater._task

    meta = await store.load_session_meta("sess1")
    assert meta is not None
    assert meta.generated_title == ""


async def test_the_title_call_is_traced_against_the_session_it_summarizes(tmp_path: Path, monkeypatch) -> None:
    """A switch during the store read must not move the request into the new session's trajectory."""
    store = await _saved_store(tmp_path)
    engine = _FakeEngine()
    engine.bound_context = "sess1-recorder"
    bus = EventBus()
    updater = SessionTitleUpdater(bus, store, engine_getter=lambda: engine)
    traced: list[object] = []
    real_load = store.load_session_meta

    async def switching_load(session_id: str):
        meta = await real_load(session_id)
        # The window the title task spends awaiting the store is long enough
        # to switch sessions in; the engine now records somewhere else.
        engine.bound_context = "sess2-recorder"
        return meta

    async def recording_generate_title(self, engine, session_id, transcript, profile, *, existing_title, trajectory):
        traced.append(trajectory)
        return "Login bug fix"

    monkeypatch.setattr(store, "load_session_meta", switching_load)
    monkeypatch.setattr(SessionTitleUpdater, "_generate_title", recording_generate_title)

    updater.on_turn_finished()
    assert updater._task is not None
    await updater._task

    assert traced == ["sess1-recorder"]


async def test_sanitize_title_strips_think_blocks() -> None:
    assert _sanitize_title("<think>我要输出一个标题</think> 标题正文") == "标题正文"
    assert _sanitize_title("<think>\nmultiline\nreasoning\n</think>Fix login bug") == "Fix login bug"
    assert _sanitize_title("<THINKING>loud</THINKING>Quiet title") == "Quiet title"
    # Unmatched closing tag: everything before it was reasoning.
    assert _sanitize_title("leaked reasoning</think> Real title") == "Real title"
    # Unmatched opening tag: everything after it is reasoning.
    assert _sanitize_title("Real title <think>trailing reasoning") == "Real title"
    # Pure reasoning yields an empty title (caller skips persisting it).
    assert _sanitize_title("<think>only reasoning</think>") == ""
