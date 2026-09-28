# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Recovery checkpoint persistence, and the primary-save / shutdown contract for the recovery sidecar."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from threading import Event as ThreadEvent
from typing import Any

import pytest

from chrys.foundation.config.settings import (
    Settings,
)
from chrys.foundation.events.bus import EventBus
from chrys.foundation.models.invocations import PassHandle
from chrys.foundation.observability.sink import OtelSessionSink, get_otel_sink, set_otel_sink
from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.foundation.recovery import RecoveryPersistOutcome
from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.kernel import Content, LoopRecorder, Message
from chrys.orchestration.engine.assembly import assemble_agent_engine
from chrys.orchestration.invoker.contracts import AbortCause, AbortResult
from chrys.orchestration.invoker.resources import PreparedAgent
from chrys.orchestration.sub_agents.tools import SubAgentTools
from chrys.service.state.store import SESSION_RECOVERY_FILE_NAME, JsonFileStateStore
from tests.orchestration.engine._recovery_helpers import (
    _HistoryStateExecutor,
    _seed_checkpoint_engine,
    _seed_recovery_sidecar,
)
from tests.support.loaded_agents import install_loaded_agent
from tests.support.waiting import wait_for, wait_until


async def test_recovery_checkpoint_write_is_backgrounded_then_flushed(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "bg")
    sidecar = store.session_dir("bg") / SESSION_RECOVERY_FILE_NAME

    await engine.writer.save_checkpoint()

    # The disk write is dispatched to a background task, not awaited inline, so
    # the LLM round trip never blocks on the sidecar's fsync/lock.
    assert engine.writer.write_task is not None

    await engine.writer.flush()
    assert sidecar.exists()


async def test_persist_recovery_now_strictly_writes_current_snapshot(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "strict-now")

    assert await engine.writer.persist_now() is True

    restored = await store.load_recovery_session("strict-now")
    assert restored is not None
    assert any(
        content.type == "function_result" and content.result == "done"
        for message in restored["messages"]
        for content in message.contents
    )


async def test_typed_recovery_barrier_persists_committed_journal_exchange(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = assemble_agent_engine(EventBus(), settings=Settings(), state_store=store)
    engine.session.session_id = "journal-barrier"
    user = Message("user", ["do work"])

    install_loaded_agent(
        engine,
        bindings=_HistoryStateExecutor(  # type: ignore[assignment]
            {"messages": [user], "compressed_msgs": [], "turn_counter": 1}
        ),
    )
    recorder = LoopRecorder(on_pre_wire_barrier=engine.writer.persist_barrier)
    install_loaded_agent(engine, loop_recorder=recorder)
    engine.turns.turn_state.set_current_input("do work", None, None)
    first_call = Content.from_function_call("committed-call-1", "write_file", arguments={})
    first_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    first_commit = recorder.stage_exchange(
        [Message("assistant", [first_call])], [first_call], result_carrier_item_id="a" * 32
    )[0]
    first_result = Content.from_function_result("committed-call-1", result="written-1")
    first_commit.commit_final(first_result)
    recorder.seal_exchange(Message("tool", [first_result]))

    second_call = Content.from_function_call("committed-call-2", "write_file", arguments={})
    second_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 1
    second_commit = recorder.stage_exchange(
        [Message("assistant", [second_call])], [second_call], result_carrier_item_id="b" * 32
    )[0]
    second_commit.commit_final(Content.from_function_result("committed-call-2", result="written-2"))

    await recorder.record_pre_call([user])

    restored = await store.load_recovery_session("journal-barrier")
    assert restored is not None
    assert [
        (content.call_id, content.result)
        for message in restored["messages"]
        for content in message.contents
        if content.type == "function_result"
    ] == [
        ("committed-call-1", "written-1"),
        ("committed-call-2", "written-2"),
    ]


async def test_persisted_pre_wire_barrier_is_the_only_pre_call_snapshot(tmp_path: Path) -> None:
    """After committed tool work the strict barrier's snapshot is the one build
    and write of the pre-call state; a best-effort checkpoint of that same
    state would only repeat it."""
    store = JsonFileStateStore(tmp_path)
    engine = assemble_agent_engine(EventBus(), settings=Settings(), state_store=store)
    engine.session.session_id = "single-barrier"
    user = Message("user", ["do work"])

    install_loaded_agent(
        engine,
        bindings=_HistoryStateExecutor(  # type: ignore[assignment]
            {"messages": [user], "compressed_msgs": [], "turn_counter": 1}
        ),
    )
    recorder = LoopRecorder(
        on_pre_wire_barrier=engine.writer.persist_barrier,
        on_result_checkpoint=engine.writer.save_checkpoint,
    )
    install_loaded_agent(engine, loop_recorder=recorder)
    engine.turns.turn_state.set_current_input("do work", None, None)
    call = Content.from_function_call("barrier-call", "write_file", arguments={})
    call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    commit = recorder.stage_exchange([Message("assistant", [call])], [call], result_carrier_item_id="a" * 32)[0]
    result = Content.from_function_result("barrier-call", result="written")
    commit.commit_final(result)
    recorder.seal_exchange(Message("tool", [result]))
    await wait_for(lambda: engine.writer.snapshot_seq == 1, description="the slot-fill checkpoint kick built")
    await engine.writer.flush()
    assert engine.writer.persisted_seq == 1

    await recorder.record_pre_call([user])

    assert engine.writer.snapshot_seq == 2
    assert engine.writer.persisted_seq == 2
    assert engine.writer.pending is None
    restored = await store.load_recovery_session("single-barrier")
    assert restored is not None
    assert [
        (content.call_id, content.result)
        for message in restored["messages"]
        for content in message.contents
        if content.type == "function_result"
    ] == [("barrier-call", "written")]


async def test_stale_background_checkpoint_cannot_downgrade_barrier_snapshot(tmp_path: Path) -> None:
    """A coalesced writer that froze state before a strict barrier must not
    overwrite the sidecar with its older snapshot after the barrier ran."""
    store = JsonFileStateStore(tmp_path)
    engine = assemble_agent_engine(EventBus(), settings=Settings(), state_store=store)
    engine.session.session_id = "stale-guard"
    user = Message("user", ["do work"])

    install_loaded_agent(
        engine,
        bindings=_HistoryStateExecutor(  # type: ignore[assignment]
            {"messages": [user], "compressed_msgs": [], "turn_counter": 1}
        ),
    )
    recorder = LoopRecorder()
    install_loaded_agent(engine, loop_recorder=recorder)
    engine.turns.turn_state.set_current_input("do work", None, None)

    first_call = Content.from_function_call("stale-call-1", "write_file", arguments={})
    first_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    first_commit = recorder.stage_exchange(
        [Message("assistant", [first_call])], [first_call], result_carrier_item_id="a" * 32
    )[0]
    first_result = Content.from_function_result("stale-call-1", result="written-1")
    first_commit.commit_final(first_result)
    recorder.seal_exchange(Message("tool", [first_result]))

    await engine.writer.save_checkpoint()
    assert engine.writer.pending is not None
    stale_seq = engine.writer.snapshot_seq

    second_call = Content.from_function_call("stale-call-2", "write_file", arguments={})
    second_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 1
    second_commit = recorder.stage_exchange(
        [Message("assistant", [second_call])], [second_call], result_carrier_item_id="b" * 32
    )[0]
    second_result = Content.from_function_result("stale-call-2", result="written-2")
    second_commit.commit_final(second_result)
    recorder.seal_exchange(Message("tool", [second_result]))

    assert await engine.writer.persist_barrier() is RecoveryPersistOutcome.PERSISTED

    await engine.writer.flush()
    assert engine.writer.persisted_seq > stale_seq

    restored = await store.load_recovery_session("stale-guard")
    assert restored is not None
    results = [
        content.result
        for message in restored["messages"]
        for content in message.contents
        if content.type == "function_result"
    ]
    assert results == ["written-1", "written-2"]


async def test_recovery_persistence_apis_keep_typed_and_bool_contracts_distinct(tmp_path: Path) -> None:
    configured = _seed_checkpoint_engine(JsonFileStateStore(tmp_path), "typed-outcomes")
    assert await configured.writer.persist_barrier() is RecoveryPersistOutcome.PERSISTED
    assert await configured.writer.persist_now() is True

    empty = assemble_agent_engine(EventBus(), settings=Settings(), state_store=JsonFileStateStore(tmp_path / "empty"))
    assert await empty.writer.persist_barrier() is RecoveryPersistOutcome.NOTHING_TO_PERSIST
    assert await empty.writer.persist_now() is False

    unconfigured = assemble_agent_engine(EventBus(), settings=Settings(), state_store=None)
    assert await unconfigured.writer.persist_barrier() is RecoveryPersistOutcome.UNCONFIGURED
    assert await unconfigured.writer.persist_now() is False


async def test_persist_recovery_now_propagates_strict_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, engine_services
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "strict-failure")

    async def _fail_write(*_args: object, **_kwargs: object) -> bool:
        raise OSError("recovery fsync failed")

    monkeypatch.setattr(engine_services(engine).persistence, "save_recovery_session_strict", _fail_write)

    assert await engine.writer.persist_barrier() is RecoveryPersistOutcome.FAILED
    with pytest.raises(OSError, match="recovery fsync failed"):
        await engine.writer.persist_now()


async def test_persist_recovery_now_drains_stale_background_write_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, engine_services
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "strict-order")
    background_started = asyncio.Event()
    release_background = asyncio.Event()
    order: list[str] = []
    original_background = engine_services(engine).persistence.save_recovery_session
    original_strict = engine_services(engine).persistence.save_recovery_session_strict

    async def _parked_background(*args: Any, **kwargs: Any) -> None:
        order.append("background-start")
        background_started.set()
        await release_background.wait()
        await original_background(*args, **kwargs)
        order.append("background-end")

    async def _ordered_strict(*args: Any, **kwargs: Any) -> bool:
        order.append(f"strict-{sum(item.startswith('strict-') for item in order) + 1}")
        return await original_strict(*args, **kwargs)

    monkeypatch.setattr(engine_services(engine).persistence, "save_recovery_session", _parked_background)
    monkeypatch.setattr(engine_services(engine).persistence, "save_recovery_session_strict", _ordered_strict)

    await engine.writer.save_checkpoint()
    await background_started.wait()
    assert engine.current.loaded.loop_recorder is not None
    assert engine.current.loaded.loop_recorder._captured is not None
    engine.current.loaded.loop_recorder._captured[-1] = Message(
        "tool", [Content.from_function_result("c1", result="newer")]
    )

    strict_task = asyncio.create_task(engine.writer.persist_now())
    release_background.set()

    assert await strict_task is True
    assert order == ["background-start", "strict-1", "background-end", "strict-2"]
    restored = await store.load_recovery_session("strict-order")
    assert restored is not None
    assert any(
        content.type == "function_result" and content.result == "newer"
        for message in restored["messages"]
        for content in message.contents
    )


async def test_persist_recovery_now_cancellation_keeps_clean_save_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "strict-cancel")
    sidecar = store.session_dir("strict-cancel") / SESSION_RECOVERY_FILE_NAME
    strict_started = ThreadEvent()
    release_strict = ThreadEvent()
    clean_started = ThreadEvent()
    original_recovery = store.save_recovery_session
    original_primary = store._save_session_sync

    def _parked_recovery(*args: Any, **kwargs: Any) -> None:
        strict_started.set()
        release_strict.wait()
        original_recovery(*args, **kwargs)

    def _observed_primary(*args: Any, **kwargs: Any) -> None:
        clean_started.set()
        original_primary(*args, **kwargs)

    monkeypatch.setattr(store, "save_recovery_session", _parked_recovery)
    monkeypatch.setattr(store, "_save_session_sync", _observed_primary)

    strict_task = asyncio.create_task(engine.writer.persist_now())
    assert await wait_until(strict_started.is_set, timeout=1.0, interval=0.01)
    strict_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await strict_task

    clean_task = asyncio.create_task(engine.writer.save_current_session())
    clean_started_early = await wait_until(clean_started.is_set, timeout=0.2, interval=0.01)
    release_strict.set()
    assert await clean_task is True

    assert not clean_started_early
    assert clean_started.is_set()
    assert not sidecar.exists()


async def test_llm_boundary_checkpoint_never_awaits_the_writer(tmp_path: Path, *, engine_services) -> None:
    """The plain ``on_checkpoint`` path stays fire-and-forget (§2.5).

    The consumed-injection callback awaits the sidecar write, but that flush
    must not creep into the per-LLM-boundary hot path: with the disk writer
    parked on an event nobody has set yet, ``_save_recovery_checkpoint`` (the
    exact callback wired to ``LoopRecorder.on_checkpoint``) still returns —
    an implementation that awaited the writer would deadlock here.
    """
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "hot")
    sidecar = store.session_dir("hot") / SESSION_RECOVERY_FILE_NAME

    release = asyncio.Event()
    original = engine_services(engine).persistence.save_recovery_session

    async def parked_save(*args: Any, **kwargs: Any) -> None:
        await release.wait()
        await original(*args, **kwargs)

    engine_services(engine).persistence.save_recovery_session = parked_save  # type: ignore[method-assign]

    # Two boundary checkpoints while the writer is parked: both return, the
    # snapshots coalesce newest-wins, and nothing has reached disk yet.
    await engine.writer.save_checkpoint()
    await engine.writer.save_checkpoint()

    assert engine.writer.write_task is not None
    assert not engine.writer.write_task.done()
    assert not sidecar.exists()

    release.set()
    await engine.writer.flush()
    assert sidecar.exists()


async def test_primary_save_flushes_pending_checkpoint_then_deletes_sidecar(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "ord")
    sidecar = store.session_dir("ord") / SESSION_RECOVERY_FILE_NAME

    await engine.writer.save_checkpoint()
    assert engine.writer.write_task is not None

    # A clean primary save must flush the in-flight checkpoint first, so its
    # structural sidecar delete is the last write — no stale sidecar survives.
    await engine.writer.save_current_session()

    assert not sidecar.exists()
    assert engine.writer.pending is None


async def test_flush_recovery_checkpoint_survives_outer_cancellation(tmp_path: Path, *, engine_services) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "cancel")
    sidecar = store.session_dir("cancel") / SESSION_RECOVERY_FILE_NAME

    started = asyncio.Event()
    release = asyncio.Event()
    original = engine_services(engine).persistence.save_recovery_session

    async def slow_save(*args: Any, **kwargs: Any) -> None:
        started.set()
        await release.wait()
        await original(*args, **kwargs)

    engine_services(engine).persistence.save_recovery_session = slow_save  # type: ignore[method-assign]

    await engine.writer.save_checkpoint()
    assert engine.writer.write_task is not None

    # Mimic the run task parking in the post-run flush, then being cancelled by
    # the graceful-shutdown timeout while the writer is mid-write.
    waiter = asyncio.create_task(engine.writer.flush())
    await started.wait()
    waiter.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await waiter

    # The writer must survive the outer cancellation (shield), not be aborted,
    # so the trailing shutdown flush can still drain it to disk.
    assert not engine.writer.write_task.cancelled()

    release.set()
    await engine.writer.flush()
    assert sidecar.exists()


async def test_successful_primary_save_clears_recovered_sidecar_marker(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = assemble_agent_engine(EventBus(), settings=Settings(), state_store=store)
    engine.session.session_id = "restore_me"
    engine.session.recovered_from_sidecar = True

    install_loaded_agent(
        engine,
        bindings=_HistoryStateExecutor(  # type: ignore[assignment]
            {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 10}
        ),
    )
    await _seed_recovery_sidecar(store)

    await engine.writer.save_current_session()

    assert engine.recovered_from_sidecar is False
    assert not (store.session_dir("restore_me") / SESSION_RECOVERY_FILE_NAME).exists()


async def test_save_current_session_finalizes_pending_records_only_after_written(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = assemble_agent_engine(EventBus(), settings=Settings(), state_store=store)
    engine.session.session_id = "pending-save"
    session_dir = store.session_dir("pending-save")
    tools = SubAgentTools(event_bus=EventBus(), session_id="pending-save", session_dir=session_dir)
    install_loaded_agent(engine, sub_agent_tools=tools)

    pending_file = session_dir / "sub_agents" / "pending" / "done.json"
    pending_file.parent.mkdir(parents=True)
    # Write the record the way production does (owner-only). finalize_pending_cleanups
    # deletes it through secure_unlink_owner_verified, which requires an owner-verified
    # file; a plain write_text on an elevated Windows runner is owned by Administrators
    # (the token's default owner), so the secure unlink would correctly refuse it.
    atomic_write_owner_only_text(pending_file, "{}")
    tools.queue_pending_cleanup(pending_file)
    install_loaded_agent(
        engine,
        bindings=_HistoryStateExecutor(  # type: ignore[assignment]
            {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1}
        ),
    )

    assert await engine.writer.save_current_session() is True
    assert not pending_file.exists()

    skipped_file = session_dir / "sub_agents" / "pending" / "skipped.json"
    atomic_write_owner_only_text(skipped_file, "{}")
    tools.queue_pending_cleanup(skipped_file)
    install_loaded_agent(
        engine,
        bindings=_HistoryStateExecutor(  # type: ignore[assignment]
            {"messages": [], "compressed_msgs": [], "turn_counter": 1}
        ),
    )

    assert await engine.writer.save_current_session() is False
    assert skipped_file.exists()


async def test_save_current_session_flushes_buffered_otel_lines_after_materializing_root(tmp_path: Path) -> None:
    previous_sink = get_otel_sink()
    sink = OtelSessionSink(write_files=True)
    set_otel_sink(sink)
    try:
        store = JsonFileStateStore(tmp_path)
        engine = assemble_agent_engine(EventBus(), settings=Settings(), state_store=store)
        engine.session.session_id = "otel-save"
        session_dir = store.session_dir("otel-save")

        install_loaded_agent(
            engine,
            bindings=_HistoryStateExecutor(  # type: ignore[assignment]
                {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1}
            ),
        )
        sink.activate("otel-save", session_dir=session_dir)
        sink.write_logs(['{"before_save": true}\n'])

        assert not session_dir.exists()

        assert await engine.writer.save_current_session() is True

        logs = (session_dir / "otel" / "logs.jsonl").read_text(encoding="utf-8")
        assert "before_save" in logs
    finally:
        set_otel_sink(previous_sink)


async def test_shutdown_timeout_preserves_recovery_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = assemble_agent_engine(EventBus(), settings=Settings(), state_store=store)
    engine.session.session_id = "shutdown_timeout"
    sidecar = store.session_dir("shutdown_timeout") / SESSION_RECOVERY_FILE_NAME
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("chrys.orchestration.engine.session_lifecycle._SHUTDOWN_POST_RUN_TIMEOUT_SECONDS", 0.01)

    class _HangingExecutor:
        running = True

        def __init__(self) -> None:
            self.active_handle = PassHandle("shutdown_timeout", "pass")
            self.interrupt_called = False
            self.close_called = False

        async def abort(self, handle: PassHandle, cause: AbortCause) -> AbortResult:
            assert handle is self.active_handle
            assert cause is AbortCause.OWNER_CLOSE
            self.interrupt_called = True
            return AbortResult.REQUESTED

        async def close(self) -> None:
            self.close_called = True

        @property
        def backend(self):
            return self

        @property
        def inputs(self):
            return self

        @property
        def state(self):
            return self

        @property
        def approval(self):
            return self

        @property
        def tool_events(self):
            return self

    executor = _HangingExecutor()
    install_loaded_agent(engine, bindings=executor)  # type: ignore[assignment]
    install_loaded_agent(engine, prepared=PreparedAgent())
    engine.current.loaded.prepared.own(executor.close)
    save_suppress_values: list[bool] = []

    async def fake_save_current_session() -> None:
        save_suppress_values.append(engine.session.suppress_save)
        if not engine.session.suppress_save:
            sidecar.unlink(missing_ok=True)

    async def never_finishes() -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(engine.writer, "save_current_session", fake_save_current_session)
    engine.turns.turn_state.lease.run_task = asyncio.create_task(never_finishes())

    await engine.lifecycle.close_session()

    assert engine.turns.turn_state.shutdown_used_cancel_fallback is True
    assert save_suppress_values == [True]
    assert sidecar.exists()
    assert engine.session.suppress_save is False
    assert executor.interrupt_called is True
    assert executor.close_called is True
