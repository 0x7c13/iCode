# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the main-screen engine read model."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from chrys.app.tui.screens.main.engine_read_model import EngineReadModel


def test_engine_read_model_loads_rollback_state_off_event_loop() -> None:
    event_loop_thread = threading.get_ident()
    state_threads: list[int] = []

    def available_rollback_turns() -> list[int]:
        state_threads.append(threading.get_ident())
        return [1, 2]

    engine = SimpleNamespace(
        mutation_tracker=object(),
        mutation_coordinator=None,
        current_turn_number=3,
        conversation_revision=7,
        build_generation=11,
        workspace_primary_cwd="/repo/workspace",
        available_rollback_turns=available_rollback_turns,
        turn_prompt_previews=dict,
    )

    state = asyncio.run(EngineReadModel(engine).load_rollback_state())

    assert state is not None
    assert len(state_threads) == 1
    assert state_threads[0] != event_loop_thread


def test_engine_read_model_returns_no_rollback_state_without_tracker() -> None:
    engine = SimpleNamespace(
        mutation_tracker=None,
        mutation_coordinator=None,
        current_turn_number=3,
        available_rollback_turns=lambda: [1, 2],
        turn_prompt_previews=lambda: {1: "first"},
    )

    assert EngineReadModel(engine).rollback_state() is None


def test_engine_read_model_returns_no_rollback_state_without_available_turns() -> None:
    engine = SimpleNamespace(
        mutation_tracker=object(),
        mutation_coordinator=None,
        current_turn_number=3,
        available_rollback_turns=list,
        turn_prompt_previews=lambda: {1: "first"},
    )

    assert EngineReadModel(engine).rollback_state() is None


def test_engine_read_model_builds_rollback_state() -> None:
    tracker = object()
    engine = SimpleNamespace(
        mutation_tracker=tracker,
        mutation_coordinator=None,
        current_turn_number=3,
        conversation_revision=7,
        build_generation=11,
        workspace_primary_cwd="/repo/workspace",
        available_rollback_turns=lambda: [1, 2],
        turn_prompt_previews=lambda: {1: "first", 2: "second"},
    )

    state = EngineReadModel(engine).rollback_state()

    assert state is not None
    assert state.tracker is tracker
    assert state.available_turns == [1, 2]
    assert state.current_turn == 3
    assert state.conversation_revision == 7
    assert state.build_generation == 11
    assert state.workspace_cwd == "/repo/workspace"
    assert state.turn_prompts == {1: "first", 2: "second"}
    # No coordinator surface on the double → no peer hooks.
    assert state.plan_augment is None
    assert state.attribution_refresh is None


def test_engine_read_model_wires_peer_hooks_from_coordinator() -> None:
    """With a live coordinator the read state carries both peer hooks.

    ``plan_augment`` must forward the tracker and use the cwd as both
    warning scope and non-git registry fallback — mirroring the engine's
    rollback-time augmentation.  ``attribution_refresh`` must call the
    ENGINE refresh (reclassify + save-if-changed), never a raw
    ``coordinator.reclassify``: a direct reclassify updates the
    short-circuit signatures without persisting, so a later engine
    refresh reports "unchanged" and serialized read surfaces stay
    stale forever.
    """
    import asyncio

    calls: list[tuple[object, ...]] = []

    class _Coordinator:
        def augment_rollback_plan(self, tracker, plan, *, scope_paths=None, fallback_root=None):
            calls.append(("augment", tracker, plan, scope_paths, fallback_root))
            return "augmented"

    async def _refresh(*, force: bool = False) -> bool:
        calls.append(("refresh", force))
        return True

    tracker = object()
    engine = SimpleNamespace(
        mutation_tracker=tracker,
        mutation_coordinator=_Coordinator(),
        refresh_mutation_attribution=_refresh,
        current_turn_number=3,
        conversation_revision=7,
        build_generation=11,
        workspace_primary_cwd="/repo/workspace",
        available_rollback_turns=lambda: [1, 2],
        turn_prompt_previews=dict,
    )

    state = EngineReadModel(engine).rollback_state()

    assert state is not None
    assert state.plan_augment is not None
    assert state.plan_augment("plan", "/ws") == "augmented"
    assert calls == [("augment", tracker, "plan", ["/ws"], "/ws")]

    calls.clear()
    assert state.attribution_refresh is not None
    assert asyncio.run(state.attribution_refresh()) is True
    assert asyncio.run(state.attribution_refresh()) is True
    assert calls == [("refresh", True), ("refresh", False)]
