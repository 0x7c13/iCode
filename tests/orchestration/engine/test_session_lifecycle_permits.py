# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Startup and reload preserve the ownership of their lifecycle permits."""

from __future__ import annotations

import asyncio
from unittest.mock import create_autospec

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.orchestration.engine.assembly import assemble_agent_engine
from chrys.orchestration.engine.state.lifecycle_permits import RebuildPermit, RebuildPermitDenied
from chrys.service.profiles.agents.schema import AgentProfile


@pytest.mark.parametrize("operation", ["start", "reload"])
@pytest.mark.parametrize("fails", [False, True])
async def test_load_acquires_and_releases_its_own_permit(
    operation: str, fails: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = assemble_agent_engine(EventBus(), settings=Settings())
    observed: list[bool] = []

    async def load(*args: object, **kwargs: object) -> None:
        observed.append(engine.permits.current_task_owns_rebuild_permit())
        if fails:
            raise ValueError("load failed")

    replacement = create_autospec(
        engine.loader.load if operation == "start" else engine.loader.reload, side_effect=load
    )
    monkeypatch.setattr(engine.loader, "load" if operation == "start" else "reload", replacement)
    entry = engine.lifecycle.start if operation == "start" else engine.lifecycle.reload
    if fails:
        with pytest.raises(ValueError, match="load failed"):
            await entry(AgentProfile(name="Code"))
    else:
        await entry(AgentProfile(name="Code"))
    assert observed == [True]
    assert not engine.permits.current_task_owns_rebuild_permit()
    assert not engine.permits.gate_lock.locked()


@pytest.mark.parametrize("operation", ["start", "reload"])
@pytest.mark.parametrize("permit_kind", ["transition", "rebuild"])
async def test_load_uses_the_callers_existing_permit(
    operation: str, permit_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = assemble_agent_engine(EventBus(), settings=Settings())
    if permit_kind == "transition":
        permit = await engine.permits.begin_session_transition("restore")
    else:
        permit = await engine.permits.acquire_rebuild_permit(engine.permits.capture_control_token())
        assert isinstance(permit, RebuildPermit)
    acquire = create_autospec(engine.permits.acquire_rebuild_permit, side_effect=AssertionError("already owned"))
    release = create_autospec(engine.permits.release_rebuild_permit, side_effect=AssertionError("caller owns release"))
    real_release = engine.permits.release_rebuild_permit
    monkeypatch.setattr(engine.permits, "acquire_rebuild_permit", acquire)
    monkeypatch.setattr(engine.permits, "release_rebuild_permit", release)
    loader = create_autospec(engine.loader.load if operation == "start" else engine.loader.reload)
    monkeypatch.setattr(engine.loader, "load" if operation == "start" else "reload", loader)
    try:
        entry = engine.lifecycle.start if operation == "start" else engine.lifecycle.reload
        await entry(AgentProfile(name="Code"))
        loader.assert_awaited_once()
        acquire.assert_not_awaited()
        release.assert_not_called()
        assert engine.permits.gate_lock.locked()
    finally:
        if isinstance(permit, str):
            engine.permits.finish_session_transition(permit)
        else:
            real_release(permit)


@pytest.mark.parametrize("operation", ["start", "reload"])
@pytest.mark.parametrize("permit_kind", ["valid", "released_stale", "other_task"])
async def test_load_with_rebuild_permit_validates_caller_ownership(
    operation: str, permit_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = assemble_agent_engine(EventBus(), settings=Settings())
    real_release = engine.permits.release_rebuild_permit
    finish = asyncio.Event()
    owned: asyncio.Future[RebuildPermit] = asyncio.get_running_loop().create_future()
    owner_valid: list[bool] = []

    async def owner() -> None:
        permit = await engine.permits.acquire_rebuild_permit(engine.permits.capture_control_token())
        assert isinstance(permit, RebuildPermit)
        try:
            owned.set_result(permit)
            await finish.wait()
            engine.permits.ensure_rebuild_permit(permit)
            owner_valid.append(engine.permits.current_task_owns_rebuild_permit())
            assert engine.permits.gate_lock.locked()
        finally:
            real_release(permit)

    task = asyncio.create_task(owner()) if permit_kind == "other_task" else None
    if task is not None:
        permit = await owned
    else:
        permit = await engine.permits.acquire_rebuild_permit(engine.permits.capture_control_token())
        assert isinstance(permit, RebuildPermit)
        if permit_kind == "released_stale":
            real_release(permit)

    gate_locked = engine.permits.gate_lock.locked()
    acquire = create_autospec(engine.permits.acquire_rebuild_permit, side_effect=AssertionError("must not acquire"))
    release = create_autospec(engine.permits.release_rebuild_permit, side_effect=AssertionError("caller owns release"))
    load = create_autospec(engine.loader.load)
    reload = create_autospec(engine.loader.reload)
    monkeypatch.setattr(engine.permits, "acquire_rebuild_permit", acquire)
    monkeypatch.setattr(engine.permits, "release_rebuild_permit", release)
    monkeypatch.setattr(engine.loader, "load", load)
    monkeypatch.setattr(engine.loader, "reload", reload)
    try:
        entry = (
            engine.lifecycle.start_with_rebuild_permit
            if operation == "start"
            else engine.lifecycle.reload_with_rebuild_permit
        )
        if permit_kind == "valid":
            await entry(permit, AgentProfile(name="Code"))
            (load if operation == "start" else reload).assert_awaited_once()
            (reload if operation == "start" else load).assert_not_awaited()
            assert engine.permits.current_task_owns_rebuild_permit()
        else:
            with pytest.raises(RuntimeError, match="Invalid rebuild permit"):
                await entry(permit, AgentProfile(name="Code"))
            load.assert_not_awaited()
            reload.assert_not_awaited()
        acquire.assert_not_awaited()
        release.assert_not_called()
        assert engine.permits.gate_lock.locked() is gate_locked
    finally:
        if task is not None:
            finish.set()
            await task
        elif permit_kind == "valid":
            real_release(permit)
    if task is not None:
        assert owner_valid == [True]
    assert not engine.permits.gate_lock.locked()


@pytest.mark.parametrize("operation", ["start", "reload"])
async def test_denied_load_does_not_release_another_tasks_permit(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = assemble_agent_engine(EventBus(), settings=Settings())
    owned = asyncio.Event()
    finish = asyncio.Event()

    async def owner() -> None:
        permit = await engine.permits.acquire_rebuild_permit(engine.permits.capture_control_token())
        assert isinstance(permit, RebuildPermit)
        owned.set()
        try:
            await finish.wait()
        finally:
            engine.permits.release_rebuild_permit(permit)

    task = asyncio.create_task(owner())
    await owned.wait()
    denied = RebuildPermitDenied(reason="busy", code="busy", message="other owner")
    acquire = create_autospec(engine.permits.acquire_rebuild_permit, return_value=denied)
    monkeypatch.setattr(engine.permits, "acquire_rebuild_permit", acquire)
    try:
        entry = engine.lifecycle.start if operation == "start" else engine.lifecycle.reload
        with pytest.raises(RuntimeError, match="other owner"):
            await entry(AgentProfile(name="Code"))
        assert engine.permits.gate_lock.locked()
    finally:
        finish.set()
        await task
    assert not engine.permits.gate_lock.locked()
