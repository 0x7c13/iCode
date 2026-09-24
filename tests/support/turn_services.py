# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Explicit assembly of turn services for tests with narrow resource doubles."""

from __future__ import annotations

from chrys.orchestration.engine.run.active_injection import ActiveTurnInjector
from chrys.orchestration.engine.run.coordinator import TurnCoordinator
from chrys.orchestration.engine.run.finalizer import TurnFinalizer
from chrys.orchestration.engine.run.prompt_content import PromptContentPreparer
from chrys.orchestration.engine.run.retry import RetryCoordinator
from chrys.orchestration.engine.run.runner import TurnRunner
from chrys.orchestration.engine.run.runtime_skills import RuntimeSkillRefresher
from chrys.orchestration.engine.run.turn_hooks import PromptSubmitGate
from tests.support.components import make_hooks


def make_turn_coordinator(
    *,
    current=None,
    session=None,
    turn_state=None,
    permits=None,
    writer=None,
    loader=None,
    bus=None,
    fsm=None,
    history=None,
    trajectory_recorder=None,
    workspace_change_tracker=None,
    settings_handle=None,
    persistence=None,
    on_successful_turn=None,
    on_turn_started=None,
    prompt_content_preparer=None,
    run_and_save=None,
    retry_and_save=None,
):
    """Assemble only explicit dependencies, retaining absent test resources as None."""
    hooks = make_hooks(session=session, current=current)
    coordinator = TurnCoordinator(
        current=current,
        session=session,
        turn_state=turn_state,
        permits=permits,
        writer=writer,
        loader=loader,
        bus=bus,
        fsm=fsm,
        history=history,
        trajectory_recorder=trajectory_recorder,
        workspace_change_tracker=workspace_change_tracker,
        settings_handle=settings_handle,
        persistence=persistence,
        on_successful_turn=on_successful_turn,
        on_turn_started=on_turn_started,
        hooks=hooks,
        prompt_content_preparer=prompt_content_preparer,
    )
    if run_and_save is not None:
        coordinator.run_and_save = run_and_save
    if retry_and_save is not None:
        coordinator.retry_and_save = retry_and_save
    return coordinator


def make_turn_runner(
    *,
    current=None,
    session=None,
    turn_state=None,
    permits=None,
    writer=None,
    loader=None,
    bus=None,
    fsm=None,
    history=None,
    trajectory_recorder=None,
    workspace_change_tracker=None,
    settings_handle=None,
    persistence=None,
    on_successful_turn=None,
    on_turn_started=None,
    prompt_content_preparer=None,
    retry_and_save=None,
):
    """Assemble only explicit dependencies, retaining absent test resources as None."""
    hooks = make_hooks(session=session, current=current)
    gate = PromptSubmitGate(session=session, current=current, bus=bus, fsm=fsm)
    content = prompt_content_preparer or PromptContentPreparer(
        session=session, current=current, bus=bus, history=history, fsm=fsm
    )
    skills = RuntimeSkillRefresher(current=current, loader=loader, session=session, bus=bus)
    injector = ActiveTurnInjector(
        turn_state=turn_state,
        current=current,
        permits=permits,
        session=session,
        fsm=fsm,
        bus=bus,
        gate=gate,
        content=content,
        skills=skills,
    )
    finalizer = TurnFinalizer(
        current=current,
        session=session,
        turn_state=turn_state,
        writer=writer,
        history=history,
        trajectory_recorder=trajectory_recorder,
        fsm=fsm,
        workspace_change_tracker=workspace_change_tracker,
        settings_handle=settings_handle,
        bus=bus,
        persistence=persistence,
        on_successful_turn=on_successful_turn,
        hooks=hooks,
    )

    def retry_factory():
        return RetryCoordinator(
            turn_state=turn_state,
            current=current,
            permits=permits,
            session=session,
            fsm=fsm,
            history=history,
            bus=bus,
            trajectory_recorder=trajectory_recorder,
            gate=gate,
            injector=injector,
            content=content,
            skills=skills,
            retry_and_save=retry_and_save,
        )

    return TurnRunner(
        current=current,
        session=session,
        turn_state=turn_state,
        history=history,
        workspace_change_tracker=workspace_change_tracker,
        settings_handle=settings_handle,
        trajectory_recorder=trajectory_recorder,
        fsm=fsm,
        on_turn_started=on_turn_started,
        finalizer=finalizer,
        hooks=hooks,
        skills=skills,
        content=content,
        retry_factory=retry_factory,
    )


def make_turn_retry(
    *,
    current=None,
    session=None,
    turn_state=None,
    permits=None,
    loader=None,
    bus=None,
    fsm=None,
    history=None,
    trajectory_recorder=None,
    prompt_content_preparer=None,
    retry_and_save=None,
):
    """Assemble only explicit dependencies, retaining absent test resources as None."""
    gate = PromptSubmitGate(session=session, current=current, bus=bus, fsm=fsm)
    content = prompt_content_preparer or PromptContentPreparer(
        session=session, current=current, bus=bus, history=history, fsm=fsm
    )
    skills = RuntimeSkillRefresher(current=current, loader=loader, session=session, bus=bus)
    injector = ActiveTurnInjector(
        turn_state=turn_state,
        current=current,
        permits=permits,
        session=session,
        fsm=fsm,
        bus=bus,
        gate=gate,
        content=content,
        skills=skills,
    )
    return RetryCoordinator(
        turn_state=turn_state,
        current=current,
        permits=permits,
        session=session,
        fsm=fsm,
        history=history,
        bus=bus,
        trajectory_recorder=trajectory_recorder,
        gate=gate,
        injector=injector,
        content=content,
        skills=skills,
        retry_and_save=retry_and_save,
    )


def make_turn_finalizer(
    *,
    current=None,
    session=None,
    turn_state=None,
    writer=None,
    bus=None,
    fsm=None,
    history=None,
    trajectory_recorder=None,
    workspace_change_tracker=None,
    settings_handle=None,
    persistence=None,
    on_successful_turn=None,
):
    """Assemble only explicit dependencies, retaining absent test resources as None."""
    hooks = make_hooks(session=session, current=current)
    finalizer = TurnFinalizer(
        current=current,
        session=session,
        turn_state=turn_state,
        writer=writer,
        history=history,
        trajectory_recorder=trajectory_recorder,
        fsm=fsm,
        workspace_change_tracker=workspace_change_tracker,
        settings_handle=settings_handle,
        bus=bus,
        persistence=persistence,
        on_successful_turn=on_successful_turn,
        hooks=hooks,
    )
    return finalizer
