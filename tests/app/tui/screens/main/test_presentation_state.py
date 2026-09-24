# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Attempt-scoped provisional presentation state tests."""

from dataclasses import replace

from chrys.app.tui.screens.main.state import TurnRenderGate
from chrys.foundation.events.types import InvocationMessage, ProvisionalPresentation
from chrys.foundation.models.invocations import InvocationOrigin


def test_render_gate_commits_or_drops_deferred_provisional_messages() -> None:
    gate = TurnRenderGate()
    gate.begin()
    accepted = InvocationMessage(
        text="accepted",
        is_intermediate=True,
        presentation=ProvisionalPresentation("attempt-1", "segment-1"),
        origin=InvocationOrigin("turn", "", "turn-test", None),
    )
    dropped = InvocationMessage(
        text="dropped",
        is_intermediate=True,
        presentation=ProvisionalPresentation("attempt-1", "segment-2"),
        origin=InvocationOrigin("turn", "", "turn-test", None),
    )
    rejected = InvocationMessage(
        text="rejected",
        is_intermediate=True,
        presentation=ProvisionalPresentation("attempt-2", "segment-3"),
        origin=InvocationOrigin("turn", "", "turn-test", None),
    )
    gate.defer(accepted)
    gate.defer(dropped)
    gate.defer(rejected)

    gate.accept_presentation_attempt("attempt-1", ("segment-1",))
    gate.reject_presentation_attempt("attempt-2")

    assert accepted.presentation == ProvisionalPresentation("attempt-1", "segment-1")
    accepted = replace(accepted, presentation=None)
    assert gate.consume_deferred() == [accepted]
    assert accepted.presentation is None
