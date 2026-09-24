# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Backend admission precedence rejects conflicting errors before side effects."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.models.invocations import PassHandle
from chrys.kernel import AgentResponse, Message
from chrys.orchestration.invoker.contracts import (
    AbortCause,
    ContinuationCapability,
    Failed,
    Ok,
    OverlappingRun,
    PreparedClosed,
    RunIntent,
    RunRequest,
    StaleContinuation,
    UnsupportedRequest,
)
from tests.orchestration.sub_agents._acp_fakes import make_controller
from tests.service.acp_client.helpers import make_spec


@pytest.mark.parametrize("backend_kind", ["kernel", "acp"])
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("closing_ticket", StaleContinuation),
        ("closing_no_ticket", PreparedClosed),
        ("overlap", OverlappingRun),
        ("fresh_ticket", UnsupportedRequest),
        ("continue_ticket", UnsupportedRequest),
        ("missing_ticket", StaleContinuation),
        ("wrong_capability", StaleContinuation),
        ("copied_ticket", StaleContinuation),
        ("foreign_origin", StaleContinuation),
    ],
)
async def test_backend_admission_error_priority(executor, tmp_path, backend_kind, case, expected):
    controller = None
    if backend_kind == "kernel":
        backend = executor.backend
        backend._attempts.run = create_autospec(backend._attempts.run, side_effect=ValueError("failed"))
        origin = executor.inputs.origin
        run = backend.run
    else:
        controller = make_controller(
            EventBus(), tmp_path, spec_factory=lambda ordinal: make_spec(tmp_path, scenario="crash_mid_prompt")
        )
        origin = controller.origin
        run = controller.policy.backend.run
    try:
        failed = await run(RunRequest([Message("user", ["first"])], RunIntent.FRESH, origin))
        assert isinstance(failed, Failed)
        assert failed.continuation is not None
        ticket = failed.continuation
        request = RunRequest([Message("user", ["retry"])], RunIntent.RETRY, origin, ticket)
        if case in {"closing_ticket", "closing_no_ticket", "overlap"}:
            # An active pass conflicts with invalid input and a copied ticket.
            request = replace(request, messages=[], intent=RunIntent.FRESH, continuation=replace(ticket))
            if backend_kind == "kernel":
                backend._active = object()
            else:
                controller.policy.backend._active_handle = PassHandle(origin.invocation_id, "active")
            if case.startswith("closing"):
                if backend_kind == "kernel":
                    await backend.owner.aclose()
                else:
                    controller.policy.backend._owner_close_cause = AbortCause.OWNER_CLOSE
            if case == "closing_no_ticket":
                request = replace(request, continuation=None)
        elif case == "fresh_ticket":
            request = replace(request, intent=RunIntent.FRESH, continuation=replace(ticket))
        elif case == "continue_ticket":
            request = replace(request, intent=RunIntent.CONTINUE, continuation=replace(ticket))
        elif case == "missing_ticket":
            request = replace(request, continuation=None)
        elif case == "wrong_capability":
            other = (
                ContinuationCapability.FRESH_SESSION
                if backend_kind == "kernel"
                else ContinuationCapability.CONTINUE_HISTORY
            )
            request = replace(request, continuation=replace(ticket, capability=other))
        elif case == "copied_ticket":
            request = replace(request, continuation=replace(ticket))
        else:
            request = replace(request, origin=replace(origin, invocation_id="foreign"))

        if backend_kind == "kernel":
            generation = backend.state_generation
            history = backend.export_audit()
            attempts = backend._attempts.run.await_count
        else:
            ordinal = controller.policy.backend.transport_ordinal
            prompt = controller.policy.backend._prompt
            factory = create_autospec(
                controller.policy.backend._spec_factory, side_effect=AssertionError("must not create transport")
            )
            controller.policy.backend._spec_factory = factory
            files = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        with pytest.raises(expected) as caught:
            await run(request)
        assert type(caught.value) is expected
        if backend_kind == "kernel":
            assert backend.state_generation == generation
            assert backend.export_audit() == history
            assert backend._attempts.run.await_count == attempts
        else:
            assert controller.policy.backend.transport_ordinal == ordinal
            assert controller.policy.backend._prompt == prompt
            factory.assert_not_called()
            assert files == {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    finally:
        if controller is not None:
            controller.policy.backend._active_handle = None
            await controller.policy.backend.aclose()
        else:
            backend._active = None


@pytest.mark.parametrize("backend_kind", ["kernel", "acp"])
async def test_foreign_fresh_origin_backend_boundary(executor, tmp_path, backend_kind):
    if backend_kind == "kernel":
        backend = executor.backend
        backend._attempts.run = create_autospec(backend._attempts.run, return_value=AgentResponse(messages=[]))
        foreign = replace(executor.inputs.origin, invocation_id="new-invocation")
        outcome = await backend.run(RunRequest([Message("user", ["new"])], RunIntent.FRESH, foreign))
        assert isinstance(outcome, Ok)
        assert outcome.handle.invocation_id == foreign.invocation_id
    else:
        factory = create_autospec(lambda ordinal: make_spec(tmp_path))
        controller = make_controller(EventBus(), tmp_path, spec_factory=factory)
        try:
            foreign = replace(controller.origin, invocation_id="new-invocation")
            with pytest.raises(UnsupportedRequest) as caught:
                await controller.policy.backend.run(RunRequest([Message("user", ["new"])], RunIntent.FRESH, foreign))
            assert type(caught.value) is UnsupportedRequest
            factory.assert_not_called()
            assert controller.policy.backend.transport_ordinal == 0
        finally:
            await controller.policy.backend.aclose()
