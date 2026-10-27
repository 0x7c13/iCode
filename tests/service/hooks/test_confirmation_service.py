# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The session's hook confirmation service: one pending table, one way each ask ends.

Written from the plan's rules (``.plan/hook-ask-confirm.md`` §3): an ask that
cannot reach anyone ends at once, every ask ends exactly once, answers are
taken only from the right session for a request still pending, and a frontend
still showing a request that ended some other way is told.
"""

from __future__ import annotations

import asyncio

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.service.hooks.confirmation import HookConfirmationService, argument_changes
from chrys.service.hooks.operation import HookOperation
from tests.support.hook_confirmations import FakeConfirmationFrontend
from tests.support.waiting import wait_for

_SESSION = "s1"


def _tool_operation(args: dict[str, object] | None = None) -> HookOperation:
    return HookOperation(
        subject="tool",
        session_id=_SESSION,
        tool_name="edit_file",
        tool_kind="edit",
        arguments=args if args is not None else {"path": "prod.yaml"},
    )


async def _service(bus: EventBus, *, enabled: bool = True, timeout: float | None = None) -> HookConfirmationService:
    service = HookConfirmationService(bus=bus, enabled=enabled, timeout_seconds=timeout)
    await service.start()
    return service


async def _ask(service: HookConfirmationService, operation: HookOperation):
    return await service.ask(operation, run_id="run-1", hook_id="guard", hook_source="global", reason="prod change")


@pytest.mark.asyncio
async def test_a_disabled_service_answers_unavailable_without_publishing() -> None:
    bus = EventBus()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION)
    await frontend.attach()
    service = await _service(bus, enabled=False)

    result = await _ask(service, _tool_operation())

    assert (result.outcome, result.end_reason) == ("unavailable", "cannot_ask")
    assert frontend.requests == []
    assert service.available() is False


@pytest.mark.asyncio
async def test_a_service_not_started_cannot_ask() -> None:
    bus = EventBus()
    service = HookConfirmationService(bus=bus, enabled=True)

    result = await _ask(service, _tool_operation())

    assert result.outcome == "unavailable"


@pytest.mark.asyncio
async def test_an_unclaimed_request_ends_unavailable_at_once() -> None:
    bus = EventBus()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION, claim=False)
    await frontend.attach()
    service = await _service(bus)

    result = await asyncio.wait_for(_ask(service, _tool_operation()), timeout=5)

    assert (result.outcome, result.end_reason) == ("unavailable", "unclaimed")
    assert len(frontend.requests) == 1
    assert frontend.withdrawn == []


@pytest.mark.asyncio
async def test_the_request_shows_a_copy_of_what_is_asked_about() -> None:
    bus = EventBus()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION, answer_inline="approved")
    await frontend.attach()
    service = await _service(bus)
    args: dict[str, object] = {"path": "prod.yaml", "nested": {"n": 1}}
    operation = _tool_operation(args)

    result = await _ask(service, operation)

    assert result.outcome == "approved"
    request = frontend.requests[0]
    assert (request.session_id, request.operation_id, request.revision) == (_SESSION, operation.operation_id, 1)
    assert (request.run_id, request.hook_id, request.hook_source, request.reason) == (
        "run-1",
        "guard",
        "global",
        "prod change",
    )
    assert (request.subject, request.tool_name, request.tool_kind) == ("tool", "edit_file", "edit")
    assert request.arguments == args
    request.arguments["path"] = "changed by a frontend"
    assert operation.arguments() == args


@pytest.mark.asyncio
async def test_an_answer_published_while_the_request_is_published_is_not_lost() -> None:
    bus = EventBus()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION, answer_inline="declined")
    await frontend.attach()
    service = await _service(bus)

    result = await asyncio.wait_for(_ask(service, _tool_operation()), timeout=5)

    assert (result.outcome, result.end_reason) == ("declined", "user")


@pytest.mark.parametrize(
    ("answer", "outcome", "end_reason"),
    [
        ("approved", "approved", "user"),
        ("declined", "declined", "user"),
        ("dismissed", "declined", "dismissed"),
        ("frontend_unavailable", "unavailable", "frontend_unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_each_answer_maps_to_its_outcome_and_reason(answer, outcome, end_reason) -> None:
    bus = EventBus()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION)
    await frontend.attach()
    service = await _service(bus)

    task = asyncio.create_task(_ask(service, _tool_operation()))
    await wait_for(lambda: frontend.claimed, description="request claimed")
    assert not task.done()
    await frontend.answer(frontend.claimed[0], answer)
    result = await asyncio.wait_for(task, timeout=5)

    assert (result.outcome, result.end_reason) == (outcome, end_reason)
    # The frontend answered: it closed the question itself, nothing to withdraw.
    assert frontend.withdrawn == []


@pytest.mark.asyncio
async def test_answers_from_another_session_or_for_another_request_are_ignored() -> None:
    bus = EventBus()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION)
    await frontend.attach()
    service = await _service(bus)

    task = asyncio.create_task(_ask(service, _tool_operation()))
    await wait_for(lambda: frontend.claimed, description="request claimed")
    request_id = frontend.claimed[0]
    await frontend.answer(request_id, "approved", session_id="another-session")
    await frontend.answer("another-request", "approved")
    assert not task.done()
    await frontend.answer(request_id, "declined")
    result = await asyncio.wait_for(task, timeout=5)
    # A late or repeated answer changes nothing once the ask ended.
    await frontend.answer(request_id, "approved")

    assert result.outcome == "declined"


@pytest.mark.asyncio
async def test_an_invalidated_operation_ends_its_ask_and_withdraws_the_question() -> None:
    bus = EventBus()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION)
    await frontend.attach()
    service = await _service(bus)
    operation = _tool_operation()

    task = asyncio.create_task(_ask(service, operation))
    await wait_for(lambda: frontend.claimed, description="request claimed")
    operation.invalidate("turn_ended")
    result = await asyncio.wait_for(task, timeout=5)

    assert (result.outcome, result.end_reason) == ("invalidated", "turn_ended")
    assert [(w.request_id, w.outcome) for w in frontend.withdrawn] == [(frontend.claimed[0], "invalidated")]
    # The ask is over: an answer now is too late.
    await frontend.answer(frontend.claimed[0], "approved")
    assert operation.state == "invalidated"


@pytest.mark.asyncio
async def test_an_operation_already_ended_is_not_asked_about() -> None:
    bus = EventBus()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION)
    await frontend.attach()
    service = await _service(bus)
    operation = _tool_operation()
    operation.invalidate("withdrawn")

    result = await _ask(service, operation)

    assert (result.outcome, result.end_reason) == ("invalidated", "withdrawn")
    assert frontend.requests == []


@pytest.mark.asyncio
async def test_an_approval_of_a_version_that_changed_since_does_not_count() -> None:
    bus = EventBus()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION)
    await frontend.attach()
    service = await _service(bus)
    operation = _tool_operation({"path": "a"})

    task = asyncio.create_task(_ask(service, operation))
    await wait_for(lambda: frontend.claimed, description="request claimed")
    assert operation.replace_arguments({"path": "b"}) is True
    await frontend.answer(frontend.claimed[0], "approved")
    result = await asyncio.wait_for(task, timeout=5)

    assert (result.outcome, result.end_reason) == ("invalidated", "revision_changed")
    # The frontend answered, whatever the answer came to: it put the question away itself.
    assert frontend.withdrawn == []


@pytest.mark.asyncio
async def test_an_ask_times_out_after_the_configured_time_and_withdraws_the_question() -> None:
    bus = EventBus()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION)
    await frontend.attach()
    service = await _service(bus, timeout=0.05)

    result = await asyncio.wait_for(_ask(service, _tool_operation()), timeout=5)

    assert (result.outcome, result.end_reason) == ("timed_out", "timeout")
    assert [w.outcome for w in frontend.withdrawn] == ["timed_out"]


@pytest.mark.asyncio
async def test_a_cancelled_ask_withdraws_the_question_and_cancels_the_caller() -> None:
    bus = EventBus()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION)
    await frontend.attach()
    service = await _service(bus)

    task = asyncio.create_task(_ask(service, _tool_operation()))
    await wait_for(lambda: frontend.claimed, description="request claimed")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [w.outcome for w in frontend.withdrawn] == ["interrupted"]
    # Nothing waits any more: an answer now finds no pending ask.
    await frontend.answer(frontend.claimed[0], "approved")


@pytest.mark.asyncio
async def test_only_one_of_two_frontends_claims_a_request() -> None:
    bus = EventBus()
    first, second = (
        FakeConfirmationFrontend(bus, session_id=_SESSION),
        FakeConfirmationFrontend(bus, session_id=_SESSION),
    )
    await first.attach()
    await second.attach()
    service = await _service(bus)

    task = asyncio.create_task(_ask(service, _tool_operation()))
    await wait_for(lambda: first.requests and second.requests, description="both frontends saw the request")
    assert len(first.claimed) + len(second.claimed) == 1
    claimer = first if first.claimed else second
    await claimer.answer(claimer.claimed[0], "approved")

    assert (await asyncio.wait_for(task, timeout=5)).outcome == "approved"


@pytest.mark.asyncio
async def test_closing_the_service_ends_every_pending_ask() -> None:
    bus = EventBus()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION)
    await frontend.attach()
    service = await _service(bus)

    tasks = [asyncio.create_task(_ask(service, _tool_operation())) for _ in range(2)]
    await wait_for(lambda: len(frontend.claimed) == 2, description="both requests claimed")
    await service.close()
    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)

    assert {(r.outcome, r.end_reason) for r in results} == {("unavailable", "closed")}
    assert sorted(w.request_id for w in frontend.withdrawn) == sorted(frontend.claimed)
    assert (await _ask(service, _tool_operation())).outcome == "unavailable"


class _HoldingFrontend(FakeConfirmationFrontend):
    """Claims a request, then holds its handler (and so the publish) until released."""

    def __init__(self, bus: EventBus) -> None:
        super().__init__(bus, session_id=_SESSION)
        self.release = asyncio.Event()

    async def _on_request(self, event):  # type: ignore[override]
        await super()._on_request(event)
        await self.release.wait()


@pytest.mark.parametrize("ending", ["invalidated", "timed_out", "closed"])
@pytest.mark.asyncio
async def test_a_publish_held_by_a_handler_does_not_hold_up_the_end(ending: str) -> None:
    bus = EventBus()
    frontend = _HoldingFrontend(bus)
    await frontend.attach()
    service = await _service(bus, timeout=0.05 if ending == "timed_out" else None)
    operation = _tool_operation()

    task = asyncio.create_task(_ask(service, operation))
    await wait_for(lambda: frontend.claimed, description="request claimed, its handler held")
    if ending == "invalidated":
        operation.invalidate("turn_ended")
    elif ending == "closed":
        await asyncio.wait_for(service.close(), timeout=5)
    result = await asyncio.wait_for(task, timeout=5)

    assert result.outcome == ("unavailable" if ending == "closed" else ending)
    # Still claimed by the held handler: told the request ended.
    assert [w.request_id for w in frontend.withdrawn] == frontend.claimed
    frontend.release.set()


@pytest.mark.asyncio
async def test_a_cancelled_close_still_winds_every_ask_down() -> None:
    bus = EventBus()
    frontend = _HoldingFrontend(bus)
    await frontend.attach()
    service = await _service(bus)

    task = asyncio.create_task(_ask(service, _tool_operation()))
    await wait_for(lambda: frontend.claimed, description="request claimed, its handler held")
    closing = asyncio.create_task(service.close())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    result = await asyncio.wait_for(task, timeout=5)
    # A second close shares the first one's work and returns once it is done.
    await asyncio.wait_for(service.close(), timeout=5)

    assert (result.outcome, result.end_reason) == ("unavailable", "closed")
    assert [w.request_id for w in frontend.withdrawn] == frontend.claimed
    # No longer subscribed: an answer now reaches nothing.
    assert service.available() is False
    frontend.release.set()


@pytest.mark.asyncio
async def test_an_ask_cancelled_twice_is_still_wound_down() -> None:
    bus = EventBus()
    frontend = _HoldingFrontend(bus)
    await frontend.attach()
    service = await _service(bus)

    task = asyncio.create_task(_ask(service, _tool_operation()))
    await wait_for(lambda: frontend.claimed, description="request claimed, its handler held")
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)

    assert [w.outcome for w in frontend.withdrawn] == ["interrupted"]
    frontend.release.set()


def test_a_long_argument_change_is_shown_where_it_changed_not_from_its_shared_start() -> None:
    script = "echo start\n" + "x" * 2000 + "\nrm -rf build\n" + "y" * 2000
    (change,) = argument_changes({"command": script}, {"command": script.replace("rm -rf build", "rm -rf /")})

    assert change.name == "command"
    assert change.before is not None and change.after is not None
    assert "rm -rf build" in change.before and "rm -rf /" in change.after
    # A little shared text around the change, the rest left out on both sides.
    assert change.before.startswith("…") and change.before.endswith("…")
    assert len(change.before) < 300 and len(change.after) < 300


def test_short_argument_changes_are_shown_whole_and_long_added_or_removed_ones_are_cut() -> None:
    changes = argument_changes({"path": "x", "old": "a" * 1000}, {"path": "y", "force": True})

    assert [(c.name, c.before, c.after) for c in changes[:2]] == [("path", '"x"', '"y"'), ("force", None, "true")]
    removed = changes[2]
    assert (removed.name, removed.after) == ("old", None)
    assert removed.before is not None and removed.before.startswith('"aaa') and removed.before.endswith("…")
    assert len(removed.before) < 300
