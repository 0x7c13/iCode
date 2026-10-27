# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The tool path's check and commit: one operation per call, checked by its hooks, committed at the last step.

Written from the plan (``.plan/hook-ask-confirm.md`` §1, §7): only the version
the hooks checked through to the end runs; a change after the check, an
operation that ended meanwhile, or a call that skipped the check does not run;
an approval edit makes a new version that is checked (and asked about) again.
"""

from __future__ import annotations

import sys
import textwrap
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.service.agent_middleware._metadata_keys import (
    _APPROVAL_REJECTED_KEY,
    _HOOK_OPERATION_KEY,
    _REJECTION_SOURCE_KEY,
)
from chrys.service.agent_middleware.control.hook_commit import HookCommitMiddleware
from chrys.service.agent_middleware.events.hook_dispatch import apply_before_tool_hooks
from chrys.service.hooks.confirmation import HookConfirmationService
from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.manager import HookManager
from chrys.service.hooks.operation import HookOperation
from chrys.service.hooks.schema import HookConfig, HookExecution, HookRun, HooksFile
from tests.service.agent_middleware._event_fakes import _ctx
from tests.support.hook_confirmations import FakeConfirmationFrontend

_SESSION = "s1"


def _manager(hooks_dir: Path, *results: dict[str, Any]) -> HookManager:
    hooks = []
    for index, result in enumerate(results):
        script = textwrap.dedent(
            f"""
            import json, os
            with open(os.environ["CHRYS_HOOK_RESULT"], "w") as f:
                json.dump({result!r}, f)
            """
        )
        hooks.append(
            HookConfig(
                id=f"h{index}",
                event=HookEvent.BEFORE_TOOL_CALL,
                run=HookRun(type="command", argv=[sys.executable, "-c", script]),
                execution=HookExecution(mode="blocking", timeout_seconds=10),
            )
        )
    return HookManager(file=HooksFile(hooks=hooks), hooks_dir=hooks_dir)


async def _check(manager: HookManager | None, context: Any) -> bool:
    return await apply_before_tool_hooks(
        manager=manager,
        context=context,
        session_id=_SESSION,
        profile_name="Code",
        tool_name=context.function.name,
        kind="",
        call_id="call-1",
        args=context.arguments,
        workspace_cwd="",
    )


class _Tool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def next_for(self, context: Any):
        async def _next() -> None:
            self.calls.append(dict(context.arguments))
            context.result = "ran"

        return _next


@pytest.fixture
def hooks_dir(tmp_path: Path) -> Path:
    path = tmp_path / "hooks"
    path.mkdir()
    return path


@pytest.mark.asyncio
async def test_a_checked_call_runs_once_with_the_checked_arguments() -> None:
    context = _ctx("edit", args={"path": "a"})
    tool = _Tool()

    assert await _check(None, context) is False
    await HookCommitMiddleware().process(context, tool.next_for(context))
    # A second pass through the commit (a replayed middleware chain) does not run the tool again.
    await HookCommitMiddleware().process(context, tool.next_for(context))

    assert tool.calls == [{"path": "a"}]
    assert context.metadata[_REJECTION_SOURCE_KEY] == "hook"
    assert "already ran" in context.result


@pytest.mark.asyncio
async def test_arguments_changed_after_the_check_do_not_run() -> None:
    context = _ctx("edit", args={"path": "a"})
    tool = _Tool()

    await _check(None, context)
    context.arguments["path"] = "b"
    await HookCommitMiddleware().process(context, tool.next_for(context))

    assert tool.calls == []
    assert context.metadata[_APPROVAL_REJECTED_KEY] is True
    assert "arguments changed after the before_tool_call hooks checked them" in context.result


@pytest.mark.asyncio
async def test_a_call_that_skipped_the_check_is_refused_not_run() -> None:
    context = _ctx("edit", args={"path": "a"})
    tool = _Tool()

    await HookCommitMiddleware().process(context, tool.next_for(context))

    assert tool.calls == []
    assert "did not finish checking it" in context.result


@pytest.mark.asyncio
async def test_an_operation_ended_after_its_check_does_not_run() -> None:
    context = _ctx("edit", args={"path": "a"})
    tool = _Tool()

    await _check(None, context)
    operation = context.metadata[_HOOK_OPERATION_KEY]
    assert isinstance(operation, HookOperation)
    operation.invalidate("rebuilt")
    await HookCommitMiddleware().process(context, tool.next_for(context))

    assert tool.calls == []
    assert "dropped before it ran (rebuilt)" in context.result


@pytest.mark.asyncio
async def test_a_hook_rewrite_is_the_version_that_runs(hooks_dir: Path) -> None:
    manager = _manager(hooks_dir, {"action": "modify", "args_override": {"path": "safe"}})
    context = _ctx("edit", args={"path": "danger", "mode": "w"})
    tool = _Tool()

    assert await _check(manager, context) is False
    await HookCommitMiddleware().process(context, tool.next_for(context))

    assert tool.calls == [{"path": "safe", "mode": "w"}]


@pytest.mark.asyncio
async def test_an_approval_edit_is_a_new_version_asked_about_again(hooks_dir: Path) -> None:
    bus = EventBus()
    service = HookConfirmationService(bus=bus, enabled=True)
    await service.start()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION, answer_inline="approved")
    await frontend.attach()
    manager = _manager(hooks_dir, {"action": "ask"})
    manager.confirmations = service
    context = _ctx("edit", args={"path": "a"})
    tool = _Tool()

    assert await _check(manager, context) is False
    # The approval dialog's edit, then the approval middleware's recheck of the same call.
    context.arguments = {"path": "b"}
    assert await _check(manager, context) is False
    await HookCommitMiddleware().process(context, tool.next_for(context))

    assert [request.arguments for request in frontend.requests] == [{"path": "a"}, {"path": "b"}]
    assert frontend.requests[0].operation_id == frontend.requests[1].operation_id
    assert tool.calls == [{"path": "b"}]


class _NoTextForm:
    def __str__(self) -> str:
        raise RuntimeError("no text form")


@pytest.mark.asyncio
async def test_arguments_without_a_json_form_run_where_no_hook_checks_them() -> None:
    value = _NoTextForm()
    context = _ctx("edit", args={"value": value})
    tool = _Tool()

    assert await _check(None, context) is False
    await HookCommitMiddleware().process(context, tool.next_for(context))

    assert tool.calls == [{"value": value}]


@pytest.mark.asyncio
async def test_arguments_without_a_json_form_are_refused_where_a_hook_would_check_them(hooks_dir: Path) -> None:
    manager = _manager(hooks_dir, {"action": "allow"})
    context = _ctx("edit", args={"value": _NoTextForm(), "nested": {"path": "safe"}})
    tool = _Tool()

    assert await _check(manager, context) is True
    assert "no JSON form" in str(context.result)
    # Even reached anyway, the commit step finds no checked version to run.
    await HookCommitMiddleware().process(context, tool.next_for(context))

    assert tool.calls == []


@pytest.mark.asyncio
async def test_a_hook_reads_typed_arguments_as_compared_and_the_tool_keeps_their_objects(hooks_dir: Path) -> None:
    class _Plain:
        pass

    # The hook reads the date key as the canonical form writes it, and asks only when it does.
    script = textwrap.dedent(
        """
        import json, os
        with open(os.environ["CHRYS_HOOK_PAYLOAD_FILE"], encoding="utf-8") as f:
            args = json.load(f)["tool"]["args"]
        action = "ask" if args["values"] == {"2026-07-18": "ok"} else "block"
        with open(os.environ["CHRYS_HOOK_RESULT"], "w") as f:
            json.dump({"action": action, "reason": "typed"}, f)
        """
    )
    manager = HookManager(
        file=HooksFile(
            hooks=[
                HookConfig(
                    id="typed",
                    event=HookEvent.BEFORE_TOOL_CALL,
                    run=HookRun(type="command", argv=[sys.executable, "-c", script]),
                    execution=HookExecution(mode="blocking", timeout_seconds=10),
                )
            ]
        ),
        hooks_dir=hooks_dir,
    )
    bus = EventBus()
    service = HookConfirmationService(bus=bus, enabled=True)
    await service.start()
    frontend = FakeConfirmationFrontend(bus, session_id=_SESSION, answer_inline="approved")
    await frontend.attach()
    manager.confirmations = service
    when, plain = date(2026, 7, 18), _Plain()
    context = _ctx("edit", args={"values": {when: "ok"}, "plain": plain})
    tool = _Tool()
    try:
        assert await _check(manager, context) is False
        await HookCommitMiddleware().process(context, tool.next_for(context))
    finally:
        await service.close()

    assert len(frontend.requests) == 1
    ((call,),) = [tool.calls]
    assert call["values"] == {when: "ok"} and call["plain"] is plain


@pytest.mark.asyncio
async def test_a_nested_change_after_the_check_is_a_change() -> None:
    context = _ctx("edit", args={"nested": {"path": "safe"}})
    tool = _Tool()

    assert await _check(None, context) is False
    context.arguments["nested"]["path"] = "danger"
    await HookCommitMiddleware().process(context, tool.next_for(context))

    assert tool.calls == []
    assert "changed after the before_tool_call hooks checked them" in str(context.result)
