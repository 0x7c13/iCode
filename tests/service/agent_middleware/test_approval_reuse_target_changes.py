# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Approval must cover the actual destination and the actual typed arguments."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from enum import Enum, IntEnum, StrEnum
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import create_model

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import ApprovalRequest, ApprovalResponse
from chrys.foundation.models.session_env import SessionEnvironment
from chrys.foundation.models.workspace import Workspace
from chrys.kernel import FunctionTool
from chrys.kernel.middleware import FunctionInvocationContext
from chrys.service.agent_middleware.control.approval import ApprovalMiddleware
from chrys.service.approval.argument_snapshot import argument_snapshot
from chrys.service.approval.policy import ApprovalPolicy
from chrys.service.approval.reuse_binding import ApprovalReuseBinding
from chrys.service.profiles.agents.schema import ApprovalConfig
from chrys.service.tools.builtins import filesystem
from chrys.service.tools.builtins.filesystem import FilesystemTools
from tests.kernel._fakes import _call_response, _result_contents, _stack, _text_response, _user
from tests.support.symlinks import symlink_or_skip


@pytest.fixture
def setup(tmp_path):
    runtime = SessionEnvironment.capture("session-a", Workspace.from_cwd(str(tmp_path)))
    runtime = replace(runtime, platform=replace(runtime.platform, config_dir=tmp_path / "config"))
    tools = FilesystemTools(runtime).tools()
    return tools, ApprovalReuseBinding(runtime, tools), EventBus(), ApprovalPolicy(ApprovalConfig(default="require"))


@pytest.fixture
def destinations(tmp_path):
    first, second, link = (tmp_path / name for name in ("first", "second", "output"))
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / "result.txt").write_text("original", encoding="utf-8")
    symlink_or_skip(link, first, target_is_directory=True)
    return first, second, link


def retarget(link, destination):
    link.unlink()
    symlink_or_skip(link, destination, target_is_directory=True)


def file_context(tools, operation):
    tool = next(tool for tool in tools if tool.name == operation)
    args = {"path": "output/result.txt"}
    if operation == "write_file":
        args.update(content="changed", overwrite=True)
    else:
        args.update(old_string="original", new_string="changed")
    return FunctionInvocationContext(tool, args)


@pytest.mark.parametrize("operation", ["write_file", "edit_file"])
@pytest.mark.parametrize("phase", ["before_call", "match", "approval", "remember", "execution"])
async def test_retargeted_directory_cannot_use_old_file_approval(setup, destinations, monkeypatch, operation, phase):
    tools, binding, bus, policy = setup
    first, second, link = destinations
    context = file_context(tools, operation)
    candidate = binding.candidate(context)
    assert candidate is not None
    if phase in {"before_call", "match", "execution"}:
        assert await asyncio.to_thread(binding.service.remember, candidate, "EXACT_SESSION")
    if phase == "before_call":
        await asyncio.to_thread(retarget, link, second)
    elif phase in {"match", "remember"}:
        original = binding.service.match if phase == "match" else binding.service.remember

        def changed_match(current):
            result = original(current)
            retarget(link, second)
            return result

        def changed_remember(current, choice):
            result = original(current, choice)
            retarget(link, second)
            return result

        monkeypatch.setattr(binding.service, phase, changed_match if phase == "match" else changed_remember)
    requests = []

    async def respond(event):
        requests.append(event)
        if phase == "approval" and len(requests) == 1:
            await asyncio.to_thread(retarget, link, second)
        await bus.publish(
            ApprovalResponse(
                request_id=event.request_id,
                approved=phase in {"approval", "remember"} and len(requests) == 1,
                remember_choice="EXACT_SESSION",
            )
        )

    async def execute():
        if phase == "execution":
            await asyncio.to_thread(retarget, link, second)
        context.result = await context.function.invoke(context=context, skip_parsing=True)

    middleware = ApprovalMiddleware(policy, bus, reuse=binding)
    await bus.subscribe(ApprovalRequest, respond)
    try:
        await middleware.process(context, execute)
        assert (second / "result.txt").read_text(encoding="utf-8") == "original"
        assert (first / "result.txt").read_text(encoding="utf-8") == "original"
        assert str(context.result).startswith("Error:")
        assert len(requests) == (0 if phase == "execution" else 2 if phase in {"approval", "remember"} else 1)
    finally:
        await middleware.close()
        await bus.unsubscribe(ApprovalRequest, respond)


@pytest.mark.parametrize("operation", ["write_file", "edit_file"])
async def test_stable_directory_alias_reuses_file_approval(setup, destinations, operation):
    tools, binding, bus, policy = setup
    first, second, _ = destinations
    context = file_context(tools, operation)
    candidate = binding.candidate(context)
    assert candidate is not None
    assert await asyncio.to_thread(binding.service.remember, candidate, "EXACT_SESSION")
    request = AsyncMock(spec=lambda event: None)
    await bus.subscribe(ApprovalRequest, request)
    middleware = ApprovalMiddleware(policy, bus, reuse=binding)

    async def execute():
        context.result = await context.function.invoke(context=context, skip_parsing=True)

    try:
        await middleware.process(context, execute)
        request.assert_not_awaited()
        assert (first / "result.txt").read_text(encoding="utf-8") == "changed"
        assert (second / "result.txt").read_text(encoding="utf-8") == "original"
    finally:
        await middleware.close()
        await bus.unsubscribe(ApprovalRequest, request)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (("safe",), ("changed",)),
        ({"safe"}, {"changed"}),
        (Path("safe"), Path("changed")),
        (float("nan"), float("inf")),
        ((["safe"],), (["changed"],)),
        ((1,), (True,)),
    ],
)
def test_snapshot_preserves_typed_argument_identity(before, after):
    assert argument_snapshot(before) != argument_snapshot(after)


def test_old_unstructured_grants_are_not_adopted(setup, destinations):
    tools, binding, _, _ = setup
    candidate = binding.candidate(file_context(tools, "write_file"))
    assert candidate is not None
    old = {"id": "old", "scope": "SESSION", "display": "file_paths", "path_resolution": "physical"}
    assert binding.service.session_store.add_many([old])
    assert not binding.service.match(candidate)
    assert binding.service.session_store.load() == [old]
    assert binding.service.remember(candidate, "EXACT_SESSION")
    assert binding.service.match(candidate)


@pytest.mark.parametrize("operation", ["write_file", "edit_file"])
async def test_final_file_symlink_uses_ordinary_approval_without_changing_replace_semantics(setup, tmp_path, operation):
    tools, binding, bus, policy = setup
    target = tmp_path / "target.txt"
    target.write_text("original", encoding="utf-8")
    link = tmp_path / "leaf.txt"
    symlink_or_skip(link, target)
    context = file_context(tools, operation)
    context.arguments["path"] = str(link)
    assert binding.candidate(context) is None
    requests = []

    async def respond(event):
        requests.append(event)
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True, remember_choice="EXACT_SESSION"))

    async def execute():
        context.result = await context.function.invoke(context=context, skip_parsing=True)

    middleware = ApprovalMiddleware(policy, bus, reuse=binding)
    await bus.subscribe(ApprovalRequest, respond)
    try:
        await middleware.process(context, execute)
        assert len(requests) == 1 and not requests[0].reuse_offer
        assert not binding.service.rules()
        assert not link.is_symlink()
        assert link.read_text(encoding="utf-8") == "changed"
        assert target.read_text(encoding="utf-8") == "original"
    finally:
        await middleware.close()
        await bus.unsubscribe(ApprovalRequest, respond)


async def test_opaque_custom_arguments_keep_ordinary_approval(setup):
    _, binding, bus, policy = setup
    context = FunctionInvocationContext(FunctionTool(name="opaque"), {"value": object()})
    middleware = ApprovalMiddleware(policy, bus, reuse=binding)
    called = AsyncMock(spec=lambda: None)

    async def respond(event):
        assert event.reuse_offer is None
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True))

    await bus.subscribe(ApprovalRequest, respond)
    try:
        await middleware.process(context, called)
        called.assert_awaited_once()
        assert not binding.service.rules()
    finally:
        await middleware.close()
        await bus.unsubscribe(ApprovalRequest, respond)


@pytest.mark.parametrize("operation", ["write_file", "edit_file"])
@pytest.mark.parametrize("phase", ["approval", "execution"])
@pytest.mark.parametrize("changed_link", ["parent", "leaf"])
async def test_final_symlink_destination_remains_tracked_for_one_time_approval(
    setup, destinations, operation, phase, changed_link
):
    tools, binding, bus, policy = setup
    first, second, link = destinations
    for directory in (first, second):
        (directory / "referent.txt").write_text("original", encoding="utf-8")
        (directory / "result.txt").unlink()
        symlink_or_skip(directory / "result.txt", directory / "referent.txt")
    context = file_context(tools, operation)
    assert binding.candidate(context) is None
    requests = []

    def change_target():
        if changed_link == "parent":
            retarget(link, second)
        else:
            (first / "result.txt").unlink()
            symlink_or_skip(first / "result.txt", second / "referent.txt")

    async def respond(event):
        requests.append(event)
        if len(requests) == 1 and phase == "approval":
            await asyncio.to_thread(change_target)
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=len(requests) == 1))

    async def execute():
        if phase == "execution":
            await asyncio.to_thread(change_target)
        context.result = await context.function.invoke(context=context, skip_parsing=True)

    middleware = ApprovalMiddleware(policy, bus, reuse=binding)
    await bus.subscribe(ApprovalRequest, respond)
    try:
        await middleware.process(context, execute)
        assert (first / "result.txt").is_symlink()
        assert (second / "result.txt").is_symlink()
        assert (first / "referent.txt").read_text(encoding="utf-8") == "original"
        assert (second / "referent.txt").read_text(encoding="utf-8") == "original"
        assert len(requests) == (2 if phase == "approval" else 1)
        assert all(not request.reuse_offer for request in requests)
        assert str(context.result).startswith("Error:")
    finally:
        await middleware.close()
        await bus.unsubscribe(ApprovalRequest, respond)


class _StringChoice(StrEnum):
    SAFE = "safe"
    CHANGED = "changed"


class _IntegerChoice(IntEnum):
    SAFE = 1
    CHANGED = 2


class _PlainChoice(Enum):
    SAFE = "safe"
    CHANGED = "changed"


@pytest.mark.parametrize("enum_type", [_StringChoice, _IntegerChoice, _PlainChoice])
async def test_enum_arguments_keep_ordinary_approval(setup, enum_type):
    _, binding, bus, policy = setup
    requests, executed = [], []

    async def consume(value) -> str:
        executed.append(value)
        return "enum tool completed"

    tool = FunctionTool(name="enum_tool", func=consume, input_model=create_model("EnumTool", value=(enum_type, ...)))

    async def respond(event):
        requests.append(event)
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=len(requests) == 1))

    middleware = ApprovalMiddleware(policy, bus, reuse=binding)
    await bus.subscribe(ApprovalRequest, respond)
    try:
        layer, _ = _stack(
            [_call_response(("call-enum", "enum_tool", {"value": enum_type.SAFE.value})), _text_response()],
            middleware=middleware,
        )
        response = await layer.get_response([_user()], options={"tools": [tool]})
        assert [request.args["value"] for request in requests] == [enum_type.SAFE]
        assert isinstance(requests[0].args["value"], enum_type) and requests[0].reuse_offer is None
        assert executed == [enum_type.SAFE]
        assert _result_contents(response)[0].result == "enum tool completed"
        assert not binding.service.rules()
    finally:
        await middleware.close()
        await bus.unsubscribe(ApprovalRequest, respond)


@pytest.mark.parametrize("link_kind", ["dangling", "looping"])
async def test_unchanged_final_symlink_can_still_be_replaced_once(setup, tmp_path, link_kind):
    tools, binding, bus, policy = setup
    link = tmp_path / "result.txt"
    symlink_or_skip(link, link if link_kind == "looping" else tmp_path / "missing.txt")
    context = file_context(tools, "write_file")
    context.arguments["path"] = str(link)
    requests = []

    async def respond(event):
        requests.append(event)
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True, remember_choice="EXACT_SESSION"))

    async def execute():
        context.result = await context.function.invoke(context=context, skip_parsing=True)

    middleware = ApprovalMiddleware(policy, bus, reuse=binding)
    await bus.subscribe(ApprovalRequest, respond)
    try:
        await middleware.process(context, execute)
        assert len(requests) == 1 and not requests[0].reuse_offer
        assert not binding.service.rules()
        assert not link.is_symlink()
        assert link.read_text(encoding="utf-8") == "changed"
        assert not (tmp_path / "missing.txt").exists()
    finally:
        await middleware.close()
        await bus.unsubscribe(ApprovalRequest, respond)


async def test_unresolved_one_time_target_does_not_disable_worker_guard(setup, destinations, monkeypatch):
    tools, binding, bus, policy = setup
    first, _, _ = destinations
    context = file_context(tools, "write_file")

    def unavailable(path, *, base_cwd=None):
        raise OSError("Cannot resolve the target during approval")

    # Only the approval adapter fails; the real worker can resolve the path.
    monkeypatch.setattr(filesystem, "file_write_target", unavailable)

    async def respond(event):
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True))

    async def execute():
        context.result = await context.function.invoke(context=context, skip_parsing=True)

    middleware = ApprovalMiddleware(policy, bus, reuse=binding)
    await bus.subscribe(ApprovalRequest, respond)
    try:
        await middleware.process(context, execute)
        assert (first / "result.txt").read_text(encoding="utf-8") == "original"
        assert str(context.result).startswith("Error:")
    finally:
        await middleware.close()
        await bus.unsubscribe(ApprovalRequest, respond)


@pytest.mark.parametrize("replacement", ["safe", _PlainChoice.SAFE])
def test_enum_snapshot_does_not_alias_a_string_or_another_enum(replacement):
    assert argument_snapshot(_StringChoice.SAFE) != argument_snapshot(replacement)
