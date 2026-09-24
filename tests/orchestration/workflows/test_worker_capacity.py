# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Request admission, executor isolation, per-pool exhaustion and byte-accurate join limits."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import replace
from pathlib import Path
from unittest.mock import create_autospec

import pytest

import chrys.orchestration.workflows.worker_client as worker_client
from chrys.orchestration.workflows.worker_client import (
    WorkerLostError,
    WorkerRpcError,
    WorkflowWorkerClient,
    _Pending,
)
from chrys.service.workflows.protocol import LIMITS, ErrorCode, Method, ref_to_wire
from chrys.service.workflows.scheduler import AttemptRef
from chrys.service.workflows.sdk import SourceValue, WorkflowValue
from chrys.service.workflows.values import canonical_json, source_to_wire
from tests.orchestration.workflows.conftest import FAKE_WORKER, Launcher
from tests.support.waiting import wait_for

pytestmark = pytest.mark.asyncio


def _ref(node: str, activation: str) -> AttemptRef:
    return AttemptRef("run", node, activation, 1)


def _evaluation_workflow(phase: str, *, offload: bool = False, stuck: bool = False) -> bytes:
    lines = [
        "import asyncio, os, pathlib, threading",
        "from chrys.workflows import WorkflowBuilder",
        "gate = threading.Event()",
        "def hold_sync(value, ctx):\n    ctx.emit('started')\n    gate.wait()\n    return 'held'",
        "async def hold_async(value, ctx):\n    return await asyncio.to_thread(hold_sync, value, ctx)",
        "def quick(text):\n    return text",
        "def evaluate(value):\n    pathlib.Path(threading.current_thread().name + '.entered').touch()\n    "
        + ("threading.Event().wait()" if stuck else "gate.set()")
        + "\n    return True",
        "def combine(sources):\n    evaluate(None)\n    return 'combined'",
        "wf = WorkflowBuilder('pools')",
        f"hold = wf.python('hold', {'hold_async' if offload else 'hold_sync'})",
        "done = wf.python('done', quick)",
        "wf.start(hold)",
    ]
    if phase == Method.EVAL_OUTGOING:
        lines.append("wf.edge(hold, done, when=evaluate)")
    elif phase == Method.COMBINE:
        lines.append("wf.join([hold], done, combine=combine)")
    else:
        lines += [
            "def body(scope):\n    end = scope.python('end', quick)\n    return end, end",
            "loop = wf.loop('loop', body=body, until=evaluate, max_iterations=1)",
            "wf.chain(hold, loop, done)",
        ]
    lines += ["wf.output(done)", "workflow = wf.build()"]
    return ("\n".join(lines) + "\n").encode()


async def _evaluate(client: WorkflowWorkerClient, phase: str, activation: str, *, timeout: float) -> None:
    if phase == Method.EVAL_OUTGOING:
        edge = "hold->done"
        assert (
            await client.eval_outgoing(_ref("hold", activation), WorkflowValue(text="x"), (edge,), timeout=timeout)
        ).value
    elif phase == Method.EVAL_LOOP_UNTIL:
        assert (
            await client.eval_loop_until(_ref("loop", activation), 1, WorkflowValue(text="x"), timeout=timeout)
        ).value
    else:
        sources = (SourceValue("hold", "held", WorkflowValue(text="x")),)
        assert (
            (await client.combine(_ref("join:done", activation), sources, timeout=timeout)).value
        ).text == "combined"


@pytest.mark.parametrize("phase", [Method.EVAL_OUTGOING, Method.EVAL_LOOP_UNTIL, Method.COMBINE])
@pytest.mark.parametrize("offload", [False, True], ids=["sync-body", "to-thread"])
async def test_evaluations_can_release_bodies_that_occupy_every_body_thread(
    launch: Launcher, interpreter: str, workspace: Path, phase: str, offload: bool
) -> None:
    started: set[str] = set()
    occupied = asyncio.Event()
    width = LIMITS.worker_thread_pool_size

    async def emitted(ref: AttemptRef, ordinal: int, text: str) -> None:
        started.add(ref.activation_id)
        if len(started) == width:
            occupied.set()

    client = await launch(interpreter=interpreter, emit_handler=emitted)
    await client.load(
        _evaluation_workflow(phase, offload=offload), filename=str(workspace / "wf.py"), workspace=workspace
    )
    bodies = [
        asyncio.create_task(client.run_python(_ref("hold", f"h{i}"), WorkflowValue(text=""), blocking=False))
        for i in range(width)
    ]
    try:
        await asyncio.wait_for(occupied.wait(), 5)
        # Only the evaluation releases the barrier. On the old shared pool it cannot start.
        await _evaluate(client, phase, "evaluation", timeout=2)
        results = await asyncio.wait_for(asyncio.gather(*bodies), 5)
        assert [result.value.text for result in results] == ["held"] * width
        assert client._leaked_threads_by_pool == {"body": 0, "eval": 0}
    finally:
        for body in bodies:
            body.cancel()
        await asyncio.gather(*bodies, return_exceptions=True)
        await client.close()


async def test_two_stuck_evaluations_exhaust_the_eval_pool_without_waiting_for_eight_leaks(
    launch: Launcher, workspace: Path
) -> None:
    client = await launch()
    phase = Method.EVAL_LOOP_UNTIL
    await client.load(_evaluation_workflow(phase, stuck=True), filename=str(workspace / "wf.py"), workspace=workspace)
    tasks = [asyncio.create_task(_evaluate(client, phase, f"e{i}", timeout=10)) for i in range(2)]

    try:
        await wait_for(
            lambda: len(list(workspace.glob("*.entered"))) == 2,
            description="both evaluation threads entered their barriers",
        )
        await client.cancel(_ref("loop", "e0"))
        assert client.lost is None
        assert client._leaked_threads_by_pool == {"body": 0, "eval": 1}
        assert client._capacity["eval"] == 1
        with pytest.raises(WorkerLostError, match="eval thread leak budget"):
            await client.cancel(_ref("loop", "e1"))
        assert client._leaked_threads_by_pool == {"body": 0, "eval": 2}
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(result, WorkerRpcError | WorkerLostError) for result in results)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()


@pytest.mark.parametrize("lane,method", [("body", Method.RUN_PYTHON), ("eval", Method.EVAL_OUTGOING)])
async def test_cancelled_waiters_keep_capacity_until_the_late_reply_and_duplicate_replies_do_not_release_twice(
    launch: Launcher, monkeypatch: pytest.MonkeyPatch, lane: str, method: str
) -> None:
    client = await launch(host_path=FAKE_WORKER)
    client._capacity[lane] = 1
    if lane == "eval":
        monkeypatch.setattr(worker_client, "LIMITS", replace(LIMITS, eval_thread_pool_size=1))
    queued = asyncio.Event()
    original_wait = client._capacity_changed.wait

    async def waiting() -> None:
        queued.set()
        await original_wait()

    monkeypatch.setattr(client._capacity_changed, "wait", create_autospec(original_wait, side_effect=waiting))
    timed: set[str] = set()
    original_deadline = client._deadline

    async def deadline(request: Awaitable[dict], pending: _Pending, ref: AttemptRef, timeout: float | None) -> dict:
        timed.add(ref.activation_id)
        return await original_deadline(request, pending, ref, timeout)

    monkeypatch.setattr(client, "_deadline", create_autospec(original_deadline, side_effect=deadline))

    async def request(node: str, activation: str, *, timeout: float | None = None) -> dict:
        ref = _ref(node, activation)
        return await client._request(
            method, {"ref": ref_to_wire(ref), "value": {"text": "", "data": None}}, key=None, ref=ref, timeout=timeout
        )

    first = asyncio.create_task(request("hang", "first"))
    tasks = [first]
    try:
        await wait_for(
            lambda: any(pending.sent for pending in client._pending.values()), description="first request sent"
        )
        first_id = next(iter(client._pending))
        second = asyncio.create_task(request("ok", "second", timeout=1))
        tasks.append(second)
        await asyncio.wait_for(queued.wait(), 5)
        assert timed == {"first"}  # neither lane starts a deadline while waiting for its slot
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        assert client._in_flight[lane] == 1
        assert first_id in client._pending
        assert not second.done()
        # The control request bypasses both limits; its terminal reply returns the original slot.
        await client.cancel(_ref("hang", "first"))
        await asyncio.wait_for(second, 5)
        assert timed == {"first", "second"}
        assert client._in_flight[lane] == 0

        third = asyncio.create_task(request("hang", "third"))
        tasks.append(third)
        await wait_for(lambda: client._in_flight[lane] == 1, description="third request owns the slot")
        client._on_frame({"id": first_id, "result": {}})  # replay of the already-consumed terminal
        assert client._in_flight[lane] == 1
        queued.clear()
        fourth = asyncio.create_task(request("ok", "fourth"))
        tasks.append(fourth)
        await asyncio.wait_for(queued.wait(), 5)
        assert not fourth.done()
        await client.cancel(_ref("hang", "third"))
        await asyncio.wait_for(fourth, 5)
        await third
        assert client._in_flight == {"body": 0, "eval": 0, "sync": 0}
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()


async def test_worker_loss_returns_all_slots_and_wakes_unregistered_waiters(
    launch: Launcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = await launch(host_path=FAKE_WORKER)
    client._capacity["body"] = 1
    queued = asyncio.Event()
    original_wait = client._capacity_changed.wait

    async def waiting() -> None:
        queued.set()
        await original_wait()

    monkeypatch.setattr(client._capacity_changed, "wait", create_autospec(original_wait, side_effect=waiting))
    first = asyncio.create_task(client.run_python(_ref("hang", "first"), WorkflowValue(text=""), blocking=False))
    tasks = [first]
    try:
        await wait_for(lambda: client._in_flight["body"] == 1, description="first slot registered")
        second = asyncio.create_task(client.run_python(_ref("ok", "second"), WorkflowValue(text=""), blocking=False))
        tasks.append(second)
        await asyncio.wait_for(queued.wait(), 5)
        client._mark_lost("injected worker loss")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(result, WorkerLostError) for result in results)
        assert client._pending == {}
        assert client._in_flight == {"body": 0, "eval": 0, "sync": 0}
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()


async def test_cancelling_before_write_returns_the_registered_slot(
    launch: Launcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = await launch(host_path=FAKE_WORKER)
    registered = asyncio.Event()
    original = client._exchange

    async def before_write(data: bytes, pending: _Pending) -> dict:
        registered.set()
        await asyncio.Event().wait()
        return await original(data, pending)

    with monkeypatch.context() as patch:
        patch.setattr(client, "_exchange", create_autospec(original, side_effect=before_write))
        task = asyncio.create_task(client.run_python(_ref("ok", "unsent"), WorkflowValue(text=""), blocking=False))
        try:
            await asyncio.wait_for(registered.wait(), 5)
            assert client._in_flight["body"] == 1
            assert all(not pending.sent for pending in client._pending.values())
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert client._pending == {}
    assert client._in_flight == {"body": 0, "eval": 0, "sync": 0}
    assert (await client.run_python(_ref("ok", "sent"), WorkflowValue(text=""), blocking=False)).value.text == "ok"


@pytest.mark.parametrize(
    "verdict",
    [
        {"leaked_threads_by_pool": {"body": 0, "eval": -1}},
        {"leaked_threads_by_pool": "invalid"},
    ],
)
async def test_invalid_leak_verdict_settles_the_popped_request_too(launch: Launcher, verdict: dict) -> None:
    client = await launch(host_path=FAKE_WORKER)
    task = asyncio.create_task(client.run_python(_ref("hang", "invalid"), WorkflowValue(text=""), blocking=False))
    try:
        await wait_for(lambda: client._in_flight["body"] == 1, description="pending request registered")
        request_id = next(iter(client._pending))
        client._on_frame({"id": request_id, "result": verdict})
        with pytest.raises(WorkerLostError):
            await asyncio.wait_for(task, 5)
        assert client._pending == {}
        assert client._in_flight == {"body": 0, "eval": 0, "sync": 0}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await client.close()


@pytest.mark.parametrize("character", ["汉", "😀"], ids=["cjk", "emoji"])
async def test_join_limit_counts_encoded_array_bytes_and_reports_join_before_frame_overflow(
    launch: Launcher, workspace: Path, character: str
) -> None:
    client = await launch()
    await client.load(_evaluation_workflow(Method.COMBINE), filename=str(workspace / "wf.py"), workspace=workspace)
    empty = SourceValue("hold", "source", WorkflowValue(text=""))
    overhead = len(canonical_json([source_to_wire(empty)]).encode("utf-8", "surrogatepass"))
    remaining = LIMITS.max_join_retained_bytes - overhead
    width = len(character.encode())
    content = character * (remaining // width) + "x" * (remaining % width)
    sources = (SourceValue("hold", "source", WorkflowValue(text=content)),)
    assert ((await client.combine(_ref("join:done", "fits"), sources)).value).text == "combined"
    oversized = (SourceValue("hold", "source", WorkflowValue(text=content + character)),)
    with pytest.raises(WorkerRpcError, match="join sources exceed max_join_retained_bytes") as failure:
        await client.combine(_ref("join:done", "too-big"), oversized)
    assert failure.value.code == ErrorCode.VALUE_TOO_LARGE
    assert client.lost is None
    assert client._in_flight == {"body": 0, "eval": 0, "sync": 0}
