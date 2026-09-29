# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Evaluation semantics on the real host: switch order, loop iterations, awaitables, fences, and bad text."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from pathlib import Path

import psutil
import pytest

from chrys.foundation.models.ask_user import AskUserAnswer, AskUserQuestion
from chrys.orchestration.workflows.worker_client import (
    AskUnavailable,
    AttemptTimeout,
    WorkerLostError,
    WorkerRpcError,
)
from chrys.service.workflows.protocol import LIMITS, ErrorCode
from chrys.service.workflows.scheduler import AttemptRef
from chrys.service.workflows.sdk import SourceValue, WorkflowValue
from tests.orchestration.workflows.conftest import Launcher
from tests.support.waiting import wait_for
from tests.support.workflow_workers import python_workflow

SWITCH = b"""
from chrys.workflows import WorkflowBuilder
def same(text):
    return text
def raises(value):
    raise RuntimeError("a later case must not be evaluated")
wf = WorkflowBuilder("switch")
src, a, b, c = (wf.python(name, same) for name in ("src", "a", "b", "c"))
wf.switch(src, cases=[(lambda value: value.text == "first", a), (raises, b)], default=c)
wf.start(src)
wf.output(a)
wf.output(b)
wf.output(c)
workflow = wf.build()
"""
LOOP = b"""
from chrys.workflows import WorkflowBuilder
def bang(text):
    return text + "!"
def body(scope):
    entry = scope.python("e", bang)
    exit_ = scope.python("x", bang)
    scope.edge(entry, exit_)
    return entry, exit_
wf = WorkflowBuilder("loop")
loop = wf.loop("L", body=body, until=lambda value: value.text.endswith("!!"), max_iterations=3)
wf.start(loop)
wf.output(loop)
workflow = wf.build()
"""
AWAITABLE = python_workflow(
    "import asyncio\nasync def inner(text):\n    await asyncio.sleep(0)\n    return text.text + '!'\n"
    "def fn(text):\n    return inner(text)\n",
    "fn",
)
HANG = python_workflow("import asyncio\nasync def fn(value, ctx):\n    await asyncio.sleep(3600)\n", "fn")
STUCK = python_workflow("import time\nasync def fn(value, ctx):\n    time.sleep(3600)\n", "fn")
LATE_AWAITABLE = python_workflow(
    "import asyncio, os, time\n"
    "async def inner():\n    os.write(1, b'LATE\\n')\n    await asyncio.sleep(3600)\n"
    "def fn(text):\n    time.sleep(0.2)\n    os.write(1, b'THREAD_DONE\\n')\n    return inner()\n",
    "fn",
)
HUGE_INT = python_workflow(
    "def fn(value, ctx):\n    ctx.emit('one')\n    return WorkflowValue(text='ok', data=10 ** 5000)\n",
    "fn",
)
WIDE_EMIT = python_workflow(
    "def fn(value, ctx):\n    ctx.emit(chr(0) * (3 * 1024 * 1024))\n    return 'x'\n"
    "def quick(text):\n    return text\n",
    "fn",
    "quick",
)
HUGE = python_workflow(
    "def fn(value, ctx):\n    ctx.emit('one')\n"
    "    return WorkflowValue(text='ok', data=['x' * (4 * 1024 * 1024)] * 5)\n",
    "fn",
)
CLEANUP = python_workflow(
    "import asyncio, os\n"
    "async def fn(value, ctx):\n    os.write(1, b'STARTED\\n')\n    try:\n        await asyncio.sleep(3600)\n"
    "    finally:\n        await asyncio.sleep(float(value.text))\n        os.write(1, b'CLEANED\\n')\n",
    "fn",
)
STUBBORN = python_workflow(
    "import asyncio, os\n"
    "async def fn(value, ctx):\n    os.write(1, b'STARTED\\n')\n"
    "    while True:\n        try:\n            await asyncio.sleep(3600)\n"
    "        except asyncio.CancelledError:\n            pass\n"
    "def quick(text):\n    return text\n",
    "fn",
    "quick",
)
OFFLOAD = python_workflow(
    "import asyncio, os, threading\n"
    "started = threading.Event()\n"
    "def stuck():\n    started.set()\n    os.write(1, b'STARTED\\n')\n    threading.Event().wait(3600)\n"
    "async def fn(value, ctx):\n    await asyncio.to_thread(stuck)\n"
    "async def pair(value, ctx):\n    await asyncio.gather(asyncio.to_thread(stuck), asyncio.to_thread(stuck))\n"
    "async def boom():\n    while not started.is_set():\n        await asyncio.sleep(0.01)\n    raise RuntimeError('boom')\n"
    "async def fail(value, ctx):\n    await asyncio.gather(asyncio.to_thread(stuck), boom())\n"
    "async def leave(value, ctx):\n    asyncio.get_running_loop().run_in_executor(None, stuck)\n"
    "    while not started.is_set():\n        await asyncio.sleep(0.01)\n    return 'left'\n"
    "def quick(text):\n    return text\n",
    "fn",
    "pair",
    "fail",
    "leave",
    "quick",
)
CLEANUP_OFFLOAD = python_workflow(
    "import asyncio, os, threading\n"
    "def cleanup():\n    os.write(1, b'CLEANING\\n')\n    threading.Event().wait(3600)\n"
    "async def fn(value, ctx):\n    os.write(1, b'STARTED\\n')\n"
    "    try:\n        await asyncio.sleep(3600)\n"
    "    finally:\n        try:\n            await asyncio.wait_for(asyncio.to_thread(cleanup), 0.3)\n"
    "        except asyncio.TimeoutError:\n            pass\n",
    "fn",
)
PHASES = (
    b"import asyncio, os, threading\n"
    b"from chrys.workflows import WorkflowBuilder\n"
    b"started = {}\n"
    b"def stuck(name):\n    started[name] = True\n    os.write(1, b'STARTED\\n')\n    threading.Event().wait(3600)\n"
    b"async def body(value, ctx):\n"
    b"    asyncio.get_running_loop().run_in_executor(None, stuck, 'body')\n"
    b"    while 'body' not in started:\n        await asyncio.sleep(0.001)\n"
    b"    return 'left'\n"
    b"def predicate(value):\n    stuck('predicate')\n    return True\n"
    b"wf = WorkflowBuilder('phases')\n"
    b"a = wf.python('a', body)\nb = wf.python('b', body)\n"
    b"wf.edge(a, b, when=predicate)\nwf.start(a)\nwf.output(b)\nworkflow = wf.build()\n"
)
LEAKY_CONDITIONS = (
    b"import asyncio, threading\n"
    b"from chrys.workflows import WorkflowBuilder\n"
    b"def stuck(ready):\n    ready.set()\n    threading.Event().wait()\n"
    b"def leave():\n    ready = threading.Event()\n"
    b"    asyncio.get_event_loop().run_in_executor(None, stuck, ready)\n    ready.wait()\n"
    b"def predicate(value):\n    leave()\n    return True\n"
    b"def until(value):\n    leave()\n    return True\n"
    b"def combine(sources):\n    leave()\n    return 'combined'\n"
    b"def same(text):\n    return text\n"
    b"wf = WorkflowBuilder('leaky-conditions')\n"
    b"a, b, c = wf.python('a', same), wf.python('b', same), wf.python('c', same)\n"
    b"loop = wf.loop('L', body=lambda scope: (scope.python('e', same),) * 2, until=until, max_iterations=2)\n"
    b"wf.edge(a, b, when=predicate)\nwf.join([b], c, combine=combine)\nwf.edge(c, loop)\n"
    b"wf.start(a)\nwf.output(loop)\nworkflow = wf.build()\n"
)
ORPHANS = python_workflow(
    "import asyncio, os, threading\n"
    "def stuck():\n    os.write(1, b'STUCK\\n')\n    threading.Event().wait(3600)\n"
    "async def sibling():\n    await asyncio.sleep(0.05)\n    await asyncio.to_thread(stuck)\n"
    "async def boom():\n    raise ValueError('boom')\n"
    "async def fail(value, ctx):\n    await asyncio.gather(boom(), *(sibling() for _ in range(8)))\n"
    "async def leave(value, ctx):\n    for _ in range(8):\n        asyncio.create_task(sibling())\n    return 'left'\n"
    "def quick(text):\n    return text\n",
    "fail",
    "leave",
    "quick",
)
ORPHAN_CLEANUP = python_workflow(
    "import asyncio, os, threading\n"
    "started = []\n"
    "def cleanup():\n    os.write(1, b'CLEANING\\n')\n    threading.Event().wait(3600)\n"
    "async def child():\n    try:\n        started.append(1)\n        await asyncio.sleep(3600)\n"
    "    finally:\n        try:\n            await asyncio.wait_for(asyncio.to_thread(cleanup), 0.3)\n"
    "        except asyncio.TimeoutError:\n            pass\n"
    "async def spawn():\n    asyncio.create_task(child())\n    while not started:\n        await asyncio.sleep(0.001)\n"
    "async def fail(value, ctx):\n    await spawn()\n    raise ValueError('boom')\n"
    "async def leave(value, ctx):\n    await spawn()\n    return 'left'\n"
    "async def hang(value, ctx):\n    await spawn()\n    await asyncio.sleep(3600)\n"
    "def quick(text):\n    return text\n",
    "fail",
    "leave",
    "hang",
    "quick",
)
UNWINDING_CHILD = python_workflow(
    "import asyncio, os\n"
    "started = []\n"
    "async def child():\n    try:\n        started.append(1)\n        await asyncio.sleep(3600)\n"
    "    finally:\n        os.write(1, b'STARTED\\n')\n        await asyncio.sleep(0.5)\n        os.write(1, b'CLEANED\\n')\n"
    "async def fn(value, ctx):\n    asyncio.create_task(child())\n    while not started:\n        await asyncio.sleep(0.001)\n"
    "    return 'left'\n",
    "fn",
)
CLEANUP_SPAWNS = python_workflow(
    "import asyncio, os, threading\n"
    "started = []\n"
    "def stuck():\n    os.write(1, b'STUCK\\n')\n    threading.Event().wait(3600)\n"
    "async def cleanup():\n    await asyncio.sleep(0.05)\n    await asyncio.to_thread(stuck)\n"
    "async def child():\n    try:\n        started.append(1)\n        await asyncio.sleep(3600)\n"
    "    finally:\n        asyncio.create_task(cleanup())\n"
    "async def fn(value, ctx):\n    for _ in range(8):\n        asyncio.create_task(child())\n"
    "    while len(started) < 8:\n        await asyncio.sleep(0.001)\n    return 'left'\n"
    "def quick(text):\n    return text\n",
    "fn",
    "quick",
)
ASYNCGEN_CLEANUP = python_workflow(
    "import asyncio, os\n"
    "async def rows():\n    fd = os.open(os.devnull, os.O_RDONLY)\n    try:\n        yield fd\n"
    "    finally:\n        await asyncio.sleep(0)\n        os.close(fd)\n"
    "async def consume(value, ctx):\n    async for fd in rows():\n        break\n    await asyncio.sleep(0)\n"
    "    return str(fd)\n"
    "def check(text):\n    try:\n        os.fstat(int(text.text))\n    except OSError:\n        return 'closed'\n"
    "    return 'open'\n",
    "consume",
    "check",
)
PROCESS_POOL = python_workflow(
    "import concurrent.futures, multiprocessing\n"
    "def square(x):\n    return x * x\n"
    "def fn(text):\n"
    "    context = multiprocessing.get_context('spawn')\n"
    "    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:\n"
    "        return str(list(pool.map(square, [2, 3])))\n",
    "fn",
)
LOCAL_ANNOTATION = python_workflow(
    "def fn(text):\n"
    "    class Local:\n        pass\n"
    "    def g(value: Local):\n        pass\n"
    "    return str(g.__annotations__['value'] is Local)\n",
    "fn",
)
HUGE_ERROR = python_workflow("def fn(text):\n    raise ValueError('x' * (9 * 1024 * 1024))\n", "fn")
STDIN_READER = python_workflow("import sys\ndef fn(text):\n    return repr(sys.stdin.readline())\n", "fn")
GATHER_EXIT = python_workflow(
    "import asyncio\n"
    "async def inner():\n    raise SystemExit(5)\n"
    "async def fn(value, ctx):\n    await asyncio.gather(inner())\n"
    "def quick(text):\n    return text.text + '!'\n",
    "fn",
    "quick",
)
ASK_ONCE = python_workflow("async def fn(value, ctx):\n    return await ctx.ask('?')\n", "fn")
CANCEL_UNSTARTED = python_workflow(
    "import asyncio, inspect\n"
    "async def inner():\n    return 1\n"
    "async def fn(value, ctx):\n"
    "    coro = inner()\n    asyncio.create_task(coro).cancel()\n    await asyncio.sleep(0)\n"
    "    return inspect.getcoroutinestate(coro)\n",
    "fn",
)
EXITS = python_workflow(
    "import asyncio\n"
    "async def leave(value, ctx):\n    await asyncio.sleep(0)\n    raise SystemExit(3)\n"
    "def wrapped(text):\n    return leave(None, None)\n"
    "def quick(text):\n    return text.text + '!'\n",
    "leave",
    "wrapped",
    "quick",
)
ORPHAN = python_workflow(
    "import subprocess, sys\n"
    "def fn(text):\n"
    "    child = subprocess.Popen(\n"
    "        [sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print(1, flush=True); time.sleep(3600)'],\n"
    "        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,\n"
    "    )\n"
    "    assert child.stdout.readline() == b'1\\n'  # the handler is installed before the body hands the pid back\n"
    "    return str(child.pid)\n",
    "fn",
)
MODULE_PRIMITIVES = python_workflow(
    "import asyncio\n"
    "gate = asyncio.Semaphore(1)\n"
    "queue = asyncio.Queue()\n"
    "async def fn(value, ctx):\n"
    "    async with gate:\n        await queue.put('ok')\n"
    "    return await queue.get()\n",
    "fn",
)
FORK_THEN_DIE = python_workflow(
    "import multiprocessing, os, time\n"
    "def linger():\n    time.sleep(3600)\n"
    "def fn(text):\n"
    "    multiprocessing.get_context('fork').Process(target=linger).start()\n"
    "    os._exit(7)\n",
    "fn",
)
WIDE_ASK = python_workflow("async def fn(value, ctx):\n    return await ctx.ask('x' * (17 * 1024 * 1024))\n", "fn")
SYNC_ASK = python_workflow(
    "import asyncio\n"
    "from chrys.workflows import Question\n"
    "def fn(value, ctx):\n    return asyncio.run(ctx.ask(Question('?', options=['a']))).text\n",
    "fn",
)
ASYNC_CONDITIONS = b"""
from chrys.workflows import WorkflowBuilder
async def verdict(value):
    return False
def same(text):
    return text
wf = WorkflowBuilder("async-conditions")
src, dst = wf.python("src", same), wf.python("dst", same)
loop = wf.loop("L", body=lambda scope: (scope.python("e", same),) * 2, until=lambda value: verdict(value), max_iterations=2)
wf.edge(src, dst, when=lambda value: verdict(value))
wf.edge(dst, loop)
wf.start(src)
wf.output(loop)
workflow = wf.build()
"""
QUEUED = python_workflow(
    "import os, threading, time\n"
    "def sleeper(text):\n    os.write(1, b'S\\n')\n"
    "    open(text.text + f'.{threading.get_ident()}', 'w').close()\n"
    "    for _ in range(1000):\n        if os.path.exists(text.text):\n            break\n        time.sleep(0.01)\n"
    "    return text\n"
    "def marker(text):\n    open(text.text, 'w').close()\n    return text\n",
    "sleeper",
    "marker",
)
SURROGATES = python_workflow(
    "def emit_fn(value, ctx):\n    ctx.emit('\\ud800')\n    return 'x'\n"
    "def text_fn(text):\n    return '\\udcff'\n"
    "def print_fn(text):\n    print('\\udcff')\n    return text\n"
    "async def ask_fn(value, ctx):\n    await ctx.ask('\\ud800')\n    return 'x'\n",
    "emit_fn",
    "text_fn",
    "print_fn",
    "ask_fn",
)


def ref(node: str, *, attempt: int = 1) -> AttemptRef:
    return AttemptRef(run_id="run", node_id=node, activation_id=f"{node}@iter#1", attempt=attempt)


async def _loaded(launch: Launcher, workspace: Path, source: bytes, **kwargs: object) -> object:
    client = await launch(**kwargs)
    await client.load(source, filename=str(workspace / "wf.py"), workspace=workspace)
    return client


async def _announced(client: object, marker: str, count: int = 1) -> None:
    """Wait until *marker* appeared *count* times on the native tail; its reader thread runs behind the writer."""
    for _ in range(250):
        if (await client.native_output()).text.count(marker) >= count:
            return
        await asyncio.sleep(0.02)
    pytest.fail(f"the worker never wrote {marker!r} to the native tail")


async def _started(client: object, count: int = 1) -> None:
    """Wait until a body (or *count* of its threads) announced itself on the native tail."""
    await _announced(client, "STARTED", count)


async def test_switch_cases_are_evaluated_in_order_and_stop_at_the_first_hit(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, SWITCH)
    edges = ("src->a", "src->b", "src->c")

    decisions = (await client.eval_outgoing(ref("src"), WorkflowValue(text="first"), edges)).value
    assert decisions == {"src->a": True, "src->b": False, "src->c": False}

    with pytest.raises(WorkerRpcError) as failure:
        await client.eval_outgoing(ref("src", attempt=2), WorkflowValue(text="none"), edges)
    assert failure.value.code == ErrorCode.USER_EXCEPTION
    assert "later case" in failure.value.message


async def test_loop_until_is_evaluated_once_per_iteration_on_one_ref(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, LOOP)
    loop_ref = AttemptRef(run_id="run", node_id="L", activation_id="L@iter#1", attempt=1)

    assert (await client.eval_loop_until(loop_ref, 1, WorkflowValue(text="go!"))).value is False
    assert (await client.eval_loop_until(loop_ref, 2, WorkflowValue(text="go!!"))).value is True
    with pytest.raises(WorkerRpcError, match="already started"):
        await client.eval_loop_until(loop_ref, 2, WorkflowValue(text="go!!"))


async def test_sync_body_returning_an_awaitable_is_awaited(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, AWAITABLE)
    result = await client.run_python(ref("fn"), WorkflowValue(text="x"), blocking=False)
    assert result.value.text == "x!"


async def test_late_awaitable_from_a_cancelled_sync_body_never_starts(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, LATE_AWAITABLE)
    with pytest.raises(AttemptTimeout):
        await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False, timeout=0.05)
    # The thread announces its return on the native tail; a coroutine that was started writes at once.
    for _ in range(250):
        if "THREAD_DONE" in (await client.native_output()).text:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.1)
    text = (await client.native_output()).text
    assert "THREAD_DONE" in text
    assert "LATE" not in text


async def test_cancelled_attempt_rejects_every_later_phase(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, HANG)
    with pytest.raises(AttemptTimeout):
        await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False, timeout=0.2)

    with pytest.raises(WorkerRpcError) as failure:
        await client.eval_outgoing(ref("fn"), WorkflowValue(text=""), ())
    assert failure.value.code == ErrorCode.ATTEMPT_TERMINATED

    assert ((await client.eval_outgoing(ref("fn", attempt=2), WorkflowValue(text=""), ())).value) == {}


async def test_oversized_result_fails_through_the_projection_barrier(launch: Launcher, workspace: Path) -> None:
    seen: list[int] = []

    async def slow(attempt: AttemptRef, ordinal: int, line: str) -> None:
        await asyncio.sleep(0.05)
        seen.append(ordinal)

    client = await _loaded(launch, workspace, HUGE, emit_handler=slow)
    with pytest.raises(WorkerRpcError) as failure:
        await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False)
    assert failure.value.code == ErrorCode.VALUE_TOO_LARGE
    assert seen == [1]


async def test_unframeable_result_fails_through_the_projection_barrier(launch: Launcher, workspace: Path) -> None:
    seen: list[int] = []

    async def slow(attempt: AttemptRef, ordinal: int, line: str) -> None:
        await asyncio.sleep(0.05)
        seen.append(ordinal)

    client = await _loaded(launch, workspace, HUGE_INT, emit_handler=slow)
    with pytest.raises(WorkerRpcError) as failure:
        await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False)
    assert failure.value.code == ErrorCode.VALUE_NOT_SERIALIZABLE
    assert seen == [1]


async def test_emit_whose_frame_outgrows_the_cap_fails_only_its_attempt(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, WIDE_EMIT)
    with pytest.raises(WorkerRpcError) as failure:
        await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False)
    assert failure.value.code == ErrorCode.PROTOCOL_LIMIT
    assert client.lost is None
    assert (await client.run_python(ref("quick"), WorkflowValue(text="alive"), blocking=False)).value.text == "alive"


async def test_lone_surrogates_fail_the_attempt_instead_of_hanging_it(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, SURROGATES)

    with pytest.raises(WorkerRpcError) as emit_failure:
        await client.run_python(ref("emit_fn"), WorkflowValue(text=""), blocking=False, timeout=5.0)
    assert emit_failure.value.code == ErrorCode.USER_EXCEPTION
    assert "not valid Unicode" in emit_failure.value.message

    with pytest.raises(WorkerRpcError) as text_failure:
        await client.run_python(ref("text_fn"), WorkflowValue(text=""), blocking=False, timeout=5.0)
    assert text_failure.value.code == ErrorCode.VALUE_NOT_SERIALIZABLE

    with pytest.raises(WorkerRpcError) as ask_failure:
        await client.run_python(ref("ask_fn"), WorkflowValue(text=""), blocking=False, timeout=5.0)
    assert ask_failure.value.code == ErrorCode.USER_EXCEPTION
    assert "Question.question is not valid Unicode" in ask_failure.value.message

    printed = await client.run_python(ref("print_fn"), WorkflowValue(text="kept"), blocking=False, timeout=5.0)
    assert printed.value.text == "kept"
    assert printed.stdout.text == "?\n"  # diagnostics are replaced, never a reason to fail the attempt


async def test_worker_whose_loop_is_blocked_is_lost_when_cancel_goes_unanswered(
    launch: Launcher, workspace: Path
) -> None:
    client = await _loaded(launch, workspace, STUCK)
    with pytest.raises(WorkerLostError, match="not acknowledged"):
        await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False, timeout=0.1)
    assert client.lost is not None


async def test_cancelled_body_still_queued_behind_the_pool_never_starts(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, QUEUED)
    pool = LIMITS.worker_thread_pool_size
    release = workspace / "release"
    marker = workspace / "ran.marker"
    sleepers = [
        asyncio.create_task(
            client.run_python(
                AttemptRef(run_id="run", node_id="sleeper", activation_id=f"sleeper@iter#{i}", attempt=1),
                WorkflowValue(text=str(release)),
                blocking=False,
            )
        )
        for i in range(1, pool + 1)
    ]
    try:
        await wait_for(lambda: len(list(workspace.glob("release.*"))) == pool, description="body pool occupied")
        assert (await client.native_output()).text.count("S\n") == pool

        with pytest.raises(AttemptTimeout) as timeout:  # raised once the worker acknowledged the cancel
            await client.run_python(ref("marker"), WorkflowValue(text=str(marker)), blocking=False, timeout=0.1)
        assert timeout.value.leaked_thread is False  # withdrawn from the queue: no thread was ever taken
        assert sum(client._leaked_threads_by_pool.values()) == 0
    finally:
        release.touch()
        assert len(await asyncio.gather(*sleepers)) == pool
    assert not marker.exists()


async def test_cancel_is_acknowledged_only_after_the_async_body_unwound(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, CLEANUP)
    called_at = time.monotonic()
    with pytest.raises(AttemptTimeout) as timeout:
        await client.run_python(ref("fn"), WorkflowValue(text="0.2"), blocking=False, timeout=0.05)
    assert time.monotonic() - called_at >= 0.2  # the ack waited for the finally block's sleep
    assert timeout.value.leaked_thread is False
    await _announced(client, "CLEANED")  # and the finally block ran to its end


async def test_direct_cancel_answers_the_body_only_after_it_unwound(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, CLEANUP)
    body = asyncio.create_task(client.run_python(ref("fn"), WorkflowValue(text="0.2"), blocking=False))
    await _started(client)
    cancelled_at = time.monotonic()
    cancel = asyncio.create_task(client.cancel(ref("fn")))

    with pytest.raises(WorkerRpcError) as failure:  # the body's own envelope, not the ack
        await body
    assert failure.value.code == ErrorCode.ATTEMPT_TERMINATED
    assert time.monotonic() - cancelled_at >= 0.2  # that envelope waited for the finally block's sleep
    await _announced(client, "CLEANED")  # and the finally block ran to its end
    outcome = await cancel
    assert outcome.cancelled is True
    assert sum(outcome.leaked_threads_by_pool.values()) == 0


async def test_overlapping_cancel_and_timeout_share_one_drain(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, CLEANUP)
    body = asyncio.create_task(client.run_python(ref("fn"), WorkflowValue(text="0.6"), blocking=False, timeout=0.3))
    await _started(client)
    cancelled_at = time.monotonic()
    cancel = asyncio.create_task(client.cancel(ref("fn")))  # drains for 0.6 s; the timeout cancels meanwhile

    with pytest.raises(AttemptTimeout) as timeout:
        await body
    assert time.monotonic() - cancelled_at >= 0.6  # the second cancel waited for the same drain
    await _announced(client, "CLEANED")
    assert timeout.value.leaked_thread is False
    outcome = await cancel
    assert outcome.cancelled is True
    assert sum(outcome.leaked_threads_by_pool.values()) == 0
    assert sum(client._leaked_threads_by_pool.values()) == 0


async def test_overlapping_cancels_charge_the_leak_budget_once(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, STUBBORN)
    body = asyncio.create_task(client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False, timeout=0.3))
    await _started(client)
    cancel = asyncio.create_task(client.cancel(ref("fn")))

    with pytest.raises(AttemptTimeout) as timeout:
        await body
    assert timeout.value.leaked_thread is True  # the shared verdict, not an early empty ack
    assert sum((await cancel).leaked_threads_by_pool.values()) == 1
    assert sum(client._leaked_threads_by_pool.values()) == 1
    assert client.lost is None


@pytest.mark.parametrize(("node", "threads"), [("fn", 1), ("pair", 2)])
async def test_async_body_offloads_stuck_on_threads_count_as_leaked(
    launch: Launcher, workspace: Path, interpreter: str, node: str, threads: int
) -> None:
    client = await _loaded(launch, workspace, OFFLOAD, interpreter=interpreter)
    body = asyncio.create_task(client.run_python(ref(node), WorkflowValue(text=""), blocking=False))
    await _started(client, threads)  # announced from the offloaded threads: running, not queued

    outcome = await client.cancel(ref(node))
    assert sum(outcome.leaked_threads_by_pool.values()) == threads  # every stuck offload is one thread of the budget
    assert sum(client._leaked_threads_by_pool.values()) == threads
    with pytest.raises(WorkerRpcError) as failure:
        await body
    assert failure.value.code == ErrorCode.ATTEMPT_TERMINATED
    assert (await client.run_python(ref("quick"), WorkflowValue(text="alive"), blocking=False)).value.text == "alive"


@pytest.mark.parametrize("node", ["fail", "leave"])
async def test_offloads_still_running_when_a_body_ends_are_charged(
    launch: Launcher, workspace: Path, interpreter: str, node: str
) -> None:
    """A body that fails or returns while its offload is on a thread has leaked it, cancel or no cancel."""
    client = await _loaded(launch, workspace, OFFLOAD, interpreter=interpreter)
    if node == "fail":
        with pytest.raises(WorkerRpcError) as failure:
            await client.run_python(ref(node), WorkflowValue(text=""), blocking=False)
        assert failure.value.code == ErrorCode.USER_EXCEPTION
    else:
        assert (await client.run_python(ref(node), WorkflowValue(text=""), blocking=False)).value.text == "left"
    assert sum(client._leaked_threads_by_pool.values()) == 1
    assert (
        sum((await client.cancel(ref(node))).leaked_threads_by_pool.values()) == 0
    )  # nothing live; the verdict was in the envelope
    assert sum(client._leaked_threads_by_pool.values()) == 1
    assert (await client.run_python(ref("quick"), WorkflowValue(text="alive"), blocking=False)).value.text == "alive"


async def test_an_offload_made_while_a_cancelled_body_unwinds_is_charged(
    launch: Launcher, workspace: Path, interpreter: str
) -> None:
    client = await _loaded(launch, workspace, CLEANUP_OFFLOAD, interpreter=interpreter)
    body = asyncio.create_task(client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False))
    await _started(client)

    outcome = await client.cancel(ref("fn"))
    assert sum(outcome.leaked_threads_by_pool.values()) == 1  # the cleanup thread, offloaded after the fence went up
    assert sum(client._leaked_threads_by_pool.values()) == 1
    with pytest.raises(WorkerRpcError) as failure:
        await body
    assert failure.value.code == ErrorCode.ATTEMPT_TERMINATED
    await _announced(client, "CLEANING")


async def test_each_phase_of_a_ref_settles_its_own_leak(launch: Launcher, workspace: Path, interpreter: str) -> None:
    """A body that leaked and then a predicate that leaks on the same ref are two threads of the budget."""
    client = await _loaded(launch, workspace, PHASES, interpreter=interpreter)
    assert (await client.run_python(ref("a"), WorkflowValue(text=""), blocking=False)).value.text == "left"
    assert sum(client._leaked_threads_by_pool.values()) == 1
    evaluation = asyncio.create_task(client.eval_outgoing(ref("a"), WorkflowValue(text=""), ("a->b",)))
    await _started(client, 2)

    assert (
        sum((await client.cancel(ref("a"))).leaked_threads_by_pool.values()) == 1
    )  # the predicate's thread; the body's was settled
    assert sum(client._leaked_threads_by_pool.values()) == 2
    with pytest.raises(WorkerRpcError) as failure:
        await evaluation
    assert failure.value.code == ErrorCode.ATTEMPT_TERMINATED


async def test_successful_evaluations_settle_their_own_leaks(
    launch: Launcher, workspace: Path, interpreter: str
) -> None:
    """A predicate, a loop condition and a combine that each left an offload on a thread are three threads."""
    client = await _loaded(launch, workspace, LEAKY_CONDITIONS, interpreter=interpreter)
    assert (await client.eval_outgoing(ref("a"), WorkflowValue(text=""), ("a->b",))).value == {"a->b": True}
    assert sum(client._leaked_threads_by_pool.values()) == 1
    sources = (SourceValue(node_id="b", activation_id="b@iter#1", value=WorkflowValue(text="")),)
    assert ((await client.combine(ref("join:c"), sources)).value).text == "combined"
    assert sum(client._leaked_threads_by_pool.values()) == 2
    assert (await client.eval_loop_until(ref("L"), 1, WorkflowValue(text=""))).value is True
    assert sum(client._leaked_threads_by_pool.values()) == 3
    assert client._leaked_threads_by_pool == {"body": 3, "eval": 0}
    assert client.lost is None


@pytest.mark.parametrize("node", ["fail", "leave"])
async def test_tasks_a_body_leaves_behind_end_with_it(
    launch: Launcher, workspace: Path, interpreter: str, node: str
) -> None:
    """A failed gather's siblings and a fire-and-forget task are cancelled with the body, before they can offload."""
    client = await _loaded(launch, workspace, ORPHANS, interpreter=interpreter)
    if node == "fail":
        with pytest.raises(WorkerRpcError) as failure:
            await client.run_python(ref(node), WorkflowValue(text=""), blocking=False)
        assert failure.value.code == ErrorCode.USER_EXCEPTION
    else:
        assert (await client.run_python(ref(node), WorkflowValue(text=""), blocking=False)).value.text == "left"
    await asyncio.sleep(0.2)  # had the eight survivors lived, they would hold every pool thread by now
    assert sum(client._leaked_threads_by_pool.values()) == 0
    assert (
        await client.run_python(ref("quick"), WorkflowValue(text="alive"), blocking=False, timeout=1)
    ).value.text == "alive"
    assert "STUCK" not in (await client.native_output()).text


@pytest.mark.parametrize("node", ["fail", "leave", "hang"])
async def test_a_cleanup_offload_of_a_task_the_body_left_behind_is_charged(
    launch: Launcher, workspace: Path, interpreter: str, node: str
) -> None:
    """The body's tasks are unwound before its verdict, so a child's finally-block offload counts like the body's."""
    client = await _loaded(launch, workspace, ORPHAN_CLEANUP, interpreter=interpreter)
    if node == "fail":
        with pytest.raises(WorkerRpcError) as failure:
            await client.run_python(ref(node), WorkflowValue(text=""), blocking=False)
        assert failure.value.code == ErrorCode.USER_EXCEPTION
    elif node == "leave":
        assert (await client.run_python(ref(node), WorkflowValue(text=""), blocking=False)).value.text == "left"
    else:
        with pytest.raises(AttemptTimeout) as timeout:
            await client.run_python(ref(node), WorkflowValue(text=""), blocking=False, timeout=0.1)
        assert timeout.value.leaked_thread is True
    assert (
        sum(client._leaked_threads_by_pool.values()) == 1
    )  # the cleanup thread; the child itself unwound within the drain
    await _announced(client, "CLEANING")
    assert (await client.run_python(ref("quick"), WorkflowValue(text="alive"), blocking=False)).value.text == "alive"


async def test_a_cancel_during_the_unwind_joins_it_instead_of_cancelling_the_child_again(
    launch: Launcher, workspace: Path, interpreter: str
) -> None:
    """A second CancelledError would cut the child's finally block short; the drain waits for the unwind under way."""
    client = await _loaded(launch, workspace, UNWINDING_CHILD, interpreter=interpreter)
    body = asyncio.create_task(client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False))
    await _started(client)  # the child's cleanup began: the body has returned and its unwind is under way

    outcome = await client.cancel(ref("fn"))
    assert outcome.cancelled is True
    assert sum(outcome.leaked_threads_by_pool.values()) == 0
    with pytest.raises(WorkerRpcError) as failure:
        await body
    assert failure.value.code == ErrorCode.ATTEMPT_TERMINATED
    await _announced(client, "CLEANED")  # the cleanup ran to its end
    assert sum(client._leaked_threads_by_pool.values()) == 0


async def test_tasks_started_while_the_body_unwinds_are_unwound_too(
    launch: Launcher, workspace: Path, interpreter: str
) -> None:
    """A finally block that starts a cleanup task hands it to the same drain: it cannot outlive the verdict either."""
    client = await _loaded(launch, workspace, CLEANUP_SPAWNS, interpreter=interpreter)
    assert (await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False)).value.text == "left"
    await asyncio.sleep(0.2)  # had the eight cleanup tasks lived, they would hold every pool thread by now
    assert sum(client._leaked_threads_by_pool.values()) == 0
    assert (
        await client.run_python(ref("quick"), WorkflowValue(text="alive"), blocking=False, timeout=1)
    ).value.text == "alive"
    assert "STUCK" not in (await client.native_output()).text


async def test_an_async_generators_cleanup_gets_the_allowance_instead_of_a_cancel(
    launch: Launcher, workspace: Path, interpreter: str
) -> None:
    """The loop finalizes a dropped async generator with an aclose() task; the unwind waits for it, as asyncio would."""
    client = await _loaded(launch, workspace, ASYNCGEN_CLEANUP, interpreter=interpreter)
    fd = (await client.run_python(ref("consume"), WorkflowValue(text=""), blocking=False)).value.text
    assert (await client.run_python(ref("check"), WorkflowValue(text=fd), blocking=False)).value.text == "closed"
    assert sum(client._leaked_threads_by_pool.values()) == 0


async def test_a_body_can_hand_its_module_functions_to_a_process_pool(
    launch: Launcher, workspace: Path, interpreter: str
) -> None:
    """A spawned child re-imports the workflow module by name, as it would re-import a script's __main__."""
    (workspace / "wf.py").write_bytes(PROCESS_POOL)
    client = await _loaded(launch, workspace, PROCESS_POOL, interpreter=interpreter)
    result = await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False, timeout=60)
    assert result.value.text == "[4, 9]"


async def test_a_workflow_file_gets_its_own_future_flags(launch: Launcher, workspace: Path, interpreter: str) -> None:
    """The host's own ``from __future__ import annotations`` must not leak into the user's file."""
    client = await _loaded(launch, workspace, LOCAL_ANNOTATION, interpreter=interpreter)
    assert (await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False)).value.text == "True"


async def test_async_body_that_keeps_swallowing_cancellation_counts_as_leaked(
    launch: Launcher, workspace: Path
) -> None:
    client = await _loaded(launch, workspace, STUBBORN)
    with pytest.raises(AttemptTimeout) as timeout:
        await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False, timeout=0.05)
    assert timeout.value.leaked_thread is True
    assert sum(client._leaked_threads_by_pool.values()) == 1
    assert (await client.run_python(ref("quick"), WorkflowValue(text="alive"), blocking=False)).value.text == "alive"


async def test_oversized_diagnostics_fail_the_attempt_not_the_worker(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, HUGE_ERROR)
    with pytest.raises(WorkerRpcError) as failure:
        await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False)
    assert failure.value.code == ErrorCode.USER_EXCEPTION
    assert failure.value.message.startswith("ValueError: xxx")
    assert "chars dropped" in failure.value.message
    assert client.lost is None


async def test_user_stdin_reads_eof_not_the_protocol_channel(launch: Launcher, workspace: Path) -> None:
    """A node that reads fd 0 gets EOF; the protocol reader keeps a private input fd off the shared channel."""
    client = await _loaded(launch, workspace, STDIN_READER)
    result = await client.run_python(ref("fn"), WorkflowValue(text="x"), blocking=False, timeout=5.0)
    assert result.value.text == "''"
    assert client.lost is None


async def test_system_exit_from_a_child_task_fails_only_its_attempt(launch: Launcher, workspace: Path) -> None:
    """The round-12 guard covered the body coroutine; a task the body spawns needs the same protection."""
    client = await _loaded(launch, workspace, GATHER_EXIT)
    with pytest.raises(WorkerRpcError) as failure:
        await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False)
    assert failure.value.code == ErrorCode.USER_EXCEPTION
    assert failure.value.message == "SystemExit: 5"
    assert (await client.run_python(ref("quick"), WorkflowValue(text="x"), blocking=False)).value.text == "x!"
    assert client.lost is None


async def test_cancelling_a_task_before_its_first_step_closes_its_coroutine(launch: Launcher, workspace: Path) -> None:
    """The guard must forward the pre-start cancel to the original coroutine, as the default task factory does."""
    client = await _loaded(launch, workspace, CANCEL_UNSTARTED)
    result = await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False)
    assert result.value.text == "CORO_CLOSED"
    assert client.lost is None


async def test_a_giant_ask_error_still_reaches_the_worker(launch: Launcher, workspace: Path) -> None:
    """An oversized error reply must be bounded, not silently dropped, or the worker's ctx.ask hangs forever."""

    async def refuse(ref_: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        raise AskUnavailable("x" * (17 * 1024 * 1024))

    client = await _loaded(launch, workspace, ASK_ONCE, ask_handler=refuse)
    with pytest.raises(WorkerRpcError) as failure:
        await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False, timeout=5.0)
    assert failure.value.code == ErrorCode.ASK_UNAVAILABLE
    assert client.lost is None


async def test_system_exit_inside_an_async_body_fails_only_its_attempt(launch: Launcher, workspace: Path) -> None:
    """asyncio lets SystemExit out of the event loop; the worker must survive it like the sync path does."""
    client = await _loaded(launch, workspace, EXITS)
    for node in ("leave", "wrapped"):  # a native async body, and a sync wrapper handing back its coroutine
        with pytest.raises(WorkerRpcError) as failure:
            await client.run_python(ref(node), WorkflowValue(text=""), blocking=False)
        assert failure.value.code == ErrorCode.USER_EXCEPTION
        assert failure.value.message == "SystemExit: 3"
    assert (await client.run_python(ref("quick"), WorkflowValue(text="x"), blocking=False)).value.text == "x!"
    assert client.lost is None


@pytest.mark.skipif(sys.platform == "win32", reason="the Windows job object takes the whole tree down on close")
async def test_close_reaps_children_that_outlive_the_worker(launch: Launcher, workspace: Path) -> None:
    """A body's child that ignores SIGTERM must not survive close() just because the worker exited promptly."""
    client = await _loaded(launch, workspace, ORPHAN)
    pid = int((await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False)).value.text)
    try:
        orphan = psutil.Process(pid)
        assert orphan.is_running()
        await client.close(grace=0.3)
        # The killed orphan is init's to reap; until then it is a zombie that still answers by pid.
        await wait_for(lambda: not orphan.is_running(), description="orphan reaped after close")
    finally:
        with contextlib.suppress(psutil.NoSuchProcess):
            psutil.Process(pid).kill()


async def test_module_level_asyncio_primitives_bind_to_the_workers_loop(
    launch: Launcher, workspace: Path, interpreter: str
) -> None:
    """A workflow file may build asyncio primitives at import time: the load thread must present the bodies' loop."""
    client = await _loaded(launch, workspace, MODULE_PRIMITIVES, interpreter=interpreter)
    assert (await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False)).value.text == "ok"


@pytest.mark.skipif(sys.platform == "win32", reason="fork is a POSIX start method")
async def test_worker_death_is_noticed_while_a_forked_child_holds_its_pipes(launch: Launcher, workspace: Path) -> None:
    """The child inherits the protocol pipe, so EOF never comes; the exit status alone must settle the request."""
    client = await _loaded(launch, workspace, FORK_THEN_DIE)
    with pytest.raises(WorkerLostError, match="status 7"):
        await asyncio.wait_for(client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False), timeout=20)
    await client.close(grace=0.3)


async def test_ask_from_a_sync_body_is_unavailable(launch: Launcher, workspace: Path) -> None:
    async def never(ref_: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        raise AssertionError("a sync body's ask must not reach the main process")

    client = await _loaded(launch, workspace, SYNC_ASK, ask_handler=never)
    with pytest.raises(WorkerRpcError) as failure:
        await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False, timeout=5.0)
    assert failure.value.code == ErrorCode.ASK_UNAVAILABLE
    assert "inside an async def node body" in failure.value.message


async def test_ask_prompt_is_checked_before_it_reaches_the_wire(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, WIDE_ASK)
    with pytest.raises(WorkerRpcError) as failure:
        await client.run_python(ref("fn"), WorkflowValue(text=""), blocking=False)
    assert failure.value.code == ErrorCode.PROTOCOL_LIMIT
    assert client.lost is None


async def test_conditions_that_return_awaitables_fail_instead_of_passing(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, ASYNC_CONDITIONS)
    with pytest.raises(WorkerRpcError) as edge:
        await client.eval_outgoing(ref("src"), WorkflowValue(text=""), ("src->dst",))
    assert edge.value.code == ErrorCode.USER_EXCEPTION
    assert "awaitable" in edge.value.message
    with pytest.raises(WorkerRpcError) as until:
        await client.eval_loop_until(ref("L"), 1, WorkflowValue(text=""))
    assert until.value.code == ErrorCode.USER_EXCEPTION
    assert "awaitable" in until.value.message


async def test_cancel_is_never_refused_for_lack_of_request_capacity(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, HANG)
    limit = LIMITS.max_pending_requests
    refs = [
        AttemptRef(run_id="run", node_id="fn", activation_id=f"fn@iter#{i}", attempt=1) for i in range(1, limit + 1)
    ]
    bodies = [asyncio.create_task(client.run_python(each, WorkflowValue(text=""), blocking=False)) for each in refs]
    try:
        await wait_for(lambda: client._in_flight["body"] == client._capacity["body"], description="normal lane full")
        # Queued requests own no wire entry; cancellation can still reach the worker.
        assert len(client._pending) < limit
        assert (await client.cancel(refs[0])).cancelled is True
        with pytest.raises(WorkerRpcError) as failure:
            await bodies[0]
        assert failure.value.code == ErrorCode.ATTEMPT_TERMINATED
    finally:
        for body in bodies:
            body.cancel()
        await asyncio.gather(*bodies, return_exceptions=True)
        await client.close()
