# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Output ownership in the worker: attribution of prints, bounds, and protection of the protocol channel."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from chrys.orchestration.workflows.worker_client import AttemptTimeout, WorkerRpcError
from chrys.service.workflows.protocol import LIMITS
from chrys.service.workflows.scheduler import AttemptRef
from chrys.service.workflows.sdk import WorkflowValue
from tests.orchestration.workflows.conftest import Launcher
from tests.support.workflow_workers import python_workflow

CAPTURE_LIMIT = LIMITS.captured_output_bytes

NOISY_IMPORT = python_workflow("print('module says hi')\ndef fn(text):\n    return text\n", "fn")
CONCURRENT = python_workflow(
    "import asyncio, time\n"
    "async def afn(value, ctx):\n"
    "    for i in range(20):\n        print(value.text, i)\n        await asyncio.sleep(0)\n"
    "    return value.text\n"
    "def sfn(text):\n"
    "    for i in range(20):\n        print(text.text, i)\n        time.sleep(0.001)\n"
    "    return text\n",
    "afn",
    "sfn",
)
OVERSIZED = python_workflow(f"def fn(text):\n    print('x' * {CAPTURE_LIMIT * 3})\n    return text\n", "fn")
TRACEBACK = python_workflow(
    "import sys\ndef fn(text):\n    print('before')\n    sys.stderr.write('warned\\n')\n    raise ValueError('boom')\n",
    "fn",
)
CANCELLED = python_workflow(
    "import asyncio\n"
    "async def fn(value, ctx):\n"
    "    if value.text == 'quick':\n        print('second')\n        return 'done'\n"
    "    print('first')\n"
    "    try:\n        await asyncio.sleep(3600)\n"
    "    finally:\n        print('after cancel')\n",
    "fn",
)
RAW_FD = python_workflow("import os\ndef fn(text):\n    os.write(1, b'raw bytes\\n')\n    return text\n", "fn")
CHILD = python_workflow(
    "import subprocess, sys\n"
    "def fn(text):\n"
    "    subprocess.run([sys.executable, '-c', \"print('child output')\"], stdin=subprocess.DEVNULL, check=True)\n"
    "    return text\n",
    "fn",
)
FORKED = python_workflow(
    "import multiprocessing, os\n"
    "def child():\n    print('forked print', flush=True)\n    os.write(1, b'forked raw\\n')\n"
    "def fn(text):\n"
    "    process = multiprocessing.get_context('fork').Process(target=child)\n"
    "    process.start()\n    process.join(30)\n"
    "    return text\n",
    "fn",
)
SPLIT_BYTES = python_workflow(
    "import asyncio, sys\n"
    "gates = {}\n"
    "def gate(name):\n    return gates.setdefault(name, asyncio.Event())\n"
    "async def a(value, ctx):\n"
    "    sys.stdout.buffer.write(b'\\xc3')\n    gate('a').set()\n    await gate('b').wait()\n"
    "    sys.stdout.buffer.write(b'\\xa9\\n')\n    return 'a'\n"
    "async def b(value, ctx):\n"
    "    await gate('a').wait()\n    sys.stdout.buffer.write(b'hello\\n')\n    gate('b').set()\n    return 'b'\n",
    "a",
    "b",
)
OPEN_BYTES = python_workflow(
    "import sys\n"
    "def mixed(text):\n"
    "    sys.stdout.buffer.write(b'prefix-\\xc3')\n    sys.stdout.write('status')\n    sys.stdout.buffer.write(b'\\xa9')\n"
    "    return text\n"
    "def trailing(text):\n    sys.stdout.buffer.write(b'prefix-\\xc3')\n    return text\n",
    "mixed",
    "trailing",
)
STREAM_STAND_INS = python_workflow(
    "import subprocess, sys\n"
    "def fn(text):\n"
    "    sys.stdout.buffer.write(b'bytes: ' + text.text.encode('utf-8')[:1])\n"
    "    sys.stdout.buffer.write(text.text.encode('utf-8')[1:] + b'\\n')\n"
    "    subprocess.run(\n"
    "        [sys.executable, '-c', \"print('child via stream')\"],\n"
    "        stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr, check=True,\n"
    "    )\n"
    "    return text\n",
    "fn",
)


def ref(node: str, *, attempt: int = 1, activation: str | None = None) -> AttemptRef:
    return AttemptRef(run_id="run", node_id=node, activation_id=activation or f"{node}@iter#1", attempt=attempt)


async def _loaded(launch: Launcher, workspace: Path, source: bytes, **kwargs: object) -> object:
    client = await launch(**kwargs)
    await client.load(source, filename=str(workspace / "wf.py"), workspace=workspace)
    return client


async def native_text(client: object, needle: str) -> str:
    """The native tail is drained by a host thread: poll until *needle* shows up (or give up)."""
    text = ""
    for _ in range(250):
        text = (await client.native_output()).text  # type: ignore[attr-defined]
        if needle in text:
            break
        await asyncio.sleep(0.02)
    return text


async def test_module_level_print_lands_in_load_output(launch: Launcher, workspace: Path, interpreter: str) -> None:
    client = await launch(interpreter=interpreter)
    loaded = await client.load(NOISY_IMPORT, filename=str(workspace / "wf.py"), workspace=workspace)
    assert loaded.stdout.text == "module says hi\n"
    assert (await client.native_output()).text == ""


async def test_concurrent_nodes_do_not_cross_talk(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, CONCURRENT)
    first, second = await asyncio.gather(
        client.run_python(ref("afn"), WorkflowValue(text="A"), blocking=False),
        client.run_python(ref("sfn"), WorkflowValue(text="B"), blocking=False),
    )
    assert first.stdout.text == "".join(f"A {i}\n" for i in range(20))
    assert second.stdout.text == "".join(f"B {i}\n" for i in range(20))


async def test_oversized_output_is_truncated_and_flagged(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, OVERSIZED)
    result = await client.run_python(ref("fn"), WorkflowValue(text="x"), blocking=False)
    assert result.stdout.truncated is True
    assert len(result.stdout.text.encode("utf-8")) <= CAPTURE_LIMIT


async def test_failure_carries_the_attempts_output_and_traceback(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, TRACEBACK)
    with pytest.raises(WorkerRpcError) as failure:
        await client.run_python(ref("fn"), WorkflowValue(text="x"), blocking=False)
    assert failure.value.stdout.text == "before\nwarned\n"
    assert "ValueError: boom" in (failure.value.traceback or "")
    assert "wf.py" in (failure.value.traceback or "")


async def test_capture_is_per_attempt_and_late_writes_are_dropped(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, CANCELLED)
    with pytest.raises(AttemptTimeout):
        await client.run_python(ref("fn", attempt=1), WorkflowValue(text="slow"), blocking=False, timeout=0.3)

    result = await client.run_python(ref("fn", attempt=2), WorkflowValue(text="quick"), blocking=False)

    assert result.stdout.text == "second\n"
    assert "after cancel" not in (await client.native_output()).text


async def test_raw_fd_writes_cannot_reach_the_protocol(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, RAW_FD)
    result = await client.run_python(ref("fn"), WorkflowValue(text="x"), blocking=False)
    assert result.value.text == "x"
    assert result.stdout.text == ""
    assert "raw bytes" in await native_text(client, "raw bytes")


async def test_child_process_inheriting_stdout_cannot_reach_the_protocol(launch: Launcher, workspace: Path) -> None:
    client = await _loaded(launch, workspace, CHILD)
    result = await client.run_python(ref("fn"), WorkflowValue(text="x"), blocking=False)
    assert result.value.text == "x"
    assert "child output" in await native_text(client, "child output")


@pytest.mark.skipif(sys.platform == "win32", reason="fork is a POSIX start method")
async def test_a_forked_child_prints_to_the_native_tail(launch: Launcher, workspace: Path, interpreter: str) -> None:
    """A fork copies the capture sinks; the child's prints join the run-level tail instead of vanishing."""
    client = await _loaded(launch, workspace, FORKED, interpreter=interpreter)
    result = await client.run_python(ref("fn"), WorkflowValue(text="x"), blocking=False, timeout=60)
    assert result.value.text == "x"
    assert result.stdout.text == ""  # the child's output is not the attempt's
    assert "forked print" in await native_text(client, "forked raw")


async def test_stream_stand_ins_offer_a_descriptor_and_a_buffer(
    launch: Launcher, workspace: Path, interpreter: str
) -> None:
    client = await _loaded(launch, workspace, STREAM_STAND_INS, interpreter=interpreter)
    result = await client.run_python(ref("fn"), WorkflowValue(text="é"), blocking=False)
    assert result.value.text == "é"
    assert result.stdout.text == "bytes: é\n"  # bytes split mid-character still decode as one
    assert "child via stream" in await native_text(client, "child via stream")


async def test_a_byte_sequence_left_open_shows_as_replacement_where_it_was_written(
    launch: Launcher, workspace: Path, interpreter: str
) -> None:
    """Text written in between, or the end of the attempt, closes the sequence; later bytes never complete it."""
    client = await _loaded(launch, workspace, OPEN_BYTES, interpreter=interpreter)
    mixed = await client.run_python(ref("mixed"), WorkflowValue(text="x"), blocking=False)
    assert mixed.stdout.text == "prefix-\ufffdstatus\ufffd"
    trailing = await client.run_python(ref("trailing"), WorkflowValue(text="x"), blocking=False)
    assert trailing.stdout.text == "prefix-\ufffd"
    assert trailing.stdout.truncated is False


async def test_a_character_split_across_byte_writes_stays_with_its_attempt(
    launch: Launcher, workspace: Path, interpreter: str
) -> None:
    client = await _loaded(launch, workspace, SPLIT_BYTES, interpreter=interpreter)
    a, b = await asyncio.gather(
        client.run_python(ref("a"), WorkflowValue(text=""), blocking=False),
        client.run_python(ref("b"), WorkflowValue(text=""), blocking=False),
    )
    assert a.stdout.text == "é\n"  # decoded by a's own sink, not completed with b's bytes
    assert b.stdout.text == "hello\n"
