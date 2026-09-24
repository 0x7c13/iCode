# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Round-trip cost of the worker protocol; the bounds are generous, the measured numbers are the point."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from chrys.service.workflows.scheduler import AttemptRef
from chrys.service.workflows.sdk import WorkflowValue
from tests.orchestration.workflows.conftest import Launcher
from tests.support.workflow_workers import python_workflow

NOOP = python_workflow("def fn(text):\n    return text\n", "fn")
FAN_OUT = python_workflow(
    "import asyncio\nasync def fn(value, ctx):\n    await asyncio.sleep(0.05)\n    return 'x'\n", "fn"
)


def ref(activation: str) -> AttemptRef:
    return AttemptRef(run_id="run", node_id="fn", activation_id=activation, attempt=1)


async def test_thousand_sequential_activations(
    launch: Launcher, workspace: Path, record_property: pytest.FixtureRequest
) -> None:
    client = await launch()
    await client.load(NOOP, filename=str(workspace / "wf.py"), workspace=workspace)
    started = time.perf_counter()
    for index in range(1000):
        await client.run_python(ref(f"a{index}"), WorkflowValue(text="x"), blocking=False)
    elapsed = time.perf_counter() - started
    record_property("sequential_1000_seconds", round(elapsed, 3))
    assert elapsed < 20.0


async def test_hundred_concurrent_activations(
    launch: Launcher, workspace: Path, record_property: pytest.FixtureRequest
) -> None:
    client = await launch()
    await client.load(FAN_OUT, filename=str(workspace / "wf.py"), workspace=workspace)
    started = time.perf_counter()
    results = await asyncio.gather(
        *(client.run_python(ref(f"c{i}"), WorkflowValue(text=""), blocking=False) for i in range(100))
    )
    elapsed = time.perf_counter() - started
    record_property("concurrent_100_seconds", round(elapsed, 3))
    assert len(results) == 100
    assert elapsed < 2.5, "100 concurrent 50ms sleeps must overlap, not serialize"
