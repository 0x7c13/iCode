# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Callback values and structured authoring errors across the real worker boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.orchestration.workflows.worker_client import WorkerRpcError
from chrys.service.workflows.protocol import ErrorCode
from chrys.service.workflows.scheduler import AttemptRef
from chrys.service.workflows.sdk import SourceValue, WorkflowValue
from tests.orchestration.workflows.conftest import Launcher
from tests.support.workflow_workers import python_workflow


@pytest.mark.parametrize("arity", [1, 2])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_python_bodies_receive_text_and_data(
    launch: Launcher, interpreter: str, workspace: Path, arity: int, asynchronous: bool
) -> None:
    parameters = "value" if arity == 1 else "value, ctx"
    source = python_workflow(
        f"{'async ' if asynchronous else ''}def body({parameters}):\n"
        "    return value.text + ':' + str(value.data['answer'])\n",
        "body",
    )
    client = await launch(interpreter=interpreter)
    await client.load(source, filename=str(workspace / "values.py"), workspace=workspace)
    result = await client.run_python(
        AttemptRef("run", "body", "opaque-activation", 1),
        WorkflowValue(text="input", data={"answer": 42}),
        blocking=False,
    )
    assert result.value == WorkflowValue(text="input:42")


async def test_join_combiner_receives_source_identities(launch: Launcher, interpreter: str, workspace: Path) -> None:
    source = b"""from chrys.workflows import WorkflowBuilder, WorkflowValue
def combine(sources):
    return WorkflowValue(text='joined', data=[
        [source.node_id, source.activation_id, source.value.data] for source in sources
    ])
wf = WorkflowBuilder('join')
a = wf.python('a', lambda value: value)
b = wf.python('b', lambda value: value)
wf.start(a)
wf.join([a], b, combine=combine)
wf.output(b)
workflow = wf.build()
"""
    client = await launch(interpreter=interpreter)
    await client.load(source, filename=str(workspace / "join.py"), workspace=workspace)
    result = (
        await client.combine(
            AttemptRef("run", "join:b", "opaque-join", 1),
            (SourceValue("a", "opaque-source", WorkflowValue(text="source", data={"n": 2})),),
        )
    ).value
    assert result == WorkflowValue(text="joined", data=[["a", "opaque-source", {"n": 2}]])


@pytest.mark.parametrize("policy", ["Retry(0)", "Retry(1, backoff=-1)"])
async def test_retry_validation_keeps_structured_load_diagnostics(
    launch: Launcher, workspace: Path, policy: str
) -> None:
    client = await launch()
    source = f"from chrys.workflows import Retry\n{policy}\n".encode()
    with pytest.raises(WorkerRpcError) as error:
        await client.load(source, filename=str(workspace / "invalid.py"), workspace=workspace)
    assert error.value.code == ErrorCode.LOAD_FAILED
    assert "Retry." in error.value.data["validation"]["message"]
