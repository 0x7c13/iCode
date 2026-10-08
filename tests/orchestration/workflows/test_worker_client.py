# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Worker client contract: hello/load/manifest, ask, the attempt fence, the projection barrier, loss."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import importlib.util
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

import psutil
import pytest

import chrys.orchestration.workflows.worker_client as worker_client_module
from chrys.foundation.models.ask_user import AskUserAnswer, AskUserOption, AskUserQuestion
from chrys.orchestration.workflows.worker_client import (
    AskUnavailable,
    AttemptTimeout,
    WorkerLostError,
    WorkerRpcError,
    WorkerStartError,
    WorkflowWorkerClient,
)
from chrys.service.workflows.environment import WorkflowEnvironmentManager, parse_environment_request, plan_environment
from chrys.service.workflows.protocol import LIMITS, ErrorCode
from chrys.service.workflows.scheduler import AttemptRef
from chrys.service.workflows.sdk import WorkflowValue
from chrys.service.workflows.sdk_artifact import SdkArtifact
from chrys.service.workflows.values import canonical_json
from tests.orchestration.workflows.conftest import FAKE_WORKER, Launcher, prepared_environment
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for
from tests.support.workflow_workers import create_venv, python_workflow

FIXTURE = Path(__file__).resolve().parents[2] / "service" / "workflows" / "fixtures" / "code-review.py"

STRUCTURED_ASK_WORKFLOW = python_workflow(
    "import json\n"
    "from chrys.workflows import Option, Question\n"
    "async def fn(value, ctx):\n"
    "    answers = await ctx.ask([\n"
    "        Question('Branch?', header='Branch', options=[Option('main', 'the default'), 'release']),\n"
    "        Question('Areas?', options=['API', 'Storage', ' UI '], multi_select=True),\n"
    "        Question('Branch name?', options=['main']),\n"
    "        Question('Anything else?'),\n"
    "    ])\n"
    "    return json.dumps([[list(a.selected), a.text] for a in answers])\n",
    "fn",
)
ASK_WORKFLOW = python_workflow(
    "async def fn(value, ctx):\n    answer = await ctx.ask('color?')\n    return 'answer=' + answer\n",
    "fn",
)
PREFIX_WORKFLOW = python_workflow("import sys\ndef fn(text):\n    return sys.prefix\n", "fn")
HANG_WORKFLOW = python_workflow(
    "import asyncio, time\n"
    "async def hang(value, ctx):\n    await asyncio.sleep(3600)\n"
    "def block(text):\n    time.sleep(3600)\n    return text\n"
    "def quick(text):\n    return text.text + '!'\n",
    "hang",
    "block",
    "quick",
)
EMIT_WORKFLOW = python_workflow(
    "def fn(value, ctx):\n"
    "    for i in range(1, 51):\n        ctx.emit('line %d' % i)\n"
    "    if value.text == 'fail':\n        raise ValueError('after emits')\n"
    "    return value.text\n",
    "fn",
)


def ref(node: str, *, attempt: int = 1, activation: str | None = None, run: str = "run") -> AttemptRef:
    return AttemptRef(run_id=run, node_id=node, activation_id=activation or f"{node}@iter#1", attempt=attempt)


def text(value: str = "") -> WorkflowValue:
    return WorkflowValue(text=value)


async def probe(client: WorkflowWorkerClient) -> list[dict[str, Any]]:
    """Frames the fake worker received so far (its ``probe`` method)."""
    result = await client._call("probe", {}, key=None)
    return result["received"]


def _in_process_manifest() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("golden_code_review", FIXTURE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.workflow.manifest()


# --------------------------------------------------------------------------- real host, both interpreters


async def test_hello_load_manifest_golden(launch: Launcher, interpreter: str, workspace: Path) -> None:
    """Hello names the interpreter, load returns the digests, and the manifest equals the in-process one."""
    client = await launch(interpreter=interpreter)

    source = FIXTURE.read_bytes()
    loaded = await client.load(source, filename=str(FIXTURE), workspace=workspace)
    manifest = loaded.manifest

    assert loaded.entry_digest == hashlib.sha256(source).hexdigest()
    assert loaded.manifest_digest == hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    assert loaded.manifest["schema_version"] == 1
    assert loaded.manifest["warnings"] == []
    assert manifest == _in_process_manifest()


async def test_run_python_ask_roundtrip(launch: Launcher, interpreter: str, workspace: Path) -> None:
    asked: list[tuple[str, tuple[AskUserQuestion, ...]]] = []

    async def answer(attempt: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        asked.append((attempt.activation_id, questions))
        return (AskUserAnswer(values=("blue",)),)

    client = await launch(interpreter=interpreter, ask_handler=answer)
    await client.load(ASK_WORKFLOW, filename=str(workspace / "wf.py"), workspace=workspace)

    result = await client.run_python(ref("fn"), text("x"), blocking=False)

    assert result.value.text == "answer=blue"
    assert asked == [("fn@iter#1", (AskUserQuestion("color?"),))]


async def test_run_python_structured_ask_roundtrip(launch: Launcher, interpreter: str, workspace: Path) -> None:
    """Questions reach the handler as chat ask-user questions; its answers come back as SDK Answers."""
    asked: list[tuple[AskUserQuestion, ...]] = []

    async def answer(attempt: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        asked.append(questions)
        return (
            AskUserAnswer(values=("main",), note="be quick"),
            AskUserAnswer(values=("UI", "API")),
            AskUserAnswer(values=("release candidate",)),
            AskUserAnswer(),
        )

    client = await launch(interpreter=interpreter, ask_handler=answer)
    await client.load(STRUCTURED_ASK_WORKFLOW, filename=str(workspace / "wf.py"), workspace=workspace)

    result = await client.run_python(ref("fn"), text("x"), blocking=False)

    assert json.loads(result.value.text) == [
        [["main"], "be quick"],
        [["API", "UI"], ""],
        [[], "release candidate"],
        [[], ""],
    ]
    assert asked == [
        (
            AskUserQuestion(
                "Branch?", header="Branch", options=(AskUserOption("main", "the default"), AskUserOption("release"))
            ),
            AskUserQuestion(
                "Areas?",
                options=(AskUserOption("API"), AskUserOption("Storage"), AskUserOption("UI")),
                multi_select=True,
            ),
            AskUserQuestion("Branch name?", options=(AskUserOption("main"),)),
            AskUserQuestion("Anything else?"),
        )
    ]


async def test_ask_without_handler_is_ask_unavailable(launch: Launcher, workspace: Path) -> None:
    client = await launch()
    await client.load(ASK_WORKFLOW, filename=str(workspace / "wf.py"), workspace=workspace)

    with pytest.raises(WorkerRpcError) as failure:
        await client.run_python(ref("fn"), text(), blocking=False)

    assert failure.value.code == ErrorCode.ASK_UNAVAILABLE


async def test_emit_projection_barrier_before_result_and_error(launch: Launcher, workspace: Path) -> None:
    """Every emit passes through a slow handler before ``run_python`` returns, on success and on failure."""
    seen: dict[str, list[int]] = {}

    async def slow(attempt: AttemptRef, ordinal: int, line: str) -> None:
        await asyncio.sleep(0.001)
        seen.setdefault(attempt.activation_id, []).append(ordinal)

    client = await launch(emit_handler=slow)
    await client.load(EMIT_WORKFLOW, filename=str(workspace / "wf.py"), workspace=workspace)

    result = await client.run_python(ref("fn", activation="ok"), text("ok"), blocking=False)
    assert result.last_emit_ordinal == 50
    assert seen["ok"] == list(range(1, 51))

    with pytest.raises(WorkerRpcError) as failure:
        await client.run_python(ref("fn", activation="fail"), text("fail"), blocking=False)
    assert failure.value.code == ErrorCode.USER_EXCEPTION
    assert "after emits" in (failure.value.traceback or "")
    assert seen["fail"] == list(range(1, 51))


async def test_sync_timeout_leaks_thread_and_worker_survives(launch: Launcher, workspace: Path) -> None:
    client = await launch()
    await client.load(HANG_WORKFLOW, filename=str(workspace / "wf.py"), workspace=workspace)

    with pytest.raises(AttemptTimeout) as timeout:
        await client.run_python(ref("block"), text("x"), blocking=False, timeout=0.3)
    assert timeout.value.leaked_thread is True
    assert sum(client._leaked_threads_by_pool.values()) == 1

    with pytest.raises(AttemptTimeout) as timeout:
        await client.run_python(ref("hang"), text("x"), blocking=False, timeout=0.3)
    assert timeout.value.leaked_thread is False

    quick = await client.run_python(ref("quick"), text("still"), blocking=False)
    assert quick.value.text == "still!"


async def test_outstanding_requests_reject_when_the_process_dies(launch: Launcher, workspace: Path) -> None:
    client = await launch()
    await client.load(HANG_WORKFLOW, filename=str(workspace / "wf.py"), workspace=workspace)
    pending = [
        asyncio.create_task(client.run_python(ref("hang", activation=f"h{i}"), text(), blocking=False))
        for i in range(5)
    ]
    await asyncio.sleep(0.2)
    assert not any(task.done() for task in pending)

    os.kill(client._process.pid, signal.SIGTERM)

    outcomes = await asyncio.gather(*pending, return_exceptions=True)
    assert all(isinstance(outcome, WorkerLostError) for outcome in outcomes)
    assert client.lost is not None
    with pytest.raises(WorkerLostError):
        await client.native_output()


async def test_loader_contract_sibling_import_file_and_cwd(launch: Launcher, workspace: Path) -> None:
    project = workspace / "proj"
    project.mkdir()
    (project / "helper.py").write_text("VALUE = 'from helper'\n", encoding="utf-8")
    source = python_workflow(
        "import json, os\nimport helper\n"
        "def fn(text):\n"
        "    return json.dumps({'helper': helper.VALUE, 'file': __file__, 'cwd': os.getcwd()})\n",
        "fn",
    )
    entry = project / "wf.py"
    entry.write_bytes(source)
    client = await launch()
    await client.load(source, filename=str(entry), workspace=workspace)

    facts = json.loads((await client.run_python(ref("fn"), text(), blocking=False)).value.text)

    assert facts["helper"] == "from helper"
    assert Path(facts["file"]).resolve() == entry.resolve()
    assert Path(facts["cwd"]).resolve() == workspace.resolve()


async def test_a_utf8_bom_is_tolerated_and_stays_in_the_entry_digest(launch: Launcher, workspace: Path) -> None:
    source = b"\xef\xbb\xbf" + FIXTURE.read_bytes()
    client = await launch()
    loaded = await client.load(source, filename=str(FIXTURE), workspace=workspace)
    assert loaded.entry_digest == hashlib.sha256(source).hexdigest()


async def test_a_surrogateescaped_entry_path_keeps_its_identity_in_the_worker(
    launch: Launcher, interpreter: str, workspace: Path
) -> None:
    filename = os.path.join(str(workspace), "raw-\udcff.py")  # what os.fsdecode() makes of a b"\xff" byte on POSIX
    source = python_workflow(
        "import os\ndef fn(text):\n    return repr((__file__, os.environ['CHRYS_WORKFLOW_ENTRY']))\n", "fn"
    )
    client = await launch(interpreter=interpreter)
    await client.load(source, filename=filename, workspace=workspace)
    assert (await client.run_python(ref("fn"), text(), blocking=False)).value.text == repr((filename, filename))


async def test_load_failure_reports_traceback_and_module_output(launch: Launcher, workspace: Path) -> None:
    client = await launch()
    with pytest.raises(WorkerRpcError) as failure:
        await client.load(b"print('loading')\nraise RuntimeError('bad file')\n", filename="x.py", workspace=workspace)
    assert failure.value.code == ErrorCode.LOAD_FAILED
    assert "bad file" in failure.value.data["traceback"]
    assert failure.value.data["stdout"] == {"text": "loading\n", "truncated": False}


@pytest.mark.parametrize(
    "first_line",
    [b"VALUE = 1\n", 'VALUE = "a\u2028b"\r\n'.encode()],
    ids=["plain", "line-separator-in-a-string"],
)
async def test_a_load_traceback_quotes_the_line_that_ran(launch: Launcher, workspace: Path, first_line: bytes) -> None:
    entry = workspace / "wf.py"
    entry.write_text("# what the file holds by now\n# is not what ran\n", encoding="utf-8")
    client = await launch()

    with pytest.raises(WorkerRpcError) as failure:
        await client.load(first_line + b'raise RuntimeError("boom")\n', filename=str(entry), workspace=workspace)

    assert 'raise RuntimeError("boom")' in failure.value.data["traceback"]


async def test_oversized_validation_diagnostics_fail_the_load_not_the_worker(launch: Launcher, workspace: Path) -> None:
    source = (
        b"from chrys.workflows import WorkflowBuilder\n"
        b"def same(text):\n    return text\n"
        b"wf = WorkflowBuilder('dup')\n"
        b"name = 'x' * (9 * 1024 * 1024)\n"
        b"wf.python(name, same)\nwf.python(name, same)\n"
        b"workflow = wf.build()\n"
    )
    client = await launch()
    with pytest.raises(WorkerRpcError) as failure:
        await client.load(source, filename="x.py", workspace=workspace)
    assert failure.value.code == ErrorCode.LOAD_FAILED
    assert "chars dropped" in failure.value.data["validation"]["message"]
    assert failure.value.data["validation"]["location"].endswith("chars dropped]")
    assert client.lost is None


async def test_stale_chrys_on_pythonpath_does_not_capture_the_sdk(
    launch: Launcher, tmp_path: Path, workspace: Path
) -> None:
    stale = tmp_path / "stale" / "chrys"
    (stale / "workflows").mkdir(parents=True)
    (stale / "__init__.py").write_text("", encoding="utf-8")
    (stale / "workflows" / "__init__.py").write_text("STALE = True\n", encoding="utf-8")

    client = await launch(env={"PYTHONPATH": str(tmp_path / "stale")})
    await client.load(ASK_WORKFLOW, filename=str(workspace / "wf.py"), workspace=workspace)


async def test_namespace_portion_artifact_loses_to_a_regular_package(
    sdk: SdkArtifact, tmp_path: Path, workspace: Path, bytecode_cache: Path
) -> None:
    stale = tmp_path / "stale" / "chrys"
    (stale / "workflows").mkdir(parents=True)
    (stale / "__init__.py").write_text("", encoding="utf-8")
    (stale / "workflows" / "__init__.py").write_text("STALE = True\n", encoding="utf-8")
    environment = await prepared_environment(sys.executable, sdk)
    (sdk.path / "chrys" / "__init__.py").unlink()

    with pytest.raises(WorkerStartError, match="SDK injection failed"):
        await WorkflowWorkerClient.launch(
            environment=environment,
            sdk=sdk,
            workspace=workspace,
            bytecode_cache=bytecode_cache,
            env={"PYTHONPATH": str(tmp_path / "stale")},
        )


async def test_cancelled_launch_reaps_the_worker_it_spawned(
    sdk: SdkArtifact, workspace: Path, bytecode_cache: Path, tmp_path: Path
) -> None:
    """A host that never says hello is still ours to kill when the launch is cancelled."""
    host = tmp_path / "sleepy_host.py"
    pid_file = tmp_path / "pid"
    host.write_text(
        f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(3600)\n", encoding="utf-8"
    )
    environment = await prepared_environment(sys.executable, sdk)
    launching = asyncio.create_task(
        WorkflowWorkerClient.launch(
            environment=environment, sdk=sdk, workspace=workspace, bytecode_cache=bytecode_cache, host_path=host
        )
    )
    await wait_for(
        lambda: launching.done() or bool(pid_file.exists() and pid_file.read_text(encoding="utf-8")),
        description="sleepy host wrote its pid",
        timeout=ENGINE_TURN_TIMEOUT,
    )
    if launching.done():
        await launching
    # Identity, not the number: a freed pid can be reused by another test worker's process before we look.
    host_process = psutil.Process(int(pid_file.read_text(encoding="utf-8")))
    launching.cancel()
    with pytest.raises(asyncio.CancelledError):
        await launching
    await wait_for(lambda: not host_process.is_running(), description="cancelled launch reaped its host")


async def test_launch_rejects_an_interpreter_that_vanished_after_preparation(
    sdk: SdkArtifact, workspace: Path, bytecode_cache: Path
) -> None:
    prepared = await prepared_environment(sys.executable, sdk)
    environment = dataclasses.replace(prepared, executable=str(workspace / "no-such-python"))
    with pytest.raises(WorkerStartError, match="Cannot start"):
        await WorkflowWorkerClient.launch(
            environment=environment, sdk=sdk, workspace=workspace, bytecode_cache=bytecode_cache
        )


async def test_launch_names_a_deleted_workspace_rather_than_the_interpreter(
    sdk: SdkArtifact, workspace: Path, bytecode_cache: Path
) -> None:
    environment = await prepared_environment(sys.executable, sdk)
    workspace.rmdir()
    with pytest.raises(WorkerStartError, match="working directory no longer exists") as caught:
        await WorkflowWorkerClient.launch(
            environment=environment, sdk=sdk, workspace=workspace, bytecode_cache=bytecode_cache
        )
    assert environment.executable not in str(caught.value)


async def test_launch_refuses_an_environment_prepared_for_another_sdk_build(
    sdk: SdkArtifact, workspace: Path, bytecode_cache: Path
) -> None:
    prepared = await prepared_environment(sys.executable, sdk)
    environment = dataclasses.replace(prepared, sdk_digest="0" * 64)
    with pytest.raises(WorkerStartError, match="different SDK build"):
        await WorkflowWorkerClient.launch(
            environment=environment, sdk=sdk, workspace=workspace, bytecode_cache=bytecode_cache
        )


async def test_hello_facts_must_match_the_prepared_environment(
    sdk: SdkArtifact, workspace: Path, bytecode_cache: Path
) -> None:
    """The environment was probed before launch; a worker that reports other facts is not the one prepared."""
    prepared = await prepared_environment(sys.executable, sdk)
    environment = dataclasses.replace(prepared, python_version="0.0.0")
    with pytest.raises(WorkerStartError, match="prepared as"):
        await WorkflowWorkerClient.launch(
            environment=environment, sdk=sdk, workspace=workspace, bytecode_cache=bytecode_cache
        )


async def test_a_byo_venv_runs_the_workflow_in_its_own_interpreter(
    sdk: SdkArtifact, tmp_path: Path, workspace: Path, bytecode_cache: Path
) -> None:
    """Declaration to result: the venv's interpreter, with no chrys installed, runs a node through the injected SDK."""
    venv = create_venv(tmp_path / ".venv")
    request = parse_environment_request(b"# /// script\n# [tool.chrys]\n# python = '.venv'\n# ///\n")
    plan = plan_environment(request, entry_path=tmp_path / "wf.py")
    environment = await WorkflowEnvironmentManager(sdk_digest=sdk.digest).prepare(plan)
    client = await WorkflowWorkerClient.launch(
        environment=environment, sdk=sdk, workspace=workspace, bytecode_cache=bytecode_cache
    )
    try:
        assert environment.mode == "byo"
        await client.load(PREFIX_WORKFLOW, filename=str(workspace / "wf.py"), workspace=workspace)
        result = await client.run_python(ref("fn"), text(), blocking=False)
        assert Path(result.value.text).resolve() == venv.resolve()
    finally:
        await client.close()


# --------------------------------------------------------------------------- scripted worker


@pytest.fixture
async def fake(launch: Launcher) -> Any:
    async def _fake(**kwargs: Any) -> WorkflowWorkerClient:
        return await launch(host_path=FAKE_WORKER, **kwargs)

    return _fake


async def test_fence_drops_late_emits_for_a_terminal_attempt(fake: Any) -> None:
    seen: list[int] = []

    async def emitted(attempt: AttemptRef, ordinal: int, line: str) -> None:
        seen.append(ordinal)

    client = await fake(emit_handler=emitted)

    result = await client.run_python(ref("late_emit"), text(), blocking=False)

    assert result.value.text == "late"
    await client._call("probe", {}, key=None)  # replies after both late notifications were read
    assert seen == []


async def test_fence_answers_a_late_ask_with_attempt_terminated(fake: Any) -> None:
    asked = False

    async def answer(attempt: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        nonlocal asked
        asked = True
        return (AskUserAnswer(values=("never",)),)

    client = await fake(ask_handler=answer)
    await client.run_python(ref("late_ask"), text(), blocking=False)

    async def late_reply() -> dict[str, Any] | None:
        replies = [frame for frame in await probe(client) if frame.get("id") == 2 and "method" not in frame]
        return replies[0] if replies else None

    reply = await late_reply()
    for _ in range(50):
        if reply is not None:
            break
        await asyncio.sleep(0.02)
        reply = await late_reply()
    assert reply is not None
    assert reply["error"]["code"] == ErrorCode.ATTEMPT_TERMINATED
    assert asked is False


async def test_barrier_waits_for_the_dispatcher_on_result_and_error(fake: Any) -> None:
    seen: list[tuple[str, int]] = []

    async def slow(attempt: AttemptRef, ordinal: int, line: str) -> None:
        await asyncio.sleep(0.01)
        seen.append((attempt.node_id, ordinal))

    client = await fake(emit_handler=slow)
    result = await client.run_python(ref("emits"), text(), blocking=False)
    assert result.last_emit_ordinal == 3
    assert seen == [("emits", 1), ("emits", 2), ("emits", 3)]

    with pytest.raises(WorkerRpcError):
        await client.run_python(ref("emits_error"), text(), blocking=False)
    assert seen[3:] == [("emits_error", 1), ("emits_error", 2)]


async def test_emit_handler_failure_does_not_stall_the_barrier(fake: Any) -> None:
    async def broken(attempt: AttemptRef, ordinal: int, line: str) -> None:
        raise RuntimeError("handler bug")

    client = await fake(emit_handler=broken)
    result = await client.run_python(ref("emits"), text(), blocking=False)
    assert result.last_emit_ordinal == 3


async def test_ask_roundtrip_and_ask_handler_exception(fake: Any) -> None:
    async def answer(attempt: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        if attempt.activation_id == "boom":
            raise RuntimeError("handler exploded")
        return (AskUserAnswer(values=(f"{questions[0].question}!",)),)

    client = await fake(ask_handler=answer)
    result = await client.run_python(ref("ask"), text(), blocking=False)
    assert result.value.text == "answer=q!"

    with pytest.raises(WorkerRpcError) as failure:
        await client.run_python(ref("ask", activation="boom"), text(), blocking=False)
    assert failure.value.code == ErrorCode.ASK_UNAVAILABLE


async def test_cancel_terminates_exactly_one_attempt_and_its_ask(fake: Any) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking(attempt: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        entered.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return (AskUserAnswer(values=("never",)),)

    client = await fake(ask_handler=blocking)
    first = asyncio.create_task(client.run_python(ref("ask_then_hang", attempt=1), text(), blocking=False))
    second = asyncio.create_task(client.run_python(ref("hang", attempt=2), text(), blocking=False))
    await entered.wait()

    outcome = await client.cancel(ref("ask_then_hang", attempt=1))
    assert outcome.cancelled is True

    with pytest.raises(WorkerRpcError) as failure:
        await first
    assert failure.value.code == ErrorCode.ATTEMPT_TERMINATED
    await cancelled.wait()
    assert not second.done()

    replies = [frame for frame in await probe(client) if frame.get("id") == 2 and "method" not in frame]
    assert replies and replies[0]["error"]["code"] == ErrorCode.ATTEMPT_TERMINATED

    assert (await client.cancel(ref("hang", attempt=2))).cancelled is True
    with pytest.raises(WorkerRpcError):
        await second
    assert (await client.cancel(ref("hang", attempt=2))).cancelled is False


async def test_close_waits_for_ask_handlers_to_finish_cleaning_up(fake: Any) -> None:
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cleaned = False

    async def blocking(attempt: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        nonlocal cleaned
        entered.set()
        try:
            await asyncio.sleep(3600)
        finally:
            cleaning.set()
            await release.wait()
            cleaned = True
        return (AskUserAnswer(values=("never",)),)

    client = await fake(ask_handler=blocking)
    body = asyncio.create_task(client.run_python(ref("ask_then_hang"), text(), blocking=False))
    await entered.wait()
    assert (await client.cancel(ref("ask_then_hang"))).cancelled is True
    with pytest.raises(WorkerRpcError):
        await body
    await cleaning.wait()  # settlement cancelled the ask; its handler is now inside its cleanup

    closing = asyncio.create_task(client.close())
    done, _ = await asyncio.wait({closing}, timeout=0.5)
    assert not done, "close returned while an ask handler was still cleaning up"
    release.set()
    await closing
    assert cleaned is True


async def test_an_ask_from_a_worker_just_marked_lost_starts_no_handler(
    fake: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frame that exhausts the run budget marks the worker lost; if it is an ask, no handler may start (close waits for them)."""
    entered = asyncio.Event()

    async def blocking(attempt: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        entered.set()
        await asyncio.sleep(3600)
        return (AskUserAnswer(values=("never",)),)

    client = await fake(ask_handler=blocking)
    # The request fits; the ask it draws is one frame over the budget.
    monkeypatch.setattr(worker_client_module, "LIMITS", dataclasses.replace(LIMITS, max_run_frames=client._frames + 1))
    with pytest.raises(WorkerLostError, match="frame budget"):
        await client.run_python(ref("ask"), text(), blocking=False)
    closing = asyncio.create_task(client.close(grace=0.1))
    done, _ = await asyncio.wait({closing}, timeout=5.0)
    for task in list(client._asks):
        task.cancel()  # a red run must not also hang the fixture's close on that handler
    assert done, "close waited for an ask handler that should never have started"
    assert not entered.is_set()


async def test_concurrent_closes_share_one_close_and_survive_a_cancelled_waiter(fake: Any) -> None:
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cleaned = False

    async def blocking(attempt: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        nonlocal cleaned
        entered.set()
        try:
            await asyncio.sleep(3600)
        finally:
            cleaning.set()
            await release.wait()
            cleaned = True
        return (AskUserAnswer(values=("never",)),)

    client = await fake(ask_handler=blocking)
    body = asyncio.create_task(client.run_python(ref("ask_then_hang"), text(), blocking=False))
    await entered.wait()
    assert (await client.cancel(ref("ask_then_hang"))).cancelled is True
    with pytest.raises(WorkerRpcError):
        await body
    await cleaning.wait()

    first = asyncio.create_task(client.close())
    second = asyncio.create_task(client.close())
    done, _ = await asyncio.wait({first, second}, timeout=0.5)
    assert not done, "a close returned while the shared close was still draining"
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    release.set()
    await first
    assert cleaned is True


async def test_worker_exit_rejects_outstanding_and_cancels_asks(fake: Any) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking(attempt: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        entered.set()
        try:
            await asyncio.sleep(3600)
        finally:
            cancelled.set()
        return (AskUserAnswer(values=("never",)),)

    client = await fake(ask_handler=blocking)
    hanging = [
        asyncio.create_task(client.run_python(ref("hang", activation=f"h{i}"), text(), blocking=False))
        for i in range(3)
    ]
    asking = asyncio.create_task(client.run_python(ref("ask_then_hang"), text(), blocking=False))
    await entered.wait()

    with pytest.raises(WorkerLostError):
        await client.run_python(ref("exit"), text(), blocking=False)

    outcomes = await asyncio.gather(*hanging, asking, return_exceptions=True)
    assert all(isinstance(outcome, WorkerLostError) for outcome in outcomes)
    await cancelled.wait()
    assert client.lost is not None


async def test_worker_loss_while_emits_wait_for_projection_is_reported_as_lost(fake: Any) -> None:
    blocked = asyncio.Event()

    async def stuck(attempt: AttemptRef, ordinal: int, line: str) -> None:
        blocked.set()
        await asyncio.sleep(3600)

    client = await fake(emit_handler=stuck)
    running = asyncio.create_task(client.run_python(ref("emits"), text(), blocking=False))
    await blocked.wait()
    await asyncio.sleep(0.05)  # let the result frame land behind the stuck emits
    assert not running.done()

    os.kill(client._process.pid, signal.SIGTERM)

    with pytest.raises(WorkerLostError):
        await running


@pytest.mark.parametrize("shape", ["answer", "error"])
async def test_unframeable_ask_answers_and_errors_still_get_a_reply(fake: Any, shape: str) -> None:
    async def bad(attempt: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        if shape == "error":
            raise AskUnavailable("x" * LIMITS.max_frame_bytes)
        # A well-formed answer that is too big to frame, not one the answer mapping rejects.
        return (AskUserAnswer(values=("x" * LIMITS.max_frame_bytes,)),)

    client = await fake(ask_handler=bad)
    with pytest.raises(WorkerRpcError) as failure:
        await asyncio.wait_for(client.run_python(ref("ask"), text(), blocking=False), timeout=5.0)
    assert failure.value.code == ErrorCode.ASK_UNAVAILABLE


async def test_timeout_leak_budget_marks_the_worker_lost(fake: Any) -> None:
    client = await fake()
    budget = LIMITS.worker_leak_budget
    for index in range(budget - 1):
        with pytest.raises(AttemptTimeout) as timeout:
            await client.run_python(ref("leak", activation=f"l{index}"), text(), blocking=False, timeout=0.05)
        assert timeout.value.leaked_thread is True
    assert sum(client._leaked_threads_by_pool.values()) == budget - 1

    with pytest.raises(WorkerLostError, match="leak budget"):
        await client.run_python(ref("leak", activation="last"), text(), blocking=False, timeout=0.05)


async def test_leaks_reported_by_body_envelopes_count_against_the_budget(fake: Any) -> None:
    client = await fake()
    budget = LIMITS.worker_leak_budget
    for index in range(budget):
        assert (
            await client.run_python(ref("leaky", activation=f"l{index}"), text(), blocking=False)
        ).value.text == "leaky"
    assert sum(client._leaked_threads_by_pool.values()) == budget
    assert client.lost is not None  # the last result still landed; the pool is exhausted for anything after it

    with pytest.raises(WorkerLostError, match="leak budget"):
        await client.run_python(ref("ok"), text(), blocking=False)


async def test_a_deadline_that_lands_during_transmission_still_charges_the_envelope(fake: Any) -> None:
    """Once written, a frame reaches the worker; cutting the wait short must not orphan its terminal envelope."""
    client = await fake()
    stdin = client._process.stdin
    real_drain = stdin.drain

    async def stalled_drain() -> None:
        stdin.drain = real_drain  # only this first frame is caught in flight
        await asyncio.Event().wait()

    stdin.drain = stalled_drain
    with pytest.raises(AttemptTimeout) as timeout:
        await client.run_python(ref("leak"), text(), blocking=False, timeout=0.05)
    assert timeout.value.leaked_thread
    assert sum(client._leaked_threads_by_pool.values()) == 1


async def test_async_timeout_does_not_count_against_the_budget(fake: Any) -> None:
    client = await fake()
    with pytest.raises(AttemptTimeout) as timeout:
        await client.run_python(ref("hang"), text(), blocking=False, timeout=0.05)
    assert timeout.value.leaked_thread is False
    assert sum(client._leaked_threads_by_pool.values()) == 0
    assert (await client.run_python(ref("ok"), text(), blocking=False)).value.text == "ok"


async def test_unknown_reverse_request_is_answered_not_fatal(fake: Any) -> None:
    client = await fake()
    result = await client.run_python(ref("bogus_reverse"), text(), blocking=False)
    assert result.value.text == "bogus"
    replies = [frame for frame in await probe(client) if frame.get("id") == 2 and "method" not in frame]
    assert replies and replies[0]["error"]["code"] == ErrorCode.UNKNOWN_METHOD


async def test_malformed_ask_questions_lose_the_worker(fake: Any) -> None:
    asked = False

    async def answer(attempt: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        nonlocal asked
        asked = True
        return (AskUserAnswer(),)

    client = await fake(ask_handler=answer)
    with pytest.raises(WorkerLostError, match="protocol error: ask option is malformed"):
        await client.run_python(ref("malformed_ask"), text(), blocking=False)
    assert asked is False


async def test_mapping_failure_still_answers_the_worker(fake: Any) -> None:
    """An answer that does not fit its questions is answered ask_unavailable instead of leaving the worker waiting."""

    async def unfit(attempt: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        return (AskUserAnswer(), AskUserAnswer())

    client = await fake(ask_handler=unfit)
    with pytest.raises(WorkerRpcError) as failure:
        await asyncio.wait_for(client.run_python(ref("ask"), text(), blocking=False), timeout=ENGINE_TURN_TIMEOUT)
    assert failure.value.code == ErrorCode.ASK_UNAVAILABLE


async def test_protocol_garbage_loses_the_worker(fake: Any) -> None:
    client = await fake()
    with pytest.raises(WorkerLostError, match="protocol error"):
        await client.run_python(ref("garbage"), text(), blocking=False)
    assert client.lost is not None
    with pytest.raises(WorkerLostError):
        await client.run_python(ref("ok"), text(), blocking=False)


async def test_close_is_idempotent_and_shuts_the_worker_down(fake: Any) -> None:
    client = await fake()
    await client.close()
    await client.close()
    with pytest.raises(WorkerLostError):
        await client.run_python(ref("ok"), text(), blocking=False)
