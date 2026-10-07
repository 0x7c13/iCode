# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Only confirmed source runs: the worker's bytecode cache is private and a workflow folder's part of it is cleared."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import stat
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from chrys.orchestration.workflows.worker_client import WorkerRpcError
from chrys.service.workflows import worker_host
from chrys.service.workflows.protocol import ErrorCode
from chrys.service.workflows.scheduler import AttemptRef
from chrys.service.workflows.sdk import WorkflowValue
from tests.orchestration.workflows.conftest import Launcher
from tests.support.waiting import ENGINE_TURN_TIMEOUT
from tests.support.workflow_workers import python_workflow

_PLANT = (
    "import importlib.util, py_compile, sys\n"
    "source, poison = sys.argv[1], sys.argv[2]\n"
    "py_compile.compile(poison, cfile=importlib.util.cache_from_source(source), dfile=source, doraise=True,\n"
    "                   invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)\n"
)


@pytest.fixture
def bytecode_cache(tmp_path: Path) -> Path:
    """A cache of this test's own, so what it finds there was written by this test."""
    return tmp_path / "pycache"


def _ref() -> AttemptRef:
    return AttemptRef(run_id="run", node_id="fn", activation_id="fn@iter#1", attempt=1)


def _probing_entry(word: str, *, header: bytes = b"") -> bytes:
    """An entry whose node reports its own word and its helper's, here and in a spawned child."""
    return header + python_workflow(
        "import concurrent.futures, multiprocessing\n"
        "import helpers as helper\n"
        f"ENTRY = {word!r}\n"
        "def probe(_):\n"
        "    return ENTRY + '/' + helper.VALUE\n"
        "def fn(text):\n"
        "    context = multiprocessing.get_context('spawn')\n"
        "    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:\n"
        "        child = pool.submit(probe, None).result()\n"
        "    return probe(None) + '|' + child\n",
        "fn",
    )


def _env_without_prefix() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key != "PYTHONPYCACHEPREFIX"}


def _plant(interpreter: str, source: Path, poison: bytes, scratch: Path) -> Path:
    """Compile *poison* into the unchecked-hash cache Python would use for *source* without a private prefix."""
    scratch.mkdir(parents=True, exist_ok=True)
    poison_file = scratch / f"poison-{source.name}"
    poison_file.write_bytes(poison)
    subprocess.run(
        [interpreter, "-c", _PLANT, str(source), str(poison_file)],
        env=_env_without_prefix(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    )
    [planted] = (source.parent / "__pycache__").glob(f"{source.stem}.*.pyc")
    return planted


def _import_value(interpreter: str, directory: Path, module: str) -> str:
    code = f"import sys; sys.path.insert(0, {str(directory)!r}); import {module}; print({module}.VALUE)"
    done = subprocess.run(
        [interpreter, "-c", code],
        env=_env_without_prefix(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@pytest.mark.parametrize("layout", ["folder", "folder-as-cwd", "file"])
async def test_bytecode_planted_next_to_the_source_never_runs(
    launch: Launcher, tmp_path: Path, workspace: Path, bytecode_cache: Path, interpreter: str, layout: str
) -> None:
    folder = tmp_path / "pkg" if layout.startswith("folder") else workspace
    entry = folder / ("pkg.py" if layout.startswith("folder") else "wf.py")
    helper = folder / "helpers.py"
    folder.mkdir(exist_ok=True)
    entry.write_bytes(_probing_entry("source"))
    helper.write_bytes(b"VALUE = 'source'\n")
    planted = {
        _plant(interpreter, entry, _probing_entry("poison"), tmp_path / "scratch"),
        _plant(interpreter, helper, b"VALUE = 'poison'\n", tmp_path / "scratch"),
    }
    # The planted cache is what Python itself would run.
    assert await asyncio.to_thread(_import_value, interpreter, folder, "helpers") == "poison"

    client = await launch(interpreter=interpreter, workspace=folder if layout == "folder-as-cwd" else workspace)
    await client.load(
        entry.read_bytes(),
        filename=str(entry),
        workspace=folder if layout == "folder-as-cwd" else workspace,
        package_dir=str(folder) if layout.startswith("folder") else None,
    )
    result = await client.run_python(_ref(), WorkflowValue(text=""), blocking=False, timeout=ENGINE_TURN_TIMEOUT)

    assert result.value.text == "source/source|source/source"
    # Nothing new is written next to the source; the worker's own cache is private.
    assert set((folder / "__pycache__").iterdir()) == planted
    assert list(bytecode_cache.rglob("helpers.*.pyc"))
    if sys.platform != "win32":
        assert stat.S_IMODE(bytecode_cache.stat().st_mode) == 0o700


@pytest.mark.skipif(sys.platform == "win32", reason="the wrapper is a POSIX shell script")
async def test_an_interpreter_that_ignores_the_private_cache_cannot_load(
    launch: Launcher, tmp_path: Path, workspace: Path
) -> None:
    wrapper = tmp_path / "python-E"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -E "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    marker = tmp_path / "ran"
    source = python_workflow(
        f"import pathlib\npathlib.Path({str(marker)!r}).touch()\ndef fn(text):\n    return text\n", "fn"
    )

    client = await launch(interpreter=str(wrapper))
    with pytest.raises(WorkerRpcError) as failure:
        await client.load(source, filename=str(workspace / "wf.py"), workspace=workspace)

    assert failure.value.code == ErrorCode.LOAD_FAILED
    assert "ignores PYTHONPYCACHEPREFIX" in failure.value.message
    assert not marker.exists()


def _helper_entry(helper: str, *, optimized_child: bool = False) -> bytes:
    """An entry reporting its helper's value here, in a spawned child and, optionally, in a ``python -O`` child."""
    optimized = (
        "    code = 'import sys; sys.path.insert(0, ' + repr(os.path.dirname(__file__)) + '); "
        f"import {helper}; print({helper}.VALUE)'\n"
        "    done = subprocess.run([sys.executable, '-O', '-c', code], stdin=subprocess.DEVNULL,\n"
        "                          capture_output=True, text=True, check=True)\n"
        "    parts.append(done.stdout.strip())\n"
        if optimized_child
        else ""
    )
    return python_workflow(
        "import concurrent.futures, multiprocessing, os, subprocess, sys\n"
        f"import {helper} as helper\n"
        "def probe(_):\n    return helper.VALUE\n"
        "def fn(text):\n"
        "    context = multiprocessing.get_context('spawn')\n"
        "    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:\n"
        "        parts = [probe(None), pool.submit(probe, None).result()]\n"
        + optimized
        + "    return '|'.join(parts)\n",
        "fn",
    )


def _rewrite_keeping_size_and_mtime(path: Path, body: bytes) -> None:
    before = path.stat()
    assert len(body) == before.st_size
    path.write_bytes(body)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


@pytest.mark.parametrize("variant", ["spawn", "pyw", "optimized-child", "pythonoptimize-3"])
async def test_a_folder_reloads_from_its_current_source_even_when_the_cache_looks_fresh(
    launch: Launcher, tmp_path: Path, workspace: Path, interpreter: str, variant: str
) -> None:
    if variant == "pyw" and sys.platform != "win32":
        pytest.skip(".pyw is a source suffix only on Windows")
    if variant != "spawn" and interpreter != sys.executable:
        pytest.skip("one interpreter is enough for the cache variants")
    folder = tmp_path / "pkg"
    folder.mkdir()
    helper = folder / ("steps.pyw" if variant == "pyw" else "steps.py")
    entry = folder / "pkg.py"
    entry.write_bytes(_helper_entry("steps", optimized_child=variant == "optimized-child"))
    helper.write_bytes(b"VALUE = 'one'\n")
    env = {"PYTHONOPTIMIZE": "3"} if variant == "pythonoptimize-3" else {}
    expected_parts = 3 if variant == "optimized-child" else 2

    async def load(package_dir: str | None) -> str:
        client = await launch(interpreter=interpreter, env=env)
        try:
            await client.load(entry.read_bytes(), filename=str(entry), workspace=workspace, package_dir=package_dir)
            result = await client.run_python(
                _ref(), WorkflowValue(text=""), blocking=False, timeout=ENGINE_TURN_TIMEOUT
            )
        finally:
            await client.close()
        return result.value.text

    assert await load(str(folder)) == "|".join(["one"] * expected_parts)
    _rewrite_keeping_size_and_mtime(helper, b"VALUE = 'two'\n")
    # Without the folder named, the timestamp cache can't tell: the edit would not be seen.
    assert await load(None) == "|".join(["one"] * expected_parts)

    assert await load(str(folder)) == "|".join(["two"] * expected_parts)


@pytest.mark.parametrize("init", [True, False], ids=["package", "namespace"])
async def test_a_folder_imports_from_its_subfolders_here_and_in_a_spawned_child(
    launch: Launcher, tmp_path: Path, workspace: Path, interpreter: str, init: bool
) -> None:
    folder = tmp_path / "pkg"
    helper = folder / "helpers" / "git.py"
    helper.parent.mkdir(parents=True)
    if init:
        (helper.parent / "__init__.py").write_bytes(b"")
    helper.write_bytes(b"VALUE = 'one'\n")
    entry = folder / "pkg.py"
    entry.write_bytes(_helper_entry("helpers.git"))

    async def load() -> str:
        client = await launch(interpreter=interpreter)
        try:
            await client.load(entry.read_bytes(), filename=str(entry), workspace=workspace, package_dir=str(folder))
            result = await client.run_python(
                _ref(), WorkflowValue(text=""), blocking=False, timeout=ENGINE_TURN_TIMEOUT
            )
        finally:
            await client.close()
        return result.value.text

    assert await load() == "one|one"
    _rewrite_keeping_size_and_mtime(helper, b"VALUE = 'two'\n")

    assert await load() == "two|two"


async def test_a_spawned_child_runs_the_entry_as_the_worker_does(
    launch: Launcher, workspace: Path, interpreter: str
) -> None:
    """The child decodes the entry as strict UTF-8, as the worker does, whatever its coding declaration says."""
    entry = workspace / "wf.py"
    (workspace / "helpers.py").write_bytes(b"VALUE = 'helper'\n")
    entry.write_bytes(_probing_entry("é", header=b"# coding: latin-1\n"))
    client = await launch(interpreter=interpreter)
    await client.load(entry.read_bytes(), filename=str(entry), workspace=workspace)

    result = await client.run_python(_ref(), WorkflowValue(text=""), blocking=False, timeout=ENGINE_TURN_TIMEOUT)

    parent, child = result.value.text.split("|")
    assert parent.split("/")[0] == child.split("/")[0] == "é"


def test_clearing_a_folder_removes_every_cached_variant_of_its_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "pycache"
    monkeypatch.setattr(sys, "pycache_prefix", str(prefix))
    folder = tmp_path / "pkg"
    sources = [folder / "Steps.py", folder / "café.py", folder / "sub" / "deep.py", folder / "__pycache__" / "tools.py"]
    for source in sources:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"VALUE = 1\n")
    (folder / ".venv" / "lib").mkdir(parents=True)
    (folder / ".venv" / "lib" / "third.py").write_bytes(b"")
    tag = sys.implementation.cache_tag

    def mirror(source: Path) -> Path:
        directory = Path(importlib.util.cache_from_source(str(source))).parent
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    top, deep, venv = mirror(sources[0]), mirror(sources[2]), mirror(folder / ".venv" / "lib" / "third.py")
    cleared = [
        mirror(sources[3]) / f"tools.{tag}.pyc",
        top / f"steps.{tag}.pyc",
        top / "steps.cpython-39.pyc",
        top / f"Steps.{tag}.opt-3.pyc",
        top / f"{unicodedata.normalize('NFD', 'café')}.{tag}.pyc",
        deep / f"deep.{tag}.opt-1.pyc",
    ]
    kept = [top / f"stepsx.{tag}.pyc", top / f"other.{tag}.pyc", top / "steps.txt", venv / f"third.{tag}.pyc"]
    for path in [*cleared, *kept]:
        path.write_bytes(b"\0")

    worker_host._clear_cached_bytecode(str(folder))

    assert [path for path in cleared if path.exists()] == []
    assert [path for path in kept if not path.exists()] == []


def test_a_load_is_refused_when_the_interpreter_caches_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "pycache_prefix", None)
    assert "ignores PYTHONPYCACHEPREFIX" in (worker_host._bytecode_cache_problem(str(tmp_path), None) or "")
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "other"))
    assert "ignores PYTHONPYCACHEPREFIX" in (worker_host._bytecode_cache_problem(str(tmp_path), None) or "")
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path))
    assert worker_host._bytecode_cache_problem(str(tmp_path), None) is None
