# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A ``diagnose`` load places what went wrong as a compiler would: file, line, column, notes and a hint."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from chrys.orchestration.workflows.worker_client import (
    LoadDiagnostic,
    LoadNote,
    WorkerRpcError,
    WorkflowWorkerClient,
    load_diagnostics,
)
from chrys.service.workflows import worker_host
from chrys.service.workflows.protocol import ErrorCode
from tests.orchestration.workflows.conftest import Launcher

BUILDER = "from chrys.workflows import WorkflowBuilder\n"
VALID_TAIL = "wf = WorkflowBuilder('t')\na = wf.python('a', lambda text: text)\nwf.start(a)\nwf.output(a)\nworkflow = wf.build()\n"


def _write(folder: Path, files: dict[str, str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder


async def _failed_load(
    client: WorkflowWorkerClient,
    entry: Path,
    workspace: Path,
    *,
    folder: bool = False,
    precompile: tuple[str, ...] = (),
) -> tuple[tuple[LoadDiagnostic, ...], bool]:
    with pytest.raises(WorkerRpcError) as failed:
        await client.load(
            entry.read_bytes(),
            filename=str(entry),
            workspace=workspace,
            package_dir=str(entry.parent) if folder else None,
            diagnose=True,
            precompile=precompile,
        )
    assert failed.value.code == ErrorCode.LOAD_FAILED
    return load_diagnostics(failed.value.data)


async def test_a_syntax_error_in_the_entry_is_placed_at_its_line_and_column(
    launch: Launcher, interpreter: str, workspace: Path, tmp_path: Path
) -> None:
    entry = _write(tmp_path, {"wf.py": BUILDER + "x = 1\nnames = ['中文', 1 +]\n"}) / "wf.py"
    client = await launch(interpreter=interpreter)
    (diagnostic,), truncated = await _failed_load(client, entry, workspace)
    assert truncated is False
    assert (diagnostic.code, diagnostic.file, diagnostic.line) == ("syntax_error", str(entry), 3)
    assert diagnostic.message.startswith("SyntaxError: ")
    assert diagnostic.source_line == "names = ['中文', 1 +]"
    assert diagnostic.notes == ()
    # Columns count characters and are given only where Python reports them that way (3.11+).
    assert diagnostic.column == (19 if _counts_columns(interpreter) else None)


def _counts_columns(interpreter: str) -> bool:
    return interpreter == sys.executable and sys.version_info >= (3, 11)


async def test_a_failure_in_a_helper_is_placed_there_with_the_way_the_entry_reached_it(
    launch: Launcher, interpreter: str, workspace: Path, tmp_path: Path
) -> None:
    folder = _write(
        tmp_path / "wf",
        {
            "wf.py": BUILDER + "import helpers\n" + VALID_TAIL,
            "helpers.py": "def boom():\n    return ['中文', 1 + 'a']\n\nboom()\n",
        },
    )
    client = await launch(interpreter=interpreter)
    (diagnostic,), _ = await _failed_load(client, folder / "wf.py", workspace, folder=True)
    helpers = str(folder / "helpers.py")
    assert (diagnostic.code, diagnostic.file, diagnostic.line) == ("load_error", helpers, 2)
    assert diagnostic.message.startswith("TypeError: ")
    assert diagnostic.source_line == "    return ['中文', 1 + 'a']"
    assert diagnostic.notes == (
        LoadNote("called from", helpers, 4),
        LoadNote("imported from", str(folder / "wf.py"), 2),
    )
    if _counts_columns(interpreter):
        assert (diagnostic.column, diagnostic.end_line, diagnostic.end_column) == (19, 2, 26)
    else:
        assert diagnostic.column is None


async def test_the_traceback_keeps_only_the_workflow_s_own_frames_of_every_chained_exception(
    launch: Launcher, interpreter: str, workspace: Path, tmp_path: Path
) -> None:
    helpers_text = (
        "import json\n\ntry:\n    json.loads('{')\nexcept ValueError as exc:\n    raise RuntimeError('bad') from exc\n"
    )
    folder = _write(tmp_path / "wf", {"wf.py": BUILDER + "import helpers\n" + VALID_TAIL, "helpers.py": helpers_text})
    client = await launch(interpreter=interpreter)
    entry = folder / "wf.py"
    with pytest.raises(WorkerRpcError) as failed:
        await client.load(
            entry.read_bytes(), filename=str(entry), workspace=workspace, package_dir=str(folder), diagnose=True
        )
    text = failed.value.traceback
    helpers = str(folder / "helpers.py")
    # The cause's frames (json's own dropped), then the raise reached through the import (the worker's dropped).
    assert re.findall(r'File "([^"]+)", line (\d+)', text) == [(helpers, "4"), (str(entry), "2"), (helpers, "6")]
    assert "direct cause" in text
    assert text.endswith("RuntimeError: bad\n")


async def test_every_broken_file_of_a_folder_is_reported_before_anything_runs(
    launch: Launcher, workspace: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "ran"
    folder = _write(
        tmp_path / "wf",
        {
            "wf.py": BUILDER + f"open({str(marker)!r}, 'w').close()\n" + VALID_TAIL,
            "a.py": "def f(:\n    pass\n",
            "b.py": "x = 1\ny = (\n",
            "notes.txt": "def (:\n",
        },
    )
    (folder / "c.py").write_bytes("x = 1\n".encode("utf-16"))  # null bytes: Python names no file for these
    files = tuple(str(folder / name) for name in ("a.py", "b.py", "c.py", "notes.txt", "wf.py"))
    diagnostics, _ = await _failed_load(await launch(), folder / "wf.py", workspace, folder=True, precompile=files)
    assert [(d.code, d.file, d.line) for d in diagnostics] == [
        ("syntax_error", files[0], 1),
        ("syntax_error", files[1], 2),
        ("syntax_error", files[2], None),
    ]
    assert not marker.exists()

    (folder / "wf.py").write_text(BUILDER + "workflow = (\n", encoding="utf-8")
    diagnostics, _ = await _failed_load(await launch(), folder / "wf.py", workspace, folder=True, precompile=files)
    assert [d.file for d in diagnostics] == [str(folder / "wf.py"), *files[:3]]


async def test_a_build_error_points_at_the_declaration_and_says_where_build_ran(
    launch: Launcher, workspace: Path, tmp_path: Path
) -> None:
    source = (
        BUILDER
        + "wf = WorkflowBuilder('t')\n"
        + "a = wf.python('a', lambda text: text)\n"
        + "wf.python('y', lambda text: text)\n"
        + "wf.start(a)\nwf.output(a)\n"
        + "workflow = wf.build()\n"
    )
    entry = _write(tmp_path, {"wf.py": source}) / "wf.py"
    (diagnostic,), _ = await _failed_load(await launch(), entry, workspace)
    assert (diagnostic.code, diagnostic.line, diagnostic.node) == ("sdk_validation_error", 4, "y")
    assert diagnostic.source_line == "wf.python('y', lambda text: text)"
    assert diagnostic.notes == (LoadNote("build() was called at", str(entry), 7),)


async def test_a_registration_error_is_placed_at_the_call_that_made_it(
    launch: Launcher, workspace: Path, tmp_path: Path
) -> None:
    source = BUILDER + "wf = WorkflowBuilder('t')\nwf.python('a', len)\nwf.python('a', len)\n"
    entry = _write(tmp_path, {"wf.py": source}) / "wf.py"
    (diagnostic,), _ = await _failed_load(await launch(), entry, workspace)
    assert (diagnostic.code, diagnostic.file, diagnostic.line, diagnostic.node) == (
        "sdk_validation_error",
        str(entry),
        4,
        "a",
    )
    assert "already used" in diagnostic.message
    assert diagnostic.notes == ()


async def test_code_compiled_under_a_made_up_name_is_placed_at_the_line_that_compiled_it(
    launch: Launcher, workspace: Path, tmp_path: Path
) -> None:
    entry = _write(tmp_path, {"wf.py": BUILDER + "x = 1\ncompile('\\n(', '<gen\\x1b>', 'exec')\n"}) / "wf.py"
    (diagnostic,), _ = await _failed_load(await launch(), entry, workspace)
    assert (diagnostic.code, diagnostic.file, diagnostic.line) == ("syntax_error", str(entry), 3)
    assert diagnostic.message.endswith("(in <gen�>, line 2)")


@pytest.mark.parametrize(
    ("tail", "message"),
    [
        ("x = 1\n", "the workflow file defines no module-level `workflow`"),
        ("workflow = 3\n", "module-level `workflow` is a int, not the Workflow that build() returns"),
    ],
)
async def test_a_file_that_builds_no_workflow_says_how_to_bind_one(
    launch: Launcher, workspace: Path, tmp_path: Path, tail: str, message: str
) -> None:
    entry = _write(tmp_path, {"wf.py": BUILDER + tail}) / "wf.py"
    (diagnostic,), _ = await _failed_load(await launch(), entry, workspace)
    assert (diagnostic.code, diagnostic.message, diagnostic.file, diagnostic.line) == (
        "missing_workflow",
        message,
        str(entry),
        None,
    )
    assert diagnostic.hint == "assign workflow = wf.build() at module level"


@pytest.mark.parametrize(
    ("line", "hint"),
    [
        ("from .helpers import TITLE\n", "write 'from helpers import ...'"),
        ("from . import helpers\n", "write 'import ...'"),
    ],
)
async def test_a_relative_import_in_a_folder_suggests_importing_by_name(
    launch: Launcher, workspace: Path, tmp_path: Path, line: str, hint: str
) -> None:
    folder = _write(tmp_path / "wf", {"wf.py": BUILDER + line + VALID_TAIL, "helpers.py": "TITLE = 't'\n"})
    (diagnostic,), _ = await _failed_load(await launch(), folder / "wf.py", workspace, folder=True)
    assert (diagnostic.code, diagnostic.line) == ("load_error", 2)
    assert diagnostic.message.startswith("ImportError: ")
    assert diagnostic.hint is not None and diagnostic.hint.endswith(hint)


async def test_a_relative_import_inside_a_real_package_gets_no_such_hint(
    launch: Launcher, workspace: Path, tmp_path: Path
) -> None:
    folder = _write(tmp_path / "wf", {"wf.py": BUILDER + "from lib import run\n" + VALID_TAIL})
    _write(folder / "lib", {"__init__.py": "from .core import run\n", "core.py": "def runn():\n    pass\n"})
    (diagnostic,), _ = await _failed_load(await launch(), folder / "wf.py", workspace, folder=True)
    assert (diagnostic.code, diagnostic.file, diagnostic.line, diagnostic.hint) == (
        "load_error",
        str(folder / "lib" / "__init__.py"),
        1,
        None,
    )


async def test_a_folder_file_named_like_a_stdlib_module_leaves_the_diagnosis_working(
    launch: Launcher, workspace: Path, tmp_path: Path
) -> None:
    folder = _write(tmp_path / "wf", {"wf.py": BUILDER + VALID_TAIL, "sysconfig.py": "SETTING = True\n"})
    entry = folder / "wf.py"
    loaded = await (await launch()).load(
        entry.read_bytes(),
        filename=str(entry),
        workspace=workspace,
        package_dir=str(folder),
        diagnose=True,
        precompile=(str(folder / "sysconfig.py"), str(entry)),
    )
    assert loaded.sites == {"a": (str(entry), 3)}


async def test_a_diagnose_load_reports_where_each_node_was_declared(
    launch: Launcher, workspace: Path, tmp_path: Path
) -> None:
    folder = _write(
        tmp_path / "wf",
        {
            "wf.py": BUILDER + "from nodes import add\nwf = WorkflowBuilder('t')\na = wf.python('a', len)\n"
            "b = add(wf)\nwf.start(a)\nwf.edge(a, b)\nwf.output(b)\nworkflow = wf.build()\n",
            "nodes.py": "def add(wf):\n    return wf.python('b', len)\n",
        },
    )
    entry = folder / "wf.py"
    client = await launch()
    loaded = await client.load(
        entry.read_bytes(), filename=str(entry), workspace=workspace, package_dir=str(folder), diagnose=True
    )
    assert loaded.sites == {"a": (str(entry), 4), "b": (str(folder / "nodes.py"), 2)}
    assert loaded.sites_truncated is False

    plain = await (await launch()).load(
        entry.read_bytes(), filename=str(entry), workspace=workspace, package_dir=str(folder)
    )
    assert (plain.sites, plain.sites_truncated) == ({}, False)


def test_reports_stay_bounded_whatever_the_code_raised(tmp_path: Path) -> None:
    sites = {"files": ["/w/wf.py"], "nodes": {f"n{index}": [0, index + 1] for index in range(100)}}
    result = {"manifest": {}, "stdout": {}}
    assert worker_host._with_sites(1, result, sites, 1 << 20)["sites"] == sites
    assert worker_host._with_sites(1, result, sites, 512) == dict(result, sites_truncated=True)

    many = [worker_host._record("load_error", "m", None) for _ in range(25)]
    data = worker_host._diagnostics_data(many)
    assert (len(data["diagnostics"]), data["diagnostics_truncated"]) == (20, True)

    long_line = tmp_path / "long.py"
    long_line.write_text("x = '" + "y" * 2000 + "'\n", encoding="utf-8")
    record = worker_host._record("load_error", "m" * 5000, worker_host._Spot(str(long_line), 1))
    assert record["source_line"] is None  # a cut excerpt would misplace the caret
    assert record["message"].endswith("... [904 chars dropped]")

    chain = [worker_host._Spot(f"/w/{index}.py", index + 1, module=True) for index in range(9)]
    # A long chain marks where it was cut; that is not a diagnostic left out.
    assert [(note["message"], note["file"]) for note in worker_host._notes(chain, None)] == [
        ("imported from", "/w/1.py"),
        ("imported from", "/w/2.py"),
        ("imported from", "/w/3.py"),
        ("4 more frames not shown", None),
        ("imported from", "/w/8.py"),
    ]
