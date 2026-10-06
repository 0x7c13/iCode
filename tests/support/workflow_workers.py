# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Helpers for tests that spawn a workflow worker: interpreter discovery, workflow sources and the bytecode cache."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from chrys.orchestration.workflows import catalog as catalog_module
from chrys.orchestration.workflows import coordinator as coordinator_module
from chrys.orchestration.workflows import validation as validation_module

PY39_ENV = "CHRYS_PY39_INTERPRETER"
_HOMEBREW_PY39 = "/opt/homebrew/opt/python@3.9/bin/python3.9"
INTERPRETER_IDS = ("current", "py39")
"""Parametrization ids: the test interpreter and the oldest supported one."""


def py39_interpreter() -> str | None:
    """Env var, then ``python3.9`` on PATH, then the Homebrew keg."""
    configured = os.environ.get(PY39_ENV)
    if configured:
        return configured
    found = shutil.which("python3.9")
    if found:
        return found
    return _HOMEBREW_PY39 if Path(_HOMEBREW_PY39).is_file() else None


def require_py39() -> str:
    """The 3.9 interpreter: CI on Linux/Windows must provide one, elsewhere its absence skips."""
    interpreter = py39_interpreter()
    if interpreter is not None:
        return interpreter
    if os.environ.get("CI") and sys.platform in {"linux", "win32"}:
        pytest.fail(f"CI must provide a Python 3.9 interpreter via {PY39_ENV} or python3.9 on PATH.")
    pytest.skip("no Python 3.9 interpreter available")


def resolve_interpreter(name: str) -> str:
    if name == "current":
        return sys.executable
    return require_py39()


def create_venv(root: Path) -> Path:
    """A real virtual environment on the test interpreter, without pip: chrys is not installed in it."""
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(root)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    )
    return root


def share_worker_bytecode_cache(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """Point catalog previews, run admission and validation at one bytecode cache instead of each test's own.

    Every test has its own configuration directory, so the private cache would start cold and the
    worker would recompile the standard library on every launch. Workflow files sit under per-test
    paths, so their entries in a shared cache never meet.
    """

    def shared(_config_dir: Path) -> Path:
        return directory

    monkeypatch.setattr(catalog_module, "worker_bytecode_cache_dir", shared)
    monkeypatch.setattr(coordinator_module, "worker_bytecode_cache_dir", shared)
    monkeypatch.setattr(validation_module, "worker_bytecode_cache_dir", shared)


def python_workflow(definitions: str, *fns: str, title: str = "t") -> bytes:
    """A workflow file: *definitions* (must define every function in *fns*) and one python node per function.

    Nodes are named after their functions, chained in order, the first is the
    start node and the last the output.
    """
    lines = [
        "from chrys.workflows import BuilderScope, NodeContext, NodeHandle, Retry, SourceValue, Workflow",
        "from chrys.workflows import WorkflowBuilder, WorkflowValue",
        definitions,
        f"wf = WorkflowBuilder({title!r})",
    ]
    lines.extend(f"_{fn} = wf.python({fn!r}, {fn})" for fn in fns)
    lines.append(f"wf.start(_{fns[0]})")
    if len(fns) > 1:
        lines.append("wf.chain(" + ", ".join(f"_{fn}" for fn in fns) + ")")
    lines.append(f"wf.output(_{fns[-1]})")
    lines.append("workflow = wf.build()")
    return "\n".join(lines).encode("utf-8") + b"\n"


CONDITIONAL_LOOP_WORKFLOW = b"""from chrys.workflows import WorkflowBuilder
def body(scope):
    entry = scope.python('entry', lambda value: value)
    exit_node = scope.python('exit', lambda value: value)
    scope.edge(entry, exit_node, when=lambda value: True)
    return entry, exit_node
wf = WorkflowBuilder('Conditional loop [literal]')
node = wf.loop('loop', body, until=lambda value: True, max_iterations=2)
wf.start(node)
wf.output(node)
workflow = wf.build()
"""
