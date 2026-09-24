# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Standalone guard: every CLI-dispatched runtime module bootstraps the runtime or is explicitly exempt."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.architecture._hygiene_core import _qualified_name, _src_sources, _tree
from tests.support.ci import CI_LINUX_ONLY

# Platform-independent source analysis: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY

_AGENTS_BOOTSTRAP_SECTION = 'AGENTS.md "Architecture & control flow" section'

_ENTRYPOINT_BOOTSTRAP_EXEMPT = {
    Path("src/chrys/app/cli/serve.py"): "spawns the TUI child, which performs bootstrap_runtime itself",
    Path("src/chrys/app/installer.py"): (
        "copies the PyApp binary and edits PATH without touching sessions, config, or the agent runtime; "
        "bootstrapping would be dead weight and would let a broken config block the installer"
    ),
}


def _function_import_bindings(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    bindings: dict[str, str] = {}

    class ImportVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
            if child is node:
                self.generic_visit(child)

        def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
            if child is node:
                self.generic_visit(child)

        def visit_ClassDef(self, child: ast.ClassDef) -> None:
            return

        def visit_ImportFrom(self, child: ast.ImportFrom) -> None:
            if child.module is None:
                return
            for alias in child.names:
                bindings[alias.asname or alias.name] = child.module

    ImportVisitor().visit(node)
    return bindings


def _is_bootstrap_entrypoint_module(module: str) -> bool:
    """Recognise an app-tier module a dispatch helper hands control to.

    Restricting this to ``chrys.app.cli.*`` dropped ``chrys install``, which
    dispatches into chrys.app.installer — a real command that was therefore
    neither checked nor exempted.
    """
    return module.startswith("chrys.app.")


def _entrypoint_dispatch_paths(source: str) -> set[Path]:
    """Resolve the runtime modules ``app.py::main`` dispatches to.

    This reads the two dispatch shapes the CLI actually uses today: ``main``
    returning a call into an imported ``chrys.app.cli.*`` module, and ``main``
    returning a ``_run_*`` helper that calls into an app-tier module. A command
    written some third way would not be discovered here — following arbitrary
    call graphs was weighed and rejected as more machinery than the dispatcher
    warrants while that convention holds. Adding a dispatch shape means
    teaching this function about it.
    """
    tree = ast.parse(source, filename="src/chrys/app/cli/app.py")
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    main = functions.get("main")
    if main is None:
        return set()

    dispatched_modules: set[str] = set()
    helpers: set[str] = set()
    main_bindings = _function_import_bindings(main)
    for node in ast.walk(main):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Call):
            continue
        called = _qualified_name(node.value.func)
        if called.startswith("_run_"):
            helpers.add(called)
            continue
        module = main_bindings.get(called)
        if module is not None and module.startswith("chrys.app.cli."):
            dispatched_modules.add(module)

    for helper_name in helpers:
        helper = functions.get(helper_name)
        if helper is None:
            continue
        bindings = _function_import_bindings(helper)
        for node in ast.walk(helper):
            if not isinstance(node, ast.Call):
                continue
            module = bindings.get(_qualified_name(node.func))
            if module is not None and _is_bootstrap_entrypoint_module(module):
                dispatched_modules.add(module)

    return {Path("src") / Path(*module.split(".")).with_suffix(".py") for module in dispatched_modules}


def _calls_bootstrap_runtime(path: Path, source: str) -> bool:
    return any(
        isinstance(node, ast.Call) and _qualified_name(node.func).rsplit(".", maxsplit=1)[-1] == "bootstrap_runtime"
        for node in ast.walk(_tree(path, source))
    )


def _uses_shared_headless_bootstrap(path: Path, source: str, sources: dict[Path, str]) -> bool:
    """Follow the shared headless preparation edge without exempting its callers."""
    tree = _tree(path, source)
    delegates = {
        f"{alias.asname or alias.name}.prepare_runtime"
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "chrys.app.cli"
        for alias in node.names
        if alias.name == "headless"
    }
    if not any(isinstance(node, ast.Call) and _qualified_name(node.func) in delegates for node in ast.walk(tree)):
        return False
    helper_path = Path("src/chrys/app/cli/headless.py")
    helper_source = sources.get(helper_path)
    if helper_source is None:
        return False
    return any(
        isinstance(node, ast.FunctionDef)
        and node.name == "prepare_runtime"
        and _calls_bootstrap_runtime(helper_path, ast.unparse(node))
        for node in _tree(helper_path, helper_source).body
    )


@pytest.mark.parametrize(
    ("caller", "helper", "expected"),
    [
        (
            "from chrys.app.cli import headless\nheadless.prepare_runtime()",
            "def prepare_runtime(): bootstrap_runtime()",
            True,
        ),
        (
            "from chrys.app.cli import headless as runtime\nruntime.prepare_runtime()",
            "def prepare_runtime(): bootstrap_runtime()",
            True,
        ),
        ("from chrys.app.cli import headless", "def prepare_runtime(): bootstrap_runtime()", False),
        ("headless.prepare_runtime()", "def prepare_runtime(): bootstrap_runtime()", False),
        ("from chrys.app.cli import headless\nheadless.prepare_runtime()", "def prepare_runtime(): pass", False),
        (
            "from chrys.app.cli import headless\nheadless.prepare_runtime()",
            "def unrelated(): bootstrap_runtime()",
            False,
        ),
        ("from chrys.app.cli import headless\nheadless.prepare_runtime()", None, False),
    ],
)
def test_shared_headless_bootstrap_requires_the_live_preparation_edge(
    caller: str, helper: str | None, expected: bool
) -> None:
    sources = {} if helper is None else {Path("src/chrys/app/cli/headless.py"): helper}
    assert _uses_shared_headless_bootstrap(Path("src/chrys/app/cli/run.py"), caller, sources) is expected


def test_entrypoint_bootstrap_completeness_and_exemptions_are_live() -> None:
    """Every CLI-dispatched runtime module must bootstrap or remain explicitly exempt."""
    sources = _src_sources()
    dispatcher_path = Path("src/chrys/app/cli/app.py")
    dispatcher_source = sources.get(dispatcher_path)
    if dispatcher_source is None:
        pytest.fail(
            f"{dispatcher_path}:1: entrypoint-bootstrap-completeness cannot find the dispatcher; violates "
            f"{_AGENTS_BOOTSTRAP_SECTION} (every entrypoint calls bootstrap_runtime). Fix: restore app/cli/app.py "
            "or update this guard to the real dispatcher"
        )
    dispatched_paths = _entrypoint_dispatch_paths(dispatcher_source)
    problems: list[str] = []

    for path, reason in _ENTRYPOINT_BOOTSTRAP_EXEMPT.items():
        if path not in sources:
            problems.append(
                f"{path}:1: entrypoint bootstrap exemption ({reason}) names no real file; violates the "
                "AGENTS.md testing-rules section (\"A guard that can't go red is worse than none — it is "
                'believed."). Fix: remove the stale exemption or point it at the live dispatched module'
            )
        elif path not in dispatched_paths:
            problems.append(
                f"{path}:1: entrypoint bootstrap exemption ({reason}) is not reachable from app.py::main; violates "
                "the AGENTS.md testing-rules section (\"A guard that can't go red is worse than none — it is "
                'believed."). Fix: remove the stale exemption or restore the real dispatch edge'
            )

    for path in sorted(dispatched_paths):
        source = sources.get(path)
        if source is None:
            problems.append(
                f"{path}:1: app.py::main dispatches to a module with no source file; violates "
                f"{_AGENTS_BOOTSTRAP_SECTION} (every entrypoint calls bootstrap_runtime). Fix: restore the "
                "dispatched module or correct the dispatcher import"
            )
        elif (
            path not in _ENTRYPOINT_BOOTSTRAP_EXEMPT
            and not _calls_bootstrap_runtime(path, source)
            and not _uses_shared_headless_bootstrap(path, source, sources)
        ):
            problems.append(
                f"{path}:1: entrypoint-bootstrap-completeness found no bootstrap_runtime() call; violates "
                f"{_AGENTS_BOOTSTRAP_SECTION} (every entrypoint calls bootstrap_runtime; never duplicate). Fix: "
                "call bootstrap_runtime through the module's runtime-preparation path, or add a narrowly "
                "justified live exemption"
            )

    _tree.cache_clear()
    assert problems == [], "\n".join(problems)
