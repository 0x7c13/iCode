# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Src rule: every subprocess spawn must pass stdin explicitly, plus its red/green proofs."""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePath, PureWindowsPath

import pytest

from tests.architecture._hygiene_core import _qualified_name, _tree
from tests.support.ci import CI_LINUX_ONLY

# Platform-independent source analysis: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY

_AGENTS_SUBPROCESS_SECTION = 'AGENTS.md "Architecture & control flow" section'

_SUBPROCESS_FUNCTIONS = {
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
}


def _subprocess_call_names(tree: ast.Module) -> set[str]:
    """Resolve supported subprocess functions through ordinary import aliases."""
    names = set(_SUBPROCESS_FUNCTIONS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {"asyncio", "subprocess"}:
                    continue
                bound = alias.asname or alias.name
                names.update(
                    f"{bound}.{qualified.rsplit('.', maxsplit=1)[1]}"
                    for qualified in _SUBPROCESS_FUNCTIONS
                    if qualified.startswith(f"{alias.name}.")
                )
        elif isinstance(node, ast.ImportFrom) and node.module in {"asyncio", "subprocess"}:
            for alias in node.names:
                qualified = f"{node.module}.{alias.name}"
                if qualified in _SUBPROCESS_FUNCTIONS:
                    names.add(alias.asname or alias.name)
    return names


def _statement_stdin_bindings(statement: ast.stmt) -> set[str]:
    """Return the mappings *statement* itself gives a stdin entry."""
    names: set[str] = set()
    if isinstance(statement, ast.Assign):
        if _dict_has_literal_stdin(statement.value):
            names.update(target.id for target in statement.targets if isinstance(target, ast.Name))
        if not _inherits_parent_stdin(statement.value):
            names.update(
                target.value.id
                for target in statement.targets
                if isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and _is_stdin_key(target.slice)
            )
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        target = statement.target
        if isinstance(target, ast.Name) and _dict_has_literal_stdin(statement.value):
            names.add(target.id)
        elif (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and _is_stdin_key(target.slice)
            and not _inherits_parent_stdin(statement.value)
        ):
            names.add(target.value.id)
    elif isinstance(statement, ast.Expr):
        call = statement.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "setdefault"
            and isinstance(call.func.value, ast.Name)
            and call.args
            and _is_stdin_key(call.args[0])
            and not (len(call.args) > 1 and _inherits_parent_stdin(call.args[1]))
        ):
            names.add(call.func.value.id)
    return names


def _unscoped_nodes(statement: ast.stmt) -> Iterator[ast.AST]:
    """Walk *statement* without entering a nested function, class, or lambda."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    stack: list[ast.AST] = [statement]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(
            child
            for child in ast.iter_child_nodes(node)
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
        )


def _statement_stdin_invalidations(statement: ast.stmt) -> set[str]:
    """Return mappings whose prior stdin evidence *statement* may destroy.

    Invalidation is the mirror of evidence and takes the opposite quantifier.
    Evidence must *dominate* the call — a default set inside a branch licenses
    nothing, because it may not run. A removal inside a branch is the reverse:
    it may run, and then the child inherits our stdin again. So this walks the
    whole statement rather than the statement alone.

    Only statically certain removals count. Reassigning ``x["stdin"]`` to a
    real handle is not one — otherwise re-defaulting inside a branch would
    read as a retraction of the default above it.
    """
    names: set[str] = set()
    for node in _unscoped_nodes(statement):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
            names.update(
                target.value.id
                for target in targets
                if isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and _is_stdin_key(target.slice)
                and _inherits_parent_stdin(node.value)
            )
        elif isinstance(node, ast.Delete):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
            names.update(
                target.value.id
                for target in node.targets
                if isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and _is_stdin_key(target.slice)
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            if (
                node.func.attr == "clear"
                or (node.func.attr == "pop" and node.args and _is_stdin_key(node.args[0]))
                or (node.func.attr == "update" and node.args and _dict_maps_stdin_to_none(node.args[0]))
            ):
                names.add(node.func.value.id)
    return names


def _dict_maps_stdin_to_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Dict) and any(
        _is_stdin_key(key) and _inherits_parent_stdin(value) for key, value in zip(node.keys, node.values, strict=True)
    )


def _nested_blocks(statement: ast.stmt) -> list[list[ast.stmt]]:
    """Return the statement lists *statement* opens, excluding nested scopes."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return []
    blocks = [
        block for field in ("body", "orelse", "finalbody") if isinstance(block := getattr(statement, field, None), list)
    ]
    if isinstance(statement, ast.Try):
        blocks.extend(handler.body for handler in statement.handlers)
    return blocks


def _stdin_evidence_by_node(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[int, frozenset[str]]:
    """Map every descendant to the mappings already given stdin where it sits.

    A bare parameter is not proof: a wrapper that forwards ``**kwargs`` it
    never defaults leaves its own callers free to omit stdin, and the sweep
    would still be green. Neither is a binding found anywhere in the body — a
    ``setdefault`` *after* the spawn, or one inside a branch that may not run,
    licenses nothing. Evidence therefore has to dominate the call: it must be
    a statement of a block enclosing the call, positioned before it. That is
    weaker than real dominance analysis and stronger than a flat scan, and it
    is what the two live wrappers actually do — one defaults at the top of the
    function, the other builds the mapping before entering the ``try`` that
    spawns.
    """
    evidence: dict[int, frozenset[str]] = {}

    def walk_block(statements: list[ast.stmt], inherited: frozenset[str]) -> None:
        available = inherited
        for statement in statements:
            for child in ast.walk(statement):
                evidence[id(child)] = available
            for block in _nested_blocks(statement):
                walk_block(block, available)
            available = (available - _statement_stdin_invalidations(statement)) | _statement_stdin_bindings(statement)

    walk_block(node.body, frozenset())
    return evidence


def _is_stdin_key(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value == "stdin"


def _dict_has_literal_stdin(node: ast.expr) -> bool:
    return isinstance(node, ast.Dict) and any(
        _is_stdin_key(key) and not _inherits_parent_stdin(value)
        for key, value in zip(node.keys, node.values, strict=True)
    )


def _inherits_parent_stdin(node: ast.expr | None) -> bool:
    """Report whether a statically known value leaves the child on our stdin.

    ``stdin=None`` is the inheriting default spelled out, not a decision, so it
    must not count as evidence. Anything the sweep cannot evaluate is left to
    the author.
    """
    return isinstance(node, ast.Constant) and node.value is None


_THREAD_OFFLOAD_FUNCTIONS = {"asyncio.to_thread"}


_PARTIAL_FUNCTIONS = {"functools.partial"}


def _thread_offload_names(tree: ast.Module) -> set[str]:
    """Resolve thread-offload helpers through ordinary import aliases."""
    names = set(_THREAD_OFFLOAD_FUNCTIONS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "asyncio":
                    names.add(f"{alias.asname or alias.name}.to_thread")
        elif isinstance(node, ast.ImportFrom) and node.module == "asyncio":
            names.update(alias.asname or alias.name for alias in node.names if alias.name == "to_thread")
    return names


def _partial_names(tree: ast.Module) -> set[str]:
    """Resolve ``functools.partial`` through ordinary import aliases."""
    names = set(_PARTIAL_FUNCTIONS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "functools":
                    names.add(f"{alias.asname or alias.name}.partial")
        elif isinstance(node, ast.ImportFrom) and node.module == "functools":
            names.update(alias.asname or alias.name for alias in node.names if alias.name == "partial")
    return names


def _spawn_site(
    node: ast.Call,
    call_names: set[str],
    offload_names: set[str],
    partial_names: set[str],
) -> ast.Call | None:
    """Return the call whose keywords reach a subprocess constructor, if any.

    Spawning through ``asyncio.to_thread`` or ``run_in_executor`` hands the
    constructor over as a value, so the visited callee is the offload helper
    and a callee-name test sees nothing. The hook worker in
    ``service/hooks/runner.py`` is spawned exactly that way — the one shape
    this rule most needs to cover.
    """
    if _qualified_name(node.func) in call_names:
        return node
    if _qualified_name(node.func) in offload_names and node.args:
        target = node.args[0]
    elif isinstance(node.func, ast.Attribute) and node.func.attr == "run_in_executor" and len(node.args) > 1:
        target = node.args[1]
    else:
        return None
    if isinstance(target, ast.Call):
        # ``partial(subprocess.Popen, ...)`` carries the keywords itself.
        if _qualified_name(target.func) in partial_names and target.args:
            inner = target.args[0]
            return target if _qualified_name(inner) in call_names else None
        return None
    return node if _qualified_name(target) in call_names else None


class _SubprocessVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        path: PurePath,
        call_names: set[str],
        offload_names: set[str],
        partial_names: set[str],
        violations: list[str],
    ) -> None:
        self._path = path
        self._call_names = call_names
        self._offload_names = offload_names
        self._partial_names = partial_names
        self._violations = violations
        self._scope_stack: list[dict[int, frozenset[str]]] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._scope_stack.append(_stdin_evidence_by_node(node))
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        spawn = _spawn_site(node, self._call_names, self._offload_names, self._partial_names)
        if spawn is not None and not _subprocess_call_has_stdin(spawn, self._scope_stack):
            self._violations.append(
                f"{self._path.as_posix()}:{node.lineno}: subprocess-explicit-stdin requires stdin= or input= on every "
                f"non-interactive subprocess; violates {_AGENTS_SUBPROCESS_SECTION} (every non-interactive "
                "subprocess MUST set stdin explicitly). Fix: pass stdin=subprocess.DEVNULL (or an intentional "
                "PIPE/input), or forward a mapping an enclosing block gives a non-None stdin entry before this call"
            )
        self.generic_visit(node)


def _assert_subprocess_stdin_is_explicit(sources: Mapping[PurePath, str]) -> None:
    """Keep non-interactive subprocesses detached from ACP's protocol stdin."""
    violations: list[str] = []
    for path, source in sources.items():
        tree = _tree(path, source)
        _SubprocessVisitor(
            path=path,
            call_names=_subprocess_call_names(tree),
            offload_names=_thread_offload_names(tree),
            partial_names=_partial_names(tree),
            violations=violations,
        ).visit(tree)
    assert violations == [], "\n".join(violations)


def _subprocess_call_has_stdin(
    node: ast.Call,
    scope_stack: Sequence[dict[int, frozenset[str]]],
) -> bool:
    if any(
        keyword.arg in {"stdin", "input"} and not _inherits_parent_stdin(keyword.value) for keyword in node.keywords
    ):
        return True
    if not scope_stack:
        return False
    dominating = scope_stack[-1].get(id(node))
    if dominating is None:
        return False
    return any(
        keyword.arg is None and isinstance(keyword.value, ast.Name) and keyword.value.id in dominating
        for keyword in node.keywords
    )


@pytest.mark.parametrize(
    "body",
    [
        "    subprocess.run(['probe'], stdin=None)\n",
        "    subprocess.run(['probe'], input=None)\n",
        "    kwargs = {'stdin': None}\n    subprocess.run(['probe'], **kwargs)\n",
    ],
)
def test_subprocess_stdin_guard_rejects_explicit_none(body: str) -> None:
    """``stdin=None`` is the inheriting default spelled out, not a decision."""
    source = "import subprocess\ndef spawn(**kwargs):\n" + body

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    assert _AGENTS_SUBPROCESS_SECTION in str(exc_info.value)


@pytest.mark.parametrize(
    "invalidation",
    [
        '    kwargs["stdin"] = None',
        "    kwargs = {}",
        '    del kwargs["stdin"]',
        '    kwargs.pop("stdin")',
        "    kwargs.clear()",
        '    saved = kwargs.pop("stdin")',
        '    kwargs.update({"stdin": None})',
        '    if cond:\n        kwargs["stdin"] = None',
        '    if cond:\n        del kwargs["stdin"]',
    ],
    ids=[
        "overwrite-none",
        "rebind",
        "delete",
        "pop",
        "clear",
        "pop-as-value",
        "update-none",
        "conditional-overwrite",
        "conditional-delete",
    ],
)
def test_subprocess_stdin_guard_kills_invalidated_evidence(invalidation: str) -> None:
    """Evidence is a *must* judgement; its retraction is a *may* judgement.

    A removal that only sometimes runs still leaves the child able to inherit
    our stdin, so it has to count even from inside a branch.
    """
    source = (
        "import subprocess\n"
        "def spawn(cond, **kwargs):\n"
        '    kwargs["stdin"] = subprocess.DEVNULL\n'
        f"{invalidation}\n"
        "    subprocess.run(['probe'], **kwargs)\n"
    )

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    assert _AGENTS_SUBPROCESS_SECTION in str(exc_info.value)


@pytest.mark.parametrize(
    "statement",
    [
        '    if cond:\n        kwargs["stdin"] = subprocess.DEVNULL',
        '    if cond:\n        kwargs["creationflags"] = 0',
        "    kwargs.update(extra())",
        '    def later():\n        kwargs.pop("stdin")',
        '    later = lambda: kwargs.pop("stdin")',
    ],
    ids=["re-default", "unrelated-key", "opaque-update", "nested-def", "lambda"],
)
def test_subprocess_stdin_guard_keeps_evidence_through_harmless_statements(statement: str) -> None:
    """Only a statically certain retraction counts, or the rule cries wolf.

    Re-defaulting inside a branch must not read as a retraction, and a removal
    parked in a callable that may never run is not one either.
    """
    source = (
        "import subprocess\n"
        "def spawn(cond, extra, **kwargs):\n"
        '    kwargs["stdin"] = subprocess.DEVNULL\n'
        f"{statement}\n"
        "    subprocess.run(['probe'], **kwargs)\n"
    )

    _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})


@pytest.mark.parametrize(
    ("label", "spawn"),
    [
        ("to_thread", "asyncio.to_thread(subprocess.Popen, ['probe'], **kwargs)"),
        ("run_in_executor", "loop.run_in_executor(None, subprocess.Popen, ['probe'], **kwargs)"),
        ("partial", "loop.run_in_executor(None, partial(subprocess.Popen, ['probe'], **kwargs))"),
    ],
)
def test_subprocess_stdin_guard_sees_through_thread_offload(label: str, spawn: str) -> None:
    """Handing the constructor to a thread pool must not hide the spawn.

    The detached hook worker is launched exactly this way, so a callee-name
    test alone would leave the rule blind to its own motivating case.
    """
    source = f"import asyncio\nimport subprocess\nfrom functools import partial\ndef go(loop, **kwargs):\n    {spawn}\n"

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    assert _AGENTS_SUBPROCESS_SECTION in str(exc_info.value), label


@pytest.mark.parametrize(
    ("import_statement", "partial_call"),
    [
        ("from functools import partial as bind", "bind"),
        ("import functools as ft", "ft.partial"),
    ],
    ids=["member-alias", "module-alias"],
)
def test_subprocess_stdin_guard_sees_partial_aliases(import_statement: str, partial_call: str) -> None:
    source = (
        "import subprocess\n"
        f"{import_statement}\n"
        "def go(loop):\n"
        f"    loop.run_in_executor(None, {partial_call}(subprocess.Popen, ['probe']))\n"
    )

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    assert _AGENTS_SUBPROCESS_SECTION in str(exc_info.value)


def test_subprocess_stdin_guard_accepts_offload_with_dominating_default() -> None:
    source = (
        "import asyncio\n"
        "import subprocess\n"
        "def go(**kwargs):\n"
        '    kwargs["stdin"] = subprocess.DEVNULL\n'
        "    try:\n"
        "        return asyncio.to_thread(subprocess.Popen, ['probe'], **kwargs)\n"
        "    finally:\n"
        "        pass\n"
    )

    _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})


def test_subprocess_stdin_guard_rejects_call_result_splat() -> None:
    source = (
        "import subprocess\n"
        "def helper():\n"
        "    return {'stdin': subprocess.DEVNULL}\n"
        "subprocess.run(['probe'], **helper())\n"
    )

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    message = str(exc_info.value)
    assert "src/chrys/foundation/bad_probe.py:4" in message
    assert _AGENTS_SUBPROCESS_SECTION in message
    assert "Fix:" in message


def test_subprocess_stdin_guard_names_a_violation_by_its_posix_path() -> None:
    """A Windows checkout hands the guard backslashed paths; the report must read the same everywhere."""
    source = "import subprocess\nsubprocess.run(['probe'], check=False)\n"

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({PureWindowsPath("src/chrys/foundation/bad_probe.py"): source})

    assert "src/chrys/foundation/bad_probe.py:2" in str(exc_info.value)


def test_subprocess_stdin_guard_rejects_undefaulted_kwargs_forwarding() -> None:
    """A wrapper that only forwards kwargs proves nothing about its callers."""
    source = (
        "import asyncio\n"
        "async def wrapper(*args, **kwargs):\n"
        "    return await asyncio.create_subprocess_exec(*args, **kwargs)\n"
    )

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    message = str(exc_info.value)
    assert "src/chrys/foundation/bad_probe.py:3" in message
    assert _AGENTS_SUBPROCESS_SECTION in message


@pytest.mark.parametrize(
    ("label", "body"),
    [
        (
            "binding after the spawn",
            (
                "    proc = await asyncio.create_subprocess_exec(*args, **kwargs)\n"
                '    kwargs.setdefault("stdin", asyncio.subprocess.DEVNULL)\n'
                "    return proc\n"
            ),
        ),
        (
            "binding inside a branch",
            (
                "    if args:\n"
                '        kwargs.setdefault("stdin", asyncio.subprocess.DEVNULL)\n'
                "    return await asyncio.create_subprocess_exec(*args, **kwargs)\n"
            ),
        ),
        (
            "binding inside an unexecuted lambda",
            (
                '    later = lambda: kwargs.setdefault("stdin", asyncio.subprocess.DEVNULL)\n'
                "    return await asyncio.create_subprocess_exec(*args, **kwargs)\n"
            ),
        ),
    ],
)
def test_subprocess_stdin_guard_requires_the_binding_to_dominate(label: str, body: str) -> None:
    """Evidence gathered anywhere in the body would accept code that never runs."""
    source = "import asyncio\nasync def wrapper(*args, **kwargs):\n" + body

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})

    assert _AGENTS_SUBPROCESS_SECTION in str(exc_info.value), label


@pytest.mark.parametrize(
    "binding",
    [
        'kwargs.setdefault("stdin", asyncio.subprocess.DEVNULL)',
        'kwargs["stdin"] = asyncio.subprocess.DEVNULL',
    ],
)
def test_subprocess_stdin_guard_accepts_forwarding_that_defaults_stdin(binding: str) -> None:
    """The splat is proof once the wrapper is seen binding stdin itself."""
    source = (
        "import asyncio\n"
        "async def wrapper(*args, **kwargs):\n"
        f"    {binding}\n"
        "    return await asyncio.create_subprocess_exec(*args, **kwargs)\n"
    )

    _assert_subprocess_stdin_is_explicit({Path("src/chrys/foundation/bad_probe.py"): source})
