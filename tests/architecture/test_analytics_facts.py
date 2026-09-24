# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Guard the ingest-write / resolve-read analytics fact boundary."""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterator

import pytest

from tests.support.ci import CI_LINUX_ONLY
from tests.support.paths import REPO_ROOT, SRC_ROOT

# Platform-independent source analysis: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY

_FACTS_PATH = SRC_ROOT / "chrys" / "service" / "analytics" / "_facts.py"
_ANALYTICS_ROOT = _FACTS_PATH.parent
_INTERMEDIATE_MUTATING_METHODS = frozenset({"absorb_batch", "consume", "mark_resolved"})
_ALLOWED_PHASE_METHODS = {
    ("aggregation.py", "TrajectoryAnalyzer", "_load_opened"): _INTERMEDIATE_MUTATING_METHODS,
    ("aggregation.py", "TrajectoryAnalyzer", "refresh"): _INTERMEDIATE_MUTATING_METHODS,
}
_MUTATING_METHODS = {
    "add",
    "append",
    "clear",
    "difference_update",
    "discard",
    "extend",
    "insert",
    "intersection_update",
    "pop",
    "popitem",
    "remove",
    "reverse",
    "setdefault",
    "sort",
    "symmetric_difference_update",
    "update",
}
_CONTAINER_ACCESSORS = {"get", "items", "keys", "setdefault", "values"}


def _terminal_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _annotation_names(node: ast.expr | None) -> set[str]:
    if node is None:
        return set()
    return {
        name
        for part in ast.walk(node)
        if (name := _terminal_name(part) if isinstance(part, ast.expr) else None) is not None
    }


def _is_self_intermediate(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == "_intermediate"
    )


def _is_intermediate_reference(node: ast.expr, object_aliases: set[str]) -> bool:
    return (isinstance(node, ast.Name) and node.id in object_aliases) or _is_self_intermediate(node)


def _is_fact_reference(node: ast.expr, object_aliases: set[str], container_aliases: set[str]) -> bool:
    if _is_intermediate_reference(node, object_aliases):
        return True
    if isinstance(node, ast.Name):
        return node.id in container_aliases
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _is_fact_reference(node.value, object_aliases, container_aliases)
    if isinstance(node, ast.NamedExpr):
        return _is_fact_reference(node.value, object_aliases, container_aliases)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in _CONTAINER_ACCESSORS and _is_fact_reference(
            node.func.value,
            object_aliases,
            container_aliases,
        )
    return False


def _local_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.AST]:
    """Walk one lexical function body without leaking aliases across nested scopes."""
    pending = list(reversed(function.body))
    while pending:
        node = pending.pop()
        yield node
        children = list(ast.iter_child_nodes(node))
        for child in reversed(children):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            pending.append(child)


def _target_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        return tuple(name for item in target.elts for name in _target_names(item))
    return ()


def _is_intermediate_source(node: ast.expr, object_aliases: set[str]) -> bool:
    if isinstance(node, ast.Call) and _terminal_name(node.func) == "_Intermediate":
        return True
    if isinstance(node, ast.Name) and node.id in object_aliases:
        return True
    return _is_self_intermediate(node)


def _is_container_source(node: ast.expr, object_aliases: set[str], container_aliases: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in container_aliases
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _is_fact_reference(node.value, object_aliases, container_aliases)
    if isinstance(node, ast.NamedExpr):
        return _is_container_source(node.value, object_aliases, container_aliases)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in _CONTAINER_ACCESSORS and _is_fact_reference(
            node.func.value,
            object_aliases,
            container_aliases,
        )
    return False


def _bind_fact_aliases(
    target: ast.expr,
    value: ast.expr,
    object_aliases: set[str],
    container_aliases: set[str],
) -> bool:
    if (
        isinstance(target, (ast.List, ast.Tuple))
        and isinstance(value, (ast.List, ast.Tuple))
        and len(target.elts) == len(value.elts)
        and not any(isinstance(item, ast.Starred) for item in target.elts)
    ):
        changed = False
        for target_item, value_item in zip(target.elts, value.elts, strict=True):
            if _bind_fact_aliases(target_item, value_item, object_aliases, container_aliases):
                changed = True
        return changed

    names = _target_names(target)
    if _is_intermediate_source(value, object_aliases):
        before = len(object_aliases)
        object_aliases.update(names)
        return len(object_aliases) != before
    if _is_container_source(value, object_aliases, container_aliases):
        before = len(container_aliases)
        container_aliases.update(names)
        return len(container_aliases) != before
    return False


def _fact_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    nodes: tuple[ast.AST, ...],
) -> tuple[set[str], set[str]]:
    parameters = (
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    )
    object_aliases = {
        parameter.arg for parameter in parameters if "_Intermediate" in _annotation_names(parameter.annotation)
    }
    container_aliases: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if isinstance(node, (ast.For, ast.AsyncFor)):
                if _is_container_source(node.iter, object_aliases, container_aliases):
                    before = len(container_aliases)
                    container_aliases.update(_target_names(node.target))
                    changed |= len(container_aliases) != before
                continue
            if isinstance(node, ast.NamedExpr):
                changed |= _bind_fact_aliases(node.target, node.value, object_aliases, container_aliases)
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    changed |= _bind_fact_aliases(target, node.value, object_aliases, container_aliases)
                continue
            if isinstance(node, ast.AnnAssign) and node.value is not None:
                changed |= _bind_fact_aliases(node.target, node.value, object_aliases, container_aliases)
    return object_aliases, container_aliases


def _target_mutates_fact_state(
    target: ast.expr,
    object_aliases: set[str],
    container_aliases: set[str],
) -> bool:
    if isinstance(target, ast.Name) or _is_self_intermediate(target):
        return False
    return _is_fact_reference(target, object_aliases, container_aliases)


_COMPREHENSION_TYPES = (ast.DictComp, ast.GeneratorExp, ast.ListComp, ast.SetComp)


class _MutationVisitor(ast.NodeVisitor):
    def __init__(
        self,
        object_aliases: set[str],
        container_aliases: set[str],
        allowed_phase_methods: frozenset[str],
    ) -> None:
        self.object_aliases = object_aliases
        self.container_aliases = container_aliases
        self.function_object_aliases = object_aliases
        self.function_container_aliases = container_aliases
        self.allowed_phase_methods = allowed_phase_methods
        self.comprehension_depth = 0
        self.violations: list[tuple[int, str]] = []

    def visit(self, node: ast.AST) -> None:
        self._record_violation(node)
        super().visit(node)

    def _record_violation(self, node: ast.AST) -> None:
        targets: tuple[ast.expr, ...] = ()
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = (node.target,)
        elif isinstance(node, ast.Delete):
            targets = tuple(node.targets)
        if any(_target_mutates_fact_state(target, self.object_aliases, self.container_aliases) for target in targets):
            self.violations.append((node.lineno, ast.unparse(node)))
            return
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _INTERMEDIATE_MUTATING_METHODS
            and _is_intermediate_reference(node.value, self.object_aliases)
        ):
            if node.attr not in self.allowed_phase_methods:
                self.violations.append((node.lineno, ast.unparse(node)))
            return
        if not isinstance(node, ast.Call):
            return
        if isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATING_METHODS:
            if _is_fact_reference(node.func.value, self.object_aliases, self.container_aliases):
                self.violations.append((node.lineno, ast.unparse(node)))
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id in {"delattr", "setattr"}
            and node.args
            and _is_fact_reference(node.args[0], self.object_aliases, self.container_aliases)
        ):
            self.violations.append((node.lineno, ast.unparse(node)))

    def visit_FunctionDef(self, _node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, _node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, _node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, _node: ast.Lambda) -> None:
        return

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        _bind_fact_aliases(node.target, node.value, self.object_aliases, self.container_aliases)
        if self.comprehension_depth:
            names = _target_names(node.target)
            if _is_intermediate_source(node.value, self.object_aliases):
                self.function_object_aliases.update(names)
            elif _is_container_source(node.value, self.object_aliases, self.container_aliases):
                self.function_container_aliases.update(names)

    def _visit_comprehension(
        self,
        comprehension: ast.DictComp | ast.GeneratorExp | ast.ListComp | ast.SetComp,
    ) -> None:
        outer_object_aliases = self.object_aliases
        outer_container_aliases = self.container_aliases
        self.object_aliases = set(outer_object_aliases)
        self.container_aliases = set(outer_container_aliases)
        self.comprehension_depth += 1
        for generator in comprehension.generators:
            self.visit(generator.iter)
            target_from_facts = _is_container_source(
                generator.iter,
                self.object_aliases,
                self.container_aliases,
            )
            target_names = _target_names(generator.target)
            self.object_aliases.difference_update(target_names)
            self.container_aliases.difference_update(target_names)
            if target_from_facts:
                self.container_aliases.update(target_names)
            for condition in generator.ifs:
                self.visit(condition)

        if isinstance(comprehension, ast.DictComp):
            self.visit(comprehension.key)
            self.visit(comprehension.value)
        else:
            self.visit(comprehension.elt)
        self.comprehension_depth -= 1
        self.object_aliases = outer_object_aliases
        self.container_aliases = outer_container_aliases

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)


def _function_violations(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    allowed_phase_methods: frozenset[str] = frozenset(),
) -> list[tuple[int, str]]:
    nodes = tuple(_local_nodes(function))
    object_aliases, container_aliases = _fact_aliases(function, nodes)
    visitor = _MutationVisitor(
        object_aliases,
        container_aliases,
        allowed_phase_methods,
    )
    for statement in function.body:
        visitor.visit(statement)
    return visitor.violations


def _enclosing_class_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.ClassDef):
            return parent.name
        parent = parents.get(parent)
    return None


def _phase_method_uses(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    nodes = tuple(_local_nodes(function))
    object_aliases, _container_aliases = _fact_aliases(function, nodes)
    return [
        node.attr
        for node in nodes
        if isinstance(node, ast.Attribute)
        and node.attr in _INTERMEDIATE_MUTATING_METHODS
        and _is_intermediate_reference(node.value, object_aliases)
    ]


def _source_violations(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(textwrap.dedent(source))
    return [
        violation
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for violation in _function_violations(node)
    ]


def test_resolvers_do_not_mutate_ingested_facts() -> None:
    violations: list[str] = []
    for path in sorted(_ANALYTICS_ROOT.rglob("*.py")):
        if path == _FACTS_PATH:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        module = path.relative_to(_ANALYTICS_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            owner = _enclosing_class_name(node, parents)
            allowed_phase_methods = _ALLOWED_PHASE_METHODS.get((module, owner, node.name), frozenset())
            for line, expression in _function_violations(node, allowed_phase_methods=allowed_phase_methods):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{line}: {expression}")

    assert violations == [], (
        "Only _facts.py and the allowlisted analyzer ingest phases may mutate _Intermediate state; "
        "resolvers consume the completed fact index read-only:\n" + "\n".join(violations)
    )


def test_intermediate_phase_methods_have_exact_owners() -> None:
    uses: list[tuple[str, str | None, str, str]] = []
    for path in sorted(_ANALYTICS_ROOT.rglob("*.py")):
        if path == _FACTS_PATH:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        module = path.relative_to(_ANALYTICS_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            owner = _enclosing_class_name(node, parents)
            uses.extend((module, owner, node.name, method) for method in _phase_method_uses(node))

    expected = [
        (path, owner, function, method)
        for (path, owner, function), methods in _ALLOWED_PHASE_METHODS.items()
        for method in methods
    ]
    assert sorted(uses) == sorted(expected)


@pytest.mark.parametrize(
    "source",
    [
        """
        class Owner:
            def mutate(self):
                self._intermediate.dirty = None
        """,
        """
        class Owner:
            def mutate(self):
                intermediate = self._intermediate
                intermediate.dirty = None
        """,
        """
        def mutate(intermediate: _Intermediate):
            del intermediate.nodes["y"]
        """,
        """
        def mutate(intermediate: _Intermediate):
            intermediate.counter += 1
        """,
        """
        def mutate(intermediate: _Intermediate):
            intermediate.nodes["x"] = {}
        """,
        """
        def mutate(intermediate: _Intermediate):
            intermediate.turns.clear()
        """,
        """
        def mutate(intermediate: _Intermediate):
            alias = intermediate
            alias.x.pop()
        """,
        """
        def mutate(intermediate: _Intermediate):
            values = intermediate.rollback_ranges
            values.append((1, 2))
        """,
        """
        def mutate(intermediate: _Intermediate):
            intermediate.coverage_starts.get("coverage", []).append(endpoint)
        """,
        """
        def resolve(intermediate: _Intermediate):
            intermediate.mark_resolved()
        """,
        """
        def resolve(intermediate: _Intermediate):
            intermediate.absorb_batch(batch)
        """,
        """
        def resolve(intermediate: _Intermediate):
            scanner(intermediate.consume)
        """,
        """
        def resolve(intermediate: _Intermediate):
            for endpoints in intermediate.coverage_starts.values():
                endpoints.append(endpoint)
        """,
        """
        def resolve(intermediate: _Intermediate):
            [nodes.pop("tool.operation") for nodes in intermediate.nodes.values()]
        """,
        """
        def resolve(intermediate: _Intermediate):
            if rows := intermediate.usage_by_turn.get(turn_id):
                rows.append(sample)
        """,
        """
        def resolve(intermediate: _Intermediate):
            turns, nodes = intermediate.turns, intermediate.nodes
            nodes.clear()
        """,
        """
        def resolve(intermediate: _Intermediate):
            [selected for nodes in intermediate.nodes.values() if (selected := nodes.get("selected"))]
            selected.clear()
        """,
    ],
    ids=[
        "direct-self-attribute",
        "self-attribute-source",
        "delete",
        "augmented-assignment",
        "subscript-assignment",
        "container-mutator",
        "object-alias",
        "container-alias",
        "accessor-chain",
        "phase-method",
        "absorb-phase-method",
        "consume-phase-method",
        "iteration-alias",
        "comprehension-alias",
        "named-expression-alias",
        "parallel-assignment-alias",
        "comprehension-named-expression-alias",
    ],
)
def test_fact_mutation_guard_can_go_red(source: str) -> None:
    assert len(_source_violations(source)) == 1


def test_fact_mutation_guard_allows_reads() -> None:
    assert (
        _source_violations(
            """
            def resolve(intermediate: _Intermediate):
                turns = intermediate.turns
                return tuple(turns.values())
            """
        )
        == []
    )


def test_comprehension_aliases_do_not_leak_into_the_function_scope() -> None:
    assert (
        _source_violations(
            """
            def resolve(intermediate: _Intermediate):
                [nodes for nodes in intermediate.nodes.values()]
                nodes = []
                nodes.append(local_node)
            """
        )
        == []
    )

    assert (
        _source_violations(
            """
            def resolve(intermediate: _Intermediate):
                nodes = intermediate.nodes
                [nodes.append(local_node) for nodes in local_node_groups]
            """
        )
        == []
    )
