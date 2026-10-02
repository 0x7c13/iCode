# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AST inventory for waits that can block trajectory-accounted execution.

The manifest (schema 2) is a change-registration gate, not a coverage proof:
a ``rules`` table of reviewed cases, then ``modules`` → function qualname → one
``[primitive, expression, rule, wrapper_target?]`` row per wait in source
order, the row position being the wait's ordinal. Refresh it with
``uv run python -m tests.support.trajectory_wait_inventory``.
"""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tests.support.paths import REPO_ROOT, SRC_ROOT

MANIFEST_PATH = REPO_ROOT / "tests" / "architecture" / "trajectory_wait_manifest.json"
_SCAN_ROOTS = tuple(SRC_ROOT / "chrys" / name for name in ("foundation", "kernel", "service", "orchestration"))
_ASYNCIO_CALLS = {
    "asyncio.gather": "gather",
    "asyncio.shield": "shield",
    "asyncio.sleep": "sleep",
    "asyncio.to_thread": "thread_pool",
    "asyncio.timeout": "timeout",
    "asyncio.wait": "multi_wait",
    "asyncio.wait_for": "timeout",
}
_METHOD_CALLS = {
    "acquire": "sync_primitive",
    "communicate": "subprocess",
    "get": "queue",
    "put": "queue",
    "run_in_executor": "thread_pool",
    "wait": "sync_primitive",
}
_FUTURE_NAME_MARKERS = ("future", "task", "pending", "settlement", "decision")


@dataclass(frozen=True)
class WaitNode:
    """One stable AST wait identity and its review-facing source facts."""

    identity: str
    module: str
    qualname: str
    ordinal: int
    primitive: str
    source_column: int
    source_line: int
    expression: str
    wrapper_target: str | None = None


@dataclass(frozen=True)
class _Candidate:
    call_target: str | None
    module: str
    qualname: str
    column: int
    line: int
    primitive: str
    expression: str
    wrapper_target: str | None = None


class _ModuleScan(ast.NodeVisitor):
    def __init__(self, *, module: str) -> None:
        self.module = module
        self.aliases: dict[str, str] = {}
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.direct: list[_Candidate] = []
        self._direct_keys: set[tuple[str, int, int, str]] = set()
        self._awaited_roots: set[int] = set()

    @property
    def qualname(self) -> str:
        names = [*self.class_stack, *self.function_stack]
        return ".".join(names) if names else "<module>"

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.aliases[alias.asname or alias.name.split(".")[0]] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = self._resolve_import_from_module(node)
        for alias in node.names:
            self.aliases[alias.asname or alias.name] = f"{module}.{alias.name}" if module else alias.name

    def _resolve_import_from_module(self, node: ast.ImportFrom) -> str:
        if node.level == 0:
            return node.module or ""
        package = self.module.split(".")[:-1]
        ascend = node.level - 1
        if ascend:
            package = package[:-ascend] if ascend <= len(package) else []
        if node.module is not None:
            package.extend(node.module.split("."))
        return ".".join(package)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if id(node) not in self._awaited_roots:
            resolved = self._resolve_call(node.func)
            primitive = _ASYNCIO_CALLS.get(resolved)
            if primitive is not None and not self._is_zero_sleep(node, resolved):
                self._add(node, primitive, call_target=resolved)
            elif isinstance(node.func, ast.Attribute) and node.func.attr in {"communicate", "wait"}:
                receiver = self._resolve_name(node.func.value).lower()
                if "proc" in receiver or "process" in receiver:
                    self._add(node, "subprocess", call_target=resolved)
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        value = node.value
        if isinstance(value, ast.Call):
            self._awaited_roots.add(id(value))
            resolved = self._resolve_call(value.func)
            primitive = _ASYNCIO_CALLS.get(resolved)
            if primitive is None and isinstance(value.func, ast.Attribute):
                primitive = _METHOD_CALLS.get(value.func.attr)
            if primitive is None:
                primitive = "awaitable"
            self._add(node, primitive, call_target=resolved)
        elif isinstance(value, ast.Name | ast.Attribute | ast.Subscript):
            self._add(node, "future" if self._looks_future(value) else "awaitable")
        else:
            self._add(node, "awaitable")
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        call_target = self._resolve_call(node.iter.func) if isinstance(node.iter, ast.Call) else None
        self._add(
            node,
            "async_iteration",
            expression=f"async for {ast.unparse(node.target)} in {ast.unparse(node.iter)}",
            call_target=call_target,
        )
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        if node.is_async:
            iterator = node.iter
            call_target = self._resolve_call(iterator.func) if isinstance(iterator, ast.Call) else None
            self._add(
                iterator,
                "async_iteration",
                expression=f"async comprehension for {ast.unparse(node.target)} in {ast.unparse(iterator)}",
                call_target=call_target,
            )
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            expression = item.context_expr
            resolved = (
                self._resolve_call(expression.func)
                if isinstance(expression, ast.Call)
                else self._resolve_name(expression)
            )
            lowered = resolved.lower()
            if resolved == "asyncio.timeout":
                primitive = "timeout"
            elif "lock" in lowered or "semaphore" in lowered or "permit" in lowered:
                primitive = "sync_primitive"
            else:
                primitive = "async_context"
            self._add(
                expression,
                primitive,
                expression=f"async with {ast.unparse(expression)}",
                call_target=resolved if isinstance(expression, ast.Call) else None,
            )
        self.generic_visit(node)

    def _add(
        self,
        node: ast.AST,
        primitive: str,
        *,
        expression: str | None = None,
        call_target: str | None = None,
    ) -> None:
        key = (self.qualname, node.lineno, node.col_offset, primitive)
        if key in self._direct_keys:
            return
        self._direct_keys.add(key)
        self.direct.append(
            _Candidate(
                call_target=call_target,
                module=self.module,
                qualname=self.qualname,
                column=node.col_offset,
                line=node.lineno,
                primitive=primitive,
                expression=expression if expression is not None else ast.unparse(node),
            )
        )

    def _resolve_name(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            parent = self._resolve_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    def _resolve_call(self, node: ast.expr) -> str:
        resolved = self._resolve_name(node)
        if resolved.startswith("self.") and self.class_stack:
            return f"{self.module}.{'.'.join(self.class_stack)}.{resolved.removeprefix('self.')}"
        if "." not in resolved and resolved not in self.aliases:
            return f"{self.module}.{resolved}"
        return resolved

    @staticmethod
    def _is_zero_sleep(call: ast.Call, resolved: str) -> bool:
        if resolved != "asyncio.sleep" or not call.args:
            return False
        value = call.args[0]
        return isinstance(value, ast.Constant) and value.value == 0

    def _looks_future(self, node: ast.expr) -> bool:
        name = self._resolve_name(node).lower()
        return any(marker in name for marker in _FUTURE_NAME_MARKERS)


def _module_name(path: Path) -> str:
    relative = path.relative_to(SRC_ROOT).with_suffix("")
    return ".".join(relative.parts)


def scan_wait_nodes() -> list[WaitNode]:
    """Return every await plus direct primitives and their transitive wrappers."""
    scans: list[_ModuleScan] = []
    for path in sorted(candidate for root in _SCAN_ROOTS for candidate in root.rglob("*.py")):
        scan = _ModuleScan(module=_module_name(path))
        scan.visit(ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix()))
        scans.append(scan)

    candidates = [candidate for scan in scans for candidate in scan.direct]
    waiting_functions = {f"{candidate.module}.{candidate.qualname}" for candidate in candidates}
    candidates = [
        replace(candidate, primitive="wrapper", wrapper_target=candidate.call_target)
        if candidate.call_target in waiting_functions
        else candidate
        for candidate in candidates
    ]

    nodes: list[WaitNode] = []
    by_function: dict[tuple[str, str], list[_Candidate]] = {}
    for candidate in candidates:
        by_function.setdefault((candidate.module, candidate.qualname), []).append(candidate)
    for (module, qualname), function_candidates in sorted(by_function.items()):
        ordered = sorted(
            function_candidates,
            key=lambda item: (item.line, item.column, item.primitive, item.expression),
        )
        for ordinal, candidate in enumerate(ordered, start=1):
            identity = f"{module}:{qualname}:{ordinal}"
            nodes.append(
                WaitNode(
                    identity=identity,
                    module=module,
                    qualname=qualname,
                    ordinal=ordinal,
                    primitive=candidate.primitive,
                    source_column=candidate.column,
                    source_line=candidate.line,
                    expression=candidate.expression,
                    wrapper_target=candidate.wrapper_target,
                )
            )
    return nodes


MANIFEST_SCHEMA_VERSION = 2
UNRESOLVED_RULE = "unresolved"
EVENT_LOOP_YIELD_RULE = "event-loop-yield"
UNRESOLVED_DEGRADATION = "Mark the containing residual Unresolved; never report it as exact."
_CASE_FIELDS = ("when", "classification", "container_rule", "degradation_rule", "reason")

# The built-in rules are owned here: a refresh rewrites their text in the
# manifest, so changing a default is a code change plus a refresh. Every other
# rule is a reviewed decision that lives only in the manifest.
DEFAULT_RULES: dict[str, dict[str, str]] = {
    UNRESOLVED_RULE: {
        "when": "all_paths",
        "classification": "B",
        "container_rule": "unresolved_outer_container",
        "degradation_rule": UNRESOLVED_DEGRADATION,
        "reason": "No enclosing trajectory interval is machine-proven for every call path; fail closed until reviewed.",
    },
    EVENT_LOOP_YIELD_RULE: {
        "when": "all_paths",
        "classification": "C",
        "container_rule": "none",
        "degradation_rule": "none",
        "reason": (
            "A zero-delay event-loop yield needs no wait interval of its own; "
            "its time stays in the enclosing interval or the residual."
        ),
    },
}

# Read-only migration of schema 1 manifests (one cases list per node): each
# reviewed case reason maps to the rule name it became.
_V1_EVENT_LOOP_YIELD_REASON = "Event-loop yielding does not block measured execution."
_V1_RULE_NAMES = {
    DEFAULT_RULES[UNRESOLVED_RULE]["reason"]: UNRESOLVED_RULE,
    _V1_EVENT_LOOP_YIELD_REASON: EVENT_LOOP_YIELD_RULE,
    (
        "Workflow persistence, answer acknowledgement or native-output draining "
        "can block without a proven enclosing trajectory interval."
    ): "workflow-blocking",
    (
        "Execution-state notification delivery and draining have no enclosing "
        "trajectory interval proven for every call path."
    ): "execution-state-notification",
}
# The schema 1 case each built-in rule replaces; any other case under a
# built-in reason was reviewed differently and cannot migrate silently.
_V1_BUILT_IN_CASES = {
    UNRESOLVED_RULE: DEFAULT_RULES[UNRESOLVED_RULE],
    EVENT_LOOP_YIELD_RULE: {**DEFAULT_RULES[EVENT_LOOP_YIELD_RULE], "reason": _V1_EVENT_LOOP_YIELD_REASON},
}


@dataclass(frozen=True)
class ManifestNode:
    """One manifest row: a wait's stable identity, its shape and its rule."""

    module: str
    qualname: str
    ordinal: int
    primitive: str
    expression: str
    rule: str
    wrapper_target: str | None = None

    @property
    def identity(self) -> str:
        return f"{self.module}:{self.qualname}:{self.ordinal}"


@dataclass(frozen=True)
class Manifest:
    """The reviewed rules and one node per wait, in scan order."""

    rules: dict[str, dict[str, str]]
    nodes: tuple[ManifestNode, ...]


def default_rule(node: WaitNode) -> str:
    """The rule a wait gets until someone reviews it."""
    if node.primitive == "sleep" and node.expression.lower().endswith("sleep(0)"):
        return EVENT_LOOP_YIELD_RULE
    return UNRESOLVED_RULE


def _carried_rules(reviewed: Manifest | None, nodes: Sequence[WaitNode | ManifestNode]) -> dict[str, str]:
    """The reviewed rule each of *nodes* keeps, by identity.

    A rule carries over only while its identity still names the same wait:
    the same expression and primitive, in a function with as many waits as
    before. An await added or removed shifts the ordinals after it, and a
    copy of a reviewed wait inserted above it would take its rule, so a
    function whose wait count changed is reopened whole.
    """
    if reviewed is None:
        return {}
    before = Counter((node.module, node.qualname) for node in reviewed.nodes)
    now = Counter((node.module, node.qualname) for node in nodes)
    rows = {node.identity: node for node in reviewed.nodes}
    return {
        node.identity: row.rule
        for node in nodes
        if (row := rows.get(node.identity)) is not None
        and (row.expression, row.primitive) == (node.expression, node.primitive)
        and before[node.module, node.qualname] == now[node.module, node.qualname]
    }


def build_manifest(reviewed: Manifest | None = None) -> Manifest:
    """Build the manifest from source, keeping reviewed rules for unchanged waits (``_carried_rules``)."""
    reviewed_rules = {} if reviewed is None else reviewed.rules
    scanned = scan_wait_nodes()
    carried = _carried_rules(reviewed, scanned)
    nodes: list[ManifestNode] = []
    for node in scanned:
        rule = carried.get(node.identity)
        if rule is None or (rule not in DEFAULT_RULES and rule not in reviewed_rules):
            rule = default_rule(node)
        nodes.append(
            ManifestNode(
                module=node.module,
                qualname=node.qualname,
                ordinal=node.ordinal,
                primitive=node.primitive,
                expression=node.expression,
                rule=rule,
                wrapper_target=node.wrapper_target,
            )
        )
    referenced = sorted({node.rule for node in nodes})
    rules = {name: dict(DEFAULT_RULES.get(name) or reviewed_rules[name]) for name in referenced}
    return Manifest(rules=rules, nodes=tuple(nodes))


def reopened_nodes(reviewed: Manifest | None, built: Manifest) -> list[tuple[str, str | None, str | None]]:
    """The waits whose reviewed rule did not carry over, so they got the default.

    Each comes with the rule its identity had in *reviewed* and the wait it
    then named, or None twice for an identity that is new: after an insertion
    the identity names another wait, so the rule belongs to that expression.
    """
    carried = _carried_rules(reviewed, built.nodes)
    before = {} if reviewed is None else {node.identity: node for node in reviewed.nodes}
    return [
        (node.identity, None, None)
        if (row := before.get(node.identity)) is None
        else (node.identity, row.rule, row.expression)
        for node in built.nodes
        if node.identity not in carried
    ]


def encode_manifest(manifest: Manifest) -> str:
    """Serialize *manifest* with one line per wait, grouped by module and function.

    Rows carry no ordinal: a function's ordinals must run 1..n, or decoding
    would renumber them.
    """
    grouped: dict[str, dict[str, list[ManifestNode]]] = {}
    for node in manifest.nodes:
        grouped.setdefault(node.module, {}).setdefault(node.qualname, []).append(node)
    for module, functions in grouped.items():
        for qualname, rows in functions.items():
            if sorted(node.ordinal for node in rows) != list(range(1, len(rows) + 1)):
                raise ValueError(f"{module}:{qualname}: ordinals are not 1..n")
    lines = ["{", f'  "schema_version": {MANIFEST_SCHEMA_VERSION},', '  "rules": {']
    rule_names = sorted(manifest.rules)
    for index, name in enumerate(rule_names):
        body = json.dumps(manifest.rules[name], indent=2, sort_keys=True).replace("\n", "\n    ")
        lines.append(f"    {json.dumps(name)}: {body}{',' if index < len(rule_names) - 1 else ''}")
    lines.extend(["  },", '  "modules": {'])
    modules = sorted(grouped)
    for module_index, module in enumerate(modules):
        lines.append(f"    {json.dumps(module)}: {{")
        functions = sorted(grouped[module])
        for function_index, qualname in enumerate(functions):
            lines.append(f"      {json.dumps(qualname)}: [")
            rows = sorted(grouped[module][qualname], key=lambda node: node.ordinal)
            for row_index, node in enumerate(rows):
                row = [node.primitive, node.expression, node.rule]
                if node.wrapper_target is not None:
                    row.append(node.wrapper_target)
                lines.append(f"        {json.dumps(row)}{',' if row_index < len(rows) - 1 else ''}")
            lines.append(f"      ]{',' if function_index < len(functions) - 1 else ''}")
        lines.append(f"    }}{',' if module_index < len(modules) - 1 else ''}")
    lines.extend(["  }", "}"])
    return "\n".join(lines) + "\n"


def _parse_json(text: str) -> tuple[Any, list[str]]:
    duplicates: list[str] = []

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    try:
        data = json.loads(text, object_pairs_hook=unique_pairs)
    except json.JSONDecodeError as exc:
        return None, [f"manifest is not valid JSON: {exc}"]
    return data, [f"duplicate JSON key {key!r}" for key in duplicates]


def _rule_errors(name: str, rule: Any) -> list[str]:
    if not isinstance(rule, dict) or set(rule) != set(_CASE_FIELDS):
        return [f"rule {name!r} must have exactly the fields {', '.join(_CASE_FIELDS)}"]
    errors: list[str] = []
    if any(not isinstance(rule[field], str) or not rule[field].strip() for field in _CASE_FIELDS):
        errors.append(f"rule {name!r}: every field must be a non-empty string")
        return errors
    if rule["when"] != "all_paths":
        errors.append(f"rule {name!r}: when must be all_paths (the single exhaustive case)")
    if rule["classification"] not in {"A", "B", "C"}:
        errors.append(f"rule {name!r}: invalid classification")
    if rule["classification"] == "B" and rule["degradation_rule"] == "none":
        errors.append(f"rule {name!r}: a B rule needs a degradation rule")
    return errors


def _row_errors(where: str, row: Any) -> list[str]:
    if not isinstance(row, list) or len(row) not in (3, 4):
        return [f"{where}: row must be [primitive, expression, rule] or [primitive, expression, rule, wrapper_target]"]
    if any(not isinstance(value, str) or not value for value in row):
        return [f"{where}: row fields must be non-empty strings"]
    if (row[0] == "wrapper") != (len(row) == 4):
        return [f"{where}: a wrapper_target goes with the wrapper primitive and nothing else"]
    return []


def decode_manifest(text: str) -> tuple[Manifest, list[str]]:
    """Strictly decode a schema 2 manifest; every structural problem is reported."""
    data, errors = _parse_json(text)
    empty = Manifest(rules={}, nodes=())
    if not isinstance(data, dict):
        return empty, errors or ["manifest must be a JSON object"]
    if set(data) != {"schema_version", "rules", "modules"}:
        errors.append("manifest must have exactly schema_version, rules and modules")
    if type(data.get("schema_version")) is not int or data["schema_version"] != MANIFEST_SCHEMA_VERSION:
        errors.append(f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION}")
    raw_rules = data.get("rules")
    raw_modules = data.get("modules")
    if not isinstance(raw_rules, dict) or not raw_rules:
        errors.append("rules must be a non-empty object")
        raw_rules = {}
    if not isinstance(raw_modules, dict) or not raw_modules:
        errors.append("modules must be a non-empty object")
        raw_modules = {}
    rules: dict[str, dict[str, str]] = {}
    for name, rule in raw_rules.items():
        if not name:
            errors.append("rule names must be non-empty")
            continue
        rule_errors = _rule_errors(name, rule)
        errors.extend(rule_errors)
        if not rule_errors:
            rules[name] = rule
    nodes: list[ManifestNode] = []
    for module, functions in raw_modules.items():
        if not module or not isinstance(functions, dict) or not functions:
            errors.append(f"module {module!r} must be a non-empty name mapping to a non-empty object")
            continue
        for qualname, rows in functions.items():
            if not qualname or not isinstance(rows, list) or not rows:
                errors.append(f"{module}:{qualname!r} must be a non-empty name mapping to a non-empty list")
                continue
            for ordinal, row in enumerate(rows, start=1):
                where = f"{module}:{qualname}:{ordinal}"
                row_errors = _row_errors(where, row)
                if row_errors:
                    errors.extend(row_errors)
                    continue
                if row[2] not in raw_rules:
                    errors.append(f"{where}: unknown rule {row[2]!r}")
                nodes.append(
                    ManifestNode(
                        module=module,
                        qualname=qualname,
                        ordinal=ordinal,
                        primitive=row[0],
                        expression=row[1],
                        rule=row[2],
                        wrapper_target=row[3] if len(row) == 4 else None,
                    )
                )
    unreferenced = sorted(set(raw_rules) - {node.rule for node in nodes})
    if unreferenced:
        errors.append("rules no wait uses: " + ", ".join(unreferenced))
    return Manifest(rules=rules, nodes=tuple(nodes)), errors


def manifest_from_v1(data: dict[str, Any]) -> Manifest:
    """Convert a schema 1 manifest; raises ValueError on anything it cannot map."""
    rules: dict[str, dict[str, str]] = {}
    nodes: list[ManifestNode] = []
    seen: set[str] = set()
    ordinals: dict[tuple[str, str], list[int]] = {}
    for entry in data.get("nodes", []):
        identity = entry["identity"]
        if identity in seen:
            raise ValueError(f"duplicate identity {identity}")
        seen.add(identity)
        if identity != f"{entry['module']}:{entry['qualname']}:{entry['ordinal']}":
            raise ValueError(f"{identity}: identity does not match module, qualname and ordinal")
        ordinals.setdefault((entry["module"], entry["qualname"]), []).append(entry["ordinal"])
        cases = entry["cases"]
        if len(cases) != 1:
            raise ValueError(f"{identity}: schema 1 nodes carry exactly one case")
        name = _V1_RULE_NAMES.get(cases[0].get("reason"))
        if name is None:
            raise ValueError(f"{identity}: no schema 2 rule name for its case; add the reason to _V1_RULE_NAMES")
        case = {field: cases[0].get(field) for field in _CASE_FIELDS}
        if not all(isinstance(value, str) and value.strip() for value in case.values()):
            raise ValueError(f"{identity}: case fields must be non-empty strings")
        if case != (_V1_BUILT_IN_CASES.get(name) or rules.setdefault(name, case)):
            raise ValueError(f"{identity}: its case does not match rule {name!r}; migrate it by hand")
        nodes.append(
            ManifestNode(
                module=entry["module"],
                qualname=entry["qualname"],
                ordinal=entry["ordinal"],
                primitive=entry["primitive"],
                expression=entry["expression"],
                rule=name,
                wrapper_target=entry["wrapper_target"],
            )
        )
    for (module, qualname), numbers in ordinals.items():
        if sorted(numbers) != list(range(1, len(numbers) + 1)):
            raise ValueError(f"{module}:{qualname}: ordinals are not 1..n")
    return Manifest(rules=rules, nodes=tuple(nodes))


def load_manifest_text() -> str:
    return MANIFEST_PATH.read_text(encoding="utf-8")


def read_reviewed(text: str) -> Manifest:
    """The reviewed manifest in *text*, schema 1 or 2; raises ValueError when unusable."""
    data, errors = _parse_json(text)
    if isinstance(data, dict) and data.get("schema_version") == 1 and not errors:
        return manifest_from_v1(data)
    manifest, errors = decode_manifest(text)
    if errors:
        raise ValueError("cannot refresh from an invalid manifest:\n" + "\n".join(errors))
    return manifest


def manifest_errors(text: str, scanned: Sequence[WaitNode]) -> list[str]:
    """Problems with the manifest in *text* against the waits *scanned* from source.

    Source line and column movement is not drift: rows carry no positions. A
    manifest that is otherwise right must be in the layout a refresh writes.
    """
    manifest, errors = decode_manifest(text)
    for name, rule in sorted(manifest.rules.items()):
        if name in DEFAULT_RULES and rule != DEFAULT_RULES[name]:
            errors.append(f"rule {name!r} differs from its built-in definition")
    expected = {node.identity: node for node in manifest.nodes}
    actual = {node.identity: node for node in scanned}
    missing = sorted(actual.keys() - expected.keys())
    stale = sorted(expected.keys() - actual.keys())
    if missing:
        errors.append("unclassified wait nodes: " + ", ".join(missing))
    if stale:
        errors.append("stale wait nodes: " + ", ".join(stale))
    for identity in sorted(expected.keys() & actual.keys()):
        row, wait = expected[identity], actual[identity]
        for field, recorded, scanned_value in (
            ("primitive", row.primitive, wait.primitive),
            ("expression", row.expression, wait.expression),
            ("wrapper_target", row.wrapper_target, wait.wrapper_target),
        ):
            if recorded != scanned_value:
                errors.append(f"{identity}: {field} drifted")
    if not errors and encode_manifest(manifest) != text:
        errors.append("manifest is not in the layout a refresh writes")
    return errors


def main() -> None:
    """Refresh source facts without erasing rules already reviewed by humans."""
    reviewed = read_reviewed(load_manifest_text()) if MANIFEST_PATH.exists() else None
    built = build_manifest(reviewed)
    MANIFEST_PATH.write_text(encode_manifest(built), encoding="utf-8")
    reopened = reopened_nodes(reviewed, built)
    sys.stdout.write(f"{len(built.nodes)} waits, {len(reopened)} reopened (new or changed; default rule applied):\n")
    for identity, rule, expression in reopened:
        sys.stdout.write(f"  {identity}" + ("" if rule is None else f" (was {rule} on `{expression}`)") + "\n")


if __name__ == "__main__":
    main()
