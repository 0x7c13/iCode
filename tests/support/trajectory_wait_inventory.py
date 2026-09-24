# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AST inventory for waits that can block trajectory-accounted execution."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, replace
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


def _classification(node: WaitNode) -> tuple[str, str, str]:
    """Return a conservative bootstrap classification for the signed baseline."""
    identity = node.identity.lower()
    expression = node.expression.lower()
    if node.primitive == "sleep" and expression.endswith("sleep(0)"):
        return ("C", "none", "Event-loop yielding does not block measured execution.")
    if "kernel.loop:toollooplayer._poll_continuation" in identity:
        return (
            "B",
            "unresolved_continuation_poll",
            "Continuation polling remains an explicitly known unresolved wait until its producer gains a span.",
        )
    return (
        "B",
        "unresolved_outer_container",
        "No enclosing trajectory interval is machine-proven for every call path; fail closed until reviewed.",
    )


def build_manifest(reviewed: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the baseline while retaining reviewed cases for unchanged waits.

    A case carries over only while its identity still names the same wait: an
    await inserted above shifts the ordinals, and a reviewed classification must
    not move onto the unreviewed wait that now holds its identity.
    """
    nodes = scan_wait_nodes()
    reviewed_cases = {
        (entry["identity"], entry.get("expression"), entry.get("primitive")): entry["cases"]
        for entry in (reviewed or {}).get("nodes", [])
        if isinstance(entry, dict)
        and isinstance(entry.get("identity"), str)
        and isinstance(entry.get("cases"), list)
        and entry["cases"]
    }
    entries: list[dict[str, Any]] = []
    for node in nodes:
        classification, container_rule, reason = _classification(node)
        default_cases = [
            {
                "when": "all_paths",
                "classification": classification,
                "container_rule": container_rule,
                "degradation_rule": (
                    "none"
                    if classification != "B"
                    else "Mark the containing residual Unresolved; never report it as exact."
                ),
                "reason": reason,
            }
        ]
        entries.append(
            {
                **asdict(node),
                "cases": reviewed_cases.get((node.identity, node.expression, node.primitive), default_cases),
            }
        )
    manifest = {"schema_version": 1, "nodes": entries}
    manifest["inventory_sha256"] = manifest_signature(manifest)
    return manifest


def manifest_signature(manifest: dict[str, Any]) -> str:
    """Sign stable identities, primitive shapes, and reviewed case decisions."""
    signature_input = [
        {
            "identity": entry["identity"],
            "primitive": entry["primitive"],
            "wrapper_target": entry["wrapper_target"],
            "cases": entry["cases"],
        }
        for entry in manifest.get("nodes", [])
    ]
    return hashlib.sha256(
        json.dumps(signature_input, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def manifest_drift(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Return stable-identity drift; source line movement alone is informational."""
    errors: list[str] = []
    if expected.get("schema_version") != 1:
        errors.append("manifest schema_version must be 1")
    if expected.get("inventory_sha256") != manifest_signature(expected):
        errors.append("manifest signature does not match its reviewed node cases")
    expected_nodes = {node["identity"]: node for node in expected.get("nodes", [])}
    actual_nodes = {node["identity"]: node for node in actual["nodes"]}
    missing = sorted(actual_nodes.keys() - expected_nodes.keys())
    stale = sorted(expected_nodes.keys() - actual_nodes.keys())
    if missing:
        errors.append("unclassified wait nodes: " + ", ".join(missing))
    if stale:
        errors.append("stale wait nodes: " + ", ".join(stale))
    for identity in sorted(expected_nodes.keys() & actual_nodes.keys()):
        expected_node = expected_nodes[identity]
        actual_node = actual_nodes[identity]
        for field in ("module", "qualname", "ordinal", "primitive", "expression", "wrapper_target"):
            if expected_node.get(field) != actual_node.get(field):
                errors.append(f"{identity}: {field} drifted")
        cases = expected_node.get("cases")
        if not isinstance(cases, list) or not cases:
            errors.append(f"{identity}: cases must be non-empty")
            continue
        if [case.get("when") for case in cases] != ["all_paths"]:
            errors.append(f"{identity}: cases must be the single mutually-exclusive/exhaustive all_paths case")
        for case in cases:
            if case.get("classification") not in {"A", "B", "C"}:
                errors.append(f"{identity}: invalid classification")
            if not case.get("container_rule") or not case.get("reason"):
                errors.append(f"{identity}: missing container rule or reason")
            if case.get("classification") == "B" and case.get("degradation_rule") in (None, "none"):
                errors.append(f"{identity}: B case lacks a degradation rule")
    return errors


def main() -> None:
    """Refresh source facts without erasing cases already reviewed by humans."""
    reviewed = load_manifest() if MANIFEST_PATH.exists() else None
    MANIFEST_PATH.write_text(
        json.dumps(build_manifest(reviewed), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
