# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The Python 3.9, standard-library-only floor check shared by the worker-side and builtin-template gates."""

from __future__ import annotations

import ast
import sys

FLOOR = (3, 9)

# Modules and attributes that do not exist on the floor interpreter (or are third-party).
_BANNED_MODULES = {"tomllib", "typing_extensions"}
# The one chrys name floor code may import: the SDK the worker host injects onto sys.path itself.
INJECTED_SDK = "chrys.workflows"
_BANNED_ATTRIBUTES = {
    ("typing", "Self"),
    ("typing", "TypeAlias"),
    ("typing", "ParamSpec"),
    ("typing", "TypeGuard"),
    ("asyncio", "TaskGroup"),
    ("asyncio", "timeout"),
    ("asyncio", "Runner"),
    ("enum", "StrEnum"),
}


def violations(source: str, *, path: str = "<source>") -> list[str]:
    """Return every floor violation in *source*: syntax past the floor, chrys imports, banned names."""
    try:
        tree = ast.parse(source, filename=path, feature_version=FLOOR)
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax past Python {FLOOR[0]}.{FLOOR[1]}: {exc.msg}"]
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.extend(_module_problem(alias.name, node.lineno, path))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0:
                found.extend(_module_problem(module, node.lineno, path))
            for alias in node.names:
                if (module, alias.name) in _BANNED_ATTRIBUTES:
                    found.append(f"{path}:{node.lineno}: {module}.{alias.name} does not exist on the floor")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if (node.value.id, node.attr) in _BANNED_ATTRIBUTES:
                found.append(f"{path}:{node.lineno}: {node.value.id}.{node.attr} does not exist on the floor")
        elif isinstance(node, ast.Call) and _is_dataclass(node.func) and any(k.arg == "slots" for k in node.keywords):
            found.append(f"{path}:{node.lineno}: dataclass(slots=...) does not exist on the floor")
    return found


def _module_problem(module: str, line: int, path: str) -> list[str]:
    root = module.split(".")[0]
    if root == "chrys":
        if module == INJECTED_SDK:
            return []
        return [f"{path}:{line}: imports {module}; the worker side must not depend on chrys"]
    if root in _BANNED_MODULES:
        return [f"{path}:{line}: imports {module}, which does not exist on the floor"]
    if root not in sys.stdlib_module_names:
        return [f"{path}:{line}: imports {module}, which is not in the standard library"]
    return []


def _is_dataclass(func: ast.expr) -> bool:
    return (isinstance(func, ast.Name) and func.id == "dataclass") or (
        isinstance(func, ast.Attribute) and func.attr == "dataclass"
    )
