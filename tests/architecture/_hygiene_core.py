# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared source-scanning primitives for the architecture hygiene rule modules."""

from __future__ import annotations

import ast
import functools
import zlib
from collections.abc import Callable
from pathlib import Path, PurePath

from tests.support.paths import REPO_ROOT, SRC_ROOT, TESTS_ROOT

# Shard count for the sweep test. One monolithic sweep measured 53-55s of
# worker time on contended CI draws — 90% of the global 60s per-test timeout —
# and was the suite's #1 tail pole on every platform. Shards are disjoint by
# construction (stable path hash modulo), so every file is checked exactly once
# and worksteal can spread the shards across workers.
_SWEEP_SHARDS = 4


# An explicit decorator registry ties each allowlist to an executable pin.
# Source-text matching is shorter, but a comment or dead reference could satisfy
# it; the registry adds one declaration per pin in exchange for structural proof.
_ALLOWLIST_PIN_REGISTRY: dict[str, list[Callable[..., None]]] = {}


def _pins_allowlist(name: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    def _register(pin: Callable[..., None]) -> Callable[..., None]:
        _ALLOWLIST_PIN_REGISTRY.setdefault(name, []).append(pin)
        return pin

    return _register


def _shard_of(relative_path: Path) -> int:
    """Stable shard assignment: seed-independent, identical on every platform."""
    return zlib.crc32(relative_path.as_posix().encode("utf-8")) % _SWEEP_SHARDS


def _test_sources(shard: int | None = None) -> dict[Path, str]:
    relative = ((path, path.relative_to(REPO_ROOT)) for path in sorted(TESTS_ROOT.rglob("*.py")))
    return {rel: path.read_text(encoding="utf-8") for path, rel in relative if shard is None or _shard_of(rel) == shard}


def _src_sources(shard: int | None = None) -> dict[Path, str]:
    relative = ((path, path.relative_to(REPO_ROOT)) for path in sorted((SRC_ROOT / "chrys").rglob("*.py")))
    return {rel: path.read_text(encoding="utf-8") for path, rel in relative if shard is None or _shard_of(rel) == shard}


_ARCHITECTURE_DIR = Path(__file__).resolve().parent


def _rule_module_paths() -> tuple[Path, ...]:
    """Every hygiene family module on disk, by naming convention.

    The entry module pins its ``_RULE_MODULES`` tuple against this listing, so
    a new family file cannot be added without joining the meta-guards' scan.
    """
    return tuple(sorted(_ARCHITECTURE_DIR.glob("test_hygiene_*.py")))


@functools.cache
def _architecture_definitions() -> dict[Path, dict[str, int]]:
    """Top-level definition lines of every architecture module, by module path.

    The hygiene rules and their allowlists live in sibling family modules, so a
    meta-guard message must resolve a name to the module that DEFINES it rather
    than assume this file. Parsing the directory (instead of importing it) also
    lets the entry module's meta-guards see a rule or allowlist that was never
    imported anywhere — the exact silent-unenforcement failure they exist for.
    """
    definitions: dict[Path, dict[str, int]] = {}
    for module_path in sorted(_ARCHITECTURE_DIR.glob("*.py")):
        if module_path.name == "__init__.py":
            continue
        relative = module_path.relative_to(REPO_ROOT)
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=relative.as_posix())
        module_lines: dict[str, int] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                module_lines[node.name] = node.lineno
                continue
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    module_lines[target.id] = node.lineno
        definitions[relative] = module_lines
    return definitions


@functools.cache
def _definition_sites() -> dict[str, tuple[Path, int]]:
    """Flatten the per-module definitions into ``name -> (module, line)``.

    Every rule, allowlist and helper name is defined in exactly one family
    module, so first-writer-wins over a sorted scan is deterministic.
    """
    sites: dict[str, tuple[Path, int]] = {}
    for module_path, module_lines in sorted(_architecture_definitions().items()):
        for name, line in module_lines.items():
            sites.setdefault(name, (module_path, line))
    return sites


def _definition_location(name: str) -> str:
    definition_sites = _definition_sites()
    module_path, line = definition_sites.get(name, definition_sites["_ALLOWLIST_PIN_REGISTRY"])
    return f"{module_path.as_posix()}:{line}"


def _meta_guard_problem(name: str, problem: str, fix: str) -> str:
    return (
        f"{_definition_location(name)}: {problem}; violates the AGENTS.md testing-rules section "
        '("A guard that can\'t go red is worse than none — it is believed."). '
        f"Fix: {fix}"
    )


@functools.cache
def _allowlist_target_trees(relative_paths: frozenset[Path]) -> dict[Path, ast.Module]:
    """Read and parse one pinned allowlist's target files exactly once.

    The pinned allowlists now live beside the rules they justify, so the target
    set arrives as an argument instead of being collected from module globals.
    """
    trees: dict[Path, ast.Module] = {}
    for relative_path in sorted(relative_paths):
        absolute_path = REPO_ROOT / relative_path
        if absolute_path.is_file():
            trees[relative_path] = ast.parse(
                absolute_path.read_text(encoding="utf-8"),
                filename=relative_path.as_posix(),
            )
    return trees


@functools.cache
def _tree(path: PurePath, source: str) -> ast.Module:
    return ast.parse(source, filename=path.as_posix())


def _qualified_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


# Generic AST predicates shared by several rule families.
def _literal_string(node: ast.expr) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


_SCOPE_BOUNDARY_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _resolved_import_module(path: Path, node: ast.ImportFrom) -> str | None:
    """Resolve a possibly-relative import to its absolute module name."""
    if node.level == 0:
        return node.module
    parts = list(path.with_suffix("").parts)
    if "src" in parts:
        parts = parts[parts.index("src") + 1 :]
    package = parts[:-1]
    ascent = node.level - 1
    if ascent > len(package):
        return None
    base = package[: len(package) - ascent] if ascent else package
    prefix = ".".join(base)
    if node.module:
        return f"{prefix}.{node.module}" if prefix else node.module
    return prefix or None


_TUI_ROOT = Path("src/chrys/app/tui")
