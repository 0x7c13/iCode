# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Src rule: imports of optional-extra distributions must be function-scoped, plus its pins and proofs."""

from __future__ import annotations

import ast
import functools
import importlib.metadata
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.architecture._hygiene_core import _meta_guard_problem, _pins_allowlist, _qualified_name, _tree
from tests.support.ci import CI_LINUX_ONLY
from tests.support.paths import REPO_ROOT

# Platform-independent source analysis: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY

_AGENTS_PROJECT_OVERVIEW = "AGENTS.md project overview"

_OPTIONAL_IMPORT_ALLOWLIST = {
    Path("src/chrys/foundation/observability/exporters.py"): (
        "every symbol in the module subclasses an OTel SDK type, so it cannot be written without the extra; "
        "it stays safe because nothing imports it at module scope — observability/setup.py reaches it from "
        "inside a function, under the same gate that decides whether telemetry runs at all"
    ),
}


_DEPENDENCY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


_LOWER_TIER_ROOTS = tuple(Path("src/chrys") / tier for tier in ("foundation", "kernel", "service", "orchestration"))


def _normalized_distribution_name(requirement: str) -> str:
    match = _DEPENDENCY_NAME.match(requirement)
    assert match is not None
    return re.sub(r"[-_.]+", "-", match.group()).lower()


# Import roots for the distributions whose root is not just the normalized
# distribution name. This is stated rather than discovered because
# ``packages_distributions()`` only knows what the running interpreter has
# installed, and a distribution that is absent resolves to nothing — its roots
# drop out of the rule silently. That is not hypothetical: pywinpty is
# Windows-marked, so the Linux job that runs this sweep never installs it, and
# a module-scope ``import winpty`` in a lower tier would sail straight through.
# Only public roots are listed; private C-extension and mypyc build artifacts
# (``_yaml``, ``_watchdog_fsevents``, the charset-normalizer hash module) are
# not things source code imports, and their names vary by platform and wheel.
_DISTRIBUTION_IMPORT_ROOTS: Mapping[str, frozenset[str]] = {
    "agent-client-protocol": frozenset({"acp"}),
    "beautifulsoup4": frozenset({"bs4"}),
    # Qualified, because the base API and the optional SDK/exporter share a
    # namespace: subtracting at top-level granularity would cancel the whole
    # of ``opentelemetry`` out of the rule and leave the extra uncovered.
    "opentelemetry-api": frozenset({"opentelemetry"}),
    "opentelemetry-exporter-otlp-proto-grpc": frozenset({"opentelemetry.exporter"}),
    "opentelemetry-instrumentation-logging": frozenset({"opentelemetry.instrumentation"}),
    "opentelemetry-sdk": frozenset({"opentelemetry.sdk"}),
    "pillow": frozenset({"PIL"}),
    "pyjwt": frozenset({"jwt"}),
    "pytest": frozenset({"py", "pytest"}),
    "pytest-xdist": frozenset({"xdist"}),
    "python-docx": frozenset({"docx"}),
    "python-dotenv": frozenset({"dotenv"}),
    "python-pptx": frozenset({"pptx"}),
    "pywinpty": frozenset({"winpty"}),
    "pyyaml": frozenset({"yaml"}),
}


def _import_roots(distribution: str) -> frozenset[str]:
    """Return the import roots a distribution supplies, independent of install state."""
    return _DISTRIBUTION_IMPORT_ROOTS.get(distribution, frozenset({distribution.replace("-", "_")}))


@functools.cache
def _project_distributions() -> tuple[frozenset[str], frozenset[str]]:
    """Return the (base, optional) distribution names declared in pyproject."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    base = {_normalized_distribution_name(item) for item in project["dependencies"]}
    optional = {
        _normalized_distribution_name(item)
        for dependencies in project["optional-dependencies"].values()
        for item in dependencies
    }
    optional.discard(_normalized_distribution_name(project["name"]))
    return frozenset(base), frozenset(optional)


@functools.cache
def _optional_only_import_prefixes() -> frozenset[str]:
    """Resolve import prefixes supplied only by project optional dependencies."""
    base_distributions, optional_distributions = _project_distributions()
    base_prefixes = {prefix for item in base_distributions for prefix in _import_roots(item)}
    optional_prefixes = {prefix for item in optional_distributions for prefix in _import_roots(item)}
    return frozenset(optional_prefixes - base_prefixes)


def _optional_prefix_for(imported: str) -> str | None:
    """Return the optional-only prefix *imported* falls under, if any."""
    return next(
        (
            prefix
            for prefix in _optional_only_import_prefixes()
            if imported == prefix or imported.startswith(f"{prefix}.")
        ),
        None,
    )


class _ModuleScopeImportCollector(ast.NodeVisitor):
    """Collect static imports without entering runtime or type-checking scopes."""

    def __init__(self) -> None:
        self.imports: list[tuple[str, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_If(self, node: ast.If) -> None:
        if _qualified_name(node.test) in {"TYPE_CHECKING", "typing.TYPE_CHECKING"}:
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend((alias.name, node.lineno) for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None:
            self.imports.append((node.module, node.lineno))


def _source_module_parts(path: Path) -> tuple[str, ...]:
    """Return the importable module parts for one source path."""
    absolute = path if path.is_absolute() else REPO_ROOT / path
    relative = absolute.relative_to(REPO_ROOT / "src").with_suffix("")
    parts = relative.parts
    return parts[:-1] if parts and parts[-1] == "__init__" else parts


def _resolved_import_from_base(path: Path, node: ast.ImportFrom) -> str | None:
    """Resolve an absolute or relative ``from`` import to its base module."""
    if node.level == 0:
        return node.module
    module_parts = _source_module_parts(path)
    absolute = path if path.is_absolute() else REPO_ROOT / path
    package_parts = module_parts if absolute.name == "__init__.py" else module_parts[:-1]
    ascend = node.level - 1
    if ascend > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - ascend]
    if node.module is not None:
        base_parts = (*base_parts, *node.module.split("."))
    return ".".join(base_parts) or None


def _first_party_module_exists(module: str) -> bool:
    target = REPO_ROOT / "src" / Path(*module.split("."))
    return target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()


class _ResolvedModuleScopeImportCollector(_ModuleScopeImportCollector):
    """Collect module-scope imports with first-party submodules resolved."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = _resolved_import_from_base(self._path, node)
        if base is None:
            return
        self.imports.append((base, node.lineno))
        self.imports.extend(
            (candidate, node.lineno)
            for alias in node.names
            if alias.name != "*"
            if _first_party_module_exists(candidate := f"{base}.{alias.name}")
        )


def _assert_optional_extra_imports_are_function_scoped(sources: Mapping[Path, str]) -> None:
    """Keep optional extras out of import-time dependencies below the app tier."""
    violations: list[str] = []
    for path, source in sources.items():
        if not any(path.is_relative_to(root) for root in _LOWER_TIER_ROOTS):
            continue
        if path in _OPTIONAL_IMPORT_ALLOWLIST:
            continue
        collector = _ModuleScopeImportCollector()
        collector.visit(_tree(path, source))
        for imported, line in collector.imports:
            prefix = _optional_prefix_for(imported)
            if prefix is not None:
                violations.append(
                    f"{path}:{line}: optional-extra-module-scope-imports forbids module-scope import of optional-only "
                    f"package {prefix!r}; violates {_AGENTS_PROJECT_OVERVIEW} (pyproject.toml is the source of truth "
                    "for deps). Fix: move the import inside the runtime function under try/except ImportError, or "
                    "make the dependency a project base dependency"
                )
    assert violations == [], "\n".join(violations)


def _observed_public_import_roots() -> dict[str, set[str]]:
    """Group installed import roots by distribution, dropping non-source names."""
    roots: dict[str, set[str]] = {}
    for package, distributions in importlib.metadata.packages_distributions().items():
        if package.startswith("_") or "__mypyc" in package:
            continue
        for distribution in distributions:
            roots.setdefault(_normalized_distribution_name(distribution), set()).add(package)
    return roots


def test_stated_import_roots_match_the_installed_environment() -> None:
    """Catch table drift wherever the environment can still adjudicate it."""
    base, optional = _project_distributions()
    observed = _observed_public_import_roots()
    problems: list[str] = []
    for distribution in sorted(base | optional):
        if distribution not in observed:
            continue
        stated = _import_roots(distribution)
        # The environment only reports top-level roots, so a qualified prefix
        # is checked at the granularity it can actually adjudicate.
        stated_roots = {prefix.split(".", maxsplit=1)[0] for prefix in stated}
        if stated_roots != observed[distribution]:
            problems.append(
                _meta_guard_problem(
                    "_DISTRIBUTION_IMPORT_ROOTS",
                    f"{distribution} supplies {sorted(observed[distribution])} but the table resolves it to "
                    f"{sorted(stated)}",
                    "correct the _DISTRIBUTION_IMPORT_ROOTS entry (or add one) so the stated roots match",
                )
            )

    assert problems == [], "\n".join(problems)


def test_stated_import_roots_have_no_stale_entries() -> None:
    base, optional = _project_distributions()
    declared = base | optional
    problems = [
        _meta_guard_problem(
            "_DISTRIBUTION_IMPORT_ROOTS",
            f"entry {distribution} is not a declared project dependency",
            "remove the stale entry or restore the dependency in pyproject.toml",
        )
        for distribution in sorted(_DISTRIBUTION_IMPORT_ROOTS)
        if distribution not in declared
    ]

    assert problems == [], "\n".join(problems)


def test_optional_roots_survive_an_uninstalled_distribution() -> None:
    """The counterexample the environment-derived mapping used to lose.

    pywinpty is Windows-marked, so it is absent on the job that runs this
    sweep. Its import root must still be covered, or a module-scope
    ``import winpty`` below the app tier would go unnoticed.
    """
    assert "winpty" in _optional_only_import_prefixes()


@_pins_allowlist("_OPTIONAL_IMPORT_ALLOWLIST")
def test_optional_import_allowlist_entries_are_live() -> None:
    """An exempted module must still be the thing the rule would flag."""
    problems: list[str] = []
    for path in sorted(_OPTIONAL_IMPORT_ALLOWLIST):
        absolute = REPO_ROOT / path
        if not absolute.is_file():
            problems.append(
                _meta_guard_problem(
                    "_OPTIONAL_IMPORT_ALLOWLIST",
                    f"entry {path} names a missing file",
                    "remove the stale entry or update it to the live module",
                )
            )
            continue
        collector = _ModuleScopeImportCollector()
        collector.visit(ast.parse(absolute.read_text(encoding="utf-8")))
        if not any(_optional_prefix_for(imported) for imported, _line in collector.imports):
            problems.append(
                _meta_guard_problem(
                    "_OPTIONAL_IMPORT_ALLOWLIST",
                    f"entry {path} has no module-scope optional-only import, so the rule would not flag it",
                    "remove the stale entry now that the module no longer needs the exemption",
                )
            )

    assert problems == [], "\n".join(problems)


def test_optional_import_allowlist_modules_stay_lazily_reached() -> None:
    """The exemption's premise: nothing pulls these in at import time.

    A module-scope import of an exempted module anywhere in the tree would
    carry the optional dependency straight back onto the bootstrap path, which
    is the crash the rule exists to prevent.
    """
    exempted = {".".join(path.relative_to(Path("src")).with_suffix("").parts) for path in _OPTIONAL_IMPORT_ALLOWLIST}
    violations: list[str] = []
    for path in sorted((REPO_ROOT / "src" / "chrys").rglob("*.py")):
        collector = _ResolvedModuleScopeImportCollector(path)
        collector.visit(ast.parse(path.read_text(encoding="utf-8")))
        relative = path.relative_to(REPO_ROOT)
        violations.extend(
            f"{relative}:{line}: {imported!r} is exempted from optional-extra-module-scope-imports only because "
            f"nothing imports it at import time; violates {_AGENTS_PROJECT_OVERVIEW} (pyproject.toml is the source "
            "of truth for deps). Fix: move this import inside the function that gates the optional dependency"
            for imported, line in collector.imports
            if imported in exempted and relative not in _OPTIONAL_IMPORT_ALLOWLIST
        )

    assert violations == [], "\n".join(violations)


@pytest.mark.parametrize(
    "source",
    [
        "from chrys.foundation.observability import exporters\n",
        "from . import exporters\n",
    ],
    ids=["package-form", "relative-package-form"],
)
def test_optional_import_allowlist_lazy_pin_resolves_submodule_aliases(source: str) -> None:
    """Package-form imports must not hide an exempted optional module."""
    path = REPO_ROOT / "src/chrys/foundation/observability/consumer.py"
    collector = _ResolvedModuleScopeImportCollector(path)
    collector.visit(ast.parse(source))

    assert ("chrys.foundation.observability.exporters", 1) in collector.imports
