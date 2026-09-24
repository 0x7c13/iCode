# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Global src rule: locale-aware TUI constructors must receive a locale controller explicitly, plus its proofs."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.architecture._hygiene_core import _TUI_ROOT, _qualified_name, _tree
from tests.support.ci import CI_LINUX_ONLY

# Platform-independent source analysis: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY


# --- Global TUI locale-context propagation guard ---------------------------


def _locale_aware_tui_class_index(
    sources: Mapping[Path, str],
) -> tuple[dict[str, list[tuple[Path, int]]], list[str]]:
    """Index locale consumers and return fail-closed inheritance problems."""
    definitions: dict[str, list[tuple[Path, ast.ClassDef]]] = {}
    aliases_by_path: dict[Path, dict[str, str]] = {}
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        tree = _tree(path, source)
        definitions_in_file = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        for node in definitions_in_file:
            definitions.setdefault(node.name, []).append((path, node))
        aliases_by_path[path] = {}

    all_class_names = set(definitions)
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        tree = _tree(path, source)
        aliases_by_path[path].update(
            {
                alias.asname or alias.name: alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
                if alias.name in all_class_names
            }
        )

    locale_aware: set[str] = set()
    for name, class_definitions in definitions.items():
        for _path, node in class_definitions:
            constructor = _class_constructor(node)
            if constructor is not None and _constructor_accepts_locale_controller(constructor):
                locale_aware.add(name)

    ambiguous_inheritance: list[str] = []
    changed = True
    while changed:
        changed = False
        for name, class_definitions in definitions.items():
            if name in locale_aware:
                continue
            inherited_definitions: list[tuple[Path, ast.ClassDef]] = []
            for path, node in class_definitions:
                if _class_constructor(node) is not None:
                    continue
                base_names = {
                    base_name
                    for base in node.bases
                    if (base_name := _local_base_class_name(base, aliases_by_path[path])) is not None
                }
                if base_names.intersection(locale_aware):
                    inherited_definitions.append((path, node))
            if not inherited_definitions:
                continue
            if len(class_definitions) != 1:
                sites = ", ".join(f"{path}:{node.lineno}" for path, node in class_definitions)
                ambiguous_inheritance.append(
                    f"{name} has duplicate definitions and inherits locale context ({sites}); "
                    "class-name graph is ambiguous"
                )
                continue
            path, node = inherited_definitions[0]
            if len(node.bases) != 1:
                ambiguous_inheritance.append(
                    f"{path}:{node.lineno}: {name} inherits locale context through multiple bases; "
                    "define an explicit __init__"
                )
                continue
            locale_aware.add(name)
            changed = True

    sites = {name: [(path, node.lineno) for path, node in definitions[name]] for name in locale_aware}
    return sites, sorted(set(ambiguous_inheritance))


def _locale_aware_tui_class_sites(sources: Mapping[Path, str]) -> dict[str, list[tuple[Path, int]]]:
    """Index explicit and inherited TUI consumers of locale context."""
    sites, problems = _locale_aware_tui_class_index(sources)
    if problems:
        raise AssertionError("\n".join(problems))
    return sites


def _class_constructor(node: ast.ClassDef) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next(
        (
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__"
        ),
        None,
    )


def _constructor_accepts_locale_controller(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    return any(parameter.arg == "locale_controller" for parameter in parameters)


def _local_base_class_name(node: ast.expr, aliases: Mapping[str, str]) -> str | None:
    while isinstance(node, ast.Subscript):
        node = node.value
    qualified = _qualified_name(node)
    if not qualified:
        return None
    bare_name = qualified.rsplit(".", maxsplit=1)[-1]
    return aliases.get(bare_name, bare_name)


def _tui_class_sites(sources: Mapping[Path, str]) -> dict[str, list[tuple[Path, int]]]:
    """Index every TUI class name so bare-name matching can fail closed."""
    sites: dict[str, list[tuple[Path, int]]] = {}
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        for node in ast.walk(_tree(path, source)):
            if isinstance(node, ast.ClassDef):
                sites.setdefault(node.name, []).append((path, node.lineno))
    return sites


def _locale_aware_constructor_references(tree: ast.Module, class_names: set[str]) -> set[str]:
    """Return bare and imported aliases for locale-aware constructor names."""
    references = set(class_names)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        references.update(alias.asname or alias.name for alias in node.names if alias.name in class_names)
    return references


def _assert_tui_locale_controller_propagation_is_explicit(sources: Mapping[Path, str]) -> None:
    """Require explicit locale-context propagation at every TUI-internal constructor edge.

    This is deliberately a structural rule: it requires the
    ``locale_controller=`` keyword and rejects a literal ``None`` on any TUI
    call, but it does not claim that a dynamic expression such as
    ``getattr(...)`` is non-null at runtime. Discovery and call scanning both
    stop at ``_TUI_ROOT``; callers outside that layer are out of scope. Bare
    class-name matching catches package imports without interpreting their
    ``__init__`` re-export chain; class-name ambiguity fails closed instead of
    guessing.
    """
    class_sites, violations = _locale_aware_tui_class_index(sources)
    all_class_sites = _tui_class_sites(sources)
    conflicts = {name: all_class_sites[name] for name in class_sites if len(all_class_sites[name]) != 1}
    if conflicts:
        details = [
            f"{name}: " + ", ".join(f"{path}:{line}" for path, line in sites)
            for name, sites in sorted(conflicts.items())
        ]
        violations.append(
            "locale-aware TUI class names must be unique because the propagation guard "
            "matches bare names across package re-exports:\n" + "\n".join(details)
        )

    class_names = set(class_sites)
    for path, source in sources.items():
        if not path.is_relative_to(_TUI_ROOT):
            continue
        tree = _tree(path, source)
        references = _locale_aware_constructor_references(tree, class_names)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _qualified_name(node.func)
            keyword = next((item for item in node.keywords if item.arg == "locale_controller"), None)
            if keyword is not None and isinstance(keyword.value, ast.Constant) and keyword.value.value is None:
                violations.append(f"{path}:{node.lineno}: {callee}(...) must not pass literal locale_controller=None")
                continue
            bare_name = callee.rsplit(".", maxsplit=1)[-1]
            if callee not in references and bare_name not in class_names:
                continue
            if keyword is None:
                violations.append(f"{path}:{node.lineno}: {callee}(...) must forward locale_controller= explicitly")
    assert violations == [], "\n".join(violations)


@pytest.mark.parametrize(
    "consumer",
    [
        "from chrys.app.tui.widgets import LocalizedWidget\nwidget = LocalizedWidget()\n",
        "from chrys.app.tui import widgets\nwidget = widgets.LocalizedWidget()\n",
        ("from chrys.app.tui.widgets.localized import LocalizedWidget as Widget\nwidget = Widget()\n"),
    ],
    ids=["package-import-bare-name", "module-qualified", "constructor-alias"],
)
def test_locale_controller_guard_covers_package_imports_qualification_and_aliases(consumer: str) -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/localized.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
        ),
        Path("src/chrys/app/tui/widgets/__init__.py"): (
            "from chrys.app.tui.widgets.localized import LocalizedWidget\n"
        ),
        Path("src/chrys/app/tui/screen.py"): consumer,
    }

    with pytest.raises(AssertionError, match=r"must forward locale_controller="):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_rejects_literal_none() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/localized.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
            "\n"
            "widget = LocalizedWidget(locale_controller=None)\n"
        )
    }

    with pytest.raises(AssertionError, match="must not pass literal locale_controller=None"):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_rejects_literal_none_on_super_call() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/localized.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
            "\n"
            "class SpecializedWidget(LocalizedWidget):\n"
            "    def __init__(self):\n"
            "        super().__init__(locale_controller=None)\n"
        )
    }

    with pytest.raises(AssertionError, match=r"__init__\(\.\.\.\) must not pass literal locale_controller=None"):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_rejects_positional_propagation() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/localized.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
            "\n"
            "widget = LocalizedWidget(controller)\n"
        )
    }

    with pytest.raises(AssertionError, match="must forward locale_controller= explicitly"):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


@pytest.mark.parametrize(
    "specialization",
    [
        (
            "from chrys.app.tui.widgets.base import LocalizedWidget\n"
            "class SpecializedWidget(LocalizedWidget):\n"
            "    pass\n"
        ),
        (
            "from chrys.app.tui.widgets.base import LocalizedWidget\n"
            "class IntermediateWidget(LocalizedWidget):\n"
            "    pass\n"
            "class SpecializedWidget(IntermediateWidget):\n"
            "    pass\n"
        ),
        (
            "from chrys.app.tui.widgets.base import LocalizedWidget as BaseWidget\n"
            "class SpecializedWidget(BaseWidget):\n"
            "    pass\n"
        ),
    ],
    ids=["direct", "transitive", "base-alias"],
)
def test_locale_controller_guard_covers_inherited_constructors(specialization: str) -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/base.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
        ),
        Path("src/chrys/app/tui/widgets/specialized.py"): specialization + "widget = SpecializedWidget()\n",
    }

    with pytest.raises(AssertionError, match=r"SpecializedWidget\(\.\.\.\) must forward locale_controller="):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_accepts_explicit_keyword_for_inherited_constructor() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/base.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
        ),
        Path("src/chrys/app/tui/widgets/specialized.py"): (
            "from chrys.app.tui.widgets.base import LocalizedWidget\n"
            "class SpecializedWidget(LocalizedWidget):\n"
            "    pass\n"
            "widget = SpecializedWidget(locale_controller=controller)\n"
        ),
    }

    sites = _locale_aware_tui_class_sites(sources)
    assert "SpecializedWidget" in sites
    _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_fails_closed_on_ambiguous_multiple_inheritance() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/base.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
            "\n"
            "class Mixin:\n"
            "    pass\n"
            "\n"
            "class SpecializedWidget(Mixin, LocalizedWidget):\n"
            "    pass\n"
        )
    }

    with pytest.raises(AssertionError, match="inherits locale context through multiple bases"):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_fails_closed_on_duplicate_class_names() -> None:
    locale_aware_constructor = (
        "class LocalizedWidget:\n"
        "    def __init__(self, *, locale_controller=None):\n"
        "        self.locale_controller = locale_controller\n"
    )
    sources = {
        Path("src/chrys/app/tui/one.py"): locale_aware_constructor,
        Path("src/chrys/app/tui/two.py"): "class LocalizedWidget:\n    pass\n",
    }

    with pytest.raises(AssertionError, match="locale-aware TUI class names must be unique"):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_fails_closed_on_duplicate_inherited_class_names() -> None:
    sources = {
        Path("src/chrys/app/tui/base.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
        ),
        Path("src/chrys/app/tui/one.py"): (
            "from chrys.app.tui.base import LocalizedWidget\nclass SpecializedWidget(LocalizedWidget):\n    pass\n"
        ),
        Path("src/chrys/app/tui/two.py"): "class SpecializedWidget:\n    pass\n",
    }

    with pytest.raises(AssertionError, match="duplicate definitions and inherits locale context"):
        _assert_tui_locale_controller_propagation_is_explicit(sources)


def test_locale_controller_guard_aggregates_discovery_and_call_violations() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/localized.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
            "\n"
            "class Mixin:\n"
            "    pass\n"
            "\n"
            "class SpecializedWidget(Mixin, LocalizedWidget):\n"
            "    pass\n"
            "\n"
            "widget = LocalizedWidget()\n"
        )
    }

    with pytest.raises(AssertionError) as exc_info:
        _assert_tui_locale_controller_propagation_is_explicit(sources)

    message = str(exc_info.value)
    assert "inherits locale context through multiple bases" in message
    assert "LocalizedWidget(...) must forward locale_controller=" in message


def test_locale_controller_guard_accepts_dynamic_keyword_and_ignores_unrelated_constructors() -> None:
    sources = {
        Path("src/chrys/app/tui/widgets/localized.py"): (
            "class LocalizedWidget:\n"
            "    def __init__(self, *, locale_controller=None):\n"
            "        self.locale_controller = locale_controller\n"
            "\n"
            "class PlainWidget:\n"
            "    pass\n"
        ),
        Path("src/chrys/app/tui/screen.py"): (
            "from chrys.app.tui.widgets.localized import LocalizedWidget as Widget, PlainWidget\n"
            "widget = Widget(locale_controller=getattr(app, 'locale_controller', None))\n"
            "plain = PlainWidget()\n"
        ),
    }

    _assert_tui_locale_controller_propagation_is_explicit(sources)
