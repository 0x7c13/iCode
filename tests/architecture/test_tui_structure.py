# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Architecture checks for the Textual TUI package structure."""

from __future__ import annotations

import ast
import contextlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

from tests.support.ci import CI_LINUX_ONLY
from tests.support.paths import REPO_ROOT, SRC_ROOT

# Platform-independent source analysis: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY

ROOT = REPO_ROOT
SRC = SRC_ROOT / "chrys"
TUI = SRC_ROOT / "chrys" / "app" / "tui"
WIDGETS = TUI / "widgets"
CHAT_WIDGETS = WIDGETS / "chat"
SCREENS = TUI / "screens"
MAIN_SCREEN = SCREENS / "main"
TERMINAL_EMULATOR = TUI / "terminal" / "emulator"

SCREEN_BASES = {"BaseDialog", "Screen", "ModalScreen"}
AGENT_SCREEN_MODULES = {"__init__.py", "config.py", "picker.py"}
KNOWN_WIDGET_TO_SCREEN_IMPORTS: set[str] = set()
KNOWN_CHAT_HELPER_PANEL_IMPORTS: set[str] = set()
KNOWN_SCREEN_PRIVATE_IMPORTS: set[str] = set()
MAIN_SCREEN_BOUNDARY_EXEMPT_FILES = {
    "__init__.py",  # Package export re-exports MainScreen.
    "screen.py",  # Textual Screen owner.
    "view_adapter.py",  # Private Textual adapter and only extracted _screen owner.
}


@dataclass(frozen=True)
class ImportRef:
    source: Path
    target: str
    line: int
    members: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        member_suffix = f":{','.join(self.members)}" if self.members else ""
        return f"{self.source.relative_to(ROOT).as_posix()}:{self.line}:{self.target}{member_suffix}"

    def format(self) -> str:
        member_suffix = f" ({', '.join(self.members)})" if self.members else ""
        return f"{self.source.relative_to(ROOT).as_posix()}:{self.line}: imports {self.target}{member_suffix}"


@dataclass(frozen=True)
class MainScreenBoundaryViolation:
    path: Path
    line: int
    kind: str
    detail: str

    def format(self) -> str:
        return f"{self.path.relative_to(ROOT).as_posix()}:{self.line}: {self.kind}: {self.detail}"


class _ImportCollector(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.imports: list[ImportRef] = []

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_guard(node.test):
            for stmt in node.orelse:
                self.visit(stmt)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            for target, members in _resolve_relative_import_refs(self.path, node):
                self._add(target, node.lineno, members)
            return
        if node.module is None:
            return
        for target, members in _expand_import_from_refs(node.module, node.names):
            self._add(target, node.lineno, members)

    def visit_Call(self, node: ast.Call) -> None:
        if _call_name(node.func) in {"__import__", "import_module"} and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                self._add(first_arg.value, node.lineno)
        self.generic_visit(node)

    def _add(self, target: str, line: int, members: tuple[str, ...] = ()) -> None:
        self.imports.append(ImportRef(source=self.path, target=target, line=line, members=members))


def test_widgets_do_not_define_screens() -> None:
    violations: list[str] = []
    for path in sorted(WIDGETS.rglob("*.py")):
        tree = _parse(path)
        screen_base_names = _screen_base_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                if _base_name(base) in screen_base_names:
                    violations.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: {node.name} subclasses {_base_name(base)}"
                    )

    assert violations == []


def test_widgets_do_not_import_screens() -> None:
    violations = [
        ref for path in sorted(WIDGETS.rglob("*.py")) for ref in _imports(path) if _is_screen_module_target(ref.target)
    ]
    actual = sorted(ref.key for ref in violations)

    if violations:
        assert actual == sorted(KNOWN_WIDGET_TO_SCREEN_IMPORTS)
        pytest.xfail("Known widget-to-screen imports remain until the owning feature is decoupled.")


def test_chat_helper_modules_do_not_import_chat_panel() -> None:
    violations = [
        ref
        for path in sorted(CHAT_WIDGETS.rglob("*.py"))
        if path.name not in {"panel.py", "__init__.py"}
        for ref in _imports(path)
        if ref.target == "chrys.app.tui.widgets.chat.panel"
    ]
    actual = sorted(ref.key for ref in violations)

    if violations:
        assert actual == sorted(KNOWN_CHAT_HELPER_PANEL_IMPORTS)
        pytest.xfail("Known chat helper panel imports remain until migration ports replace them.")


def test_chat_replay_does_not_import_textual_or_sidebar_toc_widgets() -> None:
    disallowed_widget_modules = {
        "chrys.app.tui.widgets.sidebar.toc",
    }
    violations = [
        ref
        for ref in _imports(CHAT_WIDGETS / "replay.py")
        if ref.target.startswith("textual") or ref.target in disallowed_widget_modules
    ]

    assert [ref.format() for ref in violations] == []


def test_terminal_emulator_imports_neither_textual_nor_the_rest_of_the_tui() -> None:
    """The emulator is a model of a terminal. What draws it and what runs a program on it import it, never the reverse."""
    violations = _terminal_emulator_import_violations(
        ref for path in sorted(TERMINAL_EMULATOR.glob("*.py")) for ref in _imports(path)
    )

    assert sorted(TERMINAL_EMULATOR.glob("*.py"))
    assert [ref.format() for ref in violations] == []


@pytest.mark.parametrize(
    "source",
    [
        "from textual.strip import Strip\n",
        "import textual\n",
        "from chrys.app.tui.terminal.widget import Terminal\n",
        "from chrys.app.tui.terminal import rendering\n",
    ],
)
def test_terminal_emulator_import_detection_flags_synthetic_bad_shapes(source: str) -> None:
    path = TERMINAL_EMULATOR / "core.py"

    assert _terminal_emulator_import_violations(_imports_from_source(path, source))


def test_terminal_emulator_import_detection_allows_its_own_package_and_type_only_imports() -> None:
    source = (
        "from chrys.app.tui.terminal.emulator.pen import Pen\n"
        "from rich.cells import cell_len\n"
        "if TYPE_CHECKING:\n    from textual.geometry import Size\n"
    )

    assert _terminal_emulator_import_violations(_imports_from_source(TERMINAL_EMULATOR / "core.py", source)) == []


def _terminal_emulator_import_violations(refs: Iterable[ImportRef]) -> list[ImportRef]:
    own_package = "chrys.app.tui.terminal.emulator"
    return [
        ref
        for ref in refs
        if ref.target == "textual"
        or ref.target.startswith("textual.")
        or (
            ref.target.startswith("chrys")
            and not (ref.target == own_package or ref.target.startswith(f"{own_package}."))
        )
    ]


def test_chat_toc_model_does_not_import_textual_widgets() -> None:
    disallowed_widget_modules = {
        "chrys.app.tui.widgets.chat.messages",
        "chrys.app.tui.widgets.sidebar.toc",
    }
    violations = [
        ref
        for ref in _imports(CHAT_WIDGETS / "toc_model.py")
        if ref.target.startswith("textual") or ref.target in disallowed_widget_modules
    ]

    assert [ref.format() for ref in violations] == []


def test_no_cross_feature_imports_of_private_screen_helpers() -> None:
    violations = [ref for path in sorted(SRC.rglob("*.py")) for ref in _imports(path) if _is_private_screen_import(ref)]
    actual = sorted(ref.key for ref in violations)

    if violations:
        assert actual == sorted(KNOWN_SCREEN_PRIVATE_IMPORTS)
        pytest.xfail("Known cross-screen private-helper imports remain until helpers move out of screen packages.")


def test_screen_private_modules_are_derived_across_feature_packages() -> None:
    private_modules = _screen_private_modules()

    assert "chrys.app.tui.screens.agents.panels.basic" in private_modules
    assert "chrys.app.tui.screens.dialogs.approval.body" in private_modules
    assert "chrys.app.tui.screens.dialogs.approval.bodies.file_edit" in private_modules
    assert "chrys.app.tui.screens.dialogs.approval.mode" not in private_modules
    assert "chrys.app.tui.screens.dialogs.confirm" not in private_modules
    assert "chrys.app.tui.screens.agents.config" not in private_modules


def test_private_screen_import_detection_flags_private_members_outside_feature() -> None:
    ref = ImportRef(
        source=SCREENS / "main" / "recent_dirs.py",
        target="chrys.app.tui.screens.dialogs.file_picker",
        line=1,
        members=("_PATH_DISPLAY_MAX",),
    )

    assert _is_private_screen_import(ref)


def test_private_screen_import_detection_allows_same_feature_private_members() -> None:
    ref = ImportRef(
        source=SCREENS / "dialogs" / "file_picker.py",
        target="chrys.app.tui.screens.dialogs.file_picker",
        line=1,
        members=("_PATH_DISPLAY_MAX",),
    )

    assert not _is_private_screen_import(ref)


def test_import_collector_expands_package_alias_imports() -> None:
    refs = _imports_from_source(
        WIDGETS / "fake_widget.py",
        "from chrys.app.tui import screens\n"
        "from chrys.app.tui.screens import dialogs\n"
        "from chrys.app.tui.screens.agents.panels import memory\n",
    )

    assert {ref.target for ref in refs} == {
        "chrys.app.tui",
        "chrys.app.tui.screens",
        "chrys.app.tui.screens.dialogs",
        "chrys.app.tui.screens.agents.panels",
        "chrys.app.tui.screens.agents.panels.memory",
    }


def test_screen_module_target_includes_root_screen_package() -> None:
    assert _is_screen_module_target("chrys.app.tui.screens")
    assert _is_screen_module_target("chrys.app.tui.screens.dialogs")
    assert not _is_screen_module_target("chrys.app.tui.widgets")


def test_screen_base_names_include_textual_screen_import_aliases() -> None:
    tree = ast.parse(
        "from textual.screen import ModalScreen as Modal\n"
        "from textual.screen import Screen as TextualScreen\n"
        "from chrys.app.tui.screens.dialogs import BaseDialog as Dialog\n"
        "class Bad(Modal[None]):\n"
        "    pass\n"
        "class AlsoBad(TextualScreen):\n"
        "    pass\n"
        "class DialogBad(Dialog[None]):\n"
        "    pass\n"
    )

    screen_base_names = _screen_base_names(tree)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]

    assert _base_name(classes[0].bases[0]) in screen_base_names
    assert _base_name(classes[1].bases[0]) in screen_base_names
    assert _base_name(classes[2].bases[0]) in screen_base_names


def test_import_collector_expands_relative_package_alias_imports() -> None:
    refs = _imports_from_source(
        SCREENS / "agents" / "config.py",
        "from .panels import memory\n",
    )

    assert {ref.target for ref in refs} == {
        "chrys.app.tui.screens.agents.panels",
        "chrys.app.tui.screens.agents.panels.memory",
    }


def test_main_screen_ports_do_not_expose_concrete_textual_widgets() -> None:
    violations = _main_screen_boundary_violations(MAIN_SCREEN / "ports.py")

    assert [violation.format() for violation in violations] == []


def test_main_screen_adapter_facades_do_not_store_screen_references() -> None:
    checked = [MAIN_SCREEN / "main_screen_presenter.py", MAIN_SCREEN / "dialog_gateway.py"]

    violations = [violation for path in checked for violation in _main_screen_boundary_violations(path)]

    assert violations == []


def test_main_screen_view_adapter_is_the_extracted_screen_back_reference() -> None:
    checked = _main_screen_boundary_checked_files()
    violations = [violation for path in checked for violation in _main_screen_boundary_violations(path)]

    assert violations == []


@pytest.mark.parametrize(
    ("source", "expected_kind"),
    [
        (
            "from chrys.app.tui.screens.main.screen import MainScreen\n",
            "screen-import",
        ),
        (
            "from chrys.app.tui.screens.main.screen import _SESSIONS_BINDING\n",
            "screen-import",
        ),
        (
            "from .screen import MainScreen\n",
            "screen-import",
        ),
        (
            "from chrys.app.tui.screens.main import MainScreen\n",
            "screen-import",
        ),
        (
            (
                "from typing import TYPE_CHECKING\n"
                "if TYPE_CHECKING:\n"
                "    from chrys.app.tui.screens.main.screen import MainScreen\n"
            ),
            "screen-import",
        ),
        (
            "class Controller:\n    def __init__(self, screen):\n        self.view = screen\n",
            "screen-constructor-param",
        ),
        (
            (
                "from chrys.app.tui.screens.main.screen import MainScreen\n"
                "class Controller:\n"
                "    def __init__(self, view: MainScreen):\n"
                "        self.view = view\n"
            ),
            "screen-constructor-param",
        ),
        (
            "class Controller:\n    def __init__(self, view):\n        self._screen = view\n",
            "screen-backref",
        ),
        (
            "class Controller:\n    def handle(self):\n        s = self._screen\n        return s._agent_running\n",
            "private-screen-alias",
        ),
        (
            (
                "class Controller:\n"
                "    def __init__(self, owner: object):\n"
                "        self._owner = owner\n"
                "    def handle(self):\n"
                "        return self._owner._agent_running\n"
            ),
            "private-constructor-backref",
        ),
        (
            (
                "class Controller:\n"
                "    def __init__(self, host):\n"
                "        self._host = host\n"
                "    def handle(self):\n"
                "        host = self._host\n"
                "        host.action_quit()\n"
            ),
            "private-constructor-backref",
        ),
        (
            "class Controller:\n    def handle(self):\n        return self.query_one(object)\n",
            "query-one",
        ),
        (
            "class Facade:\n    def __getattr__(self, name):\n        return getattr(self._view, name)\n",
            "broad-facade-getattr",
        ),
        (
            (
                "from chrys.app.tui.screens.main.view_adapter import MainScreenViewAdapter\n"
                "class Facade:\n"
                "    def __init__(self, view):\n"
                "        self._view = MainScreenViewAdapter(view)\n"
            ),
            "adapter-wrap-backdoor",
        ),
        (
            (
                "from typing import Protocol\n"
                "from chrys.app.tui.widgets.chat.panel import ChatPanel\n"
                "class BadPort(Protocol):\n"
                "    def chat(self) -> ChatPanel: ...\n"
            ),
            "concrete-port-type",
        ),
    ],
)
def test_main_screen_boundary_detection_flags_synthetic_bad_shapes(source: str, expected_kind: str) -> None:
    violations = _main_screen_boundary_violations_from_source(MAIN_SCREEN / "synthetic.py", source)

    assert expected_kind in {violation.kind for violation in violations}


_MAIN_SCREEN_CONCRETE_PORT_TYPES = {
    "App",
    "ChatPanel",
    "Footer",
    "InputBar",
    "Screen",
    "SessionJsonPanel",
    "ShellPanel",
    "SidebarPanel",
    "StatusBar",
    "SuggestionList",
    "Widget",
}


def _main_screen_boundary_violations(path: Path) -> list[MainScreenBoundaryViolation]:
    return _main_screen_boundary_violations_from_source(path, path.read_text(encoding="utf-8"))


def _main_screen_boundary_checked_files() -> list[Path]:
    return sorted(path for path in MAIN_SCREEN.glob("*.py") if path.name not in MAIN_SCREEN_BOUNDARY_EXEMPT_FILES)


def _main_screen_boundary_violations_from_source(path: Path, source: str) -> list[MainScreenBoundaryViolation]:
    tree = ast.parse(source, filename=str(path))
    violations: list[MainScreenBoundaryViolation] = []
    screen_aliases: set[str] = set()
    stored_constructor_params = _stored_constructor_param_attrs(tree)
    constructor_backref_aliases: set[str] = set()

    for node in ast.walk(tree):
        if _is_main_screen_import(path, node):
            violations.append(MainScreenBoundaryViolation(path, node.lineno, "screen-import", "imports MainScreen"))
        if isinstance(node, ast.FunctionDef):
            if node.name == "__getattr__":
                violations.append(
                    MainScreenBoundaryViolation(
                        path,
                        node.lineno,
                        "broad-facade-getattr",
                        "delegates unknown attributes",
                    )
                )
            if node.name == "__init__":
                for arg in node.args.args[1:]:
                    if arg.arg == "screen" or "MainScreen" in _annotation_names(arg.annotation):
                        violations.append(
                            MainScreenBoundaryViolation(
                                path,
                                arg.lineno,
                                "screen-constructor-param",
                                f"constructor accepts {arg.arg}",
                            )
                        )
        if isinstance(node, ast.Assign):
            if _is_self_screen(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        screen_aliases.add(target.id)
            if _is_stored_constructor_param_attr(node.value, stored_constructor_params):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constructor_backref_aliases.add(target.id)
            for target in node.targets:
                if _is_self_screen(target):
                    violations.append(
                        MainScreenBoundaryViolation(path, target.lineno, "screen-backref", "stores self._screen")
                    )
        if isinstance(node, ast.Attribute):
            if _is_self_screen(node):
                violations.append(MainScreenBoundaryViolation(path, node.lineno, "screen-backref", "uses self._screen"))
            if isinstance(node.value, ast.Name) and node.value.id in screen_aliases and node.attr.startswith("_"):
                violations.append(
                    MainScreenBoundaryViolation(
                        path,
                        node.lineno,
                        "private-screen-alias",
                        f"accesses {node.value.id}.{node.attr}",
                    )
                )
            if _is_private_or_action_attr(node.attr) and (
                _is_stored_constructor_param_attr(node.value, stored_constructor_params)
                or (isinstance(node.value, ast.Name) and node.value.id in constructor_backref_aliases)
            ):
                violations.append(
                    MainScreenBoundaryViolation(
                        path,
                        node.lineno,
                        "private-constructor-backref",
                        f"accesses stored constructor backref {node.attr}",
                    )
                )
        if isinstance(node, ast.Call) and _call_name(node.func) == "query_one":
            violations.append(
                MainScreenBoundaryViolation(path, node.lineno, "query-one", "calls query_one outside adapter")
            )
        if isinstance(node, ast.Call) and _call_name(node.func) == "MainScreenViewAdapter":
            violations.append(
                MainScreenBoundaryViolation(
                    path,
                    node.lineno,
                    "adapter-wrap-backdoor",
                    "constructs MainScreenViewAdapter outside screen composition root",
                )
            )
        if isinstance(node, ast.FunctionDef):
            for annotation in (node.returns, *(arg.annotation for arg in node.args.args)):
                concrete = _annotation_names(annotation) & _MAIN_SCREEN_CONCRETE_PORT_TYPES
                for name in sorted(concrete):
                    violations.append(
                        MainScreenBoundaryViolation(
                            path,
                            node.lineno,
                            "concrete-port-type",
                            f"exposes {name}",
                        )
                    )

    return violations


def _stored_constructor_param_attrs(tree: ast.Module) -> set[str]:
    attrs: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        constructor_params = {arg.arg for arg in node.args.args[1:]}
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign) or not isinstance(child.value, ast.Name):
                continue
            if child.value.id not in constructor_params:
                continue
            for target in child.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    attrs.add(target.attr)
    return attrs


def _is_stored_constructor_param_attr(node: ast.AST, attrs: set[str]) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr in attrs
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _is_private_or_action_attr(name: str) -> bool:
    return name.startswith(("_", "action_"))


def _is_main_screen_import(path: Path, node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        refs = (
            _resolve_relative_import_refs(path, node)
            if node.level
            else _expand_import_from_refs(node.module or "", node.names)
        )
        return any(
            target == "chrys.app.tui.screens.main.screen"
            or (target == "chrys.app.tui.screens.main" and (not members or "MainScreen" in members))
            for target, members in refs
        )
    if isinstance(node, ast.Import):
        return any(alias.name == "chrys.app.tui.screens.main.screen" for alias in node.names)
    return False


def _is_self_screen(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "_screen"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _annotation_names(annotation: ast.AST | None) -> set[str]:
    if annotation is None:
        return set()
    return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", ast.unparse(annotation)))


@cache
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_screen_module_target(target: str) -> bool:
    return target == "chrys.app.tui.screens" or target.startswith("chrys.app.tui.screens.")


def _screen_base_names(tree: ast.Module) -> set[str]:
    names = set(SCREEN_BASES)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module not in {"textual.screen", "chrys.app.tui.screens.dialogs", "chrys.app.tui.screens.dialogs.base"}:
            continue
        for alias in node.names:
            if alias.name in SCREEN_BASES:
                names.add(alias.asname or alias.name)
    return names


def _is_private_screen_import(ref: ImportRef) -> bool:
    if not _is_screen_module_target(ref.target):
        return False
    if _source_screen_feature(ref.source) == _target_screen_feature(ref.target):
        return False
    return ref.target in _screen_private_modules() or any(member.startswith("_") for member in ref.members)


@cache
def _screen_private_modules() -> set[str]:
    modules = _agent_private_modules()
    for path in SCREENS.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        module = _module_name_for_path(path)
        if path.stem.startswith("_"):
            modules.add(module)
            continue
        if not _module_defines_screen(path):
            modules.add(module)
    return modules


def _agent_private_modules() -> set[str]:
    """Return feature-private ``screens.agents`` modules, including future panel moves."""
    modules: set[str] = set()
    agents_package = SCREENS / "agents"
    for path in agents_package.rglob("*.py"):
        if path.parent == agents_package and path.name in AGENT_SCREEN_MODULES:
            continue
        rel = (
            path.parent.relative_to(ROOT / "src")
            if path.name == "__init__.py"
            else path.relative_to(ROOT / "src").with_suffix("")
        )
        modules.add(".".join(rel.parts))
    return modules


@cache
def _module_defines_screen(path: Path) -> bool:
    tree = _parse(path)
    screen_base_names = _screen_base_names(tree)
    return any(
        _base_name(base) in screen_base_names
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for base in node.bases
    )


def _module_name_for_path(path: Path) -> str:
    rel = (
        path.parent.relative_to(ROOT / "src")
        if path.name == "__init__.py"
        else path.relative_to(ROOT / "src").with_suffix("")
    )
    return ".".join(rel.parts)


def _source_screen_feature(path: Path) -> str | None:
    with contextlib.suppress(ValueError):
        rel_parts = path.relative_to(SCREENS).parts
        if rel_parts:
            return rel_parts[0]
    return None


def _target_screen_feature(target: str) -> str | None:
    prefix = "chrys.app.tui.screens."
    if not target.startswith(prefix):
        return None
    remainder = target.removeprefix(prefix)
    return remainder.split(".", 1)[0] if remainder else None


@cache
def _imports(path: Path) -> tuple[ImportRef, ...]:
    collector = _ImportCollector(path)
    collector.visit(_parse(path))
    return tuple(collector.imports)


def _imports_from_source(path: Path, source: str) -> list[ImportRef]:
    collector = _ImportCollector(path)
    collector.visit(ast.parse(source, filename=str(path)))
    return collector.imports


def teardown_module() -> None:
    """Release cached source trees after this worker finishes the module."""
    _imports.cache_clear()
    _parse.cache_clear()
    _screen_private_modules.cache_clear()
    _module_defines_screen.cache_clear()


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return ""


def _resolve_relative_import_refs(path: Path, node: ast.ImportFrom) -> list[tuple[str, tuple[str, ...]]]:
    package_parts = ("chrys", *path.relative_to(SRC).parts[:-1])
    if node.level > len(package_parts):
        return []
    base_parts = package_parts[: len(package_parts) - node.level + 1]
    base = ".".join(base_parts)
    if node.module is not None:
        module = f"{base}.{node.module}" if base else node.module
        return _expand_import_from_refs(module, node.names)
    return _expand_import_from_refs(base, node.names)


def _expand_import_from_refs(module: str, aliases: list[ast.alias]) -> list[tuple[str, tuple[str, ...]]]:
    refs = {(module, tuple(alias.name for alias in aliases if alias.name != "*"))}
    for alias in aliases:
        if alias.name == "*":
            continue
        candidate = f"{module}.{alias.name}"
        if _module_path_exists(candidate):
            refs.add((candidate, ()))
    return sorted(refs)


def _module_path_exists(module: str) -> bool:
    if not module.startswith("chrys."):
        return False
    rel = Path(*module.split(".")[1:])
    return (SRC / f"{rel}.py").exists() or (SRC / rel / "__init__.py").exists()


def _is_type_checking_guard(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
        isinstance(node, ast.Attribute)
        and node.attr == "TYPE_CHECKING"
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
    )


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""
