# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Install file-patched class members on classes that are already imported.

A ``FilePatch`` rewrites the installed module for the next process, but classes imported before
``apply_all()`` keep the members they were built with. Instead of duplicating each patched body,
a runtime patch stages the same fragments on the installed source in memory, compiles only the
members it names and sets them on the live classes, so imported aliases, subclasses and existing
instances all see them, and the file and runtime paths share one source of truth.

Members are compiled inside a same-named stand-in class, so decorators (``property`` and its
setter) and class-level assignments stage as they do in the real class body. Their globals are
the live module's, and a zero-argument ``super()`` is rebound to the live class. A class the
patch adds to a module is compiled whole and set on the module.
"""

from __future__ import annotations
import __future__

import ast
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import FunctionType, ModuleType

from chrys.foundation.patches.patcher import FilePatch

logger = logging.getLogger(__name__)


class StagedSourceDriftError(Exception):
    """The installed source no longer matches a fragment or a member the patch stages."""


@dataclass(frozen=True, slots=True)
class StagedMembers:
    """Compiled members ready to be set on their live classes."""

    replacements: tuple[tuple[type | ModuleType, str, object], ...]

    def install(self, marker: str) -> None:
        """Set every member on its live class, marking each compiled function with ``marker``.

        Members are set one by one, with nothing to roll back: drift is found while staging, so
        a caller that stages every module before installing any installs nothing on drift.

        A plain value (a class-level default or counter, or a class added to a module) is only
        added, never overwritten, so live state survives when the class or module already has it
        (an already file-patched install). A patch therefore cannot change a plain value the
        upstream class already defines: the file patch would write it, while the runtime install
        keeps the upstream value. Such a patch adds a new name instead.
        """
        for target, name, value in self.replacements:
            functions = _functions_of(value)
            if not functions and name in vars(target):
                continue
            for function in functions:
                setattr(function, marker, True)
            setattr(target, name, value)


def install_patched_members(
    module: ModuleType,
    patches: Sequence[FilePatch],
    members: Mapping[str, Sequence[str]],
    *,
    marker: str,
    label: str,
) -> bool:
    """Install ``members`` (class name → member names) of ``module`` from its patched source.

    Every compiled function carries ``marker``, so repeated calls are no-ops. Returns whether the
    members are installed. Drift is found while staging, before any member is set, so on drift
    nothing is changed and a warning is logged.
    """
    if members_installed(module, members, marker):
        return True
    try:
        staged = stage_members(module, patches, members, label=label)
    except StagedSourceDriftError as exc:
        logger.warning("Skipping Textual %s runtime patch: %s", label, exc)
        return False
    staged.install(marker)
    return True


def members_installed(module: ModuleType, members: Mapping[str, Sequence[str]], marker: str) -> bool:
    """Whether every function-bearing member of ``members`` on ``module`` already carries ``marker``."""
    functions: list[FunctionType] = []
    for class_name, names in members.items():
        live_class = vars(module).get(class_name)
        if not isinstance(live_class, type):
            return False
        for name in names:
            functions.extend(_functions_of(vars(live_class).get(name)))
    return bool(functions) and all(vars(function).get(marker, False) for function in functions)


def stage_members(
    module: ModuleType,
    patches: Sequence[FilePatch],
    members: Mapping[str, Sequence[str]],
    *,
    label: str,
    classes: Sequence[str] = (),
) -> StagedMembers:
    """Compile ``members`` of ``module`` from its installed source with ``patches`` applied.

    ``classes`` names top-level classes the patches add to the module, compiled whole. Raises
    ``StagedSourceDriftError`` when a fragment or member no longer matches; nothing is changed
    until ``StagedMembers.install``, so a patch spanning modules can stage them all first.
    """
    source = stage_patched_source(module, patches)
    return StagedMembers(tuple(_compile_members(module, source, members, label=label, classes=classes)))


def stage_patched_source(module: ModuleType, patches: Sequence[FilePatch]) -> str:
    """Return ``module``'s installed source with every fragment of ``patches`` applied."""
    if module.__file__ is None:
        raise StagedSourceDriftError(f"{module.__name__} has no source file")
    source = Path(module.__file__).read_text(encoding="utf-8")
    for patch in patches:
        if patch.new_fragment in source or any(fragment in source for fragment in patch.equivalent_fragments):
            continue
        if patch.old_fragment not in source:
            raise StagedSourceDriftError(f"fragment drifted: {patch.description}")
        source = source.replace(patch.old_fragment, patch.new_fragment, 1)
    return source


def _member_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _compile_members(
    module: ModuleType, source: str, members: Mapping[str, Sequence[str]], *, label: str, classes: Sequence[str]
) -> list[tuple[type | ModuleType, str, object]]:
    tree = ast.parse(source)
    class_nodes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    stand_ins: list[ast.stmt] = []
    for class_name in classes:
        class_node = class_nodes.get(class_name)
        if class_node is None:
            raise StagedSourceDriftError(f"class {class_name} is missing")
        stand_ins.append(class_node)
    for class_name, names in members.items():
        class_node = class_nodes.get(class_name)
        if class_node is None:
            raise StagedSourceDriftError(f"class {class_name} is missing")
        body = [node for node in class_node.body if _member_name(node) in names]
        missing = set(names) - {_member_name(node) for node in body}
        if missing:
            raise StagedSourceDriftError(f"{class_name} is missing {sorted(missing)}")
        stand_ins.append(ast.ClassDef(name=class_name, bases=[], keywords=[], body=body, decorator_list=[]))
    code = compile(
        ast.fix_missing_locations(ast.Module(body=stand_ins, type_ignores=[])),
        f"<Chrys {label} patch: {module.__name__}>",
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    namespace: dict[str, object] = {}
    # Globals stay bound to the live module; only the stand-in classes land in ``namespace``.
    exec(code, vars(module), namespace)  # noqa: S102
    replacements: list[tuple[type | ModuleType, str, object]] = [
        (module, class_name, namespace[class_name]) for class_name in classes
    ]
    for class_name, names in members.items():
        live_class = getattr(module, class_name)
        stand_in = namespace[class_name]
        if not isinstance(live_class, type) or not isinstance(stand_in, type):
            raise StagedSourceDriftError(f"{class_name} is not a class")
        for name in names:
            value = vars(stand_in)[name]
            for function in _functions_of(value):
                _rebind_class_cell(function, live_class)
            replacements.append((live_class, name, value))
    return replacements


def _functions_of(value: object) -> list[FunctionType]:
    if isinstance(value, FunctionType):
        return [value]
    if isinstance(value, property):
        return [function for function in (value.fget, value.fset, value.fdel) if isinstance(function, FunctionType)]
    if isinstance(value, staticmethod | classmethod) and isinstance(value.__func__, FunctionType):
        return [value.__func__]
    if isinstance(value, cached_property) and isinstance(value.func, FunctionType):
        return [value.func]
    return []


def _rebind_class_cell(function: FunctionType, live_class: type) -> None:
    """Point a zero-argument ``super()`` (the ``__class__`` cell) at the live class."""
    code = function.__code__
    if "__class__" in code.co_freevars and function.__closure__ is not None:
        function.__closure__[code.co_freevars.index("__class__")].cell_contents = live_class
