# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Src rule: outside the reminder middleware, only reviewed ``middleware.sources`` members are used.

A reminder source's other members are turn transitions the middleware drives
at fixed points (``prepare_turn``, request observers): draining the
file-change notice, capturing the archive pointer, arming the context warning,
consuming a profile switch.  Called from anywhere else they lose a notice or
overwrite persisted state, so each outside use is an allowlisted
``.sources.<source>.<member>`` chain.  Binding a source object to a name hides
the member used later, so it is rejected too: bind a reviewed member instead.
A name bound to the bundle itself (``sources = mw.sources`` or a
``ReminderSources`` parameter) is followed within its function.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

import pytest

from chrys.service.agent_middleware.reminders import ReminderSources
from tests.architecture._hygiene_core import _meta_guard_problem, _pins_allowlist, _src_sources, _tree
from tests.support.ci import CI_LINUX_ONLY

# Platform-independent source analysis: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY

_REMINDER_SOURCE_NAMES = frozenset(field.name for field in fields(ReminderSources))
_MIDDLEWARE_DIR = "src/chrys/service/agent_middleware/"

# (source, member) → why a caller outside the middleware may use it.
_REMINDER_SOURCE_MEMBER_ALLOWLIST = {
    ("archive_pointer", "record_count_state"): "save and rebuild persist the turn-start count the turn captured",
    ("archive_pointer", "restore_record_count"): "session restore stashes the persisted count for the next turn",
    ("profile_switch", "set_profile_switch"): "a profile switch records the notice the next turn announces",
    ("profile_switch", "snapshot_pending_switch"): "a rebuild carries a switch no request announced yet",
    ("profile_switch", "consumed_switch_to"): "the runner reads which switch the finished turn announced",
}


def _reads_sources(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "sources"


def _names_bundle_type(annotation: ast.expr | None) -> bool:
    return (
        (isinstance(annotation, ast.Name) and annotation.id == "ReminderSources")
        or (isinstance(annotation, ast.Attribute) and annotation.attr == "ReminderSources")
        or (isinstance(annotation, ast.Constant) and annotation.value == "ReminderSources")
    )


def _bundle_names(scope: ast.AST) -> frozenset[str]:
    """Names *scope* binds to the bundle: ``x = <y>.sources`` or an ``x: ReminderSources`` annotation."""
    names: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and _reads_sources(node.value):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if _names_bundle_type(node.annotation) or _reads_sources(node.value):
                names.add(node.target.id)
        elif isinstance(node, ast.arg) and _names_bundle_type(node.annotation):
            names.add(node.arg)
    return frozenset(names)


def _source_accesses(tree: ast.Module) -> Iterator[tuple[ast.Attribute, str, str | None]]:
    """Yield ``(node, source, member)`` for each ``<bundle>.<source>``; *member* is the attribute read on it."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    scope_names: dict[ast.AST, frozenset[str]] = {}

    def is_bundle(node: ast.expr) -> bool:
        if _reads_sources(node):
            return True
        if not isinstance(node, ast.Name):
            return False
        scope: ast.AST = node
        while not isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            scope = parents[scope]
        if scope not in scope_names:
            scope_names[scope] = _bundle_names(scope)
        return node.id in scope_names[scope]

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _REMINDER_SOURCE_NAMES and is_bundle(node.value):
            parent = parents.get(node)
            member = parent.attr if isinstance(parent, ast.Attribute) and parent.value is node else None
            yield node, node.attr, member


def _outside_uses(sources: Mapping[Path, str]) -> Iterator[tuple[Path, ast.Attribute, str, str | None]]:
    for path, source in sources.items():
        if path.as_posix().startswith(_MIDDLEWARE_DIR):
            continue
        for node, name, member in _source_accesses(_tree(path, source)):
            yield path, node, name, member


def _assert_reminder_source_members_are_reviewed(sources: Mapping[Path, str]) -> None:
    violations = [
        f"{path.as_posix()}:{node.lineno}: .sources.{name}"
        + (f".{member}" if member is not None else "")
        + " is not a reviewed reminder-source member outside the middleware; the middleware drives its other "
        "members (see tests/architecture/test_hygiene_reminder_sources.py), so use a reviewed member or add an "
        "_REMINDER_SOURCE_MEMBER_ALLOWLIST entry saying why this caller may"
        for path, node, name, member in _outside_uses(sources)
        if (name, member) not in _REMINDER_SOURCE_MEMBER_ALLOWLIST
    ]
    assert violations == [], "\n".join(violations)


@_pins_allowlist("_REMINDER_SOURCE_MEMBER_ALLOWLIST")
def test_reminder_source_member_allowlist_entries_are_live() -> None:
    """Each entry must name a real member some caller outside the middleware still uses."""
    used = {(name, member) for _path, _node, name, member in _outside_uses(_src_sources())}
    problems = [
        _meta_guard_problem(
            "_REMINDER_SOURCE_MEMBER_ALLOWLIST",
            f"entry {key} is used by no caller outside {_MIDDLEWARE_DIR}",
            "remove the stale entry",
        )
        for key in sorted(set(_REMINDER_SOURCE_MEMBER_ALLOWLIST) - used)
    ]
    source_types = get_type_hints(ReminderSources)
    problems += [
        _meta_guard_problem(
            "_REMINDER_SOURCE_MEMBER_ALLOWLIST",
            f"entry {(name, member)} names no member of {source_types[name].__name__}",
            "re-key the entry to the source member callers use",
        )
        for name, member in sorted(_REMINDER_SOURCE_MEMBER_ALLOWLIST)
        if member not in dir(source_types[name])
    ]
    _tree.cache_clear()
    assert problems == [], "\n".join(problems)


_OUTSIDE = Path("src/chrys/orchestration/engine/example.py")


@pytest.mark.parametrize(
    "source",
    [
        "def f(middleware):\n    return middleware.sources.file_change.drain()\n",
        "def f(loaded):\n    loaded.reminder_middleware.sources.archive_pointer.capture(7)\n",
        "def f(middleware, switch):\n    middleware.sources.profile_switch.carried(switch)\n",
        "def f(middleware):\n    source = middleware.sources.context_warning\n    source.delivered([])\n",
        "def f(loaded):\n    sources = loaded.reminder_middleware.sources\n    return sources.file_change.drain()\n",
        "def f(sources: ReminderSources):\n    sources.archive_pointer.capture(3)\n",
    ],
)
def test_reminder_source_guard_rejects_unreviewed_members_outside_the_middleware(source: str) -> None:
    with pytest.raises(AssertionError, match=r"example\.py:\d+: \.sources\.\w+.* is not a reviewed"):
        _assert_reminder_source_members_are_reviewed({_OUTSIDE: source})


def test_reminder_source_guard_accepts_reviewed_members_and_the_middleware_itself() -> None:
    _assert_reminder_source_members_are_reviewed(
        {
            _OUTSIDE: (
                "def f(loaded):\n"
                "    count = loaded.reminder_middleware.sources.archive_pointer.record_count_state()\n"
                "    set_switch = loaded.reminder_middleware.sources.profile_switch.set_profile_switch\n"
                "    return count, set_switch, loaded.sources.usage_publisher.drain()\n"
                "def g(loaded, sources):\n"
                "    bundle = loaded.reminder_middleware.sources\n"
                "    bundle.archive_pointer.restore_record_count(4)\n"
                "    return sources.skills.refresh()\n"
            ),
            Path("src/chrys/service/agent_middleware/system_reminder.py"): (
                "def f(self):\n    return self.sources.file_change.drain()\n"
            ),
        }
    )
