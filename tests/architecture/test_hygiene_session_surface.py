# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Src rule: every engine or session-host launch states which surface its turns record."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.architecture._hygiene_core import _tree
from tests.support.ci import CI_LINUX_ONLY

pytestmark = CI_LINUX_ONLY

# ``surface`` defaults to None (record nothing), so a launch that forgets it
# silently leaves its sessions filed under whatever surface worked in them
# last. ``surface=None`` stays legal: it is a stated decision, not an omission.
_LAUNCHES = frozenset({"assemble_agent_engine", "ChrysSessionHost"})


def _called_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _assert_launches_state_their_surface(sources: dict[Path, str]) -> None:
    violations = [
        f"{path.as_posix()}:{node.lineno}: {name}(...) must pass surface= (SessionSurface.TUI/CLI/ACP, "
        "or None for a launch whose turns are no user's)"
        for path, source in sources.items()
        for node in ast.walk(_tree(path, source))
        if isinstance(node, ast.Call)
        and (name := _called_name(node)) in _LAUNCHES
        and not any(keyword.arg == "surface" for keyword in node.keywords)
    ]
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize(
    "source",
    [
        "engine = assemble_agent_engine(bus, settings)\n",
        "engine = assembly.assemble_agent_engine(bus, state_store=store)\n",
        "host = ChrysSessionHost(profile_name='Code')\n",
        "host = session_host.ChrysSessionHost(**options)\n",
        "def launch():\n    return ChrysSessionHost(profile_name='Code', cwd=cwd)\n",
    ],
)
def test_launch_surface_guard_rejects_launches_without_a_surface(source: str) -> None:
    with pytest.raises(AssertionError, match=r"src/example.py:\d+: \w+\(\.\.\.\) must pass surface="):
        _assert_launches_state_their_surface({Path("src/example.py"): source})


def test_launch_surface_guard_allows_stated_surfaces_and_non_calls() -> None:
    _assert_launches_state_their_surface(
        {
            Path("src/example.py"): '''"""Example: engine = assemble_agent_engine(bus, settings)"""
from chrys.orchestration.engine.assembly import assemble_agent_engine
engine = assemble_agent_engine(bus, settings, surface=SessionSurface.TUI)
host = ChrysSessionHost(profile_name="Code", surface=None, **options)
factory = assemble_agent_engine
''',
        }
    )
