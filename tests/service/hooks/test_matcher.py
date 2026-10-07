# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for hook matching."""

from __future__ import annotations

import pytest

from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.matcher import matches
from chrys.service.hooks.schema import HookArgMatch, HookConfig, HookExecution, HookMatch, HookRun


def _hook(match: HookMatch, enabled: bool = True) -> HookConfig:
    return HookConfig(
        id="x",
        event=HookEvent.BEFORE_TOOL_CALL,
        run=HookRun(type="command", argv=["/bin/true"]),
        execution=HookExecution(),
        match=match,
        enabled=enabled,
    )


def test_empty_match_fires() -> None:
    assert matches(_hook(HookMatch()), {"profile": "Code", "tool": {"name": "shell"}})


def test_disabled_hook_never_fires() -> None:
    assert not matches(_hook(HookMatch(), enabled=False), {})


def test_profile_exact() -> None:
    h = _hook(HookMatch(profile="Code"))
    assert matches(h, {"profile": "Code"})
    assert not matches(h, {"profile": "QA"})


def test_profiles_any_of() -> None:
    h = _hook(HookMatch(profiles=["Code", "QA"]))
    assert matches(h, {"profile": "QA"})
    assert not matches(h, {"profile": "Explore"})


def test_tool_kind_and_name() -> None:
    h = _hook(HookMatch(tool_kind="shell", tool_name="shell"))
    payload = {"tool": {"kind": "shell", "name": "shell"}}
    assert matches(h, payload)
    assert not matches(h, {"tool": {"kind": "filesystem.read", "name": "shell"}})
    assert not matches(h, {"tool": {"kind": "shell", "name": "other"}})


def test_tool_kind_matches_pwsh_and_powershell() -> None:
    h = _hook(HookMatch(tool_kind="shell"))
    assert matches(h, {"tool": {"kind": "shell", "name": "pwsh"}})
    assert matches(h, {"tool": {"kind": "shell", "name": "powershell"}})


def test_tool_clause_requires_tool_payload() -> None:
    h = _hook(HookMatch(tool_kind="shell"))
    assert not matches(h, {"profile": "Code"})


def test_args_regex() -> None:
    h = _hook(HookMatch(args={"command": HookArgMatch(regex=r"rm\s+-rf")}))
    assert matches(h, {"tool": {"args": {"command": "rm -rf /tmp/x"}}})
    assert not matches(h, {"tool": {"args": {"command": "ls /tmp"}}})


def test_args_contains() -> None:
    h = _hook(HookMatch(args={"path": HookArgMatch(contains=".env")}))
    assert matches(h, {"tool": {"args": {"path": "/secrets/.env.prod"}}})
    assert not matches(h, {"tool": {"args": {"path": "/tmp/foo"}}})


def test_args_equals() -> None:
    h = _hook(HookMatch(args={"path": HookArgMatch(equals="/etc/passwd")}))
    assert matches(h, {"tool": {"args": {"path": "/etc/passwd"}}})
    assert not matches(h, {"tool": {"args": {"path": "/etc/passwd.bak"}}})


def test_args_combined_clauses_are_and() -> None:
    h = _hook(
        HookMatch(
            args={"command": HookArgMatch(regex=r"^git\b", contains="push")},
        )
    )
    assert matches(h, {"tool": {"args": {"command": "git push origin"}}})
    assert not matches(h, {"tool": {"args": {"command": "git status"}}})


def test_missing_arg_fails_match() -> None:
    h = _hook(HookMatch(args={"command": HookArgMatch(contains="x")}))
    # The "args" dict has no "command" key at all.
    assert not matches(h, {"tool": {"args": {"path": "y"}}})


@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param("[invalid(", id="re-error"),
        pytest.param("a{4294967296}", id="repeat-count-overflow"),
        pytest.param("(" * 5000 + ")" * 5000, id="nesting-too-deep"),
    ],
)
def test_invalid_regex_logs_and_no_match(pattern: str) -> None:
    h = _hook(HookMatch(args={"command": HookArgMatch(regex=pattern)}))
    # No exception, just a non-match.
    assert not matches(h, {"tool": {"args": {"command": "anything"}}})
