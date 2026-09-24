# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Lifecycle checks for Textual dispatch isolation in the pytest harness."""

from __future__ import annotations

import pytest

from tests.support.paths import REPO_ROOT


def test_dispatch_isolation_allows_a_non_tui_install(pytester: pytest.Pytester) -> None:
    """The global fixture must remain inert when the optional Textual extra is absent."""
    pytester.makeconftest(
        f"""
import sys
from importlib.abc import MetaPathFinder
sys.path.insert(0, {str(REPO_ROOT)!r})


class WithoutTextual(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "textual" or fullname.startswith("textual."):
            raise ModuleNotFoundError("optional tui extra is absent", name=fullname)


sys.meta_path.insert(0, WithoutTextual())
from tests.conftest import _isolated_textual_dispatch_plans, notification_driver
"""
    )
    pytester.makepyfile(
        """
def test_headless(notification_driver):
    import sys
    from chrys.foundation.patches import textual_dispatch_cache
    textual_dispatch_cache.apply_runtime_patch()
    assert notification_driver is None
    assert not any(name == "textual" or name.startswith("textual.") for name in sys.modules)
"""
    )
    result = pytester.runpytest_subprocess("-n0", "-q")
    result.assert_outcomes(passed=1)


@pytest.mark.parametrize("fail_body", [False, True])
def test_dispatch_plans_are_cleared_around_tests(pytester: pytest.Pytester, fail_body: bool) -> None:
    pytester.makeconftest(
        f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})

import pytest
from textual.message import Message
from textual.message_pump import MessagePump
from tests.conftest import _isolated_textual_dispatch_plans


class Ping(Message):
    pass


class Probe(MessagePump):
    def on_ping(self):
        pass


@pytest.fixture(scope="session", autouse=True)
def seed_and_check_cache():
    cache = MessagePump._get_dispatch_methods._plan_cache
    list(Probe()._get_dispatch_methods("on_ping", Ping()))
    assert cache
    yield
    assert cache == {{}}, "test teardown retained handler plans"
"""
    )
    pytester.makepyfile(
        f"""
import pytest
from textual.message_pump import MessagePump
from conftest import Ping, Probe


def test_first():
    cache = MessagePump._get_dispatch_methods._plan_cache
    assert cache == {{}}, "test setup retained earlier handler plans"
    probe = Probe()
    list(probe._get_dispatch_methods("on_ping", Ping()))
    plan = cache[(Probe, "on_ping", Ping)]
    list(probe._get_dispatch_methods("on_ping", Ping()))
    assert cache[(Probe, "on_ping", Ping)] is plan
    if {fail_body!r}:
        pytest.fail("injected body failure")


def test_second():
    cache = MessagePump._get_dispatch_methods._plan_cache
    assert cache == {{}}, "previous test leaked handler plans"
    list(Probe()._get_dispatch_methods("on_ping", Ping()))
"""
    )
    # Both inner tests must share a worker to exercise the cross-test boundary.
    result = pytester.runpytest_subprocess("-n0", "-q")
    result.assert_outcomes(passed=1 if fail_body else 2, failed=1 if fail_body else 0)


@pytest.mark.parametrize("override_in_body", [False, True])
def test_dispatch_patch_fixture_restores_the_incoming_dispatcher(
    pytester: pytest.Pytester, override_in_body: bool
) -> None:
    pytester.makeconftest(
        f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})

import pytest
from textual.message_pump import MessagePump
from tests.conftest import _isolated_textual_dispatch_plans
from tests.foundation.patches.test_textual_dispatch_cache import _patch_and_reset


@pytest.fixture(scope="session", autouse=True)
def preserve_incoming_dispatcher():
    incoming = MessagePump._get_dispatch_methods
    yield
    assert MessagePump._get_dispatch_methods is incoming
    assert incoming._plan_cache == {{}}
"""
    )
    pytester.makepyfile(
        f"""
from textual.message import Message
from textual.message_pump import MessagePump


def test_dispatch(monkeypatch):
    list(MessagePump()._get_dispatch_methods("on_ping", Message()))
    assert MessagePump._get_dispatch_methods._plan_cache
    if {override_in_body!r}:
        monkeypatch.setattr(MessagePump, "_get_dispatch_methods", MessagePump._get_dispatch_methods._original)
"""
    )
    result = pytester.runpytest_subprocess("-n0", "-q")
    result.assert_outcomes(passed=1)
