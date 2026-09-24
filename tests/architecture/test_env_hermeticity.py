# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Lifecycle self-tests for the environment-hermeticity guard."""

from __future__ import annotations

import os

import pytest

from tests.support.ci import CI_LINUX_ONLY
from tests.support.paths import REPO_ROOT

# Platform-independent test-infra self-checks: the Linux CI job covers them.
pytestmark = CI_LINUX_ONLY

_NESTED_CONFTEST = """
import pytest

from tests.support.env_guard import (
    report_environment_leak,
    restore_environment_and_describe_changes,
    snapshot_environment,
)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    before = snapshot_environment()
    result = yield
    changes = restore_environment_and_describe_changes(before)
    if changes is not None:
        report_environment_leak(item, changes)
    return result
"""


@pytest.fixture
def _nested_pytest_import_path(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    existing = os.environ.get("PYTHONPATH")
    value = str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    monkeypatch.setenv("PYTHONPATH", value)
    # A nested pytest process must not inherit the outer test's transient
    # bookkeeping key as its own pre-protocol baseline.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    return monkeypatch


def test_monkeypatch_environment_change_is_restored_before_protocol_exit(
    pytester: pytest.Pytester,
    _nested_pytest_import_path: pytest.MonkeyPatch,
) -> None:
    pytester.makeconftest(_NESTED_CONFTEST)
    pytester.makepyfile(
        """
        def test_green(monkeypatch):
            monkeypatch.setenv("CHRYS_ENV_GUARD_SELF_TEST", "temporary")
        """
    )

    _nested_pytest_import_path.delenv("PYTEST_CURRENT_TEST", raising=False)
    result = pytester.runpytest_subprocess("-q")

    result.assert_outcomes(passed=1)


def test_raw_environment_leak_guard_goes_red_and_names_key(
    pytester: pytest.Pytester,
    _nested_pytest_import_path: pytest.MonkeyPatch,
) -> None:
    pytester.makeconftest(_NESTED_CONFTEST)
    pytester.makepyfile(
        """
        import os

        def test_red():
            os.environ["CHRYS_ENV_GUARD_SELF_TEST"] = "leaked"
        """
    )

    _nested_pytest_import_path.delenv("PYTEST_CURRENT_TEST", raising=False)
    result = pytester.runpytest_subprocess("-q")

    assert result.ret != pytest.ExitCode.OK
    result.stdout.fnmatch_lines(["*Environment leak after*CHRYS_ENV_GUARD_SELF_TEST*"])
