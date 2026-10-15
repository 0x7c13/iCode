# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The closing summary Smart Test prints after its pytest suites."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest
from scripts import chrys_test

from tests.support.ci import CI_LINUX_ONLY
from tests.support.paths import REPO_ROOT

pytestmark = CI_LINUX_ONLY

_RESULT_LINE_PREFIX = "--chrys-result-line="


def _report_path(args: list[str]) -> Path:
    (option,) = (arg for arg in args if arg.startswith(_RESULT_LINE_PREFIX))
    return Path(option.removeprefix(_RESULT_LINE_PREFIX))


def _install_plugin(pytester: pytest.Pytester) -> None:
    """The suites load the script as a pytest plugin, from their working directory."""
    (pytester.path / "scripts").mkdir()
    (pytester.path / "scripts" / "chrys_test.py").write_text(
        (REPO_ROOT / "scripts" / "chrys_test.py").read_text(encoding="utf-8"), encoding="utf-8"
    )


def _summary(output: str, rows: int) -> list[str]:
    """The last lines of the run, which must be its summary."""
    return output.splitlines()[-(rows + 1) :]


def test_failing_suite_summary_repeats_pytests_result_line_and_skips_architecture(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_plugin(pytester)
    pytester.makepyfile(
        test_mixed="""
        import pytest

        @pytest.fixture
        def broken_setup():
            raise RuntimeError("setup")

        @pytest.fixture
        def broken_teardown():
            yield
            raise RuntimeError("teardown")

        def test_pass(): pass
        def test_fail(): assert False
        def test_skip(): pytest.skip("skip")
        @pytest.mark.xfail(strict=False)
        def test_xfail(): assert False
        @pytest.mark.xfail(strict=False)
        def test_xpass(): pass
        def test_setup_error(broken_setup): pass
        def test_teardown_error(broken_teardown): pass
        """
    )
    selection = chrys_test.Selection()
    selection.add("test_mixed.py", "changed directly")
    selection.add("tests/architecture/test_layout.py", "test tree changed")
    reports: list[Path] = []

    def run(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
        assert args[:3] == [sys.executable, "-m", "pytest"]
        reports.append(_report_path(args))
        result = pytester.runpytest_subprocess(*args[3:])
        result.assert_outcomes(passed=2, failed=1, skipped=1, xfailed=1, xpassed=1, errors=2)
        return subprocess.CompletedProcess(args, result.ret)

    monkeypatch.setattr(chrys_test, "_run_process", run)

    assert chrys_test._run_pytest_targets(selection) == 1
    assert _summary(capsys.readouterr().out, 2) == [
        "Summary: FAILED (exit 1)",
        "  tests         1 failed, 2 passed, 1 skipped, 1 xfailed, 1 xpassed, 2 errors (exit 1)",
        "  architecture  not run: the suite above failed",
    ]
    assert len(reports) == 1 and not reports[0].exists()


def test_passing_suites_summary_reports_both_suites_including_xdist_runs(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three regular files run on xdist workers, which get the same options."""
    _install_plugin(pytester)
    pytester.makepyfile(
        test_a="def test_one(): pass\ndef test_two(): pass",
        test_b="def test_three(): pass",
        test_c="def test_four(): pass",
    )
    (pytester.path / "tests" / "architecture").mkdir(parents=True)
    (pytester.path / "tests" / "architecture" / "test_rule.py").write_text("def test_rule(): pass\n", encoding="utf-8")
    selection = chrys_test.Selection()
    for name in ("test_a.py", "test_b.py", "test_c.py"):
        selection.add(name, "changed directly")
    selection.add("tests/architecture/test_rule.py", "rule watch")
    commands: list[list[str]] = []

    def run(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
        commands.append(args)
        return subprocess.CompletedProcess(args, pytester.runpytest_subprocess(*args[3:]).ret)

    monkeypatch.setattr(chrys_test, "_run_process", run)

    assert chrys_test._run_pytest_targets(selection) == 0
    assert commands[0][commands[0].index("-n") + 1] == "3"
    assert _summary(capsys.readouterr().out, 2) == [
        "Summary: PASSED (exit 0)",
        "  tests         4 passed (exit 0)",
        "  architecture  1 passed (exit 0)",
    ]


def test_summary_names_an_empty_selection_and_a_run_without_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    selection = chrys_test.Selection()
    selection.add("tests/service/mcp/test_cache_e2e.py", "changed directly")
    selection.add("tests/architecture/test_test_layout.py", "test tree changed")
    return_codes = iter((chrys_test._PYTEST_NO_TESTS_COLLECTED, 4))

    def run(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
        # Neither run writes a result line: exit 5 is described by its code,
        # and a usage error (exit 4) stops pytest before it writes one.
        return subprocess.CompletedProcess(args, next(return_codes))

    monkeypatch.setattr(chrys_test, "_run_process", run)

    assert chrys_test._run_pytest_targets(selection) == 4
    assert _summary(capsys.readouterr().out, 2) == [
        "Summary: FAILED (exit 4)",
        "  tests         no tests collected (exit 5)",
        "  architecture  no result line (exit 4)",
    ]


def test_architecture_only_selection_summarizes_one_suite(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    selection = chrys_test.Selection()
    selection.add("tests/architecture/test_test_layout.py", "test tree changed")

    def run(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
        _report_path(args).write_text("3 passed", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(chrys_test, "_run_process", run)

    assert chrys_test._run_pytest_targets(selection) == 0
    assert _summary(capsys.readouterr().out, 1) == [
        "Summary: PASSED (exit 0)",
        "  architecture  3 passed (exit 0)",
    ]


@pytest.mark.parametrize(
    ("regular_failed", "expected"),
    [
        (
            0,
            [
                "Summary: PASSED (exit 0)",
                "  tests         5 passed (exit 0)",
                "  architecture  5 passed (exit 0)",
            ],
        ),
        (
            2,
            [
                "Summary: FAILED (exit 1)",
                "  tests         2 failed, 5 passed (exit 1)",
                "  architecture  not run: the suite above failed",
            ],
        ),
    ],
)
def test_full_mode_ends_with_the_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], regular_failed: int, expected: list[str]
) -> None:
    commands: list[list[str]] = []

    def run(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
        failed = 0 if commands else regular_failed
        commands.append(args)
        line = f"{failed} failed, 5 passed" if failed else "5 passed"
        _report_path(args).write_text(line, encoding="utf-8")
        return subprocess.CompletedProcess(args, 1 if failed else 0)

    monkeypatch.setattr(chrys_test, "_run_process", run)

    assert chrys_test._run_full() == (1 if regular_failed else 0)
    assert len(commands) == (1 if regular_failed else 2)
    assert _summary(capsys.readouterr().out, 2) == expected


@pytest.mark.parametrize("worker", [False, True])
def test_only_the_xdist_controller_writes_the_result_line(tmp_path: Path, worker: bool) -> None:
    """Workers run the hook too, each with its own share of the tests; only the
    controller's line covers the whole suite."""
    destination = tmp_path / "result.txt"

    class Config:
        def getoption(self, name: str) -> str:
            assert name == "--chrys-result-line"
            return str(destination)

    class WorkerConfig(Config):
        workerinput: ClassVar[dict[str, str]] = {}

    class Reporter:
        def build_summary_stats_line(self) -> tuple[list[tuple[str, dict[str, bool]]], str]:
            return [("1 failed", {"red": True}), ("2 passed", {"green": True})], "red"

    chrys_test.pytest_terminal_summary(Reporter(), WorkerConfig() if worker else Config())

    written = destination.read_text(encoding="utf-8") if destination.exists() else None
    assert written == (None if worker else "1 failed, 2 passed")
