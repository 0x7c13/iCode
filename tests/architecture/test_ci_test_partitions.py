# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Every platform's CI shards retain the complete test inventory and required gates."""

from __future__ import annotations

import shlex
from collections import Counter
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("platform", ["linux", "macos", "windows"])
def test_platform_shards_cover_every_test_module_once(platform: str) -> None:
    jobs = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    inventory = {
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "tests").rglob("*.py")
        if path.name.startswith("test_") or path.name.endswith("_test.py")
    }
    expected = {path for path in inventory if not path.is_relative_to("tests/architecture")}
    job_id = f"test_{platform}"
    rows = jobs[job_id]["strategy"]["matrix"]["include"]
    selected: Counter[Path] = Counter()
    for row in rows:
        if platform != "windows":
            assert row["os"] == platform
        args = shlex.split(row["pytest_args"])
        roots = [Path(arg) for arg in args if not arg.startswith("-")]
        ignored = [Path(arg.removeprefix("--ignore=")) for arg in args if arg.startswith("--ignore=")]
        assert all(not arg.startswith("-") or arg.startswith("--ignore=") for arg in args), args
        assert roots and all((REPO_ROOT / root).is_dir() for root in roots)
        selected.update(
            path
            for path in inventory
            if any(path.is_relative_to(root) for root in roots)
            and not any(path.is_relative_to(ignore) for ignore in ignored)
        )
    assert set(selected) == expected, {"missing": expected - selected.keys(), "extra": selected.keys() - expected}
    assert all(count == 1 for count in selected.values()), [path for path, count in selected.items() if count != 1]


@pytest.mark.parametrize("platform", ["linux", "macos", "windows"])
def test_required_platform_gate_includes_every_shard(platform: str) -> None:
    jobs = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    gate = jobs[f"test_{platform}_gate"]
    shard_job = f"test_{platform}"
    assert gate["if"] == "always()"
    assert set(gate["needs"]) == {shard_job, "architecture"}
    assert gate["name"] == f"Test ({platform})"
    step = gate["steps"][0]
    assert f"${{{{ needs.{shard_job}.result }}}}" in step["env"].values()
    assert "${{ needs.architecture.result }}" in step["env"].values()
    for variable in step["env"]:
        assert f'test "${variable}" = "success"' in step["run"]


@pytest.mark.parametrize("platform", ["linux", "macos", "windows"])
def test_typecheck_remains_enabled_for_unrecognized_core_labels(platform: str) -> None:
    job = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"][
        f"test_{platform}"
    ]
    step = next(step for step in job["steps"] if step.get("name") == "Typecheck (ty)")
    # A status function prevents the implicit success() gate from suppressing
    # type diagnostics after pytest fails, while respecting cancellation.
    assert step["if"] == "${{ !cancelled() && !endsWith(matrix.label, '/tui') && !matrix.skip_typecheck }}"
    test_step = next(step for step in job["steps"] if step.get("name") == "Run tests")
    assert job["steps"].index(test_step) < job["steps"].index(step)
    assert not step.get("continue-on-error", False)
    rows = job["strategy"]["matrix"]["include"]
    # Opting out is an explicit boolean per row; a quoted "false" would be truthy
    # in the workflow expression, and a core shard without the key typechecks.
    assert all(isinstance(row.get("skip_typecheck", False), bool) for row in rows)
    labels = [row["label"] for row in rows if not row["label"].endswith("/tui") and not row.get("skip_typecheck")]
    assert step["run"] == "uv run ty check --error-on-warning src/chrys"
    assert len(labels) == 1
