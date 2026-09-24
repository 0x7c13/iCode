# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Python 3.9 floor harness: the host and the injected SDK compile, import, and serve under a real 3.9."""

from __future__ import annotations

import json
import runpy
import subprocess
from pathlib import Path

from chrys.orchestration.workflows.worker_client import HOST_PATH
from chrys.service.workflows import sdk
from chrys.service.workflows.interpreter import probe_interpreter
from chrys.service.workflows.protocol import PYTHON_FLOOR
from chrys.service.workflows.sdk_artifact import materialize_sdk_artifact
from tests.support.workflow_workers import require_py39

GOLDEN = Path(__file__).with_name("fixtures") / "code-review.py"


async def test_the_floor_interpreter_is_really_the_floor() -> None:
    probe = await probe_interpreter(require_py39())
    assert probe.version_tuple[:2] == PYTHON_FLOOR


def test_artifact_and_host_compile_under_py39(tmp_path: Path) -> None:
    artifact = materialize_sdk_artifact(tmp_path / "sdk")
    completed = subprocess.run(
        [require_py39(), "-m", "compileall", "-q", str(artifact.path), str(HOST_PATH)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_the_golden_example_builds_the_same_manifest_under_py39(tmp_path: Path) -> None:
    """The SDK's builder and validation, run on the floor interpreter, agree with the project interpreter."""
    artifact = materialize_sdk_artifact(tmp_path / "sdk")
    build = (
        "import json, runpy, sys; sys.path.insert(0, sys.argv[1]); "
        "print(json.dumps(runpy.run_path(sys.argv[2])['workflow'].manifest(), sort_keys=True))"
    )
    completed = subprocess.run(
        [require_py39(), "-I", "-c", build, str(artifact.path), str(GOLDEN)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == runpy.run_path(str(GOLDEN))["workflow"].manifest()


def test_sdk_imports_under_py39_in_isolation(tmp_path: Path) -> None:
    artifact = materialize_sdk_artifact(tmp_path / "sdk")
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); import chrys.workflows as w; "
        "print(','.join(w.__all__)); print(w.__file__)"
    )
    completed = subprocess.run(
        [require_py39(), "-I", "-c", probe, str(artifact.path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    exports_line, file_line = completed.stdout.splitlines()
    assert tuple(exports_line.split(",")) == tuple(sdk.__all__)
    assert Path(file_line).resolve().is_relative_to(artifact.path.resolve())
