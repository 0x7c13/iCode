# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The injected SDK artifact: regular-package layout, byte-for-byte copies, and a stable digest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from chrys.service.workflows import sdk
from chrys.service.workflows.sdk_artifact import (
    SDK_MODULES,
    SDK_SOURCE_DIR,
    materialize_sdk_artifact,
    sdk_artifact_digest,
)


def test_artifact_is_a_regular_package_copy_of_the_sdk(tmp_path: Path) -> None:
    artifact = materialize_sdk_artifact(tmp_path / "sdk")
    assert artifact.path == tmp_path / "sdk" / artifact.digest
    assert (artifact.path / "chrys" / "__init__.py").is_file()
    for name in SDK_MODULES:
        assert (artifact.path / "chrys" / "workflows" / name).read_bytes() == (SDK_SOURCE_DIR / name).read_bytes()
    assert artifact.digest == sdk_artifact_digest()
    assert materialize_sdk_artifact(tmp_path / "again").digest == artifact.digest


def test_the_artifact_carries_every_sdk_module() -> None:
    """A module added to the SDK package must be listed, or workers would import a partial SDK."""
    assert tuple(sorted(path.name for path in SDK_SOURCE_DIR.glob("*.py"))) == SDK_MODULES


def test_artifact_imports_in_isolation_with_the_eight_exports(tmp_path: Path) -> None:
    artifact = materialize_sdk_artifact(tmp_path / "sdk")
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); import chrys.workflows as w; "
        "print(w.__file__); print(','.join(w.__all__))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(artifact.path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    file_line, exports_line = completed.stdout.splitlines()
    assert Path(file_line).resolve() == (artifact.path / "chrys" / "workflows" / "__init__.py").resolve()
    assert tuple(exports_line.split(",")) == tuple(sdk.__all__)


def test_concurrent_publishers_reuse_one_complete_directory(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        artifacts = list(pool.map(materialize_sdk_artifact, [tmp_path / "sdk"] * 4))
    assert len({artifact.path for artifact in artifacts}) == 1
    artifact = artifacts[0]
    assert (artifact.path / ".complete").read_text() == artifact.digest
    assert list((tmp_path / "sdk").iterdir()) == [artifact.path]
    before = {path: path.stat().st_mtime_ns for path in artifact.path.rglob("*.py")}
    assert materialize_sdk_artifact(tmp_path / "sdk") == artifact
    assert {path: path.stat().st_mtime_ns for path in artifact.path.rglob("*.py")} == before


def test_different_sdk_bytes_publish_separate_complete_packages(tmp_path: Path, monkeypatch) -> None:
    from unittest.mock import create_autospec

    import chrys.service.workflows.sdk_artifact as module

    first = materialize_sdk_artifact(tmp_path)
    original = module._artifact_files
    files = original()
    different = [(name, payload + b"\n") for name, payload in files]
    reader = create_autospec(original, return_value=different)
    monkeypatch.setattr(module, "_artifact_files", reader)
    second = materialize_sdk_artifact(tmp_path)
    reader.assert_called_once()
    assert first.path != second.path
    for name, payload in files:
        assert (first.path / name).read_bytes() == payload
        assert (second.path / name).read_bytes() == payload + b"\n"
