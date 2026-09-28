# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The SDK artifact injected into a worker: a regular ``chrys`` package holding ``chrys.workflows``.

The SDK's single source is ``chrys.service.workflows.sdk``. A worker cannot
import that path (it runs in the user's interpreter, without chrys), so the
main process materializes a copy laid out as ``chrys/__init__.py`` +
``chrys/workflows/{__init__,_builder,_values}.py`` and the host prepends that
directory to ``sys.path``. The top-level ``__init__.py`` makes it a regular
package: a regular package on ``sys.path`` wins over any namespace portion or
later regular package, so a stale ``chrys`` in the user's environment cannot
capture the import. The host still verifies the import origin.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

from chrys.foundation.platform.files import atomic_write_owner_only_bytes
from chrys.service.workflows import sdk as _sdk_package

SDK_SOURCE_DIR = Path(_sdk_package.__file__).resolve().parent
SDK_MODULES = ("__init__.py", "_ask.py", "_builder.py", "_values.py")

_PACKAGE_INIT = b'"""Injected by chrys: only ``chrys.workflows`` is supported inside a workflow worker."""\n'


@dataclass(frozen=True, slots=True)
class SdkArtifact:
    """A materialized artifact directory and the digest of the bytes it holds."""

    path: Path
    digest: str


def sdk_artifact_digest() -> str:
    """SHA-256 over the artifact's relative paths and bytes, independent of where it is written."""
    return _digest(_artifact_files())


def _digest(files: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for relative, payload in files:
        digest.update(relative.encode("utf-8") + b"\0" + payload + b"\0")
    return digest.hexdigest()


def materialize_sdk_artifact(root: Path) -> SdkArtifact:
    """Publish a complete, immutable artifact under its full content digest."""
    files = _artifact_files()
    digest = _digest(files)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / digest
    if not (destination / ".complete").is_file():
        with tempfile.TemporaryDirectory(prefix=".sdk-", dir=root) as staging:
            directory = Path(staging)
            for relative, payload in files:
                atomic_write_owner_only_bytes(directory / relative, payload)
            atomic_write_owner_only_bytes(directory / ".complete", digest.encode("ascii"))
            try:
                directory.rename(destination)
            except OSError:
                if not (destination / ".complete").is_file():
                    raise
    return SdkArtifact(path=destination, digest=digest)


def _artifact_files() -> list[tuple[str, bytes]]:
    files = [("chrys/__init__.py", _PACKAGE_INIT)]
    files.extend((f"chrys/workflows/{name}", (SDK_SOURCE_DIR / name).read_bytes()) for name in SDK_MODULES)
    return files
