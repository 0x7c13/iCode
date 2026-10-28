# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Private copies of installed packages, for tests that patch or launch them."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


def _skipped(directory: str, names: list[str]) -> set[str]:
    """Leave out bytecode, and the temporary file of a patch still being written.

    Every launch patches the installed files, as a test starting the real entrypoint does on a fresh venv such as CI's,
    which may run beside the copy. Each file is replaced whole from a temporary file written beside it, so a copy can
    list that temporary file, named ``<file>.<random>.tmp``, and then find it gone, but reads every file it copies whole.
    """
    present = set(names)
    return {
        name
        for name in names
        if name == "__pycache__" or (name.endswith(".tmp") and name.removesuffix(".tmp").rpartition(".")[0] in present)
    }


def copy_installed_package(package: str, parent: Path) -> Path:
    """Copy the installed top-level *package* to ``parent / package`` and return the copy."""
    if "." in package:
        raise ValueError(f"{package!r} is not a top-level package.")
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise ImportError(f"Package {package!r} is not installed.")
    if spec.submodule_search_locations is None:
        # A module's directory is the one it was installed into, with every other package there.
        raise ValueError(f"{package!r} is a module, not a package.")
    copy = parent / package
    shutil.copytree(Path(spec.origin).parent, copy, ignore=_skipped)
    return copy
