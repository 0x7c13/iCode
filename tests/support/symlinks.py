# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Create real test links, skipping only unavailable filesystem capabilities."""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

import pytest


def symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    """Exercise symlinks on every capable host, including Windows CI workers."""
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except NotImplementedError:
        if os.environ.get("CI"):
            raise
        pytest.skip("This host does not implement symbolic links")
    except OSError as error:
        if error.errno in {errno.ENOSYS, errno.ENOTSUP} or getattr(error, "winerror", None) == 1314:
            if os.environ.get("CI"):
                raise
            pytest.skip(f"Symbolic link capability unavailable: {error}")
        raise


def junction_or_skip(link: Path, target: Path) -> None:
    """A Windows directory junction at *link*; other platforms have none, so the test is skipped there."""
    if sys.platform != "win32":
        pytest.skip("Directory junctions exist only on Windows")
    import _winapi

    _winapi.CreateJunction(str(target), str(link))
