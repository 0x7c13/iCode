# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Create real test links, skipping only unavailable filesystem capabilities."""

from __future__ import annotations

import errno
import os
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
