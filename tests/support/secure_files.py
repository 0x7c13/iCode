# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Plant files the way the product creates them, on every platform.

A test that pre-creates a file the product will later open through
``secure_open_owner_only_append`` cannot use ``Path.write_bytes``: on Windows
the new file takes the token's default owner — the Administrators group on an
elevated runner — and the product's owner-only verification rejects it, so the
planted file reads as one written by somebody else. Creating it through the
same helper the product uses gives it the owner and DACL the product would
have written itself, and is a plain 0600 file on POSIX.
"""

from __future__ import annotations

import os
from pathlib import Path

from chrys.foundation.platform import files


def plant_owner_only_bytes(path: Path, payload: bytes = b"") -> None:
    """Create *path* owner-only, holding *payload* as its whole content."""
    handle = files.secure_open_owner_only_append(path)
    try:
        if payload:
            os.write(handle.fd, payload)
    finally:
        os.close(handle.fd)
