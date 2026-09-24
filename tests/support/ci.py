# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Markers for tests whose CI cost is platform-independent."""

from __future__ import annotations

import os
import sys

import pytest

# Full-repo analysis (catalog extraction, AST sweeps) produces identical
# results on every OS, so repeating it per CI matrix row only spends wall
# clock. Local runs are never skipped: developers on any platform still
# execute these tests before pushing.
CI_LINUX_ONLY = pytest.mark.skipif(
    bool(os.environ.get("CI")) and sys.platform != "linux",
    reason="platform-independent full-repo analysis; the Linux CI job covers it",
)
