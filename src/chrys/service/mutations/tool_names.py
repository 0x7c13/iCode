# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tool-name sets used by mutation tracking and event middleware."""

from __future__ import annotations

_FILE_TOOLS = frozenset({"edit_file", "write_file"})
"""Tool names whose target files should be snapshotted before/after execution."""

_IMPLICIT_WRITE_TOOLS = frozenset({"run_skill_script"})
"""Tools that run subprocesses which may implicitly modify workspace files.

These get git calibration (before/after diff) but NOT heuristic shell
scanning (no command string to parse).  Shell tools get both.
"""
