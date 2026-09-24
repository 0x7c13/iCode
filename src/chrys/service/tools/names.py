# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Canonical Chrys-owned tool names reserved from MCP catalogs."""

from __future__ import annotations

from chrys.foundation.platform import get_platform

_FIXED_CHRYS_TOOL_NAMES = frozenset(
    {
        # Built-in tools.
        "ask_user",
        "convert_document",
        "edit_file",
        "glob",
        "grep",
        "list_workspace_dirs",
        "read_file",
        "sleep",
        "todo_write",
        "view_image",
        "write_file",
        # Context-management tools contributed at run time.
        "compress_context",
        "list_compressed_contexts",
        "recall_context",
        # Skill tools contributed at run time.
        "load_skill",
        "read_skill_resource",
        "run_skill_script",
    }
)


def chrys_reserved_tool_names() -> set[str]:
    """Return names that an enabled MCP catalog must not shadow.

    Shell tool names are platform-derived (for example ``zsh``, ``pwsh``, or
    ``git_bash``), so they are added at call time rather than frozen into a
    cross-platform constant.
    """
    platform = get_platform()
    return {
        *_FIXED_CHRYS_TOOL_NAMES,
        platform.shell.name,
        *(shell.name for shell in platform.extra_shells),
    }
