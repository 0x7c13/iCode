# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Locate the product documentation shared by the user guide and agent runtime."""

from __future__ import annotations

import os
from pathlib import Path

INDEX_FILENAME = "index.yaml"
BUNDLED_DOCS_PATH = Path("app/tui/screens/guides/_docs")


def resolve_docs_root() -> Path | None:
    """Prefer an explicit override, then bundled docs, then a source checkout.

    An invalid override deliberately reports missing documentation instead of
    silently selecting a different version. Paths are absolute even when the
    override is relative to the current working directory at resolution time
    (after applying the CLI's -C option).
    """
    override = os.environ.get("CHRYS_DOCS_ROOT")
    if override:
        try:
            candidate = Path(override).expanduser().resolve()
            return candidate if (candidate / INDEX_FILENAME).is_file() else None
        except RuntimeError, OSError, ValueError:
            return None
    package_root = Path(__file__).resolve().parents[1]
    bundled = package_root / BUNDLED_DOCS_PATH
    if (bundled / INDEX_FILENAME).is_file():
        return bundled
    checkout = package_root.parent.parent
    if (checkout / "pyproject.toml").is_file() and (checkout / "docs" / INDEX_FILENAME).is_file():
        return checkout / "docs"
    return None
