# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Utility helpers for the virtualized markdown package."""

from __future__ import annotations

from pathlib import Path


def sanitize_location(location: str) -> tuple[Path, str]:
    """Given a location, break out the path and any anchor.

    Args:
        location: The location to sanitize.

    Returns:
        A tuple of the path to the location cleaned of any anchor, plus
        the anchor (or an empty string if none was found).
    """
    location, _, anchor = location.partition("#")
    return Path(location), anchor
