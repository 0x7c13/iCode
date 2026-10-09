# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tell how the running copy of the app was installed.

The commands that remove or upgrade it differ per installer, so ``icode uninstall`` and the
update check ask this module which one applies.
"""

from __future__ import annotations

import json
import os
import sys
from enum import StrEnum
from importlib import metadata
from pathlib import Path

from chrys import DISTRIBUTION_NAME


class InstallFlavor(StrEnum):
    """How the running copy was installed."""

    OFFLINE = "offline"
    """An offline package: a PyApp binary carrying its own Python."""
    UV = "uv"
    """``uv tool install``."""
    PIPX = "pipx"
    """``pipx install``."""
    PIP = "pip"
    """pip, or any other installer, into an environment of its own choosing."""
    SOURCE = "source"
    """An editable install from a checkout, such as ``uv sync``."""


def _is_editable_install() -> bool:
    """Whether the distribution was installed in editable mode (PEP 660 ``direct_url.json``)."""
    try:
        raw = metadata.distribution(DISTRIBUTION_NAME).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return False
    if not raw:
        return False
    try:
        dir_info = json.loads(raw).get("dir_info")
    except ValueError, AttributeError:
        return False
    return isinstance(dir_info, dict) and dir_info.get("editable") is True


def _runs_unpacked_by_pyapp() -> bool:
    """Whether this interpreter is one an offline package unpacked for itself."""
    # The installer is heavy to import, and only an interpreter with PYAPP set gets this far.
    from chrys.app.installer import _PYAPP_PROJECT_NAMES, _find_pyapp_version_dir

    executable = Path(sys.executable).resolve()
    chosen = [
        Path(folder) for name in _PYAPP_PROJECT_NAMES if (folder := os.environ.get(f"PYAPP_INSTALL_DIR_{name.upper()}"))
    ]
    if chosen:
        # A folder the user chose has no layout to recognize, so only being inside it counts.
        return any(executable.is_relative_to(folder.resolve()) for folder in chosen)
    return _find_pyapp_version_dir(executable) is not None


def detect_install_flavor() -> InstallFlavor:
    """Return how the running copy was installed.

    PyApp passes ``PYAPP`` on to every child process, so a shell started inside an offline
    install hands it to whatever ``icode`` runs there too. The installers that keep a receipt are
    therefore checked first, and ``PYAPP`` counts only for an interpreter outside any virtual
    environment that sits where PyApp unpacks its own.
    """
    if _is_editable_install():
        return InstallFlavor.SOURCE
    prefix = Path(sys.prefix)
    if (prefix / "uv-receipt.toml").is_file():
        return InstallFlavor.UV
    if (prefix / "pipx_metadata.json").is_file():
        return InstallFlavor.PIPX
    if os.environ.get("PYAPP") and sys.prefix == sys.base_prefix and _runs_unpacked_by_pyapp():
        return InstallFlavor.OFFLINE
    return InstallFlavor.PIP


def uninstall_command(flavor: InstallFlavor) -> str | None:
    """The command that removes a flavor iCode does not remove itself, if there is one."""
    match flavor:
        case InstallFlavor.UV:
            return f"uv tool uninstall {DISTRIBUTION_NAME}"
        case InstallFlavor.PIPX:
            return f"pipx uninstall {DISTRIBUTION_NAME}"
        case InstallFlavor.PIP:
            return f"pip uninstall {DISTRIBUTION_NAME}"
        case InstallFlavor.OFFLINE | InstallFlavor.SOURCE:
            return None
