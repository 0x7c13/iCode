# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reparse data of Windows App Execution Aliases, as ``FSCTL_GET_REPARSE_POINT`` returns it."""

from __future__ import annotations

import struct

IO_REPARSE_TAG_APPEXECLINK = 0x8000001B

STORE_PYTHON_REDIRECTOR = (
    r"C:\Program Files\WindowsApps\Microsoft.DesktopAppInstaller_1.27.0.0_x64__8wekyb3d8bbwe"
    r"\AppInstallerPythonRedirector.exe"
)


def alias_reparse_data(*strings: str, tag: int = IO_REPARSE_TAG_APPEXECLINK) -> bytes:
    """A REPARSE_DATA_BUFFER for an alias holding *strings*: its prefix, then each string NUL-terminated."""
    payload = struct.pack("<I", 3) + "".join(f"{value}\0" for value in strings).encode("utf-16-le")
    return struct.pack("<IHH", tag, len(payload), 0) + payload


# What Windows puts at WindowsApps\python.exe and python3.exe while no Python is installed.
STORE_PYTHON_STUB = alias_reparse_data(
    "Microsoft.DesktopAppInstaller_8wekyb3d8bbwe",
    "Microsoft.DesktopAppInstaller_8wekyb3d8bbwe!PythonRedirector",
    STORE_PYTHON_REDIRECTOR,
    "0",
)
# The alias of a Python installed from the Microsoft Store.
STORE_PYTHON = alias_reparse_data(
    "PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0",
    "PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0!Python",
    r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.5.0_x64__qbz5n2kfra8p0\python3.13.exe",
    "0",
)
