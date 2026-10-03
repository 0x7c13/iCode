# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Table-driven ctypes declarations for the native APIs this package loads.

ctypes passes a Python ``int`` argument of an undeclared export as a C
``int`` and reads its result as one, so 64-bit handles and pointers are
rejected or truncated. Each loader therefore lists its exports as
``{name: (restype, [argtypes])}``, in C prototype order, and structures list
their members in runs of one type, as a C declaration does, so one line
describes one export and a layout reads like its header.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def declare_functions(library: Any, signatures: Mapping[str, tuple[Any, Sequence[Any]]]) -> None:
    """Pin the result and argument types of each named export of ``library``.

    Each signature is ``(restype, argtypes)``; a ``None`` restype is ``void``.
    """
    for name, (restype, argtypes) in signatures.items():
        function = getattr(library, name)
        function.restype = restype
        function.argtypes = list(argtypes)


def struct_fields(*runs: tuple[Any, str]) -> list[tuple[str, Any]]:
    """Expand ``(ctype, "name name …")`` runs into a ``_fields_`` list, in order."""
    return [(name, ctype) for ctype, names in runs for name in names.split()]
