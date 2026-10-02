# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shell token splitting helpers shared by tools and mutation detection."""

from __future__ import annotations


def split_on_operators(command: str) -> list[str]:
    """Split *command* on unquoted ``|``, ``&&``, ``||``, ``&``, ``;``, or newline operators.

    Respects single and double quotes so that ``echo 'a|b'`` is not split.
    """
    segments: list[str] = []
    current: list[str] = []
    i = 0
    in_single = False
    in_double = False

    while i < len(command):
        ch = command[i]

        # Backslash escape (not inside single quotes)
        if ch == "\\" and not in_single and i + 1 < len(command):
            current.append(ch)
            current.append(command[i + 1])
            i += 2
            continue

        # Quote tracking
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            i += 1
            continue

        # Operators (only outside quotes)
        if not in_single and not in_double:
            if ch == "|":
                if i + 1 < len(command) and command[i + 1] == "|":
                    # ||
                    segments.append("".join(current))
                    current = []
                    i += 2
                    continue
                # |
                segments.append("".join(current))
                current = []
                i += 1
                continue
            if ch == "&" and i + 1 < len(command) and command[i + 1] == "&":
                # &&
                segments.append("".join(current))
                current = []
                i += 2
                continue
            if ch == "&":
                prev_ch = command[i - 1] if i > 0 else ""
                next_ch = command[i + 1] if i + 1 < len(command) else ""
                if prev_ch in "><" or next_ch == ">":
                    current.append(ch)
                    i += 1
                    continue
                segments.append("".join(current))
                current = []
                i += 1
                continue
            if ch in ";\r\n":
                segments.append("".join(current))
                current = []
                i += 1
                continue

        current.append(ch)
        i += 1

    if current:
        segments.append("".join(current))

    return [s for s in segments if s.strip()]


_split_on_operators = split_on_operators
"""Backward-compatible alias for callers predating the public helper."""
