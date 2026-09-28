# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Show arguments as a Windows command line in a settings field, and split what was typed there back into arguments."""

from __future__ import annotations

from collections.abc import Sequence

# The runtime separates on spaces and tabs only. A command typed or pasted over several lines of a
# field is still one command line, so line breaks separate too.
_SEPARATORS = " \t\r\n"


def join_windows_command_line(args: Sequence[str]) -> str:
    """Join *args* into the command line :func:`split_windows_command_line` splits back into *args*.

    Quotes as ``subprocess.list2cmdline`` does, and also an argument holding a line break.
    """
    return " ".join(_quote(arg) for arg in args)


def _quote(arg: str) -> str:
    needs_quotes = not arg or any(char in _SEPARATORS for char in arg)
    parts: list[str] = []
    backslashes = 0
    for char in arg:
        if char == "\\":
            backslashes += 1
            continue
        if char == '"':
            # The backslashes before a quote double, and one more makes the quote literal.
            parts.append("\\" * (backslashes * 2 + 1) + '"')
        else:
            parts.append("\\" * backslashes + char)
        backslashes = 0
    # Trailing backslashes double only before the closing quote; anywhere else they are literal.
    parts.append("\\" * (backslashes * 2 if needs_quotes else backslashes))
    text = "".join(parts)
    return f'"{text}"' if needs_quotes else text


def split_windows_command_line(text: str) -> list[str]:
    """Split *text* into arguments by the Windows C runtime rules, the ones ``subprocess.list2cmdline`` quotes for.

    Spaces and tabs separate arguments outside double quotes, and so, unlike in
    the runtime, do line breaks. Backslashes are literal unless they come before
    a double quote: then each pair is one backslash, and an odd one out makes the
    quote literal. Inside quotes, ``""`` is a literal quote. Unlike the runtime,
    an unclosed quote raises ``ValueError``: in a command someone typed it is
    most likely a mistake.
    """
    args: list[str] = []
    current: list[str] = []
    in_arg = False
    in_quotes = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            end = index
            while end < len(text) and text[end] == "\\":
                end += 1
            count = end - index
            if end < len(text) and text[end] == '"':
                current.append("\\" * (count // 2))
                if count % 2:
                    current.append('"')
                    end += 1
            else:
                current.append("\\" * count)
            in_arg = True
            index = end
        elif char == '"':
            if in_quotes and text[index + 1 : index + 2] == '"':
                current.append('"')
                index += 1
            else:
                in_quotes = not in_quotes
            in_arg = True
            index += 1
        elif char in _SEPARATORS and not in_quotes:
            if in_arg:
                args.append("".join(current))
                current = []
                in_arg = False
            index += 1
        else:
            current.append(char)
            in_arg = True
            index += 1
    if in_quotes:
        raise ValueError("No closing quotation")
    if in_arg:
        args.append("".join(current))
    return args
