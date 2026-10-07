# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shell tokenization and command parsing helpers."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from chrys.foundation.util.shell_tokens import split_on_operators


# Coarse legacy parser used by configured filters and mutation hints.
def parse_commands(command: str) -> list[str]:
    """Extract all command names from a shell command string.

    Handles pipes (``|``), chaining (``&&``, ``||``), backgrounding
    (``&``), semicolons, and shell command newlines.
    Skips leading environment variable assignments (``KEY=val cmd``).

    Returns a list of command names (the first token of each segment).
    """
    segments = split_on_operators(command)
    commands: list[str] = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError:
            # Unbalanced quotes etc. — treat the raw first word as the command
            tokens = segment.split()
        if not tokens:
            continue
        # Skip env var assignments: KEY=val KEY2=val2 actual_command ...
        idx = 0
        while idx < len(tokens) and "=" in tokens[idx] and not tokens[idx].startswith("="):
            idx += 1
        if idx < len(tokens):
            commands.append(tokens[idx])
    return commands


def _has_unquoted(command: str, targets: set[str]) -> str | None:
    """Return the first *target* string found outside quotes, or ``None``."""
    i = 0
    in_single = False
    in_double = False

    while i < len(command):
        ch = command[i]

        if ch == "\\" and not in_single and i + 1 < len(command):
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue

        if not in_single and not in_double:
            for target in targets:
                if command[i : i + len(target)] == target:
                    return target

        i += 1

    return None


# Shell-aware token stream used by read-only approval analysis.
@dataclass(frozen=True)
class _ShellToken:
    text: str
    quoted: bool = False


_SHELL_OPERATORS = (
    "<<-",
    "<<<",
    "&>>",
    ">>",
    ">|",
    "&>",
    ">&",
    "<&",
    "&&",
    "||",
    "|&",
    "<<",
    "<>",
    ">",
    "<",
    "|",
    "&",
    ";",
    "(",
    ")",
)


def _shell_escape_char(shell_name: str) -> str:
    if shell_name in ("pwsh", "powershell"):
        return "`"
    if shell_name == "cmd":
        return "^"
    return "\\"


def _has_unquoted_for_shell(command: str, targets: set[str], *, shell_name: str = "") -> str | None:
    """Return the first target found outside shell-specific quotes/escapes."""
    i = 0
    in_single = False
    in_double = False
    escape_char = _shell_escape_char(shell_name)
    supports_single_quotes = shell_name != "cmd"

    while i < len(command):
        ch = command[i]

        if ch == escape_char and not in_single and i + 1 < len(command):
            i += 2
            continue
        if supports_single_quotes and ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue

        if not in_single:
            for target in targets:
                if command[i : i + len(target)] == target:
                    return target

        i += 1

    return None


def _operator_tokens(command: str, *, shell_name: str = "") -> list[_ShellToken]:
    """Tokenize shell operators while preserving whether words were quoted."""
    tokens: list[_ShellToken] = []
    current: list[str] = []
    current_quoted = False
    in_single = False
    in_double = False
    escape_char = _shell_escape_char(shell_name)
    supports_single_quotes = shell_name != "cmd"
    i = 0

    def flush() -> None:
        nonlocal current_quoted
        if current or current_quoted:
            tokens.append(_ShellToken("".join(current), current_quoted))
            current.clear()
            current_quoted = False

    while i < len(command):
        ch = command[i]

        if ch == escape_char and not in_single and i + 1 < len(command):
            current.append(command[i + 1])
            current_quoted = True
            i += 2
            continue

        if supports_single_quotes and ch == "'" and not in_double:
            in_single = not in_single
            current_quoted = True
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            current_quoted = True
            i += 1
            continue

        if not in_single and not in_double:
            if ch in "\r\n":
                flush()
                tokens.append(_ShellToken(";"))
                i += 1
                continue

            if ch.isspace():
                flush()
                i += 1
                continue

            matched_operator = next((op for op in _SHELL_OPERATORS if command.startswith(op, i)), "")
            if matched_operator:
                flush()
                tokens.append(_ShellToken(matched_operator))
                i += len(matched_operator)
                continue

        current.append(ch)
        i += 1

    if in_single or in_double:
        msg = "No closing quotation"
        raise ValueError(msg)

    flush()
    return tokens
