# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shell syntax checks used by read-only command analysis."""

from __future__ import annotations

from chrys.service.tools.builtins.shell_command_filtering.tokenizer import _operator_tokens, _ShellToken

_OUTPUT_REDIRECT_OPERATORS = frozenset({">", ">>", ">|", "&>", "&>>"})
_INPUT_REDIRECT_OPERATORS = frozenset({"<", "<<", "<<-", "<<<", "<>"})
_FD_REDIRECT_OPERATORS = frozenset({">&", "<&"})
_REDIRECT_OPERATORS = _OUTPUT_REDIRECT_OPERATORS | _INPUT_REDIRECT_OPERATORS | _FD_REDIRECT_OPERATORS
_POSIX_NULL_TARGETS = frozenset({"/dev/null"})
_POWERSHELL_NULL_TARGETS = frozenset({"$null"})
_CMD_NULL_TARGETS = frozenset({"nul"})
_STANDARD_FDS = frozenset({"0", "1", "2"})
_FD_DUPLICATION_TARGETS = _STANDARD_FDS | frozenset({"-"})


def _is_null_redirect_target(token: _ShellToken, *, shell_name: str) -> bool:
    if shell_name in ("pwsh", "powershell"):
        return not token.quoted and token.text in _POWERSHELL_NULL_TARGETS
    if shell_name == "cmd":
        return token.text.lower() in _CMD_NULL_TARGETS
    return token.text in _POSIX_NULL_TARGETS


def _has_unsafe_redirection(command: str, *, shell_name: str = "") -> str | None:
    """Return the first redirection that can persist data, or ``None`` if safe.

    The auto-approval path only permits redirections that cannot create,
    truncate, append to, or read from normal files: output to the shell's null
    device and fd duplication/closure such as ``2>&1`` or ``2>&-``.
    """
    try:
        tokens = _operator_tokens(command, shell_name=shell_name)
    except ValueError:
        return "malformed redirection"

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if (
            token.text.isdigit()
            and not token.quoted
            and i + 1 < len(tokens)
            and not tokens[i + 1].quoted
            and tokens[i + 1].text in _REDIRECT_OPERATORS
        ):
            i += 1
            token = tokens[i]

        if not token.quoted and token.text in _OUTPUT_REDIRECT_OPERATORS:
            target = tokens[i + 1] if i + 1 < len(tokens) else None
            if target is None or not _is_null_redirect_target(target, shell_name=shell_name):
                return token.text
            i += 2
            continue

        if not token.quoted and token.text in _INPUT_REDIRECT_OPERATORS:
            target = tokens[i + 1] if i + 1 < len(tokens) else None
            if token.text != "<" or target is None or not _is_null_redirect_target(target, shell_name=shell_name):
                return token.text
            i += 2
            continue

        if not token.quoted and token.text in _FD_REDIRECT_OPERATORS:
            target = tokens[i + 1] if i + 1 < len(tokens) else None
            if target is not None and not target.quoted and target.text in _FD_DUPLICATION_TARGETS:
                i += 2
                continue
            if token.text == ">&" and target is not None and _is_null_redirect_target(target, shell_name=shell_name):
                i += 2
                continue
            return token.text

        i += 1

    return None


def _subshell_targets(shell_name: str) -> set[str]:
    if shell_name in ("pwsh", "powershell"):
        return {"$("}
    if shell_name == "cmd":
        return set()
    return {"$(", "`"}
