# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Generic shell command allow/deny policy filtering."""

from __future__ import annotations

from dataclasses import dataclass

from chrys.service.tools.builtins.shell_command_filtering.tokenizer import _has_unquoted, parse_commands


@dataclass(frozen=True)
class FilterResult:
    """Result of a command validation check."""

    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class ShellCommandFilter:
    """Validates shell commands against a whitelist/blacklist policy.

    - If *whitelist* is set, only commands in the set are allowed.
    - If *blacklist* is set, commands in the set are blocked.
    - *allow_redirections*: if ``False``, unquoted ``>``, ``>>``, ``<`` are rejected.
    - *allow_subshells*: if ``False``, ``$(...)``, backticks are rejected.

    Both whitelist and blacklist can be set simultaneously — whitelist is checked
    first (must pass), then blacklist (must not match).
    """

    whitelist: frozenset[str] | None = None
    blacklist: frozenset[str] | None = None
    allow_redirections: bool = True
    allow_subshells: bool = True

    def validate(self, command: str) -> FilterResult:
        """Check *command* against the policy. Returns a :class:`FilterResult`."""
        # 1. Subshell check
        if not self.allow_subshells:
            match = _has_unquoted(command, {"$(", "`"})
            if match is not None:
                return FilterResult(False, f"Subshell/command substitution ({match}...) not allowed")

        # 2. Redirection check
        if not self.allow_redirections:
            match = _has_unquoted(command, {">>", ">", "<"})
            if match is not None:
                return FilterResult(False, f"Redirection ({match}) not allowed")

        # 3. Parse and check each command
        commands = parse_commands(command)
        for cmd in commands:
            if self.whitelist is not None and cmd not in self.whitelist:
                return FilterResult(False, f"Command '{cmd}' not in whitelist")
            if self.blacklist is not None and cmd in self.blacklist:
                return FilterResult(False, f"Command '{cmd}' is blacklisted")

        return FilterResult(True)
