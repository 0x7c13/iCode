# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared command-line parsing helpers."""

from __future__ import annotations

import argparse
from typing import Never

from chrys.foundation.i18n.formatting import sanitize_legacy_scalar


class SanitizingArgumentParser(argparse.ArgumentParser):
    """Argument parser that sanitizes attacker-controlled error details."""

    def error(self, message: str) -> Never:
        super().error(sanitize_legacy_scalar(message))


def default_to_list(argv: list[str]) -> list[str]:
    """*argv* for a command group with ``list`` filled in when no subcommand was named.

    A group's arguments name no subcommand when they are empty or start with an
    option other than help, so ``workflow --json`` means ``workflow list --json``
    while ``workflow --help`` still shows the group's own help; after ``--`` the
    subcommand is named next, as in ``workflow -- run demo``.
    """
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help", "--")):
        return ["list", *argv]
    return argv
