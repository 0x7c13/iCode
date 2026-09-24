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
