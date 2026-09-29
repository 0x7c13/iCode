# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Exception text as the formatter and every classification rule read it."""

from __future__ import annotations


def _clean_exception_text(exc: BaseException) -> str:
    """Return ``str(exc)`` normalized for client wrapper prefixes."""
    msg = str(exc).strip()
    if msg.startswith("<class "):
        msg = msg.split("> ", 1)[-1]
    return msg
