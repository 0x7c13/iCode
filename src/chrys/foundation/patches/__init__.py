# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Patches for third-party dependencies.

This module provides a mechanism to apply targeted patches to installed
site-packages when upstream libraries lack required features. Each patch
is defined as a pair of code fragments (original → patched). The patcher
validates the source file before applying:

- If it matches the original → apply the patch
- If it already matches the patched version → skip (idempotent)
- If it matches neither → raise an error (upstream changed, patch needs update)

Call ``apply_all()`` at startup before the patched code is imported.
"""

import logging

from chrys.foundation.patches.patcher import apply_all

__all__ = ["apply_all"]

# Without a handler of its own, a process that configures no logging, such as
# ``icode agent``, would print every patch record on its stderr at each launch,
# a patch the install cannot take among them. Handlers configured on the root
# still get them: the TUI's debug log, ACP's stderr.
logging.getLogger(__name__).addHandler(logging.NullHandler())
