# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Platform fakes for tests that must not lie about the running OS.

Named ``platform_fakes`` — never ``platform`` — because the ACP client tests
spawn a stub agent as ``sys.executable <tests/support/...>``, which puts this
directory at the front of the child's ``sys.path``; a ``platform.py`` here would
shadow the stdlib ``platform`` module in that subprocess and crash the SDK.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from chrys.foundation.platform import PlatformInfo, get_platform


def platform_with_config_dir(config_dir: Path) -> PlatformInfo:
    """Return the real detected platform with only ``config_dir`` redirected.

    Tests isolate ``config_dir`` to a tmp path, but they must not hardcode the
    OS booleans while doing so: a fake carrying ``is_windows=False`` lies on the
    Windows runner and defeats genuine production platform guards (the
    ``files.py`` POSIX/Windows syscall branch, ``sub_agent_logs._fsync_parent``,
    ...), turning them into ``os.open('C:\\')``-style crashes. ``is_windows`` /
    ``is_macos`` are computed properties, so replacing just the ``config_dir``
    field keeps them truthful to the real kernel.
    """
    return dataclasses.replace(get_platform(), config_dir=config_dir)
