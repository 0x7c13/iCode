# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Threads a test starts and must see finish."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading
    from collections.abc import Sequence


def run_to_the_end(workers: Sequence[threading.Thread], *, within: float) -> None:
    """Start *workers* and see all of them finish inside one shared budget of *within* seconds."""
    deadline = time.monotonic() + within
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(max(0.0, deadline - time.monotonic()))
    assert not any(thread.is_alive() for thread in workers), "a worker thread did not finish in time"
