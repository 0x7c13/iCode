# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Sampling guard for tests that bound deterministic CPU work."""

from __future__ import annotations

import time
from collections.abc import Callable

# These guards pin bounded CPU work, not user-visible latency or timeout
# semantics; pytest's per-test wall-clock timeout remains the hang guard.
CPU_TIME_BOUND_SECONDS = 0.5

# A single clean sample already proves the bound, so a healthy call is timed
# exactly once and the repeats are only ever paid after a dirty sample.
_CPU_TIME_SAMPLES = 3


def cpu_bounded[T](call: Callable[[], T], bound_seconds: float = CPU_TIME_BOUND_SECONDS) -> T:
    """Return ``call()``'s result, requiring its FASTEST run to fit ``bound_seconds``.

    The guarded operations are deterministic: they are metered by visit and
    size budgets, never by a clock, so every sample measures the same work and
    any disturbance is strictly additive. Current-thread CPU time already
    excludes xdist descheduling and unrelated background threads, but it still
    bills the measuring thread for transient kernel work it did not ask for --
    a page-fault or memory-compression storm under parallel-worker memory
    pressure lands inside one window and inflates that one sample. Taking the
    fastest of a few samples rejects that noise by construction, while the
    runaway regressions these guards exist to catch (unmemoized re-expansion,
    exponential path walks) inflate *every* sample and still trip the bound.
    """
    samples: list[float] = []
    for _ in range(_CPU_TIME_SAMPLES):
        started = time.thread_time()
        result = call()
        samples.append(time.thread_time() - started)
        if samples[-1] < bound_seconds:
            return result
    raise AssertionError(
        f"deterministic call exceeded {bound_seconds}s in all {len(samples)} samples: "
        + ", ".join(f"{sample:.3f}s" for sample in samples)
    )
