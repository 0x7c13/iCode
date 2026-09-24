# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Additive and multiplicative token calibration."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:

    class _CalibrationStrategy(Protocol):
        max_context_tokens: int


# Maximum calibration ratio.  With the additive overhead model, the ratio
# only captures tokenizer drift (should be close to 1.0).  Values above
# this indicate a measurement anomaly — cap to avoid false compaction.
_MAX_CALIBRATION_RATIO = 1.5


class TokenCalibration:
    """Own the strategy's token-overhead and calibration model."""

    def __init__(self, strategy: _CalibrationStrategy) -> None:
        self._strategy = strategy
        self._calibration_ratio: float = 1.0
        self._overhead_initialized: bool = False
        self._system_overhead: int = 0
        self._request_overhead_floor: int = 0

    def calibrate(self, api_input_tokens: int, *, local_included: int) -> None:
        """Update calibration using an already-selected local baseline."""
        if local_included <= 0 or api_input_tokens <= 0:
            return
        # First call: learn system overhead.
        if not self._overhead_initialized:
            self._overhead_initialized = True
            overhead = api_input_tokens - local_included
            self._system_overhead = max(0, overhead)
            if overhead <= 0:
                # Local already exceeds API — calibrate ratio immediately
                # so the very first usage event reports a corrected value.
                self._calibration_ratio = api_input_tokens / local_included
            return
        # Subsequent calls: calibrate ratio against known overhead.
        estimated = local_included + self._system_overhead
        if estimated > 0:
            ratio = api_input_tokens / estimated
            self._calibration_ratio = min(ratio, _MAX_CALIBRATION_RATIO)

    def restore(self, system_overhead_tokens: object, calibration_ratio: object) -> bool:
        """Restore validated calibration values from provenance-gated state."""
        if (
            not isinstance(system_overhead_tokens, int)
            or isinstance(system_overhead_tokens, bool)
            or system_overhead_tokens < 0
            or system_overhead_tokens >= self._strategy.max_context_tokens
            or not isinstance(calibration_ratio, int | float)
            or isinstance(calibration_ratio, bool)
            or not math.isfinite(calibration_ratio)
            or not 0 < calibration_ratio <= _MAX_CALIBRATION_RATIO
        ):
            return False
        self._system_overhead = system_overhead_tokens
        self._calibration_ratio = float(calibration_ratio)
        self._overhead_initialized = True
        return True

    def usage_pct(self, included: int) -> float:
        """Compute effective context usage from live strategy configuration."""
        return self.estimated_tokens(included) / self._strategy.max_context_tokens

    def estimated_tokens(self, included: int) -> int:
        """Estimate provider-visible input tokens from a local message count."""
        overhead = max(self._system_overhead, self._request_overhead_floor)
        return math.ceil((included + overhead) * self._calibration_ratio)
