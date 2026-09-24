# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Process-wide telemetry gate shared by foundation setup and kernel spans."""

from __future__ import annotations


class TelemetryGate:
    """Process-wide switch shared by Chrys telemetry emitters.

    Both flags default off. ``configure_telemetry`` sets them during provider
    setup; the gate itself reads no environment variables.
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self.sensitive_data: bool = False

    @property
    def sensitive_enabled(self) -> bool:
        """Allow sensitive capture only when telemetry and sensitive data are both enabled."""
        return self.enabled and self.sensitive_data


TELEMETRY_GATE = TelemetryGate()


def configure_telemetry(*, enabled: bool, sensitive_data: bool = False) -> None:
    """Flip the telemetry gate (``setup_otel`` closes it at entry, reopens on success)."""
    TELEMETRY_GATE.enabled = enabled
    TELEMETRY_GATE.sensitive_data = sensitive_data
