# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Sleep tool — lets the agent pause briefly before continuing."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Annotated

from chrys.service.tools.kinds import KIND_SLEEP, tool
from chrys.service.tools.result_metadata import tool_error

MAX_SLEEP_SECONDS = 3600
"""Maximum sleep duration exposed to the model."""


def normalize_sleep_seconds(value: object) -> int | None:
    """Return a valid integer sleep duration, or ``None`` for invalid input."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        with suppress(ValueError, OverflowError):
            numeric = float(stripped)
            return int(numeric) if numeric.is_integer() else None
    return None


def validate_sleep_seconds(seconds: int | None) -> str:
    """Return an error string for invalid durations, or ``""`` when valid."""
    if seconds is None:
        return tool_error("invalid_sleep_duration", "sleep duration must be an integer number of seconds.")
    if seconds < 0:
        return tool_error(
            "invalid_sleep_duration",
            "sleep duration must be 0 seconds or greater.",
            details={"seconds": seconds},
        )
    if seconds > MAX_SLEEP_SECONDS:
        return tool_error(
            "sleep_duration_too_long",
            (
                f"sleep duration cannot exceed {MAX_SLEEP_SECONDS} seconds. "
                "Use a background command or monitor-style workflow for longer waits."
            ),
            details={"seconds": seconds, "max_seconds": MAX_SLEEP_SECONDS},
        )
    return ""


def format_sleep_seconds(seconds: int) -> str:
    """Format seconds with a singular/plural unit."""
    unit = "second" if seconds == 1 else "seconds"
    return f"{seconds} {unit}"


@tool(kind=KIND_SLEEP)
async def sleep(
    seconds: Annotated[int, f"How long to sleep, in seconds. Must be between 0 and {MAX_SLEEP_SECONDS}."],
    reason: Annotated[
        str,
        "Brief reason for sleeping, shown to the user. Prefer background monitoring for long-running work.",
    ] = "",
) -> str:
    """Sleep briefly before continuing.

    Use this when a short delay is genuinely useful before the next action.
    Do not use sleep loops to monitor long-running commands, files, or services;
    prefer background execution and monitoring for those workflows.
    """
    duration = normalize_sleep_seconds(seconds)
    error = validate_sleep_seconds(duration)
    if error:
        return error
    assert duration is not None
    await asyncio.sleep(duration)
    return f"Slept for {format_sleep_seconds(duration)}."
