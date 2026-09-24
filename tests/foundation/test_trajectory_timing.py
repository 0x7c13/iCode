# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the persisted trajectory timing envelope."""

from __future__ import annotations

from typing import Any

import pytest

from chrys.foundation.trajectory_timing import (
    TRAJECTORY_TIMING_KEY,
    build_trajectory_timing,
    stamp_trajectory_timing,
    trajectory_timing_from_metadata,
)


def test_build_trajectory_timing_canonicalizes_utc() -> None:
    timing = build_trajectory_timing(
        started_at="2026-08-19T09:02:03+08:00",
        finished_at="2026-08-19T09:02:04+08:00",
        duration_ms=987,
    )

    assert timing == {
        "started_at": "2026-08-19T01:02:03.000000+00:00",
        "finished_at": "2026-08-19T01:02:04.000000+00:00",
        "duration_ms": 987,
    }


@pytest.mark.parametrize("duration_ms", [-1, True, 1.5])
def test_build_trajectory_timing_rejects_invalid_duration(duration_ms: Any) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        build_trajectory_timing(
            started_at="2026-08-19T01:02:03+00:00",
            finished_at="2026-08-19T01:02:04+00:00",
            duration_ms=duration_ms,
        )


def test_trajectory_timing_from_metadata_rejects_corrupt_envelope() -> None:
    metadata = {
        TRAJECTORY_TIMING_KEY: {
            "started_at": "2026-08-19T01:02:03+00:00",
            "finished_at": "2026-08-19T01:02:04+00:00",
            "duration_ms": -1,
        }
    }

    assert trajectory_timing_from_metadata(metadata) is None


def test_stamp_trajectory_timing_reports_only_real_changes() -> None:
    timing = build_trajectory_timing(
        started_at="2026-08-19T01:02:03+00:00",
        finished_at="2026-08-19T01:02:04+00:00",
        duration_ms=987,
    )
    metadata: dict[str, Any] = {}

    assert stamp_trajectory_timing(metadata, timing) is True
    assert stamp_trajectory_timing(metadata, timing, overwrite=True) is False
    assert metadata[TRAJECTORY_TIMING_KEY] == timing
