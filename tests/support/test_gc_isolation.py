# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A collection inside the frozen block walks only what the block allocated."""

from __future__ import annotations

import gc
from typing import Any

import pytest

from tests.support.gc_isolation import earlier_tests_heap_frozen


def test_a_full_collection_in_the_block_skips_what_was_alive_before_it() -> None:
    earlier = [[] for _ in range(20_000)]
    walked: list[int] = []

    def on_gc(phase: str, info: dict[str, Any]) -> None:
        if phase == "start" and info["generation"] == 2:
            walked.append(len(gc.get_objects()))

    gc.callbacks.append(on_gc)
    try:
        gc.collect()
        with earlier_tests_heap_frozen():
            assert gc.get_freeze_count() >= len(earlier)
            during = [[] for _ in range(100)]
            gc.collect()
        assert gc.get_freeze_count() == 0
    finally:
        gc.callbacks.remove(on_gc)

    unfrozen, frozen = walked
    assert unfrozen >= len(earlier)
    assert len(during) <= frozen < len(earlier)


def test_the_block_unfreezes_when_its_body_raises() -> None:
    with pytest.raises(RuntimeError, match="boom"), earlier_tests_heap_frozen():
        raise RuntimeError("boom")
    assert gc.get_freeze_count() == 0
