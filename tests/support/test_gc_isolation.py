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

    # Only collect inside the block: a collection outside it would walk the worker's whole heap.
    with earlier_tests_heap_frozen():
        assert gc.get_freeze_count() >= len(earlier)
        during = [[] for _ in range(100)]
        gc.callbacks.append(on_gc)
        try:
            gc.collect()
        finally:
            gc.callbacks.remove(on_gc)
    assert gc.get_freeze_count() == 0

    assert len(walked) == 1
    assert len(during) <= walked[0] < len(earlier)


def test_the_block_unfreezes_when_its_body_raises() -> None:
    with pytest.raises(RuntimeError, match="boom"), earlier_tests_heap_frozen():
        raise RuntimeError("boom")
    assert gc.get_freeze_count() == 0
