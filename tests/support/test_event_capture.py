# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cross-family capture preserves associations and owns subscriptions."""

from __future__ import annotations

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationRetryAttempt, InvocationStarted
from chrys.foundation.models.invocations import InvocationOrigin
from tests.support.event_capture import EventNormalizer, capture_event_sequence


@pytest.mark.parametrize("fail", [False, True])
async def test_capture_releases_subscriptions_and_retains_cross_type_order(fail: bool) -> None:
    bus = EventBus()
    start = InvocationStarted(
        agent_name="Child",
        tool_name="Explore",
        origin=InvocationOrigin("sub_agent", "", "random", None),
    )
    retry = InvocationRetryAttempt(
        agent_name="Child",
        message="reset",
        attempt=2,
        max_attempts=3,
        delay_seconds=7,
        origin=InvocationOrigin("sub_agent", "", "random", None),
    )
    try:
        async with capture_event_sequence(bus, type(start), type(retry), type(start)) as events:
            await bus.publish(start)
            await bus.publish(retry)
            if fail:
                raise RuntimeError("leave capture")
    except RuntimeError:
        assert fail
    assert events == [start, retry]
    assert events[0] is start
    normalizer = EventNormalizer()
    first = normalizer.event(start)
    second = normalizer.event(retry, clock_fields=("delay_seconds",))
    assert first[1]["origin"]["invocation_id"] == second[1]["origin"]["invocation_id"] == "id-1"
    assert second[1]["attempt"] == 2
    assert second[1]["max_attempts"] == 3
    assert second[1]["delay_seconds"] == "<clock>"
    await bus.publish(start)
    assert events == [start, retry]


def test_normalizer_preserves_identity_across_origin_parent_chain() -> None:
    parent = InvocationOrigin("turn", "session-random", "parent-random", None)
    child = InvocationOrigin("sub_agent", parent.session_id, "child-random", parent)
    normalizer = EventNormalizer()
    _, child_values = normalizer.event(InvocationStarted(origin=child, agent_name="Child", tool_name="Explore"))
    _, parent_values = normalizer.event(InvocationStarted(origin=parent, agent_name="Code", tool_name=""))
    assert child_values["origin"]["parent"] == parent_values["origin"]
    assert child_values["origin"]["session_id"] == parent_values["origin"]["session_id"] == "id-1"
    assert child_values["origin"]["invocation_id"] == "id-2"
    assert parent_values["origin"]["invocation_id"] == "id-3"
