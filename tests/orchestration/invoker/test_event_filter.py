# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Headless/ACP's existing allow/drop boundary for text and presentation facts."""

import pytest

from chrys.foundation.events.types import (
    InvocationMessage,
    InvocationPaused,
    InvocationPresentationAttemptAccepted,
    InvocationPresentationAttemptRejected,
    InvocationProgress,
    InvocationResumed,
    InvocationRetryAttempt,
    InvocationStarted,
    InvocationToolCallResult,
)
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.orchestration.session_host import allows_headless_event, allows_workflow_node_event
from tests.support.invocation_events import ORIGIN_IDS, ORIGINS, PHASES, PROSE_PHASES, projection_event


def test_headless_allows_main_text_and_child_activity_but_drops_child_prose() -> None:
    main = InvocationOrigin("turn", "session", "main", None)
    child = InvocationOrigin("sub_agent", "session", "child", main)
    prose = (InvocationMessage, InvocationPresentationAttemptAccepted, InvocationPresentationAttemptRejected)
    activity = (
        InvocationStarted,
        InvocationPaused,
        InvocationResumed,
        InvocationRetryAttempt,
        InvocationProgress,
        InvocationToolCallResult,
    )
    assert all(allows_headless_event(event_type(origin=main)) for event_type in prose)
    assert all(allows_headless_event(event_type(origin=child)) for event_type in activity)
    assert all(not allows_headless_event(event_type(origin=child)) for event_type in prose)


def test_headless_drops_a_workflow_node_and_its_children_entirely() -> None:
    node = InvocationOrigin("workflow_node", "session", "node", None)
    node_child = InvocationOrigin("sub_agent", "session", "node-child", node)
    facts = (
        InvocationMessage,
        InvocationStarted,
        InvocationPaused,
        InvocationResumed,
        InvocationRetryAttempt,
        InvocationProgress,
        InvocationToolCallResult,
    )
    assert all(not allows_headless_event(event_type(origin=node)) for event_type in facts)
    assert all(not allows_headless_event(event_type(origin=node_child)) for event_type in facts)


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("origin", ORIGINS, ids=ORIGIN_IDS)
def test_workflow_node_activity_is_the_node_rooted_complement_minus_child_prose(
    origin: InvocationOrigin, phase: str
) -> None:
    event = projection_event(origin, phase)
    node_rooted = origin.root.kind == "workflow_node"
    child_prose = origin.kind == "sub_agent" and phase in PROSE_PHASES

    assert allows_workflow_node_event(event) is (node_rooted and not child_prose)
    # The two filters never both admit an event: a stream asking for node activity adds to the headless set.
    assert not (allows_workflow_node_event(event) and allows_headless_event(event))
