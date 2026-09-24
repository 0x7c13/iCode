# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""An invocation origin names the chat turn or workflow node it ultimately runs under."""

from __future__ import annotations

import pytest

from chrys.foundation.models.invocations import InvocationOrigin


def test_root_walks_to_the_outermost_bound_ancestor() -> None:
    node = InvocationOrigin("workflow_node", "s1", "node", None)
    child = InvocationOrigin("sub_agent", "s1", "child", node)
    grandchild = InvocationOrigin("sub_agent", "s1", "grandchild", child)

    assert grandchild.root is node
    assert child.root is node


@pytest.mark.parametrize("kind", ["turn", "sub_agent", "workflow_node"])
def test_an_unbound_origin_is_its_own_root(kind: str) -> None:
    origin = InvocationOrigin(kind, "s1", "alone", None)  # type: ignore[arg-type]

    assert origin.root is origin
