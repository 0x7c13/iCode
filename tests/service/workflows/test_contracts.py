# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Authoring validation, immutable adjacency and shared run status contracts."""

from __future__ import annotations

import pytest

from chrys.service.workflows.graph import AgentSpec, GraphSpec, ManifestError
from chrys.service.workflows.outcomes import (
    REASON_DEADLINE_EXCEEDED,
    REASON_INTERNAL_ERROR,
    REASON_SHUTDOWN,
    RunOutcome,
    run_status,
)
from chrys.service.workflows.sdk import WorkflowBuilder, WorkflowValidationError


def test_adjacency_is_ordered_frozen_and_reused() -> None:
    builder = WorkflowBuilder("fanout")
    root, left, right, output = [builder.python(name, lambda value: value) for name in ("root", "left", "right", "out")]
    builder.start(root)
    builder.edge(root, right)
    builder.edge(root, left)
    builder.join([left, right], output)
    builder.output(output)
    definition = builder.build().definition
    assert [edge.dst for edge in definition.out_edges("root")] == ["right", "left"]
    assert [edge.src for edge in definition.in_edges("join:out")] == ["left", "right"]
    assert definition.out_edges("root") is definition.out_edges("root")
    with pytest.raises(TypeError):
        definition.outgoing["root"] = ()  # type: ignore[index]


@pytest.mark.parametrize("kind", ["agent", "python", "loop", "body"])
def test_public_names_reserve_the_synthesized_join_prefix(kind: str) -> None:
    builder = WorkflowBuilder("reserved")
    with pytest.raises(WorkflowValidationError, match="reserved") as error:
        if kind == "agent":
            builder.agent("join:out", profile="Code")
        elif kind == "python":
            builder.python("join:out", lambda value: value)
        elif kind == "loop":
            builder.loop("join:out", lambda scope: (), lambda value: True, 1)  # type: ignore[arg-type]
        else:
            builder.loop(
                "loop", lambda scope: (scope.python("join:out", lambda value: value),) * 2, lambda value: True, 1
            )
    assert error.value.location == "join:out"


@pytest.mark.parametrize(
    "agent",
    [
        None,
        {},
        {"profile": ""},
        {"profile": " "},
        {"profile": 7},
        {"profile": "Code", "model": 7},
        {"profile": "Code", "model": ""},
        {"profile": "Code", "instructions_suffix": []},
    ],
)
def test_agent_metadata_is_validated_at_the_manifest_boundary(agent: object) -> None:
    builder = WorkflowBuilder("agent")
    node = builder.agent("review", profile="Code")
    builder.start(node)
    builder.output(node)
    manifest = builder.build().manifest()
    manifest["nodes"][0]["agent"] = agent
    with pytest.raises(ManifestError):
        GraphSpec.from_manifest(manifest)


def test_graph_retains_agent_metadata() -> None:
    builder = WorkflowBuilder("agent")
    node = builder.agent("review", profile="Code", model="fast", instructions_suffix="Be terse.")
    builder.start(node)
    builder.output(node)
    graph = GraphSpec.from_manifest(builder.build().manifest())
    assert graph.nodes["review"].agent == AgentSpec("Code", "fast", "Be terse.")


@pytest.mark.parametrize(
    ("outcome", "reason", "active", "status"),
    [
        (RunOutcome.COMPLETED.value, "", False, "completed"),
        (RunOutcome.CANCELLED.value, "", False, "cancelled"),
        (RunOutcome.CANCELLED.value, REASON_DEADLINE_EXCEEDED, False, "failed"),
        (RunOutcome.CANCELLED.value, REASON_INTERNAL_ERROR, False, "failed"),
        (RunOutcome.CANCELLED.value, REASON_SHUTDOWN, False, "interrupted"),
        (RunOutcome.ORPHANED.value, "", False, "interrupted"),
        (RunOutcome.STORAGE_FAILED.value, "", False, "failed"),
        (None, None, True, "running"),
        (None, None, False, "interrupted"),
    ],
)
def test_run_status(outcome: str | None, reason: str | None, active: bool, status: str) -> None:
    assert run_status(outcome, reason, active=active) == status


@pytest.mark.parametrize("warnings", [None, {}, ["text"], [{"code": "warning", "node_id": "n", "message": 7}]])
def test_manifest_warning_messages_are_validated(warnings: object) -> None:
    from chrys.service.workflows.graph import ManifestError, manifest_warnings

    with pytest.raises(ManifestError, match="manifest warnings"):
        manifest_warnings({"warnings": warnings})
