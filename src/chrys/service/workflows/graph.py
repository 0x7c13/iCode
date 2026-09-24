# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Data-only graph the main process schedules on, built from a worker manifest.

The manifest is produced by our SDK, but it arrives from the worker process,
so its shape and referential integrity are checked here before anything is
scheduled on it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from chrys.service.workflows.sdk._builder import (
    KIND_AGENT,
    KIND_JOIN,
    KIND_LOOP,
    KIND_PYTHON,
    ON_EXHAUSTED_CONTINUE,
    ON_EXHAUSTED_FAIL,
    SCHEMA_VERSION,
)

NODE_KINDS = frozenset({KIND_AGENT, KIND_PYTHON, KIND_JOIN, KIND_LOOP})
ON_EXHAUSTED_VALUES = frozenset({ON_EXHAUSTED_CONTINUE, ON_EXHAUSTED_FAIL})


class ManifestError(ValueError):
    """The manifest is not a well-formed workflow graph."""


@dataclass(frozen=True, slots=True)
class ManifestWarning:
    code: str
    node_id: str
    message: str


def manifest_warnings(manifest: Mapping[str, Any]) -> tuple[ManifestWarning, ...]:
    """Validate diagnostic text before a frontend displays it."""
    warnings = manifest.get("warnings", [])
    if not isinstance(warnings, list):
        raise ManifestError("manifest warnings must be a list")
    result: list[ManifestWarning] = []
    for warning in warnings:
        if not isinstance(warning, dict) or any(
            not isinstance(warning.get(key), str) or not warning[key].strip() for key in ("code", "node_id", "message")
        ):
            raise ManifestError("manifest warnings require non-empty code, node_id and message strings")
        result.append(ManifestWarning(warning["code"], warning["node_id"], warning["message"]))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class RetrySpec:
    """Node retry policy with backoff measured in seconds."""

    max_attempts: int
    backoff: float


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    edge_id: str
    src: str
    dst: str
    conditional: bool
    switch_group: str | None
    switch_position: int | None
    switch_default: bool


@dataclass(frozen=True, slots=True)
class LoopSpec:
    entry: str
    exit: str
    body: tuple[str, ...]
    max_iterations: int
    on_exhausted: str


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Validated agent profile selector and optional node overrides."""

    profile: str
    model: str | None
    instructions_suffix: str

    @classmethod
    def from_manifest(cls, raw: Mapping[str, Any]) -> AgentSpec:
        profile = _string(raw, "profile")
        if not profile.strip():
            raise ManifestError("agent.profile must be a non-empty selector.")
        suffix = raw.get("instructions_suffix")
        if suffix is not None and not isinstance(suffix, str):
            raise ManifestError("agent.instructions_suffix must be a str or null.")
        return cls(profile, _optional_string(raw, "model"), suffix or "")


@dataclass(frozen=True, slots=True)
class NodeSpec:
    node_id: str
    kind: str
    parent_loop: str | None
    retry: RetrySpec
    timeout: float | None
    in_edges: tuple[str, ...]
    out_edges: tuple[str, ...]
    has_combine: bool
    loop: LoopSpec | None
    agent: AgentSpec | None = None
    fn_is_async: bool = False

    @property
    def is_loop(self) -> bool:
        return self.kind == KIND_LOOP

    def require_loop(self) -> LoopSpec:
        if self.loop is None:
            raise ManifestError(f"Node {self.node_id!r} has no loop specification.")
        return self.loop


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """Validated, immutable graph: what the scheduler and the runner share."""

    title: str
    start: str
    outputs: tuple[str, ...]
    nodes: Mapping[str, NodeSpec]
    node_order: tuple[str, ...]
    edges: Mapping[str, EdgeSpec]

    top_level: tuple[str, ...] = field(init=False)
    conditional_edges: Mapping[str, tuple[str, ...]] = field(init=False)
    switch_groups: Mapping[str, Mapping[str, tuple[str, ...]]] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "top_level", tuple(n for n in self.node_order if self.nodes[n].parent_loop is None))
        object.__setattr__(
            self,
            "conditional_edges",
            MappingProxyType(
                {
                    node.node_id: tuple(e for e in node.out_edges if self.edges[e].conditional)
                    for node in self.nodes.values()
                }
            ),
        )
        object.__setattr__(
            self,
            "switch_groups",
            MappingProxyType(
                {
                    node.node_id: MappingProxyType(
                        {group: tuple(ids) for group, ids in _switch_groups(node, self.edges).items()}
                    )
                    for node in self.nodes.values()
                }
            ),
        )

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> GraphSpec:
        if not isinstance(manifest, Mapping):
            raise ManifestError("manifest must be an object.")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ManifestError(f"manifest schema_version {manifest.get('schema_version')!r} is not {SCHEMA_VERSION}.")
        title = _string(manifest, "title")
        start = _string(manifest, "start")
        outputs = tuple(_string_list(manifest, "outputs"))
        raw_nodes = manifest.get("nodes")
        raw_edges = manifest.get("edges")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise ManifestError("manifest.nodes must be a non-empty list.")
        if not isinstance(raw_edges, list):
            raise ManifestError("manifest.edges must be a list.")

        node_ids: list[str] = []
        raw_by_id: dict[str, Mapping[str, Any]] = {}
        for raw in raw_nodes:
            if not isinstance(raw, Mapping):
                raise ManifestError("manifest.nodes entries must be objects.")
            node_id = _string(raw, "id")
            if node_id in raw_by_id:
                raise ManifestError(f"duplicate node id {node_id!r}.")
            node_ids.append(node_id)
            raw_by_id[node_id] = raw

        edges: dict[str, EdgeSpec] = {}
        in_edges: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        out_edges: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        pairs: set[tuple[str, str]] = set()
        for raw in raw_edges:
            if not isinstance(raw, Mapping):
                raise ManifestError("manifest.edges entries must be objects.")
            edge = _edge_from(raw)
            if edge.edge_id in edges:
                raise ManifestError(f"duplicate edge id {edge.edge_id!r}.")
            if edge.src not in raw_by_id or edge.dst not in raw_by_id:
                raise ManifestError(f"edge {edge.edge_id!r} references an unknown node.")
            if edge.src == edge.dst or (edge.src, edge.dst) in pairs:
                raise ManifestError(f"edge {edge.edge_id!r} is a self-edge or a duplicate.")
            pairs.add((edge.src, edge.dst))
            edges[edge.edge_id] = edge
            out_edges[edge.src].append(edge.edge_id)
            in_edges[edge.dst].append(edge.edge_id)

        nodes: dict[str, NodeSpec] = {}
        for node_id in node_ids:
            nodes[node_id] = _node_from(raw_by_id[node_id], in_edges[node_id], out_edges[node_id], raw_by_id)
        if start not in nodes or nodes[start].parent_loop is not None:
            raise ManifestError("manifest.start must name a top-level node.")
        for output in outputs:
            if output not in nodes or nodes[output].parent_loop is not None:
                raise ManifestError(f"output {output!r} must name a top-level node.")
        _check_scopes(nodes, edges, start)
        for node in nodes.values():
            for group, edge_ids in _switch_groups(node, edges).items():
                defaults = [eid for eid in edge_ids if edges[eid].switch_default]
                if group != node.node_id or len(defaults) != 1:
                    raise ManifestError(f"switch on {node.node_id!r} must have exactly one default edge.")
        return cls(
            title=title,
            start=start,
            outputs=outputs,
            nodes=nodes,
            node_order=tuple(node_ids),
            edges=edges,
        )


def _string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"manifest field {key!r} must be a non-empty str.")
    return value


def _string_list(raw: Mapping[str, Any], key: str) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ManifestError(f"manifest field {key!r} must be a list of str.")
    return value


def _optional_string(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ManifestError(f"manifest field {key!r} must be a str or null.")
    return value


def _edge_from(raw: Mapping[str, Any]) -> EdgeSpec:
    switch = raw.get("switch")
    group: str | None = None
    position: int | None = None
    default = False
    if switch is not None:
        if not isinstance(switch, Mapping):
            raise ManifestError("edge.switch must be an object or null.")
        group = _string(switch, "group")
        raw_position = switch.get("position")
        if raw_position is not None and (isinstance(raw_position, bool) or not isinstance(raw_position, int)):
            raise ManifestError("edge.switch.position must be an int or null.")
        position = raw_position
        default = switch.get("default") is True
        if default == (position is not None):
            raise ManifestError("edge.switch must be either a positioned case or the default.")
    conditional = raw.get("conditional")
    if not isinstance(conditional, bool):
        raise ManifestError("edge.conditional must be a bool.")
    if group is not None and not conditional:
        raise ManifestError("switch edges are conditional by definition.")
    return EdgeSpec(
        edge_id=_string(raw, "id"),
        src=_string(raw, "src"),
        dst=_string(raw, "dst"),
        conditional=conditional,
        switch_group=group,
        switch_position=position,
        switch_default=default,
    )


def _node_from(
    raw: Mapping[str, Any],
    in_edges: list[str],
    out_edges: list[str],
    raw_by_id: Mapping[str, Mapping[str, Any]],
) -> NodeSpec:
    node_id = _string(raw, "id")
    kind = _string(raw, "kind")
    if kind not in NODE_KINDS:
        raise ManifestError(f"node {node_id!r} has unknown kind {kind!r}.")
    parent_loop = _optional_string(raw, "parent_loop")
    if parent_loop is not None and (parent_loop not in raw_by_id or raw_by_id[parent_loop].get("kind") != KIND_LOOP):
        raise ManifestError(f"node {node_id!r} parent_loop {parent_loop!r} is not a loop node.")
    retry_raw = raw.get("retry")
    if not isinstance(retry_raw, Mapping):
        raise ManifestError(f"node {node_id!r} retry must be an object.")
    max_attempts = retry_raw.get("max_attempts")
    backoff = retry_raw.get("backoff")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ManifestError(f"node {node_id!r} retry.max_attempts must be an int >= 1.")
    if isinstance(backoff, bool) or not isinstance(backoff, int | float) or backoff < 0:
        raise ManifestError(f"node {node_id!r} retry.backoff must be a non-negative number.")
    timeout = raw.get("timeout")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, int | float) or timeout <= 0):
        raise ManifestError(f"node {node_id!r} timeout must be a positive number or null.")
    callable_raw = raw.get("callable")
    has_combine = kind == KIND_JOIN and callable_raw is not None
    agent: AgentSpec | None = None
    if kind == KIND_AGENT:
        agent_raw = raw.get("agent")
        if not isinstance(agent_raw, Mapping):
            raise ManifestError(f"agent node {node_id!r} needs an agent object.")
        agent = AgentSpec.from_manifest(agent_raw)
    loop: LoopSpec | None = None
    if kind == KIND_LOOP:
        loop_raw = raw.get("loop")
        if not isinstance(loop_raw, Mapping):
            raise ManifestError(f"loop node {node_id!r} needs a loop object.")
        body = tuple(_string_list(loop_raw, "body"))
        entry = _string(loop_raw, "entry")
        exit_ = _string(loop_raw, "exit")
        max_iterations = loop_raw.get("max_iterations")
        on_exhausted = loop_raw.get("on_exhausted")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ManifestError(f"loop node {node_id!r} max_iterations must be an int >= 1.")
        if on_exhausted not in ON_EXHAUSTED_VALUES:
            raise ManifestError(f"loop node {node_id!r} on_exhausted must be continue|fail.")
        if entry not in body or exit_ not in body:
            raise ManifestError(f"loop node {node_id!r} entry/exit must be body nodes.")
        for member in body:
            if member not in raw_by_id or raw_by_id[member].get("parent_loop") != node_id:
                raise ManifestError(f"loop node {node_id!r} body member {member!r} is not its child.")
        if parent_loop is not None:
            raise ManifestError(f"loop node {node_id!r} cannot be nested.")
        for child, child_raw in raw_by_id.items():  # the iteration barrier trusts body, so it must be complete
            if child_raw.get("parent_loop") == node_id and child not in body:
                raise ManifestError(f"loop node {node_id!r} body omits its child {child!r}.")
        loop = LoopSpec(entry=entry, exit=exit_, body=body, max_iterations=max_iterations, on_exhausted=on_exhausted)
    return NodeSpec(
        node_id=node_id,
        kind=kind,
        parent_loop=parent_loop,
        retry=RetrySpec(max_attempts=max_attempts, backoff=float(backoff)),
        timeout=None if timeout is None else float(timeout),
        in_edges=tuple(in_edges),
        out_edges=tuple(out_edges),
        has_combine=has_combine,
        loop=loop,
        agent=agent,
        fn_is_async=isinstance(callable_raw, Mapping) and callable_raw.get("async") is True,
    )


def _switch_groups(node: NodeSpec, edges: Mapping[str, EdgeSpec]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for edge_id in node.out_edges:
        group = edges[edge_id].switch_group
        if group is not None:
            groups.setdefault(group, []).append(edge_id)
    return groups


def _check_scopes(nodes: Mapping[str, NodeSpec], edges: Mapping[str, EdgeSpec], start: str) -> None:
    """Edges stay inside one scope; every scope is acyclic and reachable from its root."""
    for edge in edges.values():
        if nodes[edge.src].parent_loop != nodes[edge.dst].parent_loop:
            raise ManifestError(f"edge {edge.edge_id!r} crosses a loop boundary.")
    scopes: dict[str | None, list[str]] = {}
    for node in nodes.values():
        scopes.setdefault(node.parent_loop, []).append(node.node_id)
    for scope_id, members in scopes.items():
        member_set = set(members)
        if scope_id is None:
            roots = [node_id for node_id in members if not nodes[node_id].in_edges]
            if roots != [start]:
                raise ManifestError("manifest.start must be the only top-level node without in-edges.")
            root = start
        else:
            loop = nodes[scope_id].require_loop()
            root = loop.entry
            if nodes[root].in_edges:
                raise ManifestError(f"loop {scope_id!r} entry must not have in-edges.")
        seen = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for edge_id in nodes[current].out_edges:
                target = edges[edge_id].dst
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        if seen != member_set:
            raise ManifestError(f"nodes {sorted(member_set - seen)!r} are unreachable in scope {scope_id!r}.")
        _check_acyclic(members, nodes, edges)


def _check_acyclic(members: list[str], nodes: Mapping[str, NodeSpec], edges: Mapping[str, EdgeSpec]) -> None:
    color: dict[str, int] = {}
    for start in members:
        if color.get(start, 0):
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        color[start] = 1
        while stack:
            node_id, index = stack[-1]
            outgoing = nodes[node_id].out_edges
            if index < len(outgoing):
                stack[-1] = (node_id, index + 1)
                target = edges[outgoing[index]].dst
                state = color.get(target, 0)
                if state == 1:
                    raise ManifestError(f"cycle through {target!r}.")
                if state == 0:
                    color[target] = 1
                    stack.append((target, 0))
            else:
                color[node_id] = 2
                stack.pop()
