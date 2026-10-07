# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Explicit-graph builder of the workflow SDK (pure stdlib, Python 3.9).

``WorkflowBuilder`` records nodes and edges; ``build()`` runs the structural
validation the worker can decide without chrys (unique names, acyclic,
reachable, switch/join/loop shape, callback arity) and freezes a
``WorkflowDefinition`` that keeps the live callables. ``Workflow.manifest()``
is the data-only projection the main process consumes.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Optional

from ._values import SourceValue, WorkflowValue

SCHEMA_VERSION = 1

KIND_AGENT = "agent"
KIND_PYTHON = "python"
KIND_JOIN = "join"
KIND_LOOP = "loop"

ON_EXHAUSTED_CONTINUE = "continue"
ON_EXHAUSTED_FAIL = "fail"

DEFAULT_AGENT_MAX_ATTEMPTS = 3
DEFAULT_PYTHON_TIMEOUT = 300.0
"""Default python-node attempt timeout, in seconds."""

WARNING_LOOP_EXIT_ALL_CONDITIONAL = "loop_exit_all_conditional"

_ENTRY_ENV = "CHRYS_WORKFLOW_ENTRY"  # set by the worker host to the entry's file name before the entry runs
_SDK_DIR = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))


class WorkflowValidationError(ValueError):
    """A workflow file violates the structural contract; ``location`` names the offender.

    ``site`` is the ``(file, line)`` where the offending node or edge was declared, when ``build()`` finds the
    fault and the declaration was in a file; an error raised while declaring is located by its traceback.
    """

    def __init__(self, message: str, *, location: Optional[str] = None, site: Optional[tuple[str, int]] = None) -> None:
        super().__init__(message)
        self.location = location
        self.site = site

    def to_dict(self) -> dict[str, Any]:
        file, line = self.site if self.site is not None else (None, None)
        return {"message": str(self), "location": self.location, "file": file, "line": line}


def _declaration_site() -> Optional[tuple[str, int]]:
    """The ``(file, line)`` of the innermost caller outside the SDK whose code comes from a file.

    A file is the entry as the host names it or an existing absolute path; code that a workflow builds with
    ``compile``/``exec`` under a made-up name is passed over for the line that ran it.
    """
    entry = os.environ.get(_ENTRY_ENV)
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_code.co_filename
        if os.path.normcase(os.path.dirname(os.path.abspath(name))) != _SDK_DIR and (
            name == entry or (os.path.isabs(name) and os.path.isfile(name))
        ):
            return name, frame.f_lineno
        frame = frame.f_back
    return None


@dataclass(frozen=True)
class Retry:
    """Retry policy; ``max_attempts`` counts the first execution, and ``backoff`` is in seconds."""

    max_attempts: int
    backoff: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.max_attempts, int) or isinstance(self.max_attempts, bool) or self.max_attempts < 1:
            raise WorkflowValidationError("Retry.max_attempts must be an int >= 1 (it counts the first execution).")
        if (
            isinstance(self.backoff, bool)
            or not isinstance(self.backoff, (int, float))
            or not math.isfinite(self.backoff)
            or self.backoff < 0
        ):
            raise WorkflowValidationError("Retry.backoff must be a finite non-negative number.")


@dataclass(frozen=True)
class NodeHandle:
    """Opaque reference to a node; only meaningful with the builder that made it."""

    node_id: str
    _graph: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class LoopDefinition:
    entry: str
    exit: str
    body: tuple[str, ...]
    until: Callable[[WorkflowValue], bool]
    max_iterations: int
    on_exhausted: str


@dataclass(frozen=True)
class NodeDefinition:
    node_id: str
    kind: str
    parent_loop: Optional[str]
    retry: Retry
    timeout: Optional[float]
    fn: Optional[Callable[..., Any]] = None
    fn_arity: int = 0
    fn_is_async: bool = False
    profile: Optional[str] = None
    model: Optional[str] = None
    instructions_suffix: Optional[str] = None
    loop: Optional[LoopDefinition] = None
    site: Optional[tuple[str, int]] = field(default=None, compare=False, repr=False)  # where it was declared


@dataclass(frozen=True)
class EdgeDefinition:
    edge_id: str
    src: str
    dst: str
    predicate: Optional[Callable[[WorkflowValue], bool]] = None
    switch_group: Optional[str] = None
    switch_position: Optional[int] = None
    switch_default: bool = False
    site: Optional[tuple[str, int]] = field(default=None, compare=False, repr=False)  # where it was declared

    @property
    def conditional(self) -> bool:
        return self.predicate is not None or self.switch_group is not None


@dataclass(frozen=True)
class WorkflowDefinition:
    """Frozen graph with live callables; lives only in the worker process."""

    title: str
    description: Optional[str]
    start: str
    outputs: tuple[str, ...]
    nodes: dict[str, NodeDefinition]
    node_order: tuple[str, ...]
    edges: dict[str, EdgeDefinition]
    edge_order: tuple[str, ...]
    warnings: tuple[dict[str, Any], ...]
    incoming: Mapping[str, tuple[EdgeDefinition, ...]]
    outgoing: Mapping[str, tuple[EdgeDefinition, ...]]

    def in_edges(self, node_id: str) -> tuple[EdgeDefinition, ...]:
        return self.incoming[node_id]

    def out_edges(self, node_id: str) -> tuple[EdgeDefinition, ...]:
        return self.outgoing[node_id]


class Workflow:
    """Result of ``build()``; the module-level ``workflow`` variable holds one."""

    def __init__(self, definition: WorkflowDefinition) -> None:
        self._definition = definition

    @property
    def definition(self) -> WorkflowDefinition:
        return self._definition

    def manifest(self) -> dict[str, Any]:
        """Data-only projection: what the main process, store and events carry."""
        d = self._definition
        nodes = []
        for node_id in d.node_order:
            node = d.nodes[node_id]
            entry: dict[str, Any] = {
                "id": node.node_id,
                "kind": node.kind,
                "parent_loop": node.parent_loop,
                "timeout": node.timeout,
                "retry": {"max_attempts": node.retry.max_attempts, "backoff": node.retry.backoff},
                "agent": None,
                "callable": None,
                "loop": None,
            }
            if node.kind == KIND_AGENT:
                entry["agent"] = {
                    "profile": node.profile,
                    "model": node.model,
                    "instructions_suffix": node.instructions_suffix,
                }
            if node.fn is not None:
                entry["callable"] = {
                    "name": _callable_name(node.fn),
                    "arity": node.fn_arity,
                    "async": node.fn_is_async,
                }
            if node.loop is not None:
                entry["loop"] = {
                    "entry": node.loop.entry,
                    "exit": node.loop.exit,
                    "body": list(node.loop.body),
                    "until": _callable_name(node.loop.until),
                    "max_iterations": node.loop.max_iterations,
                    "on_exhausted": node.loop.on_exhausted,
                }
            nodes.append(entry)
        edges = []
        for edge_id in d.edge_order:
            edge = d.edges[edge_id]
            switch = None
            if edge.switch_group is not None:
                switch = {
                    "group": edge.switch_group,
                    "position": edge.switch_position,
                    "default": edge.switch_default,
                }
            edges.append(
                {
                    "id": edge.edge_id,
                    "src": edge.src,
                    "dst": edge.dst,
                    "conditional": edge.conditional,
                    "predicate": _callable_name(edge.predicate) if edge.predicate is not None else None,
                    "switch": switch,
                }
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "title": d.title,
            "description": d.description,
            "start": d.start,
            "outputs": list(d.outputs),
            "nodes": nodes,
            "edges": edges,
            "warnings": [dict(w) for w in d.warnings],
        }


def _callable_name(fn: Any) -> str:
    # SDK boundary: user callables are arbitrary objects.
    name = getattr(fn, "__qualname__", None)
    if isinstance(name, str):
        return name
    return type(fn).__name__


def _callable_shape(fn: Any, *, what: str, location: str, allowed_arities: tuple[int, ...]) -> tuple[int, bool]:
    """Return ``(positional_arity, is_async)`` or raise on an illegal shape, located at the node or edge *location*."""
    if not callable(fn):
        raise WorkflowValidationError(f"{what} must be callable, got {type(fn).__name__}.", location=location)
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise WorkflowValidationError(f"{what} has no inspectable signature.", location=location) from exc
    positional = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise WorkflowValidationError(f"{what} must not take *args or **kwargs.", location=location)
        if parameter.kind == parameter.KEYWORD_ONLY:
            if parameter.default is parameter.empty:
                raise WorkflowValidationError(f"{what} must not require keyword-only parameters.", location=location)
            continue
        if parameter.default is not parameter.empty:
            raise WorkflowValidationError(
                f"{what} must declare exactly its positional parameters without defaults.", location=location
            )
        positional += 1
    if positional not in allowed_arities:
        allowed = " or ".join(str(n) for n in allowed_arities)
        raise WorkflowValidationError(
            f"{what} must take {allowed} positional parameter(s), it takes {positional}.", location=location
        )
    # An object with an ``async def __call__`` is as async as a coroutine function.
    is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(type(fn).__call__)
    return positional, is_async


def _require_sync(fn: Any, *, what: str, location: str, allowed_arities: tuple[int, ...]) -> int:
    arity, is_async = _callable_shape(fn, what=what, location=location, allowed_arities=allowed_arities)
    if is_async:
        raise WorkflowValidationError(f"{what} must be a sync function.", location=location)
    return arity


class _GraphState:
    """Mutable build-time graph shared by the builder and every body scope."""

    def __init__(self, title: str, description: Optional[str]) -> None:
        self.title = title
        self.description = description
        self.nodes: dict[str, NodeDefinition] = {}
        self.node_order: list[str] = []
        self.edges: dict[str, EdgeDefinition] = {}
        self.edge_order: list[str] = []
        self.edge_pairs: set[tuple[str, str]] = set()
        self.start: Optional[str] = None
        self.outputs: list[str] = []
        self.building_loop: Optional[str] = None
        self.built = False
        self.warnings: list[dict[str, Any]] = []

    def add_node(self, node: NodeDefinition) -> NodeHandle:
        if self.built:
            raise WorkflowValidationError("The builder is frozen after build().", location=node.node_id)
        if node.node_id in self.nodes:
            raise WorkflowValidationError(f"Node name {node.node_id!r} is already used.", location=node.node_id)
        self.nodes[node.node_id] = node
        self.node_order.append(node.node_id)
        return NodeHandle(node.node_id, self)

    def add_edge(self, edge: EdgeDefinition) -> None:
        if self.built:
            raise WorkflowValidationError("The builder is frozen after build().", location=edge.edge_id)
        pair = (edge.src, edge.dst)
        if pair in self.edge_pairs:
            raise WorkflowValidationError(
                f"Edge {edge.src!r} -> {edge.dst!r} is declared twice.", location=edge.edge_id
            )
        self.edge_pairs.add(pair)
        self.edges[edge.edge_id] = edge
        self.edge_order.append(edge.edge_id)


def _edge_id(src: str, dst: str) -> str:
    return f"{src}->{dst}"


class BuilderScope:
    """Node/edge methods shared by the top-level builder and loop bodies."""

    def __init__(self, state: _GraphState, scope_id: Optional[str]) -> None:
        self._state = state
        self._scope_id = scope_id

    # ----------------------------------------------------------------- nodes

    def agent(
        self,
        name: str,
        *,
        profile: str,
        model: Optional[str] = None,
        instructions_suffix: Optional[str] = None,
        timeout: Optional[float] = None,
        retry: Optional[Retry] = None,
    ) -> NodeHandle:
        """Add an agent node; timeout is in seconds per attempt, or None for no deadline."""
        _check_name(name)
        if not isinstance(profile, str) or not profile.strip():
            raise WorkflowValidationError("profile= must be a non-empty profile selector.", location=name)
        if model is not None and (not isinstance(model, str) or not model):
            raise WorkflowValidationError("model= must be a non-empty selector or None.", location=name)
        if instructions_suffix is not None and not isinstance(instructions_suffix, str):
            raise WorkflowValidationError("instructions_suffix= must be a str or None.", location=name)
        return self._state.add_node(
            NodeDefinition(
                node_id=name,
                kind=KIND_AGENT,
                parent_loop=self._scope_id,
                retry=retry if retry is not None else Retry(max_attempts=DEFAULT_AGENT_MAX_ATTEMPTS),
                timeout=_check_timeout(timeout, name),
                profile=profile,
                model=model,
                instructions_suffix=instructions_suffix,
                site=_declaration_site(),
            )
        )

    def python(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        timeout: Optional[float] = DEFAULT_PYTHON_TIMEOUT,
        retry: Optional[Retry] = None,
    ) -> NodeHandle:
        """Add a Python body with signature ``(value)`` or ``(value, ctx)``.

        Both signatures receive a WorkflowValue; read its ``text`` or ``data``.
        Return a WorkflowValue or a string (converted to WorkflowValue.text).
        Timeout is in seconds per attempt, or None for no deadline.
        """
        _check_name(name)
        arity, is_async = _callable_shape(
            fn,
            what=f"python node {name!r} body (value: WorkflowValue[, ctx: NodeContext])",
            location=name,
            allowed_arities=(1, 2),
        )
        return self._state.add_node(
            NodeDefinition(
                node_id=name,
                kind=KIND_PYTHON,
                parent_loop=self._scope_id,
                retry=retry if retry is not None else Retry(max_attempts=1),
                timeout=_check_timeout(timeout, name),
                fn=fn,
                fn_arity=arity,
                fn_is_async=is_async,
                site=_declaration_site(),
            )
        )

    # ----------------------------------------------------------------- edges

    def edge(self, src: NodeHandle, dst: NodeHandle, *, when: Optional[Callable[[WorkflowValue], bool]] = None) -> None:
        src_id, dst_id = self._endpoints(src, dst)
        if when is not None:
            _require_sync(
                when,
                what=f"edge predicate {src_id!r} -> {dst_id!r}",
                location=_edge_id(src_id, dst_id),
                allowed_arities=(1,),
            )
        self._state.add_edge(
            EdgeDefinition(
                edge_id=_edge_id(src_id, dst_id), src=src_id, dst=dst_id, predicate=when, site=_declaration_site()
            )
        )

    def switch(
        self,
        src: NodeHandle,
        cases: Sequence[tuple[Callable[[WorkflowValue], bool], NodeHandle]],
        default: NodeHandle,
    ) -> None:
        src_id = self._own(src)
        if not isinstance(default, NodeHandle):
            raise WorkflowValidationError("switch(default=...) is required and must be a node.", location=src_id)
        for edge in self._state.edges.values():
            if edge.switch_group == src_id:
                raise WorkflowValidationError(f"Node {src_id!r} already has a switch.", location=src_id)
        for position, case in enumerate(cases):
            if not isinstance(case, tuple) or len(case) != 2:
                raise WorkflowValidationError("switch cases must be (predicate, node) pairs.", location=src_id)
            predicate, target = case
            _, dst_id = self._endpoints(src, target)
            _require_sync(
                predicate,
                what=f"switch case {position} of {src_id!r}",
                location=_edge_id(src_id, dst_id),
                allowed_arities=(1,),
            )
            self._state.add_edge(
                EdgeDefinition(
                    edge_id=_edge_id(src_id, dst_id),
                    src=src_id,
                    dst=dst_id,
                    predicate=predicate,
                    switch_group=src_id,
                    switch_position=position,
                    site=_declaration_site(),
                )
            )
        _, default_id = self._endpoints(src, default)
        self._state.add_edge(
            EdgeDefinition(
                edge_id=_edge_id(src_id, default_id),
                src=src_id,
                dst=default_id,
                switch_group=src_id,
                switch_default=True,
                site=_declaration_site(),
            )
        )

    def join(
        self,
        sources: Sequence[NodeHandle],
        dst: NodeHandle,
        *,
        combine: Optional[Callable[[list[SourceValue]], Any]] = None,
    ) -> None:
        """Join the sources; a custom synchronous combine has a fixed 30-second deadline.

        The internal join node has no configurable body timeout. Its combine is an evaluation,
        unlike a python/agent body for which an explicit timeout=None means no deadline.
        """
        dst_id = self._own(dst)
        if not sources:
            raise WorkflowValidationError(f"join into {dst_id!r} needs at least one source.", location=dst_id)
        if combine is not None:
            _require_sync(combine, what=f"join combine into {dst_id!r}", location=dst_id, allowed_arities=(1,))
        join_id = f"join:{dst_id}"
        join_handle = self._state.add_node(
            NodeDefinition(
                node_id=join_id,
                kind=KIND_JOIN,
                parent_loop=self._scope_id,
                retry=Retry(max_attempts=1),
                timeout=None,
                fn=combine,
                fn_arity=1 if combine is not None else 0,
                site=_declaration_site(),
            )
        )
        for source in sources:
            self.edge(source, join_handle)
        self.edge(join_handle, dst)

    def chain(self, *nodes: NodeHandle) -> None:
        if len(nodes) < 2:
            raise WorkflowValidationError("chain() needs at least two nodes.")
        for src, dst in zip(nodes, nodes[1:]):
            self.edge(src, dst)

    # --------------------------------------------------------------- helpers

    def _own(self, handle: NodeHandle) -> str:
        if not isinstance(handle, NodeHandle):
            raise WorkflowValidationError(f"Expected a node handle, got {type(handle).__name__}.")
        if handle._graph is not self._state:
            raise WorkflowValidationError(
                f"Node handle {handle.node_id!r} belongs to another builder.", location=handle.node_id
            )
        node = self._state.nodes.get(handle.node_id)
        if node is None:
            raise WorkflowValidationError(f"Unknown node {handle.node_id!r}.", location=handle.node_id)
        if node.parent_loop != self._scope_id:
            where = "the loop body" if self._scope_id is not None else "the top-level graph"
            raise WorkflowValidationError(
                f"Node {handle.node_id!r} does not belong to {where}; edges cannot cross a loop boundary.",
                location=handle.node_id,
            )
        return node.node_id

    def _endpoints(self, src: NodeHandle, dst: NodeHandle) -> tuple[str, str]:
        src_id = self._own(src)
        dst_id = self._own(dst)
        if src_id == dst_id:
            raise WorkflowValidationError(f"Self-edge on {src_id!r} would form a cycle.", location=src_id)
        return src_id, dst_id


class WorkflowBuilder(BuilderScope):
    """Top-level builder: nodes/edges plus loops, start, outputs and ``build()``."""

    def __init__(self, title: str, *, description: Optional[str] = None) -> None:
        if not isinstance(title, str) or not title.strip():
            raise WorkflowValidationError("WorkflowBuilder title must be a non-empty str.")
        if description is not None and not isinstance(description, str):
            raise WorkflowValidationError("description= must be a str or None.")
        super().__init__(_GraphState(title, description), None)

    def loop(
        self,
        name: str,
        body: Callable[[BuilderScope], tuple[NodeHandle, NodeHandle]],
        until: Callable[[WorkflowValue], bool],
        max_iterations: int,
        on_exhausted: str = ON_EXHAUSTED_CONTINUE,
    ) -> NodeHandle:
        _check_name(name)
        site = _declaration_site()
        state = self._state
        if state.building_loop is not None:
            raise WorkflowValidationError(
                f"Nested loops are not supported: {name!r} declared inside {state.building_loop!r}.",
                location=name,
            )
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise WorkflowValidationError("loop max_iterations must be an int >= 1.", location=name)
        if on_exhausted not in (ON_EXHAUSTED_CONTINUE, ON_EXHAUSTED_FAIL):
            raise WorkflowValidationError("loop on_exhausted must be 'continue' or 'fail'.", location=name)
        _require_sync(until, what=f"loop {name!r} until", location=name, allowed_arities=(1,))
        if not callable(body):
            raise WorkflowValidationError("loop body must be callable.", location=name)
        if name in state.nodes:
            raise WorkflowValidationError(f"Node name {name!r} is already used.", location=name)
        state.building_loop = name
        try:
            returned = body(BuilderScope(state, name))
        finally:
            state.building_loop = None
        if not isinstance(returned, tuple) or len(returned) != 2:
            raise WorkflowValidationError(f"loop {name!r} body must return (entry, exit).", location=name)
        entry, exit_ = returned
        body_ids = tuple(nid for nid in state.node_order if state.nodes[nid].parent_loop == name)
        for handle in (entry, exit_):
            if not isinstance(handle, NodeHandle) or handle._graph is not state or handle.node_id not in body_ids:
                raise WorkflowValidationError(f"loop {name!r} entry/exit must be nodes of its body.", location=name)
        loop = LoopDefinition(
            entry=entry.node_id,
            exit=exit_.node_id,
            body=body_ids,
            until=until,
            max_iterations=max_iterations,
            on_exhausted=on_exhausted,
        )
        return state.add_node(
            NodeDefinition(
                node_id=name,
                kind=KIND_LOOP,
                parent_loop=None,
                retry=Retry(max_attempts=1),
                timeout=None,
                loop=loop,
                site=site,
            )
        )

    def start(self, node: NodeHandle) -> None:
        self._top_level_only("start")
        node_id = self._own(node)
        if self._state.start is not None:
            raise WorkflowValidationError("start() may be called once.", location=node_id)
        self._state.start = node_id

    def output(self, node: NodeHandle) -> None:
        self._top_level_only("output")
        node_id = self._own(node)
        if node_id in self._state.outputs:
            raise WorkflowValidationError(f"output({node_id!r}) declared twice.", location=node_id)
        self._state.outputs.append(node_id)

    def build(self) -> Workflow:
        state = self._state
        if state.built:
            raise WorkflowValidationError("build() may be called once.")
        if state.start is None:
            raise WorkflowValidationError("start() was never called.")
        if not state.outputs:
            raise WorkflowValidationError("At least one output() is required.")
        incoming: dict[str, list[EdgeDefinition]] = {node_id: [] for node_id in state.node_order}
        outgoing: dict[str, list[EdgeDefinition]] = {node_id: [] for node_id in state.node_order}
        for edge_id in state.edge_order:
            edge = state.edges[edge_id]
            incoming[edge.dst].append(edge)
            outgoing[edge.src].append(edge)
        definition = WorkflowDefinition(
            title=state.title,
            description=state.description,
            start=state.start,
            outputs=tuple(state.outputs),
            nodes=dict(state.nodes),
            node_order=tuple(state.node_order),
            edges=dict(state.edges),
            edge_order=tuple(state.edge_order),
            warnings=(),
            incoming=MappingProxyType({node_id: tuple(edges) for node_id, edges in incoming.items()}),
            outgoing=MappingProxyType({node_id: tuple(edges) for node_id, edges in outgoing.items()}),
        )
        warnings = _validate(definition)
        state.built = True
        return Workflow(replace(definition, warnings=tuple(warnings)))

    def _top_level_only(self, what: str) -> None:
        if self._state.building_loop is not None:
            raise WorkflowValidationError(
                f"{what}() cannot be used inside a loop body.", location=self._state.building_loop
            )


def _check_name(name: Any) -> None:
    if not isinstance(name, str) or not name.strip():
        raise WorkflowValidationError("Node names must be non-empty strings.")
    if name.startswith("join:"):
        raise WorkflowValidationError("The 'join:' node-name prefix is reserved for synthesized joins.", location=name)
    if "->" in name:
        raise WorkflowValidationError(
            "Node names cannot contain '->'; edge ids are written as src->dst.", location=name
        )


def _check_timeout(timeout: Any, name: str) -> Optional[float]:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise WorkflowValidationError("timeout must be a finite positive number or None.", location=name)
    return float(timeout)


def _validate(d: WorkflowDefinition) -> list[dict[str, Any]]:
    """Structural checks that need the whole graph; returns build warnings."""
    warnings: list[dict[str, Any]] = []
    for node in d.nodes.values():
        if node.kind != KIND_JOIN:
            continue
        (dst_id,) = [e.dst for e in d.out_edges(node.node_id)]
        extra = [e for e in d.in_edges(dst_id) if e.src != node.node_id]
        if extra:
            others = [e.src for e in extra]
            raise WorkflowValidationError(
                f"join target {dst_id!r} must not have other direct in-edges (from {others!r}).",
                location=dst_id,
                site=extra[0].site,
            )
    scopes: dict[Optional[str], list[str]] = {None: []}
    for node_id in d.node_order:
        scopes.setdefault(d.nodes[node_id].parent_loop, []).append(node_id)
    for scope_id, members in scopes.items():
        _check_acyclic(d, members)
        if scope_id is None:
            root = d.start
        else:
            loop = d.nodes[scope_id].loop
            if loop is None:
                raise WorkflowValidationError("A loop scope requires a loop definition.")
            root = loop.entry
        unreachable = _unreachable(d, members, root)
        if unreachable:
            where = "start" if scope_id is None else f"loop {scope_id!r} entry"
            first = min(unreachable)
            raise WorkflowValidationError(
                f"Nodes {sorted(unreachable)!r} are not reachable from {where}.",
                location=first,
                site=d.nodes[first].site,
            )
    for node in d.nodes.values():
        if node.loop is None:
            continue
        exit_edges = d.in_edges(node.loop.exit)
        if exit_edges and all(e.conditional for e in exit_edges):
            warnings.append(
                {
                    "code": WARNING_LOOP_EXIT_ALL_CONDITIONAL,
                    "node_id": node.node_id,
                    "message": f"loop {node.node_id!r} exit {node.loop.exit!r} only has conditional in-edges; "
                    "an iteration with no exit value fails the run with loop_no_value.",
                }
            )
    return warnings


def _check_acyclic(d: WorkflowDefinition, members: list[str]) -> None:
    """Iterative three-colour DFS: a long chain must not depend on the recursion limit."""
    member_set = set(members)
    color: dict[str, int] = {}
    for start in members:
        if color.get(start, 0):
            continue
        color[start] = 1
        trail = [start]
        stack = [iter(d.out_edges(start))]
        while stack:
            edge = next(stack[-1], None)
            if edge is None:
                stack.pop()
                color[trail.pop()] = 2
                continue
            if edge.dst not in member_set:
                continue
            state = color.get(edge.dst, 0)
            if state == 1:
                cycle = [*trail[trail.index(edge.dst) :], edge.dst]
                raise WorkflowValidationError(
                    f"Cycle detected: {' -> '.join(cycle)}. Use wf.loop() for repetition.",
                    location=edge.dst,
                    site=edge.site,
                )
            if state == 0:
                color[edge.dst] = 1
                trail.append(edge.dst)
                stack.append(iter(d.out_edges(edge.dst)))


def _unreachable(d: WorkflowDefinition, members: list[str], root: str) -> set[str]:
    member_set = set(members)
    seen = {root}
    stack = [root]
    while stack:
        current = stack.pop()
        for edge in d.out_edges(current):
            if edge.dst in member_set and edge.dst not in seen:
                seen.add(edge.dst)
                stack.append(edge.dst)
    return member_set - seen
