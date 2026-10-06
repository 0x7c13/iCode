# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow SDK: builder contract, structural validation, the data-only manifest, and the graph reader."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import chrys.service.workflows.sdk as sdk
from chrys.service.workflows.graph import GraphSpec, ManifestError
from chrys.service.workflows.sdk import (
    BuilderScope,
    NodeContext,
    NodeHandle,
    Retry,
    SourceValue,
    Workflow,
    WorkflowBuilder,
    WorkflowValidationError,
    WorkflowValue,
)
from chrys.service.workflows.values import canonical_json
from chrys.workflows import __all__ as facade_exports
from tests.service.workflows.driver import body_fn, yes

FIXTURE = Path(__file__).parent / "fixtures" / "code-review.py"
AUTHORING_NAMES = [
    "Answer",
    "BuilderScope",
    "NodeContext",
    "NodeHandle",
    "Option",
    "Question",
    "Retry",
    "SourceValue",
    "Workflow",
    "WorkflowBuilder",
    "WorkflowValue",
]


def load_golden() -> Workflow:
    spec = importlib.util.spec_from_file_location("code_review_fixture", FIXTURE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    workflow = module.workflow
    assert isinstance(workflow, Workflow)
    return workflow


def _node(manifest: dict[str, Any], node_id: str) -> dict[str, Any]:
    return next(node for node in manifest["nodes"] if node["id"] == node_id)


def _edge(manifest: dict[str, Any], edge_id: str) -> dict[str, Any]:
    return next(edge for edge in manifest["edges"] if edge["id"] == edge_id)


def test_facade_and_sdk_export_exactly_the_authoring_names() -> None:
    assert facade_exports == AUTHORING_NAMES
    assert sdk.__all__ == AUTHORING_NAMES


def test_golden_example_manifest() -> None:
    manifest = load_golden().manifest()

    assert manifest["schema_version"] == 1
    assert manifest["title"] == "代码评审"
    assert manifest["start"] == "准备上下文"
    assert manifest["outputs"] == ["排版报告", "整改建议"]
    assert manifest["warnings"] == []
    assert [node["id"] for node in manifest["nodes"]] == [
        "准备上下文",
        "分发",
        "正确性评审",
        "安全评审",
        "性能评审",
        "汇总",
        "裁决",
        "join:汇总",
        "评审轮",
        "排版报告",
        "整改建议",
    ]
    assert [edge["id"] for edge in manifest["edges"]] == [
        "分发->正确性评审",
        "分发->安全评审",
        "分发->性能评审",
        "正确性评审->join:汇总",
        "安全评审->join:汇总",
        "性能评审->join:汇总",
        "join:汇总->汇总",
        "汇总->裁决",
        "准备上下文->评审轮",
        "评审轮->排版报告",
        "评审轮->整改建议",
    ]

    security = _node(manifest, "安全评审")
    assert security["kind"] == "agent"
    assert security["parent_loop"] == "评审轮"
    assert security["retry"] == {"max_attempts": 4, "backoff": 10.0}
    assert security["agent"] == {"profile": "QA", "model": "glm-4.7", "instructions_suffix": None}
    assert security["callable"] is None
    assert _node(manifest, "性能评审")["timeout"] == 600.0
    assert _node(manifest, "正确性评审")["retry"] == {"max_attempts": 3, "backoff": 0.0}

    summary = _node(manifest, "汇总")
    assert summary["kind"] == "python"
    assert summary["callable"] == {"name": "summarize", "arity": 2, "async": False}
    assert summary["timeout"] == 300.0
    assert summary["retry"] == {"max_attempts": 1, "backoff": 0.0}

    join = _node(manifest, "join:汇总")
    assert join["kind"] == "join"
    assert join["parent_loop"] == "评审轮"
    assert join["callable"] == {"name": "merge_reviews", "arity": 1, "async": False}

    loop = _node(manifest, "评审轮")
    assert loop["kind"] == "loop"
    assert loop["parent_loop"] is None
    assert loop["loop"] == {
        "entry": "分发",
        "exit": "裁决",
        "body": ["分发", "正确性评审", "安全评审", "性能评审", "汇总", "裁决", "join:汇总"],
        "until": "passed",
        "max_iterations": 3,
        "on_exhausted": "continue",
    }

    case = _edge(manifest, "评审轮->排版报告")
    assert case["conditional"] is True
    assert case["predicate"] == "passed"
    assert case["switch"] == {"group": "评审轮", "position": 0, "default": False}
    default = _edge(manifest, "评审轮->整改建议")
    assert default["conditional"] is True
    assert default["predicate"] is None
    assert default["switch"] == {"group": "评审轮", "position": None, "default": True}
    plain = _edge(manifest, "汇总->裁决")
    assert plain["conditional"] is False
    assert plain["predicate"] is None
    assert plain["switch"] is None

    assert canonical_json(manifest)


def test_golden_example_reads_into_a_graph_spec() -> None:
    graph = GraphSpec.from_manifest(load_golden().manifest())

    assert graph.top_level == ("准备上下文", "评审轮", "排版报告", "整改建议")
    assert graph.nodes["join:汇总"].has_combine is True
    assert graph.nodes["汇总"].has_combine is False
    assert graph.nodes["汇总"].in_edges == ("join:汇总->汇总",)
    assert graph.switch_groups["评审轮"] == {"评审轮": ("评审轮->排版报告", "评审轮->整改建议")}
    loop = graph.nodes["评审轮"].loop
    assert loop is not None and (loop.entry, loop.exit, loop.max_iterations) == ("分发", "裁决", 3)


# ---------------------------------------------------------------------------
# Builder validation


def _two(value: WorkflowValue, ctx: NodeContext) -> str:
    return value.text


async def _async_two(value: WorkflowValue, ctx: NodeContext) -> str:
    return value.text


def _zero() -> str:
    return ""


def _three(a: str, b: str, c: str) -> str:
    return a


def _star(*args: str) -> str:
    return ""


def _kw_only(text: str, *, flag: bool) -> str:
    return text


def _defaulted(text: str = "x") -> str:
    return text


def test_python_node_accepts_both_abis_and_detects_async() -> None:
    wf = WorkflowBuilder("abi")
    one = wf.python("one", body_fn)
    two = wf.python("two", _two)
    three = wf.python("three", _async_two)
    wf.start(one)
    wf.chain(one, two, three)
    wf.output(three)
    manifest = wf.build().manifest()
    assert _node(manifest, "one")["callable"] == {"name": "body_fn", "arity": 1, "async": False}
    assert _node(manifest, "two")["callable"] == {"name": "_two", "arity": 2, "async": False}
    assert _node(manifest, "three")["callable"] == {"name": "_async_two", "arity": 2, "async": True}


@pytest.mark.parametrize("fn", [_zero, _three, _star, _kw_only, _defaulted, "not callable"])
def test_python_node_rejects_callables_outside_the_abi(fn: Any) -> None:
    wf = WorkflowBuilder("abi")
    with pytest.raises(WorkflowValidationError):
        wf.python("bad", fn)


def test_predicates_and_until_must_be_sync_unary() -> None:
    wf = WorkflowBuilder("pred")
    a = wf.python("a", body_fn)
    b = wf.python("b", body_fn)
    with pytest.raises(WorkflowValidationError):
        wf.edge(a, b, when=_two)  # type: ignore[arg-type]
    with pytest.raises(WorkflowValidationError):
        wf.loop("L", body=lambda scope: (a, a), until=_async_two, max_iterations=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "make",
    [
        lambda: Retry(max_attempts=0),
        lambda: Retry(max_attempts=True),  # type: ignore[arg-type]
        lambda: Retry(max_attempts=2, backoff=-1),
        lambda: Retry(max_attempts=2, backoff=float("inf")),
        lambda: Retry(max_attempts=2, backoff=float("nan")),
        lambda: Retry(max_attempts="3"),  # type: ignore[arg-type]
    ],
)
def test_retry_validates_its_fields(make: Callable[[], Retry]) -> None:
    with pytest.raises(WorkflowValidationError, match="Retry"):
        make()


# A non-finite value would build, then fail to serialize in the worker.
@pytest.mark.parametrize("timeout", [0, -1, "5", True, float("inf"), float("nan")])
def test_timeouts_must_be_finite_positive_numbers(timeout: Any) -> None:
    wf = WorkflowBuilder("timeout")
    with pytest.raises(WorkflowValidationError):
        wf.agent("a", profile="A", timeout=timeout)
    with pytest.raises(WorkflowValidationError):
        wf.python("p", body_fn, timeout=timeout)


def test_agent_timeout_may_be_unbounded_but_python_defaults_to_300s() -> None:
    wf = WorkflowBuilder("timeout")
    a = wf.agent("a", profile="A")
    p = wf.python("p", body_fn)
    wf.start(a)
    wf.edge(a, p)
    wf.output(p)
    manifest = wf.build().manifest()
    assert _node(manifest, "a")["timeout"] is None
    assert _node(manifest, "p")["timeout"] == 300.0


@pytest.mark.parametrize("name", ["", "  ", 3, None])
def test_node_names_must_be_non_empty_strings(name: Any) -> None:
    wf = WorkflowBuilder("names")
    with pytest.raises(WorkflowValidationError):
        wf.python(name, body_fn)


def test_node_names_cannot_contain_the_edge_id_separator() -> None:
    wf = WorkflowBuilder("names")
    with pytest.raises(WorkflowValidationError, match="->"):
        wf.python("a->b", body_fn)


def test_duplicate_names_and_duplicate_edges_are_rejected() -> None:
    wf = WorkflowBuilder("dupes")
    a = wf.python("a", body_fn)
    b = wf.python("b", body_fn)
    with pytest.raises(WorkflowValidationError, match="already used"):
        wf.agent("a", profile="A")
    wf.edge(a, b)
    with pytest.raises(WorkflowValidationError):
        wf.edge(a, b, when=yes)


def test_foreign_handles_and_cross_scope_edges_are_rejected() -> None:
    wf = WorkflowBuilder("scopes")
    outside = wf.python("outside", body_fn)

    def body(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
        inside = scope.python("inside", body_fn)
        with pytest.raises(WorkflowValidationError):
            scope.edge(outside, inside)
        with pytest.raises(WorkflowValidationError):
            wf.edge(inside, outside)
        return inside, inside

    wf.loop("L", body=body, until=yes, max_iterations=1)
    with pytest.raises(WorkflowValidationError):
        wf.edge(NodeHandle("nope"), outside)

    twin = WorkflowBuilder("twin").python("outside", body_fn)  # same name, another builder
    with pytest.raises(WorkflowValidationError, match="another builder"):
        wf.edge(twin, outside)
    with pytest.raises(WorkflowValidationError, match="another builder"):
        wf.start(twin)


class _AsyncCallable:
    async def __call__(self, value: Any) -> bool:
        return True


def test_objects_with_an_async_call_count_as_async() -> None:
    wf = WorkflowBuilder("calls")
    body = wf.python("body", _AsyncCallable())
    other = wf.python("other", body_fn)
    with pytest.raises(WorkflowValidationError, match="sync function"):
        wf.edge(body, other, when=_AsyncCallable())
    with pytest.raises(WorkflowValidationError, match="sync function"):
        wf.loop("L", body=lambda scope: (scope.python("x", body_fn),) * 2, until=_AsyncCallable(), max_iterations=1)
    wf.edge(body, other)
    wf.start(body)
    wf.output(other)
    assert _node(wf.build().manifest(), "body")["callable"]["async"] is True


def test_loop_entry_and_exit_must_belong_to_the_body() -> None:
    wf = WorkflowBuilder("loop")
    outside = wf.python("outside", body_fn)
    with pytest.raises(WorkflowValidationError, match="entry/exit"):
        wf.loop("L", body=lambda scope: (outside, outside), until=yes, max_iterations=1)
    with pytest.raises(WorkflowValidationError, match="entry, exit"):
        wf.loop("M", body=lambda scope: scope.python("x", body_fn), until=yes, max_iterations=1)  # type: ignore[arg-type,return-value]

    foreign = WorkflowBuilder("twin").python("inside", body_fn)  # same name as a body node, another builder

    def body(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
        scope.python("inside", body_fn)
        return foreign, foreign

    with pytest.raises(WorkflowValidationError, match="entry/exit"):
        WorkflowBuilder("other").loop("N", body=body, until=yes, max_iterations=1)


@pytest.mark.parametrize(("max_iterations", "on_exhausted"), [(0, "continue"), (True, "continue"), (2, "retry")])
def test_loop_parameters_are_validated(max_iterations: Any, on_exhausted: str) -> None:
    wf = WorkflowBuilder("loop")

    def body(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
        node = scope.python("x", body_fn)
        return node, node

    with pytest.raises(WorkflowValidationError):
        wf.loop("L", body=body, until=yes, max_iterations=max_iterations, on_exhausted=on_exhausted)


def test_switch_is_declared_once_per_source() -> None:
    wf = WorkflowBuilder("switch")
    a = wf.python("a", body_fn)
    b = wf.python("b", body_fn)
    c = wf.python("c", body_fn)
    d = wf.python("d", body_fn)
    wf.switch(a, cases=[(yes, b)], default=c)
    with pytest.raises(WorkflowValidationError):
        wf.switch(a, cases=[(yes, d)], default=b)


def test_join_target_must_not_have_other_direct_in_edges() -> None:
    wf = WorkflowBuilder("join")
    a = wf.python("a", body_fn)
    b = wf.python("b", body_fn)
    c = wf.python("c", body_fn)
    d = wf.python("d", body_fn)
    wf.start(a)
    wf.edge(a, b)
    wf.edge(a, c)
    wf.edge(a, d)
    wf.join([b, c], d)
    wf.output(d)
    with pytest.raises(WorkflowValidationError, match="join target"):
        wf.build()


def test_start_and_outputs_are_top_level_and_declared_once() -> None:
    wf = WorkflowBuilder("io")
    a = wf.python("a", body_fn)
    b = wf.python("b", body_fn)
    body_node: list[NodeHandle] = []

    def body(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
        node = scope.python("inner", body_fn)
        body_node.append(node)
        return node, node

    loop = wf.loop("L", body=body, until=yes, max_iterations=1)
    wf.start(a)
    with pytest.raises(WorkflowValidationError):
        wf.start(b)
    with pytest.raises(WorkflowValidationError):
        wf.output(body_node[0])
    wf.output(b)
    with pytest.raises(WorkflowValidationError):
        wf.output(b)
    wf.chain(a, loop, b)
    assert isinstance(wf.build(), Workflow)


def test_build_requires_start_and_an_output_and_full_reachability() -> None:
    wf = WorkflowBuilder("build")
    a = wf.python("a", body_fn)
    b = wf.python("b", body_fn)
    with pytest.raises(WorkflowValidationError, match="start"):
        wf.build()
    wf.start(a)
    with pytest.raises(WorkflowValidationError, match="output"):
        wf.build()
    wf.output(b)
    with pytest.raises(WorkflowValidationError, match="reachable"):
        wf.build()


def test_validation_error_carries_a_location() -> None:
    error = WorkflowValidationError("bad", location="node")
    assert error.to_dict() == {"message": "bad", "location": "node", "file": None, "line": None}
    assert str(error) == "bad"
    sited = WorkflowValidationError("bad", location="node", site=("/w/flow.py", 7))
    assert sited.to_dict() == {"message": "bad", "location": "node", "file": "/w/flow.py", "line": 7}


def _line() -> int:
    """The line of the caller's call to this function."""
    frame = inspect.currentframe()
    assert frame is not None and frame.f_back is not None
    return frame.f_back.f_lineno


def test_nodes_and_edges_remember_the_line_that_declared_them() -> None:
    wf = WorkflowBuilder("sites")
    a = wf.python("a", body_fn)
    a_line = _line() - 1
    b = wf.agent("b", profile="Code")
    b_line = _line() - 1

    def body(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
        node = scope.python("inner", body_fn)
        lines["inner"] = _line() - 1
        return node, node

    lines: dict[str, int] = {}
    loop = wf.loop("L", body=body, until=yes, max_iterations=1)
    loop_line = _line() - 1
    wf.start(a)
    wf.edge(a, b)
    edge_line = _line() - 1
    wf.chain(b, loop)
    chain_line = _line() - 1
    wf.output(loop)
    definition = wf.build().definition
    here = __file__
    assert definition.nodes["a"].site == (here, a_line)
    assert definition.nodes["b"].site == (here, b_line)
    assert definition.nodes["inner"].site == (here, lines["inner"])
    assert definition.nodes["L"].site == (here, loop_line)
    assert {edge_id: edge.site for edge_id, edge in definition.edges.items()} == {
        "a->b": (here, edge_line),
        "b->L": (here, chain_line),
    }


def test_switch_and_join_edges_remember_their_declaring_line() -> None:
    wf = WorkflowBuilder("sites")
    a, b, c, d = (wf.python(name, body_fn) for name in "abcd")
    wf.start(a)
    wf.switch(a, [(yes, b)], default=c)
    switch_line = _line() - 1
    wf.join([b, c], d)
    join_line = _line() - 1
    wf.output(d)
    definition = wf.build().definition
    assert {edge_id: edge.site for edge_id, edge in definition.edges.items()} == {
        "a->b": (__file__, switch_line),
        "a->c": (__file__, switch_line),
        "b->join:d": (__file__, join_line),
        "c->join:d": (__file__, join_line),
        "join:d->d": (__file__, join_line),
    }
    assert definition.nodes["join:d"].site == (__file__, join_line)


def test_declaration_sites_stay_out_of_the_manifest_and_of_equality() -> None:
    source = (
        "wf = WorkflowBuilder('same')\na = wf.python('a', body_fn)\nb = wf.python('b', body_fn)\n"
        "wf.start(a)\nwf.edge(a, b)\nwf.output(b)\nworkflow = wf.build()\n"
    )
    first, second = ({"WorkflowBuilder": WorkflowBuilder, "body_fn": body_fn} for _ in range(2))
    exec(compile(source, __file__, "exec"), first)
    exec(compile("\n\n\n" + source, __file__, "exec"), second)  # the same workflow, declared three lines lower
    one, other = first["workflow"].definition, second["workflow"].definition
    assert (one.nodes["a"].site, other.nodes["a"].site) == ((__file__, 2), (__file__, 5))
    assert (one.edges["a->b"].site, other.edges["a->b"].site) == ((__file__, 5), (__file__, 8))
    assert canonical_json(first["workflow"].manifest()) == canonical_json(second["workflow"].manifest())
    assert "site" not in canonical_json(first["workflow"].manifest())
    assert one.nodes == other.nodes
    assert one.edges == other.edges


def test_code_built_under_a_made_up_name_is_located_at_the_line_that_ran_it(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace: dict[str, Any] = {"WorkflowBuilder": WorkflowBuilder, "body_fn": body_fn}
    generated = (
        "wf = WorkflowBuilder('g')\na = wf.python('a', body_fn)\nwf.start(a)\nwf.output(a)\nworkflow = wf.build()\n"
    )
    exec(compile(generated, "<generated>", "exec"), namespace)
    exec_line = _line() - 1
    assert namespace["workflow"].definition.nodes["a"].site == (__file__, exec_line)
    missing = str(Path(__file__).parent / "no-such-workflow.py")
    exec(compile(generated, missing, "exec"), namespace)
    exec_line = _line() - 1
    assert namespace["workflow"].definition.nodes["a"].site == (__file__, exec_line)
    monkeypatch.setenv("CHRYS_WORKFLOW_ENTRY", missing)
    exec(compile(generated, missing, "exec"), namespace)
    assert namespace["workflow"].definition.nodes["a"].site == (missing, 2)


def test_a_callable_of_the_wrong_shape_is_located_at_its_node_or_edge() -> None:
    wf = WorkflowBuilder("shapes")
    with pytest.raises(WorkflowValidationError, match="must be callable") as body:
        wf.python("a", 3)
    a, b, c = (wf.python(name, body_fn) for name in "abc")
    with pytest.raises(WorkflowValidationError, match="sync function") as predicate:
        wf.edge(a, b, when=_AsyncCallable())
    with pytest.raises(WorkflowValidationError, match="positional parameter") as case:
        wf.switch(a, [(lambda: True, b)], c)
    with pytest.raises(WorkflowValidationError, match="sync function") as until:
        wf.loop("L", body=lambda scope: (scope.python("x", body_fn),) * 2, until=_AsyncCallable(), max_iterations=1)
    assert [error.value.location for error in (body, predicate, case, until)] == ["a", "a->b", "a->b", "L"]


def test_build_errors_point_at_the_offending_declaration() -> None:
    wf = WorkflowBuilder("unreachable")
    a = wf.python("a", body_fn)
    wf.python("z", body_fn)
    wf.python("y", body_fn)
    y_line = _line() - 1
    wf.start(a)
    wf.output(a)
    with pytest.raises(WorkflowValidationError) as unreachable:
        wf.build()
    assert unreachable.value.site == (__file__, y_line)
    assert unreachable.value.location == "y"

    wf = WorkflowBuilder("cycle")
    a, b, c = (wf.python(name, body_fn) for name in "abc")
    wf.start(a)
    wf.edge(a, b)
    wf.edge(b, c)
    wf.edge(c, b)
    closing_line = _line() - 1
    wf.output(c)
    with pytest.raises(WorkflowValidationError, match="Cycle") as cycle:
        wf.build()
    assert cycle.value.site == (__file__, closing_line)

    wf = WorkflowBuilder("join")
    a, b, c, d = (wf.python(name, body_fn) for name in "abcd")
    wf.start(a)
    wf.edge(a, b)
    wf.edge(a, c)
    wf.edge(a, d)
    extra_line = _line() - 1
    wf.join([b, c], d)
    wf.output(d)
    with pytest.raises(WorkflowValidationError, match="join target") as join:
        wf.build()
    assert join.value.site == (__file__, extra_line)

    wf = WorkflowBuilder("no start")
    wf.output(wf.python("a", body_fn))
    with pytest.raises(WorkflowValidationError, match="start") as no_start:
        wf.build()
    assert no_start.value.site is None


def test_node_context_type_checks_and_delegates() -> None:
    emitted: list[str] = []

    async def ask(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"selected": [], "text": f"answer:{questions[0]['question']}"}]

    ctx = NodeContext(emit=emitted.append, ask=ask)
    ctx.emit("hello")
    assert emitted == ["hello"]
    with pytest.raises(TypeError):
        ctx.emit(1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        asyncio.run(ctx.ask(1))  # type: ignore[call-overload]
    assert asyncio.run(ctx.ask("q")) == "answer:q"


def test_value_types_are_frozen_plain_data() -> None:
    value = WorkflowValue("t", {"k": 1})
    source = SourceValue("n", "n@iter#1", value)
    with pytest.raises(AttributeError):
        value.text = "u"  # type: ignore[misc]
    assert source.value is value
    assert WorkflowValue("t").data is None


# ---------------------------------------------------------------------------
# Manifest reader


def _small_manifest() -> dict[str, Any]:
    wf = WorkflowBuilder("small")
    a = wf.python("a", body_fn)

    def body(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
        entry = scope.python("e", body_fn)
        exit_ = scope.python("x", body_fn)
        scope.edge(entry, exit_)
        return entry, exit_

    loop = wf.loop("L", body=body, until=yes, max_iterations=2)
    b = wf.python("b", body_fn)
    c = wf.python("c", body_fn)
    wf.start(a)
    wf.edge(a, loop)
    wf.switch(loop, cases=[(yes, b)], default=c)
    wf.output(b)
    wf.output(c)
    return wf.build().manifest()


def _set_schema(manifest: dict[str, Any]) -> None:
    manifest["schema_version"] = 2


def _unknown_kind(manifest: dict[str, Any]) -> None:
    _node(manifest, "a")["kind"] = "shell"


def _dangling_edge(manifest: dict[str, Any]) -> None:
    manifest["edges"].append({"id": "a->zz", "src": "a", "dst": "zz", "conditional": False, "switch": None})


def _duplicate_pair(manifest: dict[str, Any]) -> None:
    manifest["edges"].append({"id": "again", "src": "a", "dst": "L", "conditional": False, "switch": None})


def _no_default(manifest: dict[str, Any]) -> None:
    _edge(manifest, "L->c")["switch"] = {"group": "L", "position": 1, "default": False}


def _cross_scope(manifest: dict[str, Any]) -> None:
    manifest["edges"].append({"id": "a->x", "src": "a", "dst": "x", "conditional": False, "switch": None})


def _orphan_member(manifest: dict[str, Any]) -> None:
    _node(manifest, "x")["parent_loop"] = None


def _output_in_loop(manifest: dict[str, Any]) -> None:
    manifest["outputs"] = ["x"]


def _bad_retry(manifest: dict[str, Any]) -> None:
    _node(manifest, "a")["retry"] = {"max_attempts": 0, "backoff": 0}


def _bad_conditional(manifest: dict[str, Any]) -> None:
    _edge(manifest, "a->L")["conditional"] = "yes"


def _unreachable(manifest: dict[str, Any]) -> None:
    manifest["nodes"].append(_node(manifest, "b") | {"id": "lonely"})
    manifest["edges"].append({"id": "lonely->b", "src": "lonely", "dst": "b", "conditional": False, "switch": None})


def _second_root(manifest: dict[str, Any]) -> None:
    manifest["nodes"].append(_node(manifest, "b") | {"id": "root2"})
    manifest["edges"].append({"id": "root2->b", "src": "root2", "dst": "b", "conditional": False, "switch": None})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set_schema, "schema_version"),
        (_unknown_kind, "unknown kind"),
        (_dangling_edge, "unknown node"),
        (_duplicate_pair, "duplicate"),
        (_no_default, "exactly one default"),
        (_cross_scope, "crosses"),
        (_orphan_member, "not its child"),
        (_output_in_loop, "top-level"),
        (_bad_retry, "max_attempts"),
        (_bad_conditional, "conditional"),
        (_unreachable, "unreachable|only top-level node"),
        (_second_root, "only top-level node"),
    ],
)
def test_manifest_reader_rejects_tampered_manifests(mutate: Callable[[dict[str, Any]], None], message: str) -> None:
    manifest = _small_manifest()
    assert GraphSpec.from_manifest(manifest)
    mutate(manifest)
    with pytest.raises(ManifestError, match=message):
        GraphSpec.from_manifest(manifest)


def test_a_deep_chain_builds_without_touching_the_recursion_limit() -> None:
    """The builder's cycle check walks iteratively: a 1,200-node chain is a valid workflow, not a RecursionError."""
    wf = WorkflowBuilder("deep")
    handles = [wf.python(f"n{index}", body_fn) for index in range(1200)]
    wf.start(handles[0])
    wf.chain(*handles)
    wf.output(handles[-1])
    manifest = wf.build().manifest()
    assert len(GraphSpec.from_manifest(manifest).node_order) == 1200
