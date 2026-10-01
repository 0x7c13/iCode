# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""CI gate for the signed, AST-generated trajectory wait inventory."""

from __future__ import annotations

import ast

import pytest

import tests.support.trajectory_wait_inventory as wait_inventory
from tests.support.ci import CI_LINUX_ONLY
from tests.support.paths import SRC_ROOT
from tests.support.trajectory_wait_inventory import build_manifest, load_manifest, manifest_drift, scan_wait_nodes

pytestmark = CI_LINUX_ONLY


def test_signed_wait_manifest_covers_every_ast_node() -> None:
    expected = load_manifest()
    errors = manifest_drift(expected, build_manifest())
    assert errors == [], "\n".join(errors)


def test_manifest_refresh_preserves_reviewed_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    node = scan_wait_nodes()[0]
    reviewed_case = {
        "when": "all_paths",
        "classification": "B",
        "container_rule": "reviewed_container",
        "degradation_rule": "Mark the containing residual Unresolved; never report it as exact.",
        "reason": "Human-reviewed explanation that regeneration must retain.",
    }
    monkeypatch.setattr(wait_inventory, "scan_wait_nodes", lambda: [node])

    reviewed = {
        "identity": node.identity,
        "expression": node.expression,
        "primitive": node.primitive,
        "cases": [reviewed_case],
    }

    assert build_manifest({"nodes": [reviewed]})["nodes"][0]["cases"] == [reviewed_case]
    # An await inserted above shifts the ordinals: the wait now holding this
    # identity was never reviewed, so it gets the conservative default.
    moved = build_manifest({"nodes": [{**reviewed, "expression": "await some_other_wait()"}]})
    assert moved["nodes"][0]["cases"] != [reviewed_case]
    assert moved["nodes"][0]["cases"][0]["container_rule"] != "reviewed_container"


def test_wait_inventory_covers_every_explicit_and_implicit_async_wait() -> None:
    nodes = scan_wait_nodes()
    inventoried = {(node.module, node.source_line, node.source_column, node.expression) for node in nodes}
    missing: list[str] = []
    for layer in ("foundation", "kernel", "service", "orchestration"):
        for path in sorted((SRC_ROOT / "chrys" / layer).rglob("*.py")):
            module = ".".join(path.relative_to(SRC_ROOT).with_suffix("").parts)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
            for node in ast.walk(tree):
                expected: list[tuple[int, int, str]] = []
                if isinstance(node, ast.Await):
                    expected.append((node.lineno, node.col_offset, ast.unparse(node)))
                elif isinstance(node, ast.AsyncFor):
                    expected.append(
                        (
                            node.lineno,
                            node.col_offset,
                            f"async for {ast.unparse(node.target)} in {ast.unparse(node.iter)}",
                        )
                    )
                elif isinstance(node, ast.comprehension) and node.is_async:
                    expected.append(
                        (
                            node.iter.lineno,
                            node.iter.col_offset,
                            f"async comprehension for {ast.unparse(node.target)} in {ast.unparse(node.iter)}",
                        )
                    )
                elif isinstance(node, ast.AsyncWith):
                    expected.extend(
                        (
                            item.context_expr.lineno,
                            item.context_expr.col_offset,
                            f"async with {ast.unparse(item.context_expr)}",
                        )
                        for item in node.items
                    )
                for line, column, expression in expected:
                    if (module, line, column, expression) not in inventoried:
                        missing.append(f"{path.relative_to(SRC_ROOT)}:{line}: {expression}")
    assert missing == [], "async waits missing from trajectory wait inventory:\n" + "\n".join(missing)


def test_wait_inventory_scans_async_comprehension_iterators() -> None:
    tree = ast.parse(
        "async def consume():\n"
        "    return [item async for item in stream_items()]\n"
        "\n"
        "async def consume_set():\n"
        "    return {item async for item in other_items()}\n"
    )
    scan = wait_inventory._ModuleScan(module="chrys.synthetic.async_comprehensions")
    scan.visit(tree)

    candidates = [candidate for candidate in scan.direct if candidate.primitive == "async_iteration"]
    assert [(candidate.line, candidate.expression, candidate.call_target) for candidate in candidates] == [
        (2, "async comprehension for item in stream_items()", "chrys.synthetic.async_comprehensions.stream_items"),
        (5, "async comprehension for item in other_items()", "chrys.synthetic.async_comprehensions.other_items"),
    ]


def test_wait_inventory_pins_blocking_stream_io_representatives() -> None:
    nodes = scan_wait_nodes()
    expected = {
        ("chrys.service.tools.builtins.shell", "await reader.read("),
        ("chrys.service.mcp._stdio_transport", "await receive()"),
    }
    missing = {
        (module, expression)
        for module, expression in expected
        if not any(node.module == module and expression in node.expression for node in nodes)
    }
    assert missing == set()


def test_wait_inventory_pins_implicit_stream_and_context_manager_waits() -> None:
    nodes = scan_wait_nodes()
    expected = {
        ("chrys.service.llm.openai_responses.client", "async for event in parsed_events"),
        ("chrys.service.mcp._stdio_transport", "async for session_message in write_stream_reader"),
        ("chrys.service.mcp._http_transport", "async with streamable_http_client("),
    }
    missing = {
        (module, expression)
        for module, expression in expected
        if not any(node.module == module and expression in node.expression for node in nodes)
    }
    assert missing == set()


def test_wait_inventory_resolves_transitive_wrapper_targets() -> None:
    nodes = scan_wait_nodes()
    expected = {
        "chrys.orchestration.invoker.acp.AcpConversation._wait_backoff",
        "chrys.orchestration.sub_agents.kernel_policy.KernelSubAgentPolicy._interruptible_sleep",
        "chrys.service.trajectory.preparation.preparation_lock",
    }
    assert expected <= {node.wrapper_target for node in nodes if node.wrapper_target is not None}


def test_wait_inventory_resolves_relative_import_wrapper_targets() -> None:
    nodes = scan_wait_nodes()
    expected = {
        "validate_chat_options": "chrys.kernel._types.validate_chat_options",
        "apply_compaction": "chrys.kernel.compaction.apply_compaction",
        "spawn_acp_process": "chrys.service.acp_client.spawn.spawn_acp_process",
    }
    for expression, target in expected.items():
        matches = [node for node in nodes if expression in node.expression]
        assert matches, expression
        assert target in {node.wrapper_target for node in matches}


def test_mcp_owner_connect_waits_fail_closed_without_path_proof() -> None:
    nodes = [
        node
        for node in build_manifest()["nodes"]
        if node["module"] == "chrys.service.mcp.owned" and node["qualname"] == "MCPTool._connect_on_owner"
    ]
    assert nodes
    assert {case["classification"] for node in nodes for case in node["cases"]} == {"B"}
    assert {case["degradation_rule"] for node in nodes for case in node["cases"]} == {
        "Mark the containing residual Unresolved; never report it as exact."
    }


@pytest.mark.parametrize(
    ("identity", "case"),
    [
        (node["identity"], case)
        for node in load_manifest()["nodes"]
        for case in node["cases"]
        if case["classification"] == "B"
    ],
)
def test_unrecorded_wait_case_has_a_non_exact_counterexample(identity: str, case: dict[str, object]) -> None:
    """Pin the handoff: every known gap explicitly propagates Unresolved."""
    assert case["container_rule"] != "nearest_recorded_operation", identity
    assert case["degradation_rule"] == "Mark the containing residual Unresolved; never report it as exact.", identity


def test_pending_retry_clear_calls_declare_a_terminal_reason() -> None:
    violations: list[str] = []
    for path in sorted((SRC_ROOT / "chrys").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "clear_pending_retry"
            ):
                continue
            outcome = next((keyword.value for keyword in node.keywords if keyword.arg == "outcome"), None)
            if not (
                isinstance(outcome, ast.Attribute)
                and isinstance(outcome.value, ast.Name)
                and outcome.value.id == "PreparationOutcome"
                and outcome.attr in {"DROPPED", "RETRY_TURN"}
            ):
                violations.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno}")
    assert violations == [], "clear_pending_retry calls without dropped/retry_turn reason:\n" + "\n".join(violations)
