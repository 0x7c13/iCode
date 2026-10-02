# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""CI gate for the AST-generated trajectory wait manifest."""

from __future__ import annotations

import ast
import functools
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import tests.support.trajectory_wait_inventory as wait_inventory
from tests.support.ci import CI_LINUX_ONLY
from tests.support.paths import SRC_ROOT
from tests.support.trajectory_wait_inventory import (
    DEFAULT_RULES,
    EVENT_LOOP_YIELD_RULE,
    UNRESOLVED_DEGRADATION,
    UNRESOLVED_RULE,
    Manifest,
    ManifestNode,
    WaitNode,
    build_manifest,
    decode_manifest,
    default_rule,
    encode_manifest,
    load_manifest_text,
    manifest_errors,
    manifest_from_v1,
    read_reviewed,
    reopened_nodes,
    scan_wait_nodes,
)

pytestmark = CI_LINUX_ONLY

_REVIEWED_RULE = {
    "when": "all_paths",
    "classification": "B",
    "container_rule": "reviewed_container",
    "degradation_rule": UNRESOLVED_DEGRADATION,
    "reason": "Human-reviewed explanation that regeneration must retain.",
}


@functools.cache
def _scanned() -> tuple[WaitNode, ...]:
    """One source scan per worker, shared read-only by the tests below."""
    return tuple(scan_wait_nodes())


def _wait(
    expression: str = "await reviewed()",
    *,
    ordinal: int = 1,
    primitive: str = "awaitable",
    wrapper_target: str | None = None,
    line: int = 10,
    column: int = 4,
    module: str = "chrys.synthetic",
) -> WaitNode:
    return WaitNode(
        identity=f"{module}:run:{ordinal}",
        module=module,
        qualname="run",
        ordinal=ordinal,
        primitive=primitive,
        source_column=column,
        source_line=line,
        expression=expression,
        wrapper_target=wrapper_target,
    )


_SYNTHETIC_WAITS = (
    _wait(),
    _wait("await helper()", ordinal=2, primitive="wrapper", wrapper_target="chrys.synthetic.helper", line=11),
)


def _synthetic_document() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "rules": {UNRESOLVED_RULE: dict(DEFAULT_RULES[UNRESOLVED_RULE]), "reviewed": dict(_REVIEWED_RULE)},
        "modules": {
            "chrys.synthetic": {
                "run": [
                    ["awaitable", "await reviewed()", "reviewed"],
                    ["wrapper", "await helper()", UNRESOLVED_RULE, "chrys.synthetic.helper"],
                ]
            }
        },
    }


def _reviewed(node: WaitNode, *, expression: str | None = None) -> Manifest:
    return Manifest(
        rules={"reviewed": dict(_REVIEWED_RULE)},
        nodes=(
            ManifestNode(
                module=node.module,
                qualname=node.qualname,
                ordinal=node.ordinal,
                primitive=node.primitive,
                expression=expression or node.expression,
                rule="reviewed",
            ),
        ),
    )


def test_wait_manifest_matches_source() -> None:
    errors = manifest_errors(load_manifest_text(), _scanned())
    assert errors == [], "refresh: uv run python -m tests.support.trajectory_wait_inventory\n" + "\n".join(errors)


def test_manifest_refresh_preserves_reviewed_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _wait()
    monkeypatch.setattr(wait_inventory, "scan_wait_nodes", lambda: [node])

    kept = build_manifest(_reviewed(node))
    assert [row.rule for row in kept.nodes] == ["reviewed"]
    assert kept.rules == {"reviewed": _REVIEWED_RULE}
    assert reopened_nodes(_reviewed(node), kept) == []
    # An await inserted above shifts the ordinals: the wait now holding this
    # identity was never reviewed, so it gets the conservative default.
    shifted = _reviewed(node, expression="await some_other_wait()")
    moved = build_manifest(shifted)
    assert [row.rule for row in moved.nodes] == [UNRESOLVED_RULE]
    assert moved.rules == {UNRESOLVED_RULE: DEFAULT_RULES[UNRESOLVED_RULE]}
    assert reopened_nodes(shifted, moved) == [(node.identity, "reviewed", "await some_other_wait()")]


def _in(qualname: str, node: WaitNode) -> WaitNode:
    return replace(node, qualname=qualname, identity=f"{node.module}:{qualname}:{node.ordinal}")


def test_a_wait_copied_above_a_reviewed_one_reopens_its_function(monkeypatch: pytest.MonkeyPatch) -> None:
    reviewed_node = _wait()
    elsewhere = _in("other", _wait())
    reviewed = Manifest(
        rules={"reviewed": dict(_REVIEWED_RULE)},
        nodes=(
            *_reviewed(reviewed_node).nodes,
            ManifestNode(elsewhere.module, "other", 1, elsewhere.primitive, elsewhere.expression, "reviewed"),
        ),
    )
    # The copy now holds the reviewed wait's identity and expression; the
    # reviewed wait moved to ordinal 2.
    copied = [_wait(), _wait(ordinal=2, line=11), elsewhere]
    monkeypatch.setattr(wait_inventory, "scan_wait_nodes", lambda: copied)

    built = build_manifest(reviewed)

    assert [(row.identity, row.rule) for row in built.nodes] == [
        ("chrys.synthetic:run:1", UNRESOLVED_RULE),
        ("chrys.synthetic:run:2", UNRESOLVED_RULE),
        ("chrys.synthetic:other:1", "reviewed"),
    ]
    assert reopened_nodes(reviewed, built) == [
        ("chrys.synthetic:run:1", "reviewed", reviewed_node.expression),
        ("chrys.synthetic:run:2", None, None),
    ]


def test_a_wait_whose_primitive_changed_is_reopened(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _wait()
    monkeypatch.setattr(wait_inventory, "scan_wait_nodes", lambda: [replace(node, primitive="async_iteration")])

    built = build_manifest(_reviewed(node))

    assert [row.rule for row in built.nodes] == [UNRESOLVED_RULE]
    assert reopened_nodes(_reviewed(node), built) == [(node.identity, "reviewed", node.expression)]


@pytest.mark.parametrize(
    ("primitive", "expression", "rule"),
    [
        pytest.param("sleep", "await asyncio.sleep(0)", EVENT_LOOP_YIELD_RULE, id="zero-sleep"),
        pytest.param("sleep", "await asyncio.sleep(0.5)", UNRESOLVED_RULE, id="timed-sleep"),
        pytest.param("sleep", "await asyncio.sleep(delay)", UNRESOLVED_RULE, id="variable-sleep"),
        pytest.param("awaitable", "await clock.sleep(0)", UNRESOLVED_RULE, id="other-sleep"),
    ],
)
def test_only_a_zero_sleep_defaults_to_an_event_loop_yield(primitive: str, expression: str, rule: str) -> None:
    assert default_rule(_wait(expression, primitive=primitive)) == rule


def test_a_new_zero_sleep_gets_the_yield_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wait_inventory, "scan_wait_nodes", lambda: [_wait("await asyncio.sleep(0)", primitive="sleep")])

    built = build_manifest(None)

    assert [row.rule for row in built.nodes] == [EVENT_LOOP_YIELD_RULE]
    assert built.rules == {EVENT_LOOP_YIELD_RULE: DEFAULT_RULES[EVENT_LOOP_YIELD_RULE]}


def test_the_refresh_reports_the_rule_each_reopened_wait_had(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    node = _wait()
    path = tmp_path / "manifest.json"
    path.write_text(encode_manifest(_reviewed(node)), encoding="utf-8")
    monkeypatch.setattr(wait_inventory, "MANIFEST_PATH", path)
    # A wait inserted above the reviewed one takes its identity: the rule is
    # reported with the wait it was reviewed on.
    monkeypatch.setattr(
        wait_inventory, "scan_wait_nodes", lambda: [_wait("await new_thing()"), _wait(ordinal=2, line=11)]
    )

    wait_inventory.main()

    assert capsys.readouterr().out.splitlines()[1:] == [
        "  chrys.synthetic:run:1 (was reviewed on `await reviewed()`)",
        "  chrys.synthetic:run:2",
    ]
    assert [row.rule for row in decode_manifest(path.read_text(encoding="utf-8"))[0].nodes] == [UNRESOLVED_RULE] * 2


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("{", id="not-json"),
        pytest.param(json.dumps({"schema_version": 3, "rules": {}, "modules": {}}), id="unknown-schema"),
        pytest.param(
            json.dumps(
                {
                    **_synthetic_document(),
                    "modules": {"chrys.synthetic": {"run": [["awaitable", "await reviewed()", "missing"]]}},
                }
            ),
            id="unknown-rule",
        ),
        pytest.param(
            json.dumps(_synthetic_document()).replace('"modules": {', '"modules": {"chrys.synthetic": {}, ', 1),
            id="duplicate-key",
        ),
    ],
)
def test_the_refresh_refuses_an_invalid_manifest(text: str) -> None:
    with pytest.raises(ValueError, match="cannot refresh from an invalid manifest"):
        read_reviewed(text)


def test_a_v1_manifest_with_duplicate_keys_is_not_migrated() -> None:
    text = json.dumps({"schema_version": 1, "nodes": []}).replace('"nodes"', '"nodes": [], "nodes"', 1)

    with pytest.raises(ValueError, match="duplicate JSON key 'nodes'"):
        read_reviewed(text)


def test_manifest_refresh_rewrites_built_in_rule_text(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _wait()
    monkeypatch.setattr(wait_inventory, "scan_wait_nodes", lambda: [node])
    stale = Manifest(
        rules={UNRESOLVED_RULE: {**DEFAULT_RULES[UNRESOLVED_RULE], "reason": "Older default wording."}},
        nodes=(ManifestNode("chrys.synthetic", "run", 1, "awaitable", "await reviewed()", UNRESOLVED_RULE),),
    )

    assert build_manifest(stale).rules == {UNRESOLVED_RULE: DEFAULT_RULES[UNRESOLVED_RULE]}


def _canonical(document: dict[str, Any]) -> str:
    manifest, errors = decode_manifest(json.dumps(document))
    assert errors == []
    return encode_manifest(manifest)


def test_manifest_encoding_round_trips() -> None:
    manifest, errors = decode_manifest(json.dumps(_synthetic_document()))
    assert errors == []
    assert decode_manifest(encode_manifest(manifest)) == (manifest, [])
    assert manifest_errors(encode_manifest(manifest), _SYNTHETIC_WAITS) == []


def test_the_gate_wants_the_layout_a_refresh_writes() -> None:
    assert manifest_errors(json.dumps(_synthetic_document()), _SYNTHETIC_WAITS) == [
        "manifest is not in the layout a refresh writes"
    ]


def test_encoding_refuses_ordinals_it_would_renumber() -> None:
    gap = Manifest(
        rules={UNRESOLVED_RULE: dict(DEFAULT_RULES[UNRESOLVED_RULE])},
        nodes=(
            ManifestNode("chrys.synthetic", "run", 1, "awaitable", "await one()", UNRESOLVED_RULE),
            ManifestNode("chrys.synthetic", "run", 3, "awaitable", "await three()", UNRESOLVED_RULE),
        ),
    )

    with pytest.raises(ValueError, match=r"chrys.synthetic:run: ordinals are not 1..n"):
        encode_manifest(gap)


def test_source_line_moves_are_not_drift() -> None:
    moved = tuple(replace(node, source_line=node.source_line + 40, source_column=0) for node in _SYNTHETIC_WAITS)
    assert manifest_errors(_canonical(_synthetic_document()), moved) == []


def _run_rows(document: dict[str, Any]) -> list[list[Any]]:
    return document["modules"]["chrys.synthetic"]["run"]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        pytest.param(lambda doc: _run_rows(doc)[0].pop(), "row must be", id="short-row"),
        pytest.param(lambda doc: _run_rows(doc)[0].__setitem__(1, 7), "non-empty strings", id="non-string-field"),
        pytest.param(lambda doc: _run_rows(doc)[1].pop(), "wrapper_target goes with", id="wrapper-without-target"),
        pytest.param(
            lambda doc: _run_rows(doc)[1].__setitem__(3, "chrys.synthetic.other"),
            "chrys.synthetic:run:2: wrapper_target drifted",
            id="wrapper-target-drift",
        ),
        pytest.param(
            lambda doc: _run_rows(doc)[0].__setitem__(1, "await renamed()"),
            "chrys.synthetic:run:1: expression drifted",
            id="expression-drift",
        ),
        pytest.param(
            lambda doc: _run_rows(doc)[0].__setitem__(2, "missing"), "unknown rule 'missing'", id="unknown-rule"
        ),
        pytest.param(
            lambda doc: doc["rules"].__setitem__("spare", dict(_REVIEWED_RULE)),
            "rules no wait uses: spare",
            id="unreferenced-rule",
        ),
        pytest.param(lambda doc: doc["rules"].__setitem__("empty", {}), "exactly the fields", id="empty-rule"),
        pytest.param(
            lambda doc: doc["rules"]["reviewed"].__setitem__("reason", "   "),
            "rule 'reviewed': every field must be a non-empty string",
            id="blank-rule-text",
        ),
        pytest.param(
            lambda doc: doc["rules"]["reviewed"].__setitem__("classification", "D"),
            "invalid classification",
            id="bad-classification",
        ),
        pytest.param(
            lambda doc: doc["rules"]["reviewed"].__setitem__("when", "sometimes"),
            "when must be all_paths",
            id="bad-when",
        ),
        pytest.param(
            lambda doc: doc["rules"]["reviewed"].__setitem__("degradation_rule", "none"),
            "needs a degradation rule",
            id="b-without-degradation",
        ),
        pytest.param(
            lambda doc: doc["rules"][UNRESOLVED_RULE].__setitem__("reason", "Edited by hand."),
            "differs from its built-in definition",
            id="edited-built-in-rule",
        ),
        pytest.param(
            lambda doc: doc["modules"].__setitem__("chrys.extra", {"f": [["awaitable", "await x()", UNRESOLVED_RULE]]}),
            "stale wait nodes: chrys.extra:f:1",
            id="extra-module",
        ),
        pytest.param(
            lambda doc: doc["modules"]["chrys.synthetic"].__setitem__("empty", []),
            "must be a non-empty name mapping to a non-empty list",
            id="empty-function",
        ),
        pytest.param(lambda doc: doc.__setitem__("schema_version", 1), "schema_version must be 2", id="old-schema"),
        pytest.param(lambda doc: doc.__setitem__("schema_version", 2.0), "schema_version must be 2", id="float-schema"),
        pytest.param(
            lambda doc: doc.__setitem__("inventory_sha256", "0" * 64),
            "exactly schema_version, rules and modules",
            id="extra-top-level-key",
        ),
    ],
)
def test_manifest_decoder_rejects_malformed_documents(
    mutate: Callable[[dict[str, Any]], object], expected: str
) -> None:
    document = _synthetic_document()
    mutate(document)

    errors = manifest_errors(json.dumps(document), _SYNTHETIC_WAITS)

    assert any(expected in error for error in errors), errors


def test_manifest_reports_waits_missing_from_a_module() -> None:
    other = _wait("await elsewhere()", module="chrys.other")

    assert manifest_errors(_canonical(_synthetic_document()), (*_SYNTHETIC_WAITS, other)) == [
        "unclassified wait nodes: chrys.other:run:1"
    ]


def test_manifest_rejects_duplicate_json_keys() -> None:
    text = json.dumps(_synthetic_document()).replace('"modules": {', '"modules": {"chrys.synthetic": {}, ', 1)

    assert "duplicate JSON key 'chrys.synthetic'" in manifest_errors(text, _SYNTHETIC_WAITS)


def _v1_entry(ordinal: int, **case: str) -> dict[str, Any]:
    return {
        "identity": f"chrys.synthetic:run:{ordinal}",
        "module": "chrys.synthetic",
        "qualname": "run",
        "ordinal": ordinal,
        "primitive": "awaitable",
        "expression": f"await step{ordinal}()",
        "wrapper_target": None,
        "source_line": ordinal,
        "source_column": 4,
        "cases": [{**DEFAULT_RULES[UNRESOLVED_RULE], **case}],
    }


_V1_WORKFLOW_REASON = (
    "Workflow persistence, answer acknowledgement or native-output draining "
    "can block without a proven enclosing trajectory interval."
)
_V1_YIELD_CASE = {
    "classification": "C",
    "container_rule": "none",
    "degradation_rule": "none",
    "reason": "Event-loop yielding does not block measured execution.",
}


def test_v1_manifest_migrates_reviewed_reasons_to_rules() -> None:
    nodes = [
        _v1_entry(1),
        _v1_entry(2, **_V1_YIELD_CASE),
        _v1_entry(3, reason=_V1_WORKFLOW_REASON),
        _v1_entry(4, reason=_V1_WORKFLOW_REASON),
    ]
    document = {"schema_version": 1, "inventory_sha256": "0" * 64, "nodes": nodes}

    manifest = read_reviewed(json.dumps(document))

    assert [node.rule for node in manifest.nodes] == [
        UNRESOLVED_RULE,
        EVENT_LOOP_YIELD_RULE,
        "workflow-blocking",
        "workflow-blocking",
    ]
    assert manifest.rules == {"workflow-blocking": {**DEFAULT_RULES[UNRESOLVED_RULE], "reason": _V1_WORKFLOW_REASON}}


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        pytest.param([_v1_entry(1), _v1_entry(1)], "duplicate identity", id="duplicate-identity"),
        pytest.param([_v1_entry(1), _v1_entry(3)], "ordinals are not 1..n", id="ordinal-gap"),
        pytest.param([_v1_entry(1, reason="An unmapped review.")], "no schema 2 rule name", id="unmapped-case"),
        pytest.param(
            [
                _v1_entry(1, reason=_V1_WORKFLOW_REASON, container_rule="first_reviewed_container"),
                _v1_entry(2, reason=_V1_WORKFLOW_REASON, container_rule="second_reviewed_container"),
            ],
            r"chrys.synthetic:run:2: its case does not match rule 'workflow-blocking'",
            id="same-reason-different-case",
        ),
        pytest.param(
            [_v1_entry(1, container_rule="reviewed_container")],
            r"chrys.synthetic:run:1: its case does not match rule 'unresolved'",
            id="built-in-reason-reviewed-case",
        ),
        pytest.param(
            [_v1_entry(1, **{**_V1_YIELD_CASE, "when": "some_paths"})],
            r"chrys.synthetic:run:1: its case does not match rule 'event-loop-yield'",
            id="old-yield-reason-reviewed-case",
        ),
        pytest.param(
            [_v1_entry(1, reason=_V1_WORKFLOW_REASON, degradation_rule="")],
            "case fields must be non-empty strings",
            id="empty-case-field",
        ),
    ],
)
def test_v1_migration_rejects_what_it_cannot_map(entries: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        manifest_from_v1({"schema_version": 1, "nodes": entries})


def test_wait_inventory_covers_every_explicit_and_implicit_async_wait() -> None:
    nodes = _scanned()
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
    nodes = _scanned()
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
    nodes = _scanned()
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
    nodes = _scanned()
    expected = {
        "chrys.orchestration.invoker.acp.AcpConversation._wait_backoff",
        "chrys.orchestration.sub_agents.kernel_policy.KernelSubAgentPolicy._interruptible_sleep",
        "chrys.service.trajectory.preparation.preparation_lock",
    }
    assert expected <= {node.wrapper_target for node in nodes if node.wrapper_target is not None}


def test_wait_inventory_resolves_relative_import_wrapper_targets() -> None:
    nodes = _scanned()
    expected = {
        "validate_chat_options": "chrys.kernel._types.validate_chat_options",
        "apply_compaction": "chrys.kernel.compaction.apply_compaction",
        "spawn_acp_process": "chrys.service.acp_client.spawn.spawn_acp_process",
    }
    for expression, target in expected.items():
        matches = [node for node in nodes if expression in node.expression]
        assert matches, expression
        assert target in {node.wrapper_target for node in matches}


def test_mcp_owner_connect_waits_default_to_unresolved() -> None:
    manifest, errors = decode_manifest(load_manifest_text())
    assert errors == []
    nodes = [
        node
        for node in manifest.nodes
        if node.module == "chrys.service.mcp.owned" and node.qualname == "MCPTool._connect_on_owner"
    ]
    assert nodes
    assert {node.rule for node in nodes} == {UNRESOLVED_RULE}


def test_unresolved_wait_cases_declare_degradation_policy() -> None:
    """Pin the handoff: every known gap explicitly propagates Unresolved."""
    manifest, errors = decode_manifest(load_manifest_text())
    assert errors == []
    violations = [
        name
        for name, rule in sorted(manifest.rules.items())
        if rule["classification"] == "B"
        and (
            rule["container_rule"] == "nearest_recorded_operation" or rule["degradation_rule"] != UNRESOLVED_DEGRADATION
        )
    ]
    assert violations == []


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
