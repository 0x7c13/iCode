# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Requirement and C4 parsing keeps schema fields and relationship meaning."""

from __future__ import annotations

import pytest

from chrys.app.tui.widgets.markdown.diagram import compile_mermaid
from chrys.app.tui.widgets.markdown.diagram.layout import compile_ir
from chrys.app.tui.widgets.markdown.diagram.model import DiagnosticCode, DiagramKind, Direction, NodeShape
from chrys.app.tui.widgets.markdown.diagram.parser import parse_mermaid
from chrys.app.tui.widgets.markdown.diagram.parsers.common import MAX_EDGES, MAX_NODES
from chrys.app.tui.widgets.markdown.diagram.parsers.schemas import parse_c4, parse_requirement


def test_requirement_user_example_preserves_fields_and_connections() -> None:
    source = """requirementDiagram
requirement 登录功能 {
id: REQ-01
text: "用户可以使用账号密码登录"
risk: high
verifymethod: test
}
requirement 订单查询 {
id: REQ-02
text: "用户可以查询历史订单"
risk: medium
verifymethod: inspection
}
element 登录模块 {
type: component
}
element 订单模块 {
type: component
}
登录模块 - satisfies -> 登录功能
订单模块 - satisfies -> 订单查询
"""
    parsed = parse_requirement(source)
    assert not parsed.diagnostics
    assert parsed.kind is DiagramKind.REQUIREMENT
    assert [node.node_id for node in parsed.nodes] == ["登录功能", "订单查询", "登录模块", "订单模块"]
    assert parsed.nodes[0].sections == (
        ("id: REQ-01", "text: 用户可以使用账号密码登录", "risk: high", "verifymethod: test"),
    )
    assert [(edge.source, edge.target, edge.label) for edge in parsed.edges] == [
        ("登录模块", "登录功能", "satisfies"),
        ("订单模块", "订单查询", "satisfies"),
    ]
    compiled = compile_ir(source, parsed)
    assert not compiled.diagnostics
    assert "REQ-01" in "\n".join(compiled.rows)
    assert "REQ-02" in "\n".join(compiled.rows)


@pytest.mark.parametrize(
    "kind",
    [
        "requirement",
        "functionalRequirement",
        "interfaceRequirement",
        "performanceRequirement",
        "physicalRequirement",
        "designConstraint",
    ],
)
def test_requirement_types_keep_their_annotations(kind: str) -> None:
    parsed = parse_requirement(f"requirementDiagram\n{kind} req {{\nrisk: Low\nverifyMethod: Demonstration\n}}")
    assert not parsed.diagnostics
    assert parsed.nodes[0].annotation == kind


@pytest.mark.parametrize("relation", ["contains", "copies", "derives", "satisfies", "verifies", "refines", "traces"])
@pytest.mark.parametrize("reverse", [False, True])
def test_requirement_relations_support_forward_references_and_both_arrows(relation: str, reverse: bool) -> None:
    edge = (
        f'"Second requirement" <- {relation} - "First requirement"'
        if reverse
        else f'"First requirement" - {relation} -> "Second requirement"'
    )
    parsed = parse_requirement(
        f'requirementDiagram\n{edge}\nrequirement "First requirement" {{}}\nrequirement "Second requirement" {{}}'
    )
    assert not parsed.diagnostics
    assert [(edge.source, edge.target, edge.label) for edge in parsed.edges] == [
        ("First requirement", "Second requirement", relation)
    ]


def test_requirement_quotes_protect_braces_colons_arrows_and_comments() -> None:
    source = """requirementDiagram
# source comment
direction RL
requirement "输入 -> 输出" {
text: "Default: {} and %% literal"
}
element "模块: {}" {
docref: "chapter 1: {implementation}"
}
"模块: {}" - satisfies -> "输入 -> 输出"
"""
    parsed = parse_requirement(source)
    assert not parsed.diagnostics
    assert parsed.direction is Direction.RIGHT_LEFT
    assert parsed.nodes[0].sections == (("text: Default: {} and %% literal",),)
    assert parsed.edges[0].source == "模块: {}"
    assert parsed.edges[0].target == "输入 -> 输出"


@pytest.mark.parametrize(
    ("raw_text", "text"),
    [
        ("Customer's data must survive", "Customer's data must survive"),
        ("Preserve the customers' data", "Preserve the customers' data"),
        ('"Customer\'s data: {}"', "Customer's data: {}"),
    ],
)
def test_requirement_names_fields_and_relations_accept_ordinary_apostrophes(raw_text: str, text: str) -> None:
    source = (
        f"requirementDiagram\nrequirement Customer's policy {{\ntext: {raw_text}\n}}\n"
        "element O'Brien {\ntype: Owner's module\ndocref: Owner's manual\n}\n"
        "O'Brien - satisfies -> Customer's policy"
    )
    parsed = parse_mermaid(source)
    assert not parsed.diagnostics
    assert [node.label for node in parsed.nodes] == ["Customer's policy", "O'Brien"]
    assert parsed.nodes[0].sections == ((f"text: {text}",),)
    assert parsed.nodes[1].sections == (("type: Owner's module", "docref: Owner's manual"),)
    assert [(edge.source, edge.target) for edge in parsed.edges] == [("O'Brien", "Customer's policy")]
    diagram = compile_mermaid(source)
    assert not diagram.diagnostics
    assert "".join(text.split()) in "".join("".join(diagram.rows).split())


@pytest.mark.parametrize(
    "body",
    [
        "requirement R {\nrisk: severe\n}",
        "requirement R {\nverifyMethod: magic\n}",
        "requirement R {\nother: ignored\n}",
        "element E {\nid: 1\n}",
        'requirement R {\ntext: "unclosed\n}',
        "requirement R {\nrequirement N {\n}\n}",
        "requirement R {\nid: 1",
        "requirement R {}\nR - satisfies -> Missing",
        "requirement R {}\nrequirement R {}",
        'requirement "" {}',
        "requirement R {}\nR - unknown -> R",
    ],
)
def test_requirement_unsupported_or_malformed_source_fails_closed(body: str) -> None:
    parsed = parse_requirement(f"requirementDiagram\n{body}")
    assert parsed.has_fatal_error
    assert parsed.diagnostics


def test_schema_pending_edges_are_bounded_and_rejected_nodes_never_referenced() -> None:
    nodes = "\n".join(f"requirement N{index} {{}}" for index in range(MAX_NODES + 1))
    edges = "\n".join("N0 - satisfies -> N1" for _ in range(MAX_EDGES + 1))
    parsed = parse_requirement(f"requirementDiagram\n{nodes}\n{edges}\nN0 - satisfies -> N{MAX_NODES}")
    assert parsed.has_fatal_error
    assert len(parsed.nodes) == MAX_NODES
    assert len(parsed.edges) == MAX_EDGES
    assert {DiagnosticCode.NODE_LIMIT, DiagnosticCode.EDGE_LIMIT} <= {item.code for item in parsed.diagnostics}
    assert all(edge.target != f"N{MAX_NODES}" for edge in parsed.edges)


def test_whitespace_only_graph_title_does_not_add_empty_rows() -> None:
    source = 'C4Context\nPerson(p,"P")'
    titled = compile_mermaid('C4Context\ntitle &#10;\nPerson(p,"P")')
    plain = compile_mermaid(source)
    assert not titled.diagnostics
    assert titled.rows == plain.rows
    assert titled.height == plain.height


def test_c4_context_user_example_preserves_titles_descriptions_and_external_type() -> None:
    source = """C4Context
title 订单系统上下文
Person(user, "用户", "使用系统的人")
System(sys, "订单系统", "核心业务系统")
System_Ext(pay, "支付网关", "外部支付依赖")
Rel(user, sys, "下单")
Rel(sys, pay, "发起支付")
"""
    parsed = parse_c4(source)
    assert not parsed.diagnostics
    assert parsed.title == "订单系统上下文"
    assert parsed.nodes[0].annotation == "Person"
    assert parsed.nodes[0].sections == (("使用系统的人",),)
    assert parsed.nodes[2].annotation == "System_Ext"
    assert [(edge.source, edge.target, edge.label) for edge in parsed.edges] == [
        ("user", "sys", "下单"),
        ("sys", "pay", "发起支付"),
    ]
    assert not compile_ir(source, parsed).diagnostics


@pytest.mark.parametrize("kind", ["SystemDb", "SystemQueue_Ext", "ContainerDb", "ContainerQueue_Ext", "Component_Ext"])
def test_c4_node_variants_preserve_technology_and_description(kind: str) -> None:
    tail = '"PostgreSQL", "Durable data"' if kind.startswith(("Container", "Component")) else '"Durable data"'
    parsed = parse_c4(f'C4Container\n{kind}(db, "Database", {tail})')
    assert not parsed.diagnostics
    assert parsed.nodes[0].annotation == kind
    assert parsed.nodes[0].sections[0][-1] == "Durable data"
    assert parsed.nodes[0].shape is (NodeShape.CYLINDER if "Db" in kind else NodeShape.RECTANGLE)


def test_c4_optional_empty_arguments_and_quoted_commas_are_preserved() -> None:
    parsed = parse_c4(
        r"""C4Component
Component(模块, "用户, 订单", "", "Handles \"orders\" (v2)")
Component(网关, "API 网关",, "入口")
BiRel(模块, 网关, "请求, 回复", "HTTP", "同步")
Rel_Back(模块, 网关, "reverse")
"""
    )
    assert not parsed.diagnostics
    assert parsed.nodes[0].label == "用户, 订单"
    assert parsed.nodes[0].sections == (('Handles "orders" (v2)',),)
    assert parsed.edges[0].label == "请求, 回复 / HTTP / 同步"
    assert parsed.edges[0].source_marker == "◀"
    assert (parsed.edges[1].source, parsed.edges[1].target) == ("网关", "模块")


@pytest.mark.parametrize(
    "body",
    [
        'System(a, "A")\nRel(a, missing, "missing")',
        'System_Boundary(b, "Boundary") {\nSystem(a, "A")\n}',
        'System(a, "A", "descr", "sprite")',
        'System(a, "A", $descr="named unsupported")',
        'System(a, "unclosed)',
        'Rel(a, b, "label", "protocol", "description", "unsupported")',
        'System(a, "A")\nRel_U(a, a, "unsupported directional layout")',
        'System(, "A")',
        'System(a, "")',
        'Unknown(a, "A")',
    ],
)
def test_c4_unsupported_semantics_and_malformed_arguments_fail_closed(body: str) -> None:
    parsed = parse_c4(f"C4Context\n{body}")
    assert parsed.has_fatal_error
    assert parsed.diagnostics


def test_c4_relations_may_precede_their_nodes() -> None:
    parsed = parse_c4('C4Context\nRel(a, b, "uses")\nSystem(a, "A")\nSystem(b, "B")')
    assert not parsed.diagnostics
    assert len(parsed.edges) == 1


@pytest.mark.parametrize(
    ("header", "body", "kind"),
    [
        ("requirementDiagram", "requirement Login {}", DiagramKind.REQUIREMENT),
        ("C4Context", 'System(login, "Login")', DiagramKind.C4),
        ("C4Container", 'Container(login, "Login", "Python")', DiagramKind.C4),
        ("C4Component", 'Component(login, "Login", "Python")', DiagramKind.C4),
    ],
)
def test_schema_compiler_entrypoint_preserves_titles_after_leading_comments(
    header: str, body: str, kind: DiagramKind
) -> None:
    source = f"%% first comment\n\n%% second comment\n{header} %% header comment\ntitle Login diagram\n{body}"
    parsed = parse_mermaid(source)
    assert not parsed.diagnostics
    assert parsed.kind is kind
    assert parsed.nodes[0].line == 6
    compiled = compile_mermaid(source)
    assert not compiled.diagnostics
    assert compiled.kind is kind
    assert "Login diagram" in "\n".join(compiled.rows)


@pytest.mark.parametrize("header", ["requirementDiagram", "C4Context", "C4Container", "C4Component"])
@pytest.mark.parametrize("suffix", [" unexpected", ":"])
def test_schema_compiler_entrypoint_rejects_malformed_header_suffix(header: str, suffix: str) -> None:
    body = "requirement Login {}" if header == "requirementDiagram" else 'System(login, "Login")'
    source = f"%% leading comment\n{header}{suffix}\n{body}"
    parsed = parse_mermaid(source)
    assert parsed.has_fatal_error
    assert parsed.diagnostics[0].line == 2
    assert not parsed.nodes
    assert compile_mermaid(source).diagnostics
