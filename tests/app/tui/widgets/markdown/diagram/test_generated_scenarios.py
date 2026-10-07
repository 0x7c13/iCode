# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Model-authored Mermaid scenarios and seeded structural regression checks."""

from __future__ import annotations

import random

import pytest

from chrys.app.tui.widgets.markdown.diagram import compile_mermaid
from chrys.app.tui.widgets.markdown.diagram.model import DiagnosticSeverity, EdgeStyle, NodeShape
from chrys.app.tui.widgets.markdown.diagram.parser import MAX_EDGES, parse_mermaid
from chrys.app.tui.widgets.markdown.parser import _create_markdown_parser, _parse_tokens

MODEL_DIAGRAMS = (
    'flowchart LR; Read["Read items[i] and f(x)"] --> Done[Done]',
    "flowchart TB;\nA[API] & B[Worker] --> C[(Cache)] & D[(Database)] --> E[Audit]",
    'flowchart LR\nA["Customer\'s [account]"] -->|"status | reason"| B["Retry (safe)"]',
    'flowchart LR\nA["First line\nSecond line"] --> B["`Markdown label`"]',
    "flowchart LR\nA e1@--> B\ne1@{ animate: true }\nB --> C",
    '%%{init: {\n"theme": "dark"\n}}%%\nflowchart LR\nA[Input] --> B[Result]',
    'classDiagram\nclass User { +String name; +login() bool }\nUser "1" --> "*" Session : owns',
    "classDiagram\nclass `HTTP-Client`\n`HTTP-Client` : +send() Response\n`HTTP-Client` .. Transport : uses",
    "classDiagram\nclass Repository~T~\nRepository o--* Entity : stores\nEntity <|.. DTO",
    "sequenceDiagram\nparticipant A as Client\nparticipant B as Server\nA<<->>B: negotiate\nB<<-->>A: confirmed",
    "sequenceDiagram; A->>B: start; B-->>A: ready#59; next\nA->A: inspect\nA-->B: signal",
    "stateDiagram-v2\n[*] --> Idle\nIdle --> Running : start\nRunning --> Idle : stop\nRunning --> [*] : shutdown",
    "erDiagram\nACCOUNT ||--o{ ENTRY : owns\nACCOUNT {\nint id PK\nstring name\n}\nENTRY {\nint id PK\nint account_id FK\n}",
)


@pytest.mark.parametrize("source", MODEL_DIAGRAMS)
def test_generated_scenarios_render_through_markdown(source: str) -> None:
    ir = parse_mermaid(source)
    assert not ir.has_fatal_error, ir.diagnostics
    result = compile_mermaid(source)
    assert not any(item.severity is DiagnosticSeverity.ERROR for item in result.diagnostics)
    assert result.source == source
    blocks = _parse_tokens(_create_markdown_parser().parse(f"```mermaid\n{source}\n```"))
    assert len(blocks) == 1
    assert blocks[0].diagram is not None


def test_ampersand_chains_expand_every_edge_without_cross_stage_connections() -> None:
    ir = parse_mermaid(MODEL_DIAGRAMS[1])
    assert {(edge.source, edge.target) for edge in ir.edges} == {
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
        ("C", "E"),
        ("D", "E"),
    }


def test_quoted_delimiters_stay_inside_labels() -> None:
    ir = parse_mermaid(MODEL_DIAGRAMS[2])
    assert [node.label for node in ir.nodes] == ["Customer's [account]", "Retry (safe)"]
    assert ir.edges[0].label == "status | reason"
    assert [node.label for node in parse_mermaid(MODEL_DIAGRAMS[3]).nodes] == [
        "First line Second line",
        "Markdown label",
    ]


def test_edge_metadata_does_not_create_a_spurious_node() -> None:
    ir = parse_mermaid(MODEL_DIAGRAMS[4])
    assert [node.node_id for node in ir.nodes] == ["A", "B", "C"]
    assert len(ir.edges) == 2
    assert all(item.severity is DiagnosticSeverity.WARNING for item in ir.diagnostics)


def test_inline_class_members_and_multiplicity_are_retained() -> None:
    ir = parse_mermaid(MODEL_DIAGRAMS[6])
    assert ir.nodes[0].sections == (("+String name",), ("+login() bool",))
    assert (ir.edges[0].source_label, ir.edges[0].target_label) == ("1", "*")


@pytest.mark.parametrize(
    ("relation", "source_marker", "target_marker", "directed"),
    [("..", "", "", False), ("o--*", "◇", "◆", False), ("<|--|>", "△", "△", True), ("<..>", "◀", "▶", True)],
)
def test_two_sided_class_relations_keep_both_markers(
    relation: str, source_marker: str, target_marker: str, directed: bool
) -> None:
    ir = parse_mermaid(f"classDiagram\nA {relation} B")
    assert not ir.has_fatal_error
    edge = ir.edges[0]
    assert (edge.source, edge.target) == ("A", "B")
    assert (edge.source_marker, edge.target_marker, edge.directed) == (source_marker, target_marker, directed)


def test_sequence_arrows_preserve_heads_in_both_directions() -> None:
    source = MODEL_DIAGRAMS[9]
    ir = parse_mermaid(source)
    assert all(edge.source_marker == "◀" and edge.directed for edge in ir.edges)
    assert [edge.style for edge in ir.edges] == [EdgeStyle.SOLID, EdgeStyle.DOTTED]
    rows = compile_mermaid(source).rows
    assert sum("◀" in row and "▶" in row for row in rows) == 2


def test_headless_sequence_lines_are_not_silently_given_arrowheads() -> None:
    source = "sequenceDiagram\nA->B: plain\nB-->A: dashed"
    ir = parse_mermaid(source)
    assert all(not edge.directed and not edge.target_marker for edge in ir.edges)
    assert not any("▶" in row or "◀" in row for row in compile_mermaid(source).rows)


def test_expanded_fanout_obeys_existing_resource_limit() -> None:
    group = " & ".join(f"N{i}" for i in range(30))
    ir = parse_mermaid(f"flowchart LR\n{group} --> {group}")
    assert ir.has_fatal_error
    assert len(ir.edges) == MAX_EDGES


def test_seeded_quoted_labels_and_fanout_preserve_graph_identity() -> None:
    rng = random.Random(52171)
    labels = ["input[i]", "call(x)", "a | b", "brace {x}", "Customer's order", "中文流程", "x; y"]
    for _ in range(40):
        chosen = rng.sample(labels, 4)
        source = "flowchart LR\n" + " & ".join(f'N{i}["{chosen[i]}"]' for i in range(2))
        source += " --> " + " & ".join(f'N{i}["{chosen[i]}"]' for i in range(2, 4))
        ir = parse_mermaid(source)
        assert not ir.has_fatal_error, source
        assert [node.label for node in ir.nodes] == chosen
        assert len(ir.edges) == 4
        assert not any(item.severity is DiagnosticSeverity.ERROR for item in compile_mermaid(source).diagnostics)


@pytest.mark.parametrize(
    "source",
    [
        'flowchart LR\nA["unclosed] --> B',
        "flowchart LR\nA & --> B",
        "flowchart LR\nA --> B &",
        "flowchart LR\nA --> B\n%%{init: {",
        "classDiagram\nA <|--? B",
    ],
)
def test_malformed_graphs_still_fail_closed(source: str) -> None:
    assert parse_mermaid(source).has_fatal_error


@pytest.mark.parametrize("prefix", ['%% dangling "comment\n', ""])
def test_quotes_in_comments_do_not_swallow_graph_structure(prefix: str) -> None:
    source = prefix + 'flowchart LR\nA --> B %% unmatched "comment\n%% more "comment\nB --> C'
    ir = parse_mermaid(source)
    assert not ir.has_fatal_error
    assert [node.node_id for node in ir.nodes] == ["A", "B", "C"]
    assert len(ir.edges) == 2


def test_multiline_labels_keep_metadata_and_comment_like_text() -> None:
    ir = parse_mermaid('flowchart LR\nA["line 1\naccTitle: literal title\n100%% accurate\nline 4"] --> B')
    assert not ir.has_fatal_error
    assert ir.nodes[0].label == "line 1 accTitle: literal title 100%% accurate line 4"
    assert len(ir.edges) == 1


@pytest.mark.parametrize("label", ["it's ready", "unmatched ( text", 'say "hello', "code `unfinished"])
def test_sequence_prose_does_not_swallow_following_messages(label: str) -> None:
    ir = parse_mermaid(f"sequenceDiagram\nA->>B: {label}; B->>A: ok")
    assert not ir.has_fatal_error
    assert [edge.label for edge in ir.edges] == [label, "ok"]


@pytest.mark.parametrize("relation", ["--> B", ".. B", "<|-- B", "A -->"])
def test_class_relation_requires_both_endpoints(relation: str) -> None:
    assert parse_mermaid("classDiagram\n" + relation).has_fatal_error


def test_forward_edge_metadata_keeps_only_real_nodes() -> None:
    ir = parse_mermaid("flowchart LR\ne1@{ animate: true }\nA e1@--> B")
    assert not ir.has_fatal_error
    assert [node.node_id for node in ir.nodes] == ["A", "B"]
    assert len(ir.edges) == 1
    assert ir.diagnostics and all(item.severity is DiagnosticSeverity.WARNING for item in ir.diagnostics)


def test_class_generic_does_not_consume_tildes_in_label() -> None:
    ir = parse_mermaid('classDiagram\nclass Box~T~["tilde ~ marker ~"]')
    assert not ir.has_fatal_error
    assert ir.nodes[0].label == "tilde ~ marker ~<T>"


def test_backtick_class_references_work_in_annotations_notes_and_members() -> None:
    ir = parse_mermaid(
        'classDiagram\nclass `HTTP Client`\n<<service>> `HTTP Client`\nnote for `HTTP Client` "retry"\n`HTTP Client` : +get()'
    )
    assert not ir.has_fatal_error
    assert len(ir.nodes) == 1
    assert ir.nodes[0].annotation == "service"
    assert ir.nodes[0].notes == ("retry",)
    assert ir.nodes[0].sections == ((), ("+get()",))


@pytest.mark.parametrize(
    ("metadata", "label"),
    [
        ('label: "Customer\'s order", shape: rect', "Customer's order"),
        ("label: 'brace } and comma, preserved', shape: rect", "brace } and comma, preserved"),
        ('label: "shape: diam, label: text", shape: rect', "shape: diam, label: text"),
        ('label: "escaped \\"quote\\"", shape: rect', 'escaped "quote"'),
        ("label: 'Customer''s order', shape: rect", "Customer's order"),
        ("label: 'inch \" marker', shape: rect", 'inch " marker'),
        ("label: 'brace }; preserved', shape: rect", "brace }; preserved"),
        ("label: 'backslash \\', shape: rect", "backslash \\"),
    ],
)
def test_flow_metadata_preserves_complete_quoted_labels(metadata: str, label: str) -> None:
    ir = parse_mermaid(f"flowchart LR\nA@{{{metadata}}} --> B\nB --> C")
    assert not ir.has_fatal_error
    assert ir.nodes[0].label == label
    assert len(ir.edges) == 2


@pytest.mark.parametrize("value", ['"x" junk', "'x' junk"])
def test_malformed_quoted_metadata_fails_closed(value: str) -> None:
    assert parse_mermaid(f"flowchart LR\nA@{{label: {value}, shape: rect}} --> B").has_fatal_error


@pytest.mark.parametrize(
    "label",
    [
        "one; two",
        "one; and two",
        "phase; end of work",
        "code#59; literal",
        "one; note over coffee",
        "one; note left of the building",
        "one; create participant diagrams is useful",
    ],
)
def test_sequence_keeps_legacy_semicolons_in_message_prose(label: str) -> None:
    ir = parse_mermaid(f"sequenceDiagram\nA->>B: {label}")
    assert not ir.has_fatal_error
    assert len(ir.edges) == 1
    assert ir.edges[0].label == label.replace("#59;", ";")


def test_sequence_can_mix_prose_semicolons_and_real_statement_separators() -> None:
    ir = parse_mermaid("sequenceDiagram; A->>B: one; two; B->>A: three; four; activate A")
    assert not ir.has_fatal_error
    assert [edge.label for edge in ir.edges] == ["one; two", "three; four"]


@pytest.mark.parametrize("alias", ["DB;read", "DB#59;read", "DB&#59;read", "DB#quot;read"])
@pytest.mark.parametrize("declaration", ["participant", "actor", "create participant"])
def test_sequence_separator_preserves_complete_participant_metadata(alias: str, declaration: str) -> None:
    source = f'sequenceDiagram\nA->>B: start; {declaration} C@{{"type": "database", "alias": "{alias}"}}; C->>A: ready'
    ir = parse_mermaid(source)
    assert not ir.has_fatal_error, ir.diagnostics
    assert [edge.label for edge in ir.edges] == ["start", "ready"]
    participant = next(node for node in ir.nodes if node.node_id == "C")
    assert participant.shape is NodeShape.CYLINDER
    assert participant.label == alias.replace("&#59;", ";").replace("#59;", ";").replace("#quot;", '"')


@pytest.mark.parametrize("label", ["Customer's order", "O'Reilly's docs", '12" monitor'])
def test_bare_metadata_quotes_are_literal_inside_the_value(label: str) -> None:
    ir = parse_mermaid(f"flowchart LR\nA@{{label: {label}, shape: rect}} --> B; B --> C")
    assert not ir.has_fatal_error, ir.diagnostics
    assert [node.label for node in ir.nodes] == [label, "B", "C"]
    assert [(edge.source, edge.target) for edge in ir.edges] == [("A", "B"), ("B", "C")]


@pytest.mark.parametrize("ambiguous_metadata", [False, True])
def test_sequence_lookahead_work_is_bounded_by_source_size(
    monkeypatch: pytest.MonkeyPatch, ambiguous_metadata: bool
) -> None:
    from chrys.app.tui.widgets.markdown.diagram.parsers import graphs

    suffix = "; participant C@{" * 500 if ambiguous_metadata else ";" * 6000 + "done"
    source = "sequenceDiagram\nA->>B: " + suffix
    scanned = 0
    consume = graphs._StatementBudget.consume

    def count_scan(budget: graphs._StatementBudget) -> None:
        nonlocal scanned
        scanned += 1
        consume(budget)

    monkeypatch.setattr(graphs._StatementBudget, "consume", count_scan)
    ir = parse_mermaid(source)
    assert scanned <= 5 * len(source)
    if ambiguous_metadata:
        assert ir.has_fatal_error
        blocks = _parse_tokens(_create_markdown_parser().parse(f"```mermaid\n{source}\n```"))
        assert blocks[0].diagram is None
        assert blocks[0].content.plain == source
    else:
        assert not ir.has_fatal_error
        assert [edge.label for edge in ir.edges] == [suffix]


def test_sequence_scan_budget_accepts_long_valid_statement_chains() -> None:
    source = "sequenceDiagram; " + "; ".join(f"A->>B: message {index}" for index in range(100))
    ir = parse_mermaid(source)
    assert not ir.has_fatal_error
    assert [edge.label for edge in ir.edges] == [f"message {index}" for index in range(100)]


@pytest.mark.parametrize(
    ("source", "label"),
    [("done;", "done"), ("done;   ", "done"), ("first; last;", "first; last"), ("done#59;", "done;")],
)
def test_sequence_trailing_separator_ends_message_but_escaped_semicolon_is_literal(source: str, label: str) -> None:
    ir = parse_mermaid(f"sequenceDiagram\nA->>B: {source}\nB-->>A: ready;")
    assert not ir.has_fatal_error
    assert [edge.label for edge in ir.edges] == [label, "ready"]


@pytest.mark.parametrize(
    ("source", "label"),
    [
        ("done#quot;", 'done"'),
        ("value #infin;", "value ∞"),
        ("I #9829; you #infin; times more!", "I ♥ you ∞ times more!"),
        ("#quot;quoted#quot;", '"quoted"'),
        ("#amp;lt#59;", "&lt;"),
        ("#38;lt#59;", "&lt;"),
        ("&#x3b;", ";"),
        ("#unknownName;", "#unknownName;"),
    ],
)
@pytest.mark.parametrize("suffix", ["", "; B-->>A: ready;"])
def test_sequence_entities_decode_once_without_splitting_statements(source: str, label: str, suffix: str) -> None:
    ir = parse_mermaid(f"sequenceDiagram\nA->>B: {source}{suffix}")
    assert not ir.has_fatal_error
    expected = [label, "ready"] if suffix else [label]
    assert [edge.label for edge in ir.edges] == expected


def test_flow_labels_decode_mermaid_named_entities_after_parsing() -> None:
    ir = parse_mermaid('flowchart LR; A["#quot;quoted#quot;"] --> B["#infin;"]; B --> C["#unknownName;"]')
    assert not ir.has_fatal_error
    assert [node.label for node in ir.nodes] == ['"quoted"', "∞", "#unknownName;"]


@pytest.mark.parametrize("entity", ["#amp;lt;", "#38;lt;", "&amp;lt;"])
def test_class_generic_labels_decode_entities_once(entity: str) -> None:
    ir = parse_mermaid(f'classDiagram\nclass Box~{entity}~\nclass Alias~{entity}~ as "Label"')
    assert not ir.has_fatal_error
    assert [node.label for node in ir.nodes] == ["Box<&lt;>", "Label<&lt;>"]


@pytest.mark.parametrize("entity", ["#amp;lt;", "#38;lt;", "&amp;lt;"])
def test_er_attribute_comments_decode_entities_once(entity: str) -> None:
    ir = parse_mermaid(f'erDiagram\nRecord {{\nstring value "{entity}"\n}}')
    assert not ir.has_fatal_error
    assert ir.nodes[0].sections == (("string value — &lt;",),)


@pytest.mark.parametrize("header", ["classDiagram", "sequenceDiagram"])
def test_flow_label_joining_is_selected_only_by_the_header(header: str) -> None:
    from chrys.app.tui.widgets.markdown.diagram.parsers.common import _source_lines

    source = header + '\ngraph LR\nA: unmatched "\nB --> C'
    assert _source_lines(source, join_labels=True) == list(enumerate(source.splitlines(), 1))
