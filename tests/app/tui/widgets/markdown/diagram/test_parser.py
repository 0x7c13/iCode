# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the bounded Mermaid-subset parser."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest

from chrys.app.tui.widgets.markdown.diagram import compile_mermaid
from chrys.app.tui.widgets.markdown.diagram.model import (
    DiagnosticCode,
    DiagnosticSeverity,
    DiagramKind,
    Direction,
    EdgeStyle,
    NodeShape,
    PieChart,
    QuadrantChart,
    TreemapChart,
    XYChart,
)
from chrys.app.tui.widgets.markdown.diagram.parser import (
    MAX_DIAGNOSTICS,
    MAX_EDGES,
    MAX_NODES,
    MAX_SOURCE_BYTES,
    parse_mermaid,
)
from chrys.app.tui.widgets.markdown.diagram.parsers import graphs


def test_flowchart_parses_chains_shapes_labels_and_compact_edges() -> None:
    diagram = parse_mermaid(
        """flowchart LR
        request[Request]-->|dispatch|engine(Engine)-.->decision{Ready?}
        decision --- request
        """
    )

    assert diagram.kind is DiagramKind.FLOWCHART
    assert [node.node_id for node in diagram.nodes] == ["request", "engine", "decision"]
    assert [node.shape for node in diagram.nodes] == [NodeShape.RECTANGLE, NodeShape.ROUNDED, NodeShape.DECISION]
    assert [(edge.source, edge.target) for edge in diagram.edges] == [
        ("request", "engine"),
        ("engine", "decision"),
        ("decision", "request"),
    ]
    assert diagram.edges[0].label == "dispatch"
    assert diagram.edges[1].style is EdgeStyle.DOTTED
    assert not diagram.edges[2].directed
    assert not diagram.diagnostics


def test_flowchart_accepts_common_shapes_edge_variants_and_reverse_arrows() -> None:
    diagram = parse_mermaid(
        """flowchart LR
        A((Circle)) --> B[(Database)] --> C[[Subroutine]] --> D([Stadium]) --> E{{Hexagon}}
        A <--> B
        B o--o C
        C x--x D
        D o--> E
        E <-- A
        A-. dotted .->C
        B== heavy ==>D
        """
    )

    assert [node.shape for node in diagram.nodes] == [
        NodeShape.CIRCLE,
        NodeShape.CYLINDER,
        NodeShape.SUBROUTINE,
        NodeShape.STADIUM,
        NodeShape.HEXAGON,
    ]
    bidirectional, circles, crosses, circle_arrow, reverse, dotted, heavy = diagram.edges[4:]
    assert (bidirectional.source_marker, bidirectional.target_marker) == ("◀", "▶")
    assert (circles.source_marker, circles.target_marker) == ("○", "○")
    assert (crosses.source_marker, crosses.target_marker) == ("x", "x")
    assert (circle_arrow.source_marker, circle_arrow.target_marker) == ("○", "")
    assert (reverse.source, reverse.target) == ("A", "E")
    assert (dotted.style, dotted.label) == (EdgeStyle.DOTTED, "dotted")
    assert (heavy.style, heavy.label) == (EdgeStyle.HEAVY, "heavy")
    assert not diagram.diagnostics


def test_flowchart_normalizes_variable_length_links_without_changing_structure() -> None:
    diagram = parse_mermaid(
        """flowchart LR
        A ---> B
        B ---- C
        C -..-> D
        D -..- E
        E ===> F
        F ==== G
        G -- labelled ---> H
        """
    )

    assert [(edge.style, edge.directed, edge.label) for edge in diagram.edges] == [
        (EdgeStyle.SOLID, True, ""),
        (EdgeStyle.SOLID, False, ""),
        (EdgeStyle.DOTTED, True, ""),
        (EdgeStyle.DOTTED, False, ""),
        (EdgeStyle.HEAVY, True, ""),
        (EdgeStyle.HEAVY, False, ""),
        (EdgeStyle.SOLID, True, "labelled"),
    ]
    assert not diagram.diagnostics


def test_flowchart_accepts_legacy_shape_delimiters_as_terminal_approximations() -> None:
    diagram = parse_mermaid(
        r"""flowchart LR
        A>Flag] --> B[/Input/] --> C[\Output\] --> D[/Trap\] --> E[\Trap alt/] --> F(((Done)))
        """
    )

    assert [node.shape for node in diagram.nodes] == [
        NodeShape.SLANTED,
        NodeShape.SLANTED,
        NodeShape.SLANTED,
        NodeShape.SLANTED,
        NodeShape.SLANTED,
        NodeShape.CIRCLE,
    ]
    assert [node.label for node in diagram.nodes] == ["Flag", "Input", "Output", "Trap", "Trap alt", "Done"]
    assert not diagram.diagnostics


def test_flowchart_maps_common_shape_metadata_to_terminal_shapes() -> None:
    diagram = parse_mermaid(
        """flowchart LR
        A@{ shape: diam, label: "Choose" } --> B@{ shape: database } --> C@{ shape: lean-r, label: "Output" }
        """
    )

    assert [(node.label, node.shape) for node in diagram.nodes] == [
        ("Choose", NodeShape.DECISION),
        ("B", NodeShape.CYLINDER),
        ("Output", NodeShape.SLANTED),
    ]
    assert [(edge.source, edge.target) for edge in diagram.edges] == [("A", "B"), ("B", "C")]
    assert not diagram.diagnostics


def test_frontmatter_accessibility_metadata_and_entities_do_not_change_graph_structure() -> None:
    diagram = parse_mermaid(
        """---
        title: Entity example
        config:
          theme: dark
        ---
        flowchart LR
        accTitle: Accessible title
        accDescr {
          Accessible description
        }
        A[Value #35; &amp; ready] --> B; B --> C
        """
    )

    assert [node.label for node in diagram.nodes] == ["Value # & ready", "B", "C"]
    assert [(edge.source, edge.target) for edge in diagram.edges] == [("A", "B"), ("B", "C")]
    assert not diagram.diagnostics


def test_flowchart_flattens_subgraphs_and_ignores_non_structural_directives_as_warnings() -> None:
    diagram = parse_mermaid(
        """flowchart LR
        %%{init: {"flowchart": {"curve": "basis"}}}%%
        classDef hot fill:#f00
        subgraph Workers[Workers]
          direction TB
          A[Decode]:::hot:::selected --> B[Execute]:::hot
        end
        style A fill:#fff
        class A,B hot
        linkStyle 0 stroke:#fff
        click A "https://example.com"
        """
    )

    assert diagram.direction is Direction.LEFT_RIGHT
    assert [node.node_id for node in diagram.nodes] == ["A", "B"]
    assert [(edge.source, edge.target) for edge in diagram.edges] == [("A", "B")]
    assert diagram.diagnostics
    assert {diagnostic.severity for diagnostic in diagram.diagnostics} == {DiagnosticSeverity.WARNING}


def test_flowchart_ignores_unicode_class_suffixes() -> None:
    diagram = parse_mermaid("flowchart LR\n甲[开始]:::高亮:::已选 --> 乙[完成]:::结束")

    assert [(node.node_id, node.label) for node in diagram.nodes] == [("甲", "开始"), ("乙", "完成")]
    assert [(edge.source, edge.target) for edge in diagram.edges] == [("甲", "乙")]
    assert not diagram.diagnostics


def test_class_diagram_groups_attributes_and_methods_and_relations() -> None:
    diagram = parse_mermaid(
        """classDiagram
        class User {
          +str name
          +login()
        }
        User <|-- Admin
        User "1" o-- "*" Session : owns
        """
    )

    user = next(node for node in diagram.nodes if node.node_id == "User")
    inheritance, aggregation = diagram.edges
    assert user.sections == (("+str name",), ("+login()",))
    assert (inheritance.source, inheritance.target, inheritance.target_marker) == ("Admin", "User", "△")
    assert aggregation.source_marker == "◇"
    assert (aggregation.source_label, aggregation.target_label, aggregation.label) == ("1", "*", "owns")
    assert not diagram.diagnostics


def test_class_diagram_parses_realization_in_both_directions() -> None:
    diagram = parse_mermaid(
        """classDiagram
        OperationBinding <|.. SubAgentToolShell
        AcpSubAgentPolicy ..|> Policy
        """
    )

    first, second = diagram.edges
    assert (first.source, first.target, first.style, first.target_marker) == (
        "SubAgentToolShell",
        "OperationBinding",
        EdgeStyle.DOTTED,
        "△",
    )
    assert (second.source, second.target, second.style, second.target_marker) == (
        "AcpSubAgentPolicy",
        "Policy",
        EdgeStyle.DOTTED,
        "△",
    )
    assert not diagram.diagnostics


def test_class_diagram_accepts_labels_annotations_and_ignored_decorations() -> None:
    diagram = parse_mermaid(
        """classDiagram
        class Service["Public API"] <<interface>>
        class Impl {
          <<service>>
          +run()
        }
        style Service fill:#fff
        click Service href "https://example.com"
        Service --> Impl
        """
    )

    service, implementation = diagram.nodes
    assert (service.label, service.annotation) == ("Public API", "interface")
    assert (implementation.annotation, implementation.sections) == ("service", ((), ("+run()",)))
    assert [(edge.source, edge.target) for edge in diagram.edges] == [("Service", "Impl")]
    assert {diagnostic.severity for diagnostic in diagram.diagnostics} == {DiagnosticSeverity.WARNING}


def test_class_namespaces_are_flattened_and_targeted_notes_are_retained() -> None:
    diagram = parse_mermaid(
        """classDiagram
        namespace Company {
          namespace API["Public API"] {
            class Service {
              +run()
            }
            note for Service "External\\nentry point"
          }
        }
        Service --> Impl
        """
    )

    service = next(node for node in diagram.nodes if node.node_id == "Service")
    assert service.notes == ("External entry point",)
    assert service.sections == ((), ("+run()",))
    assert [(edge.source, edge.target) for edge in diagram.edges] == [("Service", "Impl")]
    assert diagram.diagnostics
    assert {diagnostic.severity for diagnostic in diagram.diagnostics} == {DiagnosticSeverity.WARNING}


def test_class_generic_declaration_uses_plain_identifier_for_relations() -> None:
    diagram = parse_mermaid(
        """classDiagram
        class Square~Shape~ {
          +draw()
        }
        Square --> Canvas
        """
    )

    square = next(node for node in diagram.nodes if node.node_id == "Square")
    assert square.label == "Square<Shape>"
    assert square.sections == ((), ("+draw()",))
    assert [(edge.source, edge.target) for edge in diagram.edges] == [("Square", "Canvas")]
    assert not diagram.diagnostics


def test_er_diagram_parses_entities_attributes_keys_and_relationships() -> None:
    diagram = parse_mermaid(
        """erDiagram
        Student ||--o{ Enrollment : enrolls
        Course ||--o{ Enrollment : offered_in
        Professor ||--o{ Course : teaches
        Student {
            string student_id PK
            string name
            string major
        }
        Course {
            string course_id PK
            string title
            int credits
        }
        Enrollment {
            string student_id FK
            string course_id FK
            date enroll_date
        }
        Professor {
            string prof_id PK
            string name
            string dept
        }
        """
    )

    assert diagram.kind is DiagramKind.ER
    entities = {node.node_id: node for node in diagram.nodes}
    assert entities["Student"].sections == (("string student_id [PK]", "string name", "string major"),)
    assert entities["Enrollment"].sections == (("string student_id [FK]", "string course_id [FK]", "date enroll_date"),)
    assert [(edge.source, edge.target, edge.source_label, edge.target_label, edge.label) for edge in diagram.edges] == [
        ("Student", "Enrollment", "1", "0..*", "enrolls"),
        ("Course", "Enrollment", "1", "0..*", "offered_in"),
        ("Professor", "Course", "1", "0..*", "teaches"),
    ]
    assert all(not edge.directed for edge in diagram.edges)
    assert not diagram.diagnostics


def test_er_diagram_accepts_aliases_unicode_cardinalities_and_attribute_metadata() -> None:
    diagram = parse_mermaid(
        """erDiagram
        direction LR
        p[Person] ||--o| a["Customer Account"] : has
        "驾驶员 档案" }o..|{ p : uses
        p {
            string drivers_license PK, UK "政府签发编号"
            string? middle_name
            decimal(10,2) balance
        }
        """
    )

    assert diagram.direction is Direction.LEFT_RIGHT
    assert [(node.node_id, node.label) for node in diagram.nodes] == [
        ("p", "Person"),
        ("a", "Customer Account"),
        ("驾驶员 档案", "驾驶员 档案"),
    ]
    assert diagram.nodes[0].sections == (
        (
            "string drivers_license [PK,UK] — 政府签发编号",
            "string? middle_name",
            "decimal(10,2) balance",
        ),
    )
    assert [(edge.source_label, edge.target_label, edge.style) for edge in diagram.edges] == [
        ("1", "0..1", EdgeStyle.SOLID),
        ("0..*", "1..*", EdgeStyle.DOTTED),
    ]
    assert not diagram.diagnostics


def test_er_diagram_accepts_word_relationships_and_ignored_styles() -> None:
    diagram = parse_mermaid(
        """erDiagram
        CAR 1 to zero or more NAMED-DRIVER : allows
        PERSON many(0) optionally to 0+ NAMED-DRIVER : is
        CAR:::vehicle ||--|| GARAGE:::building : parks_in
        style CAR fill:#fff
        classDef vehicle stroke:#333
        class CAR vehicle
        """
    )

    assert [(edge.source_label, edge.target_label, edge.style) for edge in diagram.edges] == [
        ("1", "0..*", EdgeStyle.SOLID),
        ("0..*", "0..*", EdgeStyle.DOTTED),
        ("1", "1", EdgeStyle.SOLID),
    ]
    assert {diagnostic.severity for diagnostic in diagram.diagnostics} == {DiagnosticSeverity.WARNING}
    assert not diagram.has_fatal_error


def test_er_diagram_reopened_entity_accumulates_attributes() -> None:
    diagram = parse_mermaid(
        """erDiagram
        User {
            int id PK
        }
        User {
            string name
        }
        """
    )

    assert diagram.nodes[0].sections == (("int id [PK]", "string name"),)
    assert not diagram.diagnostics


def test_er_attribute_comments_may_contain_braces() -> None:
    source = 'erDiagram\nUser {\nstring data "Default: {}"\nint id PK "Key {user_id}"\n}'
    diagram = parse_mermaid(source)

    assert not diagram.diagnostics
    assert diagram.nodes[0].sections == (("string data — Default: {}", "int id [PK] — Key {user_id}"),)
    compiled = compile_mermaid(source)
    assert not compiled.diagnostics
    assert "Default: {}" in "\n".join(compiled.rows)
    assert "Key {user_id}" in "\n".join(compiled.rows)


def test_er_actual_nested_entity_body_still_fails_closed() -> None:
    diagram = parse_mermaid("erDiagram\nUser {\nNested {\n}")

    assert diagram.has_fatal_error
    assert diagram.diagnostics[0].code is DiagnosticCode.NESTED_ER_BODY


@pytest.mark.parametrize("direction", ["TB", "TD", "BT", "LR", "RL"])
def test_er_diagram_accepts_all_official_directions(direction: str) -> None:
    diagram = parse_mermaid(f"erDiagram\ndirection {direction}\nA ||--o{{ B : contains")

    expected = {
        "TB": Direction.TOP_DOWN,
        "TD": Direction.TOP_DOWN,
        "BT": Direction.BOTTOM_UP,
        "LR": Direction.LEFT_RIGHT,
        "RL": Direction.RIGHT_LEFT,
    }[direction]
    assert diagram.direction is expected
    assert not diagram.diagnostics


@pytest.mark.parametrize(
    "source",
    [
        "erDiagram\nEntity {\nstring id",
        "erDiagram\nEntity {\nbad\n}",
        "erDiagram\n}",
        "erDiagram\nA ||-- B : malformed",
    ],
)
def test_er_diagram_fails_closed_for_malformed_entity_syntax(source: str) -> None:
    diagram = parse_mermaid(source)

    assert diagram.kind is DiagramKind.ER
    assert diagram.has_fatal_error
    assert diagram.diagnostics


def test_state_and_sequence_create_pseudo_states_and_implicit_participants() -> None:
    state = parse_mermaid(
        """stateDiagram-v2
        [*]-->Idle
        Idle --> Running : start
        Running --> [*]
        """
    )
    sequence = parse_mermaid(
        """sequenceDiagram
        actor U as User
        U->>S: request
        S-->>U: response
        """
    )

    assert [node.shape for node in state.nodes].count(NodeShape.PSEUDO_START) == 1
    assert [node.shape for node in state.nodes].count(NodeShape.PSEUDO_END) == 1
    assert [node.node_id for node in sequence.nodes] == ["U", "S"]
    assert [edge.style for edge in sequence.edges] == [EdgeStyle.SOLID, EdgeStyle.DOTTED]
    assert not state.diagnostics
    assert not sequence.diagnostics


def test_unicode_identifiers_are_accepted_across_supported_diagrams() -> None:
    state = parse_mermaid("stateDiagram-v2\n[*] --> 待机\n待机 --> 运行中 : 触发事件")
    flow = parse_mermaid("flowchart LR\n开始 --> 完成")
    class_diagram = parse_mermaid("classDiagram\n用户 --> 订单")
    sequence = parse_mermaid("sequenceDiagram\n用户 ->> 服务 : 请求")

    assert [(edge.source, edge.target) for edge in state.edges] == [
        (state.nodes[0].node_id, "待机"),
        ("待机", "运行中"),
    ]
    assert [(edge.source, edge.target) for edge in flow.edges] == [("开始", "完成")]
    assert [(edge.source, edge.target) for edge in class_diagram.edges] == [("用户", "订单")]
    assert [(edge.source, edge.target) for edge in sequence.edges] == [("用户", "服务")]
    assert not state.diagnostics
    assert not flow.diagnostics
    assert not class_diagram.diagnostics
    assert not sequence.diagnostics


def test_state_pseudo_ids_cannot_collide_with_user_identifiers() -> None:
    start = parse_mermaid(
        """stateDiagram-v2
        state __start_1 : User start
        __start_1 --> Ready
        [*] --> Idle
        """
    )
    end = parse_mermaid(
        """stateDiagram-v2
        state __end_1 : User end
        Ready --> __end_1
        Completed --> [*]
        """
    )

    user_start = next(node for node in start.nodes if node.node_id == "__start_1")
    pseudo_start = next(node for node in start.nodes if node.shape is NodeShape.PSEUDO_START)
    user_end = next(node for node in end.nodes if node.node_id == "__end_1")
    pseudo_end = next(node for node in end.nodes if node.shape is NodeShape.PSEUDO_END)
    assert (user_start.label, user_start.shape) == ("User start", NodeShape.RECTANGLE)
    assert (user_end.label, user_end.shape) == ("User end", NodeShape.RECTANGLE)
    assert pseudo_start.node_id.startswith("@state-start:")
    assert pseudo_end.node_id.startswith("@state-end:")
    assert pseudo_start.node_id != user_start.node_id
    assert pseudo_end.node_id != user_end.node_id
    assert not start.diagnostics
    assert not end.diagnostics


def test_state_choice_and_inline_comments_preserve_transitions() -> None:
    diagram = parse_mermaid(
        """stateDiagram-v2
        state decision <<choice>>
        Ready --> decision %% choose a branch
        decision --> Done : "100%% complete"
        """
    )

    decision = next(node for node in diagram.nodes if node.node_id == "decision")
    assert decision.shape is NodeShape.DECISION
    assert [(edge.source, edge.target, edge.label) for edge in diagram.edges] == [
        ("Ready", "decision", ""),
        ("decision", "Done", "100%% complete"),
    ]
    assert not diagram.diagnostics


def test_state_fork_join_and_notes_are_retained() -> None:
    diagram = parse_mermaid(
        """stateDiagram-v2
        state fork_state <<fork>>
        state join_state <<join>>
        A --> fork_state
        fork_state --> B
        fork_state --> C
        note right of B : worker #38; one
        note left of C
          worker two
        end note
        B --> join_state
        C --> join_state
        """
    )

    fork = next(node for node in diagram.nodes if node.node_id == "fork_state")
    join = next(node for node in diagram.nodes if node.node_id == "join_state")
    assert (fork.shape, join.shape) == (NodeShape.FORK_JOIN, NodeShape.FORK_JOIN)
    assert next(node for node in diagram.nodes if node.node_id == "B").notes == ("worker & one",)
    assert next(node for node in diagram.nodes if node.node_id == "C").notes == ("worker two",)
    assert not diagram.diagnostics


def test_state_style_statements_are_warnings_that_preserve_structure() -> None:
    diagram = parse_mermaid(
        """stateDiagram-v2
        A --> B
        classDef hot fill:#f00
        class A,B hot
        style A fill:#fff
        """
    )

    assert [(edge.source, edge.target) for edge in diagram.edges] == [("A", "B")]
    assert diagram.diagnostics
    assert {diagnostic.severity for diagnostic in diagram.diagnostics} == {DiagnosticSeverity.WARNING}


def test_overlong_decimal_entities_are_safely_replaced() -> None:
    digits = "9" * 5_000
    diagram = parse_mermaid(
        f"""flowchart LR
        A["before &#{digits}; after"] --> B["before &#{digits} after"]
        """
    )

    assert next(node for node in diagram.nodes if node.node_id == "A").label == "before   after"
    assert next(node for node in diagram.nodes if node.node_id == "B").label == "before   after"
    assert not diagram.diagnostics


def test_class_and_state_accept_direction_lr() -> None:
    class_diagram = parse_mermaid("classDiagram\ndirection LR\nA --> B")
    state_diagram = parse_mermaid("stateDiagram-v2\ndirection LR\nA --> B")

    assert class_diagram.direction is Direction.LEFT_RIGHT
    assert state_diagram.direction is Direction.LEFT_RIGHT
    assert not class_diagram.diagnostics
    assert not state_diagram.diagnostics


def test_sequence_autonumber_prefixes_messages_in_source_order() -> None:
    diagram = parse_mermaid(
        """sequenceDiagram
        autonumber
        A->>B: request
        B-->>A: response
        """
    )

    assert [edge.label for edge in diagram.edges] == ["1. request", "2. response"]
    assert not diagram.diagnostics


def test_sequence_autonumber_accepts_decimal_start_and_increment() -> None:
    diagram = parse_mermaid(
        """sequenceDiagram
        autonumber 10.5 0.25
        A->>B: first
        B-->>A: second
        A->>B: third
        """
    )

    assert [edge.label for edge in diagram.edges] == ["10.5. first", "10.75. second", "11. third"]
    assert not diagram.diagnostics


def test_sequence_flattens_fragments_and_accepts_activation_markers() -> None:
    diagram = parse_mermaid(
        """sequenceDiagram
        A->>+B: request
        activate B
        alt success
          B-->>-A: response
        else failure
          B--xA: error
        end
        deactivate B
        """
    )

    assert [(edge.source, edge.target, edge.label) for edge in diagram.edges] == [
        ("A", "B", "request"),
        ("B", "A", "response"),
        ("B", "A", "error"),
    ]
    assert diagram.edges[-1].target_marker == "x"
    assert diagram.diagnostics
    assert {diagnostic.severity for diagnostic in diagram.diagnostics} == {DiagnosticSeverity.WARNING}


def test_sequence_participant_types_and_lifecycle_directives_degrade_safely() -> None:
    diagram = parse_mermaid(
        """sequenceDiagram
        participant Boundary@{ "type": "boundary", "alias": "API Boundary" }
        participant DB@{ "type": "database" } as Storage
        participant Items@{ "type": "collections" }
        create actor Worker as Background Worker
        Boundary->>DB: fetch
        DB-->>Items: rows
        destroy Worker
        """
    )

    boundary, database, collections, worker = diagram.nodes
    assert (boundary.label, boundary.annotation) == ("API Boundary", "boundary")
    assert (database.label, database.shape, database.annotation) == ("Storage", NodeShape.CYLINDER, "database")
    assert (collections.shape, collections.annotation) == (NodeShape.SUBROUTINE, "collections")
    assert (worker.label, worker.shape) == ("Background Worker", NodeShape.ACTOR)
    assert [(edge.source, edge.target) for edge in diagram.edges] == [("Boundary", "DB"), ("DB", "Items")]
    assert diagram.diagnostics
    assert {diagnostic.severity for diagnostic in diagram.diagnostics} == {DiagnosticSeverity.WARNING}


def test_sequence_notes_and_actor_links_degrade_without_executing_links() -> None:
    diagram = parse_mermaid(
        """sequenceDiagram
        participant A
        participant B
        Note right of A: owner
        Note over A,B: shared
        link A: docs @ https://example.com
        links B: {"repo": "https://example.com/repo"}
        A->>B: request
        """
    )

    first, second = diagram.nodes
    assert first.notes == ("owner", "shared")
    assert second.notes == ("shared",)
    assert [(edge.source, edge.target, edge.label) for edge in diagram.edges] == [("A", "B", "request")]
    assert diagram.diagnostics
    assert {diagnostic.severity for diagnostic in diagram.diagnostics} == {DiagnosticSeverity.WARNING}


def test_pie_chart_parses_title_show_data_and_positive_slices() -> None:
    diagram = parse_mermaid(
        """pie showData
        title 浏览器市场份额
        "Chrome" : 65
        "Safari" : 15.5
        """
    )

    assert diagram.kind is DiagramKind.PIE
    assert isinstance(diagram.chart, PieChart)
    assert (diagram.chart.title, diagram.chart.show_data) == ("浏览器市场份额", True)
    assert [(item.label, item.value) for item in diagram.chart.slices] == [
        ("Chrome", Decimal(65)),
        ("Safari", Decimal("15.5")),
    ]
    assert not diagram.diagnostics


@pytest.mark.parametrize(
    ("header", "show_data"),
    [
        ("pie title 浏览器市场份额", False),
        ("pie showData title 浏览器市场份额", True),
    ],
)
def test_pie_chart_accepts_title_on_the_header_line(header: str, show_data: bool) -> None:
    diagram = parse_mermaid(f'{header}\n"Chrome" : 65\n"Safari" : 35')

    assert isinstance(diagram.chart, PieChart)
    assert diagram.chart.title == "浏览器市场份额"
    assert diagram.chart.show_data is show_data
    assert not diagram.diagnostics


def test_xychart_parses_axes_bar_line_and_horizontal_alias() -> None:
    diagram = parse_mermaid(
        """xychart-beta horizontal
        title "Quarterly performance"
        x-axis "Quarter" [Q1, Q2, Q3, Q4]
        y-axis "Revenue" -10 --> 100
        bar [25, 45, 72, 90]
        line [20, 50, 65, 95]
        """
    )

    assert diagram.kind is DiagramKind.XYCHART
    assert isinstance(diagram.chart, XYChart)
    assert diagram.chart.horizontal
    assert diagram.chart.x_axis_title == "Quarter"
    assert diagram.chart.x_labels == ("Q1", "Q2", "Q3", "Q4")
    assert (diagram.chart.y_axis_title, diagram.chart.y_min, diagram.chart.y_max) == (
        "Revenue",
        Decimal(-10),
        Decimal(100),
    )
    assert [series.kind.value for series in diagram.chart.series] == ["bar", "line"]
    assert not diagram.diagnostics


def test_xychart_parses_named_series_and_title_only_axes() -> None:
    diagram = parse_mermaid(
        """xychart
        x-axis Month
        y-axis Revenue
        bar "Bookings" [25, 45]
        line average [20, 50]
        """
    )

    assert isinstance(diagram.chart, XYChart)
    assert diagram.chart.x_axis_title == "Month"
    assert diagram.chart.x_labels == ()
    assert diagram.chart.y_axis_title == "Revenue"
    assert (diagram.chart.y_min, diagram.chart.y_max) == (None, None)
    assert [series.name for series in diagram.chart.series] == ["Bookings", "average"]
    assert not diagram.diagnostics


@pytest.mark.parametrize("orientation", ["", " horizontal"])
@pytest.mark.parametrize("kind", ["bar", "line"])
@pytest.mark.parametrize("values", ["20, 30", "-10, -5", "-10, 20", "5, 20", "20"])
@pytest.mark.parametrize("axis_first", [False, True])
def test_xychart_out_of_range_values_fail_closed(orientation: str, kind: str, values: str, axis_first: bool) -> None:
    axis, plot = "y-axis 0 --> 10", f"{kind} [{values}]"
    body = f"{axis}\n{plot}" if axis_first else f"{plot}\n{axis}"
    diagram = parse_mermaid(f"xychart{orientation}\n{body}")
    assert diagram.has_fatal_error
    assert any(
        diagnostic.code is DiagnosticCode.UNSUPPORTED_CHART_STATEMENT and diagnostic.line == (3 if axis_first else 2)
        for diagnostic in diagram.diagnostics
    )


@pytest.mark.parametrize("orientation", ["", " horizontal"])
@pytest.mark.parametrize("axis", ["y-axis 0 --> 10\n", ""])
def test_xychart_boundary_samples_and_automatic_ranges_remain_supported(orientation: str, axis: str) -> None:
    values = "0, 10" if axis else "-10, 20"
    diagram = parse_mermaid(f"xychart{orientation}\n{axis}line [{values}]\nbar [{values}]")
    assert not diagram.diagnostics
    assert isinstance(diagram.chart, XYChart)
    expected = tuple(Decimal(value) for value in values.split(", "))
    assert all(series.values == expected for series in diagram.chart.series)


@pytest.mark.parametrize("kind", ("bar", "line"))
@pytest.mark.parametrize("name", ("Latency [ms]", "吞吐量 [请求/秒]", "Range [0, 100] --> output"))
def test_xychart_parses_series_names_containing_brackets(kind: str, name: str) -> None:
    diagram = parse_mermaid(f'xychart\n{kind} "{name}" [10, 20]')

    assert not diagram.diagnostics
    assert isinstance(diagram.chart, XYChart)
    series = diagram.chart.series[0]
    assert series.name == name
    assert series.kind.value == kind
    assert series.values == (Decimal(10), Decimal(20))


@pytest.mark.parametrize("body", ('line "Latency [ms] [10, 20]', 'line "Latency [ms]"', 'bar "Size [KB]" [10, broken]'))
def test_xychart_rejects_malformed_quoted_series(body: str) -> None:
    diagram = parse_mermaid(f"xychart\n{body}")

    assert diagram.has_fatal_error
    assert diagram.diagnostics[0].code is DiagnosticCode.UNSUPPORTED_CHART_STATEMENT


@pytest.mark.parametrize("title", ("Time [s]", "Flow --> output", "Window [0 --> 10]"))
@pytest.mark.parametrize("axis_data", ("", " 0 --> 10", ' ["A [s]", "B --> C"]'))
def test_xychart_ignores_delimiters_inside_quoted_axis_titles(title: str, axis_data: str) -> None:
    diagram = parse_mermaid(f'xychart\nx-axis "{title}"{axis_data}\ny-axis "Value [units]" 0 --> 10\nbar [1, 2]')

    assert not diagram.diagnostics
    assert isinstance(diagram.chart, XYChart)
    assert diagram.chart.x_axis_title == title
    assert diagram.chart.y_axis_title == "Value [units]"
    if axis_data.startswith(" ["):
        assert diagram.chart.x_labels == ("A [s]", "B --> C")
    else:
        assert diagram.chart.x_labels == ()
    expected_range = (Decimal(0), Decimal(10)) if axis_data == " 0 --> 10" else (None, None)
    assert (diagram.chart.x_min, diagram.chart.x_max) == expected_range


def test_xychart_accepts_quoted_title_only_y_axis_with_delimiters() -> None:
    diagram = parse_mermaid('xychart\ny-axis "Flow [units] --> output"\nbar [1, 2]')

    assert not diagram.diagnostics
    assert isinstance(diagram.chart, XYChart)
    assert diagram.chart.y_axis_title == "Flow [units] --> output"
    assert (diagram.chart.y_min, diagram.chart.y_max) == (None, None)


def test_quadrant_chart_parses_axes_labels_points_and_ignores_styles() -> None:
    diagram = parse_mermaid(
        """quadrantChart
        title Reach and engagement
        x-axis Low Reach --> High Reach
        y-axis Low Engagement --> High Engagement
        quadrant-1 Expand
        quadrant-3 Re-evaluate
        Campaign A: [0.3, 0.6]
        Campaign B:::important: [0.78, 0.34] radius: 12
        """
    )

    assert diagram.kind is DiagramKind.QUADRANT
    assert isinstance(diagram.chart, QuadrantChart)
    assert diagram.chart.x_axis == ("Low Reach", "High Reach")
    assert diagram.chart.y_axis == ("Low Engagement", "High Engagement")
    assert diagram.chart.quadrants == ("Expand", "", "Re-evaluate", "")
    assert [(point.label, point.x, point.y) for point in diagram.chart.points] == [
        ("Campaign A", Decimal("0.3"), Decimal("0.6")),
        ("Campaign B", Decimal("0.78"), Decimal("0.34")),
    ]
    assert {diagnostic.severity for diagnostic in diagram.diagnostics} == {DiagnosticSeverity.WARNING}


@pytest.mark.parametrize("suffix", ["", ":::important", " radius: 12"])
def test_quadrant_coordinates_ignore_brackets_inside_quoted_labels(suffix: str) -> None:
    label = "Baseline: [0.1, 0.2]"
    style_class = suffix if suffix.startswith(":::") else ""
    style = suffix if not style_class else ""
    source = f'quadrantChart\n"{label}"{style_class}: [0.8, 0.9]{style}'
    diagram = parse_mermaid(source)

    assert not diagram.has_fatal_error
    assert isinstance(diagram.chart, QuadrantChart)
    assert [(point.label, point.x, point.y) for point in diagram.chart.points] == [
        (label, Decimal("0.8"), Decimal("0.9"))
    ]
    assert bool(diagram.diagnostics) == bool(suffix)
    assert f"{label} [0.8, 0.9]" in "\n".join(compile_mermaid(source).rows)


@pytest.mark.parametrize("point", ['"Unclosed: [0.1, 0.2]', ": [0.1, 0.2]", '"Point" [0.1, 0.2]'])
def test_quadrant_malformed_point_delimiters_fail_closed(point: str) -> None:
    assert parse_mermaid(f"quadrantChart\n{point}").has_fatal_error


@pytest.mark.parametrize("label", ["Customer's app", "用户的 app's baseline", "'90s baseline", "'90s baseline'"])
def test_quadrant_plain_label_apostrophes_do_not_start_quoted_strings(label: str) -> None:
    diagram = parse_mermaid(f"quadrantChart\n{label}: [0.2, 0.5]")
    assert not diagram.diagnostics
    assert isinstance(diagram.chart, QuadrantChart)
    assert diagram.chart.points[0].label == label
    assert (diagram.chart.points[0].x, diagram.chart.points[0].y) == (Decimal("0.2"), Decimal("0.5"))


@pytest.mark.parametrize("suffix", ["", ":::important"])
@pytest.mark.parametrize("label", ["Baseline:::foo", "Baseline:::foo: [0.1, 0.2]", "'90s:::baseline"])
def test_quadrant_only_strips_style_suffixes_outside_double_quotes(label: str, suffix: str) -> None:
    source = f'quadrantChart\n"{label}"{suffix}: [0.8, 0.9]'
    diagram = parse_mermaid(source)
    assert not diagram.has_fatal_error
    assert isinstance(diagram.chart, QuadrantChart)
    assert diagram.chart.points[0].label == label
    assert (diagram.chart.points[0].x, diagram.chart.points[0].y) == (Decimal("0.8"), Decimal("0.9"))
    assert bool(diagram.diagnostics) == bool(suffix)
    assert all(item.severity is DiagnosticSeverity.WARNING for item in diagram.diagnostics)
    assert f"● {label} [0.8, 0.9]" in "\n".join(compile_mermaid(source).rows)


@pytest.mark.parametrize("axis", ["x-axis", "y-axis"])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"flow --> output" --> High', ("flow --> output", "High")),
        ('Low --> "flow --> output"', ("Low", "flow --> output")),
        ('"input --> flow" --> "flow --> output"', ("input --> flow", "flow --> output")),
        ('"flow --> output"', ("flow --> output", "")),
        ("'90s baseline --> Customer's app", ("'90s baseline", "Customer's app")),
        ("'90s baseline' --> High", ("'90s baseline'", "High")),
    ],
)
def test_quadrant_axis_arrows_only_split_outside_double_quotes(axis: str, raw: str, expected: tuple[str, str]) -> None:
    source = f"quadrantChart\n{axis} {raw}\nP: [0.5, 0.5]"
    diagram = parse_mermaid(source)
    assert not diagram.diagnostics
    assert isinstance(diagram.chart, QuadrantChart)
    assert (diagram.chart.x_axis if axis == "x-axis" else diagram.chart.y_axis) == expected
    rendered = "\n".join(compile_mermaid(source).rows)
    assert all(label in rendered for label in expected if label)


@pytest.mark.parametrize("axis", ["x-axis", "y-axis"])
@pytest.mark.parametrize("raw", ['"Unclosed --> High', 'Low --> "Unclosed', "Low --> Medium --> High"])
def test_quadrant_malformed_axis_delimiters_fail_closed(axis: str, raw: str) -> None:
    assert parse_mermaid(f"quadrantChart\n{axis} {raw}\nP: [0.5, 0.5]").has_fatal_error


def test_treemap_parses_indented_hierarchy_and_computes_branch_values() -> None:
    diagram = parse_mermaid(
        """treemap-beta
        "Products"
            "Desktop": 40
            "Mobile": 35
        "Services"
            "Cloud": 20
            "Support": 5
        """
    )

    assert diagram.kind is DiagramKind.TREEMAP
    assert isinstance(diagram.chart, TreemapChart)
    products, services = diagram.chart.items
    assert (products.label, products.value) == ("Products", Decimal(75))
    assert [(item.label, item.value) for item in products.children] == [
        ("Desktop", Decimal(40)),
        ("Mobile", Decimal(35)),
    ]
    assert (services.label, services.value) == ("Services", Decimal(25))
    assert not diagram.diagnostics


def test_treemap_ignores_class_suffix_after_leaf_value() -> None:
    diagram = parse_mermaid(
        """treemap-beta
        "Products":::section
            "Desktop": 40:::highlight
        """
    )

    assert isinstance(diagram.chart, TreemapChart)
    assert diagram.chart.items[0].children[0].value == Decimal(40)
    assert {diagnostic.severity for diagnostic in diagram.diagnostics} == {DiagnosticSeverity.WARNING}


@pytest.mark.parametrize(
    "source",
    [
        'pie\n"bad": 0',
        "xychart\nx-axis [A, B]\nbar [1]",
        "xychart\nbarometer [1]",
        "quadrantChart",
        "quadrantChart\nOutside: [1.1, 0.5]",
        'treemap-beta\n"Leaf": -1',
    ],
)
def test_statistical_charts_fail_closed_for_invalid_data(source: str) -> None:
    diagram = parse_mermaid(source)

    assert diagram.has_fatal_error
    assert diagram.diagnostics


def test_expected_input_failures_are_bounded_diagnostics() -> None:
    unsupported = parse_mermaid("flowchart TB\n" + "\n".join(f"subgraph bad{index}" for index in range(100)))
    oversized = parse_mermaid("x" * (MAX_SOURCE_BYTES + 1))

    assert len(unsupported.diagnostics) == MAX_DIAGNOSTICS
    assert oversized.kind is DiagramKind.UNKNOWN
    assert oversized.diagnostics[0].line == 1
    assert oversized.diagnostics[0].code is DiagnosticCode.SOURCE_LIMIT


def test_fatal_node_limit_survives_warning_cap_without_dangling_edges() -> None:
    warnings = "\n".join(f"classDef warning{index} fill:#fff" for index in range(MAX_DIAGNOSTICS))
    chain = " --> ".join(f"N{index}" for index in range(MAX_NODES + 1))
    source = f"flowchart LR\n{warnings}\n{chain}"
    diagram = parse_mermaid(source)
    compiled = compile_mermaid(source)

    node_ids = {node.node_id for node in diagram.nodes}
    assert len(diagram.diagnostics) == MAX_DIAGNOSTICS
    assert diagram.has_fatal_error
    assert any(diagnostic.code is DiagnosticCode.NODE_LIMIT for diagnostic in diagram.diagnostics)
    assert all(edge.source in node_ids and edge.target in node_ids for edge in diagram.edges)
    assert compiled.diagnostics


@pytest.mark.parametrize("limit", ["nodes", "edges", "syntax"])
def test_flow_parser_stops_work_after_first_fatal_error(limit: str) -> None:
    if limit == "nodes":
        chain = " --> ".join(f"N{index}" for index in range(5000))
        expected_calls = MAX_NODES + 1
        code = DiagnosticCode.NODE_LIMIT
    elif limit == "edges":
        chain = " --> ".join("A" for _ in range(8000))
        expected_calls = MAX_EDGES + 2
        code = DiagnosticCode.EDGE_LIMIT
    else:
        chain = "A --> ?"
        expected_calls = 2
        code = DiagnosticCode.MISSING_EDGE_TARGET
    source = f"flowchart LR\n{chain}; Later --> Ignored\nAlso --> Ignored"
    assert len(source.encode()) < MAX_SOURCE_BYTES
    with patch.object(graphs, "_parse_node_ref", wraps=graphs._parse_node_ref) as parse_node:
        diagram = parse_mermaid(source)

    assert diagram.has_fatal_error
    assert diagram.diagnostics[-1].code is code
    assert parse_node.call_count == expected_calls
    assert all(node.node_id not in {"Later", "Also"} for node in diagram.nodes)


def test_unknown_diagram_kind_has_source_positioned_diagnostic() -> None:
    diagram = parse_mermaid("%% comment\nunknownDiagram\n  root((No browser here))")

    assert diagram.kind is DiagramKind.UNKNOWN
    assert diagram.diagnostics[0].line == 2


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("BT", Direction.BOTTOM_UP), ("RL", Direction.RIGHT_LEFT)],
)
def test_reverse_flow_directions_are_preserved(direction: str, expected: Direction) -> None:
    diagram = parse_mermaid(f"flowchart {direction}\nA --> B")

    assert diagram.kind is DiagramKind.FLOWCHART
    assert diagram.direction is expected
    assert [(edge.source, edge.target) for edge in diagram.edges] == [("A", "B")]
    assert not diagram.diagnostics


@pytest.mark.parametrize(
    "source",
    (
        "flowchart TB\ndirection RL\nA --> B",
        "classDiagram\ndirection RL\nA --> B",
        "stateDiagram-v2\ndirection RL\nA --> B",
    ),
)
def test_graph_direction_statements_accept_reverse_axes(source: str) -> None:
    diagram = parse_mermaid(source)

    assert diagram.direction is Direction.RIGHT_LEFT
    assert not diagram.diagnostics
