# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared graph IR, geometry and compiler results."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from .specs.charts import ChartSeriesKind as ChartSeriesKind
from .specs.charts import PieChart as PieChart
from .specs.charts import PieSlice as PieSlice
from .specs.charts import QuadrantChart as QuadrantChart
from .specs.charts import QuadrantPoint as QuadrantPoint
from .specs.charts import TreemapChart as TreemapChart
from .specs.charts import TreemapItem as TreemapItem
from .specs.charts import XYChart as XYChart
from .specs.charts import XYSeries as XYSeries
from .specs.planning import JourneyChart, KanbanChart, TimelineChart
from .specs.schedule import GanttChart, GitChart, PacketChart
from .specs.structure import BlockChart


class DiagramKind(StrEnum):
    """Diagram syntaxes supported by the terminal renderer."""

    FLOWCHART = "flowchart"
    CLASS = "class"
    ER = "er"
    STATE = "state"
    SEQUENCE = "sequence"
    PIE = "pie"
    XYCHART = "xychart"
    QUADRANT = "quadrant"
    TREEMAP = "treemap"
    JOURNEY = "journey"
    TIMELINE = "timeline"
    KANBAN = "kanban"
    MINDMAP = "mindmap"
    GANTT = "gantt"
    GIT = "git"
    PACKET = "packet"
    REQUIREMENT = "requirement"
    C4 = "c4"
    ARCHITECTURE = "architecture"
    BLOCK = "block"
    SANKEY = "sankey"
    UNKNOWN = "unknown"


class Direction(StrEnum):
    """Primary layout direction for a graph diagram."""

    TOP_DOWN = "TB"
    BOTTOM_UP = "BT"
    LEFT_RIGHT = "LR"
    RIGHT_LEFT = "RL"


class NodeShape(StrEnum):
    """Terminal node shapes."""

    RECTANGLE = "rectangle"
    ENTITY = "entity"
    ROUNDED = "rounded"
    STADIUM = "stadium"
    SUBROUTINE = "subroutine"
    CYLINDER = "cylinder"
    CIRCLE = "circle"
    SLANTED = "slanted"
    DECISION = "decision"
    HEXAGON = "hexagon"
    FORK_JOIN = "fork_join"
    PSEUDO_START = "pseudo_start"
    PSEUDO_END = "pseudo_end"
    ACTOR = "actor"


class EdgeStyle(StrEnum):
    """Line styles retained from Mermaid source."""

    SOLID = "solid"
    DOTTED = "dotted"
    HEAVY = "heavy"


class DiagnosticSeverity(StrEnum):
    """Severity of a parser or layout diagnostic."""

    ERROR = "error"
    WARNING = "warning"


class DiagnosticCode(StrEnum):
    """Locale-neutral reasons a Mermaid diagram could not be compiled."""

    NODE_LIMIT = "node_limit"
    NODE_REDECLARED = "node_redeclared"
    EDGE_LIMIT = "edge_limit"
    UNSUPPORTED_FLOW_STATEMENT = "unsupported_flow_statement"
    EXPECTED_NODE_OR_EDGE = "expected_node_or_edge"
    MALFORMED_FLOW_EDGE = "malformed_flow_edge"
    MISSING_EDGE_TARGET = "missing_edge_target"
    UNSUPPORTED_DIRECTIVE = "unsupported_directive"
    UNEXPECTED_CLASS_TERMINATOR = "unexpected_class_terminator"
    NESTED_CLASS_BODY = "nested_class_body"
    UNSUPPORTED_CLASS_STATEMENT = "unsupported_class_statement"
    UNCLOSED_CLASS_BODY = "unclosed_class_body"
    UNEXPECTED_ER_TERMINATOR = "unexpected_er_terminator"
    NESTED_ER_BODY = "nested_er_body"
    UNSUPPORTED_ER_STATEMENT = "unsupported_er_statement"
    UNCLOSED_ER_BODY = "unclosed_er_body"
    NESTED_STATE = "nested_state"
    UNSUPPORTED_STATE_STATEMENT = "unsupported_state_statement"
    UNSUPPORTED_SEQUENCE_FRAGMENT = "unsupported_sequence_fragment"
    UNSUPPORTED_SEQUENCE_STATEMENT = "unsupported_sequence_statement"
    UNSUPPORTED_CHART_STATEMENT = "unsupported_chart_statement"
    SOURCE_LIMIT = "source_limit"
    EMPTY_SOURCE = "empty_source"
    UNSUPPORTED_DIAGRAM_TYPE = "unsupported_diagram_type"
    UNSUPPORTED_FLOW_DIRECTION = "unsupported_flow_direction"
    CANVAS_LIMIT = "canvas_limit"
    NO_NODES = "no_nodes"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A locale-neutral, source-positioned parser or layout diagnostic."""

    line: int
    code: DiagnosticCode
    parameters: tuple[tuple[str, str | int], ...] = ()
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR


@dataclass(frozen=True, slots=True)
class DiagramNode:
    """A graph node or sequence participant; min_width reserves terminal cells for overlays."""

    node_id: str
    label: str
    shape: NodeShape = NodeShape.RECTANGLE
    sections: tuple[tuple[str, ...], ...] = ()
    line: int = 0
    annotation: str = ""
    notes: tuple[str, ...] = ()
    min_width: int = 0


@dataclass(frozen=True, slots=True)
class DiagramEdge:
    """A graph edge or time-ordered sequence message."""

    source: str
    target: str
    label: str = ""
    style: EdgeStyle = EdgeStyle.SOLID
    directed: bool = True
    source_marker: str = ""
    target_marker: str = ""
    source_label: str = ""
    target_label: str = ""
    line: int = 0
    # Feedback edges remain visible without constraining forward layout ranks.
    constrains_rank: bool = True


ChartData = (
    PieChart
    | XYChart
    | QuadrantChart
    | TreemapChart
    | JourneyChart
    | TimelineChart
    | KanbanChart
    | GanttChart
    | GitChart
    | PacketChart
    | BlockChart
)


@dataclass(frozen=True, slots=True)
class DiagramIR:
    """Immutable parsed diagram before terminal layout."""

    kind: DiagramKind
    direction: Direction
    nodes: tuple[DiagramNode, ...]
    edges: tuple[DiagramEdge, ...]
    diagnostics: tuple[Diagnostic, ...] = ()
    has_fatal_error: bool = False
    chart: ChartData | None = None
    title: str = ""
    simplified: bool = False


@dataclass(frozen=True, slots=True)
class PlacedNode:
    """A node with terminal-cell geometry."""

    node: DiagramNode
    x: int
    y: int
    width: int
    height: int
    content_lines: tuple[str, ...]
    section_breaks: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Point:
    """A terminal-cell coordinate."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class RoutedEdge:
    """A graph edge with an orthogonal terminal path."""

    edge: DiagramEdge
    points: tuple[Point, ...]
    label_at: Point | None
    arrow_at: Point | None
    arrow: str = ""
    source_marker_at: Point | None = None
    target_marker_at: Point | None = None
    source_label_at: Point | None = None
    target_label_at: Point | None = None

    def translated(self, dx: int, dy: int) -> RoutedEdge:
        """Move the route and its annotations together with the rendered canvas."""

        def move(point: Point | None) -> Point | None:
            return Point(point.x + dx, point.y + dy) if point is not None else None

        return replace(
            self,
            points=tuple(Point(point.x + dx, point.y + dy) for point in self.points),
            label_at=move(self.label_at),
            arrow_at=move(self.arrow_at),
            source_marker_at=move(self.source_marker_at),
            target_marker_at=move(self.target_marker_at),
            source_label_at=move(self.source_label_at),
            target_label_at=move(self.target_label_at),
        )


@dataclass(frozen=True, slots=True)
class CompiledDiagram:
    """A safe, immutable terminal rendering of Mermaid source.

    ``rows`` omit insignificant trailing spaces. ``crop_plain_row`` returns
    exactly the requested number of terminal cells and safely handles a crop
    boundary that lands inside a wide grapheme.
    """

    source: str
    kind: DiagramKind
    width: int
    height: int
    rows: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...] = ()
    routed_edges: tuple[RoutedEdge, ...] = ()

    def crop_plain_row(self, y: int, x: int, width: int) -> str:
        """Return a padded cell crop from row *y*."""
        from .canvas import crop_cell_text

        if width <= 0:
            return ""
        row = self.rows[y] if 0 <= y < len(self.rows) else ""
        return crop_cell_text(row, max(0, x), width)
