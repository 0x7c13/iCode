# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Typed chart compiler dispatch; layouts live in renderers/."""

from __future__ import annotations

from collections.abc import Callable

from .model import (
    CompiledDiagram,
    DiagramIR,
    PieChart,
    QuadrantChart,
    TreemapChart,
    XYChart,
)
from .renderers.charts import _compile_pie, _compile_quadrant, _compile_treemap
from .renderers.common import ChartCanvasLimit
from .renderers.planning import compile_planning
from .renderers.schedule import compile_schedule
from .renderers.structure import compile_block
from .renderers.xy import _compile_xy
from .specs.planning import JourneyChart, KanbanChart, TimelineChart
from .specs.schedule import GanttChart, GitChart, PacketChart
from .specs.structure import BlockChart


def compile_chart(
    source: str,
    ir: DiagramIR,
    exceeds_budget: Callable[[int, int], bool],
) -> CompiledDiagram:
    """Compile a parsed statistical chart onto a bounded terminal canvas."""
    chart = ir.chart
    if isinstance(chart, BlockChart):
        return compile_block(source, ir, exceeds_budget)
    if isinstance(chart, (JourneyChart, TimelineChart, KanbanChart)):
        return compile_planning(source, ir, exceeds_budget)
    if isinstance(chart, (GanttChart, GitChart, PacketChart)):
        return compile_schedule(source, ir, exceeds_budget)
    if isinstance(chart, PieChart):
        return _compile_pie(source, ir, chart, exceeds_budget)
    if isinstance(chart, XYChart):
        return _compile_xy(source, ir, chart, exceeds_budget)
    if isinstance(chart, QuadrantChart):
        return _compile_quadrant(source, ir, chart, exceeds_budget)
    if isinstance(chart, TreemapChart):
        return _compile_treemap(source, ir, chart, exceeds_budget)
    raise TypeError("chart diagram is missing chart data")


__all__ = ["ChartCanvasLimit", "compile_chart"]
