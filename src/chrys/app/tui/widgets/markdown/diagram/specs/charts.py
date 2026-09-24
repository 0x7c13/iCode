# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Immutable data for statistical chart families."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class ChartSeriesKind(StrEnum):
    """Plot styles supported by an XY chart."""

    BAR = "bar"
    LINE = "line"


@dataclass(frozen=True, slots=True)
class PieSlice:
    """One positive slice in a pie chart."""

    label: str
    value: Decimal


@dataclass(frozen=True, slots=True)
class PieChart:
    """Data retained from Mermaid pie syntax."""

    title: str
    show_data: bool
    slices: tuple[PieSlice, ...]


@dataclass(frozen=True, slots=True)
class XYSeries:
    """One bar or line series in an XY chart."""

    kind: ChartSeriesKind
    values: tuple[Decimal, ...]
    name: str = ""


@dataclass(frozen=True, slots=True)
class XYChart:
    """Axes and plots retained from Mermaid XY chart syntax."""

    title: str
    horizontal: bool
    x_axis_title: str
    x_labels: tuple[str, ...]
    x_min: Decimal | None
    x_max: Decimal | None
    y_axis_title: str
    y_min: Decimal | None
    y_max: Decimal | None
    series: tuple[XYSeries, ...]


@dataclass(frozen=True, slots=True)
class QuadrantPoint:
    """One normalized point in a quadrant chart."""

    label: str
    x: Decimal
    y: Decimal


@dataclass(frozen=True, slots=True)
class QuadrantChart:
    """Labels and points retained from Mermaid quadrant syntax."""

    title: str
    x_axis: tuple[str, str]
    y_axis: tuple[str, str]
    quadrants: tuple[str, str, str, str]
    points: tuple[QuadrantPoint, ...]


@dataclass(frozen=True, slots=True)
class TreemapItem:
    """One weighted item in a treemap hierarchy."""

    label: str
    value: Decimal
    children: tuple[TreemapItem, ...] = ()


@dataclass(frozen=True, slots=True)
class TreemapChart:
    """Top-level items retained from Mermaid treemap syntax."""

    items: tuple[TreemapItem, ...]
