# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The trajectory dashboard's Timeline page: one turn's operations on a time axis, or as a dependency graph."""

from __future__ import annotations

from collections import defaultdict

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.color import Color, ColorParseError

from chrys.app.tui.util.rich_style import rich_style_from_textual_color
from chrys.app.tui.widgets.trajectory.chartkit import (
    fit_cells,
    fit_text_cells,
    time_ruler,
    timeline_bar,
    unresolved_bar,
)
from chrys.app.tui.widgets.trajectory.presentation import (
    ELAPSED_SCOPE,
    PRECISION_SYMBOLS,
    TIME_RULER,
    TURN_LABEL,
    WIDE_MIN_COLUMNS,
    DashboardLook,
    RenderContext,
    fit_text_right,
    format_duration,
    identity_with_hook_id,
    metric_value,
    precision_badge,
    precision_label,
    precision_style,
    section_style,
)
from chrys.foundation.i18n import msg
from chrys.service.analytics import FLOW_TERMINAL_INDEX, Metric, Precision, TimelineOperation, TurnAnalysis

_GRAPH_LEGEND = msg(
    "tui.trajectory.graph.legend",
    fallback="│ parent · ⇠ causal · ┄ adjacent only (no proven dependency)",
)
_GRAPH_TITLE = msg("tui.trajectory.graph.title", fallback="Dependency graph · turn {turn}")
_GRAPH_RESPONSE = msg("tui.trajectory.graph.response", fallback="response")
_GRAPH_NONE = msg(
    "tui.trajectory.graph.none",
    fallback="No dependency graph is available for this turn.",
)
_GRAPH_CYCLE_WARNING = msg(
    "tui.trajectory.graph.cycle",
    fallback="The dependency graph contains a cycle; rows fall back to first-occurrence order.",
)
_GRAPH_HINT = msg("tui.trajectory.timeline.graph_hint", fallback="Space: dependency graph")
_TIMELINE_HINT = msg("tui.trajectory.graph.timeline_hint", fallback="Space: timeline")
_CATEGORY_MODEL = msg("tui.trajectory.category.model", fallback="Model")
_CATEGORY_TOOL = msg("tui.trajectory.category.tool", fallback="Tool")
_CATEGORY_WAIT = msg("tui.trajectory.category.wait", fallback="Wait")
_CATEGORY_HOOK = msg("tui.trajectory.category.hook", fallback="Hook")
_CATEGORY_AGENT = msg("tui.trajectory.category.agent", fallback="Agent")
_CATEGORY_PREPARATION = msg("tui.trajectory.category.preparation", fallback="Prepare")
_CATEGORY_COMPACTION = msg("tui.trajectory.category.compaction", fallback="Compact")
_CATEGORY_APPROVAL = msg("tui.trajectory.category.approval", fallback="Approval")
_CATEGORY_RETRY = msg("tui.trajectory.category.retry", fallback="Retry")
_CATEGORY_OPERATION = msg("tui.trajectory.category.operation", fallback="Operation")

_OPERATION_CATEGORY_WIDTH = 8
_OPERATION_CATEGORY_SEPARATOR = " "
_TIMELINE_LABEL_WIDTH = 27
_TIMELINE_WIDE_LABEL_WIDTH = 39


def timeline_lines(look: DashboardLook, context: RenderContext, turn: TurnAnalysis) -> list[Text]:
    title = look.message(TURN_LABEL.bind(turn=turn.turn_number or "—"))
    lines = [
        Text.assemble(
            Text(title, style=section_style(look)),
            Text("  "),
            _metric_text(look, turn.elapsed_ns),
            Text(f"  {look.message(_GRAPH_HINT.bind())}", style="dim"),
        )
    ]
    width = context.width
    category_width = _OPERATION_CATEGORY_WIDTH
    suffix_reserve = 20
    # Below the narrowest fitting layout the timeline draws on a fixed
    # 92-cell canvas and scrolls horizontally instead of squeezing its
    # columns into illegibility; the Timeline view owns a horizontal
    # scrollbar, unlike Overview.
    minimum_fit_width = category_width + _TIMELINE_LABEL_WIDTH + suffix_reserve + 4 + 12
    canvas_width = width if width >= minimum_fit_width else max(width, 92)
    label_width = _TIMELINE_WIDE_LABEL_WIDTH if canvas_width >= WIDE_MIN_COLUMNS else _TIMELINE_LABEL_WIDTH
    # The duration column hugs its widest value so the bars run right up
    # to the figures instead of leaving a gutter of reserved cells.
    resolved_suffixes = [
        format_duration(operation.duration_ns or 0)
        for operation in turn.operations
        if operation.start_ns is not None and operation.end_ns is not None
    ]
    unresolved_suffixes = [
        f"{PRECISION_SYMBOLS[operation.precision]} {precision_label(look, operation.precision)}"
        for operation in turn.operations
        if operation.start_ns is None or operation.end_ns is None
    ]
    suffix_width = min(
        suffix_reserve,
        max([8, *(cell_len(suffix) for suffix in (*resolved_suffixes, *unresolved_suffixes))]),
    )
    bar_width = max(12, canvas_width - category_width - label_width - suffix_width - 4)
    prefix_width = category_width + label_width + 3
    span = max(0, turn.axis_end_ns - turn.axis_start_ns)
    ruler = time_ruler(span, width=bar_width)
    ruler.stylize("dim")
    lines.append(
        Text.assemble(
            Text(fit_cells(look.message(TIME_RULER.bind()), prefix_width), style="dim"),
            ruler,
        )
    )
    for row_index, operation in enumerate(turn.operations):
        category = _operation_category(look, operation)
        identity = _operation_identity(look, operation)
        operation_style = _operation_style(look, operation.family)
        bar_style = _operation_bar_style(look, operation.family)
        indented = Text.assemble(
            Text("│ " * operation.depth, style="dim"),
            Text(identity, style="dim"),
        )
        if operation.start_ns is None or operation.end_ns is None:
            bar = unresolved_bar(bar_width, style=precision_style(look, operation.precision))
            suffix = Text.assemble(
                precision_badge(look, operation.precision),
                Text(
                    f" {precision_label(look, operation.precision)}",
                    style=precision_style(look, operation.precision),
                ),
            )
        else:
            bar = timeline_bar(
                operation.start_ns,
                operation.end_ns,
                origin=turn.axis_start_ns,
                span=span,
                width=bar_width,
                glyph=_operation_glyph(operation.family),
                style=bar_style,
            )
            suffix = Text(format_duration(operation.duration_ns or 0), style="dim")
        line = Text.assemble(
            fit_text_cells(Text(category, style=operation_style), category_width),
            Text(_OPERATION_CATEGORY_SEPARATOR),
            fit_text_cells(indented, label_width),
            Text(" "),
            bar,
            Text("  "),
            fit_text_right(suffix, suffix_width),
        )
        if row_index % 2:
            line.stylize(_zebra_style(look), 0, len(line))
        lines.append(line)
    lines.extend([Text(), Text(look.message(ELAPSED_SCOPE.bind()), style="dim")])
    return lines


def dependency_graph_lines(look: DashboardLook, turn: TurnAnalysis) -> list[Text]:
    lines = [
        Text.assemble(
            Text(
                look.message(_GRAPH_TITLE.bind(turn=turn.turn_number or "—")),
                style=section_style(look),
            ),
            Text(f"  {look.message(_TIMELINE_HINT.bind())}", style="dim"),
        ),
        Text(look.message(_GRAPH_LEGEND.bind()), style="dim"),
        Text(),
    ]
    flow = turn.flow
    if flow is None:
        lines.append(Text(look.message(_GRAPH_NONE.bind()), style="dim"))
        return lines
    if not flow.acyclic:
        lines.append(
            Text(
                look.message(_GRAPH_CYCLE_WARNING.bind()),
                style=precision_style(look, Precision.UNRESOLVED),
            )
        )
        lines.append(Text())
    operations = turn.operations
    children: dict[int, list[int]] = defaultdict(list)
    has_parent: set[int] = set()
    for source, target in flow.parent_edges():
        if FLOW_TERMINAL_INDEX in (source, target):
            continue
        children[source].append(target)
        has_parent.add(target)
    causal_in: dict[int, list[int]] = defaultdict(list)
    terminal_fan_in = 0
    for source, target in flow.causal_edges():
        if target == FLOW_TERMINAL_INDEX:
            terminal_fan_in += 1
        elif source != FLOW_TERMINAL_INDEX:
            causal_in[target].append(source)

    def order_key(index: int) -> tuple[bool, int, int]:
        operation = operations[index]
        return (operation.start_ns is None, operation.start_ns or 0, index)

    roots = sorted((index for index in range(len(operations)) if index not in has_parent), key=order_key)
    for child_list in children.values():
        child_list.sort(key=order_key)
    seen: set[int] = set()
    stack = [(index, 0) for index in reversed(roots)]
    while stack:
        index, depth = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        unproven = depth == 0 and index != flow.root_index and not causal_in.get(index)
        lines.append(
            _dependency_node_line(
                look,
                operations,
                index,
                depth=depth,
                causal_sources=causal_in.get(index, []),
                unproven=unproven,
            )
        )
        stack.extend((child, depth + 1) for child in reversed(children.get(index, [])))
        # A parent cycle leaves its members without a reachable root; once
        # the reachable forest drains, surface them flat rather than drop
        # them (the cycle warning above explains the shape).
        if not stack:
            leftovers = sorted((index for index in range(len(operations)) if index not in seen), key=order_key)
            stack = [(index, 0) for index in reversed(leftovers)]
    if flow.has_terminal:
        lines.append(
            Text.assemble(
                Text(" " * _OPERATION_CATEGORY_WIDTH),
                Text(_OPERATION_CATEGORY_SEPARATOR),
                Text("◆ ", style=section_style(look)),
                Text(look.message(_GRAPH_RESPONSE.bind()), style=section_style(look)),
                Text(f"  ⇠ {terminal_fan_in}", style="dim"),
            )
        )
    return lines


def _dependency_node_line(
    look: DashboardLook,
    operations: tuple[TimelineOperation, ...],
    index: int,
    *,
    depth: int,
    causal_sources: list[int],
    unproven: bool,
) -> Text:
    operation = operations[index]
    style = _operation_style(look, operation.family)
    parts = [
        fit_text_cells(Text(_family_category(look, operation.family), style=style), _OPERATION_CATEGORY_WIDTH),
        Text(_OPERATION_CATEGORY_SEPARATOR),
        Text("│ " * depth, style="dim"),
    ]
    if unproven:
        parts.append(Text("┄ ", style="dim"))
    parts.append(Text(_operation_identity(look, operation), style=style))
    duration = operation.duration_ns
    if duration is not None:
        parts.append(Text(f"  {format_duration(duration)}", style="dim"))
    parts.extend((Text(" "), precision_badge(look, operation.precision)))
    if causal_sources:
        labels = ", ".join(_operation_identity(look, operations[source]) for source in causal_sources[:2])
        extra = len(causal_sources) - 2
        suffix = f" +{extra}" if extra > 0 else ""
        parts.append(Text(f"  ⇠ {labels}{suffix}", style="dim"))
    return Text.assemble(*parts)


def _operation_category(look: DashboardLook, operation: TimelineOperation) -> str:
    return _family_category(look, operation.family)


def _family_category(look: DashboardLook, family: str) -> str:
    if family.startswith("workflow."):
        from chrys.app.tui.widgets.workflow.text import TITLE

        label = TITLE
    elif family.startswith("model."):
        label = _CATEGORY_MODEL
    elif family == "tool.operation":
        label = _CATEGORY_TOOL
    elif family in {"wait", "continuation.poll", "turn.suspension"}:
        label = _CATEGORY_WAIT
    elif family == "hook.operation":
        label = _CATEGORY_HOOK
    elif family == "sub_agent":
        label = _CATEGORY_AGENT
    elif family == "preparation":
        label = _CATEGORY_PREPARATION
    elif family.startswith("compaction"):
        label = _CATEGORY_COMPACTION
    elif family == "approval":
        label = _CATEGORY_APPROVAL
    elif family == "retry":
        label = _CATEGORY_RETRY
    else:
        label = _CATEGORY_OPERATION
    return look.message(label.bind())


def _operation_identity(look: DashboardLook, operation: TimelineOperation) -> str:
    if operation.identity is not None:
        return identity_with_hook_id(operation.identity, operation.hook_id)
    if operation.family == "tool.operation":
        return look.message(_CATEGORY_TOOL.bind())
    if operation.family == "hook.operation":
        return look.message(_CATEGORY_HOOK.bind())
    if operation.family in {"wait", "continuation.poll", "turn.suspension"}:
        return look.message(_CATEGORY_WAIT.bind())
    if operation.family == "sub_agent":
        return look.message(_CATEGORY_AGENT.bind())
    return operation.family.removeprefix("model.").removeprefix("compaction.")


def _metric_text(look: DashboardLook, metric: Metric) -> Text:
    return Text.assemble(Text(metric_value(metric)), Text(" "), precision_badge(look, metric.precision))


def _operation_style(look: DashboardLook, family: str) -> Style:
    if family.startswith(("model.", "compaction", "workflow.")):
        return look.semantic_style("primary", "blue", bold=True)
    if family == "tool.operation" or family == "preparation":
        return look.semantic_style("warning", "yellow", bold=True)
    if family == "hook.operation":
        return rich_style_from_textual_color("ansi_cyan", bold=True)
    if family in {"wait", "approval", "retry", "continuation.poll", "turn.suspension"}:
        return look.semantic_style("accent", "magenta", bold=True)
    if family == "sub_agent":
        return look.semantic_style("secondary", "cyan", bold=True)
    return look.semantic_style("foreground", "white")


def _operation_bar_style(look: DashboardLook, family: str) -> Style:
    if family == "model.run":
        return look.semantic_style("accent", "magenta", bold=True)
    return _operation_style(look, family)


def _zebra_style(look: DashboardLook) -> Style:
    return _blended_style(look, 0.08, fill_background=True) or Style()


def _blended_style(look: DashboardLook, factor: float, *, fill_background: bool) -> Style | None:
    """Blend the theme background toward the foreground; None when not blendable (ANSI)."""
    variables = look.theme_variables
    try:
        background = Color.parse(variables.get("background", ""))
        foreground = Color.parse(variables.get("foreground", ""))
    except ColorParseError:
        return None
    if background.ansi is not None or foreground.ansi is not None:
        return None
    blended = background.blend(foreground, factor)
    if fill_background:
        return Style(bgcolor=blended.rich_color)
    return Style(color=blended.rich_color)


def _operation_glyph(family: str) -> str:
    if family.startswith("model."):
        return "▮"
    if family in {"wait", "approval", "retry", "continuation.poll", "turn.suspension"}:
        return "▭"
    if family == "hook.operation":
        return "◆"
    return "▨"
