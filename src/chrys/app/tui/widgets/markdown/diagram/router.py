# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Orthogonal edge routing and terminal rendering for placed graph nodes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from itertools import pairwise
from typing import NamedTuple

from rich.cells import cell_len

from .canvas import TerminalCanvas, sanitize_terminal_text
from .model import DiagramEdge, Direction, EdgeStyle, NodeShape, PlacedNode, Point, RoutedEdge


class _Net(NamedTuple):
    """Adjacent-rank edges joined by shared endpoints, with the rows (columns) their trunk spans."""

    low: int
    high: int
    leaves: frozenset[int]
    enters: frozenset[int]
    edges: tuple[int, ...]


def detour_channels(
    edges: Iterable[DiagramEdge], ranks: dict[str, int]
) -> tuple[dict[int, tuple[int, int]], dict[int, int], dict[int, int]]:
    """Reserve distinct departure/arrival tracks for non-adjacent and repeated edges."""
    departures: defaultdict[int, int] = defaultdict(int)
    arrivals: defaultdict[int, int] = defaultdict(int)
    offsets: dict[int, tuple[int, int]] = {}
    seen: set[tuple[str, str]] = set()
    for index, edge in enumerate(edges):
        pair = (edge.source, edge.target)
        source_rank, target_rank = ranks[edge.source], ranks[edge.target]
        if edge.source != edge.target and (target_rank != source_rank + 1 or pair in seen):
            offsets[index] = (departures[source_rank] * 2, arrivals[target_rank] * 2)
            departures[source_rank] += 1
            arrivals[target_rank] += 1
        seen.add(pair)
    return offsets, dict(departures), dict(arrivals)


def trunk_tracks(
    edges: Iterable[DiagramEdge],
    placed: dict[str, PlacedNode],
    ranks: dict[str, int],
    direction: Direction,
) -> tuple[dict[int, int], dict[int, int]]:
    """Give unrelated edges in one rank gap their own trunk; edges of one fan keep sharing theirs.

    Adjacent-rank edges that share an endpoint form a net with a single trunk. Nets whose spans
    along the rank meet or touch take separate tracks, and a net leaving a row stays on the source
    side of the net arriving at that row, so their runs along it never overlap. Returns the track
    of each edge that has a trunk and the number of tracks per source rank.
    """
    edges = tuple(edges)
    detours, _, _ = detour_channels(edges, ranks)
    horizontal = direction is Direction.LEFT_RIGHT
    parents: dict[tuple[int, str], tuple[int, str]] = {}

    def root(key: tuple[int, str]) -> tuple[int, str]:
        while parents.setdefault(key, key) != key:
            parents[key] = key = parents[parents[key]]
        return key

    def center(node_id: str) -> int:
        node = placed[node_id]
        return node.y + node.height // 2 if horizontal else node.x + node.width // 2

    adjacent = [
        index
        for index, edge in enumerate(edges)
        if index not in detours and ranks[edge.target] == ranks[edge.source] + 1
    ]
    for index in adjacent:
        gap = ranks[edges[index].source]
        parents[root((gap, edges[index].source))] = root((gap, edges[index].target))
    members: defaultdict[tuple[int, str], list[int]] = defaultdict(list)
    for index in adjacent:
        members[root((ranks[edges[index].source], edges[index].source))].append(index)
    nets: defaultdict[int, list[_Net]] = defaultdict(list)
    for (gap, _), indexes in members.items():
        leaves = frozenset(center(edges[index].source) for index in indexes)
        enters = frozenset(center(edges[index].target) for index in indexes)
        if len(leaves | enters) > 1:
            nets[gap].append(_Net(min(leaves | enters), max(leaves | enters), leaves, enters, tuple(indexes)))
    tracks: dict[int, int] = {}
    counts: dict[int, int] = {}
    for gap, pending in nets.items():
        pending.sort(key=lambda net: (net.low, net.high, net.edges))
        track = 0
        while pending:
            row: list[_Net] = []
            for net in pending:
                if any(net.low <= other.high + 1 and net.high >= other.low - 1 for other in row):
                    continue
                if any(other is not net and other.leaves & net.enters for other in pending):
                    continue
                row.append(net)
            # Nets that each leave where the other arrives cannot both be satisfied.
            row = row or pending[:1]
            for net in row:
                tracks.update(dict.fromkeys(net.edges, track))
            pending = [net for net in pending if net not in row]
            track += 1
        counts[gap] = track
    return tracks, counts


def _separate_detour(
    points: tuple[Point, ...],
    direction: Direction,
    forward: bool,
    label: str,
    occupied: dict[int, list[tuple[int, int, int]]],
) -> tuple[Point, ...]:
    """Keep overlapping outer runs distinct; disjoint corridors may reuse a track."""
    first, second = points[2:4]
    horizontal = direction is Direction.LEFT_RIGHT
    track = first.y if horizontal else first.x
    start, end = sorted((first.x, second.x) if horizontal else (first.y, second.y))
    label_width = cell_len(sanitize_terminal_text(label)) + 1 if label and forward and not horizontal else 0
    step = 2 if forward else -2
    while any(
        start <= right and end >= left and track <= other_track + width and track + label_width >= other_track
        for other_track, segments in occupied.items()
        for left, right, width in segments
    ):
        track += step
    occupied.setdefault(track, []).append((start, end, label_width))
    shifted = (
        (Point(first.x, track), Point(second.x, track))
        if horizontal
        else (Point(track, first.y), Point(track, second.y))
    )
    return (*points[:2], *shifted, *points[4:])


def _single_bend(
    points: tuple[Point, ...],
    direction: Direction,
    placed: dict[str, PlacedNode],
    ends: tuple[str, str],
    occupied: dict[int, list[tuple[int, int, int]]],
) -> tuple[Point, ...] | None:
    """Replace a forward detour with one bend when a straight run to the target is free.

    The run follows the source's row (column, top-down) into the arrival track, or leaves on the
    departure track and follows the target's. It must stay a cell clear of every other node and off
    the outer tracks already taken, so it never shares cells with a detour that had to go around.
    """
    horizontal = direction is Direction.LEFT_RIGHT
    start, departure, arrival, end = points[0], points[1], points[4], points[5]
    candidates = (
        (start, Point(arrival.x, start.y) if horizontal else Point(start.x, arrival.y), arrival, end),
        (start, departure, Point(departure.x, end.y) if horizontal else Point(end.x, departure.y), end),
    )
    boxes = [node for node_id, node in placed.items() if node_id not in ends]
    for index, path in enumerate(candidates):
        if any(
            min(first.x, second.x) <= box.x + box.width
            and max(first.x, second.x) >= box.x - 1
            and min(first.y, second.y) <= box.y + box.height
            and max(first.y, second.y) >= box.y - 1
            for first, second in pairwise(path)
            for box in boxes
        ):
            continue
        first, second = path[index : index + 2]
        track = first.y if horizontal else first.x
        low, high = sorted((first.x, second.x) if horizontal else (first.y, second.y))
        if any(low <= right and high >= left for left, right, _ in occupied.get(track, ())):
            continue
        occupied.setdefault(track, []).append((low, high, 0))
        return path
    return None


def arrow_for_points(points: tuple[Point, ...]) -> str:
    """Return an arrow glyph matching the final cardinal path segment."""
    if len(points) < 2:
        return ""
    previous, final = points[-2:]
    if final.x > previous.x:
        return "▶"
    if final.x < previous.x:
        return "◀"
    if final.y > previous.y:
        return "▼"
    return "▲"


def _oriented_marker(marker: str, points: tuple[Point, ...], *, at_source: bool = False) -> str:
    if marker not in {"◀", "▶", "▲", "▼", "◁", "▷", "△", "▽"} or len(points) < 2:
        return marker
    direction = arrow_for_points((points[1], points[0])) if at_source else arrow_for_points(points)
    if marker in {"◀", "▶", "▲", "▼"}:
        return direction
    return {"◀": "◁", "▶": "▷", "▲": "△", "▼": "▽"}[direction]


def _label_point(points: tuple[Point, ...], label: str) -> Point | None:
    if not label or len(points) < 2:
        return None
    segments = [(first, second) for first, second in pairwise(points) if first != second]
    if not segments:
        return None
    horizontal = [(first, second) for first, second in segments if first.y == second.y]
    candidates = horizontal or segments
    first, second = max(candidates, key=lambda pair: abs(pair[1].x - pair[0].x) + abs(pair[1].y - pair[0].y))
    if first.y == second.y:
        midpoint = (first.x + second.x) // 2
        return Point(max(0, midpoint - cell_len(sanitize_terminal_text(label)) // 2), max(0, first.y - 1))
    midpoint = (first.y + second.y) // 2
    return Point(first.x + 1, midpoint)


def _rank_bounds(placed: dict[str, PlacedNode], ranks: dict[str, int]) -> dict[int, tuple[int, int, int, int]]:
    bounds: dict[int, tuple[int, int, int, int]] = {}
    for node_id, node in placed.items():
        rank = ranks[node_id]
        current = bounds.get(rank)
        node_bounds = (node.x, node.y, node.x + node.width - 1, node.y + node.height - 1)
        if current is None:
            bounds[rank] = node_bounds
        else:
            bounds[rank] = (
                min(current[0], node_bounds[0]),
                min(current[1], node_bounds[1]),
                max(current[2], node_bounds[2]),
                max(current[3], node_bounds[3]),
            )
    return bounds


def _resolve_label_collisions(
    routed_edges: tuple[RoutedEdge, ...],
    placed: dict[str, PlacedNode],
) -> tuple[RoutedEdge, ...]:
    occupied_by_y: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for node in placed.values():
        interval = (node.x, node.x + node.width - 1)
        for y in range(node.y, node.y + node.height):
            occupied_by_y[y].append(interval)
    for routed in routed_edges:
        for first, second in pairwise(routed.points):
            if first.y == second.y:
                occupied_by_y[first.y].append((min(first.x, second.x), max(first.x, second.x)))
            else:
                for y in range(min(first.y, second.y), max(first.y, second.y) + 1):
                    occupied_by_y[y].append((first.x, first.x))
    max_occupied_y = max(occupied_by_y, default=0)
    label_count = sum(
        bool(label)
        for routed in routed_edges
        for label in (routed.edge.label, routed.edge.source_label, routed.edge.target_label)
    )

    def is_available(point: Point, width: int) -> bool:
        right = point.x + width - 1
        return not any(
            not (right < left or point.x > other_right) for left, other_right in occupied_by_y.get(point.y, ())
        )

    def resolve(point: Point, label: str) -> Point:
        width = max(1, cell_len(sanitize_terminal_text(label)))
        if not is_available(point, width):
            search_limit = max(max_occupied_y, point.y) + label_count + 2
            for distance in range(1, search_limit + 1):
                # Try beside a crossing stem before moving far along it; vertical-
                # only displacement can lift a branch label above its source node.
                candidates = (
                    Point(point.x + distance, point.y),
                    Point(point.x - distance, point.y),
                    Point(point.x, point.y - distance),
                    Point(point.x, point.y + distance),
                )
                replacement = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.x >= 0 and candidate.y >= 0 and is_available(candidate, width)
                    ),
                    None,
                )
                if replacement is not None:
                    point = replacement
                    break
        occupied_by_y[point.y].append((point.x, point.x + width - 1))
        return point

    resolved = [
        replace(routed, label_at=resolve(routed.label_at, routed.edge.label)) if routed.label_at is not None else routed
        for routed in routed_edges
    ]
    for index, routed in enumerate(resolved):
        source_label_at = (
            resolve(Point(routed.points[0].x + 1, routed.points[0].y), routed.edge.source_label)
            if routed.edge.source_label
            else None
        )
        target_label_at = (
            resolve(Point(routed.points[-1].x + 1, routed.points[-1].y), routed.edge.target_label)
            if routed.edge.target_label
            else None
        )
        resolved[index] = replace(routed, source_label_at=source_label_at, target_label_at=target_label_at)
    return tuple(resolved)


def _route_top_down(
    source: PlacedNode,
    target: PlacedNode,
    source_rank: int,
    target_rank: int,
    bounds: dict[int, tuple[int, int, int, int]],
    lane: int,
    outer_right: int,
    departure_offset: int,
    approach_offset: int,
    midpoint: int,
) -> tuple[Point, ...]:
    source_center = source.x + source.width // 2
    target_center = target.x + target.width // 2
    start = Point(source_center, source.y + source.height)
    end = Point(target_center, target.y - 1)
    if source.node.node_id == target.node.node_id:
        right = source.x + source.width + 2 + lane * 2
        approach_y = source.y - 2 - lane * 2
        return (
            Point(source.x + source.width, source.y + source.height // 2),
            Point(right, source.y + source.height // 2),
            Point(right, approach_y),
            Point(source_center, approach_y),
            end,
        )
    if target_rank == source_rank + 1:
        if lane:
            side = outer_right + lane * 2
            lane_y = bounds[source_rank][3] + 2 + departure_offset
            approach_y = end.y - 1 - approach_offset
            return (
                start,
                Point(start.x, lane_y),
                Point(side, lane_y),
                Point(side, approach_y),
                Point(end.x, approach_y),
                end,
            )
        return (start, Point(start.x, midpoint), Point(end.x, midpoint), end)
    if target_rank > source_rank:
        right = outer_right + lane * 2
        departure_y = bounds[source_rank][3] + 2 + departure_offset
        approach_y = end.y - 1 - approach_offset
        return (
            start,
            Point(start.x, departure_y),
            Point(right, departure_y),
            Point(right, approach_y),
            Point(end.x, approach_y),
            end,
        )
    source_bottom = bounds[source_rank][3] + 2 + departure_offset
    left = min(bound[0] for bound in bounds.values()) - 4 - lane * 2
    approach_y = end.y - 1 - approach_offset
    return (
        start,
        Point(start.x, source_bottom),
        Point(left, source_bottom),
        Point(left, approach_y),
        Point(end.x, approach_y),
        end,
    )


def _route_left_right(
    source: PlacedNode,
    target: PlacedNode,
    source_rank: int,
    target_rank: int,
    bounds: dict[int, tuple[int, int, int, int]],
    lane: int,
    outer_bottom: int,
    departure_offset: int,
    approach_offset: int,
    midpoint: int,
) -> tuple[Point, ...]:
    source_center = source.y + source.height // 2
    target_center = target.y + target.height // 2
    start = Point(source.x + source.width, source_center)
    end = Point(target.x - 1, target_center)
    if source.node.node_id == target.node.node_id:
        bottom = source.y + source.height + 2 + lane
        approach_x = source.x - 2
        return (
            Point(source.x + source.width // 2, source.y + source.height),
            Point(source.x + source.width // 2, bottom),
            Point(approach_x, bottom),
            Point(approach_x, source_center),
            end,
        )
    if target_rank == source_rank + 1:
        if lane:
            bottom = outer_bottom + (lane - 1) * 2
            departure_x = bounds[source_rank][2] + 2 + departure_offset
            approach_x = end.x - 1 - approach_offset
            return (
                start,
                Point(departure_x, start.y),
                Point(departure_x, bottom),
                Point(approach_x, bottom),
                Point(approach_x, end.y),
                end,
            )
        return (start, Point(midpoint, start.y), Point(midpoint, end.y), end)
    if target_rank > source_rank:
        bottom = outer_bottom + lane * 2
        departure_x = bounds[source_rank][2] + 2 + departure_offset
        approach_x = end.x - 1 - approach_offset
        return (
            start,
            Point(departure_x, start.y),
            Point(departure_x, bottom),
            Point(approach_x, bottom),
            Point(approach_x, end.y),
            end,
        )
    source_right = bounds[source_rank][2] + 2 + departure_offset
    top = min(bound[1] for bound in bounds.values()) - 2 - lane * 2
    approach_x = end.x - 1 - approach_offset
    return (
        start,
        Point(source_right, start.y),
        Point(source_right, top),
        Point(approach_x, top),
        Point(approach_x, end.y),
        end,
    )


def route_edges(
    edges: Iterable[DiagramEdge],
    placed: dict[str, PlacedNode],
    ranks: dict[str, int],
    direction: Direction,
) -> tuple[RoutedEdge, ...]:
    """Route edges through rank channels, detouring around only the ranks they cross."""
    bounds = _rank_bounds(placed, ranks)
    edges = tuple(edges)
    detours, departures, arrivals = detour_channels(edges, ranks)
    tracks, trunks = trunk_tracks(edges, placed, ranks, direction)
    outer_tracks: dict[int, list[tuple[int, int, int]]] = {}
    routed: list[RoutedEdge] = []
    parallel_indexes: defaultdict[tuple[str, str], int] = defaultdict(int)
    backward_label_indexes: defaultdict[str, int] = defaultdict(int)
    successors: defaultdict[str, set[str]] = defaultdict(set)
    predecessors: defaultdict[str, set[str]] = defaultdict(set)
    for edge in edges:
        successors[edge.source].add(edge.target)
        predecessors[edge.target].add(edge.source)
    for index, edge in enumerate(edges):
        source = placed[edge.source]
        target = placed[edge.target]
        source_rank, target_rank = ranks[edge.source], ranks[edge.target]
        first_rank, last_rank = sorted((source_rank, target_rank))
        corridor = {rank: bound for rank, bound in bounds.items() if first_rank <= rank <= last_rank}
        outer_right = max(bound[2] for bound in corridor.values()) + 3
        outer_bottom = max(bound[3] for bound in corridor.values()) + 3
        parallel_key = (edge.source, edge.target)
        lane = parallel_indexes[parallel_key]
        parallel_indexes[parallel_key] += 1
        departure_offset, approach_offset = detours.get(index, (0, 0))
        source_padding = max(0, departures.get(source_rank, 0) - 1) * 2
        target_padding = max(0, arrivals.get(target_rank, 0) - 1) * 2
        far, near = (3, 1) if direction is Direction.TOP_DOWN else (2, 0)
        channel_start = bounds[source_rank][far] + source_padding
        channel_end = bounds[target_rank][near] - target_padding
        midpoint = (channel_start + channel_end) // 2
        spread = trunks.get(source_rank, 1) - 1
        # A gap laid out without room for every trunk keeps the shared one.
        if index in tracks and spread <= (channel_end - channel_start) // 2 - 3:
            midpoint += 2 * tracks[index] - spread
        if direction is Direction.TOP_DOWN:
            points = _route_top_down(
                source,
                target,
                source_rank,
                target_rank,
                corridor,
                lane,
                outer_right,
                departure_offset,
                approach_offset,
                midpoint,
            )
        else:
            points = _route_left_right(
                source,
                target,
                source_rank,
                target_rank,
                corridor,
                lane,
                outer_bottom,
                departure_offset,
                approach_offset,
                midpoint,
            )
        bend = (
            _single_bend(points, direction, placed, (edge.source, edge.target), outer_tracks)
            if target_rank > source_rank + 1 and not lane
            else None
        )
        if bend is not None:
            points = bend
        elif index in detours:
            points = _separate_detour(points, direction, target_rank > source_rank, edge.label, outer_tracks)
        target_marker = _oriented_marker(edge.target_marker, points) or (
            arrow_for_points(points) if edge.directed else ""
        )
        label_at = _label_point(points, edge.label)
        if edge.label and direction is Direction.LEFT_RIGHT and ranks[edge.target] == ranks[edge.source] + 1:
            channel_start = source.x + source.width
            channel_width = target.x - channel_start
            # Fan-out labels follow their destination branch, fan-in their source.
            # Using the upper endpoint groups lower-branch labels on the middle edge.
            fan_out = len(successors[edge.source]) > 1
            fan_in = len(predecessors[edge.target]) > 1
            label_y = points[-1].y if fan_out else points[0].y
            label_width = cell_len(sanitize_terminal_text(edge.label))
            label_x = (
                channel_start + 1
                if lane
                else points[1].x + 1
                if fan_out
                else points[1].x - label_width
                if fan_in
                else channel_start + max(1, (channel_width - label_width) // 2)
            )
            label_at = Point(
                label_x,
                max(0, (points[2].y if lane else label_y) - 1),
            )
        elif edge.label and direction is Direction.TOP_DOWN and target_rank == source_rank + 1 and not lane:
            fan_in = len(predecessors[edge.target]) > 1 and len(successors[edge.source]) == 1
            endpoint = points[0] if fan_in else points[-1]
            label_at = Point(endpoint.x + 1, endpoint.y + 1 if fan_in else endpoint.y - 1)
        elif edge.label and direction is Direction.TOP_DOWN and ranks[edge.source] > ranks[edge.target]:
            label_lane = backward_label_indexes[edge.target]
            backward_label_indexes[edge.target] += 1
            label_at = Point(points[2].x + 1, target.y + target.height + label_lane * 2)
        if (
            edge.label
            and direction is Direction.TOP_DOWN
            and ranks[edge.target] > ranks[edge.source]
            and index in detours
            and len(points) == 6
        ):
            outer_segment_start, outer_segment_end = points[2:4]
            label_at = Point(
                outer_segment_start.x + 1,
                (outer_segment_start.y + outer_segment_end.y) // 2 + lane,
            )
        routed.append(
            RoutedEdge(
                edge,
                points,
                label_at,
                points[-1] if target_marker else None,
                target_marker,
                points[0] if edge.source_marker else None,
                points[-1] if edge.target_marker else None,
            )
        )
    return _resolve_label_collisions(tuple(routed), placed)


def _overlay_styled_segments(canvas: TerminalCanvas, routed: RoutedEdge) -> None:
    if routed.edge.style is EdgeStyle.SOLID:
        return
    horizontal_glyph = "┅" if routed.edge.style is EdgeStyle.HEAVY else "┄"
    vertical_glyph = "┇" if routed.edge.style is EdgeStyle.HEAVY else "┊"
    for first, second in pairwise(routed.points):
        distance = abs(second.x - first.x) + abs(second.y - first.y)
        if distance < 2:
            continue
        if first.y == second.y:
            step = 1 if second.x > first.x else -1
            for x in range(first.x + step, second.x, step):
                if abs(x - first.x) % 2:
                    canvas.put(x, first.y, horizontal_glyph)
        else:
            step = 1 if second.y > first.y else -1
            for y in range(first.y + step, second.y, step):
                if abs(y - first.y) % 2:
                    canvas.put(first.x, y, vertical_glyph)


def draw_edges(canvas: TerminalCanvas, routed_edges: Iterable[RoutedEdge]) -> None:
    """Draw routed edges, endpoint markers, and plain labels."""
    routed_edges = tuple(routed_edges)
    for routed in routed_edges:
        canvas.draw_path(routed.points)
        _overlay_styled_segments(canvas, routed)
        if routed.source_marker_at is not None:
            canvas.put(
                routed.source_marker_at.x,
                routed.source_marker_at.y,
                _oriented_marker(routed.edge.source_marker, routed.points, at_source=True),
            )
        if routed.arrow_at is not None:
            canvas.put(routed.arrow_at.x, routed.arrow_at.y, routed.arrow)
    for routed in routed_edges:
        if routed.label_at is not None:
            canvas.draw_text(routed.label_at.x, routed.label_at.y, routed.edge.label)
        if routed.source_label_at is not None:
            canvas.draw_text(routed.source_label_at.x, routed.source_label_at.y, routed.edge.source_label)
        if routed.target_label_at is not None:
            canvas.draw_text(routed.target_label_at.x, routed.target_label_at.y, routed.edge.target_label)


def draw_nodes(canvas: TerminalCanvas, nodes: Iterable[PlacedNode]) -> None:
    """Draw node boxes and their cell-wrapped plain-text contents."""
    for placed in nodes:
        node = placed.node
        if node.shape is NodeShape.FORK_JOIN:
            canvas.draw_text(placed.x, placed.y, "━" * placed.width)
            continue
        if placed.width == 1 and placed.height == 1:
            canvas.put(placed.x, placed.y, node.label)
            continue
        rounded = node.shape in {NodeShape.CIRCLE, NodeShape.CYLINDER, NodeShape.ROUNDED, NodeShape.STADIUM}
        canvas.draw_box(placed.x, placed.y, placed.width, placed.height, rounded=rounded)
        if node.shape is NodeShape.SUBROUTINE:
            for y in range(placed.y + 1, placed.y + placed.height - 1):
                canvas.put(placed.x + 1, y, "│")
                canvas.put(placed.x + placed.width - 2, y, "│")
        for section_break in placed.section_breaks:
            canvas.draw_horizontal(placed.x, placed.x + placed.width - 1, placed.y + 1 + section_break)
        for line_offset, content_line in enumerate(placed.content_lines):
            available = placed.width - 2
            if node.shape is NodeShape.ENTITY and placed.section_breaks and line_offset > placed.section_breaks[0]:
                left = placed.x + 2
            else:
                left = placed.x + 1 + max(0, (available - cell_len(content_line)) // 2)
            canvas.draw_text(left, placed.y + 1 + line_offset, content_line)
