# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Detours depend on the ranks they cross, while still avoiding intermediate nodes."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import pytest

from chrys.app.tui.widgets.markdown.diagram.layout import compile_ir_with_geometry
from chrys.app.tui.widgets.markdown.diagram.model import (
    DiagramEdge,
    DiagramIR,
    DiagramKind,
    DiagramNode,
    Direction,
    PlacedNode,
    Point,
    RoutedEdge,
)
from chrys.app.tui.widgets.markdown.diagram.router import route_edges


def _placements(direction: Direction) -> dict[str, PlacedNode]:
    placed = {
        name: PlacedNode(DiagramNode(name, name), x, y, 9, 5, (name,))
        for name, x, y in (("A", 40, 40), ("B", 65, 40), ("C", 90, 40), ("upper", 10, 5), ("lower", 10, 80))
    }
    if direction is Direction.TOP_DOWN:
        placed = {
            name: replace(node, x=node.y, y=node.x, width=node.height, height=node.width)
            for name, node in placed.items()
        }
    return placed


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_unrelated_parallel_rank_does_not_stretch_local_detours(direction: Direction) -> None:
    placed = _placements(direction)
    ranks = {"upper": 0, "lower": 0, "A": 1, "B": 2, "C": 3}
    edges = (
        DiagramEdge("A", "C", "skip"),
        DiagramEdge("C", "A", "repeat"),
        DiagramEdge("A", "B", "first"),
        DiagramEdge("A", "B", "second"),
        DiagramEdge("B", "B", "self"),
    )
    local = {name: placed[name] for name in ("A", "B", "C")}
    expected = route_edges(edges, local, ranks, direction)

    assert route_edges(edges, placed, ranks, direction) == expected


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_local_detours_still_clear_a_large_intermediate_rank(direction: Direction) -> None:
    placed = _placements(direction)
    # The same wide/tall branch now lies BETWEEN the endpoints and must be avoided.
    middle = placed["B"]
    for name in ("upper", "lower"):
        placed[name] = (
            replace(placed[name], x=middle.x)
            if direction is Direction.LEFT_RIGHT
            else replace(placed[name], y=middle.y)
        )
    ranks = {"A": 1, "B": 2, "upper": 2, "lower": 2, "C": 3}
    edges = (DiagramEdge("A", "C"), DiagramEdge("C", "A"))
    routed = route_edges(edges, placed, ranks, direction)
    for route in routed:
        for first, second in pairwise(route.points):
            assert first.x == second.x or first.y == second.y
            for box in placed.values():
                assert (
                    max(first.x, second.x) < box.x
                    or min(first.x, second.x) >= box.x + box.width
                    or max(first.y, second.y) < box.y
                    or min(first.y, second.y) >= box.y + box.height
                ), (route.edge, first, second, box.node.node_id)


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
@pytest.mark.parametrize("backward", [False, True])
def test_independent_detours_do_not_merge_into_one_outer_line(direction: Direction, backward: bool) -> None:
    placed = {
        name: PlacedNode(DiagramNode(name, name), x, y, 9, 5, (name,))
        for name, x, y in (("A", 20, 20), ("B", 50, 20), ("C", 80, 20), ("D", 110, 20))
    }
    if direction is Direction.TOP_DOWN:
        placed = {
            name: replace(box, x=box.y, y=box.x, width=box.height, height=box.width) for name, box in placed.items()
        }
    ranks = dict(zip(placed, range(4), strict=True))
    edges = (DiagramEdge("A", "C"), DiagramEdge("B", "D"))
    if backward:
        edges = tuple(replace(edge, source=edge.target, target=edge.source) for edge in edges)
    first, second = route_edges(edges, placed, ranks, direction)
    assert (
        (first.points[2].y != second.points[2].y)
        if direction is Direction.LEFT_RIGHT
        else (first.points[2].x != second.points[2].x)
    )


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_parallel_loops_have_separate_return_approaches(direction: Direction) -> None:
    placed = {
        name: PlacedNode(DiagramNode(name, name), x, y, 9, 5, (name,))
        for name, x, y in (("loop_a", 20, 20), ("loop_b", 20, 35), ("body_a", 55, 20), ("body_b", 55, 35))
    }
    if direction is Direction.TOP_DOWN:
        placed = {
            name: replace(box, x=box.y, y=box.x, width=box.height, height=box.width) for name, box in placed.items()
        }
    ranks = {"loop_a": 0, "loop_b": 0, "body_a": 1, "body_b": 1}
    first, second = route_edges(
        (DiagramEdge("body_a", "loop_a"), DiagramEdge("body_b", "loop_b")), placed, ranks, direction
    )
    # Distinct loop entries must not share a return stem and suggest cross-loop edges.
    assert (
        (first.points[4].x != second.points[4].x)
        if direction is Direction.LEFT_RIGHT
        else (first.points[4].y != second.points[4].y)
    )


@pytest.mark.parametrize("direction", list(Direction))
def test_dense_detours_keep_labels_arrows_and_nodes_unobstructed(direction: Direction) -> None:
    names = ("start", "validate", "assess", "approve", "report", "finish")
    edges = (
        *(DiagramEdge(a, b) for a, b in pairwise(names)),
        DiagramEdge("start", "report", "cached_result"),
        DiagramEdge("validate", "finish", "no_changes"),
        DiagramEdge("assess", "finish", "rejected"),
        DiagramEdge("approve", "finish", "dry_run"),
    )
    diagram, placed = compile_ir_with_geometry(
        "", DiagramIR(DiagramKind.FLOWCHART, direction, tuple(DiagramNode(name, name) for name in names), edges)
    )
    assert not diagram.diagnostics
    routes = diagram.routed_edges
    for route in routes:
        for first, second in pairwise(route.points):
            assert min(first.x, first.y, second.x, second.y) >= 0
            for box in placed.values():
                assert (
                    max(first.x, second.x) < box.x
                    or min(first.x, second.x) >= box.x + box.width
                    or max(first.y, second.y) < box.y
                    or min(first.y, second.y) >= box.y + box.height
                )
        if route.label_at is None:
            continue
        point, label = route.label_at, route.edge.label
        assert diagram.rows[point.y][point.x : point.x + len(label)] == label
        for other in routes:
            for first, second in pairwise(other.points):
                assert (
                    max(first.x, second.x) < point.x
                    or min(first.x, second.x) >= point.x + len(label)
                    or max(first.y, second.y) < point.y
                    or min(first.y, second.y) > point.y
                ), (label, other.edge)


@pytest.mark.parametrize("direction", list(Direction))
def test_many_loop_returns_remain_inside_canvas(direction: Direction) -> None:
    names = ("loop", *(f"branch_{index}" for index in range(6)))
    ir = DiagramIR(
        DiagramKind.FLOWCHART,
        direction,
        tuple(DiagramNode(name, name) for name in names),
        tuple(
            edge
            for name in names[1:]
            for edge in (DiagramEdge("loop", name), DiagramEdge(name, "loop", constrains_rank=False))
        ),
    )
    diagram, _ = compile_ir_with_geometry("", ir)
    assert not diagram.diagnostics
    assert len(diagram.routed_edges) == 12
    for route in diagram.routed_edges:
        for point in route.points:
            assert 0 <= point.x < diagram.width
            assert 0 <= point.y < diagram.height


@pytest.mark.parametrize("fan_in", [False, True])
def test_horizontal_branch_labels_follow_their_branch(fan_in: bool) -> None:
    placed = {
        name: PlacedNode(DiagramNode(name, name), x, y, 9, 5, (name,))
        for name, x, y in (("choice", 10, 25), ("upper", 50, 10), ("middle", 50, 25), ("lower", 50, 40))
    }
    ranks = {"choice": 0, "upper": 1, "middle": 1, "lower": 1}
    edges = tuple(DiagramEdge("choice", name, name) for name in ("upper", "middle", "lower"))
    if fan_in:
        placed = {name: replace(box, x=60 - box.x) for name, box in placed.items()}
        ranks = {name: 1 - rank for name, rank in ranks.items()}
        edges = tuple(replace(edge, source=edge.target, target=edge.source) for edge in edges)
    for route in route_edges(edges, placed, ranks, Direction.LEFT_RIGHT):
        branch = placed[route.edge.source if fan_in else route.edge.target]
        assert route.label_at is not None
        assert abs(route.label_at.y - (branch.y + branch.height // 2)) == 1


def test_vertical_branch_label_stays_between_its_nodes() -> None:
    names = ("check", "approved", "manual", "denied")
    ir = DiagramIR(
        DiagramKind.FLOWCHART,
        Direction.TOP_DOWN,
        tuple(DiagramNode(name, name) for name in names),
        tuple(DiagramEdge("check", name, f"requires_{name}_approval") for name in names[1:]),
    )
    diagram, placed = compile_ir_with_geometry("", ir)
    for route in diagram.routed_edges:
        assert route.label_at is not None
        assert placed["check"].y + placed["check"].height <= route.label_at.y < placed[route.edge.target].y


def _boxes(direction: Direction, *spots: tuple[str, int, int]) -> dict[str, PlacedNode]:
    """Nodes given as (name, x, y) for a left-right layout; transposed for top-down."""
    placed = {name: PlacedNode(DiagramNode(name, name), x, y, 9, 5, (name,)) for name, x, y in spots}
    if direction is Direction.TOP_DOWN:
        placed = {
            name: replace(box, x=box.y, y=box.x, width=box.height, height=box.width) for name, box in placed.items()
        }
    return placed


def _along(direction: Direction, point: Point) -> int:
    """The coordinate a rank-to-rank run keeps constant: its row left-right, its column top-down."""
    return point.y if direction is Direction.LEFT_RIGHT else point.x


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_a_skip_edge_with_a_free_row_bends_once_instead_of_going_around(direction: Direction) -> None:
    placed = _boxes(direction, ("side", 10, 40), ("middle", 40, 10), ("target", 70, 10))
    ranks = {"side": 0, "middle": 1, "target": 2}

    [route] = route_edges((DiagramEdge("side", "target"),), placed, ranks, direction)

    start, corner, approach, end = route.points
    assert _along(direction, corner) == _along(direction, start)
    assert _along(direction, approach) == _along(direction, end)


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_a_skip_edge_whose_own_row_is_blocked_runs_along_the_targets_row(direction: Direction) -> None:
    placed = _boxes(direction, ("source", 10, 40), ("blocker", 40, 40), ("target", 70, 10))
    ranks = {"source": 0, "blocker": 1, "target": 2}

    [route] = route_edges((DiagramEdge("source", "target"),), placed, ranks, direction)

    start, departure, corner, end = route.points
    assert _along(direction, departure) == _along(direction, start)
    assert _along(direction, corner) == _along(direction, end)


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_a_skip_edge_with_both_rows_blocked_still_goes_around(direction: Direction) -> None:
    placed = _boxes(direction, ("A", 10, 10), ("B", 40, 10), ("C", 70, 10))

    [route] = route_edges((DiagramEdge("A", "C"),), placed, {"A": 0, "B": 1, "C": 2}, direction)

    assert len(route.points) == 6
    blocker = placed["B"]
    outer = _along(direction, route.points[2])
    assert outer > (blocker.y + blocker.height if direction is Direction.LEFT_RIGHT else blocker.x + blocker.width)


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
@pytest.mark.parametrize("straight_first", [False, True])
def test_a_straight_run_never_shares_a_track_with_a_detour(direction: Direction, straight_first: bool) -> None:
    # A -> C has to go around B on the track three cells past them, which is exactly the row of `low`.
    placed = _boxes(direction, ("low", 10, 15), ("A", 40, 10), ("B", 70, 10), ("C", 100, 10), ("far", 130, 60))
    ranks = {"low": 0, "A": 1, "B": 2, "C": 3, "far": 4}
    edges = (DiagramEdge("A", "C"), DiagramEdge("low", "far"))

    routed = route_edges(edges[::-1] if straight_first else edges, placed, ranks, direction)

    runs = []
    for route in routed:
        for first, second in pairwise(route.points):
            if _along(direction, first) == _along(direction, second) and first != second:
                span = sorted((first.x, second.x) if direction is Direction.LEFT_RIGHT else (first.y, second.y))
                runs.append((route.edge.source, _along(direction, first), span))
    for index, (owner, track, (low, high)) in enumerate(runs):
        for other_owner, other_track, (other_low, other_high) in runs[index + 1 :]:
            assert owner == other_owner or track != other_track or high < other_low or other_high < low


def _across(direction: Direction, point: Point) -> int:
    """The coordinate that advances from rank to rank: the column left-right, the row top-down."""
    return point.x if direction is Direction.LEFT_RIGHT else point.y


def _overlapping_runs(direction: Direction, routed: tuple[RoutedEdge, ...]) -> list[tuple[str, str]]:
    """Pairs of sources whose routes lie on top of each other for at least a cell; crossings are fine."""
    runs = []
    for route in routed:
        for first, second in pairwise(route.points):
            if first != second:
                lengthwise = first.y == second.y
                low, high = sorted((first.x, second.x) if lengthwise else (first.y, second.y))
                runs.append((route.edge.source, lengthwise, first.y if lengthwise else first.x, low, high))
    return [
        (owner, other_owner)
        for index, (owner, lengthwise, track, low, high) in enumerate(runs)
        for other_owner, other_lengthwise, other_track, other_low, other_high in runs[index + 1 :]
        if owner != other_owner
        and (lengthwise, track) == (other_lengthwise, other_track)
        and low <= other_high
        and other_low <= high
    ]


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_unrelated_edges_in_one_gap_do_not_merge_into_one_trunk(direction: Direction) -> None:
    # `solo` sits on the row where the fan's last branch arrives, so a shared trunk reads solo -> Z.
    placed = _boxes(
        direction, ("fan", 2, 14), ("solo", 2, 26), ("X", 40, 2), ("Y", 40, 14), ("Z", 40, 26), ("Q", 40, 38)
    )
    ranks = {"fan": 0, "solo": 0, "X": 1, "Y": 1, "Z": 1, "Q": 1}
    edges = (DiagramEdge("fan", "X"), DiagramEdge("fan", "Y"), DiagramEdge("fan", "Z"), DiagramEdge("solo", "Q"))

    routed = route_edges(edges, placed, ranks, direction)

    to_x, _, to_z, solo = routed
    assert _across(direction, to_x.points[1]) == _across(direction, to_z.points[1])
    # The edge leaving that row turns before the trunk arriving at it, so neither runs over the other.
    assert _across(direction, solo.points[1]) < _across(direction, to_z.points[1])
    assert _overlapping_runs(direction, routed) == []


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_each_edge_of_a_staircase_turns_before_the_one_arriving_on_its_row(direction: Direction) -> None:
    placed = _boxes(direction, ("A", 2, 2), ("B", 2, 14), ("C", 2, 26), ("X", 40, 14), ("Y", 40, 26), ("Z", 40, 38))
    ranks = {"A": 0, "B": 0, "C": 0, "X": 1, "Y": 1, "Z": 1}
    edges = (DiagramEdge("A", "X"), DiagramEdge("B", "Y"), DiagramEdge("C", "Z"))

    routed = route_edges(edges, placed, ranks, direction)

    first, second, third = (_across(direction, route.points[1]) for route in routed)
    assert third < second < first
    assert _overlapping_runs(direction, routed) == []


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_edges_joined_by_shared_endpoints_keep_one_trunk(direction: Direction) -> None:
    placed = _boxes(direction, ("A", 2, 2), ("B", 2, 14), ("C", 40, 2), ("D", 40, 14), ("E", 40, 26))
    ranks = {"A": 0, "B": 0, "C": 1, "D": 1, "E": 1}
    edges = (DiagramEdge("A", "D"), DiagramEdge("B", "D"), DiagramEdge("B", "E"))

    routed = route_edges(edges, placed, ranks, direction)

    assert len({_across(direction, route.points[1]) for route in routed}) == 1


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_edges_that_each_leave_where_the_other_arrives_still_get_separate_trunks(direction: Direction) -> None:
    placed = _boxes(direction, ("A", 2, 2), ("B", 2, 14), ("C", 40, 2), ("D", 40, 14))
    ranks = {"A": 0, "B": 0, "C": 1, "D": 1}

    down, up = route_edges((DiagramEdge("A", "D"), DiagramEdge("B", "C")), placed, ranks, direction)

    assert _across(direction, down.points[1]) != _across(direction, up.points[1])


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_a_gap_without_room_for_every_trunk_keeps_the_shared_one(direction: Direction) -> None:
    placed = _boxes(direction, ("A", 2, 2), ("B", 2, 14), ("C", 17, 14), ("D", 17, 26))
    ranks = {"A": 0, "B": 0, "C": 1, "D": 1}

    first, second = route_edges((DiagramEdge("A", "C"), DiagramEdge("B", "D")), placed, ranks, direction)

    trunk = _across(direction, first.points[1])
    assert trunk == _across(direction, second.points[1])
    source, target = placed["A"], placed["C"]
    if direction is Direction.LEFT_RIGHT:
        assert source.x + source.width < trunk < target.x - 1
    else:
        assert source.y + source.height < trunk < target.y - 1


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_the_layout_widens_a_gap_by_the_trunks_it_holds(direction: Direction) -> None:
    def compiled(edges: tuple[DiagramEdge, ...]) -> tuple[dict[str, PlacedNode], tuple[RoutedEdge, ...]]:
        names = ("F", "S", "X", "Y", "Z", "Q")
        ir = DiagramIR(DiagramKind.FLOWCHART, direction, tuple(DiagramNode(name, name) for name in names), edges)
        diagram, geometry = compile_ir_with_geometry("", ir)
        return geometry, diagram.routed_edges

    fan = (DiagramEdge("F", "X"), DiagramEdge("F", "Y"), DiagramEdge("F", "Z"))
    shared, _ = compiled((*fan, DiagramEdge("F", "Q")))
    separate, routed = compiled((*fan, DiagramEdge("S", "Q")))

    def gap(geometry: dict[str, PlacedNode]) -> int:
        source, target = geometry["F"], geometry["Y"]
        if direction is Direction.LEFT_RIGHT:
            return target.x - (source.x + source.width)
        return target.y - (source.y + source.height)

    assert gap(separate) == gap(shared) + 2
    trunks = {route.edge.source: _across(direction, route.points[1]) for route in routed}
    assert abs(trunks["F"] - trunks["S"]) == 2
    assert _overlapping_runs(direction, routed) == []


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_trunks_that_would_meet_end_to_end_take_separate_tracks(direction: Direction) -> None:
    # One trunk ends on the cell before the other begins: on one track their corners would touch.
    placed = _boxes(direction, ("A", 2, 2), ("B", 2, 15), ("X", 40, 14), ("Y", 40, 27))
    ranks = {"A": 0, "B": 0, "X": 1, "Y": 1}

    first, second = route_edges((DiagramEdge("A", "X"), DiagramEdge("B", "Y")), placed, ranks, direction)

    assert _across(direction, first.points[1]) != _across(direction, second.points[1])
