# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which run the bottom-right hook indicator shows, and how its id is fitted."""

from __future__ import annotations

import pytest
from rich.cells import cell_len

from chrys.app.tui.widgets import hook_activity
from chrys.app.tui.widgets.hook_activity import (
    HOOK_SHOW_AFTER_SECONDS,
    hook_activity_text,
    indicator_activity,
)
from chrys.foundation.models.hook_executions import HookExecutionView

NOW = 1000.0


def _view(
    hook_id: str,
    *,
    waited_at: str = "call",
    age: float = 5.0,
    queued: bool = False,
) -> HookExecutionView:
    return HookExecutionView(
        hook_id=hook_id,
        waited_at=waited_at,  # type: ignore[arg-type]
        queued=queued,
        started_at=NOW - age,
    )


def _pick(*views: HookExecutionView, now: float = NOW) -> HookExecutionView | None:
    return indicator_activity(views, now=now)


def _id(*views: HookExecutionView, now: float = NOW) -> str | None:
    activity = _pick(*views, now=now)
    return None if activity is None else activity.hook_id


# --- indicator_activity ---------------------------------------------------------


def test_the_threshold_constant() -> None:
    assert HOOK_SHOW_AFTER_SECONDS == 1.0
    assert not hasattr(hook_activity, "HOOK_ELAPSED_AFTER_SECONDS")


def test_nothing_running_picks_nothing() -> None:
    assert _pick() is None


def test_the_chosen_run_is_returned_as_its_snapshot_view() -> None:
    chosen = _view("protect-env", age=12, queued=True)
    # The chosen run is returned as it is: the very view from the snapshot.
    assert _pick(chosen) == chosen


@pytest.mark.parametrize("waited_at", ["call", "turn", "session"])
def test_every_waited_point_is_shown(waited_at: str) -> None:
    assert _id(_view("h", waited_at=waited_at)) == "h"


def test_runs_nobody_waits_for_never_show() -> None:
    assert _pick(_view("ff", waited_at="none", age=30), _view("interrupt", waited_at="none", age=9)) is None
    # Even the latest-started one is skipped when nobody waits for it.
    assert _id(_view("guard", age=5), _view("ff", waited_at="none", age=1.5)) == "guard"


def test_a_run_shows_only_after_a_second_queue_time_included() -> None:
    assert _pick(_view("young", age=0.5)) is None
    assert _id(_view("young", age=1.5)) == "young"
    assert _pick(_view("waiting", queued=True, age=0.2)) is None
    assert _id(_view("waiting", queued=True, age=1.5)) == "waiting"


def test_the_latest_started_shown_run_wins() -> None:
    assert _id(_view("a", age=9), _view("b", age=3), _view("c", age=6)) == "b"


def test_when_the_latest_ends_the_previous_one_shows_again() -> None:
    a, b = _view("a", age=9), _view("b", age=3)
    assert _id(a, b) == "b"
    assert _id(a) == "a"
    assert _id() is None


def test_a_younger_run_does_not_take_over_until_it_has_lasted_a_second() -> None:
    a, b = _view("a", age=9), _view("b", age=0.3)
    assert _id(a, b) == "a"
    assert _id(a, b, now=NOW + 1) == "b"


# --- hook_activity_text ---------------------------------------------------------


def _activity(hook_id: str = "slow-shell-blocking", *, queued: bool = False) -> HookExecutionView:
    return _view(hook_id, queued=queued, age=12)


def test_the_text_is_just_the_id() -> None:
    assert hook_activity_text(_activity()) == "slow-shell-blocking"
    assert hook_activity_text(_activity(queued=True)) == "slow-shell-blocking"


def _assert_middle_cut(original: str, text: str, width: int) -> None:
    """*text* keeps a head and a tail of *original* joined by one ``…`` and fits *width* cells."""
    assert cell_len(text) <= width, (width, text)
    assert text.count("…") == 1, text
    head, tail = text.split("…")
    assert head and tail, (width, text)
    assert original.startswith(head) and original.endswith(tail), (width, text)


def test_a_long_id_is_cut_in_the_middle_keeping_head_and_tail() -> None:
    hook_id = "a-very-long-hook-identifier"
    activity = _activity(hook_id)
    assert hook_activity_text(activity, max_cells=27) == hook_id
    for width in range(3, 27):
        text = hook_activity_text(activity, max_cells=width)
        _assert_middle_cut(hook_id, text, width)
    for width in range(1, 30):
        assert cell_len(hook_activity_text(activity, max_cells=width)) <= width


def test_a_cjk_id_is_cut_in_the_middle_by_cells() -> None:
    hook_id = "检查环境变量的钩子"  # 18 cells
    activity = _activity(hook_id)
    assert hook_activity_text(activity, max_cells=18) == hook_id
    for width in range(5, 18):
        text = hook_activity_text(activity, max_cells=width)
        _assert_middle_cut(hook_id, text, width)
    for width in range(1, 20):
        assert cell_len(hook_activity_text(activity, max_cells=width)) <= width


def test_control_characters_are_not_shown() -> None:
    activity = _pick(_view("bad\x1b[31mid\x07", age=2))
    assert activity is not None
    text = hook_activity_text(activity)
    assert "\x1b" not in text
    assert "\x07" not in text


def test_a_running_run_outranks_a_later_queued_one() -> None:
    """A queued run shows only when none is running; the slot holder is what blocks."""
    assert _id(_view("a", age=6), _view("b", queued=True, age=2)) == "a"
    assert _id(_view("a", age=6), _view("b", queued=True, age=2), _view("c", age=4)) == "c"


def test_only_queued_runs_show_the_latest_queued() -> None:
    assert _id(_view("q-old", queued=True, age=5), _view("q-new", queued=True, age=2)) == "q-new"


def test_a_running_run_under_a_second_does_not_hide_a_queued_one_that_has_waited() -> None:
    """The threshold applies first: a running run that has not lasted a second is not yet a candidate."""
    assert _id(_view("q", queued=True, age=3), _view("young", age=0.4)) == "q"
    assert _id(_view("q", queued=True, age=3), _view("young", age=0.4), now=NOW + 0.7) == "young"


def test_a_slot_holder_nobody_waits_for_never_shows_the_queued_waited_run_does() -> None:
    """F (fire_and_forget) fills the slot and B (blocking) queues: B shows, never F."""
    f = _view("f", waited_at="none", age=6)
    b = _view("b", queued=True, age=2)
    assert _id(f, b) == "b"
    assert _pick(f, _view("b", queued=True, age=0.5)) is None, "B has not waited a second yet"
    assert _pick(f) is None
    # F ends and B takes the slot: still B.
    assert _id(_view("b", age=2)) == "b"


# --- elide_middle's width contract ----------------------------------------------


@pytest.mark.parametrize("text", ["a-very-long-hook-identifier", "检查环境变量的钩子", "ab", "x"])
def test_elide_middle_never_exceeds_its_budget(text: str) -> None:
    from chrys.app.tui.util.formatting import elide_middle

    assert elide_middle(text, 0) == ""
    assert elide_middle(text, -3) == ""
    assert elide_middle(text, 1) == (text if cell_len(text) <= 1 else "…")
    for width in range(1, cell_len(text) + 3):
        result = elide_middle(text, width)
        assert cell_len(result) <= width, (width, result)
        if cell_len(text) <= width:
            assert result == text
