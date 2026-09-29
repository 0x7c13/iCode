# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the patch that keeps Textual widgets' unchanged defaults on the class."""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import subprocess
import sys
import weakref
from collections import deque
from functools import cached_property
from pathlib import Path
from typing import Any

import pytest
import textual
from textual import _queue as queue_module
from textual import events
from textual import message_pump as message_pump_module
from textual.app import App, ComposeResult
from textual.dom import DOMNode
from textual.message_pump import MessagePump
from textual.reactive import ReactiveError
from textual.rlock import RLock
from textual.signal import Signal
from textual.widget import Widget
from textual.widgets import Static

from chrys.foundation.patches import patcher, textual_node_diet
from chrys.foundation.patches.patcher import FilePatch, PatchResult
from chrys.foundation.patches.staged_members import members_installed, stage_patched_source

Queue = queue_module.Queue


class _Value:
    pass


class _StaticApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("probe")


def _class_defaults() -> set[str]:
    """Every plain value the patch moves onto a class."""
    names: set[str] = set()
    for module_name, members in textual_node_diet._RUNTIME_MEMBERS.items():
        module = importlib.import_module(module_name)
        for class_name, member_names in members.items():
            class_vars = vars(vars(module)[class_name])
            names.update(
                name
                for name in member_names
                if not callable(class_vars[name]) and not isinstance(class_vars[name], property | cached_property)
            )
    return names


def test_the_installed_textual_is_patched() -> None:
    assert all(
        members_installed(importlib.import_module(name), members, textual_node_diet._RUNTIME_PATCH_MARKER)
        for name, members in textual_node_diet._RUNTIME_MEMBERS.items()
    )


async def test_the_queue_hands_values_over_in_order_across_compactions() -> None:
    queue: Any = Queue()
    for value in range(200):
        queue.put_nowait(value)
    received = [await queue.get() for _ in range(150)]
    for value in range(200, 260):
        queue.put_nowait(value)
    received += [queue.get_nowait() for _ in range(queue.qsize())]

    assert received == list(range(260))
    assert queue.empty()
    with pytest.raises(asyncio.QueueEmpty):
        queue.get_nowait()


async def test_a_waiting_get_takes_the_next_put() -> None:
    queue: Any = Queue()
    getter = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    assert not getter.done()

    queue.put_nowait("message")

    assert await asyncio.wait_for(getter, timeout=1) == "message"


async def test_a_get_cancelled_after_its_wake_up_leaves_the_value_queued() -> None:
    queue: Any = Queue()
    getter = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    assert not getter.done()

    queue.put_nowait("kept")
    getter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await getter

    assert queue.get_nowait() == "kept"


def test_a_consumed_value_is_released_before_the_queue_drains() -> None:
    queue: Any = Queue()
    value = _Value()
    ref = weakref.ref(value)
    queue.put_nowait(value)
    queue.put_nowait(_Value())
    del value

    queue.get_nowait()

    assert ref() is None
    assert queue.qsize() == 1


async def test_the_flag_wakes_its_waiters_and_forgets_a_cancelled_one() -> None:
    flag = vars(message_pump_module)["Flag"]()
    waiter = asyncio.create_task(flag.wait())
    cancelled = asyncio.create_task(flag.wait())
    await asyncio.sleep(0)
    assert not waiter.done()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert len(flag._waiters) == 1

    flag.set()

    assert await asyncio.wait_for(waiter, timeout=1) is True
    assert flag.is_set()
    assert await flag.wait() is True


def test_a_new_widget_stores_no_unchanged_default() -> None:
    defaults = _class_defaults()
    assert {"_running", "_timers", "_has_hover_style", "_render_cache", "_extrema"} <= defaults

    assert not defaults & set(vars(Static("probe")))


def test_disabled_messages_belong_to_one_pump() -> None:
    first, second = MessagePump(), MessagePump()

    first.disable_messages(events.Key)

    assert events.Key in first._disabled_messages
    assert not second._disabled_messages
    assert MessagePump._disabled_messages == frozenset()
    first.enable_messages(events.Key)
    assert not first._disabled_messages


def test_css_type_names_are_shared_by_a_class() -> None:
    first, second = Static("a"), Static("b")

    assert first._css_types is second._css_types
    assert first._css_types == {cls.__name__ for cls in Static._css_bases(Static)}
    assert DOMNode()._css_types == {"DOMNode"}


def test_the_widget_lock_is_made_on_first_use() -> None:
    first, second = Widget(), Widget()
    assert "lock" not in vars(first)

    assert isinstance(first.lock, RLock)
    assert first.lock is first.lock
    assert first.lock is not second.lock


def test_a_widget_that_skips_super_init_is_still_detected() -> None:
    class DaftWidget(Widget):
        def __init__(self) -> None:
            pass

    with pytest.raises(ReactiveError):
        DaftWidget().disabled = True


async def test_timers_and_signal_subscriptions_are_made_on_first_use() -> None:
    app = _StaticApp()
    async with app.run_test():
        static = app.query_one(Static)
        assert static._timers == frozenset()
        timer = static.set_timer(60, lambda: None)
        assert set(static._timers) == {timer}
        assert MessagePump._timers == frozenset()
        timer.stop()

        signal: Signal[str] = Signal(static, "probe")
        assert signal._subscriptions is None
        assert "probe" in repr(signal)
        signal.unsubscribe(static)
        signal.publish("ignored")
        received: list[str] = []
        signal.subscribe(static, received.append, immediate=True)
        signal.publish("sent")
        signal.unsubscribe(static)
        signal.publish("dropped")

        assert received == ["sent"]


def _restore_upstream(patch: FilePatch) -> list[FilePatch]:
    """Patches that turn the patched fragment, or any equivalent it accepts, back into upstream's."""
    variants = (patch.new_fragment, *patch.equivalent_fragments)
    return [
        FilePatch(
            package=patch.package,
            module_file=patch.module_file,
            old_fragment=variant,
            new_fragment=patch.old_fragment,
            description=f"Restore upstream: {patch.description}",
            equivalent_fragments=variants[index + 1 :],
        )
        for index, variant in enumerate(variants)
    ]


def test_upstream_pumps_build_their_queue_containers_up_front() -> None:
    """Pins that the patch is still needed; drop the patch once a Textual upgrade makes this fail."""
    upstream = _restore_upstream(textual_node_diet._QUEUE_PATCH)
    namespace: dict[str, Any] = {"__name__": "chrys_upstream_textual_queue"}
    exec(compile(stage_patched_source(queue_module, upstream), queue_module.__file__ or "_queue.py", "exec"), namespace)

    assert isinstance(vars(namespace["Queue"]())["values"], deque)


_LAUNCH = """\
import asyncio

import textual
from textual.app import App
from textual.widgets import Static


class Probe(App):
    def compose(self):
        yield Static("probe")


async def main():
    async with Probe().run_test() as pilot:
        await pilot.pause()


asyncio.run(main())
print(textual.__file__)
"""


def _textual_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy the installed Textual into a site directory that the patcher writes instead."""
    site = tmp_path / "site"
    shutil.copytree(Path(textual.__file__).parent, site / "textual", ignore=shutil.ignore_patterns("__pycache__"))

    def locate(package: str) -> Path:
        if package != "textual":
            raise ImportError(package)
        return site / "textual"

    monkeypatch.setattr(patcher, "_locate_package_dir", locate)
    monkeypatch.setattr(patcher, "_apply_runtime_patches", lambda _runtime_patches: None)
    return site


def _patch_files() -> dict[FilePatch, PatchResult]:
    """Patch the copied Textual's files as every launch does."""
    return {result.patch: result for result in patcher.apply_all() if result.patch.package == "textual"}


def _launch(site: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run an app in a new process that imports Textual from ``site``."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = os.environ | {
        "PYTHONPATH": os.fspath(site),
        "HOME": os.fspath(home),
        "USERPROFILE": os.fspath(home),
        "APPDATA": os.fspath(tmp_path / "appdata"),
    }
    return subprocess.run(
        [sys.executable, "-c", _LAUNCH],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=40,
        check=False,
    )


def _rewrite(path: Path, *replacements: tuple[str, str]) -> None:
    """Replace each fragment, which must occur exactly once, in ``path``."""
    source = path.read_text(encoding="utf-8")
    for fragment, replacement in replacements:
        assert source.count(fragment) == 1, fragment
        source = source.replace(fragment, replacement)
    path.write_text(source, encoding="utf-8")


def _assert_launches(site: Path, tmp_path: Path) -> None:
    launched = _launch(site, tmp_path)
    assert launched.returncode == 0, launched.stderr
    assert Path(launched.stdout.strip()).parent == site / "textual"


def test_the_next_launch_starts_when_the_queue_patch_no_longer_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _textual_copy(tmp_path, monkeypatch)
    assert all(
        result.status != "error"
        for result in patcher.apply_patch_group(_restore_upstream(textual_node_diet._QUEUE_PATCH))
    )
    _rewrite(site / "textual" / "_queue.py", ("just enough functionality", "enough functionality"))

    results = _patch_files().values()

    assert {result.status for result in results if result.patch.module_file == "_queue.py"} == {"error"}
    assert all(result.status != "error" for result in results if result.patch.module_file == "message_pump.py")
    _assert_launches(site, tmp_path)


def _write_earlier_patch_state(site: Path) -> tuple[Path, Path]:
    """Leave the copy's queue and pump as the patch wrote them while ``Flag`` lived in ``_queue.py``."""
    _patch_files()
    queue_file = site / "textual" / "_queue.py"
    pump_file = site / "textual" / "message_pump.py"
    # A development install the earlier patch wrote keeps that queue, and so does its copy.
    if textual_node_diet._QUEUE_WITH_FLAG_NEW not in queue_file.read_text(encoding="utf-8"):
        _rewrite(queue_file, (textual_node_diet._QUEUE_NEW, textual_node_diet._QUEUE_WITH_FLAG_NEW))
    _rewrite(
        pump_file,
        (textual_node_diet._PUMP_FLAG_NEW, textual_node_diet._PUMP_FLAG_OLD),
        (textual_node_diet._PUMP_IMPORT_WITH_FLAG_NEW, textual_node_diet._PUMP_IMPORT_WITH_FLAG_OLD),
    )
    return queue_file, pump_file


def test_a_launch_gives_the_pump_its_own_flag_after_an_earlier_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _textual_copy(tmp_path, monkeypatch)
    queue_file, pump_file = _write_earlier_patch_state(site)

    results = _patch_files()

    assert all(result.status != "error" for result in results.values())
    assert [
        results[patch].status for patch in (textual_node_diet._QUEUE_PATCH, *textual_node_diet._PUMP_PATCHES[:2])
    ] == ["skipped", "applied", "applied"]
    assert textual_node_diet._QUEUE_WITH_FLAG_NEW in queue_file.read_text(encoding="utf-8")
    pump_source = pump_file.read_text(encoding="utf-8")
    assert "class Flag:" in pump_source
    assert textual_node_diet._PUMP_IMPORT_WITH_FLAG_OLD not in pump_source
    _assert_launches(site, tmp_path)


def test_the_next_launch_starts_when_an_earlier_patch_state_meets_a_pump_patch_that_no_longer_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _textual_copy(tmp_path, monkeypatch)
    _queue_file, pump_file = _write_earlier_patch_state(site)
    drifted = textual_node_diet._PUMP_FLAG_OLD.replace("    pass\n", "    pass  # drifted\n")
    _rewrite(pump_file, (textual_node_diet._PUMP_FLAG_OLD, drifted))

    results = _patch_files().values()

    assert "error" in {result.status for result in results if result.patch.module_file == "message_pump.py"}
    assert all(result.status != "error" for result in results if result.patch.module_file == "_queue.py")
    _assert_launches(site, tmp_path)
