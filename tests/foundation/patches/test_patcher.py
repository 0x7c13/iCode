# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for site-packages patch staging."""

from __future__ import annotations

import stat
import sys
from typing import TYPE_CHECKING

import pytest

from chrys.foundation.patches import patcher
from chrys.foundation.patches.patcher import FilePatch

if TYPE_CHECKING:
    from pathlib import Path


def _patch(old: str, new: str, description: str) -> FilePatch:
    return FilePatch(
        package="pkg",
        module_file="target.py",
        old_fragment=old,
        new_fragment=new,
        description=description,
    )


def test_absent_package_is_skipped_not_errored(monkeypatch: pytest.MonkeyPatch) -> None:
    """An install without the ``tui`` extra has no textual, and that is normal.

    Every headless entrypoint runs ``apply_all()``, so classifying "package is
    not installed" as an error made ``chrys run``/``chrys acp`` log a warning
    per registered patch on every single invocation.
    """

    def _absent(package: str) -> Path:
        raise ImportError(f"Package {package!r} is not installed.")

    monkeypatch.setattr(patcher, "_locate_package_dir", _absent)

    single = patcher.apply_patch(_patch("a = 'old'", "a = 'new'", "single"))
    group = patcher.apply_patch_group(
        [
            _patch("a = 'old'", "a = 'new'", "first"),
            _patch("b = 'old'", "b = 'new'", "second"),
        ]
    )

    assert single.status == "skipped", single.detail
    assert [result.status for result in group] == ["skipped", "skipped"]


def test_missing_file_in_installed_package_is_still_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The absent-package carve-out must not swallow a genuinely broken target."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()  # installed, but target.py was never created

    monkeypatch.setattr(patcher, "_locate_package_dir", lambda _package: pkg)

    single = patcher.apply_patch(_patch("a = 'old'", "a = 'new'", "single"))
    group = patcher.apply_patch_group([_patch("a = 'old'", "a = 'new'", "first")])

    assert single.status == "error", single.detail
    assert [result.status for result in group] == ["error"]


def test_patch_group_does_not_write_partial_file_on_later_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "target.py"
    original = "first = 'old'\nsecond = 'old'\n"
    target.write_text(original, encoding="utf-8")

    monkeypatch.setattr(patcher, "_locate_package_dir", lambda _package: pkg)

    results = patcher.apply_patch_group(
        [
            _patch("first = 'old'", "first = 'new'", "first"),
            _patch("missing = 'old'", "missing = 'new'", "missing"),
        ]
    )

    assert [result.status for result in results] == ["error", "error"]
    assert target.read_text(encoding="utf-8") == original


def test_patch_group_writes_once_after_all_fragments_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "target.py"
    target.write_text("first = 'old'\nsecond = 'old'\n", encoding="utf-8")

    monkeypatch.setattr(patcher, "_locate_package_dir", lambda _package: pkg)

    results = patcher.apply_patch_group(
        [
            _patch("first = 'old'", "first = 'new'", "first"),
            _patch("second = 'old'", "second = 'new'", "second"),
        ]
    )

    assert [result.status for result in results] == ["applied", "applied"]
    assert target.read_text(encoding="utf-8") == "first = 'new'\nsecond = 'new'\n"


def test_patch_group_skips_equivalent_fragments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "target.py"
    target.write_text("first = 'legacy-new'\n", encoding="utf-8")

    monkeypatch.setattr(patcher, "_locate_package_dir", lambda _package: pkg)

    results = patcher.apply_patch_group(
        [
            FilePatch(
                package="pkg",
                module_file="target.py",
                old_fragment="first = 'old'",
                new_fragment="first = 'new'",
                description="first",
                equivalent_fragments=("first = 'legacy-new'",),
            )
        ]
    )

    assert [result.status for result in results] == ["skipped"]
    assert target.read_text(encoding="utf-8") == "first = 'legacy-new'\n"


def test_patch_group_can_upgrade_equivalent_fragment_in_later_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "target.py"
    target.write_text("first = 'legacy-new'\n", encoding="utf-8")

    monkeypatch.setattr(patcher, "_locate_package_dir", lambda _package: pkg)

    results = patcher.apply_patch_group(
        [
            FilePatch(
                package="pkg",
                module_file="target.py",
                old_fragment="first = 'old'",
                new_fragment="first = 'new'",
                description="first",
                equivalent_fragments=("first = 'legacy-new'",),
            ),
            _patch("first = 'legacy-new'", "first = 'new'", "upgrade first"),
        ]
    )

    assert [result.status for result in results] == ["skipped", "applied"]
    assert target.read_text(encoding="utf-8") == "first = 'new'\n"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows os.chmod only honours the read-only bit; POSIX modes are not preserved.",
)
def test_patch_group_preserves_target_file_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "target.py"
    target.write_text("first = 'old'\n", encoding="utf-8")
    target.chmod(0o644)

    monkeypatch.setattr(patcher, "_locate_package_dir", lambda _package: pkg)

    results = patcher.apply_patch_group([_patch("first = 'old'", "first = 'new'", "first")])

    assert [result.status for result in results] == ["applied"]
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_apply_all_isolates_runtime_patch_failures(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One broken runtime patch must not block startup or later patches."""
    from chrys.foundation.patches import (
        textual_block_border,
        textual_button_ansi,
        textual_callback_cache,
        textual_callback_dispatch,
        textual_compositor_cjk,
        textual_dispatch_cache,
        textual_ime_cursor_anchor,
        textual_kitty_keyboard,
        textual_message_pump,
        textual_option_list,
        textual_precompose,
        textual_selection_extract,
        textual_tab_selection,
        textual_utf8_decoder,
        textual_win_sleep,
        watchdog_fsevents,
        watchdog_windows,
    )

    calls: list[str] = []

    def broken_compositor_patch() -> None:
        calls.append("compositor_cjk")
        raise AttributeError("upstream private API moved")

    monkeypatch.setattr(patcher, "apply_patch_group", lambda _patches: [])
    monkeypatch.setattr(watchdog_windows, "apply_runtime_patch", lambda: calls.append("watchdog_windows"))
    monkeypatch.setattr(watchdog_fsevents, "apply_runtime_patch", lambda: calls.append("watchdog_fsevents"))
    monkeypatch.setattr(textual_win_sleep, "apply_runtime_patch", lambda: calls.append("win_sleep"))
    monkeypatch.setattr(textual_block_border, "apply_runtime_patch", lambda: calls.append("block_border"))
    monkeypatch.setattr(textual_button_ansi, "apply_runtime_patch", lambda: calls.append("button"))
    monkeypatch.setattr(textual_callback_cache, "apply_runtime_patch", lambda: calls.append("callback_cache"))
    monkeypatch.setattr(
        textual_callback_dispatch,
        "apply_runtime_patch",
        lambda: calls.append("callback_dispatch"),
    )
    monkeypatch.setattr(textual_compositor_cjk, "apply_runtime_patch", broken_compositor_patch)
    monkeypatch.setattr(textual_message_pump, "apply_runtime_patch", lambda: calls.append("message_pump"))
    monkeypatch.setattr(textual_dispatch_cache, "apply_runtime_patch", lambda: calls.append("dispatch_cache"))
    monkeypatch.setattr(textual_option_list, "apply_runtime_patch", lambda: calls.append("option_list"))
    monkeypatch.setattr(textual_precompose, "apply_runtime_patch", lambda: calls.append("precompose"))
    monkeypatch.setattr(textual_selection_extract, "apply_runtime_patch", lambda: calls.append("selection_extract"))
    monkeypatch.setattr(textual_tab_selection, "apply_runtime_patch", lambda: calls.append("tab_selection"))
    monkeypatch.setattr(textual_ime_cursor_anchor, "apply_runtime_patch", lambda: calls.append("ime_cursor_anchor"))
    monkeypatch.setattr(textual_kitty_keyboard, "apply_runtime_patch", lambda: calls.append("kitty_keyboard"))
    monkeypatch.setattr(textual_utf8_decoder, "apply_runtime_patch", lambda: calls.append("utf8_decoder"))

    patcher.apply_all()

    assert calls == [
        "watchdog_windows",
        "watchdog_fsevents",
        "win_sleep",
        "block_border",
        "button",
        "callback_cache",
        "callback_dispatch",
        "compositor_cjk",
        "message_pump",
        "dispatch_cache",
        "option_list",
        "precompose",
        "selection_extract",
        "tab_selection",
        "ime_cursor_anchor",
        "kitty_keyboard",
        "utf8_decoder",
    ]
    assert "Runtime patch error: textual_compositor_cjk — upstream private API moved" in caplog.text
    assert any(record.exc_info is not None for record in caplog.records)
