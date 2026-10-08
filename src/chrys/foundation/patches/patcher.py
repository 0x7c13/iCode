# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Generic site-packages patcher.

Patches are defined as ``FilePatch`` objects — each specifies a package,
a module-relative file path, and an (old_fragment, new_fragment) pair.
The patcher locates the installed file, checks its content, and applies
or skips as appropriate.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import stat
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilePatch:
    """A single find-and-replace patch for an installed package file."""

    package: str
    """Top-level package name (e.g. ``some_package``)."""

    module_file: str
    """File path relative to the package root (e.g. ``_tools.py``)."""

    old_fragment: str
    """Exact code fragment expected in the original (unpatched) file."""

    new_fragment: str
    """Replacement code fragment."""

    description: str = ""
    """Human-readable description of what this patch does."""

    equivalent_fragments: tuple[str, ...] = ()
    """Already-patched fragments that are behaviorally equivalent to ``new_fragment``."""


@dataclass
class PatchResult:
    """Outcome of attempting to apply a single patch."""

    patch: FilePatch
    status: str  # "applied", "skipped", "error"
    detail: str = ""


# Module-level registry so patch definitions can self-register.
_patches: list[FilePatch] = []


def register(*patches: FilePatch) -> None:
    """Register one or more patches."""
    _patches.extend(patches)


def _locate_package_dir(package: str) -> Path:
    """Return the directory of an installed package."""
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise ImportError(f"Package {package!r} is not installed.")
    return Path(spec.origin).parent


def _atomic_write_text(path: Path, payload: str) -> None:
    """Atomically write text to an installed patch target.

    Writes via a binary file descriptor on the same filesystem so
    ``os.replace`` is genuinely atomic on POSIX and Windows.  Binary
    mode avoids ``\\r\\n`` translation under Windows and lets
    ``os.fsync`` accept the file descriptor uniformly.  The original
    file's mode bits are preserved so a stricter umask on the temp
    file doesn't leave the patched module unreadable.
    """
    target_mode = stat.S_IMODE(path.stat().st_mode)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, target_mode)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def _patch_target(patch: FilePatch) -> Path:
    """Resolve the installed file targeted by a patch."""
    return _locate_package_dir(patch.package) / patch.module_file


def _missing_fragment_detail(patch: FilePatch, target: Path) -> str:
    return (
        f"Source file does not contain the expected original fragment.\n"
        f"The upstream package may have been updated — please review and "
        f"update the patch.\n"
        f"File: {target}"
    )


def apply_patch(patch: FilePatch) -> PatchResult:
    """Apply a single patch, returning the result."""
    try:
        target = _patch_target(patch)
    except ImportError as exc:
        # A package that is not installed has nothing to patch. Every
        # entrypoint runs apply_all(), so "error" would log a warning per
        # patch on every invocation. A missing file inside an INSTALLED
        # package stays an error below — that one really is broken.
        return PatchResult(patch, "skipped", str(exc))

    if not target.is_file():
        return PatchResult(patch, "error", f"File not found: {target}")

    source = target.read_text(encoding="utf-8")

    if patch.new_fragment in source or any(fragment in source for fragment in patch.equivalent_fragments):
        return PatchResult(patch, "skipped", "Already patched.")

    if patch.old_fragment not in source:
        return PatchResult(patch, "error", _missing_fragment_detail(patch, target))

    patched = source.replace(patch.old_fragment, patch.new_fragment, 1)
    _atomic_write_text(target, patched)
    return PatchResult(patch, "applied", f"Patched {target}")


def apply_patch_group(patches: list[FilePatch]) -> list[PatchResult]:
    """Apply same-file patches all-or-nothing.

    The patcher often registers several fragments against one upstream
    module.  Staging them as a group prevents a package upgrade from
    leaving that module half-mutated if one later fragment no longer
    matches.
    """
    if not patches:
        return []

    first = patches[0]
    if any((patch.package, patch.module_file) != (first.package, first.module_file) for patch in patches):
        return [PatchResult(patch, "error", "Patch group contains multiple target files.") for patch in patches]

    try:
        target = _patch_target(first)
    except ImportError as exc:
        # A package that is not installed has nothing to patch — see apply_patch.
        return [PatchResult(patch, "skipped", str(exc)) for patch in patches]

    if not target.is_file():
        return [PatchResult(patch, "error", f"File not found: {target}") for patch in patches]

    source = target.read_text(encoding="utf-8")
    staged_source = source
    staged_results: list[PatchResult] = []

    for index, patch in enumerate(patches):
        if patch.new_fragment in staged_source or any(
            fragment in staged_source for fragment in patch.equivalent_fragments
        ):
            staged_results.append(PatchResult(patch, "skipped", "Already patched."))
            continue
        if patch.old_fragment not in staged_source:
            detail = _missing_fragment_detail(patch, target)
            aborted_detail = f"Patch group aborted before writing because another fragment failed.\n{detail}"
            return (
                [
                    result if result.status == "skipped" else PatchResult(result.patch, "error", aborted_detail)
                    for result in staged_results
                ]
                + [PatchResult(patch, "error", detail)]
                + [
                    PatchResult(remaining, "error", "Patch group aborted before this fragment was checked.")
                    for remaining in patches[index + 1 :]
                ]
            )
        staged_source = staged_source.replace(patch.old_fragment, patch.new_fragment, 1)
        staged_results.append(PatchResult(patch, "applied", f"Patched {target}"))

    if staged_source != source:
        _atomic_write_text(target, staged_source)
    return staged_results


def _apply_runtime_patches(runtime_patches: tuple[tuple[str, Callable[[], None]], ...]) -> None:
    """Apply independent in-process patches without making startup all-or-nothing."""
    for name, apply_runtime_patch in runtime_patches:
        try:
            apply_runtime_patch()
        except Exception as exc:
            logger.warning("Runtime patch error: %s — %s", name, exc, exc_info=True)


def apply_all() -> list[PatchResult]:
    """Apply all registered patches. Returns results for each."""
    # Import patch definition modules so they call ``register()``.
    import chrys.foundation.patches.mdurl_cache as mdurl_cache
    import chrys.foundation.patches.textual_block_border as textual_block_border
    import chrys.foundation.patches.textual_button_ansi as textual_button_ansi
    import chrys.foundation.patches.textual_callback_cache as textual_callback_cache
    import chrys.foundation.patches.textual_callback_dispatch as textual_callback_dispatch
    import chrys.foundation.patches.textual_compositor_cjk as textual_compositor_cjk
    import chrys.foundation.patches.textual_dispatch_cache as textual_dispatch_cache
    import chrys.foundation.patches.textual_ime_cursor_anchor as textual_ime_cursor_anchor
    import chrys.foundation.patches.textual_kitty_keyboard as textual_kitty_keyboard
    import chrys.foundation.patches.textual_lru_acyclic as textual_lru_acyclic
    import chrys.foundation.patches.textual_message_pump as textual_message_pump
    import chrys.foundation.patches.textual_node_diet as textual_node_diet
    import chrys.foundation.patches.textual_one_shot_timer as textual_one_shot_timer
    import chrys.foundation.patches.textual_option_list as textual_option_list
    import chrys.foundation.patches.textual_precompose as textual_precompose
    import chrys.foundation.patches.textual_pruned_tabs as textual_pruned_tabs
    import chrys.foundation.patches.textual_reflow_reuse as textual_reflow_reuse
    import chrys.foundation.patches.textual_removed_node_caches as textual_removed_node_caches
    import chrys.foundation.patches.textual_removed_screen_callbacks as textual_removed_screen_callbacks
    import chrys.foundation.patches.textual_selection_extract as textual_selection_extract
    import chrys.foundation.patches.textual_strip_cycles as textual_strip_cycles
    import chrys.foundation.patches.textual_tab_selection as textual_tab_selection
    import chrys.foundation.patches.textual_timer_skip as textual_timer_skip
    import chrys.foundation.patches.textual_utf8_decoder as textual_utf8_decoder
    import chrys.foundation.patches.textual_win_sleep as textual_win_sleep
    import chrys.foundation.patches.watchdog_fsevents as watchdog_fsevents
    import chrys.foundation.patches.watchdog_windows as watchdog_windows

    grouped: dict[tuple[str, str], list[FilePatch]] = defaultdict(list)
    for patch in _patches:
        grouped[(patch.package, patch.module_file)].append(patch)

    results: list[PatchResult] = []
    for patches in grouped.values():
        group_results = apply_patch_group(patches)
        for result in group_results:
            patch = result.patch
            if result.status == "applied":
                logger.info("Patch applied: %s — %s", patch.description, result.detail)
            elif result.status == "skipped":
                logger.debug("Patch skipped: %s — %s", patch.description, result.detail)
            else:
                logger.warning("Patch error: %s — %s", patch.description, result.detail)
        results.extend(group_results)

    _apply_runtime_patches(
        (
            ("mdurl_cache", mdurl_cache.apply_runtime_patch),
            ("watchdog_windows", watchdog_windows.apply_runtime_patch),
            ("watchdog_fsevents", watchdog_fsevents.apply_runtime_patch),
            ("textual_win_sleep", textual_win_sleep.apply_runtime_patch),
            ("textual_block_border", textual_block_border.apply_runtime_patch),
            ("textual_button_ansi", textual_button_ansi.apply_runtime_patch),
            ("textual_callback_cache", textual_callback_cache.apply_runtime_patch),
            ("textual_callback_dispatch", textual_callback_dispatch.apply_runtime_patch),
            ("textual_compositor_cjk", textual_compositor_cjk.apply_runtime_patch),
            ("textual_message_pump", textual_message_pump.apply_runtime_patch),
            ("textual_dispatch_cache", textual_dispatch_cache.apply_runtime_patch),
            ("textual_one_shot_timer", textual_one_shot_timer.apply_runtime_patch),
            ("textual_timer_skip", textual_timer_skip.apply_runtime_patch),
            ("textual_reflow_reuse", textual_reflow_reuse.apply_runtime_patch),
            ("textual_strip_cycles", textual_strip_cycles.apply_runtime_patch),
            ("textual_lru_acyclic", textual_lru_acyclic.apply_runtime_patch),
            ("textual_node_diet", textual_node_diet.apply_runtime_patch),
            ("textual_option_list", textual_option_list.apply_runtime_patch),
            ("textual_precompose", textual_precompose.apply_runtime_patch),
            ("textual_pruned_tabs", textual_pruned_tabs.apply_runtime_patch),
            ("textual_removed_node_caches", textual_removed_node_caches.apply_runtime_patch),
            ("textual_removed_screen_callbacks", textual_removed_screen_callbacks.apply_runtime_patch),
            ("textual_selection_extract", textual_selection_extract.apply_runtime_patch),
            ("textual_tab_selection", textual_tab_selection.apply_runtime_patch),
            ("textual_ime_cursor_anchor", textual_ime_cursor_anchor.apply_runtime_patch),
            ("textual_kitty_keyboard", textual_kitty_keyboard.apply_runtime_patch),
            ("textual_utf8_decoder", textual_utf8_decoder.apply_runtime_patch),
        )
    )
    return results
