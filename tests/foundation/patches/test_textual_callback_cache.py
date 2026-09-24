# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the Textual callback arity cache patch."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import textual._callback as callback_mod
import textual.reactive as reactive_mod

from chrys.foundation.patches import patcher
from chrys.foundation.patches.patcher import FilePatch
from chrys.foundation.patches.textual_callback_cache import (
    _BUGGY_NEW,
    _NEW,
    _NEW_INVOKE,
    _OLD,
    _OLD_INVOKE,
    apply_runtime_patch,
)


@pytest.fixture(autouse=True)
def _reset_runtime_patch() -> Iterator[None]:
    """Force the genuine Textual callback functions back into place around each test.

    ``apply_runtime_patch`` installs its wrappers via direct module assignment
    (not ``monkeypatch``), so without an explicit teardown restore the patched
    ``count_parameters`` / ``_invoke`` would leak into other test files sharing
    the same xdist worker — there they would meet ``_param_count`` values cached
    by vanilla Textual and miscount bound-method arity. Capture the unwrapped
    originals and reinstate them both before and after the test.
    """
    original_count = getattr(callback_mod.count_parameters, "_original", callback_mod.count_parameters)
    original_invoke = getattr(callback_mod._invoke, "_original", callback_mod._invoke)
    original_reactive_count = reactive_mod.count_parameters
    callback_mod.count_parameters = original_count
    callback_mod._invoke = original_invoke
    reactive_mod.count_parameters = original_count
    try:
        yield
    finally:
        callback_mod.count_parameters = original_count
        callback_mod._invoke = original_invoke
        reactive_mod.count_parameters = original_reactive_count


def test_partial_callbacks_reuse_underlying_signature_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def count_raw(func: Any) -> int:
        nonlocal calls
        calls += 1
        return len(inspect.signature(func).parameters)

    def callback(first: object, second: object, third: object) -> None:
        pass

    monkeypatch.setattr(callback_mod, "_count_parameters", count_raw)
    apply_runtime_patch()

    assert callback_mod.count_parameters(partial(callback, "a")) == 2
    assert callback_mod.count_parameters(partial(callback, "b")) == 2
    assert calls == 1


def test_bound_methods_reuse_function_signature_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def count_raw(func: Any) -> int:
        nonlocal calls
        calls += 1
        return len(inspect.signature(func).parameters)

    class Handler:
        def callback(self, value: object) -> None:
            pass

    handler = Handler()
    monkeypatch.setattr(callback_mod, "_count_parameters", count_raw)
    apply_runtime_patch()

    assert callback_mod.count_parameters(handler.callback) == 1
    assert callback_mod.count_parameters(handler.callback) == 1
    assert calls == 1


def test_bound_method_cache_preserves_unbound_function_arity(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def count_raw(func: Any) -> int:
        nonlocal calls
        calls += 1
        return len(inspect.signature(func).parameters)

    class Handler:
        def callback(self, value: object) -> None:
            pass

    handler = Handler()
    monkeypatch.setattr(callback_mod, "_count_parameters", count_raw)
    apply_runtime_patch()

    assert callback_mod.count_parameters(handler.callback) == 1
    assert callback_mod.count_parameters(Handler.callback) == 2
    assert callback_mod.count_parameters(handler.callback) == 1
    assert calls == 1


def test_file_patch_upgrades_previous_callback_cache_fragment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "_callback.py"
    target.write_text(_BUGGY_NEW, encoding="utf-8")
    monkeypatch.setattr(patcher, "_locate_package_dir", lambda _package: tmp_path)

    results = patcher.apply_patch_group(
        [
            FilePatch(
                package="textual",
                module_file="_callback.py",
                old_fragment="upstream count_parameters",
                new_fragment=_NEW,
                description="fresh patch",
                equivalent_fragments=(_BUGGY_NEW,),
            ),
            FilePatch(
                package="textual",
                module_file="_callback.py",
                old_fragment=_BUGGY_NEW,
                new_fragment=_NEW,
                description="upgrade buggy patch",
            ),
        ]
    )

    assert [result.status for result in results] == ["skipped", "applied"]
    assert target.read_text(encoding="utf-8") == _NEW


def _registered_callback_patches() -> list[FilePatch]:
    """Return the live `textual/_callback.py` patches in registration order."""
    return [patch for patch in patcher._patches if (patch.package, patch.module_file) == ("textual", "_callback.py")]


def test_registered_callback_patches_transform_vanilla_fragments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real registered patches turn the vanilla `_OLD`/`_OLD_INVOKE` source into `_NEW`/`_NEW_INVOKE`.

    The existing upgrade test only exercises the `_BUGGY_NEW` -> `_NEW` path with a
    dummy `old_fragment`; this covers the genuine `_OLD` -> `_NEW` transform end to end.
    """
    target = tmp_path / "_callback.py"
    target.write_text(f"{_OLD}\n\n{_OLD_INVOKE}", encoding="utf-8")
    monkeypatch.setattr(patcher, "_locate_package_dir", lambda _package: tmp_path)

    patches = _registered_callback_patches()
    assert patches, "textual _callback.py patches are not registered"

    results = patcher.apply_patch_group(patches)

    statuses = [result.status for result in results]
    assert "error" not in statuses, statuses
    assert "applied" in statuses
    patched = target.read_text(encoding="utf-8")
    assert _NEW in patched
    assert _NEW_INVOKE in patched
    assert _OLD not in patched
    assert _OLD_INVOKE not in patched


def test_registered_callback_patches_match_installed_textual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registered patches must converge the *installed* Textual source to the patched state.

    This guards against a Textual bump silently invalidating the vanilla
    `_OLD`/`_OLD_INVOKE` fragments. If upstream drifts, the file patch degrades to a
    logged ``error`` and only the runtime patch survives; this turns that silent
    degradation into a hard test failure. The installed file may already be patched
    on disk (then every patch reports ``skipped``), so the assertion is convergence
    to `_NEW`/`_NEW_INVOKE` with no ``error``, not a specific status sequence.
    """
    import textual._callback as callback_mod

    target = tmp_path / "_callback.py"
    target.write_text(Path(callback_mod.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(patcher, "_locate_package_dir", lambda _package: tmp_path)

    patches = _registered_callback_patches()
    assert patches, "textual _callback.py patches are not registered"

    results = patcher.apply_patch_group(patches)

    errored = [(result.patch.description, result.detail) for result in results if result.status == "error"]
    assert not errored, f"installed Textual no longer matches the patch fragments — review the patch: {errored}"
    patched = target.read_text(encoding="utf-8")
    assert _NEW in patched
    assert _NEW_INVOKE in patched
    assert _OLD not in patched
    assert _OLD_INVOKE not in patched


async def test_invoke_skips_parameter_count_when_no_params(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_count(_func: Any) -> int:
        raise AssertionError("count_parameters should not run without supplied params")

    def callback() -> str:
        return "ok"

    apply_runtime_patch()
    monkeypatch.setattr(callback_mod, "count_parameters", fail_count)

    assert await callback_mod._invoke(callback) == "ok"


async def test_invoke_counts_parameters_when_params_are_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def count_one(_func: Any) -> int:
        nonlocal calls
        calls += 1
        return 1

    def callback(value: str) -> str:
        return value

    apply_runtime_patch()
    monkeypatch.setattr(callback_mod, "count_parameters", count_one)

    assert await callback_mod._invoke(callback, "kept", "dropped") == "kept"
    assert calls == 1
