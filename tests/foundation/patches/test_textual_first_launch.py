# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""File patch success must also repair Textual consumers already in memory."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from chrys.foundation.patches import textual_block_border, textual_callback_cache, textual_utf8_decoder


@pytest.mark.parametrize("scenario", ["block-border", "reactive-callback", "utf8-drivers"])
def test_patches_reach_preimported_consumers(tmp_path: Path, scenario: str) -> None:
    spec = importlib.util.find_spec("textual")
    assert spec is not None and spec.origin is not None
    package = tmp_path / "textual"
    shutil.copytree(Path(spec.origin).parent, package, ignore=shutil.ignore_patterns("__pycache__"))
    reversals = {
        "_styles_cache.py": [(textual_block_border._NEW, textual_block_border._OLD)],
        "_callback.py": [
            (textual_callback_cache._NEW_INVOKE, textual_callback_cache._OLD_INVOKE),
            (textual_callback_cache._NEW, textual_callback_cache._OLD),
            (textual_callback_cache._BUGGY_NEW, textual_callback_cache._OLD),
        ],
        **{
            f"drivers/{name}.py": [(textual_utf8_decoder._NEW, textual_utf8_decoder._OLD)]
            for name in ("linux_driver", "linux_inline_driver", "web_driver")
        },
    }
    for relative, replacements in reversals.items():
        target = package / relative
        source = target.read_text(encoding="utf-8")
        for new, old in replacements:
            source = source.replace(new, old, 1)
        target.write_text(source, encoding="utf-8")

    script = """
import codecs
import importlib
import sys
from contextlib import nullcontext
from functools import partial

sys.path.insert(0, sys.argv[1])
from chrys.app.tui.app import ChrysApp
from chrys.foundation.patches import apply_all
from chrys.foundation.platform import get_platform
import textual._callback as callback
import textual.reactive as reactive
import textual.widget as widget
from textual._styles_cache import StylesCache
from textual.color import Color

drivers = [importlib.import_module("textual.drivers.web_driver")]
if not get_platform().is_windows:
    drivers.extend(importlib.import_module(f"textual.drivers.{name}") for name in ("linux_driver", "linux_inline_driver"))
cache = StylesCache()
base, background = Color.parse("ansi_default"), Color.parse("#123456")
assert cache.get_inner_outer(base, background)[1].background is not None
assert reactive.count_parameters is callback.count_parameters
raw_count = callback._count_parameters
count_calls = []
def count_parameters(func):
    count_calls.append(func)
    return raw_count(func)
callback._count_parameters = count_parameters
results = apply_all()
assert not [result for result in results if result.status == "error"], results

if sys.argv[2] == "block-border":
    assert widget.StylesCache is StylesCache
    inner, outer = cache.get_inner_outer(base, background)
    assert inner.background == base + background
    assert outer.background is None, outer
    assert cache.get_inner_outer(Color(0, 0, 0, 0), background)[1].background is None
    assert cache.get_inner_outer(background, background)[1].background == background
elif sys.argv[2] == "reactive-callback":
    class Owner:
        def _context(self):
            return nullcontext()
    values = []
    def watch(prefix, value):
        values.append((prefix, value))
    owner = Owner()
    for value in (1, 2):
        reactive.invoke_watcher(owner, partial(watch, "value"), value - 1, value)
    assert values == [("value", 1), ("value", 2)], values
    assert count_calls == [watch], count_calls
    assert reactive.count_parameters is callback.count_parameters
else:
    for driver in drivers:
        decoder = driver.getincrementaldecoder("utf-8")()
        assert decoder.decode(b"\\xe4") == ""
        assert decoder.decode(b"\\xbd\\xa0\\xffz") == "你\\ufffdz"
        # The driver-local default must not weaken codecs or explicit policies.
        for factory in (codecs.getincrementaldecoder, driver.getincrementaldecoder):
            try:
                factory("utf-8")(errors="strict").decode(b"\\xff")
            except UnicodeDecodeError:
                pass
            else:
                raise AssertionError("explicit strict UTF-8 decoding was changed")
        assert driver.getincrementaldecoder("ascii") is codecs.getincrementaldecoder("ascii")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", "-c", script, str(tmp_path), scenario],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
