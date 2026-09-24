# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Already-captured input methods must decode invalid terminal bytes safely."""

from __future__ import annotations

import ast
import codecs
import importlib
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from textual.events import Key

from chrys.foundation.patches import textual_utf8_decoder as patch
from chrys.foundation.platform import get_platform


class _InputReader:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            yield b"D" + len(chunk).to_bytes(4, "big") + chunk

    def close(self) -> None:
        self.closed = True


class _Selector:
    def __init__(self, finished: Event) -> None:
        self.finished = finished
        self.closed = False

    def register(self, fileno, events):
        assert fileno == 42 and events == 1

    def unregister(self, fileno):
        assert fileno == 42

    def select(self, timeout):
        return [] if self.finished.is_set() else [(None, 1)]

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("web_driver", "WebDriver"),
        pytest.param(
            "linux_driver",
            "LinuxDriver",
            marks=pytest.mark.skipif(get_platform().is_windows, reason="POSIX driver"),
        ),
        pytest.param(
            "linux_inline_driver",
            "LinuxInlineDriver",
            marks=pytest.mark.skipif(get_platform().is_windows, reason="POSIX driver"),
        ),
    ],
)
def test_prebound_input_method_survives_invalid_utf8(monkeypatch, module_name, class_name):
    module = importlib.import_module(f"textual.drivers.{module_name}")
    driver_class = getattr(module, class_name)
    source = Path(module.__file__).read_text(encoding="utf-8").replace(patch._NEW, patch._OLD, 1)
    assert patch._OLD in source
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run_input_thread")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<stock Textual input>", "exec"), vars(module), namespace)
    monkeypatch.setattr(driver_class, "run_input_thread", namespace["run_input_thread"])
    for name in patch._DRIVER_MODULES:
        loaded = sys.modules.get(name)
        if loaded is not None:
            monkeypatch.setattr(loaded, "getincrementaldecoder", loaded.getincrementaldecoder)
    monkeypatch.setattr(module, "getincrementaldecoder", codecs.getincrementaldecoder)

    # Avoid terminal descriptors / background threads. Exercise the actual old
    # input method with owned, finite byte sources at its native I/O boundary.
    driver = object.__new__(driver_class)
    driver._debug = False
    messages = []
    driver.process_message = messages.append
    # Keep decoded data nonempty in the first packet: WebDriver treats an
    # empty parser feed as EOF, independently of its UTF-8 error policy.
    chunks = [b"A\xe4", b"\xbd\xa0\xffz"]
    if module_name == "web_driver":
        source_owner = _InputReader(chunks)
        driver._input_reader = source_owner
    else:
        driver.fileno = 42
        driver.exit_event = Event()
        source_owner = _Selector(driver.exit_event)
        monkeypatch.setattr(module, "selectors", SimpleNamespace(SelectSelector=lambda: source_owner, EVENT_READ=1))
        remaining = iter(chunks)

        def read(fileno, count):
            assert fileno == 42
            chunk = next(remaining)
            if chunk == chunks[-1]:
                driver.exit_event.set()
            return chunk

        monkeypatch.setattr(module, "os", SimpleNamespace(read=read))

    run = driver.run_input_thread
    patch.apply_runtime_patch()
    run()

    assert "".join(message.character or "" for message in messages if isinstance(message, Key)) == "A你\ufffdz"
    assert source_owner.closed
    installed = module.getincrementaldecoder
    patch.apply_runtime_patch()
    assert module.getincrementaldecoder is installed
