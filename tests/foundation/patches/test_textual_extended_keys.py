# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Modified keys survive decoding and terminal modes are paired on shutdown."""

from __future__ import annotations

from io import StringIO

import pytest
from textual import constants
from textual._xterm_parser import XTermParser
from textual.events import Key

from chrys.foundation.patches import textual_extended_keys as patch
from chrys.foundation.patches.textual_kitty_keyboard import apply_runtime_patch as apply_kitty_patch
from chrys.foundation.platform import get_platform
from tests.support.keyboard_patches import isolated_keyboard_patches as isolated_keyboard_patches


@pytest.fixture(autouse=True)
def _terminal_identity(monkeypatch: pytest.MonkeyPatch, isolated_keyboard_patches: None) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "test-terminal")
    monkeypatch.setattr(constants, "DISABLE_KITTY_KEY", False)


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("\r", [("enter", "\r")]),
        ("\n", [("ctrl+j", "\n")]),
        ("\x1b[27;2;13~", [("shift+enter", None)]),
        ("\x1b[27;5;13~", [("ctrl+enter", None)]),
        ("\x1b[27;1;13~", [("enter", None)]),
        ("\x1b[27;5;106~", [("ctrl+j", None)]),
        ("\x1b[27;2;9~", [("shift+tab", None)]),
        ("\x1b[27;2;8~", [("backspace", None)]),
        ("\x1b[27;2;127~", [("backspace", None)]),
        ("\x1b[27;2;27~", [("escape", None)]),
        ("\x1b[27;6;127~", [("ctrl+shift+backspace", None)]),
        ("\x1b[27;6;27~", [("ctrl+shift+escape", None)]),
        ("\x1b[27;5;91~", [("escape", "\x1b")]),
        ("\x1b[27;3;8~", [("ctrl+w", None)]),
        ("\x1b[27;3;127~", [("ctrl+w", None)]),
        ("\x1b[27;5;104~", [("backspace", "\x08")]),
        ("\x1b[27;5;72~", [("backspace", "\x08")]),
        ("\x1b[27;5;105~", [("tab", "\t")]),
        ("\x1b[27;5;73~", [("tab", "\t")]),
        ("\x1b[27;5;109~", [("enter", "\r")]),
        ("\x1b[27;5;77~", [("enter", "\r")]),
        ("\x1b[27;6;72~", [("ctrl+shift+h", None)]),
        ("\x1b[27;7;104~", [("alt+ctrl+h", None)]),
        ("\x1b[27;4;127~", [("alt+shift+backspace", None)]),
        ("\x1b[27;3;98~", [("ctrl+left", None)]),
        ("\x1b[27;3;102~", [("ctrl+right", None)]),
        ("\x1b[27;3;66~", [("ctrl+left", None)]),
        ("\x1b[27;4;98~", [("alt+shift+b", None)]),
        ("\x1b[27;7;102~", [("alt+ctrl+f", None)]),
        ("\x1b[27;7;91~", [("alt+ctrl+left_square_bracket", None)]),
        ("\x1b[27;6;95~", [("ctrl+underscore", None)]),
        ("\x1b[27;6;50~", [("ctrl+shift+2", None)]),
        ("\x1b[27;3;13~", [("alt+enter", None)]),
        ("\x1b[27;9;13~", [("meta+enter", None)]),
        ("\x1b[27;6;86~", [("ctrl+shift+v", None)]),
        ("\x1b[27;5;65~", [("ctrl+a", None)]),
        ("\x1b[27;2;65~", [("A", "A")]),
        ("\x1b[27;2;33~", [("exclamation_mark", "!")]),
        ("\x1b[27;2;32~", [("shift+space", " ")]),
        ("\x1b[27;1;24403~", [("当", "当")]),
        ("\x1b[27;0;13~", []),
        ("\x1b[27;17;13~", []),
        ("\x1b[27;2;1114112~", []),
        ("\x1b[27;2;55296~", []),
        ("\x1b[13;2u", [("shift+enter", None)]),
        ("\x1b[13;2;13u", [("shift+enter", "\r")]),
        ("\x1b[97;1:3;97u", []),
        ("\x1b[1;2A", [("shift+up", None)]),
    ],
)
def test_production_streaming_parser(sequence: str, expected: list) -> None:
    apply_kitty_patch()
    patch.apply_runtime_patch()
    parser = XTermParser()
    keys = [event for char in sequence for event in parser.feed(char) if isinstance(event, Key)]
    assert [(event.key, event.character) for event in keys] == expected


def test_parser_patch_is_idempotent() -> None:
    patch.apply_runtime_patch()
    first = XTermParser._sequence_to_key_events
    patch.apply_runtime_patch()
    assert XTermParser._sequence_to_key_events is first


def test_version_guard_precedes_private_access(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    import textual
    import textual._xterm_parser as parser_module

    monkeypatch.setattr(textual, "__version__", "9.0.0")
    monkeypatch.delattr(parser_module, "XTermParser")
    patch.apply_runtime_patch()
    assert "unsupported Textual 9.0.0" in caplog.text


class _Writer(StringIO):
    stopped = False

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture(params=("fullscreen", "inline"))
def driver(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if get_platform().is_windows:
        pytest.skip("POSIX driver")
    from textual.drivers.linux_driver import LinuxDriver
    from textual.drivers.linux_inline_driver import LinuxInlineDriver

    cls = LinuxDriver if request.param == "fullscreen" else LinuxInlineDriver
    patch.apply_runtime_patch()
    instance = object.__new__(cls)
    writer = _Writer()
    instance._file = writer
    instance._writer_thread = writer
    instance._app = None
    instance._in_band_window_resize = False
    instance.attrs_before = None
    instance._mouse = False
    monkeypatch.setattr(instance, "disable_input", lambda: None)
    try:
        yield instance, writer
    finally:
        instance.close()


def test_real_driver_pairs_modes_on_stop_and_resume(driver) -> None:
    instance, writer = driver
    for _ in range(2):
        instance.write("\x1b[>25u")
        instance.write("frame")
        instance.stop_application_mode()
    output = writer.getvalue()
    assert output.count("\x1b[>4;2m\x1b[>25u") == 2
    assert output.count("\x1b[>4;0m\x1b[<u") == 2
    assert output.count("frame") == 2
    instance.close()
    assert writer.getvalue() == output


def test_driver_close_resets_after_partial_startup(driver) -> None:
    instance, writer = driver
    instance.write("\x1b[>1u")
    instance.close()
    assert writer.getvalue() == "\x1b[>4;2m\x1b[>1u\x1b[>4;0m"
    instance.close()
    assert writer.getvalue().count("\x1b[>4;0m") == 1


def test_driver_respects_keyboard_protocol_opt_out(driver, monkeypatch) -> None:
    instance, writer = driver
    monkeypatch.setattr(constants, "DISABLE_KITTY_KEY", True)
    instance.write("\x1b[>25u")
    instance.write("\x1b[<u")
    assert writer.getvalue() == "\x1b[>25u\x1b[<u"


@pytest.mark.parametrize("terminal", ["Apple_Terminal", None])
def test_driver_request_does_not_depend_on_local_terminal_identity(driver, monkeypatch, terminal) -> None:
    instance, writer = driver
    if terminal is None:
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
    else:
        monkeypatch.setenv("TERM_PROGRAM", terminal)
    instance.write("\x1b[>25u")
    instance.write("\x1b[<u")
    assert writer.getvalue() == "\x1b[>4;2m\x1b[>25u\x1b[>4;0m\x1b[<u"


def test_driver_patch_is_idempotent(driver) -> None:
    instance, writer = driver
    first = type(instance).write
    patch.apply_runtime_patch()
    assert type(instance).write is first
    instance.write("\x1b[>1u")
    assert writer.getvalue() == "\x1b[>4;2m\x1b[>1u"
