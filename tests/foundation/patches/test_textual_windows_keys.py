# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Exercise lossless Windows keyboard input through the pinned Textual driver."""

from __future__ import annotations

import ctypes
import dataclasses
import importlib.util
import sys
from io import StringIO
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import textual.drivers
from textual import constants, events, messages
from textual._xterm_parser import XTermParser

from chrys.foundation.patches import textual_extended_keys as patch
from chrys.foundation.patches import textual_windows_keys as windows_keys
from chrys.foundation.patches.textual_kitty_keyboard import apply_runtime_patch
from chrys.foundation.platform import get_platform
from tests.support.keyboard_patches import isolated_keyboard_patches as isolated_keyboard_patches


@pytest.fixture(autouse=True)
def _keyboard_patch(monkeypatch: pytest.MonkeyPatch, isolated_keyboard_patches: None) -> None:
    apply_runtime_patch()
    patch.apply_runtime_patch()
    monkeypatch.setattr(constants, "DISABLE_KITTY_KEY", False)


def _keys(parser, data: str) -> list[tuple[str, str | None]]:
    return [(event.key, event.character) for event in parser.feed(data) if isinstance(event, events.Key)]


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ("13;28;13;1;16;1", [("shift+enter", None)]),
        ("13;28;13;1;272;1", [("shift+enter", None)]),  # Numpad Enter
        ("13;28;13;1;48;2", [("shift+enter", None)] * 2),  # NumLock and repeat
        ("13;28;13;1;0;1", [("enter", "\r")]),
        ("13;28;10;1;8;1", [("ctrl+enter", None)]),
        ("13;28;13;1;2;1", [("alt+enter", None)]),
        ("74;36;10;1;8;1", [("ctrl+j", None)]),
        ("65;30;1;1;8;1", [("ctrl+a", None)]),
        ("86;47;22;1;24;1", [("ctrl+shift+v", None)]),
        ("65;30;65;1;16;1", [("A", "A")]),
        ("65;30;65;1;128;1", [("A", "A")]),  # CapsLock
        ("49;2;33;1;16;1", [("exclamation_mark", "!")]),
        ("81;16;64;1;9;1", [("at", "@")]),  # AltGr
        ("221;0;0;1;9;1", []),  # AltGr dead key has not committed a character
        ("51;4;0;1;9;1", []),  # AltGr dead key on a digit
        ("65;30;0;1;9;1", []),  # AltGr dead key on a letter
        ("51;4;0;1;10;1", [("alt+ctrl+3", None)]),  # Left Ctrl + Left Alt
        ("65;30;0;1;10;1", [("alt+ctrl+a", None)]),
        ("32;57;32;1;16;1", [("shift+space", " ")]),
        ("50;3;0;1;8;1", [("ctrl+2", None)]),
        ("0;0;24403;1;0;1", [("当", "当")]),
        ("231;0;24403;1;16;1", [("当", "当")]),  # VK_PACKET
        ("18;56;233;0;32;1", [("é", "é")]),  # Alt+numpad commit
        ("97;79;49;1;2;1", []),  # Alt+numpad digit
        ("37;75;0;1;24;1", [("ctrl+shift+left", None)]),
        ("38;72;0;1;0;1", [("up", None)]),
        ("38;72;0;1;9;1", [("alt+ctrl+up", None)]),  # Functional key, not composition
        ("39;77;0;1;0;1", [("right", None)]),
        ("40;80;0;1;0;1", [("down", None)]),
        ("36;71;0;1;0;1", [("home", None)]),
        ("35;79;0;1;0;1", [("end", None)]),
        ("33;73;0;1;0;1", [("pageup", None)]),
        ("34;81;0;1;0;1", [("pagedown", None)]),
        ("45;82;0;1;0;1", [("insert", None)]),
        ("46;83;0;1;0;1", [("delete", None)]),
        ("8;14;8;1;0;1", [("backspace", "\x7f")]),
        ("8;14;8;1;16;1", [("backspace", None)]),
        ("9;15;9;1;0;1", [("tab", "\t")]),
        ("9;15;9;1;16;1", [("shift+tab", None)]),
        ("27;1;27;1;0;1", [("escape", None)]),
        ("27;1;27;1;16;1", [("escape", None)]),
        ("112;59;0;1;8;1", [("ctrl+f1", None)]),
        ("123;88;0;1;0;1", [("f12", None)]),
        ("135;0;0;1;0;1", [("f24", None)]),
        ("32;57;0;1;8;1", [("ctrl+space", None)]),
        ("219;26;27;1;8;1", [("escape", "\x1b")]),
        ("8;14;8;1;2;1", [("ctrl+w", None)]),
        ("72;35;8;1;8;1", [("backspace", "\x08")]),
        ("73;23;9;1;8;1", [("tab", "\t")]),
        ("77;50;13;1;8;1", [("enter", "\r")]),
        ("72;35;8;1;24;1", [("ctrl+shift+h", None)]),
        ("66;48;98;1;2;1", [("alt+b", None)]),
        ("70;33;102;1;2;1", [("alt+f", None)]),
        ("189;12;31;1;24;1", [("ctrl+underscore", None)]),
        ("13;28;13;1;16", [("shift+enter", None)]),  # Omitted repeat
        ("13;28;13;1;16;", [("shift+enter", None)]),
        ("13;28;13;0;16;1", []),
        ("16;42;0;1;16;1", []),
        ("13;28", []),  # Omitted key-down defaults to released
        ("13;28;13;1;16;0", []),
        ("13;28;13;2;16;1", []),
        ("0;0;65536;1;0;1", []),
        ("0;0;97;1;0;65536", []),
        ("13;28;13;1;16;1;2", []),
    ],
)
def test_native_records_across_read_boundaries(record: str, expected: list) -> None:
    parser = windows_keys.get_parser_class()()
    assert [key for char in f"\x1b[{record}_" for key in _keys(parser, char)] == expected


@pytest.mark.parametrize("digit", range(10))
def test_ctrl_digits_without_characters_remain_shortcuts(digit: int) -> None:
    parser = windows_keys.get_parser_class()()
    assert _keys(parser, f"\x1b[{48 + digit};0;0;1;8;1_") == [(f"ctrl+{digit}", None)]


@pytest.mark.parametrize(
    ("record", "mapped", "expected"),
    [
        ("191;53;0;1;8;1", ord("/"), [("ctrl+slash", None)]),
        ("191;53;0;1;8;1", ord(";"), [("ctrl+semicolon", None)]),  # Different layout
        ("191;53;0;1;2;1", ord("/"), [("alt+slash", None)]),
        ("191;53;0;1;10;1", ord("/"), [("alt+ctrl+slash", None)]),
        ("192;41;0;1;8;1", ord("`"), [("ctrl+grave_accent", None)]),
        ("221;0;0;1;8;1", 0x8000005E, [("ctrl+circumflex_accent", None)]),
        ("191;53;0;1;8;1", 0, []),  # No mapping for this layout
    ],
)
def test_missing_punctuation_uses_native_keyboard_layout(record, mapped, expected, monkeypatch) -> None:
    mapper = Mock(return_value=mapped)
    monkeypatch.setattr(windows_keys, "_load_virtual_key_mapper", lambda: mapper)
    parser = windows_keys.get_parser_class()()
    assert [key for char in f"\x1b[{record}_" for key in _keys(parser, char)] == expected
    mapper.assert_called_once_with(int(record.split(";")[0]), 2)


@pytest.mark.parametrize("virtual_key", [51, 65, 221])
@pytest.mark.parametrize("state", [0x09, 0x19, 0x89])  # AltGr, Shift+AltGr, CapsLock+AltGr
def test_altgr_dead_key_does_not_consult_layout_or_emit_a_shortcut(monkeypatch, virtual_key, state) -> None:
    mapper = Mock(return_value=ord("^"))
    monkeypatch.setattr(windows_keys, "_load_virtual_key_mapper", lambda: mapper)
    parser = windows_keys.get_parser_class()()
    assert _keys(parser, f"\x1b[{virtual_key};0;0;1;{state};1_") == []
    mapper.assert_not_called()
    assert _keys(parser, "\x1b[65;30;226;1;0;1_") == [("â", "â")]


def test_layout_lookup_does_not_cache_character_results(monkeypatch) -> None:
    mapper = Mock(side_effect=[ord("/"), ord(";")])
    monkeypatch.setattr(windows_keys, "_load_virtual_key_mapper", lambda: mapper)
    parser = windows_keys.get_parser_class()()
    sequence = "\x1b[191;53;0;1;8;1_"
    assert _keys(parser, sequence) == [("ctrl+slash", None)]
    assert _keys(parser, sequence) == [("ctrl+semicolon", None)]


def test_native_layout_api_is_loaded_with_unsigned_word_arguments(monkeypatch) -> None:
    mapper = Mock()
    loader = Mock(return_value=SimpleNamespace(MapVirtualKeyW=mapper))
    monkeypatch.setattr(windows_keys, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(ctypes, "WinDLL", loader, raising=False)
    # Bypass the process cache so the fake native function cannot escape this test.
    assert windows_keys._load_virtual_key_mapper.__wrapped__() is mapper
    loader.assert_called_once_with("user32")
    assert mapper.argtypes == [ctypes.c_uint, ctypes.c_uint]
    assert mapper.restype is ctypes.c_uint


@pytest.mark.parametrize("failure", ["library", "symbol"])
def test_native_layout_lookup_failure_does_not_stop_keyboard_input(monkeypatch, failure) -> None:
    loader = Mock(side_effect=OSError("unavailable")) if failure == "library" else Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(windows_keys, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(ctypes, "WinDLL", loader, raising=False)
    # Keep the failing lookup local, rather than caching it for later tests.
    monkeypatch.setattr(windows_keys, "_load_virtual_key_mapper", windows_keys._load_virtual_key_mapper.__wrapped__)
    parser = windows_keys.get_parser_class()()
    assert _keys(parser, "\x1b[191;53;0;1;8;1_\x1b[65;30;97;1;0;1_\x1b[65;30;1;1;8;1_") == [
        ("a", "a"),
        ("ctrl+a", None),
    ]
    loader.assert_called_once_with("user32")


def _wrapped(text: str) -> str:
    return "".join(f"\x1b[0;0;{ord(char)};1;0;1_" for char in text)


@pytest.mark.parametrize("native_letters", [False, True])
@pytest.mark.parametrize("wrapped_markers", [False, True])
@pytest.mark.parametrize("separator", ["\r", "\r\t"])
def test_bracketed_paste_preserves_native_return_and_tab_records(native_letters, wrapped_markers, separator) -> None:
    parser = windows_keys.get_parser_class()()
    start, end = "\x1b[200~", "\x1b[201~"
    if wrapped_markers:
        start, end = _wrapped(start), _wrapped(end)
    first = "\x1b[65;30;97;1;0;1_" if native_letters else _wrapped("a")
    last = "\x1b[66;48;98;1;0;1_" if native_letters else _wrapped("b")
    # Terminal/ConPTY mixes text with native controls; conhost can encode
    # every pasted character as a key record, including key-up events.
    controls = "\x1b[13;28;13;1;0;1_\x1b[13;28;13;0;0;1_"
    if "\t" in separator:
        controls += "\x1b[9;15;9;1;0;1_\x1b[9;15;9;0;0;1_"
    parsed = [event for char in start + first + controls + last + end for event in parser.feed(char)]
    assert len(parsed) == 1
    assert isinstance(parsed[0], events.Paste)
    assert parsed[0].text == f"a{separator}b"


@pytest.mark.parametrize("wrapped", [False, True])
def test_paste_mouse_focus_and_terminal_replies_remain_vt(wrapped: bool) -> None:
    parser = windows_keys.get_parser_class()()
    text = "\x1b[200~a\n当前😀\nb\x1b[201~\x1b[<0;4;5M\x1b[I\x1b[?2026;1$y\x1b[13;2u"
    if wrapped:
        # Native records contain UTF-16 code units, not astral codepoints.
        text = text.replace("😀", "\ud83d\ude00")
        text = _wrapped(text)
    parsed = [event for char in text for event in parser.feed(char)]
    assert isinstance(parsed[0], events.Paste)
    assert parsed[0].text == "a\n当前😀\nb"
    assert isinstance(parsed[1], events.MouseDown)
    assert (parsed[1].x, parsed[1].y) == (3, 4)
    assert isinstance(parsed[2], events.AppFocus)
    assert isinstance(parsed[3], messages.TerminalSupportsSynchronizedOutput)
    assert [(event.key, event.character) for event in parsed[4:]] == [("shift+enter", None)]


@pytest.mark.parametrize("sequence", ["\r", "\n", "\x1b[27;5;9~", "\x1b[32;;24403:21069u", "\x1b[97;1:3;97u"])
def test_windows_adapter_delegates_other_keyboard_protocols(sequence: str) -> None:
    parser = windows_keys.get_parser_class()()
    assert [key for char in sequence for key in _keys(parser, char)] == _keys(XTermParser(), sequence)


def test_surrogate_pair_survives_key_up_and_input_read_boundaries() -> None:
    parser = windows_keys.get_parser_class()()
    assert _keys(parser, "\x1b[231;0;55357;1;0;1_\x1b[231;0;55357;0;0;1_") == []
    assert _keys(parser, "\x1b[231;0;56832;1;0;1_") == [("grinning_face", "😀")]
    assert _keys(parser, "\x1b[231;0;56832;0;0;1_") == []


def test_legacy_escape_uses_one_configured_timeout_and_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    import textual._parser as parser_module

    now = 100.0
    monkeypatch.setattr(windows_keys, "monotonic", lambda: now)
    monkeypatch.setattr(parser_module, "get_time", lambda: now)
    monkeypatch.setattr(constants, "ESCAPE_DELAY", 0.02)
    parser = windows_keys.get_parser_class()()
    assert list(parser.feed("\x1b")) == []
    now += 0.021
    assert [event.key for event in parser.tick()] == ["escape"]
    assert _keys(parser, "a") == [("a", "a")]
    assert list(parser.feed("")) == []
    assert parser.is_eof


def test_partial_outer_record_keeps_inner_escape_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    import textual._parser as parser_module

    now = 100.0
    monkeypatch.setattr(windows_keys, "monotonic", lambda: now)
    monkeypatch.setattr(parser_module, "get_time", lambda: now)
    monkeypatch.setattr(constants, "ESCAPE_DELAY", 0.1)
    parser = windows_keys.get_parser_class()()
    assert list(parser.feed(_wrapped("\x1b"))) == []
    now += 0.08
    assert list(parser.feed("\x1b[0;0;")) == []
    now += 0.03
    assert list(parser.tick()) == []
    assert _keys(parser, "91;1;0;1_" + _wrapped("A")) == [("up", None)]


def test_truncated_record_recovers_at_eof() -> None:
    parser = windows_keys.get_parser_class()()
    assert list(parser.feed("\x1b[13;28;")) == []
    assert list(parser.feed(""))
    assert parser.is_eof


def _load_module(name: str) -> ModuleType:
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin is not None
    isolated = importlib.util.spec_from_file_location(name, spec.origin)
    assert isolated is not None and isolated.loader is not None
    module = importlib.util.module_from_spec(isolated)
    isolated.loader.exec_module(module)
    return module


@pytest.fixture
def windows_driver(monkeypatch: pytest.MonkeyPatch):
    """Load real Windows driver/monitor code, replacing only OS and thread I/O."""
    kernel = SimpleNamespace(GetStdHandle=Mock(return_value=1))
    with monkeypatch.context() as loading:
        loading.setattr(ctypes, "WinDLL", lambda name, use_last_error: kernel, raising=False)
        loading.setitem(sys.modules, "msvcrt", ModuleType("msvcrt"))
        win32 = _load_module("textual.drivers.win32")
    monkeypatch.setitem(sys.modules, "textual.drivers.win32", win32)
    monkeypatch.setattr(textual.drivers, "win32", win32, raising=False)
    module = _load_module("textual.drivers.windows_driver")
    monkeypatch.setitem(sys.modules, "textual.drivers.windows_driver", module)
    monkeypatch.setattr(textual.drivers, "windows_driver", module, raising=False)
    windows = dataclasses.replace(get_platform(), os_name="windows")
    monkeypatch.setattr(patch, "get_platform", lambda: windows)
    patch.apply_runtime_patch()

    writer = StringIO()
    writer_thread = SimpleNamespace(start=Mock(), stop=Mock(), write=writer.write)
    restore = Mock()
    monkeypatch.setattr(win32, "enable_application_mode", lambda: restore)
    monkeypatch.setattr(module, "WriterThread", lambda file: writer_thread)
    monkeypatch.setattr(module, "asyncio", SimpleNamespace(get_running_loop=lambda: None))
    # Keep EventMonitor.run intact. Run it with a finite native input batch
    # instead of owning a real background thread.
    monkeypatch.setattr(win32.EventMonitor, "start", lambda self: None)
    monkeypatch.setattr(win32.EventMonitor, "join", lambda self: None)
    driver = object.__new__(module.WindowsDriver)
    driver._file = writer
    driver._app = SimpleNamespace(log=SimpleNamespace(error=Mock()))
    driver._mouse = True
    driver._writer_thread = None
    driver._event_thread = None
    driver._restore_console = None
    driver.exit_event = Event()
    received = []
    monkeypatch.setattr(driver, "process_message", received.append)
    try:
        yield driver, writer, win32, received, restore
    finally:
        driver.close()


def test_windows_driver_enables_resets_and_reenables_protocol(windows_driver) -> None:
    driver, writer, _win32, _received, restore = windows_driver
    for _ in range(2):
        driver.start_application_mode()
        driver.stop_application_mode()
        driver.close()
    output = writer.getvalue()
    assert output.count("\x1b[?9001h\x1b[>1u") == 2
    assert output.count("\x1b[?9001l\x1b[<u") == 2
    assert output.index("\x1b[?9001l") < output.index("\x1b[?1049l")
    assert "\x1b[>4;2m" not in output
    assert restore.call_count == 2


def test_windows_close_without_stop_and_opt_out(windows_driver, monkeypatch) -> None:
    driver, writer, _win32, _received, _restore = windows_driver
    driver.start_application_mode()
    driver.close()
    driver.close()
    assert writer.getvalue().count("\x1b[?9001l") == 1
    monkeypatch.setattr(constants, "DISABLE_KITTY_KEY", True)
    writer.seek(0)
    writer.truncate()
    driver.start_application_mode()
    driver.stop_application_mode()
    assert "9001" not in writer.getvalue()


@pytest.mark.parametrize("failure", ["enable", "reset"])
def test_windows_close_restores_after_failed_mode_write(windows_driver, monkeypatch, failure: str) -> None:
    driver, writer, _win32, _received, restore = windows_driver
    driver.start_application_mode()
    if failure == "enable":
        driver.stop_application_mode()
    failing_sequence = "\x1b[?9001h" if failure == "enable" else "\x1b[?9001l"
    failed = False

    def write(data: str) -> None:
        nonlocal failed
        if not failed and data.startswith(failing_sequence):
            failed = True
            # The write may already have changed terminal state before failing.
            writer.write(failing_sequence)
            raise OSError("injected output failure")
        writer.write(data)

    monkeypatch.setattr(driver._writer_thread, "write", write)
    with pytest.raises(OSError, match="injected output failure"):
        driver.write("\x1b[>1u" if failure == "enable" else "\x1b[<u")
    driver.close()
    assert writer.getvalue().endswith("\x1b[?9001l")
    restore.assert_called_once()


def test_real_event_monitor_routes_protocol_through_production_adapter(windows_driver, monkeypatch) -> None:
    driver, _writer, win32, received, _restore = windows_driver
    first_write = type(driver).write
    patch.apply_runtime_patch()
    assert type(driver).write is first_write
    driver.start_application_mode()
    wire = "\x1b[13;28;13;1;16;1_\x1b[13;28;13;0;16;1_" + _wrapped("\x1b[200~a\nb\x1b[201~")

    def read_console(handle, records, count, read_count):
        for index, char in enumerate(wire):
            record = records._obj[index]
            record.EventType = 1
            record.Event.KeyEvent.bKeyDown = True
            record.Event.KeyEvent.uChar.UnicodeChar = char
        read_count._obj.value = len(wire)
        driver.exit_event.set()
        return 1

    monkeypatch.setattr(win32.KERNEL32, "ReadConsoleInputW", read_console, raising=False)
    monkeypatch.setattr(win32, "wait_for_handles", lambda handles, timeout: handles[0])
    driver._event_thread.run()
    driver._app.log.error.assert_not_called()
    assert isinstance(received[0], events.Key)
    assert received[0].key == "shift+enter"
    assert isinstance(received[1], events.Paste)
    assert received[1].text == "a\nb"
    assert len(received) == 2
    driver.exit_event.clear()
    driver.stop_application_mode()
