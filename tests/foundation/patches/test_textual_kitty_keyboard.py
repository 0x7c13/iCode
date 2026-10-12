# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the Textual Kitty keyboard IME patch."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest
from textual.events import Key

from chrys.foundation.patches import patcher, textual_kitty_keyboard
from chrys.foundation.patches.patcher import apply_patch_group
from chrys.foundation.patches.textual_kitty_keyboard import apply_runtime_patch
from tests.support.keyboard_patches import isolated_keyboard_patches as isolated_keyboard_patches


@pytest.fixture(autouse=True)
def _isolate_terminal_identity(monkeypatch: pytest.MonkeyPatch, isolated_keyboard_patches: None) -> None:
    """Keep runtime-patch tests independent of the developer's terminal."""
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("TERM_PROGRAM_VERSION", raising=False)


def _keys_for(sequence: str) -> list[Key]:
    from textual._xterm_parser import XTermParser

    return [event for event in XTermParser().feed(sequence) if isinstance(event, Key)]


def test_file_patch_fragments_match_installed_textual() -> None:
    spec = importlib.util.find_spec("textual")
    assert spec is not None
    assert spec.origin is not None
    source = (Path(spec.origin).parent / "_xterm_parser.py").read_text(encoding="utf-8")

    assert (
        textual_kitty_keyboard._OLD_SEARCH_THRESHOLD in source or textual_kitty_keyboard._NEW_SEARCH_THRESHOLD in source
    )
    assert textual_kitty_keyboard._OLD_REGEX in source or textual_kitty_keyboard._NEW_REGEX in source
    parse_fragments = (
        textual_kitty_keyboard._OLD_PARSE,
        textual_kitty_keyboard._NEW_PARSE,
        textual_kitty_keyboard._NEW_PARSE_WITH_RELEASE_GUARD,
    )
    assert any(fragment in source for fragment in parse_fragments)
    key_fragments = (
        textual_kitty_keyboard._OLD_KEY,
        textual_kitty_keyboard._EQUIVALENT_KEY,
        textual_kitty_keyboard._NEW_KEY,
    )
    assert any(fragment in source for fragment in key_fragments)


def test_file_patch_adds_multi_codepoint_key_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = tmp_path / "textual"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    target = package / "_xterm_parser.py"
    target.write_text(
        "\n".join(
            (
                textual_kitty_keyboard._OLD_SEARCH_THRESHOLD,
                textual_kitty_keyboard._OLD_REGEX,
                textual_kitty_keyboard._OLD_PARSE,
                textual_kitty_keyboard._OLD_KEY,
                "",
            )
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("chrys.foundation.patches.patcher._locate_package_dir", lambda _package: package)

    results = apply_patch_group(
        [patch for patch in patcher._patches if patch.package == "textual" and patch.module_file == "_xterm_parser.py"]
    )

    assert [result.status for result in results] == ["applied", "applied", "applied", "applied", "applied"]
    source = target.read_text(encoding="utf-8")
    assert textual_kitty_keyboard._NEW_PARSE_WITH_RELEASE_GUARD in source
    assert textual_kitty_keyboard._NEW_KEY in source
    assert textual_kitty_keyboard._OLD_KEY not in source


def test_file_patch_accepts_equivalent_key_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = tmp_path / "textual"
    package.mkdir()
    target = package / "_xterm_parser.py"
    target.write_text(
        "\n".join(
            (
                textual_kitty_keyboard._NEW_SEARCH_THRESHOLD,
                textual_kitty_keyboard._NEW_REGEX,
                textual_kitty_keyboard._NEW_PARSE_WITH_RELEASE_GUARD,
                textual_kitty_keyboard._EQUIVALENT_KEY,
                "",
            )
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("chrys.foundation.patches.patcher._locate_package_dir", lambda _package: package)

    results = apply_patch_group(
        [patch for patch in patcher._patches if patch.package == "textual" and patch.module_file == "_xterm_parser.py"]
    )

    assert [result.status for result in results] == ["skipped", "skipped", "skipped", "skipped", "skipped"]
    assert target.read_text(encoding="utf-8").endswith(f"{textual_kitty_keyboard._EQUIVALENT_KEY}\n")


def test_runtime_key_guard_matches_file_patch_fragment() -> None:
    assert textual_kitty_keyboard._NEW_KEY in inspect.getsource(textual_kitty_keyboard.apply_runtime_patch)


def test_runtime_patch_guard_target_matches_installed_textual() -> None:
    import textual

    assert textual.__version__ == textual_kitty_keyboard._RUNTIME_PATCH_TEXTUAL_VERSION


def test_runtime_patch_version_gate_precedes_private_api_access(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import textual
    import textual._xterm_parser as xterm_parser

    linux_driver = pytest.importorskip("textual.drivers.linux_driver")
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.setenv("TERM_PROGRAM_VERSION", "3.6.11")
    monkeypatch.setattr(linux_driver, "KITTY_REPORT_ALL_KEYS", 0b00001000)
    monkeypatch.setattr(linux_driver, "KITTY_REPORT_ASSOCIATED_TEXT", 0b00010000)
    monkeypatch.setattr(textual, "__version__", "9.0.0")
    monkeypatch.delattr(xterm_parser, "XTermParser")

    apply_runtime_patch()

    assert "loaded Textual is not the pinned 8.2.7" in caplog.text
    assert linux_driver.KITTY_REPORT_ALL_KEYS == 0
    assert linux_driver.KITTY_REPORT_ASSOCIATED_TEXT == 0


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.6.11"}, True),
        ({"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.6.0beta1"}, True),
        ({"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.7.0beta6"}, False),
        ({"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.60.0"}, False),
        ({"TERM_PROGRAM": "Apple_Terminal", "TERM_PROGRAM_VERSION": "3.6.11"}, False),
        ({"TERM_PROGRAM": "iTerm.app"}, False),
    ],
)
def test_detects_iterm_versions_with_kitty_composition_bug(env: dict[str, str], expected: bool) -> None:
    assert textual_kitty_keyboard._is_affected_iterm(env) is expected


def test_affected_iterm_keeps_only_kitty_disambiguation(monkeypatch: pytest.MonkeyPatch) -> None:
    linux_driver = pytest.importorskip("textual.drivers.linux_driver")

    monkeypatch.setattr(linux_driver, "KITTY_REPORT_ALL_KEYS", 0b00001000)
    monkeypatch.setattr(linux_driver, "KITTY_REPORT_ASSOCIATED_TEXT", 0b00010000)

    textual_kitty_keyboard._apply_iterm_kitty_flag_workaround(
        {"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.6.11"}
    )

    assert linux_driver.KITTY_REPORT_ALL_KEYS == 0
    assert linux_driver.KITTY_REPORT_ASSOCIATED_TEXT == 0
    assert (
        linux_driver.KITTY_DISAMBIGUATE_ESCAPE_CODES
        | linux_driver.KITTY_REPORT_ALL_KEYS
        | linux_driver.KITTY_REPORT_ASSOCIATED_TEXT
    ) == 1


def test_other_terminals_keep_textual_kitty_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    linux_driver = pytest.importorskip("textual.drivers.linux_driver")

    monkeypatch.setattr(linux_driver, "KITTY_REPORT_ALL_KEYS", 0b00001000)
    monkeypatch.setattr(linux_driver, "KITTY_REPORT_ASSOCIATED_TEXT", 0b00010000)

    textual_kitty_keyboard._apply_iterm_kitty_flag_workaround(
        {"TERM_PROGRAM": "ghostty", "TERM_PROGRAM_VERSION": "1.3.1"}
    )

    assert linux_driver.KITTY_REPORT_ALL_KEYS == 0b00001000
    assert linux_driver.KITTY_REPORT_ASSOCIATED_TEXT == 0b00010000
    assert (
        linux_driver.KITTY_DISAMBIGUATE_ESCAPE_CODES
        | linux_driver.KITTY_REPORT_ALL_KEYS
        | linux_driver.KITTY_REPORT_ASSOCIATED_TEXT
    ) == 25


def test_runtime_patch_wires_iterm_workaround(monkeypatch: pytest.MonkeyPatch) -> None:
    linux_driver = pytest.importorskip("textual.drivers.linux_driver")
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.setenv("TERM_PROGRAM_VERSION", "3.6.11")
    monkeypatch.setattr(linux_driver, "KITTY_REPORT_ALL_KEYS", 0b00001000)
    monkeypatch.setattr(linux_driver, "KITTY_REPORT_ASSOCIATED_TEXT", 0b00010000)

    apply_runtime_patch()

    assert linux_driver.KITTY_REPORT_ALL_KEYS == 0
    assert linux_driver.KITTY_REPORT_ASSOCIATED_TEXT == 0


def test_runtime_patch_decodes_multi_codepoint_kitty_text() -> None:
    apply_runtime_patch()

    keys = _keys_for("\x1b[32;;24403:21069u")

    assert [(key.key, key.character) for key in keys] == [("当前", "当前")]


def test_runtime_patch_decodes_long_multi_codepoint_kitty_text() -> None:
    apply_runtime_patch()
    text = "中华人民共和国"
    sequence = "\x1b[32;;" + ":".join(str(ord(character)) for character in text) + "u"
    assert len(sequence) > 32

    keys = _keys_for(sequence)

    assert [(key.key, key.character) for key in keys] == [(text, text)]


def test_runtime_patch_preserves_single_codepoint_kitty_text() -> None:
    apply_runtime_patch()

    keys = _keys_for("\x1b[24403;;24403u")

    assert [(key.key, key.character) for key in keys] == [("当", "当")]


def test_runtime_patch_ignores_kitty_release_events() -> None:
    apply_runtime_patch()

    assert _keys_for("\x1b[97;1:3;97u") == []


def test_runtime_patch_preserves_functional_keys() -> None:
    apply_runtime_patch()

    keys = _keys_for("\x1b[13;2u")

    assert [(key.key, key.character) for key in keys] == [("shift+enter", None)]


def test_runtime_patch_rejects_extra_kitty_parameter_fields() -> None:
    apply_runtime_patch()

    assert _keys_for("\x1b[1;2;3;4A") == []
    assert _keys_for("\x1b[1;;2;A") == []
    assert _keys_for("\x1b[;;;;A") == []
