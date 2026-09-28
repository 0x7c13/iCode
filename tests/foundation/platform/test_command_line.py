# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Joining and splitting a Windows command line: each the inverse of the other, on every OS."""

from __future__ import annotations

import subprocess

import pytest

from chrys.foundation.platform.command_line import join_windows_command_line, split_windows_command_line

_WITHOUT_LINE_BREAKS = [
    ["python", "server.py"],
    ["python", "[/bold] literal"],
    [""],
    ["", "x", ""],
    ["tab\there", "multi  space"],
    ['a"b c'],
    ['"'],
    ['""'],
    ['\\"'],
    ['a\\"b'],
    ['a\\\\"b c'],
    ["a\\"],
    ["a b\\"],
    ["a b\\\\"],
    ["back\\slash"],
    ["C:\\Program Files\\My Server\\", "--port", "8080"],
    ['x"', '"y'],
]


@pytest.mark.parametrize("argv", _WITHOUT_LINE_BREAKS)
def test_what_list2cmdline_quotes_splits_back_to_the_same_arguments(argv: list[str]) -> None:
    assert split_windows_command_line(subprocess.list2cmdline(argv)) == argv


@pytest.mark.parametrize("argv", _WITHOUT_LINE_BREAKS)
def test_without_line_breaks_join_quotes_as_list2cmdline_does(argv: list[str]) -> None:
    assert join_windows_command_line(argv) == subprocess.list2cmdline(argv)


@pytest.mark.parametrize(
    ("argv", "line"),
    [
        (["echo", "two\nlines"], 'echo "two\nlines"'),
        (["python\r\nserver.py"], '"python\r\nserver.py"'),
        (["\n", "\r"], '"\n" "\r"'),
        (["a backslash\\\nthen a line"], '"a backslash\\\nthen a line"'),
        (["ends in a backslash\n\\"], '"ends in a backslash\n\\\\"'),
        (['a "quoted"\nline'], '"a \\"quoted\\"\nline"'),
    ],
)
def test_an_argument_with_a_line_break_is_quoted_and_splits_back_whole(argv: list[str], line: str) -> None:
    assert join_windows_command_line(argv) == line
    assert split_windows_command_line(line) == argv


@pytest.mark.parametrize(
    ("text", "argv"),
    [
        ("", []),
        ("  python   server.py\t", ["python", "server.py"]),
        ('"C:\\Program Files\\node.exe" "my server.js"', ["C:\\Program Files\\node.exe", "my server.js"]),
        ("C:\\tools\\server.exe --root C:\\data\\", ["C:\\tools\\server.exe", "--root", "C:\\data\\"]),
        ('say "a""b"', ["say", 'a"b']),
        ('--name="my server"', ["--name=my server"]),
        ("npx -y\n@scope/server C:\\data", ["npx", "-y", "@scope/server", "C:\\data"]),
        ("python\r\nserver.py\n", ["python", "server.py"]),
        ('"two\nlines" kept', ["two\nlines", "kept"]),
        ("'single quotes' mean nothing", ["'single", "quotes'", "mean", "nothing"]),
    ],
)
def test_a_typed_command_line_splits_as_windows_programs_read_it(text: str, argv: list[str]) -> None:
    assert split_windows_command_line(text) == argv


@pytest.mark.parametrize("text", ['python "unclosed', '"', 'a "b" "c'])
def test_an_unclosed_quote_is_an_error(text: str) -> None:
    with pytest.raises(ValueError, match="No closing quotation"):
        split_windows_command_line(text)
