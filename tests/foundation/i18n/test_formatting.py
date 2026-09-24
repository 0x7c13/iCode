# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for safe locale-neutral message formatting."""

from __future__ import annotations

import pytest

from chrys.foundation.i18n import DisplayPath, msg
from chrys.foundation.i18n.formatting import (
    format_message,
    format_template,
    has_visible_content,
    sanitize_legacy_block,
    sanitize_legacy_scalar,
    validate_authored_template,
)


def test_format_template_supports_reordered_named_slots() -> None:
    rendered = format_template("{second}, {first}", {"first": "A", "second": "B"})

    assert rendered == "B, A"


def test_format_template_preserves_cjk_output() -> None:
    rendered = format_template("你好, {name}!", {"name": "小菊"})

    assert rendered == "你好, 小菊!"


def test_format_template_supports_literal_braces() -> None:
    rendered = format_template("Use {{name}} for {value}", {"value": "Chrys"})

    assert rendered == "Use {name} for Chrys"


def test_format_template_rejects_missing_parameters() -> None:
    with pytest.raises(ValueError, match="parameters"):
        format_template("Hello, {name}", {})


def test_format_template_rejects_extra_parameters() -> None:
    with pytest.raises(ValueError, match="parameters"):
        format_template("Hello", {"name": "Chrys"})


def test_count_is_the_only_unused_parameter_exemption() -> None:
    assert format_template("One file", {"count": "1"}) == "One file"


def test_english_singular_plural_fallback_may_omit_count() -> None:
    definition = msg("test.singular_without_count", fallback="One file", plural_fallback="{count} files")

    assert format_message(definition.bind(count=1)) == "One file"


@pytest.mark.parametrize(
    "template",
    [
        "{0}",
        "{}",
        "{foo-bar}",
        "{foo bar}",
        "{Value}",
        "{名字}",
        "{value.attr}",
        "{value[index]}",
        "{value!r}",
        "{value:>10}",
        "{value:{width}}",
        "{value:}",
    ],
)
def test_authored_template_accepts_only_plain_named_slots(template: str) -> None:
    with pytest.raises(ValueError, match="placeholder"):
        validate_authored_template(template, multiline=False)


def test_display_path_converts_surrogateescaped_text_for_display() -> None:
    definition = msg("test.surrogate_path", fallback="Path: {path}")

    rendered = format_message(definition.bind(path=DisplayPath("bad\udcff-name")))

    assert rendered == r"Path: bad\udcff-name"
    assert "\udcff" not in rendered


@pytest.mark.parametrize(
    "translated",
    [
        "Translated {missing}",
        "Translated {name!r}",
        "Translated \x1b[31m{name}",
        "[bold]Translated {name}[/bold]",
        "​",
    ],
)
def test_translated_template_failure_uses_english_fallback(translated: str) -> None:
    definition = msg("test.translation_failure", fallback="Hello, {name}")
    reference = definition.bind(name="Chrys")

    assert format_message(reference, template=translated) == "Hello, Chrys"


def test_translated_template_cannot_drop_a_bound_slot() -> None:
    definition = msg("test.translation_dropped_slot", fallback="{first} and {second}")
    reference = definition.bind(first="A", second="B")

    assert format_message(reference, template="Only {first}") == "A and B"


@pytest.mark.parametrize(
    "translated",
    ["​", "​other.lookup​", "​ other.lookup ​", "ᅠother.lookupᅠ", "line one\nline two"],
)
def test_translated_template_runtime_safety_failure_uses_english(translated: str) -> None:
    reference = msg("test.translation_runtime_safety", fallback="English").bind()

    assert format_message(reference, template=translated) == "English"


@pytest.mark.parametrize(
    "template",
    [
        "[bold]Unsafe[/bold]",
        r"\\[bold]Unsafe",
        r"\\\\[red]Unsafe",
        "[50%]Unsafe",
        "[ bold]Unsafe",
        "[/ bold]Unsafe",
        "[-foo=bar bold]Unsafe",
        "[3 items]Unsafe",
        "[]Unsafe",
        r"\[[bold]Unsafe",
    ],
)
def test_authored_template_rejects_every_unescaped_bracket_pair(template: str) -> None:
    with pytest.raises(ValueError, match="markup"):
        validate_authored_template(template, multiline=False)


def test_multiline_template_rejects_a_bracket_pair_spanning_lines() -> None:
    with pytest.raises(ValueError, match="markup"):
        validate_authored_template("[bold\n]Unsafe", multiline=True)


@pytest.mark.parametrize(
    "template",
    [
        r"\[bold] stays literal",
        r"\\\[bold] stays literal",
        r"\[3 items] stays literal",
        "unmatched [ bracket stays prose",
        "a lone ] bracket stays prose",
        "closing ] before an unmatched [ stays prose",
    ],
)
def test_authored_template_accepts_escaped_bracket_text(template: str) -> None:
    assert validate_authored_template(template, multiline=False) == frozenset()


@pytest.mark.parametrize("text", ["", " \n\t", "​", "​ ​", "\u2060", "\u0301"])
def test_visible_content_rejects_text_without_positive_cell_width(text: str) -> None:
    assert not has_visible_content(text)


@pytest.mark.parametrize("text", ["A", "界", " {value} ", "\u0301A"])
def test_visible_content_accepts_a_positive_cell_width_glyph(text: str) -> None:
    assert has_visible_content(text)


def test_legacy_scalar_sanitization_replaces_all_controls() -> None:
    assert sanitize_legacy_scalar("a\tb\nc\rd\x1be\x7ff\x85g") == "a�b�c�d�e�f�g"


def test_legacy_block_sanitization_preserves_only_lf() -> None:
    assert sanitize_legacy_block("a\tb\nc\rd\x1be") == "a�b\nc�d�e"
