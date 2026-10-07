# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the kernel's local tool-result ceiling transform."""

from __future__ import annotations

from chrys.foundation.text.tokenizer import MixedLanguageTokenizer
from chrys.foundation.tool_result_metadata import tool_payload_observation
from chrys.kernel._result_ceiling import apply_result_ceiling
from chrys.kernel.types import Content

_tokenizer = MixedLanguageTokenizer()


def test_string_below_ceiling_is_unchanged_and_zero_disables() -> None:
    text = "short"

    assert apply_result_ceiling(text, 100) is text
    large = "x" * 10_000
    assert apply_result_ceiling(large, 0) is large


def test_string_over_ceiling_is_bounded_and_idempotent() -> None:
    first = apply_result_ceiling("HEAD" + "x" * 10_000 + "TAIL", 100)
    second = apply_result_ceiling(first, 100)

    assert "truncated" in first
    assert first.startswith("HEAD")
    assert first.endswith("TAIL")
    assert _tokenizer.count_tokens(first) <= 100
    assert second == first


def test_content_list_is_copied_and_non_text_position_is_preserved() -> None:
    first = Content.from_text("x" * 10_000)
    image = Content.from_data(b"image", "image/png")
    later = Content.from_text("later")
    source = [first, image, later]

    result = apply_result_ceiling(source, 100)

    assert result is not source
    assert result[0] is not first
    assert result[1] is image
    assert later not in result
    joined = "\n".join(item.text for item in result if item.type == "text" and item.text)
    assert _tokenizer.count_tokens(joined) <= 100
    assert first.text == "x" * 10_000


def test_single_content_string_fields_share_one_ceiling() -> None:
    source = Content(
        "mcp_server_tool_result",
        text="a" * 400,
        output="b" * 400,
        message="c" * 400,
        stdout="d" * 400,
        stderr="e" * 400,
        exception="f" * 400,
    )

    result = apply_result_ceiling(source, 100)

    assert result is not source
    values = [result.text, result.output, result.message, result.stdout, result.stderr, result.exception]
    joined = "\n".join(value for value in values if isinstance(value, str))
    assert _tokenizer.count_tokens(joined) <= 100
    assert source.text == "a" * 400


def test_function_result_items_are_sanitized_and_result_is_recomputed() -> None:
    image = Content.from_data(b"image", "image/png")
    source = Content.from_function_result(
        "c1",
        result=[Content.from_text("x" * 10_000), image, Content.from_text("later")],
    )

    result = apply_result_ceiling(source, 100)

    assert result is not source
    assert result.items is not None
    assert image in result.items
    text = "\n".join(item.text for item in result.items if item.type == "text" and item.text)
    assert result.result == text
    assert _tokenizer.count_tokens(text) <= 100


def test_function_result_ceiling_never_erases_exception_state() -> None:
    source = Content.from_function_result("c1", result="E" * 10_000, exception="boom")

    result = apply_result_ceiling(source, 100)

    assert result.exception is not None
    assert result.items is not None
    values = [item.text for item in result.items if item.type == "text" and item.text]
    values.append(result.exception)
    assert _tokenizer.count_tokens("\n".join(values)) <= 100


def test_pathological_shapes_degrade_to_truncated_canonical_serialization() -> None:
    mapping = {"payload": "x" * 10_000}
    raw_bytes = b"x" * 10_000

    mapped = apply_result_ceiling(mapping, 100)
    bytes_result = apply_result_ceiling(raw_bytes, 100)

    assert isinstance(mapped, str)
    assert isinstance(bytes_result, str)
    assert _tokenizer.count_tokens(mapped) <= 100
    assert _tokenizer.count_tokens(bytes_result) <= 100


def _observed(result: object, ceiling: int, *, preset: dict[str, object] | None = None) -> dict[str, object]:
    """Apply the ceiling with a live payload observation and return what it recorded."""
    observation: dict[str, object] = dict(preset or {})
    token = tool_payload_observation.set(observation)
    try:
        apply_result_ceiling(result, ceiling)
    finally:
        tool_payload_observation.reset(token)
    return observation


def test_the_backstop_reports_the_truncation_it_performs() -> None:
    # Nobody else sees this cut, so a silent one would be recorded as a
    # payload that reached the model whole.
    observation = _observed("x" * 10_000, 100)

    assert observation["truncated"] is True
    assert observation["original_bytes"] == 10_000


def test_a_lone_surrogate_in_an_oversized_result_is_still_measured() -> None:
    # An MCP server can return "\ud800" in JSON, and json.loads hands it back
    # as a lone high surrogate. Sizing it must not raise: the tool already ran,
    # and an exception here would report the completed call as failed.
    text = "\ud800" + "x" * 10_000

    observation = _observed(text, 100)

    assert observation["truncated"] is True
    assert observation["original_bytes"] == len(text.encode("utf-8", errors="backslashreplace"))


def test_a_result_that_fits_reports_nothing() -> None:
    assert _observed("short", 100) == {}
    assert _observed(Content.from_text("short"), 100) == {}


def test_the_first_bounding_pass_owns_the_original_size() -> None:
    # The spill already recorded the real size; the backstop then trims the
    # short notice it left behind and must not restate the size as that.
    observation = _observed("x" * 5_000, 100, preset={"original_bytes": 9_000_000, "truncated": True})

    assert observation["original_bytes"] == 9_000_000


def test_bounded_structured_result_keeps_non_ascii_text_readable() -> None:
    result = apply_result_ceiling({"摘要": "北京" * 5_000}, 100)

    assert isinstance(result, str)
    assert result.startswith('{"摘要": "北京北京')
    assert "\\u" not in result
