# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for HTTP header charset validation helpers."""

from __future__ import annotations

import pytest

from chrys.foundation.util.header_charset import (
    api_key_charset_error,
    find_non_printable_ascii,
    header_name_charset_error,
    header_value_charset_error,
    model_id_charset_error,
)

# ───────────────────────── find_non_printable_ascii ───────────────────


@pytest.mark.parametrize("value", ["", "sk-abc123", " ~", "Bearer {{MY_KEY}}", "!#$%&'*+-.^_`|~"])
def test_find_non_printable_ascii_accepts_printable_ascii(value: str) -> None:
    assert find_non_printable_ascii(value) is None


@pytest.mark.parametrize(
    ("value", "index"),
    [
        ("sk-abc▼def", 6),  # BLACK DOWN-POINTING TRIANGLE
        ("密钥", 0),
        ("key\twith-tab", 3),
        ("key\nwith-newline", 3),
        ("key\x7fdel", 3),
        ("\x00", 0),
        ("ok😀", 2),
    ],
)
def test_find_non_printable_ascii_locates_first_bad_char(value: str, index: int) -> None:
    assert find_non_printable_ascii(value) == index


# ───────────────────────── api_key_charset_error ──────────────────────


@pytest.mark.parametrize("value", ["", "sk-ant-api03-abc", "token-abc123", "{{OPENAI_API_KEY}}", "key with space"])
def test_api_key_accepts_permissive_printable_ascii(value: str) -> None:
    assert api_key_charset_error(value) is None


def test_api_key_error_reports_position_without_echoing_content() -> None:
    message = api_key_charset_error("sk-secret▼value")

    assert message is not None
    assert "position 10" in message
    assert "printable ASCII" in message
    assert "sk-secret" not in message
    assert "▼" not in message
    assert "25BC" not in message


def test_api_key_error_rejects_control_characters_without_echo() -> None:
    message = api_key_charset_error("abc\tdef")

    assert message is not None
    assert "position 4" in message
    assert "abc" not in message


def test_api_key_error_rejects_outer_spaces_without_echo() -> None:
    """h11 rejects header values with leading/trailing space before any request is sent."""
    leading = api_key_charset_error(" sk-abc")
    trailing = api_key_charset_error("sk-abc ")

    assert leading is not None
    assert "starts with a space" in leading
    assert trailing is not None
    assert "ends with a space" in trailing
    assert "sk-abc" not in leading
    assert "sk-abc" not in trailing


# ───────────────────────── model_id_charset_error ─────────────────────


@pytest.mark.parametrize(
    "value",
    [
        "gpt-4o",
        "claude-sonnet-4-5",
        "huggingface/WizardLM/WizardCoder-Python-34B-V1.0",
        "azure/my deployment",
        "org:model@rev#tag?x=1",
        "/models/local.gguf",
    ],
)
def test_model_id_accepts_provider_and_gateway_shapes(value: str) -> None:
    assert model_id_charset_error(value) is None


def test_model_id_error_echoes_offending_character() -> None:
    message = model_id_charset_error("gpt▼4o")

    assert message is not None
    assert "'▼'" in message
    assert "U+25BC" in message
    assert "position 4" in message


def test_model_id_rejects_empty() -> None:
    assert model_id_charset_error("") == "Model ID is empty."


def test_model_id_error_describes_control_character_without_echo() -> None:
    message = model_id_charset_error("gpt\x004o")

    assert message is not None
    assert "control character" in message
    assert "U+0000" in message
    assert "position 4" in message


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        (" gpt-4o", "starts with a space"),
        ("gpt-4o ", "ends with a space"),
        (" ", "starts with a space"),
    ],
)
def test_model_id_rejects_outer_spaces(value: str, fragment: str) -> None:
    message = model_id_charset_error(value)

    assert message is not None
    assert fragment in message


# ───────────────────────── header_name_charset_error ──────────────────


@pytest.mark.parametrize("name", ["X-Api-Key", "authorization", "x-goog-api-key", "!#$%&'*+-.^_`|~", "H1"])
def test_header_name_accepts_rfc9110_tokens(name: str) -> None:
    assert header_name_charset_error(name) is None


def test_header_name_rejects_empty() -> None:
    assert header_name_charset_error("") == "Header name is empty."


@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("X Api-Key", "position 2"),  # inner space is invalid in a token
        ("X-Api▼", "U+25BC"),
        ("X:Colon", "':'"),
        ("名字", "position 1"),
    ],
)
def test_header_name_rejects_non_token_characters(name: str, fragment: str) -> None:
    message = header_name_charset_error(name)

    assert message is not None
    assert fragment in message
    assert "letters, digits" in message


# ───────────────────────── header_value_charset_error ─────────────────


@pytest.mark.parametrize("value", ["", "Bearer sk-abc", "value with spaces", "{{MY_TOKEN}}", "~!@#$%"])
def test_header_value_accepts_printable_ascii(value: str) -> None:
    assert header_value_charset_error("X-Api-Key", value) is None


def test_header_value_error_names_header_but_never_echoes_value() -> None:
    message = header_value_charset_error("X-Api-Key", "Bearer secret▼token")

    assert message is not None
    assert "'X-Api-Key'" in message
    assert "position 14" in message
    assert "secret" not in message
    assert "▼" not in message


def test_header_value_error_rejects_crlf_injection() -> None:
    message = header_value_charset_error("X-Api-Key", "ok\r\nInjected: yes")

    assert message is not None
    assert "position 3" in message


def test_header_value_error_rejects_outer_spaces_without_echoing_value() -> None:
    leading = header_value_charset_error("X-Api-Key", " token")
    trailing = header_value_charset_error("X-Api-Key", "token ")

    assert leading is not None
    assert "'X-Api-Key'" in leading
    assert "starts with a space" in leading
    assert trailing is not None
    assert "ends with a space" in trailing
    assert "token" not in leading
    assert "token" not in trailing
