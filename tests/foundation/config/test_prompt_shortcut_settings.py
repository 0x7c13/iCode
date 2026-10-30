# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``prompt_shortcuts.paths``: a JSON array of folder paths, read and stored each folder once."""

from __future__ import annotations

import json

import pytest

from chrys.foundation.config.coercion import Coerced, CoerceReason, CoerceStatus
from chrys.foundation.config.settings import Settings, parse_prompt_shortcut_paths
from chrys.foundation.config.spec import specs_by_field
from chrys.foundation.config.web_values import json_list_coercer, web_list_coercer


def _coerce_paths(raw: object) -> Coerced:
    spec = specs_by_field(Settings)["prompt_shortcuts_paths"]
    assert spec.key == "prompt_shortcuts.paths"
    return spec.coerce(raw)


# -- parse_prompt_shortcut_paths ------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[]", []),
        ([], []),
        ('["~/prompts", "/abs/dir"]', ["~/prompts", "/abs/dir"]),
        (["~/prompts", "/abs/dir"], ["~/prompts", "/abs/dir"]),
        ('["  team/prompts  ", "\\tx\\n"]', ["team/prompts", "x"]),
        (["  a  ", "a"], ["a", "a"]),
    ],
    ids=["empty-json", "empty-list", "json", "list", "strips-json", "strips-list-keeps-repeats"],
)
def test_parse_reads_a_json_array_or_a_list_of_strings(raw: object, expected: list[str]) -> None:
    assert parse_prompt_shortcut_paths(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        '"one/folder"',
        "{}",
        {"a": "b"},
        "not json",
        "",
        '["ok", 3]',
        ["ok", None],
        '["ok", ["nested"]]',
        '["ok", ""]',
        ["ok", "   "],
    ],
    ids=[
        "json-string",
        "json-object",
        "dict",
        "not-json",
        "empty-text",
        "number-item",
        "none-item",
        "list-item",
        "empty-item",
        "blank-item",
    ],
)
def test_parse_refuses_anything_but_an_array_of_non_blank_strings(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_prompt_shortcut_paths(raw)


# -- the setting's coercer -----------------------------------------------------


@pytest.mark.parametrize("raw", ["", "   ", "\n"])
def test_a_blank_field_is_the_empty_list(raw: str) -> None:
    outcome = _coerce_paths(raw)
    assert outcome.status is CoerceStatus.VALID
    assert outcome.value == "[]"


def test_nothing_set_is_missing() -> None:
    assert _coerce_paths(None).status is CoerceStatus.MISSING


def test_folders_are_stored_stripped_and_each_once_in_first_seen_order() -> None:
    outcome = _coerce_paths('[" b ", "a", "b", "a ", "c"]')

    assert outcome.status is CoerceStatus.VALID
    assert json.loads(outcome.value) == ["b", "a", "c"]


def test_a_list_value_is_stored_as_a_json_array_string() -> None:
    outcome = _coerce_paths(["~/提示", "rel/dir"])

    assert outcome.status is CoerceStatus.VALID
    # Stored as written, not \\u-escaped.
    assert outcome.value == '["~/提示", "rel/dir"]'


@pytest.mark.parametrize(
    "raw",
    ['"one/folder"', '["ok", 3]', '["ok", " "]', "not json", {"a": 1}, "[" * 200_000],
    ids=["json-string", "number-item", "blank-item", "not-json", "dict", "nested-too-deeply"],
)
def test_an_invalid_value_is_refused_as_not_a_json_array(raw: object) -> None:
    outcome = _coerce_paths(raw)

    assert outcome.status is CoerceStatus.INVALID
    assert outcome.reason is CoerceReason.EXPECTED_JSON_ARRAY


# -- the shared list coercer -------------------------------------------------------


def test_web_list_coercer_keeps_any_json_array_unchanged() -> None:
    assert web_list_coercer("  ").value == "[]"
    assert web_list_coercer(None).status is CoerceStatus.MISSING
    kept = web_list_coercer('[1, "a", {"b": null}, "a"]')
    assert kept.status is CoerceStatus.VALID
    # No item check and no dedupe: those belong to the setting's own parse.
    assert json.loads(kept.value) == [1, "a", {"b": None}, "a"]
    assert web_list_coercer(["x"]).value == '["x"]'
    refused = web_list_coercer('{"a": 1}')
    assert refused.status is CoerceStatus.INVALID
    assert refused.reason is CoerceReason.EXPECTED_JSON_ARRAY
    deep = web_list_coercer("[" * 200_000)
    assert deep.status is CoerceStatus.INVALID
    assert deep.reason is CoerceReason.EXPECTED_JSON_ARRAY


@pytest.mark.parametrize("error", [ValueError, TypeError, RecursionError])
def test_json_list_coercer_turns_a_refusal_into_expected_json_array(error: type[Exception]) -> None:
    def refuse(_raw: object) -> list[object]:
        raise error("no")

    outcome = json_list_coercer(refuse)("[1]")

    assert outcome.status is CoerceStatus.INVALID
    assert outcome.reason is CoerceReason.EXPECTED_JSON_ARRAY


def test_json_list_coercer_stores_what_its_parse_returns() -> None:
    outcome = json_list_coercer(lambda _raw: ["parsed"])("[1, 2]")

    assert outcome.status is CoerceStatus.VALID
    assert outcome.value == '["parsed"]'
