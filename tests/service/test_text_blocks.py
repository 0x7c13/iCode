# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Identity and whitespace contracts for text fragment reconstruction."""

from __future__ import annotations

import pytest

from chrys.kernel import OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY
from chrys.service.text_blocks import join_text_blocks, reconstruct_text_blocks, text_block_id


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        ([], []),
        ([("回", None), ("退", None)], ["回退"]),
        ([("基线\n", None), ("核对完毕", None)], ["基线\n核对完毕"]),
        ([("first", None), ("\n\n", None), ("second", None)], ["first\n\nsecond"]),
        ([("  ", "a"), ("\n", "a"), ("\t", "a")], ["  \n\t"]),
        ([("", None), ("", None)], [""]),
        ([("one", "a"), ("two", "a"), ("three", "b")], ["onetwo", "three"]),
        ([("one", "a"), ("two", "b"), ("three", "a")], ["one", "two", "three"]),
        ([("one", None), ("two", "a"), ("three", None)], ["one", "two", "three"]),
        ([("one", "a"), ("", "b"), ("three", "a")], ["one", "", "three"]),
        ([("one", "a"), ("", None), ("three", "a")], ["one", "", "three"]),
    ],
)
def test_reconstruct_text_blocks(parts: list[tuple[str, str | None]], expected: list[str]) -> None:
    assert reconstruct_text_blocks(iter(parts)) == expected


@pytest.mark.parametrize(
    ("blocks", "expected"),
    [
        ([], ""),
        (["", ""], ""),
        (["", "One.", "", "Three.", ""], "One.\nThree."),
        (["\n\n"], "\n\n"),
        (["  \t"], "  \t"),
        (["One.", "  ", "Three."], "One.\n  \nThree."),
    ],
)
def test_join_text_blocks_omits_only_empty_strings(blocks: list[str], expected: str) -> None:
    assert join_text_blocks(iter(blocks)) == expected


@pytest.mark.parametrize(
    "properties",
    [None, {}, [], {OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY: None}, {OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY: {"id": 42}}],
)
def test_missing_or_invalid_envelope_has_no_block_identity(properties: object) -> None:
    assert text_block_id(properties) is None


def test_only_envelope_id_identifies_a_block() -> None:
    first = {OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY: {"id": "msg_a", "status": "in_progress", "phase": "commentary"}}
    last = {OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY: {"id": "msg_a", "status": "completed", "phase": "final_answer"}}

    assert text_block_id(first) == text_block_id(last) == "msg_a"
    assert reconstruct_text_blocks([("回", text_block_id(first)), ("退", text_block_id(last))]) == ["回退"]
