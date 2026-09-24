# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Source display keeps ordinary whitespace and neutralizes terminal controls."""

from __future__ import annotations

import pytest

from chrys.app.tui.util.source_text import sanitize_source_text


@pytest.mark.parametrize("tab_size", [4, 8])
def test_source_whitespace_is_normalized_before_filtering(tab_size: int) -> None:
    source = 'if ready:\r\n\tvalue = "中文"\r\treturn value\n'
    indent = " " * tab_size
    assert sanitize_source_text(source, tab_size=tab_size) == (
        f'if ready:\n{indent}value = "中文"\n{indent}return value\n'
    )


def test_source_controls_are_inert_but_written_escape_sequences_remain_literal() -> None:
    controls = "".join(chr(code) for code in (*range(0x20), *range(0x7F, 0xA0)) if code not in (9, 10, 13))
    literal = r'"\x1b[24D\t\r\n"'
    assert sanitize_source_text(controls + "\n" + literal) == "�" * len(controls) + "\n" + literal
