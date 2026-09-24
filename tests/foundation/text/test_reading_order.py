# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Names sort the way a person reads them."""

from __future__ import annotations

from chrys.foundation.text.reading_order import reading_order


def test_names_sort_as_a_person_reads_them() -> None:
    names = ["huoshan-seed-pro-2.1", "Kimi K3", "GPT-5.10", "glm-5.3", "GPT-5.9 mini", "GPT-5.9", "DeepSeek-V4"]

    assert sorted(names, key=reading_order) == [
        "DeepSeek-V4",
        "glm-5.3",
        "GPT-5.9",
        "GPT-5.9 mini",
        "GPT-5.10",
        "huoshan-seed-pro-2.1",
        "Kimi K3",
    ]


def test_case_and_leading_zeros_do_not_separate_names() -> None:
    assert reading_order("GPT-5") == reading_order("gpt-05")
    assert reading_order("") == ()
