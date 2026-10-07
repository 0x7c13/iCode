# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for JSON text written for a model to read."""

from __future__ import annotations

import json

from chrys.foundation.text.model_json import model_json


def test_non_ascii_text_stays_as_written() -> None:
    assert model_json({"城市": "北京 🌼"}) == '{"城市": "北京 🌼"}'


def test_a_lone_surrogate_keeps_its_escape_and_reads_back() -> None:
    value = {"path": "报告\udc80.txt"}

    text = model_json(value)

    assert text == '{"path": "报告\\udc80.txt"}'
    assert text.encode("utf-8")
    assert json.loads(text) == value


def test_default_renders_values_json_cannot() -> None:
    assert model_json({"标签": {"甲"}}, default=sorted) == '{"标签": ["甲"]}'
