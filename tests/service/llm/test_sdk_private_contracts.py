# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fail-loudly contracts for the private OpenAI SDK helpers Chrys calls.

Structured output strictifies a raw JSON schema with the SDK's own recursive
helper (``service/llm/_structured_outputs.py``) and turns a model class into
a ``response_format`` envelope with another
(``service/llm/chat_completions/request.py``). Both live in private SDK
modules, so an upgrade that moves or reshapes one fails here first, in one
obvious place, instead of only through the behavior tests downstream.
"""

from __future__ import annotations

import inspect

from pydantic import BaseModel


def test_the_strict_schema_helper_takes_the_schema_then_path_and_root_by_keyword() -> None:
    from openai.lib._pydantic import _ensure_strict_json_schema

    parameters = inspect.signature(_ensure_strict_json_schema).parameters
    assert [(name, parameter.kind) for name, parameter in parameters.items()] == [
        ("json_schema", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("path", inspect.Parameter.KEYWORD_ONLY),
        ("root", inspect.Parameter.KEYWORD_ONLY),
    ]

    schema = {"type": "object", "properties": {"city": {"type": "object", "properties": {"name": {"type": "string"}}}}}
    strict = _ensure_strict_json_schema(schema, path=(), root=schema)

    assert strict["additionalProperties"] is False
    assert strict["required"] == ["city"]
    assert strict["properties"]["city"]["additionalProperties"] is False
    assert strict["properties"]["city"]["required"] == ["name"]


def test_the_response_format_helper_takes_one_model_class() -> None:
    from openai.lib._parsing._completions import type_to_response_format_param

    parameters = list(inspect.signature(type_to_response_format_param).parameters.values())
    assert [(parameter.name, parameter.kind) for parameter in parameters] == [
        ("response_format", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    class Digest(BaseModel):
        answer: str

    envelope = type_to_response_format_param(Digest)

    assert isinstance(envelope, dict)
    assert envelope["type"] == "json_schema"
    json_schema = envelope["json_schema"]
    assert json_schema["name"] == "Digest"
    assert json_schema["strict"] is True
    assert json_schema["schema"]["additionalProperties"] is False
    assert json_schema["schema"]["required"] == ["answer"]
