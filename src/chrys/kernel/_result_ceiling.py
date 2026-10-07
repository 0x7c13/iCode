# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Idempotent kernel backstop for oversized local tool results."""

from __future__ import annotations

from copy import copy
from typing import Any

from chrys.foundation.text.model_json import model_json
from chrys.foundation.text.tokenizer import MixedLanguageTokenizer
from chrys.foundation.text.tool_output import truncate_content_texts, truncate_output
from chrys.foundation.tool_result_metadata import record_payload_truncation, tool_payload_observation

from ._content import Content

_EXCEPTION_OMITTED = "[exception omitted]"

_tokenizer = MixedLanguageTokenizer()


_CONTENT_TEXT_FIELDS = ("text", "result", "output", "message", "stdout", "stderr", "exception")
"""Every string field of a plain content the ceiling bounds."""


def _content_texts(content: Content) -> list[str]:
    """The strings *content* contributes to the ceiling, in bounding order."""
    if content.type == "function_result":
        texts = [item.text for item in (content.items or []) if item.type == "text" and isinstance(item.text, str)]
        if isinstance(content.exception, str):
            texts.append(content.exception)
        return texts
    return [value for name in _CONTENT_TEXT_FIELDS if isinstance(value := getattr(content, name, None), str)]


def _original_text(result: Any) -> str:
    """What the tool produced, measured the way the ceiling measured it."""
    if isinstance(result, str):
        return result
    if isinstance(result, Content):
        return "\n".join(_content_texts(result))
    if isinstance(result, list) and all(isinstance(item, Content) for item in result):
        return "\n".join(item.text for item in result if item.type == "text" and isinstance(item.text, str))
    return _canonical_result_text(result)


def _canonical_result_text(result: Any) -> str:
    try:
        return model_json(result, default=str)
    except TypeError, ValueError:
        return str(result)


def _bound_content_list(items: list[Content], ceiling: int) -> list[Content] | None:
    texts = [item.text for item in items if item.type == "text" and isinstance(item.text, str)]
    if not texts:
        return None
    bounded = truncate_content_texts(texts, ceiling)
    if bounded is None:
        return None

    output: list[Content] = []
    text_index = 0
    for item in items:
        if item.type == "text" and item.text is not None:
            replacement = bounded[text_index]
            text_index += 1
            if replacement is None:
                continue
            if replacement == item.text:
                output.append(item)
            else:
                item_copy = copy(item)
                item_copy.text = replacement
                output.append(item_copy)
        else:
            output.append(item)
    return output


def _fit_texts_before_exception(texts: list[str], ceiling: int) -> tuple[list[str | None], str]:
    """Leave room for a non-empty exception sentinel under the same ceiling."""
    exception = truncate_output(_EXCEPTION_OMITTED, ceiling)
    item_budget = max(1, ceiling - _tokenizer.count_tokens(exception))

    while item_budget >= 1:
        bounded = truncate_content_texts(texts, item_budget)
        values: list[str | None] = list(texts) if bounded is None else bounded
        joined = "\n".join([*(value for value in values if value is not None), exception])
        total = _tokenizer.count_tokens(joined)
        if total <= ceiling:
            return values, exception
        if item_budget == 1:
            break
        item_budget = max(1, item_budget - max(1, total - ceiling))

    return [None] * len(texts), exception


def _bound_function_result(content: Content, ceiling: int) -> Content:
    items = list(content.items or [])
    item_texts = [item.text for item in items if item.type == "text" and isinstance(item.text, str)]
    fields = [*item_texts]
    if isinstance(content.exception, str):
        fields.append(content.exception)
    if not fields:
        return content
    bounded = truncate_content_texts(fields, ceiling)
    if bounded is None:
        return content
    if isinstance(content.exception, str) and bounded[-1] is None:
        bounded_items, bounded_exception = _fit_texts_before_exception(item_texts, ceiling)
        bounded = [*bounded_items, bounded_exception]

    copied = copy(content)
    output_items: list[Content] = []
    text_index = 0
    for item in items:
        if item.type == "text" and isinstance(item.text, str):
            replacement = bounded[text_index]
            text_index += 1
            if replacement is None:
                continue
            if replacement == item.text:
                output_items.append(item)
            else:
                item_copy = copy(item)
                item_copy.text = replacement
                output_items.append(item_copy)
        else:
            output_items.append(item)
    copied.items = output_items
    copied.result = "\n".join(
        item.text for item in output_items if item.type == "text" and isinstance(item.text, str) and item.text
    )
    if isinstance(content.exception, str):
        copied.exception = bounded[-1]
    return copied


def _bound_content(content: Content, ceiling: int) -> Content:
    if content.type == "function_result":
        return _bound_function_result(content, ceiling)

    candidates = tuple((name, getattr(content, name, None)) for name in _CONTENT_TEXT_FIELDS)
    field_names: list[str] = []
    values: list[str] = []
    for name, value in candidates:
        if isinstance(value, str):
            field_names.append(name)
            values.append(value)
    if not values:
        return content
    bounded = truncate_content_texts(values, ceiling)
    if bounded is None:
        return content

    copied = copy(content)
    for name, value in zip(field_names, bounded, strict=True):
        if name == "text":
            copied.text = value
        elif name == "result":
            copied.result = value
        elif name == "output":
            copied.output = value
        elif name == "message":
            copied.message = value
        elif name == "stdout":
            copied.stdout = value
        elif name == "stderr":
            copied.stderr = value
        else:
            copied.exception = value
    return copied


def apply_result_ceiling(result: Any, ceiling: int | None) -> Any:
    """Return *result* bounded to *ceiling*, copying only when it changes."""
    bounded = _bounded_result(result, ceiling)
    if bounded is not result:
        # The backstop is the last thing to shrink a result and the only one
        # the payload observation would otherwise never hear about — without
        # this the recorded payload claims it was never truncated.
        record_payload_truncation(_original_text(result))
    return bounded


def preview_result_ceiling(result: Any, ceiling: int | None, *, observation: dict[str, object]) -> Any:
    """Return what the ceiling will leave of *result*, recording the cut in *observation*.

    The middleware that reports the payload runs before this backstop and with
    the call's observation window already closed, so it reopens the window for
    this pass alone: without it the record measures text the model never sees
    and calls it untruncated.
    """
    token = tool_payload_observation.set(observation)
    try:
        return apply_result_ceiling(result, ceiling)
    finally:
        tool_payload_observation.reset(token)


def _bounded_result(result: Any, ceiling: int | None) -> Any:
    if ceiling is None or ceiling <= 0:
        return result
    if isinstance(result, str):
        return truncate_output(result, ceiling)
    if isinstance(result, list) and all(isinstance(item, Content) for item in result):
        bounded = _bound_content_list(result, ceiling)
        return result if bounded is None else bounded
    if isinstance(result, Content):
        return _bound_content(result, ceiling)

    serialized = _canonical_result_text(result)
    bounded_text = truncate_output(serialized, ceiling)
    return result if bounded_text == serialized else bounded_text
