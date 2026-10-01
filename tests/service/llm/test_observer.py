# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for what the wire-call observer derives from a response on its own."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from chrys.service.llm.observer import intermediate_text_signal


def _make_response(*messages: SimpleNamespace) -> Any:
    return SimpleNamespace(messages=list(messages))


def _make_msg(*contents: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(contents=list(contents))


def _text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t, provider_hosted=False)


def _fn_call(name: str = "tool", *, informational_only: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        name=name,
        informational_only=informational_only,
        provider_hosted=False,
    )


def _fn_result() -> SimpleNamespace:
    return SimpleNamespace(type="function_result", provider_hosted=False)


# ──────────────── text written beside tool calls ─────────────────────────


def test_signal_is_the_text_beside_a_function_call() -> None:
    resp = _make_response(_make_msg(_text("Let me check"), _fn_call()))
    assert intermediate_text_signal(resp) == "Let me check"


def test_signal_concatenates_text_parts() -> None:
    resp = _make_response(_make_msg(_text("A"), _text("B"), _fn_call()))
    assert intermediate_text_signal(resp) == "AB"


def test_signal_reads_text_and_call_across_messages() -> None:
    resp = _make_response(
        _make_msg(_text("thinking")),
        _make_msg(_fn_call()),
    )
    assert intermediate_text_signal(resp) == "thinking"


# ──────────────── batch boundary without text ────────────────────────────


def test_signal_is_empty_for_a_function_call_without_text() -> None:
    resp = _make_response(_make_msg(_fn_call()))
    assert intermediate_text_signal(resp) == ""


def test_signal_is_empty_for_several_function_calls_without_text() -> None:
    resp = _make_response(_make_msg(_fn_call("a"), _fn_call("b"), _fn_call("c")))
    assert intermediate_text_signal(resp) == ""


def test_signal_ignores_empty_text_parts() -> None:
    resp = _make_response(_make_msg(_text(""), _fn_call()))
    assert intermediate_text_signal(resp) == ""


def test_signal_defers_hosted_response_text_even_beside_a_local_call() -> None:
    """Hosted-tool text belongs to the presentation bridge; the local call
    still marks the batch boundary."""
    hosted = SimpleNamespace(type="search_tool_call", provider_hosted=True)
    resp = _make_response(_make_msg(_text("checking"), hosted, _fn_call()))

    assert intermediate_text_signal(resp) == ""


# ──────────────── no signal ──────────────────────────────────────────────


def test_no_signal_for_a_text_only_response() -> None:
    resp = _make_response(_make_msg(_text("Hello")))
    assert intermediate_text_signal(resp) is None


def test_no_signal_for_an_informational_function_call() -> None:
    resp = _make_response(_make_msg(_text("visible"), _fn_call(informational_only=True)))
    assert intermediate_text_signal(resp) is None


def test_no_signal_for_an_informational_function_call_alone() -> None:
    resp = _make_response(_make_msg(_fn_call("hosted", informational_only=True)))
    assert intermediate_text_signal(resp) is None


def test_no_signal_for_hosted_response_text() -> None:
    hosted = SimpleNamespace(type="search_tool_call", provider_hosted=True)
    resp = _make_response(_make_msg(_text("checking"), hosted, _text("answer")))

    assert intermediate_text_signal(resp) is None


def test_no_signal_for_an_empty_response() -> None:
    resp = _make_response(_make_msg())
    assert intermediate_text_signal(resp) is None


def test_no_signal_for_a_function_result() -> None:
    """A function_result is not a function_call."""
    resp = _make_response(_make_msg(_fn_result()))
    assert intermediate_text_signal(resp) is None
