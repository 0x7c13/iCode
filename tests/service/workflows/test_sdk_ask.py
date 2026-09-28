# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow SDK structured ``ctx.ask``: Question/Option/Answer validation, dispatch and the wire form."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from chrys.foundation.models import ask_user
from chrys.service.workflows.sdk import Answer, NodeContext, Option, Question, _ask


class _Asker:
    """A wire-level ask callback that records every call and replies with canned answers."""

    def __init__(self, *answers: dict[str, Any]) -> None:
        self.answers = list(answers)
        self.calls: list[list[dict[str, Any]]] = []

    async def __call__(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.calls.append(questions)
        return self.answers


def _ctx(asker: _Asker) -> NodeContext:
    return NodeContext(emit=lambda _text: None, ask=asker)


def test_caps_match_the_chat_ask_user_caps() -> None:
    assert _ask.MAX_QUESTIONS == ask_user.MAX_ASK_USER_QUESTIONS
    assert _ask.MAX_OPTIONS == ask_user.MAX_ASK_USER_OPTIONS
    assert _ask.MAX_LABEL_CHARS == ask_user.MAX_ASK_USER_LABEL_CHARS
    assert _ask.MAX_DESCRIPTION_CHARS == ask_user.MAX_ASK_USER_DESCRIPTION_CHARS
    assert _ask.MAX_HEADER_CHARS == ask_user.MAX_ASK_USER_HEADER_CHARS


def test_question_normalizes_options_and_strips_labels() -> None:
    question = Question("Pick?", header="Pick", options=["  a  ", Option(" b ", "the b")])

    assert question.options == (Option("a"), Option("b", "the b"))
    assert isinstance(question.options, tuple)
    assert Option(" c ").label == "c"
    # Question text is never capped below the frame: a review question can embed a whole draft.
    assert len(Question("x" * 50_000).question) == 50_000


@pytest.mark.parametrize(
    ("build", "error", "match"),
    [
        (lambda: Option(1), TypeError, "Option.label must be a str"),
        (lambda: Option("a", None), TypeError, "Option.description must be a str"),
        (lambda: Option("   "), ValueError, "Option.label must not be blank"),
        (lambda: Option("x" * 201), ValueError, "longer than 200"),
        (lambda: Option("a", "x" * 501), ValueError, "longer than 500"),
        (lambda: Option("\ud800"), ValueError, "lone surrogate"),
        (lambda: Question(1), TypeError, "Question.question must be a str"),
        (lambda: Question("  \n "), ValueError, "Question.question must not be blank"),
        (lambda: Question("q\udc80"), ValueError, "lone surrogate"),
        (lambda: Question("q", header="x" * 65), ValueError, "longer than 64"),
        (lambda: Question("q", header=None), TypeError, "Question.header must be a str"),
        (lambda: Question("q", options="ab"), TypeError, "list or tuple"),
        (lambda: Question("q", options={"a"}), TypeError, "list or tuple"),
        (lambda: Question("q", options=[1]), TypeError, "list or tuple of Option or str"),
        (lambda: Question("q", options=[str(n) for n in range(9)]), ValueError, "more than 8"),
        (lambda: Question("q", options=["a", " a"]), ValueError, "unique"),
        (lambda: Question("q", options=["a"], multi_select=1), TypeError, "must be a bool"),
        (lambda: Question("q", multi_select=True), ValueError, "needs options"),
    ],
)
def test_invalid_questions_fail_at_construction(build: Any, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        build()


def test_answer_choice_answered_and_types() -> None:
    assert Answer().choice is None and not Answer().answered
    assert Answer(selected=("a",)).choice == "a"
    assert Answer(text="custom").answered and Answer(text="custom").choice is None
    assert Answer(selected=["a", "b"]).selected == ("a", "b")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="several options"):
        _ = Answer(selected=("a", "b")).choice
    with pytest.raises(TypeError):
        Answer(selected="ab")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Answer(text=None)  # type: ignore[arg-type]


def test_ask_str_sends_one_open_question_and_returns_the_text() -> None:
    asker = _Asker({"selected": [], "text": "blue"})

    assert asyncio.run(_ctx(asker).ask("Color?")) == "blue"
    assert asker.calls == [[{"question": "Color?", "header": "", "options": [], "multi_select": False}]]


def test_ask_question_returns_one_answer_and_sends_the_wire_form() -> None:
    asker = _Asker({"selected": ["deep"], "text": "go"})
    question = Question("How deep?", header="Depth", options=[Option("quick", "fast"), "deep"])

    answer = asyncio.run(_ctx(asker).ask(question))

    assert answer == Answer(selected=("deep",), text="go")
    assert asker.calls == [
        [
            {
                "question": "How deep?",
                "header": "Depth",
                "options": [{"label": "quick", "description": "fast"}, {"label": "deep", "description": ""}],
                "multi_select": False,
            }
        ]
    ]


@pytest.mark.parametrize("container", [list, tuple])
def test_ask_sequence_returns_a_tuple_in_question_order(container: Any) -> None:
    asker = _Asker({"selected": ["main"], "text": ""}, {"selected": ["API", "UI"], "text": ""})
    questions = container(
        [
            Question("Branch?", options=["main", "release"]),
            Question("Areas?", options=["API", "Storage", "UI"], multi_select=True),
        ]
    )

    branch, areas = asyncio.run(_ctx(asker).ask(questions))

    assert branch.choice == "main"
    assert areas.selected == ("API", "UI")
    assert [question["multi_select"] for question in asker.calls[0]] == [False, True]


def test_a_one_element_list_still_returns_a_tuple() -> None:
    answers = asyncio.run(_ctx(_Asker({"selected": [], "text": "x"})).ask([Question("q")]))
    assert answers == (Answer(text="x"),)


@pytest.mark.parametrize(
    ("prompt", "error", "match"),
    [
        (1, TypeError, "expects a str, a Question"),
        ({"question": "q"}, TypeError, "expects a str, a Question"),
        ([], ValueError, "1 to 5"),
        ([Question("q")] * 6, ValueError, "1 to 5"),
        (["q"], TypeError, "Question"),
        (" \n", ValueError, r"ctx\.ask\(\) prompt must not be blank"),
    ],
    ids=["int", "dict", "empty", "six", "str-item", "blank-str"],
)
def test_ask_rejects_bad_arguments_before_asking(prompt: Any, error: type[Exception], match: str) -> None:
    asker = _Asker()
    with pytest.raises(error, match=match):
        asyncio.run(_ctx(asker).ask(prompt))
    assert asker.calls == []


def test_a_question_mutated_after_construction_is_rechecked_before_it_is_sent() -> None:
    asker = _Asker({"selected": [], "text": ""})
    question = Question("q", options=["a"])
    object.__setattr__(question, "options", ("a", "a"))

    with pytest.raises(ValueError, match="unique"):
        asyncio.run(_ctx(asker).ask(question))
    assert asker.calls == []
