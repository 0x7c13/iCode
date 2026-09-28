# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Main-process ``ctx.ask`` codec: strict question decoding, answer mapping and bounded summaries."""

from __future__ import annotations

from typing import Any

import pytest

from chrys.foundation.models.ask_user import AskUserAnswer, AskUserOption, AskUserQuestion
from chrys.service.workflows.asks import answer_summary, answers_to_wire, ask_summary, questions_from_wire
from chrys.service.workflows.protocol import ProtocolError
from chrys.service.workflows.sdk import Option, Question
from chrys.service.workflows.sdk._ask import questions_to_wire

AREAS = AskUserQuestion(
    question="Which areas?",
    header="Areas",
    options=(AskUserOption("API"), AskUserOption("Storage"), AskUserOption("UI")),
    multi_select=True,
)
BRANCH = AskUserQuestion(question="Which branch?", options=(AskUserOption("main"), AskUserOption("release")))
OPEN = AskUserQuestion(question="Anything else?", header="Notes")


def _wire(**changes: Any) -> dict[str, Any]:
    question: dict[str, Any] = {
        "question": "Pick?",
        "header": "Pick",
        "options": [{"label": "a", "description": "the a"}, {"label": "b", "description": ""}],
        "multi_select": False,
    }
    question.update(changes)
    return question


def test_sdk_questions_decode_to_the_chat_model() -> None:
    sdk_questions = (
        Question("Which areas?", header="Areas", options=["API", "Storage", "UI"], multi_select=True),
        Question("Which branch?", options=[Option("main"), Option("release")]),
        Question("Anything else?", header="Notes"),
        Question("x" * 50_000, header="h" * 64, options=[Option("l" * 200, "d" * 500)] + [str(n) for n in range(7)]),
    )

    decoded = questions_from_wire(questions_to_wire(sdk_questions))

    assert decoded[:3] == (AREAS, BRANCH, OPEN)
    assert len(decoded[3].options) == 8 and decoded[3].options[0] == AskUserOption("l" * 200, "d" * 500)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        pytest.param({"question": "q"}, "1 to 5", id="not-a-list"),
        pytest.param([], "1 to 5", id="empty"),
        pytest.param([_wire()] * 6, "1 to 5", id="six"),
        pytest.param(["q"], "question is malformed", id="question-not-dict"),
        pytest.param([{**_wire(), "extra": 1}], "question is malformed", id="question-extra-key"),
        pytest.param([{k: v for k, v in _wire().items() if k != "header"}], "malformed", id="question-missing-key"),
        pytest.param([_wire(question=1)], "question must be valid text", id="question-not-str"),
        pytest.param([_wire(question=" \n")], "question is blank", id="question-blank"),
        pytest.param([_wire(question="q\ud800")], "question must be valid text", id="question-surrogate"),
        pytest.param([_wire(header="h" * 65)], "longer than 64", id="header-too-long"),
        pytest.param([_wire(multi_select=1)], "must be a bool", id="multi-select-int"),
        pytest.param([_wire(options=({"label": "a", "description": ""},))], "options must be a list", id="tuple"),
        pytest.param([_wire(options=[{"label": str(n), "description": ""} for n in range(9)])], "at most 8", id="nine"),
        pytest.param([_wire(options=["a"])], "option is malformed", id="option-not-dict"),
        pytest.param([_wire(options=[{"label": "a"}])], "option is malformed", id="option-missing-key"),
        pytest.param([_wire(options=[{"label": " a", "description": ""}])], "stripped", id="label-unstripped"),
        pytest.param([_wire(options=[{"label": "", "description": ""}])], "non-empty", id="label-empty"),
        pytest.param([_wire(options=[{"label": "l" * 201, "description": ""}])], "200", id="label-too-long"),
        pytest.param([_wire(options=[{"label": "a", "description": "d" * 501}])], "500", id="description-too-long"),
        pytest.param(
            [_wire(options=[{"label": "a", "description": ""}, {"label": "a", "description": "again"}])],
            "unique",
            id="duplicate-labels",
        ),
        pytest.param([_wire(options=[], multi_select=True)], "needs options", id="multi-select-without-options"),
    ],
)
def test_malformed_questions_are_a_protocol_error(raw: object, match: str) -> None:
    with pytest.raises(ProtocolError, match=match):
        questions_from_wire(raw)


@pytest.mark.parametrize(
    ("question", "answer", "wire"),
    [
        pytest.param(AREAS, AskUserAnswer(values=("UI", "API")), {"selected": ["API", "UI"], "text": ""}, id="order"),
        pytest.param(
            AREAS, AskUserAnswer(values=("Storage",), note="soon"), {"selected": ["Storage"], "text": "soon"}, id="note"
        ),
        pytest.param(BRANCH, AskUserAnswer(values=("feature/x",)), {"selected": [], "text": "feature/x"}, id="custom"),
        # The dialog records no click provenance: typing a label exactly is choosing it.
        pytest.param(BRANCH, AskUserAnswer(values=("main",)), {"selected": ["main"], "text": ""}, id="typed-label"),
        pytest.param(BRANCH, AskUserAnswer(values=("Main",)), {"selected": [], "text": "Main"}, id="case-differs"),
        pytest.param(OPEN, AskUserAnswer(values=("later",)), {"selected": [], "text": "later"}, id="open"),
        pytest.param(BRANCH, AskUserAnswer(), {"selected": [], "text": ""}, id="skipped"),
    ],
)
def test_answers_map_to_the_sdk_shape(question: AskUserQuestion, answer: AskUserAnswer, wire: dict[str, Any]) -> None:
    assert answers_to_wire((question,), (answer,)) == [wire]


@pytest.mark.parametrize(
    "answer",
    [AskUserAnswer(values=("one", "two")), AskUserAnswer(values=("own",), note="extra"), AskUserAnswer(note="orphan")],
)
def test_unvalidated_answers_are_refused(answer: AskUserAnswer) -> None:
    with pytest.raises(ValueError, match="validated"):
        answers_to_wire((AREAS,), (answer,))


def test_summaries_prefix_each_of_several_questions_with_its_header_or_number() -> None:
    assert ask_summary((BRANCH,)) == "Which branch?"
    assert ask_summary((AREAS, BRANCH, OPEN)) == "Areas: Which areas?\nQ2: Which branch?\nNotes: Anything else?"
    assert answer_summary((BRANCH,), (AskUserAnswer(values=("main",)),)) == "main"
    assert answer_summary((BRANCH,), (AskUserAnswer(),)) == "(no answer)"
    assert (
        answer_summary(
            (AREAS, BRANCH, OPEN),
            (AskUserAnswer(values=("API", "UI"), note="both"), AskUserAnswer(), AskUserAnswer(values=("later",))),
        )
        == "Areas: API, UI — both\nQ2: (no answer)\nNotes: later"
    )
