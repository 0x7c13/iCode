# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the shared ask-user model and wire validators."""

from __future__ import annotations

import json
import operator
from dataclasses import FrozenInstanceError

import pytest

from chrys.foundation.events.types import QuestionToUser
from chrys.foundation.models.ask_user import (
    AskUserAnswer,
    AskUserOption,
    AskUserQuestion,
    Cancelled,
    format_ask_user_result,
    parse_ask_user_answers,
    parse_ask_user_questions,
    parse_recorded_ask_user_questions,
    validate_request_input_params,
    validate_request_input_response,
)


def _questions() -> tuple[AskUserQuestion, ...]:
    return (
        AskUserQuestion(
            question="Which library?",
            header="Library",
            options=(
                AskUserOption("tenacity", "Use the existing dependency."),
                AskUserOption("custom", "Build the loop locally."),
            ),
        ),
        AskUserQuestion(
            question="Which targets?",
            header="Targets",
            options=(AskUserOption("TUI"), AskUserOption("ACP")),
            multi_select=True,
        ),
        AskUserQuestion(question="Rollout plan?", header="Rollout"),
    )


def test_legacy_result_is_byte_identical() -> None:
    questions = (AskUserQuestion(question="Pick?"),)
    assert format_ask_user_result(questions, (AskUserAnswer(values=("X",)),)) == "User response: X"


def test_structured_result_is_lossless_json_with_explicit_unanswered() -> None:
    questions = _questions()
    answers = (
        AskUserAnswer(values=('value, with separator\nand "quotes"',), note="note:\nforged"),
        AskUserAnswer(values=("界面", "ACP")),
        AskUserAnswer(),
    )
    encoded = format_ask_user_result(questions, answers)
    assert (
        encoded
        == """{
  "responses": [
    {
      "question": "Which library?",
      "answers": [
        "value, with separator\\nand \\"quotes\\""
      ],
      "note": "note:\\nforged"
    },
    {
      "question": "Which targets?",
      "answers": [
        "界面",
        "ACP"
      ]
    },
    {
      "question": "Rollout plan?",
      "answers": [],
      "unanswered": true
    }
  ]
}"""
    )
    assert json.loads(encoded) == {
        "responses": [
            {
                "question": "Which library?",
                "answers": ['value, with separator\nand "quotes"'],
                "note": "note:\nforged",
            },
            {"question": "Which targets?", "answers": ["界面", "ACP"]},
            {"question": "Rollout plan?", "answers": [], "unanswered": True},
        ]
    }


def test_answer_parser_always_has_requested_cardinality() -> None:
    assert parse_ask_user_answers([{"values": ["A"]}], question_count=3) == (
        AskUserAnswer(values=("A",)),
        AskUserAnswer(),
        AskUserAnswer(),
    )
    assert parse_ask_user_answers([{}, {}, {}], question_count=1) == (AskUserAnswer(),)


def test_models_and_event_payload_are_immutable() -> None:
    question = AskUserQuestion(question="Pick?", options=(AskUserOption("A"),))
    answer = AskUserAnswer(values=("A",), note="Because")
    event = QuestionToUser(questions=(question,))
    with pytest.raises(FrozenInstanceError):
        question.question = "Changed"  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        answer.values = ("B",)  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        answer.note = "Changed"  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        operator.setitem(event.questions, 0, question)  # ty: ignore[no-matching-overload]


def test_question_parser_reads_native_only_and_replay_reads_recorded_legacy() -> None:
    mixed = {
        "questions": [{"question": "Native?", "options": [{"label": "A", "description": "Why"}]}],
        "question": "Ignored?",
        "options": ["B"],
    }
    native = (AskUserQuestion(question="Native?", options=(AskUserOption("A", "Why"),)),)
    assert parse_ask_user_questions(mixed) == native
    assert parse_recorded_ask_user_questions(mixed) == native
    legacy = {"question": "Legacy?", "options": ["A"]}
    assert parse_ask_user_questions(legacy) == ()
    assert parse_recorded_ask_user_questions(legacy) == (
        AskUserQuestion(question="Legacy?", options=(AskUserOption("A"),)),
    )
    assert parse_recorded_ask_user_questions({"question": "   "}) == ()
    assert parse_recorded_ask_user_questions([{"question": "List?"}]) == (AskUserQuestion(question="List?"),)


def test_request_validator_requires_questions_and_rejects_unknown_keys() -> None:
    questions = _questions()
    payload = {
        "sessionId": "s",
        "requestId": "r",
        "questions": [
            {
                "question": question.question,
                "header": question.header,
                "multiSelect": question.multi_select,
                "options": [{"label": option.label, "description": option.description} for option in question.options],
            }
            for question in questions
        ],
    }
    assert validate_request_input_params(payload) == questions
    for mirror in ({"question": "Which library?"}, {"options": ["tenacity"]}, {"text": "x"}):
        with pytest.raises(ValueError):
            validate_request_input_params({**payload, **mirror})
    malformed = json.loads(json.dumps(payload))
    malformed["questions"][0]["options"][0]["unknown"] = True
    with pytest.raises(ValueError):
        validate_request_input_params(malformed)
    legacy = {
        "sessionId": "s",
        "requestId": "r",
        "question": "Legacy?",
        "options": ["A", "B"],
    }
    with pytest.raises(ValueError):
        validate_request_input_params(legacy)
    with pytest.raises(ValueError):
        validate_request_input_params({"sessionId": "s", "requestId": "r"})


@pytest.mark.parametrize(
    "answers",
    [
        [{"values": ["tenacity"]}],
        [{"values": ["tenacity", "custom"]}, {}, {}],
        [{"values": ["tenacity", "tenacity"]}, {}, {}],
        [{"values": [""]}, {}, {}],
        [{"values": [], "note": "orphan"}, {}, {}],
        [{"values": ["free text"], "note": "second free text"}, {}, {}],
        [{"values": ["tenacity"], "note": 7}, {}, {}],
    ],
)
def test_malformed_structured_response_is_cancelled_atomically(answers: object) -> None:
    outcome = validate_request_input_response({"answers": answers}, questions=_questions())
    assert isinstance(outcome, Cancelled)


def test_response_validator_cancels_without_answers_and_accepts_structured() -> None:
    questions = _questions()
    for payload in ({"text": "legacy"}, {}, {"cancelled": True}, {"answers": [], "text": "x"}):
        assert isinstance(validate_request_input_response(payload, questions=questions), Cancelled)
    outcome = validate_request_input_response(
        {
            "answers": [
                {"values": ["tenacity"], "note": "only here"},
                {"values": ["TUI", "ACP"], "note": ""},
                {"values": [], "note": ""},
            ],
        },
        questions=questions,
    )
    assert outcome == (
        AskUserAnswer(values=("tenacity",), note="only here"),
        AskUserAnswer(values=("TUI", "ACP")),
        AskUserAnswer(),
    )


def test_parsers_repair_surrogates_so_questions_and_answers_can_cross_a_strict_wire() -> None:
    questions = parse_ask_user_questions(
        [
            {
                "question": "Pick\ud800?",
                "header": "\udc00",
                "options": [{"label": "x\ud83d", "description": "\ud83d\ude00 pair"}],
            }
        ]
    )
    assert questions[0].question == "Pick\ufffd?"
    assert questions[0].header == "\ufffd"
    assert questions[0].options[0].label == "x\ufffd"
    # An adjacent high/low pair is what a JSON decoder would have joined.
    assert questions[0].options[0].description == "😀 pair"

    answers = parse_ask_user_answers([{"values": ["\ud800"], "note": "n\udfff"}], question_count=1)
    assert answers == (AskUserAnswer(values=("\ufffd",), note="n\ufffd"),)
    # Everything that came through the parsers encodes strictly.
    for text in (questions[0].question, questions[0].header, *answers[0].values, answers[0].note):
        text.encode("utf-8")


def test_ready_made_answers_and_validated_responses_are_repaired_too() -> None:
    # Answers built directly (the TUI, legacy transcript replay) skip the
    # dict parser, so the repair has to run on the dataclass path as well.
    repaired = parse_ask_user_answers((AskUserAnswer(values=("\ud800",), note="n\udc00"),), question_count=1)
    assert repaired == (AskUserAnswer(values=("\ufffd",), note="n\ufffd"),)
    clean = AskUserAnswer(values=("ok",), note="fine")
    assert parse_ask_user_answers((clean,), question_count=1)[0] is clean
    format_ask_user_result((AskUserQuestion("Pick?"),), (AskUserAnswer(values=("\ud800",)),)).encode("utf-8")

    # A high/low pair passes the wire check as a pair, so the strict validator
    # must hand back the joined scalar, not two code units.
    questions = (AskUserQuestion("Pick?", options=(AskUserOption("😀"),)),)
    outcome = validate_request_input_response(
        {"answers": [{"values": ["\ud83d\ude00"], "note": " \ud83d\ude00 "}]}, questions=questions
    )
    assert outcome == (AskUserAnswer(values=("😀",), note="😀"),)


def test_wire_validators_reject_unpaired_surrogates_like_the_transport() -> None:
    base = {"sessionId": "s", "requestId": "r", "questions": [{"question": "Pick?"}]}
    with pytest.raises(ValueError, match="surrogate"):
        validate_request_input_params({**base, "questions": [{"question": "Pick\ud800?"}]})
    with pytest.raises(ValueError):
        validate_request_input_params({**base, "_meta": {"k\udc00": 1}})
    # A joined pair is fine on the wire, so it stays accepted.
    (question,) = validate_request_input_params({**base, "questions": [{"question": "\ud83d\ude00?"}]})
    assert question.question == "😀?"

    questions = (AskUserQuestion("Pick?"),)
    assert isinstance(
        validate_request_input_response({"answers": [{"values": ["\ud800"], "note": ""}]}, questions=questions),
        Cancelled,
    )
    assert isinstance(
        validate_request_input_response({"answers": [{"values": ["ok"], "note": "\udfff"}]}, questions=questions),
        Cancelled,
    )
