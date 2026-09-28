# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The main-process side of ``ctx.ask``: wire questions in, answers out, bounded summaries.

The SDK (``sdk/_ask.py``) guarantees every rule :func:`questions_from_wire`
checks, so a violation means a broken worker and ends the connection. The
chat ask-user model carries the questions to the dialog and back; answers are
validated by the shared :func:`validate_ask_user_answers` before they get here.
"""

from __future__ import annotations

from typing import Any, cast

from chrys.foundation.models.ask_user import (
    MAX_ASK_USER_DESCRIPTION_CHARS,
    MAX_ASK_USER_HEADER_CHARS,
    MAX_ASK_USER_LABEL_CHARS,
    MAX_ASK_USER_OPTIONS,
    MAX_ASK_USER_QUESTIONS,
    AskUserAnswer,
    AskUserOption,
    AskUserQuestion,
)
from chrys.foundation.util.unicode_scalars import find_unpaired_surrogate
from chrys.service.workflows.protocol import ProtocolError

_QUESTION_KEYS = frozenset({"question", "header", "options", "multi_select"})
_OPTION_KEYS = frozenset({"label", "description"})
NO_ANSWER = "(no answer)"


def questions_from_wire(raw: object) -> tuple[AskUserQuestion, ...]:
    """Strictly decode an ask frame's ``questions``; raises :class:`ProtocolError`."""
    if type(raw) is not list or not 1 <= len(raw) <= MAX_ASK_USER_QUESTIONS:
        raise ProtocolError(f"ask needs 1 to {MAX_ASK_USER_QUESTIONS} questions.")
    return tuple(_question_from_wire(item) for item in raw)


def _text(raw: dict[str, object], key: str, limit: int | None = None) -> str:
    value = raw[key]
    if type(value) is not str or find_unpaired_surrogate(value) >= 0:
        raise ProtocolError(f"ask {key} must be valid text.")
    if limit is not None and len(value) > limit:
        raise ProtocolError(f"ask {key} is longer than {limit} characters.")
    return value


def _question_from_wire(item: object) -> AskUserQuestion:
    if type(item) is not dict or set(item) != _QUESTION_KEYS:
        raise ProtocolError("ask question is malformed.")
    raw = cast("dict[str, object]", item)
    question = _text(raw, "question")
    if not question.strip():
        raise ProtocolError("ask question is blank.")
    header = _text(raw, "header", MAX_ASK_USER_HEADER_CHARS)
    multi_select = raw["multi_select"]
    options_raw = raw["options"]
    if type(multi_select) is not bool:
        raise ProtocolError("ask multi_select must be a bool.")
    if type(options_raw) is not list or len(options_raw) > MAX_ASK_USER_OPTIONS:
        raise ProtocolError(f"ask options must be a list of at most {MAX_ASK_USER_OPTIONS}.")
    options = tuple(_option_from_wire(option) for option in options_raw)
    labels = {option.label for option in options}
    if len(labels) != len(options):
        raise ProtocolError("ask option labels must be unique.")
    if multi_select and not options:
        raise ProtocolError("ask multi_select needs options.")
    return AskUserQuestion(question=question, header=header, options=options, multi_select=multi_select)


def _option_from_wire(item: object) -> AskUserOption:
    if type(item) is not dict or set(item) != _OPTION_KEYS:
        raise ProtocolError("ask option is malformed.")
    raw = cast("dict[str, object]", item)
    label = _text(raw, "label", MAX_ASK_USER_LABEL_CHARS)
    # The shared answer validator strips values, so only a stripped label can come back selected.
    if not label or label != label.strip():
        raise ProtocolError("ask option label must be stripped and non-empty.")
    return AskUserOption(label=label, description=_text(raw, "description", MAX_ASK_USER_DESCRIPTION_CHARS))


def answers_to_wire(questions: tuple[AskUserQuestion, ...], answers: tuple[AskUserAnswer, ...]) -> list[dict[str, Any]]:
    """Map validated answers to the SDK's ``{"selected", "text"}`` shape.

    Values naming offered labels are the selection (option order) and the note
    is the text; a single other value is custom text. The dialog records no
    click provenance, so a typed label counts as selected.
    """
    wire: list[dict[str, Any]] = []
    for question, answer in zip(questions, answers, strict=True):
        labels = [option.label for option in question.options]
        if answer.values and all(value in labels for value in answer.values):
            chosen = set(answer.values)
            wire.append({"selected": [label for label in labels if label in chosen], "text": answer.note})
        elif len(answer.values) == 1 and not answer.note:
            wire.append({"selected": [], "text": answer.values[0]})
        elif not answer.values and not answer.note:
            wire.append({"selected": [], "text": ""})
        else:
            raise ValueError("Workflow ask answers must be validated before they are mapped.")
    return wire


def _prefix(index: int, question: AskUserQuestion) -> str:
    return question.header or f"Q{index + 1}"


def _answer_text(answer: AskUserAnswer) -> str:
    text = ", ".join(answer.values)
    if answer.note:
        text = f"{text} — {answer.note}"
    return text or NO_ANSWER


def ask_summary(questions: tuple[AskUserQuestion, ...]) -> str:
    """English text for the run store, each of several questions under its header (the journal bounds it)."""
    if len(questions) == 1:
        return questions[0].question
    return "\n".join(f"{_prefix(index, question)}: {question.question}" for index, question in enumerate(questions))


def answer_summary(questions: tuple[AskUserQuestion, ...], answers: tuple[AskUserAnswer, ...]) -> str:
    """English text for the run store and ``WorkflowNodeAnswered``, each of several answers under its header."""
    if len(answers) == 1:
        return _answer_text(answers[0])
    return "\n".join(
        f"{_prefix(index, question)}: {_answer_text(answer)}"
        for index, (question, answer) in enumerate(zip(questions, answers, strict=True))
    )
