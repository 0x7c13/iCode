# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Structured questions for ``ctx.ask`` (pure stdlib, Python 3.9).

``Question``/``Option`` describe what a node asks; ``Answer`` is what comes
back. Everything is checked at construction, so a malformed question fails in
the node body with a traceback, and checked again when it is serialized, so a
frozen object mutated with ``object.__setattr__`` never reaches the wire.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

# Copies of the chat ask-user caps (``chrys.foundation.models.ask_user``), which
# this package cannot import; tests pin them to the originals.
MAX_QUESTIONS = 5
MAX_OPTIONS = 8
MAX_LABEL_CHARS = 200
MAX_DESCRIPTION_CHARS = 500
MAX_HEADER_CHARS = 64


def _check_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{name} is not valid Unicode (lone surrogate).") from None
    return value


def _check_option(option: Option) -> str:
    label = _check_text(option.label, "Option.label").strip()
    if not label:
        raise ValueError("Option.label must not be blank.")
    if len(label) > MAX_LABEL_CHARS:
        raise ValueError(f"Option.label is longer than {MAX_LABEL_CHARS} characters.")
    if len(_check_text(option.description, "Option.description")) > MAX_DESCRIPTION_CHARS:
        raise ValueError(f"Option.description is longer than {MAX_DESCRIPTION_CHARS} characters.")
    return label


def _check_question(question: Question) -> tuple[Option, ...]:
    if not _check_text(question.question, "Question.question").strip():
        raise ValueError("Question.question must not be blank.")
    if len(_check_text(question.header, "Question.header")) > MAX_HEADER_CHARS:
        raise ValueError(f"Question.header is longer than {MAX_HEADER_CHARS} characters.")
    if not isinstance(question.multi_select, bool):
        raise TypeError("Question.multi_select must be a bool.")
    if not isinstance(question.options, (list, tuple)):
        raise TypeError("Question.options must be a list or tuple of Option or str.")
    if len(question.options) > MAX_OPTIONS:
        raise ValueError(f"Question.options holds more than {MAX_OPTIONS} options.")
    options: list[Option] = []
    for option in question.options:
        if isinstance(option, str):
            option = Option(option)
        elif isinstance(option, Option):
            if _check_option(option) != option.label:
                option = Option(option.label, option.description)
        else:
            raise TypeError("Question.options must be a list or tuple of Option or str.")
        options.append(option)
    labels = [option.label for option in options]
    if len(set(labels)) != len(labels):
        raise ValueError("Question.options labels must be unique.")
    if question.multi_select and not options:
        raise ValueError("Question.multi_select needs options to select from.")
    return tuple(options)


@dataclass(frozen=True)
class Option:
    """One choice offered by a :class:`Question`.

    ``label`` is what the user picks and what :attr:`Answer.selected` returns;
    surrounding whitespace is removed. ``description`` is shown under it.
    """

    label: str
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _check_option(self))


@dataclass(frozen=True)
class Question:
    """One question of a ``ctx.ask`` dialog.

    ``question`` is Markdown; ``header`` is the short tab label; ``options``
    are :class:`Option` objects or plain labels (normalized to a tuple of
    :class:`Option`). Without options the question is free text; with options
    the user picks one (or several with ``multi_select=True``), and can always
    type a custom answer instead.
    """

    question: str
    header: str = ""
    options: Sequence[Union[Option, str]] = ()
    multi_select: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", _check_question(self))


@dataclass(frozen=True)
class Answer:
    """The answer to one :class:`Question`.

    ``selected`` holds the offered labels the answer names, in option order —
    clicking an option and typing its exact label are the same answer.
    ``text`` holds everything else: a custom answer that matches no label, or
    a note typed alongside a selection. Both empty means the question was
    skipped.
    """

    selected: tuple[str, ...] = ()
    text: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.selected, (list, tuple)) or not all(isinstance(item, str) for item in self.selected):
            raise TypeError("Answer.selected must be a tuple of str.")
        if not isinstance(self.text, str):
            raise TypeError("Answer.text must be a str.")
        object.__setattr__(self, "selected", tuple(self.selected))

    @property
    def answered(self) -> bool:
        """Whether anything was selected or typed."""
        return bool(self.selected) or bool(self.text)

    @property
    def choice(self) -> Optional[str]:
        """The one selected label, or None; raises ValueError when several are selected."""
        if len(self.selected) > 1:
            raise ValueError("Answer.choice is ambiguous: several options are selected; use Answer.selected.")
        return self.selected[0] if self.selected else None


def question_to_wire(question: Question) -> dict[str, Any]:
    """Re-check *question* and return its wire form."""
    if not isinstance(question, Question):
        raise TypeError("ctx.ask() questions must be Question objects.")
    options = _check_question(question)
    return {
        "question": question.question,
        "header": question.header,
        "options": [{"label": option.label, "description": option.description} for option in options],
        "multi_select": question.multi_select,
    }


def questions_to_wire(questions: Sequence[Question]) -> list[dict[str, Any]]:
    """Check the count and every question of one ``ctx.ask`` call."""
    if not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f"ctx.ask() takes 1 to {MAX_QUESTIONS} questions.")
    return [question_to_wire(question) for question in questions]


def answer_from_wire(raw: dict[str, Any]) -> Answer:
    """Build an :class:`Answer` from a reply the worker host already shape-checked."""
    return Answer(selected=tuple(raw["selected"]), text=raw["text"])
