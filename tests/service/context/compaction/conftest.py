# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Opt-in fixtures shared by the compaction test modules."""

from __future__ import annotations

import pytest

from chrys.service.context.compaction.last_words import LastWordsGenerator


@pytest.fixture
def no_note_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the note-length floor; the floor tests opt back in in-body.

    Most generator tests script single-word notes; the production floor would
    divert them into retry loops irrelevant to what they assert.  Kept opt-in
    (not autouse) so it never silently rewrites a production constant for the
    sibling modules that do not instantiate a real generator.
    """
    monkeypatch.setattr(LastWordsGenerator, "_MIN_NOTE_CHARS", 0)
