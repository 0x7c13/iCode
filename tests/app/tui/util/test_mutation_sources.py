# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for TUI mutation-source display helpers."""

from __future__ import annotations

from chrys.app.tui.util.mutation_sources import merge_mutation_source
from chrys.service.mutations.types import MutationSource


def test_merge_mutation_source_prefers_new_explicit_source() -> None:
    assert merge_mutation_source(MutationSource.WRITE_FILE.value, MutationSource.EDIT_FILE.value) == (
        MutationSource.EDIT_FILE.value
    )


def test_merge_mutation_source_preserves_current_when_new_source_is_empty() -> None:
    assert merge_mutation_source(MutationSource.SHELL.value, "") == MutationSource.SHELL.value


def test_merge_mutation_source_preserves_implicit_marker_from_either_side() -> None:
    assert merge_mutation_source(MutationSource.IMPLICIT.value, MutationSource.EDIT_FILE.value) == (
        MutationSource.IMPLICIT.value
    )
    assert merge_mutation_source(MutationSource.WRITE_FILE.value, MutationSource.IMPLICIT.value) == (
        MutationSource.IMPLICIT.value
    )
