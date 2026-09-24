# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for immutable locale-neutral message definitions and references."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from enum import Enum, IntEnum
from pathlib import Path

import pytest

from chrys.foundation.i18n import (
    DisplayBlock,
    DisplayPath,
    DisplaySequence,
    MessageDef,
    MessageRef,
    msg,
)
from chrys.foundation.i18n.formatting import format_message


class _Count(IntEnum):
    ONE = 1


class _Choice(Enum):
    VALUE = "value"


def test_msg_creates_definition_and_bind_creates_occurrence() -> None:
    definition = msg("test.greeting", fallback="Hello, {name}!")
    reference = definition.bind(name="Chrys")

    assert isinstance(definition, MessageDef)
    assert isinstance(reference, MessageRef)
    assert reference.definition is definition
    assert reference.args == (("name", "Chrys"),)
    assert reference.count is None
    assert format_message(reference) == "Hello, Chrys!"


def test_message_definitions_and_references_are_immutable() -> None:
    definition = msg("test.immutable", fallback="Immutable")
    reference = definition.bind()

    with pytest.raises(FrozenInstanceError):
        definition.fallback = "Changed"  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        reference.count = 2  # ty: ignore[invalid-assignment]


def test_direct_message_definition_construction_is_closed() -> None:
    with pytest.raises(TypeError):
        MessageDef(key="test.direct", fallback="Direct")  # ty: ignore[missing-argument]


def test_replacing_a_message_definition_is_closed() -> None:
    definition = msg("test.replace", fallback="Original")

    with pytest.raises(TypeError, match="InitVar"):
        replace(definition, fallback="Changed")


def test_replacing_a_message_reference_is_closed() -> None:
    reference = msg("test.replace_reference", fallback="Original").bind()

    with pytest.raises(TypeError, match="InitVar"):
        replace(reference, count=1)


def test_type_based_message_definition_construction_is_closed() -> None:
    definition = msg("test.dynamic_type", fallback="Original")

    with pytest.raises(TypeError):
        type(definition)(key="test.dynamic_type", fallback="Changed")  # ty: ignore[missing-argument]


def test_direct_message_reference_construction_is_closed() -> None:
    definition = msg("test.direct_reference", fallback="Direct")

    with pytest.raises(TypeError):
        MessageRef(definition=definition)  # ty: ignore[missing-argument]


@pytest.mark.parametrize("key", ["test.bad#key", "single", "Test.uppercase"])
def test_message_keys_must_follow_the_namespaced_grammar(key: str) -> None:
    with pytest.raises(ValueError, match="key"):
        msg(key, fallback="Invalid key")


def test_bound_arguments_have_deterministic_name_order() -> None:
    definition = msg("test.argument_order", fallback="{first} then {second}")

    reference = definition.bind(second="B", first="A")

    assert reference.args == (("first", "A"), ("second", "B"))
    assert format_message(reference) == "A then B"


def test_plural_definition_requires_count() -> None:
    definition = msg("test.plural_required", fallback="One file", plural_fallback="{count} files")

    with pytest.raises(ValueError, match="count"):
        definition.bind()


def test_singular_definition_rejects_count() -> None:
    definition = msg("test.singular_count", fallback="One file")

    with pytest.raises(ValueError, match="count"):
        definition.bind(count=1)


def test_message_types_cannot_be_subclassed() -> None:
    with pytest.raises(TypeError, match="subclass"):

        class _ForgedDef(MessageDef):
            pass

    with pytest.raises(TypeError, match="subclass"):

        class _ForgedRef(MessageRef):
            def __post_init__(self, _token: object) -> None:
                pass


def test_a_caller_created_unset_sentinel_is_not_an_omission() -> None:
    from chrys.foundation.i18n.messages import _UnsetCount

    with pytest.raises(ValueError, match="count"):
        msg("test.sentinel_forgery", fallback="One file").bind(count=_UnsetCount())

    plural = msg("test.sentinel_forgery_plural", fallback="One file", plural_fallback="{count} files")
    with pytest.raises(TypeError, match="count"):
        plural.bind(count=_UnsetCount())


def test_singular_definition_rejects_an_explicit_none_count() -> None:
    definition = msg("test.singular_none_count", fallback="One file")

    with pytest.raises(ValueError, match="count"):
        definition.bind(count=None)


def test_plural_definition_rejects_an_explicit_none_count() -> None:
    definition = msg("test.plural_none_count", fallback="One file", plural_fallback="{count} files")

    with pytest.raises(TypeError, match="count"):
        definition.bind(count=None)


def test_plural_count_is_injected_for_formatting() -> None:
    definition = msg("test.plural_injected", fallback="One file", plural_fallback="{count} files")

    singular = definition.bind(count=1)
    plural = definition.bind(count=3)

    assert singular.args == ()
    assert singular.count == 1
    assert format_message(singular) == "One file"
    assert format_message(plural) == "3 files"


def test_duplicate_count_argument_is_rejected() -> None:
    definition = msg("test.duplicate_count", fallback="One file", plural_fallback="{count} files")
    args = {"count": 2}

    with pytest.raises(TypeError, match="count"):
        definition.bind(count=1, **args)


@pytest.mark.parametrize("count", [True, _Count.ONE])
def test_plural_count_must_be_exactly_int(count: object) -> None:
    definition = msg("test.exact_count", fallback="One file", plural_fallback="{count} files")

    with pytest.raises(TypeError, match="count"):
        definition.bind(count=count)  # ty: ignore[invalid-argument-type]


def test_plural_forms_share_the_non_count_slot_schema() -> None:
    with pytest.raises(ValueError, match="schema"):
        msg(
            "test.plural_schema",
            fallback="One file from {path}",
            plural_fallback="{count} files",
        )


def test_singular_definition_cannot_reference_count() -> None:
    with pytest.raises(ValueError, match="count"):
        msg("test.reserved_count", fallback="Found {count} files")


def test_closed_message_argument_union_accepts_all_supported_kinds() -> None:
    definition = msg(
        "test.supported_args",
        fallback="{text} {number} {path}\n{block}",
        multiline=True,
    )

    reference = definition.bind(
        text="value",
        number=7,
        path=DisplayPath(Path("file.txt")),
        block=DisplayBlock("line one\nline two"),
    )

    assert format_message(reference) == "value 7 file.txt\nline one\nline two"


@pytest.mark.parametrize("value", [True, 1.5, None, _Choice.VALUE])
def test_closed_message_argument_union_rejects_unsupported_scalars(value: object) -> None:
    definition = msg("test.unsupported_arg", fallback="Value: {value}")

    with pytest.raises(TypeError, match="argument"):
        definition.bind(value=value)  # ty: ignore[invalid-argument-type]


def test_display_sequence_renders_with_fixed_separator_and_is_immutable() -> None:
    sequence = DisplaySequence(("alpha", 2, DisplayPath("gamma")))
    definition = msg("test.sequence", fallback="Items: {items}")

    assert format_message(definition.bind(items=sequence)) == "Items: alpha, 2, gamma"
    assert sequence.values == ("alpha", 2, DisplayPath("gamma"))
    with pytest.raises(FrozenInstanceError):
        sequence.values = ()  # ty: ignore[invalid-assignment]


def test_empty_display_sequence_is_legal_and_renders_empty_text() -> None:
    definition = msg("test.empty_sequence", fallback="Items: {items}")

    assert format_message(definition.bind(items=DisplaySequence(()))) == "Items: "


@pytest.mark.parametrize(
    "value",
    [
        DisplaySequence(("nested",)),
        DisplayBlock("block"),
    ],
)
def test_display_sequence_rejects_nested_and_block_elements(value: object) -> None:
    with pytest.raises(TypeError, match="element"):
        DisplaySequence((value,))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("value", [True, _Count.ONE, _Choice.VALUE])
def test_display_sequence_rejects_non_scalar_element_subclasses(value: object) -> None:
    with pytest.raises(TypeError, match="element"):
        DisplaySequence((value,))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("value", ["", "   ", "​"])
def test_display_sequence_rejects_invisible_elements(value: str) -> None:
    with pytest.raises(ValueError, match="visible"):
        DisplaySequence((value,))


def test_scalar_control_sequences_are_sanitized_at_rendering() -> None:
    definition = msg("test.scalar_sanitization", fallback="Value: {value}")

    rendered = format_message(definition.bind(value="safe\x1b[31mred"))

    assert rendered == "Value: safe�[31mred"
    assert "\x1b" not in rendered


def test_display_sequence_elements_use_single_line_sanitization() -> None:
    definition = msg("test.sequence_sanitization", fallback="Items: {items}")
    sequence = DisplaySequence(("one\rtwo\nthree",))

    assert format_message(definition.bind(items=sequence)) == "Items: one�two�three"


def test_display_path_uses_single_line_sanitization() -> None:
    definition = msg("test.path_sanitization", fallback="Path: {path}")

    assert format_message(definition.bind(path=DisplayPath("one\ntwo"))) == "Path: one�two"


def test_display_block_preserves_lf_and_sanitizes_other_controls() -> None:
    definition = msg("test.block_sanitization", fallback="Block:\n{block}", multiline=True)

    rendered = format_message(definition.bind(block=DisplayBlock("one\ntwo\x1b[0m")))

    assert rendered == "Block:\none\ntwo�[0m"


def test_every_textual_argument_kind_neutralizes_lone_surrogates() -> None:
    surrogate = "img\udcff.png"
    scalar = msg("test.scalar_surrogate", fallback="Value: {value}")
    sequence = msg("test.sequence_surrogate", fallback="Items: {items}")
    block = msg("test.block_surrogate", fallback="Block:\n{block}", multiline=True)
    path = msg("test.path_surrogate", fallback="Path: {path}")

    assert format_message(scalar.bind(value=surrogate)) == "Value: img\\udcff.png"
    assert format_message(sequence.bind(items=DisplaySequence((surrogate,)))) == "Items: img\\udcff.png"
    assert format_message(block.bind(block=DisplayBlock(f"one\n{surrogate}"))) == "Block:\none\nimg\\udcff.png"
    assert format_message(path.bind(path=DisplayPath(surrogate))) == "Path: img\\udcff.png"
    for rendered in (
        format_message(scalar.bind(value=surrogate)),
        format_message(block.bind(block=DisplayBlock(surrogate))),
    ):
        rendered.encode("utf-8")


@pytest.mark.parametrize("fallback", ["", "  \n ", "​", "​ ​"])
def test_message_fallback_must_have_visible_content(fallback: str) -> None:
    with pytest.raises(ValueError, match="visible"):
        msg("test.invisible_fallback", fallback=fallback, multiline=True)


def test_plural_fallback_must_have_visible_content() -> None:
    with pytest.raises(ValueError, match="visible"):
        msg("test.invisible_plural", fallback="One", plural_fallback="​")


@pytest.mark.parametrize("fallback", ["Unsafe \x1b[31m", "[bold]Unsafe[/bold]"])
def test_message_fallback_rejects_unsafe_authored_text(fallback: str) -> None:
    with pytest.raises(ValueError):
        msg("test.unsafe_fallback", fallback=fallback)


def test_single_line_definition_rejects_lf() -> None:
    with pytest.raises(ValueError, match="LF"):
        msg("test.single_line", fallback="line one\nline two")


def test_multiline_definition_accepts_lf() -> None:
    definition = msg("test.multiline", fallback="line one\nline two", multiline=True)

    assert format_message(definition.bind()) == "line one\nline two"


def test_display_block_cannot_bind_into_single_line_definition() -> None:
    definition = msg("test.block_single_line", fallback="Block: {block}")

    with pytest.raises(ValueError, match="multiline"):
        definition.bind(block=DisplayBlock("block"))


@pytest.mark.parametrize("value", ["", "​"])
def test_bind_rejects_an_invisible_formatted_fallback(value: str) -> None:
    definition = msg("test.invisible_bound", fallback="{value}")

    with pytest.raises(ValueError, match="visible"):
        definition.bind(value=value)


def test_bind_rejects_an_integer_beyond_the_decimal_conversion_limit() -> None:
    definition = msg("test.large_int", fallback="Value: {value}")

    with pytest.raises(ValueError, match="decimal"):
        definition.bind(value=10**5000)


def test_bind_rejects_exceptions_and_arbitrary_objects() -> None:
    definition = msg("test.object_arg", fallback="Value: {value}")

    for value in (RuntimeError("failure"), object()):
        with pytest.raises(TypeError, match="argument"):
            definition.bind(value=value)  # ty: ignore[invalid-argument-type]
