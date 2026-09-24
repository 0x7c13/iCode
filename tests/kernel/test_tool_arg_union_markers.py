# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for union-marker and discriminator location resolution in tool argument guidance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

import pytest
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    Tag,
    ValidationError,
    WrapValidator,
)

from chrys.kernel._tool_arg_errors import _argument_validation_message
from tests.kernel._tool_arg_helpers import _message_for


def test_argument_guidance_prefers_union_marker_over_sibling_field_name() -> None:
    class Cat(BaseModel):
        pet_type: Literal["cat"]
        dog: int

    class Dog(BaseModel):
        pet_type: Literal["dog"]
        age: str

    class PetArguments(BaseModel):
        pet: Cat | Dog = Field(discriminator="pet_type")

    arguments = {"pet": {"pet_type": "dog", "age": 7}}
    message = _message_for(PetArguments, arguments, tool_name="pet_tool", input_model=PetArguments)
    assert "pet.age: expected string, got integer" in message
    assert "pet.dog" not in message


def test_argument_guidance_drops_markers_at_mapping_unions() -> None:
    class MapUnionArguments(BaseModel):
        value: dict[str, int] | str

    arguments = {"value": {"a": "oops"}}
    message = _message_for(MapUnionArguments, arguments, tool_name="map_tool", input_model=MapUnionArguments)
    assert "value.a: expected integer" in message
    assert "dict[str,int]" not in message
    assert "value.str" not in message


def test_argument_guidance_navigates_optional_mapping_keys() -> None:
    class OptionalMapArguments(BaseModel):
        child: dict[str, int] | None = None

    arguments = {"child": {"a": "oops"}}
    message = _message_for(OptionalMapArguments, arguments, tool_name="opt_map_tool", input_model=OptionalMapArguments)
    assert "child.a: expected integer" in message

    # A nullable single-variant union gets no branch marker, so even a
    # marker-shaped key ("str", "dict[str,int]") is a real mapping key there.
    for key in ("str", "dict[str,int]"):
        arguments = {"child": {key: "oops"}}
        message = _message_for(
            OptionalMapArguments, arguments, tool_name="opt_map_tool", input_model=OptionalMapArguments
        )
        assert "expected integer" in message
        assert key in message


def test_argument_guidance_selects_marker_inside_nullable_discriminated_union() -> None:
    class Cat(BaseModel):
        kind: Literal["cat"]
        age: int

    class Dog(BaseModel):
        kind: Literal["dog"]
        age: str

    class NullablePetArguments(BaseModel):
        pet: Annotated[Cat | Dog, Field(discriminator="kind")] | None = None

    arguments = {"pet": {"kind": "dog", "age": 7}}
    message = _message_for(NullablePetArguments, arguments, tool_name="null_pet_tool", input_model=NullablePetArguments)
    assert "pet.age: expected string, got integer" in message


def test_argument_guidance_keeps_property_that_matches_variant_title() -> None:
    class Details(BaseModel):
        value: str = Field(alias="Details")

    class HolderArguments(BaseModel):
        child: Details | None = None

    arguments = {"child": {"Details": 7}}
    schema = HolderArguments.model_json_schema()
    with pytest.raises(ValidationError) as caught:
        HolderArguments.model_validate(arguments)
    # Nullable single-variant unions get no branch marker in ``loc``, so the
    # part must resolve as the real property even though it matches the
    # variant's title.
    assert [error["loc"] for error in caught.value.errors()] == [("child", "Details")]
    message = _argument_validation_message(
        tool_name="holder_tool",
        arguments=arguments,
        arguments_unparseable=False,
        schema=schema,
        exception=caught.value,
        reject_unexpected=True,
        input_model=HolderArguments,
    )
    assert "child.Details: expected string, got integer" in message


def test_argument_guidance_discriminator_tag_outranks_sibling_property_path() -> None:
    class Nested(BaseModel):
        age: int

    class Cat(BaseModel):
        kind: Literal["cat"]
        dog: Nested

    class Dog(BaseModel):
        kind: Literal["dog"]
        age: str

    class PetArguments(BaseModel):
        pet: Cat | Dog = Field(discriminator="kind")

    arguments = {"pet": {"kind": "dog", "age": 7}}
    # ``Cat.dog.age`` exists structurally, but the discriminator tag ``dog``
    # is authoritative and must win over that sibling property path.
    message = _message_for(PetArguments, arguments, tool_name="pet_tool", input_model=PetArguments)
    assert "pet.age: expected string, got integer" in message
    assert "pet.dog" not in message


def test_argument_guidance_resolves_deeply_nested_union_locations() -> None:
    level_one = dict[str, int] | str
    level_two = dict[str, level_one] | str
    level_three = dict[str, level_two] | str
    level_four = dict[str, level_three] | str
    level_five = dict[str, level_four] | str
    level_six = dict[str, level_five] | str
    level_seven = dict[str, level_six] | str

    class DeepArguments(BaseModel):
        value: level_seven

    arguments: dict[str, object] = {"value": "seed"}
    payload: object = "oops"
    for _ in range(7):
        payload = {"a": payload}
    arguments = {"value": payload}
    message = _message_for(DeepArguments, arguments, tool_name="deep_tool", input_model=DeepArguments)
    # Best-first ordering plus branch-and-bound keeps the state cap from
    # truncating the correct all-keys reading of the deepest location.
    assert "value.a.a.a.a.a.a.a: expected integer" in message
    assert "dict[str," not in message


def test_argument_guidance_selects_inner_tag_after_outer_wrapper_marker() -> None:
    class PlainNested(BaseModel):
        age: int

    class Plain(BaseModel):
        dog: PlainNested

    class Cat(BaseModel):
        kind: Literal["cat"]
        age: int

    class Dog(BaseModel):
        kind: Literal["dog"]
        age: str

    class WrapUnionArguments(BaseModel):
        value: Plain | Annotated[Cat | Dog, Field(discriminator="kind")] | None = None

    arguments = {"value": {"kind": "dog", "age": 7}}
    # The loc carries an outer wrapper marker before the inner tag
    # (('value', 'tagged-union[Cat,Dog]', 'dog', 'age')); selection must stay
    # available after the generic skip, and the tag must beat Plain.dog.age.
    message = _message_for(WrapUnionArguments, arguments, tool_name="wrap_tool", input_model=WrapUnionArguments)
    assert "value.age: expected string, got integer" in message
    assert "value.dog.age" not in message


def test_argument_guidance_trusts_title_markers_at_multi_variant_unions() -> None:
    class NestedB(BaseModel):
        age: int

    class A(BaseModel):
        B: NestedB

    class B(BaseModel):
        age: str

    class UntaggedUnionArguments(BaseModel):
        pet: A | B

    arguments = {"pet": {"age": 7}}
    # Loc ('pet', 'B', 'age'): at a union with two non-null variants Pydantic
    # always writes the branch marker, so the title match 'B' is
    # authoritative and must not lose to the sibling exact path A.B.age.
    message = _message_for(
        UntaggedUnionArguments, arguments, tool_name="untagged_tool", input_model=UntaggedUnionArguments
    )
    assert "pet.age: expected string, got integer" in message
    assert "pet.B.age" not in message


def test_argument_guidance_keeps_real_property_named_like_variant_title_after_skip() -> None:
    class NestedB(BaseModel):
        age: int

    class Plain(BaseModel):
        B: NestedB

    class B(BaseModel):
        age: str

    class WrappedVariantArguments(BaseModel):
        value: Annotated[Plain, AfterValidator(lambda v: v)] | B

    arguments = {"value": {"B": {"age": "not-int"}}}
    # Loc ('value', 'function-after[...], Plain]', 'B', 'age'): the generic
    # skip consumes this union level's one marker, so the following 'B' must
    # read as Plain's real property, never as a second marker selecting the
    # sibling variant B (which would swallow the key and report value.age).
    message = _message_for(
        WrappedVariantArguments, arguments, tool_name="wrapped_tool", input_model=WrappedVariantArguments
    )
    assert "value.B.age: expected integer" in message
    assert "value.age: expected" not in message


def test_argument_guidance_recovers_inner_markers_flattened_into_one_union() -> None:
    class Cat(BaseModel):
        kind: Literal["cat"]
        age: int

    class Dog(BaseModel):
        kind: Literal["dog"]
        age: str

    class Plain(BaseModel):
        name: str

    class FlattenedArguments(BaseModel):
        value: Plain | Annotated[Cat | Dog, AfterValidator(lambda v: v)]

    arguments = {"value": {"kind": "dog", "age": 7}}
    # Pydantic keeps two union levels in the loc (('value',
    # 'function-after[..., union[Cat,Dog]]', 'Dog', 'age')) but schema
    # generation flattens both into one anyOf, so the inner marker 'Dog'
    # lands on the already-skipped node. The demoted post-skip selection must
    # still recover it — no variant has a real 'Dog' key to prefer here.
    message = _message_for(FlattenedArguments, arguments, tool_name="flat_tool", input_model=FlattenedArguments)
    assert "value.age: expected string, got integer" in message
    assert "value.Dog.age" not in message


def test_argument_guidance_prefers_mapping_key_over_title_after_non_union_skip() -> None:
    class B(BaseModel):
        age: str

    class MappingArguments(BaseModel):
        value: Annotated[dict[str, int], AfterValidator(lambda v: v)] | B

    arguments = {"value": {"B": "oops"}}
    # Loc ('value', 'function-after[..., dict[str,int]]', 'B'): the skipped
    # marker wrapped a NON-union, so no flattened union level follows it —
    # 'B' must read as the real dictionary key (additionalProperties), never
    # as a selection of the sibling variant titled B, which would steer the
    # model to the wrong branch.
    message = _message_for(MappingArguments, arguments, tool_name="mapping_tool", input_model=MappingArguments)
    assert "value.B: expected integer" in message
    assert "value: expected" not in message


def test_argument_guidance_selects_flattened_variants_after_schemaless_wrapper() -> None:
    class Cat(BaseModel):
        kind: Literal["cat"]
        age: int

    class Dog(BaseModel):
        kind: Literal["dog"]
        age: str

    class Plain(BaseModel):
        name: str

    class WrapArguments(BaseModel):
        value: Plain | Annotated[Cat | Dog, WrapValidator(lambda v, handler: handler(v))]

    arguments = {"value": {"kind": "dog", "age": 7}}
    # A wrap validator's marker (`function-wrap[<lambda>()]`) records nothing
    # about what it wrapped, yet the union underneath it was flattened and
    # its 'Dog' marker follows at the same node. Selection must survive the
    # schema-less skip — at a weight below every real-key reading.
    message = _message_for(WrapArguments, arguments, tool_name="wrap_tool", input_model=WrapArguments)
    assert "value.age: expected string, got integer" in message
    assert "value.Dog.age" not in message


def test_argument_guidance_ignores_poisoned_union_tag_text() -> None:
    class B(BaseModel):
        age: str

    class PoisonedArguments(BaseModel):
        value: Annotated[dict[str, int], Tag("union[dict]")] | B

    arguments = {"value": {"B": "oops"}}
    # Tag() lets schema authors write arbitrary marker text, so `union[` in
    # loc ('value', 'union[dict]', 'B') is no proof of a flattened union
    # level. The classifier demands corroboration — a reachable variant
    # title inside the marker — before granting selection rights, so the
    # real dictionary key 'B' must win over the sibling variant titled B.
    message = _message_for(PoisonedArguments, arguments, tool_name="poisoned_tool", input_model=PoisonedArguments)
    assert "value.B: expected integer" in message
    assert "value: expected" not in message


def test_argument_guidance_denies_nested_tag_after_non_union_skip() -> None:
    class Cat(BaseModel):
        kind: Literal["cat"]
        age: int

    class Dog(BaseModel):
        kind: Literal["dog"]
        age: str

    class MixedArguments(BaseModel):
        value: (
            Annotated[dict[str, int], AfterValidator(lambda v: v)] | Annotated[Cat | Dog, Field(discriminator="kind")]
        )

    arguments = {"value": {"dog": "oops"}}
    # Loc ('value', 'function-after[..., dict[str,int]]', 'dog'): the skipped
    # marker named the DICT variant, so the following 'dog' is that dict's
    # real key — the nested discriminated union's identical tag must not
    # outrank the additionalProperties reading after a non-union skip.
    message = _message_for(MixedArguments, arguments, tool_name="mixed_tool", input_model=MixedArguments)
    assert "value.dog: expected integer" in message
    assert "value.age" not in message


def test_argument_guidance_blind_selection_loses_to_any_real_key_path() -> None:
    class B(BaseModel):
        age: str

    class SubsidyArguments(BaseModel):
        value: Annotated[dict[str, dict[str, int]], WrapValidator(lambda v, handler: handler(v))] | B

    arguments = {"value": {"B": {"age": "oops"}}}
    # Loc ('value', 'function-wrap[...]', 'B', 'age'): a small local weight
    # is not enough — selecting sibling B (then exact 'age') would outscore
    # the two cheap additionalProperties moves in summed weight. The blind
    # selection carries a penalty so ANY full real-key reading beats it.
    message = _message_for(SubsidyArguments, arguments, tool_name="subsidy_tool", input_model=SubsidyArguments)
    assert "value.B.age: expected integer" in message
    assert "value.age: expected" not in message


def test_argument_guidance_rejects_unbounded_title_substring_corroboration() -> None:
    class B(BaseModel):
        age: str

    class SubstringArguments(BaseModel):
        value: Annotated[dict[str, int], Tag("union[Business]")] | B

    arguments = {"value": {"B": "oops"}}
    # 'B' appears inside 'union[Business]' only as a fragment of another
    # word; corroboration requires delimiter-bounded occurrences, so the
    # marker stays uncorroborated and the real dictionary key wins.
    message = _message_for(SubstringArguments, arguments, tool_name="substring_tool", input_model=SubstringArguments)
    assert "value.B: expected integer" in message
    assert "value: expected" not in message


def test_argument_guidance_corroborates_markers_via_ref_names_despite_custom_titles() -> None:
    class Feline(BaseModel):
        model_config = ConfigDict(title="FelineTitle")

        kind: Literal["cat"]
        age: int

    class Canine(BaseModel):
        model_config = ConfigDict(title="CanineTitle")

        kind: Literal["dog"]
        age: str

    class NestedAge(BaseModel):
        age: int

    class PlainDog(BaseModel):
        dog: NestedAge

    class TitledArguments(BaseModel):
        value: PlainDog | Annotated[Feline | Canine, Field(discriminator="kind")] | None = None

    arguments = {"value": {"kind": "dog", "age": 5}}
    # The marker 'tagged-union[Feline,Canine]' uses CORE names (class/$defs
    # keys) while the advertised titles are customized, so corroboration must
    # also consult $ref and discriminator-mapping names — otherwise the
    # nested tag select is wrongly blocked and Plain's sibling dog.age wins.
    message = _message_for(TitledArguments, arguments, tool_name="titled_tool", input_model=TitledArguments)
    assert "value.age: expected string, got integer" in message
    assert "value.dog.age" not in message


def test_argument_guidance_consumes_consecutive_wrapper_markers() -> None:
    class OuterPlain(BaseModel):
        outer: str

    class InnerPlain(BaseModel):
        inner: str

    class Cat(BaseModel):
        kind: Literal["cat"]
        age: int

    class Dog(BaseModel):
        kind: Literal["dog"]
        age: str

    inner = InnerPlain | Annotated[Cat | Dog, AfterValidator(lambda v: v)]

    class DoubleWrapArguments(BaseModel):
        value: OuterPlain | Annotated[inner, WrapValidator(lambda v, handler: handler(v))]  # type: ignore[valid-type]

    arguments = {"value": {"kind": "dog", "age": 7}}
    # Loc ('value', 'function-wrap[...]', 'function-after[..., union[Cat,Dog]]',
    # 'Dog', 'age'): the schema-less wrapper hid a second wrapper level, so a
    # second same-node skip must be allowed — but only for a part carrying
    # its own marker evidence, keeping single-skip semantics elsewhere.
    message = _message_for(
        DoubleWrapArguments, arguments, tool_name="double_wrap_tool", input_model=DoubleWrapArguments
    )
    assert "missing 'value.inner'" in message
    assert 'value.kind: expected "cat"' in message
    assert "function-wrap" not in message
    assert "function-after" not in message


def test_argument_guidance_bounds_oversized_marker_corroboration() -> None:
    class B(BaseModel):
        age: str

    class OversizedArguments(BaseModel):
        value: Annotated[dict[str, int], Tag("union[B]" + "Z" * 5000)] | B

    arguments = {"value": {"B": "oops"}}
    # Corroboration work is errors x names x marker length, so oversized
    # marker text is never scanned: no genuine Pydantic repr approaches the
    # bound, and an adversarial 'union[B]...' + padding must classify opaque
    # (real key wins) instead of buying selection rights with a bounded 'B'.
    message = _message_for(OversizedArguments, arguments, tool_name="oversized_tool", input_model=OversizedArguments)
    assert "value.B: expected integer" in message
    assert "value: expected" not in message


def test_argument_guidance_survives_adversarially_wide_wrapped_unions() -> None:
    def make_level(inner: object) -> object:
        union: object = None
        for index in range(6):
            model = type(f"Level{index}", (BaseModel,), {"__annotations__": {"child": inner}})
            wrapped = Annotated[model, AfterValidator(lambda v: v)]
            union = wrapped if union is None else union | wrapped
        return union

    nested: object = int
    for _ in range(4):
        nested = make_level(nested)

    class WideArguments(BaseModel):
        value: nested  # type: ignore[valid-type]

    arguments = {"value": {"child": {"child": {"child": {"child": "not-int"}}}}}
    schema = WideArguments.model_json_schema()
    with pytest.raises(ValidationError) as caught:
        WideArguments.model_validate(arguments)
    # 6^4 = 1296 validation errors, each loc alternating wrapper markers with
    # the shared 'child' key. The greedy first-choice dive secures the full
    # reading before the state-capped refinement, and the validation-error cap
    # keeps message construction bounded.
    assert len(caught.value.errors()) > 1000
    message = _argument_validation_message(
        tool_name="wide_tool",
        arguments=arguments,
        arguments_unparseable=False,
        schema=schema,
        exception=caught.value,
        reject_unexpected=True,
        input_model=WideArguments,
    )
    assert "value.child.child.child.child: expected integer" in message
    assert "function-after" not in message


def _boolean_discriminated_pets() -> type[BaseModel]:
    class ActivePet(BaseModel):
        active: Literal[True]
        age: int

    class InactivePet(BaseModel):
        active: Literal[False]
        age: str

    class PetArguments(BaseModel):
        pet: ActivePet | InactivePet = Field(discriminator="active")

    return PetArguments


def _numeric_discriminated_pets() -> type[BaseModel]:
    class CatV1(BaseModel):
        version: Literal[1]
        age: int

    class DogV2(BaseModel):
        version: Literal[2]
        age: str

    class PetArguments(BaseModel):
        pet: CatV1 | DogV2 = Field(discriminator="version")

    return PetArguments


def _string_discriminated_pets() -> type[BaseModel]:
    class Cat(BaseModel):
        pet_type: Literal["cat"]
        age: int

    class Dog(BaseModel):
        pet_type: Literal["dog"]
        age: str

    class PetArguments(BaseModel):
        pet: Cat | Dog = Field(discriminator="pet_type")

    return PetArguments


@pytest.mark.parametrize(
    ("build_arguments_model", "pet"),
    [
        pytest.param(_boolean_discriminated_pets, {"active": False, "age": 7}, id="boolean"),
        pytest.param(_numeric_discriminated_pets, {"version": 2, "age": 7}, id="numeric"),
        pytest.param(_string_discriminated_pets, {"pet_type": "dog", "age": 7}, id="string"),
    ],
)
def test_argument_guidance_selects_union_branch_by_discriminator(
    build_arguments_model: Callable[[], type[BaseModel]], pet: dict[str, object]
) -> None:
    arguments = {"pet": pet}
    message = _message_for(build_arguments_model(), arguments, tool_name="pet_tool")
    assert "pet.age: expected string, got integer" in message
    assert "expected integer, got integer" not in message


def test_argument_guidance_drops_union_branch_markers_from_locations() -> None:
    class Cat(BaseModel):
        pet_type: Literal["cat"]
        meows: int

    class Dog(BaseModel):
        pet_type: Literal["dog"]
        barks: float

    class PetArguments(BaseModel):
        pet: Cat | Dog = Field(discriminator="pet_type")

    arguments = {"pet": {"pet_type": "dog", "barks": "loud"}}
    message = _message_for(PetArguments, arguments, tool_name="pet_tool")
    assert "pet.barks: expected number" in message
    assert "pet.dog" not in message

    class UnionArguments(BaseModel):
        value: str | list[str]

    union_arguments = {"value": 7}
    union_message = _message_for(UnionArguments, union_arguments, tool_name="union_tool")
    assert union_message.count("value: expected string | [string], got integer") == 1
    assert "value." not in union_message


def test_argument_guidance_reports_unknown_fields_inside_union_variants() -> None:
    class Cat(BaseModel):
        model_config = ConfigDict(extra="forbid")

        pet_type: Literal["cat"]
        meows: int

    class Dog(BaseModel):
        model_config = ConfigDict(extra="forbid")

        pet_type: Literal["dog"]
        barks: float

    class PetArguments(BaseModel):
        pet: Cat | Dog = Field(discriminator="pet_type")

    arguments = {"pet": {"pet_type": "dog", "barks": 1.0, "bad_field": 1}}
    message = _message_for(PetArguments, arguments, tool_name="pet_tool")
    assert "unknown 'pet.bad_field'" in message
    assert "unknown 'pet'" not in message
    assert "pet.dog" not in message
