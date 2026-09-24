# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pydantic fixture models for the null-overlay policy tests, with the call counters their serializers mutate."""

from __future__ import annotations

import dataclasses
import enum
import typing
from collections import deque
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from typing import Annotated, Any, Literal, NewType, NotRequired, TypedDict

import annotated_types
import typing_extensions
from pydantic import (
    AfterValidator,
    AnyUrl,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PositiveInt,
    RootModel,
    SecretStr,
    StringConstraints,
    computed_field,
    field_serializer,
    model_serializer,
    model_validator,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.fields import FieldInfo
from pydantic_core import core_schema

from tests.kernel._fakes import _NullableValue

_FIELD_INFO_SERIALIZER_CARRIER = FieldInfo.from_annotation(
    Annotated[list[_NullableValue], PlainSerializer(lambda values: [{"owned": True} for _ in values])]
)


class _FieldInfoCarrierArguments(BaseModel):
    items: Annotated[list[_NullableValue], _FIELD_INFO_SERIALIZER_CARRIER]


class _FieldConstraintArguments(BaseModel):
    count: int = Field(gt=0)
    positive: PositiveInt


class _CoreSchemaChild(BaseModel):
    value: str | None = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        schema = handler(source)
        schema["serialization"] = core_schema.plain_serializer_function_ser_schema(lambda _: {"owned": True})
        return schema


class _CoreSchemaArguments(BaseModel):
    child: _CoreSchemaChild


@dataclasses.dataclass
class _CoreSchemaDataclass:
    value: str | None = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        schema = handler(source)
        schema["serialization"] = core_schema.plain_serializer_function_ser_schema(lambda _: {"owned": True})
        return schema


class _CoreSchemaDataclassArguments(BaseModel):
    child: _CoreSchemaDataclass


class _CoreSchemaKey(enum.StrEnum):
    VALUE = "value"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        schema = handler(source)
        schema["serialization"] = core_schema.plain_serializer_function_ser_schema(lambda _: "SERIALIZED")
        return schema


class _CoreSchemaKeyArguments(BaseModel):
    payload: dict[_CoreSchemaKey, _NullableValue]


class _RootDumpOverrideArguments(BaseModel):
    value: str | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {"owned": True}


class _NestedDumpOverride(BaseModel):
    value: str | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {"owned": True}


class _NestedDumpOverrideArguments(BaseModel):
    child: _NestedDumpOverride


class _CollisionOwned(BaseModel):
    value: str | None = None


class _CollisionConditional(BaseModel):
    value: str | None = None


class _ExcludeIfCollisionArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True, validate_by_name=True)

    owned: _CollisionOwned = Field(serialization_alias="slot")
    conditional: _CollisionConditional = Field(serialization_alias="slot", exclude_if=lambda _: False)


class _ExcludeIfPlainArguments(BaseModel):
    safe: str | None
    conditional: _CollisionConditional = Field(exclude_if=lambda _: False)


@dataclasses.dataclass
class _ComputedAliasBaseDataclass:
    owned: _NullableValue

    @computed_field
    @property
    def view(self) -> dict[str, Any]:
        return {"base": True}


@dataclasses.dataclass
class _ComputedAliasDerivedDataclass(_ComputedAliasBaseDataclass):
    @computed_field(alias="owned")
    @property
    def view(self) -> dict[str, Any]:
        return {}


class _ComputedAliasDataclassArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    child: _ComputedAliasDerivedDataclass


class _SlotBase(BaseModel):
    value: str | None = None


class _SlotSub(_SlotBase):
    extra: str | None = None


def _promote_slot(value: _SlotBase) -> _SlotSub:
    return _SlotSub(value=value.value, extra=None)


class _UnionLeft(BaseModel):
    kind: Literal["left"]
    value: str | None = None


class _UnionRight(BaseModel):
    kind: Literal["right"]
    value: str | None = None


class _SlotClassArguments(BaseModel):
    child: Annotated[_SlotBase, AfterValidator(_promote_slot)]
    other: _SlotSub | None = None
    choice: _UnionLeft | _UnionRight


class _ChildListRoot(RootModel[list[_NullableValue]]):
    pass


class _RootModelArguments(BaseModel):
    payload: _ChildListRoot


class _SerializationDefaultsRequiredArguments(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    value: str | None = None


class _NestedArguments(BaseModel):
    child: _NullableValue


class _TupleArguments(BaseModel):
    items: tuple[_NullableValue, ...]


class _PlainMembers(TypedDict):
    value: str | None
    optional: NotRequired[str | None]


class _TypedDictArguments(BaseModel):
    payload: _PlainMembers


@dataclasses.dataclass
class _StaticDataclass:
    required: str | None
    omitted: str | None = None
    non_none_default: str | None = "seed"
    factory: str | None = dataclasses.field(default_factory=lambda: None)


class _DataclassArguments(BaseModel):
    child: _StaticDataclass


@pydantic_dataclass
class _PydanticDataclass:
    value: str | None


class _PydanticDataclassArguments(BaseModel):
    child: _PydanticDataclass


class _ExtraArguments(BaseModel):
    model_config = ConfigDict(extra="allow")

    anchor: int = 1


class _ExcludedArguments(BaseModel):
    visible: str | None
    hidden: str | None = Field(exclude=True)


class _ComputedArguments(BaseModel):
    count: int

    @computed_field
    @property
    def ghost(self) -> None:
        return None


class _SplitAliasArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    value: str | None = Field(validation_alias="inputKey", serialization_alias="callableKey")


class _EmptyAliasArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    value: str | None = Field(alias="")


class _UnschemableComputedArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    value: str | None = Field(validation_alias="inKey", serialization_alias="outKey")

    @computed_field
    @property
    def factory(self) -> Callable[[], int]:
        return lambda: 1


_StdNewTypeChild = NewType("_StdNewTypeChild", _NullableValue)
_ExtNewTypeChild = typing_extensions.NewType("_ExtNewTypeChild", _NullableValue)


def _parse_int(value: Any) -> int:
    return int(value)


class _FullGateArguments(BaseModel):
    child: _NullableValue
    members: _PlainMembers
    items: list[_NullableValue]
    mapping: dict[str, _NullableValue]
    std_new: _StdNewTypeChild
    ext_new: _ExtNewTypeChild
    validated: Annotated[int, BeforeValidator(_parse_int)]
    choice: _NullableValue | None
    sequence: Sequence[_NullableValue]
    abstract_mapping: Mapping[str, _NullableValue]
    mutable_mapping: MutableMapping[str, _NullableValue]
    members_set: set[str | None]
    frozen_members: frozenset[str | None]
    mode: Literal["plain"]


class _Mode(enum.Enum):
    PLAIN = "plain"


class _EnumGateArguments(BaseModel):
    mode: _Mode


class _ValidationMetadataArguments(BaseModel):
    count: PositiveInt | None = None
    name: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)] | None = None
    score: Annotated[int, annotated_types.Gt(0)] | None = None
    interval: Annotated[int, annotated_types.Interval(gt=0, lt=10)] | None = None
    length: Annotated[str, annotated_types.Len(1, 5)] | None = None


class _SafeMetadataWithSerializerArguments(BaseModel):
    value: (
        Annotated[
            str,
            annotated_types.MinLen(1),
            PlainSerializer(lambda value: value.upper()),
        ]
        | None
    ) = None


class _SerializerGroupedMetadata(annotated_types.GroupedMetadata):
    def __iter__(self) -> Iterator[Any]:
        yield PlainSerializer(lambda _: {"grouped_owned": True})


class _GroupedSerializerArguments(BaseModel):
    safe: str | None
    child: Annotated[_NullableValue, _SerializerGroupedMetadata()]


_CoreSchemaNewType = NewType("_CoreSchemaNewType", _NullableValue)


def _new_type_core_schema(source: Any, handler: Any) -> Any:
    schema = dict(handler(_NullableValue))
    schema["serialization"] = core_schema.plain_serializer_function_ser_schema(lambda _: {"new_type_owned": True})
    return schema


_CoreSchemaNewType.__get_pydantic_core_schema__ = _new_type_core_schema  # type: ignore[attr-defined]


class _NewTypeSerializerArguments(BaseModel):
    child: _CoreSchemaNewType


class _CoreSchemaTypedDict(typing_extensions.TypedDict):
    child: _NullableValue


def _typed_dict_core_schema(cls: Any, source: Any, handler: Any) -> Any:
    schema = handler(source)
    schema["serialization"] = core_schema.plain_serializer_function_ser_schema(lambda _: {"typed_dict_owned": True})
    return schema


_CoreSchemaTypedDict.__get_pydantic_core_schema__ = classmethod(_typed_dict_core_schema)  # type: ignore[attr-defined]


class _TypedDictSerializerArguments(BaseModel):
    payload: _CoreSchemaTypedDict


class _CoreSchemaPropertyMeta(type(BaseModel)):
    @property
    def __get_pydantic_core_schema__(cls) -> Any:
        def hook(source: Any, handler: Any) -> Any:
            return core_schema.no_info_after_validator_function(
                lambda value: value,
                handler(source),
                serialization=core_schema.plain_serializer_function_ser_schema(lambda _: {"metaclass_owned": True}),
            )

        return hook


class _MetaclassSerializerArguments(BaseModel, metaclass=_CoreSchemaPropertyMeta):
    value: str | None


class _NestedPlainSerializerArguments(BaseModel):
    safe: str | None
    child: Annotated[_NullableValue, PlainSerializer(lambda _: {"plain_owned": True})]


_ModelSerializationNewType = NewType("_ModelSerializationNewType", _NullableValue)


def _model_serialization_core_schema(source: Any, handler: Any) -> Any:
    schema = dict(handler(_NullableValue))
    schema["serialization"] = core_schema.model_ser_schema(
        _NullableValue,
        core_schema.model_fields_schema({}),
    )
    return schema


_ModelSerializationNewType.__get_pydantic_core_schema__ = _model_serialization_core_schema  # type: ignore[attr-defined]


class _ModelSerializationArguments(BaseModel):
    child: _ModelSerializationNewType


class _CustomSlotSerializer:
    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        return core_schema.no_info_after_validator_function(
            lambda value: value,
            handler(source),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda _: "SERIALIZER_OWNED",
                when_used="always",
            ),
        )


class _CustomSlotSerializerArguments(BaseModel):
    safe: str | None = None
    value: Annotated[str | None, _CustomSlotSerializer()] = None


class _PydanticInternalSerializationArguments(BaseModel):
    url: AnyUrl
    secret: SecretStr


_RootSerializationNewType = NewType("_RootSerializationNewType", dict[str, Any] | None)


def _root_serialization_core_schema(source: Any, handler: Any) -> Any:
    schema = dict(handler(dict[str, Any] | None))
    schema["serialization"] = core_schema.plain_serializer_function_ser_schema(lambda _: {"owned": True})
    return schema


_RootSerializationNewType.__get_pydantic_core_schema__ = _root_serialization_core_schema  # type: ignore[attr-defined]


class _SerializedRootArguments(RootModel[_RootSerializationNewType]):
    @model_validator(mode="before")
    @classmethod
    def _empty_to_null(cls, value: Any) -> Any:
        return None if value == {} else value


class _AnyUrlWithSafeRootFieldArguments(BaseModel):
    safe: str | None
    url: AnyUrl


class _SerializationNamedFieldArguments(BaseModel):
    serialization: str
    child: _NullableValue


class _ModelFieldsSerializerMeta(type(BaseModel)):
    @property
    def __get_pydantic_core_schema__(cls) -> Any:
        def hook(source: Any, handler: Any) -> Any:
            schema = handler(source)
            schema["schema"]["serialization"] = core_schema.plain_serializer_function_ser_schema(
                lambda _: {"owned": True}
            )
            return schema

        return hook


class _ModelFieldsSerializerArguments(BaseModel, metaclass=_ModelFieldsSerializerMeta):
    value: str | None


class _RecursiveSerializedChild(BaseModel):
    next: _RecursiveSerializedChild | None = None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, bool]:
        return {"owned": True}


class _RecursiveSerializedArguments(BaseModel):
    child: _RecursiveSerializedChild | None


class _RecursivePlainChild(BaseModel):
    next: _RecursivePlainChild | None = None


class _RecursivePlainDegradedArguments(BaseModel):
    child: _RecursivePlainChild | None
    url: AnyUrl


class _DefaultedDataclassPayload(TypedDict):
    value: str | None


@pydantic_dataclass
class _DefaultedNestedDataclass:
    payload: _DefaultedDataclassPayload = dataclasses.field(default_factory=lambda: {"value": None})


class _DefaultedNestedDataclassArguments(BaseModel):
    child: _DefaultedNestedDataclass


@pydantic_dataclass
class _RequiredNestedDataclass:
    payload: _DefaultedDataclassPayload


class _RequiredNestedDataclassArguments(BaseModel):
    child: _RequiredNestedDataclass


@pydantic_dataclass
class _NonNoneDefaultDataclass:
    value: str | None = "seed"


class _NonNoneDefaultDataclassArguments(BaseModel):
    child: _NonNoneDefaultDataclass


class _AliasedFieldWithInternalNameExtraArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True, validate_by_name=False, extra="allow")

    value: str | None = Field(alias="wire")


class _ExcludedFieldWithAliasedExtraArguments(BaseModel):
    model_config = ConfigDict(extra="allow", serialize_by_alias=True)

    hidden: str | None = Field(
        default="seed",
        validation_alias="inputHidden",
        serialization_alias="x",
        exclude=True,
    )


class _EmittingFieldWithCollidingExtraArguments(BaseModel):
    model_config = ConfigDict(extra="allow", serialize_by_alias=True)

    visible: str | None = Field(
        default=None,
        validation_alias="inputVisible",
        serialization_alias="x",
    )


class _UnhashableAnnotation:
    __hash__ = None

    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        return core_schema.str_schema()


class _UnhashableAnnotationArguments(BaseModel):
    value: _UnhashableAnnotation()


class _InheritedAliasBase(typing_extensions.TypedDict):
    __pydantic_config__ = ConfigDict(alias_generator=str.upper)

    first: str | None


class _InheritedAliasPayload(_InheritedAliasBase):
    second: str | None


class _InheritedAliasArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    payload: _InheritedAliasPayload


class _PlainInheritedBase(typing_extensions.TypedDict):
    first: str | None


class _PlainInheritedPayload(_PlainInheritedBase):
    second: str | None


class _PlainInheritedArguments(BaseModel):
    payload: _PlainInheritedPayload


def _base_alias(name: str) -> str:
    return f"base_{name}"


def _own_alias(name: str) -> str:
    return f"own_{name}"


class _PrecedenceAliasBase(typing_extensions.TypedDict):
    __pydantic_config__ = ConfigDict(alias_generator=_base_alias)

    first: str | None


class _PrecedenceAliasPayload(_PrecedenceAliasBase):
    __pydantic_config__ = ConfigDict(alias_generator=_own_alias)

    second: str | None


class _PrecedenceAliasArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    payload: _PrecedenceAliasPayload


class _CoDeclaredTupleArguments(BaseModel):
    pair: tuple[Annotated[_SlotBase, AfterValidator(_promote_slot)], _SlotSub]


class _CoDeclaredUnionArguments(BaseModel):
    choice: _SlotBase | _SlotSub


class _UnknownContainerArguments(BaseModel):
    items: deque[int]


_NESTED_SERIALIZER_CALLS = 0


class _SerializedChild(BaseModel):
    value: str | None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        global _NESTED_SERIALIZER_CALLS
        _NESTED_SERIALIZER_CALLS += 1
        return {"owned": True}


class _NestedSerializerArguments(BaseModel):
    safe: str | None
    child: _SerializedChild


class _RootSerializerArguments(BaseModel):
    value: str | None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        return {"owned": True}


_FIELD_SERIALIZER_CALLS = 0


class _FieldSerializerArguments(BaseModel):
    safe: str | None
    text: str

    @field_serializer("text")
    def _serialize_text(self, value: str) -> str:
        global _FIELD_SERIALIZER_CALLS
        _FIELD_SERIALIZER_CALLS += 1
        return value.upper()


_KEY_SERIALIZER_CALLS = 0


def _upper_key(value: str) -> str:
    global _KEY_SERIALIZER_CALLS
    _KEY_SERIALIZER_CALLS += 1
    return value.upper()


_SerializedKey = Annotated[str, PlainSerializer(_upper_key)]


class _OptionalKeyChild(BaseModel):
    value: str | None = None


class _KeySerializerArguments(BaseModel):
    payload: dict[_SerializedKey, _OptionalKeyChild]


class _SupertypeAttributeChild(_NullableValue):
    __supertype__: typing.ClassVar[type] = _NullableValue

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        return {"owned": True}


class _SupertypeAttributeArguments(BaseModel):
    child: _SupertypeAttributeChild


class _AliasedTypedDict(TypedDict):
    value: Annotated[str | None, Field(serialization_alias="wire")]


class _AliasedTypedDictArguments(BaseModel):
    payload: _AliasedTypedDict


class _DefaultedTypedDict(TypedDict, total=False):
    value: Annotated[str | None, Field(default=None)]


class _DefaultedTypedDictArguments(BaseModel):
    payload: _DefaultedTypedDict


@dataclasses.dataclass
class _AliasedDataclass:
    value: Annotated[str | None, Field(serialization_alias="wire")]


class _AliasedDataclassArguments(BaseModel):
    payload: _AliasedDataclass


@dataclasses.dataclass
class _MetadataDefaultDataclass:
    required: str | None
    annotated_default: Annotated[str | None, Field(default=None)]
    slot_default: str | None = Field(default=None)


class _MetadataDefaultDataclassArguments(BaseModel):
    child: _MetadataDefaultDataclass


_PDC_ALIAS_GENERATOR_CALLS = 0


def _pdc_alias_generator(name: str) -> str:
    global _PDC_ALIAS_GENERATOR_CALLS
    _PDC_ALIAS_GENERATOR_CALLS += 1
    return name.upper()


@pydantic_dataclass(
    config=ConfigDict(alias_generator=_pdc_alias_generator, serialize_by_alias=True, validate_by_name=True)
)
class _GeneratedPydanticDataclass:
    value: str | None


class _AnyPydanticDataclassArguments(BaseModel):
    payload: Any


class _DeclaredPydanticDataclassArguments(BaseModel):
    payload: _GeneratedPydanticDataclass
