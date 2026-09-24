# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Static policy for restoring caller-explicit null tool arguments.

Pydantic's ``model_dump(exclude_none=True)`` removes explicit ``None`` values
at typed field/member positions. Caller-explicit nulls must reach tool
callables, while omitted default-null fields must stay absent.

This module classifies each input model once and assigns one of three tiers:

``FULL``
    Every reachable annotation is structurally plain. The whitelist contains
    BaseModel, pydantic/stdlib dataclasses, TypedDict, primitive/enum/literal
    leaves, built-in collections plus Sequence/Mapping/MutableMapping,
    unions, Annotated validation metadata, Any, and real NewType objects.
    Serializer metadata, serializer decorators, alias generators, TypedDict or
    dataclass aliases/default metadata, and unknown annotation nodes reject the
    graph. Invoke-time restoration then follows runtime containers only and
    descends into model/dataclass instances whose classes the policy approved.

``TOP_LEVEL``
    The graph was not fully plain, but the root model has no model serializer.
    Only statically keyed, serializer-free, non-excluded root fields can regain
    an explicit null. Nested output is left exactly as Pydantic produced it.

``OFF``
    Even the root output is not structurally owned by its fields, normally
    because a model serializer controls it. Restoration is disabled.

Classification may execute first-party build-time behavior while Pydantic
resolves annotations and compiled field metadata. Invoke time executes no
serializer or alias-generator code beyond the single ordinary Pydantic dump.
Schema-supplied tools bypass this module. Middleware and tool-author runtime
objects are trusted: the loop may use ``deepcopy`` and ordinary equality, and
a middleware rewrite that changes only dict insertion order may be classified
as untouched.

The accepted trade-off is reduced restoration coverage for rejected graphs:
nested explicit nulls stay absent, with one build-time debug record emitted by
the FunctionTool integration. The invariant retained here is structural: the
walk never rewrites or deletes Pydantic output. It inserts ``None`` only at a
slot whose class/key/container shape the static policy approved; a runtime
shape mismatch or unapproved class is an untouched leaf.

Declared slot closures also guard runtime model/dataclass descent. A validator
substitution to a class outside that slot is left untouched, and a slot that
co-declares a subclass pair degrades at classification. The remaining bounded
residual is substitution between unrelated co-declared classes; Pydantic's
declared-schema dump already emits mismatched output before this walk runs.
A compiled default value that itself mimics a typed core-schema node may
conservatively degrade classification; this loses restoration only.
Nested nulls beneath a defaulted dataclass field are not restored because
dataclasses expose no caller-provenance set for those materialized values.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import enum
import types
import typing
import uuid
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, Literal, get_args, get_origin

import annotated_types
import typing_extensions
from pydantic import AfterValidator, BaseModel, BeforeValidator, PlainSerializer, WrapSerializer, WrapValidator
from pydantic.fields import FieldInfo
from typing_extensions import is_typeddict as _te_is_typeddict


class OverlayTier(enum.Enum):
    """Strength of the statically approved restore."""

    FULL = "full"
    TOP_LEVEL = "top_level"
    OFF = "off"


@dataclass(frozen=True, slots=True)
class OverlayPolicy:
    """Immutable output of the one-time annotation-graph classification."""

    tier: OverlayTier
    root_model: type[BaseModel]
    approved_classes: frozenset[type]
    dump_keys: Mapping[type, Mapping[str, str]]
    never_restore: Mapping[type, frozenset[str]]
    dataclass_null_restorable: Mapping[type, frozenset[str]]
    dataclass_required_fields: Mapping[type, frozenset[str]]
    computed_keys: Mapping[type, frozenset[str]]
    slot_classes: Mapping[tuple[type, str], frozenset[type]]
    top_level_restorable: frozenset[str]
    reason: str | None = None


_PLAIN_LEAVES = frozenset(
    {
        str,
        int,
        float,
        bool,
        bytes,
        complex,
        type(None),
        datetime.datetime,
        datetime.date,
        datetime.time,
        datetime.timedelta,
        decimal.Decimal,
        uuid.UUID,
    }
)
_CONTAINER_ORIGINS = frozenset({list, tuple, set, frozenset, dict, Sequence, Mapping, MutableMapping})
_TRANSPARENT_WRAPPERS = frozenset(
    {
        typing.Required,
        typing.NotRequired,
        typing.ReadOnly,
        typing_extensions.Required,
        typing_extensions.NotRequired,
        typing_extensions.ReadOnly,
    }
)
_SAFE_ANNOTATED_METADATA = (
    FieldInfo,
    BeforeValidator,
    WrapValidator,
    AfterValidator,
    annotated_types.BaseMetadata,
    annotated_types.GroupedMetadata,
    str,
)


def _is_static_member(value: Any, members: frozenset[Any]) -> bool:
    """Test a static whitelist without requiring the annotation to be hashable."""
    try:
        return value in members
    except TypeError:
        return False


def _is_typed_dict(cls: Any) -> bool:
    return isinstance(cls, type) and (typing.is_typeddict(cls) or _te_is_typeddict(cls))


def _typed_dict_config(td_cls: type, seen: set[int] | None = None) -> Mapping[str, Any] | None:
    """Resolve TypedDict config through its flattened declaration bases."""
    seen = set() if seen is None else seen
    if id(td_cls) in seen:
        return None
    seen.add(id(td_cls))
    config = td_cls.__dict__.get("__pydantic_config__")
    if isinstance(config, Mapping):
        return config
    for declared_base in td_cls.__dict__.get("__orig_bases__", ()):
        base = get_origin(declared_base) or declared_base
        if _is_typed_dict(base):
            inherited = _typed_dict_config(base, seen)
            if inherited is not None:
                return inherited
    return None


def _real_new_type(annotation: Any) -> bool:
    """Whether *annotation* is a real stdlib/backport NewType object."""
    return isinstance(annotation, typing.NewType)


def _has_custom_core_schema(cls: type, framework_base: type | None = None) -> bool:
    """Whether a class below its framework base controls core-schema output."""
    for base in cls.__mro__:
        if base is framework_base:
            return False
        if "__get_pydantic_core_schema__" in base.__dict__:
            return True
    return False


def _metadata_contains_serializer(metadata: list[Any], seen: set[int] | None = None) -> bool:
    """Whether metadata contains serializer ownership, including FieldInfo carriers."""
    seen = set() if seen is None else seen
    for item in metadata:
        if isinstance(item, (PlainSerializer, WrapSerializer)):
            return True
        if isinstance(item, FieldInfo) and id(item) not in seen:
            seen.add(id(item))
            if _metadata_contains_serializer(item.metadata, seen):
                return True
    return False


def _annotation_field_info(annotation: Any) -> FieldInfo | None:
    """Merged FieldInfo through Annotated and TypedDict wrapper layers."""
    layers: list[list[FieldInfo]] = []
    seen: set[int] = set()
    while id(annotation) not in seen:
        seen.add(id(annotation))
        if _real_new_type(annotation):
            annotation = annotation.__supertype__
            continue
        origin = get_origin(annotation)
        if origin is Annotated:
            annotation, *metadata = get_args(annotation)
            layers.append([item for item in metadata if isinstance(item, FieldInfo)])
            continue
        if _is_static_member(origin, _TRANSPARENT_WRAPPERS):
            annotation = get_args(annotation)[0]
            continue
        break
    infos = [item for layer in reversed(layers) for item in layer]
    # The overlay mirrors pydantic-internal field merging; revisit when pydantic removes it.
    return FieldInfo.merge_field_infos(*infos) if infos else None  # ty: ignore[deprecated]


def _field_dump_key(model_cls: type[BaseModel], field_name: str, field_info: FieldInfo) -> str:
    """Field key emitted by the module's ordinary model dump call."""
    if not bool(model_cls.model_config.get("serialize_by_alias", False)):
        return field_name
    if field_info.serialization_alias is not None:
        return field_info.serialization_alias
    if field_info.alias is not None:
        return field_info.alias
    return field_name


def _dataclass_mro_attribute(dataclass_cls: type, name: str) -> tuple[Any, int] | None:
    for index, base in enumerate(dataclass_cls.__mro__):
        if name in base.__dict__:
            return (base.__dict__[name], index)
    return None


def _dataclass_decorator_state(dataclass_cls: type) -> tuple[bool, dict[str, Any]]:
    """Serializer presence and computed-field metadata across the MRO."""
    computed_fields: dict[str, Any] = {}
    for base in dataclass_cls.__mro__:
        decorators = base.__dict__.get("__pydantic_decorators__")
        if decorators is not None and (decorators.model_serializers or decorators.field_serializers):
            return (True, {})
        if decorators is not None:
            for name, decorator in decorators.computed_fields.items():
                computed_fields.setdefault(name, decorator.info)
        for name, attribute in base.__dict__.items():
            info = getattr(attribute, "decorator_info", None)
            if info is None:
                continue
            info_name = type(info).__name__
            if info_name in {"ModelSerializerDecoratorInfo", "FieldSerializerDecoratorInfo"}:
                return (True, {})
            if info_name == "ComputedFieldInfo":
                computed_fields.setdefault(name, info)
    return (False, computed_fields)


def _dataclass_field_info(
    dataclass_cls: type, dataclass_field: dataclasses.Field[Any], annotation: Any
) -> FieldInfo | None:
    """Effective Pydantic field metadata for a dataclass field."""
    compiled = _dataclass_mro_attribute(dataclass_cls, "__pydantic_fields__")
    if compiled is not None:
        pydantic_fields, owner_index = compiled
        owner_fields = dataclass_cls.__mro__[owner_index].__dict__.get("__dataclass_fields__")
        if (
            isinstance(pydantic_fields, dict)
            and isinstance(owner_fields, dict)
            and owner_fields.get(dataclass_field.name) is dataclass_field
        ):
            field_info = pydantic_fields.get(dataclass_field.name)
            if isinstance(field_info, FieldInfo):
                return field_info
    default = dataclass_field.default
    attribute = default if isinstance(default, FieldInfo) else dataclass_field
    return FieldInfo.from_annotated_attribute(annotation, attribute)


class _PolicyBuilder:
    def __init__(self) -> None:
        self.approved_classes: set[type] = set()
        self.dump_keys: dict[type, dict[str, str]] = {}
        self.never_restore: dict[type, frozenset[str]] = {}
        self.dataclass_null_restorable: dict[type, frozenset[str]] = {}
        self.dataclass_required_fields: dict[type, frozenset[str]] = {}
        self.computed_keys: dict[type, frozenset[str]] = {}
        self.slot_classes: dict[tuple[type, str], frozenset[type]] = {}
        self._visiting: set[type] = set()
        self.reason: str | None = None

    def reject(self, reason: str) -> bool:
        if self.reason is None:
            self.reason = reason
        return False

    def metadata_is_plain(self, metadata: list[Any]) -> bool:
        """Validate one metadata sequence, including nested FieldInfo carriers."""
        if _metadata_contains_serializer(metadata):
            return self.reject("Annotated serializer metadata")
        for item in metadata:
            if isinstance(item, FieldInfo):
                if not self.metadata_is_plain(item.metadata):
                    return False
                continue
            if not isinstance(item, _SAFE_ANNOTATED_METADATA):
                return self.reject(f"unsupported Annotated metadata {type(item).__name__}")
        return True

    def slot_model_classes(self, annotation: Any, seen: set[int] | None = None) -> frozenset[type]:
        """Approved model/dataclass classes reachable before a nested class boundary."""
        seen = set() if seen is None else seen
        if id(annotation) in seen:
            return frozenset()
        seen.add(id(annotation))
        if _real_new_type(annotation):
            return self.slot_model_classes(annotation.__supertype__, seen)
        origin = get_origin(annotation)
        if origin is Annotated or _is_static_member(origin, _TRANSPARENT_WRAPPERS):
            return self.slot_model_classes(get_args(annotation)[0], seen)
        if origin in {dict, Mapping, MutableMapping}:
            arguments = get_args(annotation)
            return self.slot_model_classes(arguments[1], seen) if len(arguments) == 2 else frozenset()
        if origin in (typing.Union, types.UnionType) or _is_static_member(origin, _CONTAINER_ORIGINS):
            classes: set[type] = set()
            for item in get_args(annotation):
                if item is not Ellipsis:
                    classes.update(self.slot_model_classes(item, seen.copy()))
            return frozenset(classes)
        if _is_typed_dict(annotation):
            try:
                hints = typing.get_type_hints(annotation, include_extras=True)
            except Exception:
                return frozenset()
            classes = set()
            for member_annotation in hints.values():
                classes.update(self.slot_model_classes(member_annotation, seen.copy()))
            extra_items = annotation.__dict__.get("__extra_items__", typing_extensions.NoExtraItems)
            no_extra_items = typing.__dict__.get("NoExtraItems", typing_extensions.NoExtraItems)
            if extra_items is not typing_extensions.NoExtraItems and extra_items is not no_extra_items:
                classes.update(self.slot_model_classes(extra_items, seen.copy()))
            return frozenset(classes)
        if isinstance(annotation, type) and (issubclass(annotation, BaseModel) or dataclasses.is_dataclass(annotation)):
            return frozenset({annotation})
        return frozenset()

    def record_slot_classes(self, owner: type, name: str, annotation: Any) -> bool:
        """Record one slot unless its declared closure contains a subclass pair."""
        classes = self.slot_model_classes(annotation)
        ordered = sorted(classes, key=lambda cls: (cls.__module__, cls.__qualname__))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if issubclass(left, right) or issubclass(right, left):
                    return self.reject(f"co-declared subclass pair {left.__name__}/{right.__name__} in one slot")
        self.slot_classes[(owner, name)] = classes
        return True

    def annotation_is_plain(self, annotation: Any) -> bool:
        if annotation is Any or _is_static_member(annotation, _PLAIN_LEAVES):
            return True
        if _real_new_type(annotation):
            return self.annotation_is_plain(annotation.__supertype__)
        origin = get_origin(annotation)
        if origin is Annotated:
            base, *metadata = get_args(annotation)
            return self.metadata_is_plain(metadata) and self.annotation_is_plain(base)
        if _is_static_member(origin, _TRANSPARENT_WRAPPERS):
            return self.annotation_is_plain(get_args(annotation)[0])
        if origin in (typing.Union, types.UnionType):
            return all(self.annotation_is_plain(item) for item in get_args(annotation))
        if origin is Literal:
            return True
        if _is_static_member(origin, _CONTAINER_ORIGINS):
            return all(item is Ellipsis or self.annotation_is_plain(item) for item in get_args(annotation))
        if _is_static_member(annotation, _CONTAINER_ORIGINS):
            return True
        if _is_typed_dict(annotation):
            return self.typed_dict_is_plain(annotation)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return self.model_is_plain(annotation)
        if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
            return self.dataclass_is_plain(annotation)
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            if _has_custom_core_schema(annotation, enum.Enum):
                return self.reject(f"custom core schema on {annotation.__name__}")
            return True
        return self.reject(f"unsupported annotation {annotation!r}")

    def model_is_plain(self, model_cls: type[BaseModel]) -> bool:
        if model_cls in self.approved_classes or model_cls in self._visiting:
            return True
        self._visiting.add(model_cls)
        if _has_custom_core_schema(model_cls, BaseModel):
            self._visiting.discard(model_cls)
            return self.reject(f"custom core schema on {model_cls.__name__}")
        decorators = model_cls.__pydantic_decorators__
        if decorators.model_serializers or decorators.field_serializers:
            self._visiting.discard(model_cls)
            return self.reject(f"serializer decorator on {model_cls.__name__}")
        if model_cls.model_config.get("alias_generator") is not None:
            self._visiting.discard(model_cls)
            return self.reject(f"alias generator on {model_cls.__name__}")
        try:
            hints = typing.get_type_hints(model_cls, include_extras=True)
        except Exception:
            self._visiting.discard(model_cls)
            return self.reject(f"unresolved annotations on {model_cls.__name__}")

        keys: dict[str, str] = {}
        never: set[str] = set()
        claims: dict[str, int] = {}
        for field_name, field_info in model_cls.model_fields.items():
            if not self.metadata_is_plain(field_info.metadata):
                self._visiting.discard(model_cls)
                return False
            key = _field_dump_key(model_cls, field_name, field_info)
            keys[field_name] = key
            if field_info.exclude is True:
                never.add(field_name)
                continue
            claims[key] = claims.get(key, 0) + 1
            annotation = (
                field_info.annotation
                if model_cls.__pydantic_root_model__ and field_name == "root"
                else hints.get(field_name, field_info.annotation)
            )
            if not self.annotation_is_plain(annotation):
                self._visiting.discard(model_cls)
                return False
            if not self.record_slot_classes(model_cls, field_name, annotation):
                self._visiting.discard(model_cls)
                return False
            if field_info.exclude_if is not None:
                never.add(field_name)

        by_alias = bool(model_cls.model_config.get("serialize_by_alias", False))
        computed: set[str] = set()
        for field_name, field_info in model_cls.model_computed_fields.items():
            key = field_info.alias if by_alias and field_info.alias is not None else field_name
            computed.add(key)
            claims[key] = claims.get(key, 0) + 1
        if any(count > 1 for count in claims.values()):
            self._visiting.discard(model_cls)
            return self.reject(f"colliding dump keys on {model_cls.__name__}")

        self._visiting.discard(model_cls)
        self.approved_classes.add(model_cls)
        self.dump_keys[model_cls] = keys
        self.never_restore[model_cls] = frozenset(never)
        self.computed_keys[model_cls] = frozenset(computed)
        return True

    def typed_dict_is_plain(self, td_cls: type) -> bool:
        if td_cls in self.approved_classes or td_cls in self._visiting:
            return True
        self._visiting.add(td_cls)
        config = _typed_dict_config(td_cls)
        if isinstance(config, Mapping) and config.get("alias_generator") is not None:
            self._visiting.discard(td_cls)
            return self.reject(f"alias generator on {td_cls.__name__}")
        try:
            hints = typing.get_type_hints(td_cls, include_extras=True)
        except Exception:
            self._visiting.discard(td_cls)
            return self.reject(f"unresolved annotations on {td_cls.__name__}")
        keys: dict[str, str] = {}
        for member, annotation in hints.items():
            field_info = _annotation_field_info(annotation)
            if field_info is not None:
                if not self.metadata_is_plain(field_info.metadata):
                    self._visiting.discard(td_cls)
                    return False
                if (
                    field_info.alias is not None
                    or field_info.serialization_alias is not None
                    or field_info.validation_alias is not None
                ):
                    self._visiting.discard(td_cls)
                    return self.reject(f"alias metadata on {td_cls.__name__}.{member}")
                if field_info.exclude is True or field_info.exclude_if is not None:
                    self._visiting.discard(td_cls)
                    return self.reject(f"exclusion metadata on {td_cls.__name__}.{member}")
                if not field_info.is_required():
                    self._visiting.discard(td_cls)
                    return self.reject(f"default metadata on {td_cls.__name__}.{member}")
            if not self.annotation_is_plain(annotation):
                self._visiting.discard(td_cls)
                return False
            keys[member] = member
            if not self.record_slot_classes(td_cls, member, annotation):
                self._visiting.discard(td_cls)
                return False
        extra_items = td_cls.__dict__.get("__extra_items__", typing_extensions.NoExtraItems)
        no_extra_items = typing.__dict__.get("NoExtraItems", typing_extensions.NoExtraItems)
        if (
            extra_items is not typing_extensions.NoExtraItems
            and extra_items is not no_extra_items
            and not self.annotation_is_plain(extra_items)
        ):
            self._visiting.discard(td_cls)
            return False
        self._visiting.discard(td_cls)
        self.approved_classes.add(td_cls)
        self.dump_keys[td_cls] = keys
        self.never_restore[td_cls] = frozenset()
        self.computed_keys[td_cls] = frozenset()
        return True

    def dataclass_is_plain(self, dataclass_cls: type) -> bool:
        if dataclass_cls in self.approved_classes or dataclass_cls in self._visiting:
            return True
        self._visiting.add(dataclass_cls)
        if _has_custom_core_schema(dataclass_cls):
            self._visiting.discard(dataclass_cls)
            return self.reject(f"custom core schema on {dataclass_cls.__name__}")
        has_serializers, computed_fields = _dataclass_decorator_state(dataclass_cls)
        if has_serializers:
            self._visiting.discard(dataclass_cls)
            return self.reject(f"serializer decorator on {dataclass_cls.__name__}")
        if any(info.alias is not None for info in computed_fields.values()):
            self._visiting.discard(dataclass_cls)
            return self.reject(f"computed-field alias on {dataclass_cls.__name__}")
        compiled_config = _dataclass_mro_attribute(dataclass_cls, "__pydantic_config__")
        if compiled_config is not None and compiled_config[0].get("alias_generator") is not None:
            self._visiting.discard(dataclass_cls)
            return self.reject(f"alias generator on {dataclass_cls.__name__}")
        try:
            hints = typing.get_type_hints(dataclass_cls, include_extras=True)
        except Exception:
            self._visiting.discard(dataclass_cls)
            return self.reject(f"unresolved annotations on {dataclass_cls.__name__}")

        keys: dict[str, str] = {}
        never: set[str] = set()
        null_restorable: set[str] = set()
        required_fields: set[str] = set()
        # Callers admit only classes already proven by dataclasses.is_dataclass().
        for dataclass_field in dataclasses.fields(dataclass_cls):  # ty: ignore[invalid-argument-type]
            name = dataclass_field.name
            annotation = hints.get(name, dataclass_field.type)
            field_info = _dataclass_field_info(dataclass_cls, dataclass_field, annotation)
            if not self.metadata_is_plain(field_info.metadata):
                self._visiting.discard(dataclass_cls)
                return False
            if (
                field_info.alias is not None
                or field_info.serialization_alias is not None
                or field_info.validation_alias is not None
            ):
                self._visiting.discard(dataclass_cls)
                return self.reject(f"alias metadata on {dataclass_cls.__name__}.{name}")
            keys[name] = name
            if not dataclass_field.init or field_info.exclude is True or field_info.exclude_if is not None:
                never.add(name)
                continue
            if name in computed_fields:
                self._visiting.discard(dataclass_cls)
                return self.reject(f"computed-field collision on {dataclass_cls.__name__}.{name}")
            if not self.annotation_is_plain(annotation):
                self._visiting.discard(dataclass_cls)
                return False
            if not self.record_slot_classes(dataclass_cls, name, annotation):
                self._visiting.discard(dataclass_cls)
                return False
            if field_info.is_required():
                required_fields.add(name)
                null_restorable.add(name)
            elif field_info.default_factory is None and field_info.default is not None:
                null_restorable.add(name)

        self._visiting.discard(dataclass_cls)
        self.approved_classes.add(dataclass_cls)
        self.dump_keys[dataclass_cls] = keys
        self.never_restore[dataclass_cls] = frozenset(never)
        self.dataclass_null_restorable[dataclass_cls] = frozenset(null_restorable)
        self.dataclass_required_fields[dataclass_cls] = frozenset(required_fields)
        self.computed_keys[dataclass_cls] = frozenset()
        return True


def _root_output_reason(model_cls: type[BaseModel]) -> str | None:
    if model_cls.__pydantic_decorators__.model_serializers:
        return f"model serializer on {model_cls.__name__}"
    if _has_custom_core_schema(model_cls, BaseModel):
        return f"custom core schema on {model_cls.__name__}"
    if model_cls.model_dump is not BaseModel.model_dump:
        return f"model_dump override on {model_cls.__name__}"
    return None


def _node_has_untrusted_serialization(node: Mapping[str, Any]) -> bool:
    """Whether a compiled node installs non-framework serialization."""
    if not isinstance(node.get("type"), str):
        return False
    serialization = node.get("serialization")
    if not isinstance(serialization, Mapping):
        return False
    if "function" not in str(serialization.get("type", "")):
        return True
    function = serialization.get("function")
    module = getattr(function, "__module__", None)
    return not isinstance(module, str) or not module.startswith("pydantic.")


def _compiled_definition_index(schema: Any, seen: set[int] | None = None) -> dict[str, Mapping[str, Any]]:
    """Index named schemas from every compiled definitions table."""
    if not isinstance(schema, (dict, list, tuple)):
        return {}
    seen = set() if seen is None else seen
    if id(schema) in seen:
        return {}
    seen.add(id(schema))
    indexed: dict[str, Mapping[str, Any]] = {}
    if isinstance(schema, dict):
        definitions = schema.get("definitions")
        if isinstance(definitions, (list, tuple)):
            for definition in definitions:
                if isinstance(definition, Mapping) and isinstance(definition.get("ref"), str):
                    indexed.setdefault(definition["ref"], definition)
        for value in schema.values():
            indexed.update(_compiled_definition_index(value, seen))
    else:
        for value in schema:
            indexed.update(_compiled_definition_index(value, seen))
    return indexed


def _schema_has_untrusted_serialization(
    schema: Any,
    seen: set[int] | None = None,
    definitions: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """Scan a compiled core-schema graph for non-framework serialization."""
    if not isinstance(schema, (dict, list, tuple)):
        return False
    seen = set() if seen is None else seen
    if id(schema) in seen:
        return False
    seen.add(id(schema))
    if isinstance(schema, dict):
        if _node_has_untrusted_serialization(schema):
            return True
        if schema.get("type") == "definition-ref" and definitions is not None:
            schema_ref = schema.get("schema_ref")
            if not isinstance(schema_ref, str):
                return True
            target = definitions.get(schema_ref)
            if target is None:
                return True
            return _schema_has_untrusted_serialization(target, seen, definitions)
        return any(_schema_has_untrusted_serialization(value, seen, definitions) for value in schema.values())
    return any(_schema_has_untrusted_serialization(value, seen, definitions) for value in schema)


_ROOT_SCHEMA_WRAPPERS = frozenset(
    {
        "definitions",
        "function-before",
        "function-after",
        "function-wrap",
        "function-plain",
        "default",
        "nullable",
    }
)


def _root_schema_has_untrusted_serialization(schema: Any, seen: set[int] | None = None) -> bool:
    """Check non-framework serialization only along the root wrapper spine."""
    if not isinstance(schema, dict):
        return False
    seen = set() if seen is None else seen
    if id(schema) in seen:
        return False
    seen.add(id(schema))
    if _node_has_untrusted_serialization(schema):
        return True
    schema_type = schema.get("type")
    if (schema_type == "model" and not bool(schema.get("root_model", False))) or schema_type in _ROOT_SCHEMA_WRAPPERS:
        return _root_schema_has_untrusted_serialization(schema.get("schema"), seen)
    if schema_type == "lax-or-strict":
        return _root_schema_has_untrusted_serialization(
            schema.get("lax_schema"), seen
        ) or _root_schema_has_untrusted_serialization(schema.get("strict_schema"), seen)
    if schema_type == "json-or-python":
        return _root_schema_has_untrusted_serialization(
            schema.get("json_schema"), seen
        ) or _root_schema_has_untrusted_serialization(schema.get("python_schema"), seen)
    return False


def _root_model_fields(schema: Any, seen: set[int] | None = None) -> Mapping[str, Any] | None:
    """Locate the root model-fields mapping along the compiled wrapper spine."""
    if not isinstance(schema, dict):
        return None
    seen = set() if seen is None else seen
    if id(schema) in seen:
        return None
    seen.add(id(schema))
    schema_type = schema.get("type")
    if schema_type == "model-fields":
        fields = schema.get("fields")
        return fields if isinstance(fields, Mapping) else None
    if schema_type == "model" or schema_type in _ROOT_SCHEMA_WRAPPERS:
        return _root_model_fields(schema.get("schema"), seen)
    if schema_type == "lax-or-strict":
        lax_fields = _root_model_fields(schema.get("lax_schema"), seen.copy())
        strict_fields = _root_model_fields(schema.get("strict_schema"), seen.copy())
        if lax_fields is None:
            return strict_fields
        if strict_fields is None or lax_fields is strict_fields:
            return lax_fields
        return None
    if schema_type == "json-or-python":
        json_fields = _root_model_fields(schema.get("json_schema"), seen.copy())
        python_fields = _root_model_fields(schema.get("python_schema"), seen.copy())
        if json_fields is None:
            return python_fields
        if python_fields is None or json_fields is python_fields:
            return json_fields
        return None
    return None


def _off_policy(model: type[BaseModel], reason: str) -> OverlayPolicy:
    return OverlayPolicy(
        tier=OverlayTier.OFF,
        root_model=model,
        approved_classes=frozenset(),
        dump_keys=MappingProxyType({}),
        never_restore=MappingProxyType({}),
        dataclass_null_restorable=MappingProxyType({}),
        dataclass_required_fields=MappingProxyType({}),
        computed_keys=MappingProxyType({}),
        slot_classes=MappingProxyType({}),
        top_level_restorable=frozenset(),
        reason=reason,
    )


def _root_top_level_fields(
    model_cls: type[BaseModel],
    core_schema: Any,
) -> tuple[frozenset[str], dict[str, str]]:
    """Serializer-free root fields whose own null slot is statically keyed."""
    decorators = model_cls.__pydantic_decorators__
    serialized_fields: set[str] = set()
    all_fields_serialized = False
    for decorator in decorators.field_serializers.values():
        if "*" in decorator.info.fields:
            all_fields_serialized = True
        serialized_fields.update(decorator.info.fields)
    try:
        hints = typing.get_type_hints(model_cls, include_extras=True)
    except Exception:
        hints = {}

    keys = {name: _field_dump_key(model_cls, name, info) for name, info in model_cls.model_fields.items()}
    computed_keys = {
        info.alias if bool(model_cls.model_config.get("serialize_by_alias", False)) and info.alias is not None else name
        for name, info in model_cls.model_computed_fields.items()
    }
    counts: dict[str, int] = {}
    for name, key in keys.items():
        if model_cls.model_fields[name].exclude is not True:
            counts[key] = counts.get(key, 0) + 1
    for key in computed_keys:
        counts[key] = counts.get(key, 0) + 1

    compiled_fields = _root_model_fields(core_schema)
    if core_schema is not None and compiled_fields is None:
        return (frozenset(), keys)
    definitions = _compiled_definition_index(core_schema)
    restorable: set[str] = set()
    for name, info in model_cls.model_fields.items():
        if (
            all_fields_serialized
            or name in serialized_fields
            or info.exclude is True
            or info.exclude_if is not None
            or counts.get(keys[name], 0) != 1
        ):
            continue
        annotation = hints.get(name, info.annotation)
        if _annotation_contains_serializer(annotation):
            continue
        if compiled_fields is not None:
            compiled_field = compiled_fields.get(name)
            if not isinstance(compiled_field, Mapping) or _schema_has_untrusted_serialization(
                compiled_field,
                definitions=definitions,
            ):
                continue
        restorable.add(name)
    return (frozenset(restorable), keys)


def _annotation_contains_serializer(annotation: Any, seen: set[int] | None = None) -> bool:
    seen = set() if seen is None else seen
    if id(annotation) in seen:
        return False
    seen.add(id(annotation))
    if _real_new_type(annotation):
        return _annotation_contains_serializer(annotation.__supertype__, seen)
    if isinstance(annotation, (typing.TypeAliasType, typing_extensions.TypeAliasType)):
        return True
    if get_origin(annotation) is Annotated:
        base, *metadata = get_args(annotation)
        return _metadata_contains_serializer(metadata) or _annotation_contains_serializer(base, seen)
    return any(_annotation_contains_serializer(item, seen) for item in get_args(annotation))


def classify(model: type[BaseModel]) -> OverlayPolicy:
    """Classify one tool input model against the finite structural whitelist."""
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TypeError("model must be a BaseModel subclass")
    root_reason = _root_output_reason(model)
    if root_reason is not None:
        return _off_policy(model, root_reason)

    core_schema = getattr(model, "__pydantic_core_schema__", None)
    if _root_schema_has_untrusted_serialization(core_schema):
        return _off_policy(model, f"root serialization in compiled schema for {model.__name__}")
    compiled_reason = (
        "compiled core schema unavailable"
        if core_schema is None
        else "serialization in compiled schema"
        if _schema_has_untrusted_serialization(core_schema)
        else None
    )

    builder = _PolicyBuilder()
    graph_is_plain = builder.model_is_plain(model)
    if graph_is_plain and compiled_reason is None:
        tier = OverlayTier.FULL
        restorable = frozenset()
    else:
        tier = OverlayTier.TOP_LEVEL
        restorable, root_keys = _root_top_level_fields(model, core_schema)
        builder.dump_keys[model] = root_keys
        builder.never_restore[model] = frozenset(set(model.model_fields) - set(restorable))

    return OverlayPolicy(
        tier=tier,
        root_model=model,
        approved_classes=frozenset(builder.approved_classes),
        dump_keys=MappingProxyType({cls: MappingProxyType(dict(keys)) for cls, keys in builder.dump_keys.items()}),
        never_restore=MappingProxyType(dict(builder.never_restore)),
        dataclass_null_restorable=MappingProxyType(dict(builder.dataclass_null_restorable)),
        dataclass_required_fields=MappingProxyType(dict(builder.dataclass_required_fields)),
        computed_keys=MappingProxyType(dict(builder.computed_keys)),
        slot_classes=MappingProxyType(dict(builder.slot_classes)),
        top_level_restorable=restorable,
        reason=builder.reason or compiled_reason,
    )


def _classification_fallback(model: type[BaseModel], reason: str) -> OverlayPolicy:
    """Build the weakest usable policy after an unexpected classifier failure."""
    try:
        root_reason = _root_output_reason(model)
        if root_reason is not None:
            return _off_policy(model, root_reason)
        core_schema = getattr(model, "__pydantic_core_schema__", None)
        if _root_schema_has_untrusted_serialization(core_schema):
            return _off_policy(model, reason)
        restorable, keys = _root_top_level_fields(model, core_schema)
    except Exception:
        return _off_policy(model, reason)
    return OverlayPolicy(
        tier=OverlayTier.TOP_LEVEL,
        root_model=model,
        approved_classes=frozenset(),
        dump_keys=MappingProxyType({model: MappingProxyType(keys)}),
        never_restore=MappingProxyType({model: frozenset(set(model.model_fields) - set(restorable))}),
        dataclass_null_restorable=MappingProxyType({}),
        dataclass_required_fields=MappingProxyType({}),
        computed_keys=MappingProxyType({}),
        slot_classes=MappingProxyType({}),
        top_level_restorable=restorable,
        reason=reason,
    )


def _model_dynamic_collisions(value: BaseModel, policy: OverlayPolicy) -> set[str]:
    extras = value.model_extra or {}
    if not extras:
        return set()
    cls = type(value)
    claimed = {
        key for name, key in policy.dump_keys.get(cls, {}).items() if cls.model_fields[name].exclude is not True
    } | set(policy.computed_keys.get(cls, frozenset()))
    return claimed.intersection(extras)


def _restore_full(
    value: Any,
    dumped: Any,
    policy: OverlayPolicy,
    allowed_classes: frozenset[type],
) -> Any:
    if isinstance(value, BaseModel):
        cls = type(value)
        if cls not in policy.approved_classes or cls not in allowed_classes:
            return dumped
        if cls.__pydantic_root_model__:
            root = value.__dict__.get("root")
            root_classes = policy.slot_classes.get((cls, "root"), frozenset())
            return _restore_full(root, dumped, policy, root_classes)
        if not isinstance(dumped, dict):
            return dumped
        keys = policy.dump_keys[cls]
        never = policy.never_restore.get(cls, frozenset())
        extras = value.model_extra or {}
        collisions = _model_dynamic_collisions(value, policy)
        declared_fields_set = value.model_fields_set.intersection(cls.model_fields)
        for name in declared_fields_set:
            if name in never or name not in keys:
                continue
            field_value = value.__dict__.get(name)
            key = keys[name]
            if key in collisions:
                continue
            if field_value is None:
                if key not in dumped:
                    dumped[key] = None
            elif key in dumped:
                slot_classes = policy.slot_classes.get((cls, name), frozenset())
                dumped[key] = _restore_full(field_value, dumped[key], policy, slot_classes)
        for key, extra_value in extras.items():
            if key in collisions:
                continue
            if extra_value is None:
                if key not in dumped:
                    dumped[key] = None
            elif key in dumped:
                dumped[key] = _restore_full(extra_value, dumped[key], policy, frozenset())
        return dumped

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        if cls not in policy.approved_classes or cls not in allowed_classes or not isinstance(dumped, dict):
            return dumped
        keys = policy.dump_keys[cls]
        never = policy.never_restore.get(cls, frozenset())
        null_restorable = policy.dataclass_null_restorable.get(cls, frozenset())
        required_fields = policy.dataclass_required_fields.get(cls, frozenset())
        for field in dataclasses.fields(cls):
            name = field.name
            if name in never or name not in keys:
                continue
            try:
                field_value = object.__getattribute__(value, name)
            except AttributeError:
                continue
            key = keys[name]
            if field_value is None:
                if name in null_restorable and key not in dumped:
                    dumped[key] = None
            elif name in required_fields and key in dumped:
                slot_classes = policy.slot_classes.get((cls, name), frozenset())
                dumped[key] = _restore_full(field_value, dumped[key], policy, slot_classes)
        return dumped

    if isinstance(value, dict):
        if not isinstance(dumped, dict):
            return dumped
        for key, item in value.items():
            if item is None:
                if key not in dumped:
                    dumped[key] = None
            elif key in dumped:
                dumped[key] = _restore_full(item, dumped[key], policy, allowed_classes)
        return dumped

    if isinstance(value, (list, tuple)):
        expected_type = list if isinstance(value, list) else tuple
        if type(dumped) is not expected_type:
            return dumped
        none_count = sum(item is None for item in value)
        if len(dumped) == len(value):
            restored = list(dumped)
            for index, item in enumerate(value):
                if item is not None:
                    restored[index] = _restore_full(item, restored[index], policy, allowed_classes)
        elif len(dumped) == len(value) - none_count:
            restored = []
            dumped_items = iter(dumped)
            for item in value:
                if item is None:
                    restored.append(None)
                else:
                    restored.append(_restore_full(item, next(dumped_items), policy, allowed_classes))
        else:
            return dumped
        return restored if expected_type is list else tuple(restored)

    if isinstance(value, set) and isinstance(dumped, set):
        if None in value:
            dumped.add(None)
        return dumped
    if isinstance(value, frozenset) and isinstance(dumped, frozenset):
        return dumped | {None} if None in value else dumped
    return dumped


def restore_explicit_nulls(validated: BaseModel, dumped: dict[str, Any], policy: OverlayPolicy) -> dict[str, Any]:
    """Insert statically approved explicit nulls without rewriting dump data."""
    if policy.tier is OverlayTier.OFF or type(validated) is not policy.root_model:
        return dumped
    if policy.tier is OverlayTier.TOP_LEVEL:
        keys = policy.dump_keys.get(policy.root_model, {})
        for name in validated.model_fields_set.intersection(policy.top_level_restorable):
            value = validated.__dict__.get(name)
            key = keys.get(name)
            if value is None and key is not None and key not in dumped:
                dumped[key] = None
        return dumped
    restored = _restore_full(validated, dumped, policy, frozenset({policy.root_model}))
    return restored if isinstance(restored, dict) else dumped


__all__ = ["OverlayPolicy", "OverlayTier", "classify", "restore_explicit_nulls"]
