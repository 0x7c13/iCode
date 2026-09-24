# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Null-overlay policy: ``classify``/``OverlayTier`` tiers and the nulls ``FunctionTool.invoke`` restores."""

from __future__ import annotations

import logging
from typing import Any

import annotated_types
import pytest
from pydantic import BaseModel, StringConstraints

from chrys.kernel import FunctionTool
from chrys.kernel._null_overlay import OverlayTier, classify
from tests.kernel import _null_overlay_models as models
from tests.kernel._fakes import _AliasedNullable, _DivergentDefaults, _NullableValue
from tests.kernel._null_overlay_models import (
    _AliasedDataclassArguments,
    _AliasedFieldWithInternalNameExtraArguments,
    _AliasedTypedDictArguments,
    _AnyPydanticDataclassArguments,
    _AnyUrlWithSafeRootFieldArguments,
    _CoDeclaredTupleArguments,
    _CoDeclaredUnionArguments,
    _ComputedAliasDataclassArguments,
    _ComputedArguments,
    _CoreSchemaArguments,
    _CoreSchemaDataclassArguments,
    _CoreSchemaKeyArguments,
    _CustomSlotSerializerArguments,
    _DataclassArguments,
    _DeclaredPydanticDataclassArguments,
    _DefaultedNestedDataclassArguments,
    _DefaultedTypedDictArguments,
    _EmittingFieldWithCollidingExtraArguments,
    _EmptyAliasArguments,
    _EnumGateArguments,
    _ExcludedArguments,
    _ExcludedFieldWithAliasedExtraArguments,
    _ExcludeIfCollisionArguments,
    _ExcludeIfPlainArguments,
    _ExtraArguments,
    _FieldConstraintArguments,
    _FieldInfoCarrierArguments,
    _FieldSerializerArguments,
    _FullGateArguments,
    _GeneratedPydanticDataclass,
    _GroupedSerializerArguments,
    _InheritedAliasArguments,
    _KeySerializerArguments,
    _MetaclassSerializerArguments,
    _MetadataDefaultDataclassArguments,
    _ModelFieldsSerializerArguments,
    _ModelSerializationArguments,
    _NestedArguments,
    _NestedDumpOverrideArguments,
    _NestedPlainSerializerArguments,
    _NestedSerializerArguments,
    _NewTypeSerializerArguments,
    _NonNoneDefaultDataclassArguments,
    _PlainInheritedArguments,
    _PrecedenceAliasArguments,
    _PydanticDataclassArguments,
    _PydanticInternalSerializationArguments,
    _RecursivePlainDegradedArguments,
    _RecursiveSerializedArguments,
    _RequiredNestedDataclassArguments,
    _RootDumpOverrideArguments,
    _RootModelArguments,
    _RootSerializerArguments,
    _SafeMetadataWithSerializerArguments,
    _SerializationDefaultsRequiredArguments,
    _SerializationNamedFieldArguments,
    _SerializedRootArguments,
    _SlotClassArguments,
    _SplitAliasArguments,
    _SupertypeAttributeArguments,
    _TupleArguments,
    _TypedDictArguments,
    _TypedDictSerializerArguments,
    _UnhashableAnnotationArguments,
    _UnknownContainerArguments,
    _UnschemableComputedArguments,
    _ValidationMetadataArguments,
)


def _capturing_tool(
    input_model: Any,
    *,
    name: str,
    received: list[dict[str, Any]] | None = None,
) -> tuple[FunctionTool, list[dict[str, Any]]]:
    """Build a tool whose callable records every validated kwargs mapping it receives.

    Pass ``received`` to have several tools append, in invocation order, to one list.
    """
    if received is None:
        received = []

    async def capture(**kwargs: Any) -> str:
        received.append(kwargs)
        return "ok"

    capturing = FunctionTool(
        name=name,
        description=f"Capture the arguments passed to {name}.",
        func=capture,
        input_model=input_model,
    )
    return capturing, received


_STATIC_GATE_TIERS: list[tuple[type[BaseModel], OverlayTier]] = [
    (_FullGateArguments, OverlayTier.FULL),
    (_NestedSerializerArguments, OverlayTier.TOP_LEVEL),
    (_FieldSerializerArguments, OverlayTier.TOP_LEVEL),
    (_KeySerializerArguments, OverlayTier.TOP_LEVEL),
    (_AliasedTypedDictArguments, OverlayTier.TOP_LEVEL),
    (_DefaultedTypedDictArguments, OverlayTier.TOP_LEVEL),
    (_AliasedDataclassArguments, OverlayTier.TOP_LEVEL),
    (_UnknownContainerArguments, OverlayTier.TOP_LEVEL),
    (_RootSerializerArguments, OverlayTier.OFF),
    (_AnyPydanticDataclassArguments, OverlayTier.FULL),
    (_EnumGateArguments, OverlayTier.FULL),
]


class TestNullOverlayPolicy:
    """Static tier assignment and the runtime null-restoration each tier permits."""

    @pytest.mark.parametrize(
        ("input_model", "tier"),
        _STATIC_GATE_TIERS,
        ids=[input_model.__name__ for input_model, _tier in _STATIC_GATE_TIERS],
    )
    def test_static_gate_assigns_full_top_level_and_off_tiers(
        self, input_model: type[BaseModel], tier: OverlayTier
    ) -> None:
        assert classify(input_model).tier is tier

    def test_new_type_detection_does_not_unwrap_ordinary_classes(self) -> None:
        full_policy = classify(_FullGateArguments)
        assert _NullableValue in full_policy.approved_classes
        assert classify(_SupertypeAttributeArguments).tier is OverlayTier.TOP_LEVEL

    async def test_validation_metadata_is_full_and_restores_nulls(self) -> None:
        constraints = StringConstraints(min_length=1)
        assert isinstance(constraints, annotated_types.GroupedMetadata)

        metadata_tool, received = _capturing_tool(_ValidationMetadataArguments, name="validation_metadata")

        assert metadata_tool._null_overlay_policy.tier is OverlayTier.FULL
        await metadata_tool.invoke(arguments={"count": None, "name": None, "score": None})
        assert received == [{"count": None, "name": None, "score": None}]

    def test_serializer_metadata_still_degrades_with_safe_metadata(self) -> None:
        policy = classify(_SafeMetadataWithSerializerArguments)

        assert policy.tier is OverlayTier.TOP_LEVEL
        assert policy.reason == "Annotated serializer metadata"
        assert "value" not in policy.top_level_restorable

    async def test_field_info_metadata_cannot_hide_a_serializer(self) -> None:
        carrier_tool, received = _capturing_tool(_FieldInfoCarrierArguments, name="field_info_carrier")

        assert carrier_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert classify(_FieldConstraintArguments).tier is OverlayTier.FULL
        assert classify(_ValidationMetadataArguments).tier is OverlayTier.FULL
        await carrier_tool.invoke(arguments={"items": [{"value": None}]})
        assert received == [{"items": [{"owned": True}]}]

    async def test_custom_core_schema_hooks_degrade_reachable_classes(self) -> None:
        child_tool, received = _capturing_tool(_CoreSchemaArguments, name="core_schema_child")
        key_tool, _ = _capturing_tool(_CoreSchemaKeyArguments, name="core_schema_key", received=received)
        dataclass_tool, _ = _capturing_tool(
            _CoreSchemaDataclassArguments, name="core_schema_dataclass", received=received
        )

        assert child_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert key_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert dataclass_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert classify(_NestedArguments).tier is OverlayTier.FULL
        await child_tool.invoke(arguments={"child": {"value": None}})
        await key_tool.invoke(arguments={"payload": {"value": {"value": None}}})
        await dataclass_tool.invoke(arguments={"child": {"value": None}})
        assert received == [
            {"child": {"owned": True}},
            {"payload": {"SERIALIZED": {}}},
            {"child": {"owned": True}},
        ]

    async def test_root_model_dump_override_is_off_but_nested_override_is_plain(self) -> None:
        root_tool, received = _capturing_tool(_RootDumpOverrideArguments, name="root_dump_override")
        nested_tool, _ = _capturing_tool(_NestedDumpOverrideArguments, name="nested_dump_override", received=received)

        assert root_tool._null_overlay_policy.tier is OverlayTier.OFF
        assert nested_tool._null_overlay_policy.tier is OverlayTier.FULL
        await root_tool.invoke(arguments={"value": None})
        await nested_tool.invoke(arguments={"child": {"value": None}})
        assert received == [
            {"owned": True},
            {"child": {"value": None}},
        ]

    async def test_exclude_if_fields_keep_claims_and_annotation_checks(self) -> None:
        collision_tool, received = _capturing_tool(_ExcludeIfCollisionArguments, name="exclude_if_collision")
        plain_tool, _ = _capturing_tool(_ExcludeIfPlainArguments, name="exclude_if_plain", received=received)

        assert collision_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert plain_tool._null_overlay_policy.tier is OverlayTier.FULL
        await collision_tool.invoke(arguments={"owned": {"value": None}, "conditional": {"value": None}})
        await plain_tool.invoke(arguments={"safe": None, "conditional": {"value": None}})
        assert received == [
            {"slot": {}},
            {"safe": None, "conditional": {}},
        ]

    async def test_derived_dataclass_computed_metadata_wins(self) -> None:
        dataclass_tool, received = _capturing_tool(_ComputedAliasDataclassArguments, name="derived_computed_alias")

        assert dataclass_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        await dataclass_tool.invoke(arguments={"child": {"owned": {"value": None}}})
        assert received == [{"child": {"owned": {}}}]

    async def test_declared_slot_classes_guard_runtime_substitutions(self) -> None:
        slot_tool, received = _capturing_tool(_SlotClassArguments, name="slot_classes")

        assert slot_tool._null_overlay_policy.tier is OverlayTier.FULL
        await slot_tool.invoke(
            arguments={
                "child": {"value": "child"},
                "other": {"value": "other", "extra": None},
                "choice": {"kind": "right", "value": None},
            }
        )
        assert received == [
            {
                "child": {"value": "child"},
                "other": {"value": "other", "extra": None},
                "choice": {"kind": "right", "value": None},
            }
        ]

    async def test_root_model_unwraps_before_dump_shape_guard(self) -> None:
        root_tool, received = _capturing_tool(_RootModelArguments, name="root_model_restore")

        assert root_tool._null_overlay_policy.tier is OverlayTier.FULL
        await root_tool.invoke(arguments={"payload": [{"value": None}]})
        assert received == [{"payload": [{"value": None}]}]

    async def test_serialization_defaults_required_does_not_require_omitted_defaults(self) -> None:
        defaults_tool, received = _capturing_tool(
            _SerializationDefaultsRequiredArguments, name="serialization_defaults_required"
        )

        await defaults_tool.invoke(arguments={})
        await defaults_tool.invoke(arguments={"value": None})
        assert received == [{}, {"value": None}]

    @pytest.mark.parametrize(
        ("input_model", "arguments", "expected", "tier"),
        [
            (
                _GroupedSerializerArguments,
                {"safe": None, "child": {"value": None}},
                {"safe": None, "child": {"grouped_owned": True}},
                OverlayTier.TOP_LEVEL,
            ),
            (
                _NewTypeSerializerArguments,
                {"child": {"value": None}},
                {"child": {"new_type_owned": True}},
                OverlayTier.TOP_LEVEL,
            ),
            (
                _TypedDictSerializerArguments,
                {"payload": {"child": {"value": None}}},
                {"payload": {"typed_dict_owned": True}},
                OverlayTier.TOP_LEVEL,
            ),
            (
                _MetaclassSerializerArguments,
                {"value": None},
                {"metaclass_owned": True},
                OverlayTier.OFF,
            ),
        ],
        ids=["grouped-metadata", "new-type", "typed-dict", "metaclass-property"],
    )
    async def test_compiled_schema_scan_covers_every_serializer_ingress(
        self,
        input_model: type[BaseModel],
        arguments: dict[str, Any],
        expected: dict[str, Any],
        tier: OverlayTier,
    ) -> None:
        schema_tool, received = _capturing_tool(input_model, name="compiled_schema_gate")

        assert schema_tool._null_overlay_policy.tier is tier
        await schema_tool.invoke(arguments=arguments)
        assert received == [expected]

    async def test_compiled_scan_preserves_plain_and_precise_early_rules(self) -> None:
        root_policy = classify(_RootSerializerArguments)
        nested_tool, received = _capturing_tool(_NestedPlainSerializerArguments, name="nested_plain_serializer")

        assert classify(_FullGateArguments).tier is OverlayTier.FULL
        assert classify(_ValidationMetadataArguments).tier is OverlayTier.FULL
        assert root_policy.tier is OverlayTier.OFF
        assert root_policy.reason == "model serializer on _RootSerializerArguments"
        assert nested_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        await nested_tool.invoke(arguments={"safe": None, "child": {"value": None}})
        assert received == [{"safe": None, "child": {"plain_owned": True}}]

    async def test_typed_dict_config_follows_orig_bases_precedence(self) -> None:
        inherited_tool, received = _capturing_tool(_InheritedAliasArguments, name="inherited_typed_dict_config")
        precedence_value = _PrecedenceAliasArguments.model_validate(
            {"payload": {"own_first": None, "own_second": "kept"}}
        )

        assert inherited_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert classify(_PlainInheritedArguments).tier is OverlayTier.FULL
        assert classify(_PrecedenceAliasArguments).tier is OverlayTier.TOP_LEVEL
        assert precedence_value.model_dump() == {"payload": {"own_first": None, "own_second": "kept"}}
        await inherited_tool.invoke(arguments={"payload": {"FIRST": None, "SECOND": "kept"}})
        assert received == [{"payload": {"SECOND": "kept"}}]

    @pytest.mark.parametrize("input_model", [_CoDeclaredTupleArguments, _CoDeclaredUnionArguments])
    def test_subclass_pairs_in_one_slot_degrade(self, input_model: type[BaseModel]) -> None:
        policy = classify(input_model)

        assert policy.tier is OverlayTier.TOP_LEVEL
        assert policy.reason == "co-declared subclass pair _SlotBase/_SlotSub in one slot"

    async def test_tuple_subclass_pair_output_is_not_descended(self) -> None:
        pair_tool, received = _capturing_tool(_CoDeclaredTupleArguments, name="subclass_pair")
        await pair_tool.invoke(
            arguments={
                "pair": [
                    {"value": "first"},
                    {"value": "second", "extra": "kept"},
                ]
            }
        )

        assert received == [
            {
                "pair": (
                    {"value": "first"},
                    {"value": "second", "extra": "kept"},
                )
            }
        ]

    async def test_non_function_compiled_serialization_degrades_nested_restore(self) -> None:
        serialized_tool, received = _capturing_tool(_ModelSerializationArguments, name="model_serialization_node")

        assert serialized_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert classify(_FullGateArguments).tier is OverlayTier.FULL
        assert classify(_ValidationMetadataArguments).tier is OverlayTier.FULL
        assert classify(_PydanticInternalSerializationArguments).tier is OverlayTier.TOP_LEVEL
        await serialized_tool.invoke(arguments={"child": {"value": None}})
        assert received == [{"child": {}}]

    async def test_degraded_root_model_without_compiled_fields_restores_no_slots(self) -> None:
        root_tool, received = _capturing_tool(_SerializedRootArguments, name="serialized_root")

        policy = root_tool._null_overlay_policy
        assert policy.tier is OverlayTier.TOP_LEVEL
        assert policy.top_level_restorable == frozenset()
        assert classify(_AnyUrlWithSafeRootFieldArguments).top_level_restorable == frozenset({"safe", "url"})
        await root_tool.invoke(arguments={})
        assert received == [{"owned": True}]

    async def test_serialization_named_field_is_not_a_compiled_schema_node(self) -> None:
        plain_tool, received = _capturing_tool(_SerializationNamedFieldArguments, name="serialization_named_field")

        assert plain_tool._null_overlay_policy.tier is OverlayTier.FULL
        assert classify(_ModelSerializationArguments).tier is OverlayTier.TOP_LEVEL
        assert classify(_PydanticInternalSerializationArguments).tier is OverlayTier.TOP_LEVEL
        await plain_tool.invoke(arguments={"serialization": "plain", "child": {"value": None}})
        assert received == [{"serialization": "plain", "child": {"value": None}}]

    async def test_root_model_fields_serialization_disables_restoration(self) -> None:
        root_tool, received = _capturing_tool(_ModelFieldsSerializerArguments, name="model_fields_serializer")

        assert root_tool._null_overlay_policy.tier is OverlayTier.OFF
        await root_tool.invoke(arguments={"value": None})
        assert received == [{"owned": True}]

    async def test_definition_references_preserve_compiled_field_ownership(self) -> None:
        serialized_tool, received = _capturing_tool(_RecursiveSerializedArguments, name="recursive_serialized_child")

        serialized_policy = serialized_tool._null_overlay_policy
        plain_policy = classify(_RecursivePlainDegradedArguments)
        assert serialized_policy.tier is OverlayTier.TOP_LEVEL
        assert "child" not in serialized_policy.top_level_restorable
        assert plain_policy.tier is OverlayTier.TOP_LEVEL
        assert plain_policy.top_level_restorable == frozenset({"child", "url"})
        await serialized_tool.invoke(arguments={"child": None})
        assert received == [{}]

    async def test_defaulted_dataclass_fields_are_not_descended_without_provenance(self) -> None:
        defaulted_tool, received = _capturing_tool(
            _DefaultedNestedDataclassArguments, name="defaulted_dataclass_descent"
        )

        await defaulted_tool.invoke(arguments={"child": {}})
        await defaulted_tool.invoke(arguments={"child": {"payload": {"value": None}}})
        assert received == [
            {"child": {"payload": {}}},
            {"child": {"payload": {}}},
        ]

    async def test_required_dataclass_fields_descend_and_non_none_defaults_restore_null(self) -> None:
        required_tool, received = _capturing_tool(_RequiredNestedDataclassArguments, name="required_dataclass_descent")
        direct_null_tool, _ = _capturing_tool(
            _NonNoneDefaultDataclassArguments, name="non_none_dataclass_default", received=received
        )

        await required_tool.invoke(arguments={"child": {"payload": {"value": None}}})
        await direct_null_tool.invoke(arguments={"child": {"value": None}})
        assert received == [
            {"child": {"payload": {"value": None}}},
            {"child": {"value": None}},
        ]

    async def test_compiled_field_serialization_excludes_only_its_top_level_slot(self) -> None:
        serialized_tool, received = _capturing_tool(_CustomSlotSerializerArguments, name="custom_slot_serialization")

        policy = serialized_tool._null_overlay_policy
        assert policy.tier is OverlayTier.TOP_LEVEL
        assert policy.top_level_restorable == frozenset({"safe"})
        await serialized_tool.invoke(arguments={"safe": None, "value": None})
        assert received == [{"safe": None}]

    async def test_declared_field_and_same_named_extra_restore_independently(self) -> None:
        aliased_tool, received = _capturing_tool(
            _AliasedFieldWithInternalNameExtraArguments, name="field_extra_internal_name"
        )

        await aliased_tool.invoke(arguments={"wire": None, "value": "extra"})
        assert received == [{"wire": None, "value": "extra"}]

    async def test_excluded_fields_do_not_reserve_extra_dump_keys(self) -> None:
        excluded_tool, received = _capturing_tool(
            _ExcludedFieldWithAliasedExtraArguments, name="excluded_field_extra_key"
        )
        emitting_tool, _ = _capturing_tool(
            _EmittingFieldWithCollidingExtraArguments, name="emitting_field_extra_key", received=received
        )

        await excluded_tool.invoke(arguments={"x": None})
        await emitting_tool.invoke(arguments={"inputVisible": None, "x": None})
        assert received == [{"x": None}, {}]

    async def test_unhashable_annotation_degrades_without_bricking_invocation(self) -> None:
        annotation_tool, received = _capturing_tool(_UnhashableAnnotationArguments, name="unhashable_annotation")

        assert annotation_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        await annotation_tool.invoke(arguments={"value": "kept"})
        assert received == [{"value": "kept"}]

    async def test_classifier_exceptions_use_a_cached_degraded_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Arguments(BaseModel):
            value: str | None

        def fail_classification(model: type[BaseModel]) -> Any:
            raise RuntimeError("classification failed")

        monkeypatch.setattr("chrys.kernel.tools.classify", fail_classification)
        degraded_tool, received = _capturing_tool(_Arguments, name="classifier_failure")

        first = degraded_tool._null_overlay_policy
        second = degraded_tool._null_overlay_policy
        assert first is second
        assert first.tier is OverlayTier.TOP_LEVEL
        await degraded_tool.invoke(arguments={"value": None})
        assert received == [{"value": None}]

    def test_tool_caches_one_policy_and_logs_one_degraded_classification(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        degraded_tool = FunctionTool(
            name="degraded_policy",
            description="Cache a static policy.",
            func=lambda **kwargs: kwargs,
            input_model=_NestedSerializerArguments,
        )
        with caplog.at_level(logging.DEBUG, logger="chrys.kernel.tools"):
            first = degraded_tool._null_overlay_policy
            second = degraded_tool._null_overlay_policy

        assert first is second
        records = [record for record in caplog.records if "degraded_policy" in record.getMessage()]
        assert len(records) == 1

    async def test_explicit_and_nested_nulls_reach_callable_under_full_policy(self) -> None:
        explicit_tool, received = _capturing_tool(_FullGateArguments, name="full_restore")
        await explicit_tool.invoke(
            arguments={
                "child": {"value": None},
                "members": {"value": None},
                "items": [{"value": None}],
                "mapping": {"x": {"value": None}},
                "std_new": {"value": None},
                "ext_new": {"value": None},
                "validated": "7",
                "choice": None,
                "sequence": [{"value": None}],
                "abstract_mapping": {"a": {"value": None}},
                "mutable_mapping": {"b": {"value": None}},
                "members_set": [None, "x"],
                "frozen_members": [None, "x"],
                "mode": "plain",
            }
        )

        assert received == [
            {
                "child": {"value": None},
                "members": {"value": None},
                "items": [{"value": None}],
                "mapping": {"x": {"value": None}},
                "std_new": {"value": None},
                "ext_new": {"value": None},
                "validated": 7,
                "choice": None,
                "sequence": [{"value": None}],
                "abstract_mapping": {"a": {"value": None}},
                "mutable_mapping": {"b": {"value": None}},
                "members_set": {None, "x"},
                "frozen_members": frozenset({None, "x"}),
                "mode": "plain",
            }
        ]

    async def test_tuple_restore_preserves_python_mode_tuple(self) -> None:
        tuple_tool, received = _capturing_tool(_TupleArguments, name="tuple_restore")
        await tuple_tool.invoke(arguments={"items": [{"value": None}, {"value": "x"}]})

        assert received == [{"items": ({"value": None}, {"value": "x"})}]

    async def test_typed_dict_omitted_member_stays_absent(self) -> None:
        td_tool, received = _capturing_tool(_TypedDictArguments, name="typed_dict_restore")
        await td_tool.invoke(arguments={"payload": {"value": None}})

        assert received == [{"payload": {"value": None}}]

    async def test_dataclass_static_null_provenance_and_pydantic_dataclass_restore(self) -> None:
        std_tool, received = _capturing_tool(_DataclassArguments, name="stdlib_dataclass_restore")
        pdc_tool, _ = _capturing_tool(_PydanticDataclassArguments, name="pydantic_dataclass_restore", received=received)

        await std_tool.invoke(arguments={"child": {"required": None, "non_none_default": None}})
        await pdc_tool.invoke(arguments={"child": {"value": None}})

        assert received == [
            {"child": {"required": None, "non_none_default": None}},
            {"child": {"value": None}},
        ]

    async def test_dataclass_field_metadata_defaults_do_not_invent_nulls(self) -> None:
        metadata_tool, received = _capturing_tool(_MetadataDefaultDataclassArguments, name="metadata_defaults")
        await metadata_tool.invoke(arguments={"child": {"required": None}})

        assert received == [{"child": {"required": None}}]

    async def test_alias_extras_exclusion_and_computed_fields(self) -> None:
        alias_tool, received = _capturing_tool(_AliasedNullable, name="alias_restore")
        extra_tool, _ = _capturing_tool(_ExtraArguments, name="extra_restore", received=received)
        excluded_tool, _ = _capturing_tool(_ExcludedArguments, name="excluded_restore", received=received)
        computed_tool, _ = _capturing_tool(_ComputedArguments, name="computed_output", received=received)

        await alias_tool.invoke(arguments={"wire": None})
        await extra_tool.invoke(arguments={"runtime": None})
        await excluded_tool.invoke(arguments={"visible": None, "hidden": None})
        await computed_tool.invoke(arguments={"count": 2})

        assert received == [
            {"wire": None},
            {"anchor": 1, "runtime": None},
            {"visible": None},
            {"count": 2},
        ]

    async def test_top_level_tier_restores_only_safe_root_fields(self) -> None:
        models._NESTED_SERIALIZER_CALLS = 0
        models._FIELD_SERIALIZER_CALLS = 0
        nested_tool, received = _capturing_tool(_NestedSerializerArguments, name="nested_serializer_gate")
        field_tool, _ = _capturing_tool(_FieldSerializerArguments, name="field_serializer_gate", received=received)
        assert nested_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert field_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert models._NESTED_SERIALIZER_CALLS == 0
        assert models._FIELD_SERIALIZER_CALLS == 0

        await nested_tool.invoke(arguments={"safe": None, "child": {"value": None}})
        await field_tool.invoke(arguments={"safe": None, "text": "hello"})

        assert received == [
            {"safe": None, "child": {"owned": True}},
            {"safe": None, "text": "HELLO"},
        ]
        assert models._NESTED_SERIALIZER_CALLS == 1
        assert models._FIELD_SERIALIZER_CALLS == 1

    async def test_off_tier_preserves_root_serializer_output(self) -> None:
        root_tool, received = _capturing_tool(_RootSerializerArguments, name="root_serializer_gate")
        await root_tool.invoke(arguments={"value": None})

        assert received == [{"owned": True}]

    async def test_serialized_mapping_keys_are_uniformly_left_untouched(self) -> None:
        models._KEY_SERIALIZER_CALLS = 0
        key_tool, received = _capturing_tool(_KeySerializerArguments, name="serialized_keys")
        assert key_tool._null_overlay_policy.tier is OverlayTier.TOP_LEVEL
        assert models._KEY_SERIALIZER_CALLS == 0
        await key_tool.invoke(arguments={"payload": {"X": {"value": None}, "x": {}}})
        await key_tool.invoke(arguments={"payload": {"x": {"value": None}}})

        assert received == [{"payload": {"X": {}}}, {"payload": {"X": {}}}]
        assert models._KEY_SERIALIZER_CALLS == 3

    async def test_supertype_class_attribute_does_not_bypass_serializer_gate(self) -> None:
        supertype_tool, received = _capturing_tool(_SupertypeAttributeArguments, name="supertype_attribute")
        await supertype_tool.invoke(arguments={"child": {"value": None}})

        assert received == [{"child": {"owned": True}}]

    async def test_runtime_dataclass_under_any_is_an_unapproved_leaf(self) -> None:
        baseline = models._PDC_ALIAS_GENERATOR_CALLS
        any_tool, received = _capturing_tool(_AnyPydanticDataclassArguments, name="any_pydantic_dataclass")
        declared_tool, _ = _capturing_tool(
            _DeclaredPydanticDataclassArguments, name="declared_generated_dataclass", received=received
        )
        payload = _GeneratedPydanticDataclass(value=None)

        await any_tool.invoke(arguments={"payload": payload})
        await declared_tool.invoke(arguments={"payload": payload})

        assert received == [{"payload": {}}, {"payload": {}}]
        assert baseline == models._PDC_ALIAS_GENERATOR_CALLS

    @pytest.mark.parametrize("value", [None, "x"])
    async def test_split_alias_invokes_in_serialization_keyspace(self, value: Any) -> None:
        alias_tool, received = _capturing_tool(_SplitAliasArguments, name="split_alias")
        await alias_tool.invoke(arguments={"inputKey": value})

        assert received == [{"callableKey": value}]

    async def test_unschemable_serialization_mirror_degrades_without_crashing(self) -> None:
        unschemable_tool, received = _capturing_tool(_UnschemableComputedArguments, name="unschemable")
        await unschemable_tool.invoke(arguments={"inKey": None})

        assert received[0]["outKey"] is None
        assert callable(received[0]["factory"])

    async def test_empty_alias_is_a_real_dump_key(self) -> None:
        alias_tool, received = _capturing_tool(_EmptyAliasArguments, name="empty_alias")
        await alias_tool.invoke(arguments={"": None})

        assert received == [{"": None}]

    async def test_omitted_model_default_keeps_callable_default(self) -> None:
        received: list[int] = []

        async def capture(count: int = 3) -> str:
            received.append(count)
            return "ok"

        default_tool = FunctionTool(
            name="model_default",
            description="Omitted fields stay omitted.",
            func=capture,
            input_model=_DivergentDefaults,
        )
        await default_tool.invoke(arguments={})

        assert received == [7]

    async def test_schema_supplied_array_stays_strict_and_raw_null_passes_through(self) -> None:
        schema_tool, received = _capturing_tool(
            {
                "type": "object",
                "properties": {"items": {"type": "array"}, "value": {"type": ["string", "null"]}},
                "required": ["items", "value"],
                "additionalProperties": False,
            },
            name="schema_tool",
        )

        await schema_tool.invoke(arguments={"items": [1, 2], "value": None})
        with pytest.raises(TypeError, match="Invalid type for 'items'"):
            await schema_tool.invoke(arguments={"items": (1, 2), "value": None})

        assert received == [{"items": [1, 2], "value": None}]
