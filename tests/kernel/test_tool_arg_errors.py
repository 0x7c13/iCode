# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for compact, model-actionable tool argument errors."""

from __future__ import annotations

import sys
import warnings
from typing import Literal

import pytest
from pydantic import (
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationError,
    field_validator,
    model_validator,
    root_validator,
)
from pydantic_core import PydanticCustomError
from pydantic_core import core_schema as pydantic_core_schema

from chrys.kernel._tool_arg_errors import (
    _MAX_MARKER_SCAN_CHARS,
    _MAX_SAFE_ARGUMENT_ENTRIES,
    _MAX_UNEXPECTED_NAMES,
    _MAX_VALIDATION_ERRORS,
    _accepts_arbitrary_argument_names,
    _argument_validation_message,
    _capped_validation_errors,
    _display_safe_arguments,
    _rejects_unexpected_arguments,
    _schema_type_text,
    _unexpected_argument_names,
)
from tests.kernel._tool_arg_helpers import _message_for
from tests.support.cpu_guard import CPU_TIME_BOUND_SECONDS, cpu_bounded


def test_argument_validation_guidance_is_concise_and_shows_expected_shape() -> None:
    class AskArguments(BaseModel):
        question: str
        options: list[str] | None = Field(
            default=None,
            description='Flat array of strings, for example ["A", "B"].',
        )

    arguments = {"question": "Pick?", "options": 7}
    message = _message_for(AskArguments, arguments, tool_name="ask_user")

    assert message == (
        "Invalid arguments for 'ask_user': options: expected [string] | null, got integer. "
        "Expected: {question: string, options?: [string] | null}."
    )
    assert len(message) < 150


def test_argument_name_policy_rejects_typed_unknowns_but_honors_dynamic_schema() -> None:
    closed_schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    dynamic_schema = {**closed_schema, "additionalProperties": True}

    assert _rejects_unexpected_arguments(closed_schema, typed_model_dump=True)
    assert _unexpected_argument_names(
        {"path": "a.py", "file_path": "b.py"},
        closed_schema,
        reject_unexpected=True,
    ) == ["file_path"]
    assert not _rejects_unexpected_arguments(dynamic_schema, typed_model_dump=True)
    assert (
        _unexpected_argument_names(
            {"path": "a.py", "file_path": "b.py"},
            dynamic_schema,
            reject_unexpected=False,
        )
        == []
    )


def test_argument_guidance_handles_fixed_tuple_and_all_of_schema() -> None:
    class TupleArguments(BaseModel):
        pair: tuple[str, int]

    arguments = {"pair": ["ok", "bad"]}
    message = _message_for(TupleArguments, arguments, tool_name="tuple_tool")
    assert "pair[1]: expected integer" in message
    assert "Expected: {pair: [string, integer]}" in message

    all_of_schema = {
        "$defs": {"Payload": {"type": "object", "properties": {"path": {"type": "string"}}}},
        "allOf": [{"$ref": "#/$defs/Payload"}],
    }
    assert _schema_type_text(all_of_schema, all_of_schema) == "{path?: string}"


def test_argument_guidance_reports_nested_missing_and_unknown_fields() -> None:
    class Item(BaseModel):
        model_config = ConfigDict(extra="forbid")

        content: str

    class ItemsArguments(BaseModel):
        items: list[Item]

    arguments = {"items": [{"file_path": "wrong"}]}
    message = _message_for(ItemsArguments, arguments, tool_name="items_tool")
    assert "missing 'items[0].content'" in message
    assert "unknown 'items[0].file_path'" in message
    assert "Expected: {items: [{content: string}]}" in message


def test_schema_supplied_error_uses_schema_without_echoing_values() -> None:
    schema = {
        "type": "object",
        "properties": {"mode": {"enum": ["safe", "fast"]}},
        "required": ["mode"],
        "additionalProperties": False,
    }
    secret = "do-not-repeat-this-value"

    message = _argument_validation_message(
        tool_name="configure",
        arguments={"mode": secret},
        arguments_unparseable=False,
        schema=schema,
        exception=TypeError(f"Invalid value: {secret}"),
        reject_unexpected=True,
    )

    assert message == (
        'Invalid arguments for \'configure\': mode: expected "safe" | "fast", got string. '
        'Expected: {mode: "safe" | "fast"}.'
    )
    assert secret not in message


def test_argument_guidance_descends_into_mapping_value_schemas() -> None:
    class WeightsArguments(BaseModel):
        weights: dict[str, int]

    arguments = {"weights": {"alpha": "x"}}
    message = _message_for(WeightsArguments, arguments, tool_name="weights_tool")
    assert "weights.alpha: expected integer" in message
    assert "expected {string: integer}" not in message
    assert "Expected: {weights: {string: integer}}" in message


def test_argument_guidance_does_not_echo_custom_validator_messages() -> None:
    class TokenArguments(BaseModel):
        token: str
        key: str

        @field_validator("token")
        @classmethod
        def _check_token(cls, value: str) -> str:
            if not value.startswith("tok_"):
                raise ValueError(f"invalid token {value}")
            return value

        @field_validator("key")
        @classmethod
        def _check_key(cls, value: str) -> str:
            raise PydanticCustomError("key_error", "key {key} is rejected", {"key": value})

    secret = "super-secret-payload"
    arguments = {"token": secret, "key": secret}
    message = _message_for(TokenArguments, arguments, tool_name="token_tool")
    assert secret not in message
    assert "token: expected string" in message
    assert "key: expected string" in message
    assert "Expected: {token: string, key: string}" in message


def test_argument_guidance_suppresses_spoofed_safe_type_messages() -> None:
    class SpoofArguments(BaseModel):
        token: str

        @field_validator("token")
        @classmethod
        def _check_token(cls, value: str) -> str:
            raise PydanticCustomError("int_parsing", "rejected secret {secret}", {"secret": value})

    secret = "spoofed-secret-value"
    arguments = {"token": secret}
    message = _message_for(SpoofArguments, arguments, tool_name="spoof_tool")
    assert secret not in message
    assert "token: expected string" in message


def test_argument_name_policy_honors_composed_root_schemas() -> None:
    root_model = RootModel[dict[str, int] | None]
    schema = root_model.model_json_schema()
    assert not _rejects_unexpected_arguments(schema, typed_model_dump=True)


def test_argument_name_policy_accepts_pydantic_alias_spellings() -> None:
    class AliasArguments(BaseModel):
        model_config = ConfigDict(validate_by_name=True)

        path: str = Field(alias="file_path")

    schema = AliasArguments.model_json_schema()
    assert (
        _unexpected_argument_names(
            {"path": "a.py"},
            schema,
            reject_unexpected=True,
            input_model=AliasArguments,
        )
        == []
    )
    assert (
        _unexpected_argument_names(
            {"file_path": "a.py"},
            schema,
            reject_unexpected=True,
            input_model=AliasArguments,
        )
        == []
    )
    assert _unexpected_argument_names(
        {"paht": "a.py"},
        schema,
        reject_unexpected=True,
        input_model=AliasArguments,
    ) == ["paht"]


def test_argument_name_policy_rejects_field_name_pydantic_would_silently_ignore() -> None:
    class StrictAliasArguments(BaseModel):
        path: str = Field(default="DEFAULT", alias="file_path")

    schema = StrictAliasArguments.model_json_schema()
    assert StrictAliasArguments.model_validate({"path": "requested.txt"}).path == "DEFAULT"
    assert _unexpected_argument_names(
        {"path": "requested.txt"},
        schema,
        reject_unexpected=True,
        input_model=StrictAliasArguments,
    ) == ["path"]
    assert (
        _unexpected_argument_names(
            {"file_path": "requested.txt"},
            schema,
            reject_unexpected=True,
            input_model=StrictAliasArguments,
        )
        == []
    )


def test_argument_name_policy_honors_alias_and_name_validation_toggles() -> None:
    class AliasDisabled(BaseModel):
        model_config = ConfigDict(validate_by_alias=False, validate_by_name=True)

        path: str = Field(default="DEFAULT", alias="file_path")

    schema = AliasDisabled.model_json_schema()
    assert AliasDisabled.model_validate({"file_path": "ignored"}).path == "DEFAULT"
    assert _unexpected_argument_names(
        {"file_path": "ignored"},
        schema,
        reject_unexpected=True,
        input_model=AliasDisabled,
    ) == ["file_path"]
    assert (
        _unexpected_argument_names(
            {"path": "used"},
            schema,
            reject_unexpected=True,
            input_model=AliasDisabled,
        )
        == []
    )

    class NameDisabled(BaseModel):
        model_config = ConfigDict(populate_by_name=True, validate_by_name=False)

        path: str = Field(default="DEFAULT", alias="file_path")

    name_disabled_schema = NameDisabled.model_json_schema()
    assert NameDisabled.model_validate({"path": "ignored"}).path == "DEFAULT"
    assert _unexpected_argument_names(
        {"path": "ignored"},
        name_disabled_schema,
        reject_unexpected=True,
        input_model=NameDisabled,
    ) == ["path"]


def test_argument_name_policy_stands_down_for_before_validator_models() -> None:
    class PlainArguments(BaseModel):
        x: int

    class RewritingArguments(BaseModel):
        x: int

        @model_validator(mode="before")
        @classmethod
        def _unwrap(cls, data: object) -> object:
            if isinstance(data, dict) and "payload" in data:
                return {"x": data["payload"]}
            return data

    class AfterOnlyArguments(BaseModel):
        x: int

        @model_validator(mode="after")
        def _check(self) -> AfterOnlyArguments:
            return self

    assert not _accepts_arbitrary_argument_names(None)
    assert not _accepts_arbitrary_argument_names(PlainArguments)
    assert not _accepts_arbitrary_argument_names(AfterOnlyArguments)
    assert _accepts_arbitrary_argument_names(RewritingArguments)
    assert RewritingArguments.model_validate({"payload": 7}).x == 7

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        class LegacyRewritingArguments(BaseModel):
            x: int

            @root_validator(pre=True)
            @classmethod
            def _unwrap(cls, values: object) -> object:
                if isinstance(values, dict) and "payload" in values:
                    return {"x": values["payload"]}
                return values

    assert _accepts_arbitrary_argument_names(LegacyRewritingArguments)
    assert LegacyRewritingArguments.model_validate({"payload": 7}).x == 7

    class CoreSchemaRewritingArguments(BaseModel):
        x: int

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: object, handler: object) -> object:
            schema = handler(source_type)  # type: ignore[operator]
            return pydantic_core_schema.no_info_before_validator_function(cls._unwrap, schema)

        @staticmethod
        def _unwrap(data: object) -> object:
            if isinstance(data, dict) and "payload" in data:
                return {"x": data["payload"]}
            return data

    assert _accepts_arbitrary_argument_names(CoreSchemaRewritingArguments)
    assert CoreSchemaRewritingArguments.model_validate({"payload": 7}).x == 7

    class JsonOrPythonRewritingArguments(BaseModel):
        x: int

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: object, handler: object) -> object:
            schema = handler(source_type)  # type: ignore[operator]
            wrapped = pydantic_core_schema.no_info_before_validator_function(cls._unwrap, schema)
            return pydantic_core_schema.json_or_python_schema(json_schema=wrapped, python_schema=wrapped)

        @staticmethod
        def _unwrap(data: object) -> object:
            if isinstance(data, dict) and "payload" in data:
                return {"x": data["payload"]}
            return data

    assert _accepts_arbitrary_argument_names(JsonOrPythonRewritingArguments)
    assert JsonOrPythonRewritingArguments.model_validate({"payload": 7}).x == 7

    class RecursiveArguments(BaseModel):
        child: RecursiveArguments | None = None

    assert not _accepts_arbitrary_argument_names(RecursiveArguments)

    class CustomInitArguments(BaseModel):
        x: int

        def __init__(self, **data: object) -> None:
            if "payload" in data:
                data = {"x": data.pop("payload"), **data}
            super().__init__(**data)

    assert _accepts_arbitrary_argument_names(CustomInitArguments)
    assert CustomInitArguments.model_validate({"payload": 7}).x == 7

    class ValidateOverrideArguments(BaseModel):
        x: int

        @classmethod
        def model_validate(cls, obj: object, **kwargs: object) -> ValidateOverrideArguments:
            # The loop validates through this method, so a class-level
            # override rewrites keys invisibly to the core schema and the
            # decorator registries alike.
            if isinstance(obj, dict) and "payload" in obj:
                obj = {"x": obj["payload"]}
            return super().model_validate(obj, **kwargs)  # type: ignore[return-value]

    assert _accepts_arbitrary_argument_names(ValidateOverrideArguments)
    assert ValidateOverrideArguments.model_validate({"payload": 7}).x == 7


def test_argument_guidance_skips_static_missing_for_before_validator_models() -> None:
    class RewritingArguments(BaseModel):
        x: int
        y: int

        @model_validator(mode="before")
        @classmethod
        def _unwrap(cls, data: object) -> object:
            if isinstance(data, dict) and "payload" in data:
                data = {**data, "x": data["payload"]}
                del data["payload"]
            return data

    arguments = {"payload": 7, "y": "bad"}
    schema = RewritingArguments.model_json_schema()
    with pytest.raises(ValidationError) as caught:
        RewritingArguments.model_validate(arguments)

    message = _argument_validation_message(
        tool_name="rewrite_tool",
        arguments=arguments,
        arguments_unparseable=False,
        schema=schema,
        exception=caught.value,
        reject_unexpected=False,
        input_model=RewritingArguments,
    )
    assert "missing" not in message
    assert "y: expected integer" in message


def test_argument_guidance_reconciles_serialization_alias_schema_names() -> None:
    class SerializationAliasArguments(BaseModel):
        model_config = ConfigDict(json_schema_mode_override="serialization")

        path: str = Field(validation_alias="input_path", serialization_alias="output_path")

    schema = SerializationAliasArguments.model_json_schema()
    assert "output_path" in schema["properties"]

    message = _message_for(
        SerializationAliasArguments, {}, tool_name="alias_tool", input_model=SerializationAliasArguments, schema=schema
    )
    assert message.count("missing") == 1
    assert "missing 'input_path'" in message
    assert "output_path" not in message
    assert "Expected: {input_path: string}" in message


def test_argument_guidance_resolves_cross_field_alias_collisions() -> None:
    class CollidingAliasArguments(BaseModel):
        model_config = ConfigDict(json_schema_mode_override="serialization")

        first: str = Field(validation_alias="x", serialization_alias="y")
        second: int = Field(validation_alias="y", serialization_alias="z")

    schema = CollidingAliasArguments.model_json_schema()
    assert list(schema["properties"]) == ["y", "z"]
    assert CollidingAliasArguments.model_validate({"x": "a", "y": 2}).second == 2

    message = _message_for(
        CollidingAliasArguments, {}, tool_name="collide_tool", input_model=CollidingAliasArguments, schema=schema
    )
    assert message.count("missing") == 2
    assert "missing 'x'" in message
    assert "missing 'y'" in message
    assert "'z'" not in message
    assert "Expected: {x: string, y: integer}" in message


def test_argument_guidance_reconciles_field_name_error_locations() -> None:
    class LocByFieldNameArguments(BaseModel):
        model_config = ConfigDict(loc_by_alias=False)

        path: str = Field(alias="file_path")

    message = _message_for(LocByFieldNameArguments, {}, tool_name="loc_tool", input_model=LocByFieldNameArguments)
    assert message.count("missing") == 1
    assert "missing 'file_path'" in message

    arguments = {"file_path": 7}
    message = _message_for(
        LocByFieldNameArguments, arguments, tool_name="loc_tool", input_model=LocByFieldNameArguments
    )
    assert "file_path: expected string, got integer" in message
    assert " path:" not in message
    assert "missing" not in message


def test_argument_guidance_resolves_field_name_loc_collisions() -> None:
    class CollidingLocArguments(BaseModel):
        model_config = ConfigDict(loc_by_alias=False)

        first: str = Field(alias="x")
        second: int = Field(alias="first")

    arguments = {"x": 7, "first": 2}
    message = _message_for(
        CollidingLocArguments, arguments, tool_name="collide_loc_tool", input_model=CollidingLocArguments
    )
    assert "x: expected string, got integer" in message
    assert "first: expected" not in message
    assert "Expected: {x: string, first: integer}" in message


def test_argument_guidance_top_level_alias_map_stays_out_of_nested_models() -> None:
    class ChildArguments(BaseModel):
        model_config = ConfigDict(loc_by_alias=False)

        first: str = Field(alias="y")
        x: int = 0

    class OuterArguments(BaseModel):
        model_config = ConfigDict(loc_by_alias=False)

        first: str = Field(alias="x")
        child: ChildArguments

    arguments = {"x": "ok", "child": {"y": 7, "x": 1}}
    message = _message_for(OuterArguments, arguments, tool_name="outer_tool", input_model=OuterArguments)
    # The nested loc ``child.first`` must degrade to the containing node, not
    # resolve through the top-level ``first → x`` map to the sibling ``x``.
    assert "child.first: expected {y: string" in message
    assert "child.x" not in message


def test_argument_guidance_uses_accepted_spellings_when_schema_names_diverge() -> None:
    class AliasDisabled(BaseModel):
        model_config = ConfigDict(validate_by_alias=False, validate_by_name=True)

        path: str = Field(alias="file_path")

    empty_message = _message_for(AliasDisabled, {}, tool_name="alias_tool", input_model=AliasDisabled)
    assert empty_message.count("missing") == 1
    assert "missing 'path'" in empty_message
    assert "file_path" not in empty_message
    assert "Expected: {path: string}" in empty_message

    bad_message = _message_for(AliasDisabled, {"path": 7}, tool_name="alias_tool", input_model=AliasDisabled)
    assert "missing" not in bad_message
    assert "path: expected string, got integer" in bad_message
    assert "Expected: {path: string}" in bad_message


def test_argument_guidance_alias_path_fields_are_not_reported_missing() -> None:
    class PathArguments(BaseModel):
        path: str = Field(validation_alias=AliasPath("payload", "path"))
        count: int

    arguments = {"payload": {"path": "x"}, "count": "bad"}
    message = _message_for(PathArguments, arguments, tool_name="path_tool", input_model=PathArguments)
    assert "missing" not in message
    assert "count: expected integer" in message


def test_argument_guidance_resolves_alias_locations_and_missing_fields() -> None:
    class AliasArguments(BaseModel):
        model_config = ConfigDict(validate_by_name=True)

        path: str = Field(alias="file_path")

    arguments = {"path": 7}
    message = _message_for(AliasArguments, arguments, tool_name="alias_tool", input_model=AliasArguments)
    assert "missing" not in message
    assert "path: expected string, got integer" in message
    assert "Expected: {file_path: string}" in message


def test_argument_guidance_reports_non_string_argument_keys() -> None:
    class KeyedArguments(BaseModel):
        a: int

    schema = KeyedArguments.model_json_schema()
    # Mapping-form calls from custom clients are not runtime-checked, so a
    # non-string key must become an argument error, not a crash inside
    # sorted() or the close-match suggester.
    assert _unexpected_argument_names({1: 2}, schema, reject_unexpected=True, input_model=KeyedArguments) == ["1"]
    message = _argument_validation_message(
        tool_name="keyed_tool",
        arguments={"a": 1, 2: 3},
        arguments_unparseable=False,
        schema=schema,
        exception=TypeError("unexpected keyword argument"),
        reject_unexpected=True,
        input_model=KeyedArguments,
    )
    assert 'unknown "2"' in message

    class ExplodingKey:
        def __hash__(self) -> int:
            return 1

        def __eq__(self, other: object) -> bool:
            return self is other

        def __repr__(self) -> str:
            raise RuntimeError("repr must never run")

    secret = "super-secret-payload"
    # Arbitrary key types must not run user __repr__ (it may raise past the
    # loop's TypeError|ValidationError handler or embed value content like a
    # tuple key); they degrade to a type placeholder instead.
    message = _argument_validation_message(
        tool_name="keyed_tool",
        arguments={ExplodingKey(): 1, (secret,): 2, 10**9000: 3},
        arguments_unparseable=False,
        schema=schema,
        exception=TypeError("unexpected keyword argument"),
        reject_unexpected=True,
        input_model=KeyedArguments,
    )
    assert secret not in message
    assert "ExplodingKey key" in message
    assert "tuple key" in message
    assert "int key" in message


def test_argument_guidance_never_runs_user_code_while_formatting() -> None:
    class KeyedArguments(BaseModel):
        a: int

    schema = KeyedArguments.model_json_schema()

    class RaisingMeta(type):
        def __getattribute__(cls, item: str) -> object:
            if item == "__name__":
                raise RuntimeError("metaclass hook must never run")
            return type.__getattribute__(cls, item)

    hooked_key = RaisingMeta("Hooked", (), {})()
    secret_class_key = type("super-secret-class", (), {})()

    class HostileStr(str):
        def split(self, *args: object, **kwargs: object) -> list[str]:
            raise RuntimeError("split must never run")

        def __eq__(self, other: object) -> bool:
            raise RuntimeError("__eq__ must never run")

        def __lt__(self, other: object) -> bool:
            raise RuntimeError("__lt__ must never run")

        __hash__ = str.__hash__

    # Key rendering must not dispatch through a metaclass __getattribute__
    # (type's own descriptor still recovers the real name), must not leak a
    # payload-controlled dynamic class name, and must report a str-subclass
    # key via a base-slot exact-str copy instead of running its methods in
    # sorted()/membership/close-match code.
    message = _argument_validation_message(
        tool_name="keyed_tool",
        arguments={hooked_key: 1, secret_class_key: 2, HostileStr("bogus_name"): 3},
        arguments_unparseable=False,
        schema=schema,
        exception=TypeError("unexpected keyword argument"),
        reject_unexpected=True,
        input_model=KeyedArguments,
    )
    assert 'unknown "<Hooked key>"' in message
    assert "super-secret-class" not in message
    assert 'unknown "<object key>"' in message
    assert "unknown 'bogus_name'" in message

    # The exact-int display bound must hold even with the interpreter's
    # int→str digit limit disabled (sys.set_int_max_str_digits(0)).
    huge = 10**5000
    limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        message = _argument_validation_message(
            tool_name="keyed_tool",
            arguments={huge: 1},
            arguments_unparseable=False,
            schema=schema,
            exception=TypeError("unexpected keyword argument"),
            reject_unexpected=True,
            input_model=KeyedArguments,
        )
    finally:
        sys.set_int_max_str_digits(limit)
    assert 'unknown "<int key>"' in message
    assert "10000" not in message

    # A *_type error's "got <kind>" for a non-JSON value goes through the
    # same metaclass-safe name path.
    weird_value = RaisingMeta("Odd", (), {})()
    with pytest.raises(ValidationError) as caught:
        KeyedArguments.model_validate({"a": weird_value})
    message = _argument_validation_message(
        tool_name="keyed_tool",
        arguments={"a": weird_value},
        arguments_unparseable=False,
        schema=schema,
        exception=caught.value,
        reject_unexpected=True,
        input_model=KeyedArguments,
    )
    assert "a: expected integer, got Odd" in message

    # A hostile container raising mid payload-scan must degrade the echo
    # guard (suppressing detail), never abort the turn.
    class VolatileMapping(dict):
        def values(self):  # type: ignore[override]
            raise RuntimeError("scan hook must never abort the turn")

    message = _argument_validation_message(
        tool_name="keyed_tool",
        arguments={"a": VolatileMapping()},
        arguments_unparseable=False,
        schema=schema,
        exception=caught.value,
        reject_unexpected=True,
        input_model=KeyedArguments,
    )
    assert "Invalid arguments for 'keyed_tool'" in message

    # A container that yields forever must exhaust the scan's item budget,
    # not hang the turn; the truncated scan then suppresses detail rather
    # than vouching for it.
    class EndlessList(list):
        def __iter__(self):  # type: ignore[override]
            while True:
                yield "spin"

    message = cpu_bounded(
        lambda: _argument_validation_message(
            tool_name="keyed_tool",
            arguments={"a": EndlessList()},
            arguments_unparseable=False,
            schema=schema,
            exception=caught.value,
            reject_unexpected=True,
            input_model=KeyedArguments,
        )
    )
    assert "Invalid arguments for 'keyed_tool'" in message

    # enum/const comparisons in the TypeError fallback must not dispatch to a
    # value's comparison hooks; the base-slot exact-str copy still detects
    # the mismatch.
    class LiteralArguments(BaseModel):
        mode: Literal["safe"]

    class HostileEq(str):
        __hash__ = str.__hash__

        def __eq__(self, other: object) -> bool:
            raise RuntimeError("__eq__ must never run")

        def __ne__(self, other: object) -> bool:
            raise RuntimeError("__ne__ must never run")

    message = _argument_validation_message(
        tool_name="literal_tool",
        arguments={"mode": HostileEq("bad"), "unknown": 1},
        arguments_unparseable=False,
        schema=LiteralArguments.model_json_schema(),
        exception=TypeError("unexpected keyword argument"),
        reject_unexpected=True,
        input_model=LiteralArguments,
    )
    assert "unknown 'unknown'" in message
    assert 'mode: expected "safe", got string' in message

    # A str-subclass tool name must not dispatch split() to user code.
    class SplitBomb(str):
        def split(self, *args: object, **kwargs: object) -> list[str]:
            raise RuntimeError("split must never run")

    message = _argument_validation_message(
        tool_name=SplitBomb("mytool"),
        arguments={"a": 1},
        arguments_unparseable=False,
        schema=schema,
        exception=caught.value,
        reject_unexpected=True,
        input_model=KeyedArguments,
    )
    assert "Invalid arguments for 'mytool'" in message


def test_argument_guidance_echo_guard_bounds_payload_scan_depth_and_size() -> None:
    class DeepArguments(BaseModel):
        payload: object

        @field_validator("payload")
        @classmethod
        def _reject(cls, value: object) -> object:
            raise PydanticCustomError("int_parsing", "rejected secret round21-secret-payload")

    deep_schema = DeepArguments.model_json_schema()

    # A payload string buried past the scan's depth cap is unverifiable, so a
    # spoofed allowlisted error code must not echo it as detail text.
    secret = "round21-secret-payload"
    deep: object = secret
    for _ in range(8):
        deep = [deep]
    with pytest.raises(ValidationError) as caught:
        DeepArguments.model_validate({"payload": deep})
    message = _argument_validation_message(
        tool_name="deep_tool",
        arguments={"payload": deep},
        arguments_unparseable=False,
        schema=deep_schema,
        exception=caught.value,
        reject_unexpected=True,
        input_model=DeepArguments,
    )
    assert secret not in message

    # A shallow payload keeps genuine allowlisted detail text.
    class ShallowArguments(BaseModel):
        payload: object

        @field_validator("payload")
        @classmethod
        def _reject(cls, value: object) -> object:
            raise PydanticCustomError("int_parsing", "should be a plain number")

    message = _message_for(ShallowArguments, {"payload": "x"}, tool_name="shallow_tool", input_model=ShallowArguments)
    assert "should be a plain number" in message

    # Repeated references to one large str-subclass object are copied once
    # (identity memo) and large strings charge the scan budget by size.
    class FatStr(str):
        pass

    fat = FatStr("x" * 2_000_000)
    cpu_bounded(
        lambda: _argument_validation_message(
            tool_name="deep_tool",
            arguments={"payload": [fat] * 1000},
            arguments_unparseable=False,
            schema=deep_schema,
            exception=caught.value,
            reject_unexpected=True,
            input_model=DeepArguments,
        )
    )


def test_argument_guidance_bounds_enum_comparison_and_rendering_work() -> None:
    # Shared references amplify a small graph into exponentially many node
    # visits; the exact-tree walk's TOTAL budget must keep enum comparison
    # bounded.
    class LiteralArguments(BaseModel):
        mode: Literal["safe"]

    dag: object = 0
    for _ in range(3):
        dag = [dag] * 256
    cpu_bounded(
        lambda: _argument_validation_message(
            tool_name="literal_tool",
            arguments={"mode": dag},
            arguments_unparseable=False,
            schema=LiteralArguments.model_json_schema(),
            exception=TypeError("unexpected keyword argument"),
            reject_unexpected=True,
            input_model=LiteralArguments,
        )
    )

    # A pathologically wide enum renders through the value cap (the join
    # still reaches the schema-text bound and truncates identically).
    wide_schema = {
        "type": "object",
        "properties": {"mode": {"enum": [f"v{index}" for index in range(10_000)]}},
        "required": ["mode"],
        "additionalProperties": False,
    }
    message = cpu_bounded(
        lambda: _argument_validation_message(
            tool_name="wide_enum_tool",
            arguments={"mode": "nope"},
            arguments_unparseable=False,
            schema=wide_schema,
            exception=TypeError("unexpected keyword argument"),
            reject_unexpected=True,
            input_model=None,
        )
    )
    assert "…" in message


def test_argument_guidance_echo_guard_covers_keys_sets_and_hostile_types() -> None:
    # Caller strings hide in mapping KEYS and set members too; both must be
    # covered by the echo guard's containment check.
    class KeyEcho(BaseModel):
        payload: object

        @field_validator("payload")
        @classmethod
        def _reject(cls, value: object) -> object:
            raise PydanticCustomError("int_parsing", "rejected round22-secret-key")

    with pytest.raises(ValidationError) as key_caught:
        KeyEcho.model_validate({"payload": 1})
    for payload in ({"round22-secret-key": 1}, {"round22-secret-key", "x"}):
        message = _argument_validation_message(
            tool_name="key_tool",
            arguments={"payload": payload},
            arguments_unparseable=False,
            schema=KeyEcho.model_json_schema(),
            exception=key_caught.value,
            reject_unexpected=True,
            input_model=KeyEcho,
        )
        assert "round22-secret-key" not in message

    # A container subclass is never iterated at all — its hooks are user
    # code that can raise or block before the first yield; the scan marks it
    # unverified instead.
    class NeverIter(list):
        def __iter__(self):  # type: ignore[override]
            raise AssertionError("scan must never iterate a container subclass")

    message = _argument_validation_message(
        tool_name="key_tool",
        arguments={"payload": NeverIter(["x"])},
        arguments_unparseable=False,
        schema=KeyEcho.model_json_schema(),
        exception=key_caught.value,
        reject_unexpected=True,
        input_model=KeyEcho,
    )
    assert "Invalid arguments for 'key_tool'" in message

    # Exactness checks must compare types by IDENTITY: `type(x) in (...)`
    # dispatches to a caller-controlled metaclass __eq__, which could raise
    # mid-formatting or lie its way past the scan (marking a foreign object
    # "exact" so smuggled strings go unscanned yet report complete).
    class RaisingEqMeta(type):
        def __eq__(cls, other: object) -> bool:
            raise RuntimeError("metaclass __eq__ must never run")

        __hash__ = type.__hash__

    hostile = RaisingEqMeta("Hostile", (), {})()
    message = _argument_validation_message(
        tool_name="key_tool",
        arguments={"payload": hostile, hostile: 2},
        arguments_unparseable=False,
        schema=KeyEcho.model_json_schema(),
        exception=key_caught.value,
        reject_unexpected=True,
        input_model=KeyEcho,
    )
    assert 'unknown "<Hostile key>"' in message

    class LyingEqMeta(type):
        def __eq__(cls, other: object) -> bool:
            return True

        __hash__ = type.__hash__

    class MetaEcho(BaseModel):
        payload: object

        @field_validator("payload")
        @classmethod
        def _reject(cls, value: object) -> object:
            raise PydanticCustomError("int_parsing", "leak round23-meta-secret")

    with pytest.raises(ValidationError) as meta_caught:
        MetaEcho.model_validate({"payload": 1})
    message = _argument_validation_message(
        tool_name="meta_tool",
        arguments={"payload": LyingEqMeta("Liar", (), {})()},
        arguments_unparseable=False,
        schema=MetaEcho.model_json_schema(),
        exception=meta_caught.value,
        reject_unexpected=True,
        input_model=MetaEcho,
    )
    assert "round23-meta-secret" not in message


def test_argument_guidance_bounds_unknown_name_collection() -> None:
    # Vast unknown-key sets stay bounded: only the displayed issues get
    # close-match suggestions and the name list is capped (the trailing
    # count degrades, label-only).
    class KeyedArguments(BaseModel):
        a: int

    huge_arguments: dict[str, object] = {f"k{index}": 1 for index in range(20_000)}
    huge_arguments["a"] = 1
    message = cpu_bounded(
        lambda: _argument_validation_message(
            tool_name="keyed_tool",
            arguments=huge_arguments,
            arguments_unparseable=False,
            schema=KeyedArguments.model_json_schema(),
            exception=TypeError("unexpected keyword argument"),
            reject_unexpected=True,
            input_model=KeyedArguments,
        )
    )
    assert "unknown 'k0'" in message

    # Collection itself stops at the cap — the walk is bounded at
    # cap + |accepted| iterations, so growing the mapping past the cap adds
    # no per-name work (round 24: 4M keys took 0.79s when only the DISPLAY
    # was capped).
    capped_arguments: dict[str, object] = {f"k{index}": 1 for index in range(10_000)}
    capped_arguments["a"] = 1
    capped_names = _unexpected_argument_names(
        capped_arguments,
        KeyedArguments.model_json_schema(),
        reject_unexpected=True,
        input_model=KeyedArguments,
    )
    assert len(capped_names) == _MAX_UNEXPECTED_NAMES
    assert capped_names == sorted(capped_names)

    class Collider(str):
        __hash__ = object.__hash__

        def __eq__(self, other: object) -> bool:
            return self is other

        __ne__ = object.__ne__

    # A flood of distinct str-subclass keys that all normalize to an accepted
    # name never fills the unknown list, so the scan needs its own visit cap;
    # on exhaustion the pre-check stands down (Pydantic still validates the
    # full mapping) instead of walking millions of keys (round 25).
    flood: dict[object, object] = {Collider("a"): 1 for _ in range(120_000)}
    flood_names = cpu_bounded(
        lambda: _unexpected_argument_names(
            flood,
            KeyedArguments.model_json_schema(),
            reject_unexpected=True,
            input_model=KeyedArguments,
        )
    )
    assert flood_names == []
    # Below the visit cap the same shape still reports its unknown key.
    small_flood: dict[object, object] = {Collider("a"): 1 for _ in range(1_000)}
    small_flood["zzz_unknown"] = 1
    assert _unexpected_argument_names(
        small_flood,
        KeyedArguments.model_json_schema(),
        reject_unexpected=True,
        input_model=KeyedArguments,
    ) == ["zzz_unknown"]


def test_argument_guidance_display_safe_copy_flags_incomplete_scans() -> None:
    # The display-safe copy is entry-capped and reports incompleteness; a
    # string living in a dropped entry was never scanned, so a spoofed
    # allowlisted detail code must not be able to echo it.
    class CapMarker(str):
        pass

    over_cap: dict[object, object] = {CapMarker("payload"): 1}
    for index in range(_MAX_SAFE_ARGUMENT_ENTRIES):
        over_cap[f"f{index}"] = 1
    over_cap["tail"] = "round24-cap-secret"
    safe_copy, copy_complete = _display_safe_arguments(over_cap)
    assert not copy_complete
    assert len(safe_copy) == _MAX_SAFE_ARGUMENT_ENTRIES
    assert all(type(key) is str for key in safe_copy)

    class CapEcho(BaseModel):
        payload: object

        @field_validator("payload")
        @classmethod
        def _reject(cls, value: object) -> object:
            raise PydanticCustomError("int_parsing", "leak round24-cap-secret")

    with pytest.raises(ValidationError) as cap_caught:
        CapEcho.model_validate({"payload": 1})
    message = cpu_bounded(
        lambda: _argument_validation_message(
            tool_name="cap_tool",
            arguments=over_cap,
            arguments_unparseable=False,
            schema=CapEcho.model_json_schema(),
            exception=cap_caught.value,
            reject_unexpected=True,
            input_model=CapEcho,
        )
    )
    assert "round24-cap-secret" not in message
    assert "Invalid arguments for 'cap_tool'" in message
    # An uncapped mapping keeps the identity fast path and full detail flow.
    small_safe, small_complete = _display_safe_arguments({"payload": 7})
    assert small_complete and small_safe == {"payload": 7}
    detail_message = _argument_validation_message(
        tool_name="cap_tool",
        arguments={"payload": 7},
        arguments_unparseable=False,
        schema=CapEcho.model_json_schema(),
        exception=cap_caught.value,
        reject_unexpected=True,
        input_model=CapEcho,
    )
    assert "leak round24-cap-secret" in detail_message

    # Key normalization can collapse two DISTINCT caller keys (a str subclass
    # with identity hash plus the exact string); the dropped entry's value was
    # never scanned, so the copy must flag incomplete and the echo guard must
    # suppress spoofed detail (round 25).
    class Collider(str):
        __hash__ = object.__hash__

        def __eq__(self, other: object) -> bool:
            return self is other

        __ne__ = object.__ne__

    class CollisionEcho(BaseModel):
        payload: object

        @field_validator("payload")
        @classmethod
        def _reject(cls, value: object) -> object:
            raise PydanticCustomError("int_parsing", "leak round25-collision-secret")

    with pytest.raises(ValidationError) as collision_caught:
        CollisionEcho.model_validate({"payload": 1})
    colliding: dict[object, object] = {
        Collider("payload"): 1,
        "payload": "round25-collision-secret",
    }
    collided_safe, collided_complete = _display_safe_arguments(colliding)
    assert not collided_complete
    assert collided_safe == {"payload": 1}
    message = _argument_validation_message(
        tool_name="cap_tool",
        arguments=colliding,
        arguments_unparseable=False,
        schema=CollisionEcho.model_json_schema(),
        exception=collision_caught.value,
        reject_unexpected=True,
        input_model=CollisionEcho,
    )
    assert "round25-collision-secret" not in message
    assert "Invalid arguments for 'cap_tool'" in message


def test_argument_guidance_bounds_giant_loc_parts_and_vast_error_counts() -> None:
    # A giant mapping KEY lands verbatim in every error's loc tuple; the
    # marker-analysis containment scans, identifier check, JSON quoting, and
    # path join must all length-gate before running over it, or a wide union
    # multiplies an O(len) pass per error (round 26: 1.05s -> ~0.2s).
    wide_variants: object = None
    for index in range(60):
        variant = type(f"LocV{index}", (BaseModel,), {"__annotations__": {"x": int}})
        wide_variants = variant if wide_variants is None else wide_variants | variant

    class GiantKeyed(BaseModel):
        data: dict[str, wide_variants]  # type: ignore[valid-type]

    giant_key = "k" * 1_000_000
    giant_arguments = {"data": {giant_key: {}}}
    with pytest.raises(ValidationError) as giant_caught:
        GiantKeyed.model_validate(giant_arguments)
    message = cpu_bounded(
        lambda: _argument_validation_message(
            tool_name="giant_tool",
            arguments=giant_arguments,
            arguments_unparseable=False,
            schema=GiantKeyed.model_json_schema(),
            exception=giant_caught.value,
            reject_unexpected=True,
            input_model=GiantKeyed,
        )
    )
    assert giant_key[:200] not in message

    # Pydantic materializes a DISTINCT copy of a giant mapping key per error,
    # so per-object hash caching never amortizes: loc parts must be clamped
    # at materialization, before any schema-map lookup hashes them (round 27:
    # 50 refs to one 25M-char extra key, 0.54s -> ~0.3s, dominated by
    # pydantic's own errors() copies).
    class ForbidItem(BaseModel):
        model_config = ConfigDict(extra="forbid")
        x: int

    class ForbidItems(BaseModel):
        items: list[ForbidItem | None]

    shared_extra = {"x": 1, giant_key + "x": 2}
    forbid_arguments = {"items": [shared_extra] * 50}
    with pytest.raises(ValidationError) as forbid_caught:
        ForbidItems.model_validate(forbid_arguments)
    # Structural pin (wall-clock here is dominated by pydantic's own per-error
    # key copies, which flake on slow CI): every loc string part comes out of
    # materialization clamped to one char PAST the marker gate — bounded for
    # hashing, still classified oversized by every marker-analysis gate.
    for capped_error in _capped_validation_errors(forbid_caught.value):
        capped_location = capped_error.get("loc")
        assert isinstance(capped_location, tuple)
        assert all(len(part) <= _MAX_MARKER_SCAN_CHARS + 1 for part in capped_location if isinstance(part, str))
    message = cpu_bounded(
        lambda: _argument_validation_message(
            tool_name="forbid_tool",
            arguments=forbid_arguments,
            arguments_unparseable=False,
            schema=ForbidItems.model_json_schema(),
            exception=forbid_caught.value,
            reject_unexpected=True,
            input_model=ForbidItems,
        ),
        2 * CPU_TIME_BOUND_SECONDS,
    )
    assert giant_key[:200] not in message

    # A ValidationError with a vast error count is never materialized; the
    # message degrades to schema/kind comparison instead of allocating every
    # error dict.
    class ItemsArguments(BaseModel):
        items: list[int]

    bad_items = ["x"] * 60_000
    with pytest.raises(ValidationError) as wide_caught:
        ItemsArguments.model_validate({"items": bad_items})
    message = cpu_bounded(
        lambda: _argument_validation_message(
            tool_name="items_tool",
            arguments={"items": bad_items},
            arguments_unparseable=False,
            schema=ItemsArguments.model_json_schema(),
            exception=wide_caught.value,
            reject_unexpected=True,
            input_model=ItemsArguments,
        )
    )
    assert "Invalid arguments for 'items_tool'" in message


def test_argument_guidance_error_cap_keeps_breadth_across_fields() -> None:
    wide_union: object = None
    for index in range(60):
        model = type(f"Variant{index}", (BaseModel,), {"__annotations__": {"x": int}})
        wide_union = model if wide_union is None else wide_union | model

    class CappedArguments(BaseModel):
        wide: wide_union  # type: ignore[valid-type]
        target: int

    arguments = {"wide": {"x": "not-int"}, "target": "not-int"}
    schema = CappedArguments.model_json_schema()
    with pytest.raises(ValidationError) as caught:
        CappedArguments.model_validate(arguments)
    # Pydantic lists all 60 per-variant errors for 'wide' before the single
    # 'target' error. The cap selects breadth-first across top-level fields,
    # so 'target' must survive instead of losing every slot to duplicates.
    assert len(caught.value.errors()) > _MAX_VALIDATION_ERRORS
    message = _argument_validation_message(
        tool_name="capped_tool",
        arguments=arguments,
        arguments_unparseable=False,
        schema=schema,
        exception=caught.value,
        reject_unexpected=True,
        input_model=CappedArguments,
    )
    assert "wide.x: expected integer" in message
    assert "target: expected integer" in message


def test_argument_guidance_escapes_surrogates_and_control_characters() -> None:
    schema = {"type": "object", "properties": {"path": {"type": "string"}}, "additionalProperties": False}

    message = _argument_validation_message(
        tool_name="probe",
        arguments={"bad\udcffname": 1, "ansi\x9cname": 2},
        arguments_unparseable=False,
        schema=schema,
        exception=TypeError("bad"),
        reject_unexpected=True,
    )

    message.encode("utf-8")
    assert "\udcff" not in message
    assert "\\udcff" in message
    assert "\x9c" not in message
    assert "\\x9c" in message


def test_argument_guidance_sanitizes_names_and_bounds_large_messages() -> None:
    properties = {f"parameter_{index}": {"enum": [f"value_{index}_" + "x" * 100]} for index in range(20)}
    schema = {"type": "object", "properties": properties, "additionalProperties": False}

    message = _argument_validation_message(
        tool_name="unsafe\ntool",
        arguments={f"unknown_{index}\nname": index for index in range(10)},
        arguments_unparseable=False,
        schema=schema,
        exception=TypeError("bad"),
        reject_unexpected=True,
    )

    assert "\n" not in message
    assert 'unknown "unknown_0 name"' in message
    assert "… and 7 more issue(s)" in message
    assert len(message) <= 1200
