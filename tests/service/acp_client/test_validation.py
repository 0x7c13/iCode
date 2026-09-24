# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Strict JSON framing, scalar validation, and immutable spec tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from acp.schema import Implementation

from chrys.service.acp_client import (
    AcpAgentSpec,
    encode_protocol_json,
    parse_protocol_json,
    validate_json_rpc_envelope,
    validate_json_scalar_tree,
)


def test_spec_and_sdk_implementation_are_keyword_only(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        AcpAgentSpec("agent", str(tmp_path), tmp_path / "stderr")  # type: ignore[misc]
    with pytest.raises(TypeError):
        Implementation("chrys", "1")  # type: ignore[misc]

    spec = AcpAgentSpec(command="agent", cwd=str(tmp_path), stderr_log_path=tmp_path / "stderr")

    assert spec.client_info.name == "chrys"
    assert spec.client_info.version


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
        {"outer": [1, {"nested": math.inf}]},
    ],
)
def test_scalar_validator_rejects_non_finite_values(value: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_json_scalar_tree(value)


def test_scalar_validator_rejects_unpaired_surrogates_in_values_and_keys() -> None:
    unpaired = chr(0xD800)

    with pytest.raises(ValueError, match="surrogate"):
        validate_json_scalar_tree({"value": [unpaired]})
    with pytest.raises(ValueError, match="surrogate"):
        validate_json_scalar_tree({unpaired: "value"})


def test_protocol_parser_rejects_constants_and_overflowed_float() -> None:
    with pytest.raises(ValueError):
        parse_protocol_json(b'{"value":NaN}')
    with pytest.raises(ValueError, match="finite"):
        parse_protocol_json(b'{"value":1e9999}')


def test_protocol_encoder_disallows_non_finite_and_preserves_unicode() -> None:
    payload = {"text": "Chrys 你好 🌼"}

    encoded = encode_protocol_json(payload)

    assert json.loads(encoded) == payload
    with pytest.raises(ValueError):
        encode_protocol_json({"value": math.nan})


def test_protocol_encoder_accepts_a_valid_surrogate_pair_but_not_an_unpaired_half() -> None:
    paired = chr(0xD83D) + chr(0xDE00)

    encoded = encode_protocol_json({"text": paired})

    assert json.loads(encoded)["text"] == "😀"
    with pytest.raises(ValueError, match="surrogate"):
        encode_protocol_json({"text": chr(0xD83D)})


@pytest.mark.parametrize(
    "message",
    [
        [],
        {},
        {"jsonrpc": "1.0", "id": 1, "result": None},
        {"jsonrpc": "2.0", "id": True, "result": None},
        {"jsonrpc": "2.0", "id": 1.5, "result": None},
        {"jsonrpc": "2.0", "method": 7},
        {"jsonrpc": "2.0", "method": "x", "params": []},
        {"jsonrpc": "2.0", "id": 1},
        {"jsonrpc": "2.0", "id": 1, "result": None, "params": {}},
        {"jsonrpc": "2.0", "id": 1, "result": None, "error": {"code": -1, "message": "bad"}},
        {"jsonrpc": "2.0", "id": 1, "error": None},
        {"jsonrpc": "2.0", "id": 1, "error": {"code": True, "message": "bad"}},
        {"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": 7}},
    ],
)
def test_json_rpc_envelope_rejects_every_illegal_shape(message: object) -> None:
    with pytest.raises(ValueError):
        validate_json_rpc_envelope(message)


def test_json_rpc_envelope_accepts_exact_legal_shapes() -> None:
    assert validate_json_rpc_envelope({"jsonrpc": "2.0", "method": "notice"}) == "notification"
    assert validate_json_rpc_envelope({"jsonrpc": "2.0", "id": "r1", "method": "request"}) == "request"
    assert validate_json_rpc_envelope({"jsonrpc": "2.0", "id": 1, "result": None}) == "response"
    assert (
        validate_json_rpc_envelope({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "missing"}})
        == "response"
    )
