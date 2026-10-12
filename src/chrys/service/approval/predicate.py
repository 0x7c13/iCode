# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Strict predicate assets, three-valued responses, and Formal routing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from typing import Any, Literal

_DECISION_VERSION = "three-valued-or-v2"
_MAX_PRINCIPLES = 32
_MAX_TEXT_LENGTH = 4_000

type TruthValue = bool | Literal["unknown"]


class PredicateAssetError(ValueError):
    """The packaged predicate asset cannot safely be used."""


class PredicateDecision(StrEnum):
    ALLOW = "allow"
    REASONING = "reasoning"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True, slots=True)
class PrincipleExample:
    """Immutable example data; result labels remain strings from the asset."""

    prompt: str
    toolcall: str
    result: str
    reason: str


@dataclass(frozen=True, slots=True)
class Principle:
    """One independent semantic condition, including its complete definition."""

    id: str
    role: str
    question: str
    examples: tuple[PrincipleExample, ...]
    exceptions: tuple[PrincipleExample, ...]


@dataclass(frozen=True, slots=True)
class PredicateAsset:
    """Validated immutable predicate definition used for one evaluation."""

    asset_version: str
    decision_version: str
    project: str
    digest: str
    principles: tuple[Principle, ...]

    def __post_init__(self) -> None:
        validate_asset_snapshot(self)


@dataclass(frozen=True, slots=True)
class PredicateValue:
    """A validated true, false, or unknown predicate result."""

    id: str
    value: TruthValue


@dataclass(frozen=True, slots=True)
class PredicateEvaluation:
    """Result of strict parsing followed by the fixed routing decision."""

    decision: PredicateDecision
    reason: str
    values: tuple[PredicateValue, ...] = ()


def load_default_asset() -> PredicateAsset:
    """Load the packaged, read-only Formal asset."""
    raw = files("chrys.service.approval").joinpath("principles.json").read_bytes()
    return parse_asset(raw)


def parse_asset(raw: bytes | str) -> PredicateAsset:
    """Validate the complete array; no legacy schema or role classification."""
    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicate_object)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredicateAssetError("invalid predicate asset JSON") from exc
    if not isinstance(value, list) or not 1 <= len(value) <= _MAX_PRINCIPLES:
        raise PredicateAssetError("predicate asset needs principles")
    principles: list[Principle] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "role", "question", "examples", "exceptions"}:
            raise PredicateAssetError("invalid principle fields")
        groups: list[tuple[PrincipleExample, ...]] = []
        for key in ("examples", "exceptions"):
            examples = item[key]
            if not isinstance(examples, list) or any(
                not isinstance(example, dict) or set(example) != {"prompt", "toolcall", "result", "reason"}
                for example in examples
            ):
                raise PredicateAssetError("invalid principle examples")
            groups.append(tuple(PrincipleExample(**example) for example in examples))
        principles.append(Principle(item["id"], item["role"], item["question"], *groups))
    digest_source = raw.encode("utf-8") if isinstance(raw, str) else raw
    return PredicateAsset(
        asset_version="risk-predicates-v1",
        decision_version=_DECISION_VERSION,
        project="iCode",
        digest=asset_digest(digest_source),
        principles=tuple(principles),
    )


def validate_predicate_response(text: str, asset: PredicateAsset) -> tuple[PredicateValue, ...]:
    """Accept exactly one JSON boolean or literal \"unknown\" per principle."""
    lines = text.strip().splitlines()
    # Unwrap only one complete block; extra prose or nested blocks remain invalid.
    if len(lines) >= 3 and lines[0].strip().lower() in {"```", "```json"} and lines[-1].strip() == "```":
        text = "\n".join(lines[1:-1])
    try:
        data = json.loads(text, object_pairs_hook=reject_duplicate_object)
    except (json.JSONDecodeError, PredicateAssetError) as exc:
        raise PredicateAssetError("invalid predicate response JSON") from exc
    expected = {principle.id for principle in asset.principles}
    if not isinstance(data, dict) or set(data) != expected or any(not _valid_value(data[key]) for key in expected):
        raise PredicateAssetError('predicate response must contain true, false, or "unknown" per required ID')
    return tuple(PredicateValue(principle.id, data[principle.id]) for principle in asset.principles)


def decide_predicates(values: tuple[PredicateValue, ...], asset: PredicateAsset) -> PredicateEvaluation:
    """Any true requires review; all false allows; otherwise use reasoning."""
    if (
        len(values) != len(asset.principles)
        or {value.id for value in values} != {principle.id for principle in asset.principles}
        or any(not _valid_value(value.value) for value in values)
    ):
        raise PredicateAssetError("invalid predicate values")
    triggered = [value.id for value in values if value.value is True]
    if triggered:
        return PredicateEvaluation(
            PredicateDecision.NEEDS_REVIEW, f"Requires human review: {', '.join(triggered)}.", values
        )
    if all(value.value is False for value in values):
        return PredicateEvaluation(PredicateDecision.ALLOW, "All predicates are false; automatically approved.", values)
    return PredicateEvaluation(
        PredicateDecision.REASONING, "No predicate is true, but some are unknown; reasoning review required.", values
    )


def evaluate_predicate_response(text: str, asset: PredicateAsset) -> PredicateEvaluation:
    """Shared, side-effect-free online/offline predicate evaluation entry point."""
    return decide_predicates(validate_predicate_response(text, asset), asset)


def asset_digest(raw: bytes) -> str:
    """Return a stable audit identifier for a candidate or packaged asset."""
    return hashlib.sha256(raw).hexdigest()


def validate_asset_snapshot(asset: PredicateAsset) -> None:
    """Validate once when constructing the immutable asset."""
    if (
        asset.decision_version != _DECISION_VERSION
        or not _valid_text(asset.asset_version)
        or not _valid_text(asset.project)
        or not isinstance(asset.digest, str)
        or len(asset.digest) != 64
        or any(character not in "0123456789abcdef" for character in asset.digest)
        or not isinstance(asset.principles, tuple)
        or not 1 <= len(asset.principles) <= _MAX_PRINCIPLES
        or any(
            not isinstance(principle, Principle)
            or not all(_valid_text(text) for text in (principle.id, principle.role, principle.question))
            or not all(_valid_examples(group) for group in (principle.examples, principle.exceptions))
            for principle in asset.principles
        )
        or len({principle.id for principle in asset.principles}) != len(asset.principles)
    ):
        raise PredicateAssetError("invalid Formal asset snapshot")


def _valid_examples(examples: object) -> bool:
    return (
        isinstance(examples, tuple)
        and bool(examples)
        and all(
            isinstance(example, PrincipleExample)
            and all(_valid_text(text) for text in (example.prompt, example.toolcall, example.reason))
            and isinstance(example.result, str)
            and example.result in ("true", "false", "unknown")
            for example in examples
        )
    )


def _valid_value(value: object) -> bool:
    return type(value) is bool or (isinstance(value, str) and value == "unknown")


def reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PredicateAssetError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _valid_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= _MAX_TEXT_LENGTH
