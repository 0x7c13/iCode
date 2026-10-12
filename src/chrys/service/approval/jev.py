# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Translate Formal predicate requests to Jev using the existing model-profile transport."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import httpx
from openai import AsyncOpenAI

from chrys.kernel import ChatResponse, Message, UsageDetails
from chrys.service.approval.predicate import PredicateAsset, TruthValue, reject_duplicate_object
from chrys.service.profiles.models.schema import ModelProfile


def is_jev_profile(profile: ModelProfile) -> bool:
    """Jev uses an ordinary OpenAI profile selected for the approval-judge role."""
    return profile.provider == "openai" and profile.model_id.startswith(("typesafe/jev-", "~typesafe/jev-"))


class JevPredicateClient:
    """Wire adapter only: the Judge still owns validation, retries, audit and policy."""

    def __init__(self, client: AsyncOpenAI, model_id: str, asset: PredicateAsset) -> None:
        self.client = client
        self.model_id = model_id
        self.asset = asset

    async def get_response(
        self, messages: Sequence[Message], *, stream: bool = False, options: Mapping[str, Any] | None = None
    ) -> ChatResponse:
        # Decisions has no streaming or chat sampling options. The shared SDK
        # already owns profile credentials, headers, proxy/TLS, timeouts and retries.
        if stream:
            raise ValueError("Jev predicate requests do not stream")
        response = await self.client.post(
            "decisions",
            cast_to=httpx.Response,
            body={
                "model": self.model_id,
                "state": {"messages": [{"role": m.role, "content": m.text} for m in messages]},
                "questions": {
                    p.id: {
                        "type": "choice",
                        "instructions": asdict(p),
                        "criteria": {
                            "true": "The condition holds.",
                            "false": "The condition does not hold.",
                            "unknown": "The available input is insufficient to determine whether the condition holds.",
                        },
                    }
                    for p in self.asset.principles
                },
            },
        )
        # Decode before the SDK collapses duplicate answer IDs into a dict.
        try:
            data = response.json(object_pairs_hook=reject_duplicate_object)
        except ValueError:
            data = {}
        values = self._values(data)
        usage = data.get("usage") if isinstance(data, dict) else None
        tokens: UsageDetails = {}
        total = 0
        if isinstance(usage, dict):
            for source, target in (("input_tokens", "input_token_count"), ("output_tokens", "output_token_count")):
                value = usage.get(source)
                if type(value) is int and value >= 0:
                    tokens[target] = value
                    total += value
            if len(tokens) == 2:
                tokens["total_token_count"] = total
        # Invalid answers become an invalid predicate object so the existing
        # response-validation retry lane runs, retaining usage for that attempt.
        return ChatResponse(
            messages=[Message("assistant", [json.dumps(values)])],
            usage_details=tokens or None,
            model=self.model_id,
        )

    def _values(self, data: Any) -> dict[str, TruthValue]:
        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, dict) or set(answers) != {p.id for p in self.asset.principles}:
            return {}
        values: dict[str, TruthValue] = {}
        for p in self.asset.principles:
            answer = answers[p.id]
            if not isinstance(answer, dict) or answer.get("type") != "choice":
                return {}
            choice = answer.get("choice")
            if choice not in ("true", "false", "unknown"):
                return {}
            values[p.id] = "unknown" if choice == "unknown" else choice == "true"
        return values
