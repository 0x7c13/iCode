# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Provider-independent search inputs and results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchRequest:
    query: str
    allowed_domains: tuple[str, ...]
    blocked_domains: tuple[str, ...]
    limit: int


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class SearchResponse:
    provider_id: str
    hits: tuple[SearchHit, ...]
