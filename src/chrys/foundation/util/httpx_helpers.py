# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared helpers for constructing ``httpx`` clients."""

from __future__ import annotations

BYPASS_PROXY_MOUNTS: dict[str, None] = {
    "http://": None,
    "https://": None,
    "all://": None,
}
"""Mounts that force httpx to use its direct transport instead of env proxies."""
