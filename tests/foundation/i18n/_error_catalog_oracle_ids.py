# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pinned message ids for what provider and network errors, and the retries they start, mean to the user."""

from __future__ import annotations

ERROR_MESSAGE_IDS = frozenset(
    {
        "error.display.with_hint",
        "error.hint.maybe_offline",
        "error.kind.auth_failed",
        "error.kind.connect_timeout",
        "error.kind.connection_failed",
        "error.kind.connection_lost",
        "error.kind.connection_refused",
        "error.kind.content_filtered",
        "error.kind.context_overflow",
        "error.kind.context_overflow_config_mismatch",
        "error.kind.dns_failed",
        "error.kind.host_unreachable",
        "error.kind.invalid_endpoint",
        "error.kind.network_generic",
        "error.kind.no_route",
        "error.kind.overloaded",
        "error.kind.payload_too_large",
        "error.kind.proxy_auth_failed",
        "error.kind.proxy_rejected",
        "error.kind.proxy_unreachable",
        "error.kind.quota_exhausted",
        "error.kind.rate_limited",
        "error.kind.read_timeout",
        "error.kind.server_error",
        "error.kind.stream_truncated",
        "error.kind.tls_failed",
        "error.kind.via_proxy_failed",
        "error.kind.write_timeout",
        "retry.context_overflow",
        "retry.stream_stalled",
    }
)
