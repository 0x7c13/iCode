# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pinned message ids for web search/fetch and the tool sections it added."""

from __future__ import annotations

WEB_MESSAGE_IDS = frozenset(
    {
        "settings.tools.web_egress.proxy_url.label",
        "settings.tools.web_egress.proxy_dns.label",
        "settings.tools.web_fetch.allowed_origins.label",
        "settings.tools.web_fetch.denied_origins.label",
        "settings.tools.web_fetch.http_origins.label",
        "settings.tools.web_fetch.max_tokens.label",
        "settings.tools.web_fetch.mode.label",
        "settings.tools.web_fetch.private_origins.label",
        "settings.tools.web_fetch.timeout_seconds.label",
        "settings.tools.web_search.custom_endpoints.label",
        "settings.tools.web_search.http_origins.label",
        "settings.tools.web_search.mode.label",
        "settings.tools.web_search.num_results.label",
        "settings.tools.web_search.private_origins.label",
        "settings.tools.web_search.timeout_seconds.label",
        "tui.agent_config.tools.web_search",
        "tui.agent_config.tools.web_search_description",
        "tui.agent_config.tools.web_fetch",
        "tui.agent_config.tools.web_fetch_description",
        "builder.hosted_web_tools_preferred",
        "builder.web_tools_unavailable",
        "tui.main.hosted_web_tools.ok",
        "tui.settings.hint.tools.web_egress.proxy_url",
        "tui.settings.hint.tools.web_egress.proxy_dns",
        "tui.settings.hint.tools.web_fetch.mode",
        "tui.settings.hint.tools.web_search.custom_endpoints",
        "tui.settings.hint.tools.web_search.http_origins",
        "tui.settings.hint.tools.web_search.mode",
        "tui.settings.hint.tools.web_search.num_results",
        "tui.settings.hint.tools.web_search.private_origins",
        "tui.settings.section.ask_user",
        "tui.settings.section.file_search",
        "tui.settings.section.web",
        "settings.rejected.json_array",
        "settings.rejected.web_origin",
        "settings.rejected.endpoint_grant",
        "tui.settings.error.expected_json_array",
        "tui.settings.error.expected_web_origin",
        "tui.settings.error.expected_endpoint_grant",
    }
)
