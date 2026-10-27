# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Runtime/profile display helpers shared by main-screen handlers."""

from __future__ import annotations

from chrys.app.tui.screens.main.ports import StatusMessage, StatusTrail
from chrys.app.tui.screens.main.state import MainScreenServices
from chrys.app.tui.widgets.chrome.status_bar import TrailCount
from chrys.foundation.events.types import AgentRuntimeDetails
from chrys.foundation.i18n import msg

_RUNTIME_TOOLS = msg(
    "tui.status.runtime_tools",
    fallback="{count} tool",
    plural_fallback="{count} tools",
)
_RUNTIME_SKILLS = msg(
    "tui.status.runtime_skills",
    fallback="{count} skill",
    plural_fallback="{count} skills",
)
_RUNTIME_HOOKS = msg(
    "tui.status.runtime_hooks",
    fallback="{count} hook",
    plural_fallback="{count} hooks",
)
_RUNTIME_FILES = msg(
    "tui.status.runtime_files",
    fallback="{count} file",
    plural_fallback="{count} files",
)
_RUNTIME_BACKGROUND = msg(
    "tui.status.runtime_background",
    fallback="{count} background task",
    plural_fallback="{count} background tasks",
)
_RUNTIME_BACKGROUND_SHORT = msg(
    "tui.status.runtime_background_short",
    fallback="{count} task",
    plural_fallback="{count} tasks",
)


def with_background_count(trail: StatusTrail, running: int) -> StatusTrail:
    """*trail*, then how many background commands run, when any do."""
    if not running:
        return trail
    parts = trail if isinstance(trail, tuple) else ((trail,) if trail else ())
    return (
        *parts,
        TrailCount(_RUNTIME_BACKGROUND.bind(count=running), _RUNTIME_BACKGROUND_SHORT.bind(count=running)),
    )


def enabled_hook_count(details: AgentRuntimeDetails | None) -> int:
    """The hooks *details* names that are enabled: what the status trail and the Hooks tab count."""
    if details is None:
        return 0
    return sum(hook.enabled for source in details.hook_sources for hook in source.hooks)


class RegistryRuntimeInfoProvider:
    """Format runtime info and profile descriptions from configured registries."""

    def __init__(self, services: MainScreenServices) -> None:
        self._services = services

    def format_tool_info(
        self,
        tool_names: list[str],
        skill_names: list[str],
        *,
        memory_files: list[str] | None = None,
        runtime_details: AgentRuntimeDetails | None = None,
    ) -> StatusTrail:
        """Build trail text for tool/skill/hook/file counts."""
        parts: list[StatusMessage] = []
        # The Tools tab lists the context management tools too: the count takes them in.
        tools = dict.fromkeys([*tool_names, *(runtime_details.context_tools if runtime_details else ())])
        if tools:
            parts.append(_RUNTIME_TOOLS.bind(count=len(tools)))
        if skill_names:
            parts.append(_RUNTIME_SKILLS.bind(count=len(skill_names)))
        hook_count = enabled_hook_count(runtime_details)
        if hook_count:
            parts.append(_RUNTIME_HOOKS.bind(count=hook_count))
        if memory_files:
            parts.append(_RUNTIME_FILES.bind(count=len(memory_files)))
        return tuple(parts)

    def get_profile_description(self, profile_name: str) -> str:
        """Look up a profile's description from the registry."""
        registry = self._services.agent_registry
        if registry is None:
            return ""
        profile = registry.get(profile_name)
        return profile.description if profile else ""
