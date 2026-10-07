# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Hook configuration shared by Chat rebuilds and Workflow run admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chrys.foundation.events.types import Warning
from chrys.foundation.i18n import DisplayBlock, DisplayPath, msg
from chrys.foundation.platform import get_platform

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.service.hooks.manager import HookManager

_CONSTRUCTION_GLOBAL_HOOKS_INVALID = msg(
    "construction.global_hooks_invalid",
    fallback="Global hooks config could not be loaded: {detail}.  Global hooks disabled.",
    multiline=True,
)

_CONSTRUCTION_PROJECT_HOOKS_INVALID = msg(
    "construction.project_hooks_invalid",
    fallback="Project hooks config could not be loaded: {detail}. Project hooks disabled; global hooks unaffected.",
    multiline=True,
)

_CONSTRUCTION_HOOK_SKIPPED = msg(
    "construction.hook_skipped",
    fallback="Hook '{hook_id}' in {path} was skipped: {detail}. The other hooks in this file still run.",
    multiline=True,
)


class SessionHookFactory:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def __call__(
        self, *, project_root: str, project_hooks_enabled: bool, session_id: str | None, request_id: str = ""
    ) -> HookManager | None:
        """Resolve hooks for the target workspace and report warnings to its session."""
        from pathlib import Path

        from chrys.service.hooks.loader import (
            HooksConfigError,
            load_hooks_dir,
            load_hooks_project,
            merge_hooks_files,
        )
        from chrys.service.hooks.manager import HookManager
        from chrys.service.hooks.schema import HooksFile

        config_dir = get_platform().config_dir
        root = Path(project_root)

        global_hooks: HooksFile | None = None
        try:
            global_hooks = load_hooks_dir(config_dir)
            if not global_hooks.source:
                global_hooks = None
        except (HooksConfigError, OSError) as exc:
            await self._bus.publish(
                Warning(
                    code="hooks_config_invalid",
                    message=f"Global hooks config could not be loaded: {exc}.  Global hooks disabled.",
                    display_message=_CONSTRUCTION_GLOBAL_HOOKS_INVALID.bind(detail=DisplayBlock(str(exc))),
                    session_id=session_id,
                    request_id=request_id,
                )
            )
            global_hooks = None

        project_hooks: HooksFile | None = None
        try:
            if project_hooks_enabled:
                project_hooks = load_hooks_project(root)
        except (HooksConfigError, OSError) as exc:
            await self._bus.publish(
                Warning(
                    code="project_hooks_config_invalid",
                    message=(
                        f"Project hooks config could not be loaded: {exc}. Project hooks disabled; global hooks unaffected."
                    ),
                    display_message=_CONSTRUCTION_PROJECT_HOOKS_INVALID.bind(detail=DisplayBlock(str(exc))),
                    session_id=session_id,
                    request_id=request_id,
                )
            )
            project_hooks = None

        merged = merge_hooks_files(project=project_hooks, global_=global_hooks)
        for loaded in (merged.project, merged.global_):
            if loaded is None:
                continue
            for skipped in loaded.skipped_hooks:
                await self._bus.publish(
                    Warning(
                        code="hook_skipped",
                        message=(
                            f"Hook '{skipped.id}' in {loaded.source} was skipped: {skipped.reason}. "
                            "The other hooks in this file still run."
                        ),
                        display_message=_CONSTRUCTION_HOOK_SKIPPED.bind(
                            hook_id=skipped.id, path=DisplayPath(loaded.source), detail=DisplayBlock(skipped.reason)
                        ),
                        session_id=session_id,
                        request_id=request_id,
                    )
                )
        if not merged.sources:
            return None
        return HookManager(
            file=merged,
            hooks_dir=config_dir / "hooks",
        )
