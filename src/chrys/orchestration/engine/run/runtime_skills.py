# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Runtime skill refresh helpers for main-agent turns."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chrys.foundation.events.types import AgentRuntimeUpdated, Warning

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import RuntimeSkillDetails
    from chrys.orchestration.engine.build.loaded import AgentManifest
    from chrys.orchestration.engine.loader import AgentLoader
    from chrys.orchestration.engine.state.active_session import ActiveSession
    from chrys.orchestration.engine.state.current_agent import CurrentAgent
    from chrys.service.skills.model import SkillProviderWarning
    from chrys.service.skills.provider import ChrysSkillsProvider, StagedSkillRefresh


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StagedRuntimeSkillRefresh:
    """Runtime skill refresh staged before an active-injection owner is revalidated."""

    provider: ChrysSkillsProvider
    context: StagedSkillRefresh
    warnings: list[SkillProviderWarning]
    skill_names: list[str]
    skill_sources: dict[str, list[str]]
    skill_details: list[RuntimeSkillDetails]
    skill_catalog: str | None


@dataclass(frozen=True)
class CommittedSkillRefresh:
    """Warnings and the exact manifest installed by one synchronous refresh commit."""

    warnings: list[SkillProviderWarning]
    manifest: AgentManifest


class RuntimeSkillRefresher:
    """Refresh runtime skills through the engine host."""

    def __init__(
        self,
        *,
        current: CurrentAgent,
        loader: AgentLoader,
        session: ActiveSession,
        bus: EventBus,
    ) -> None:
        self._current = current
        self._loader = loader
        self._session = session
        self._bus = bus

    async def refresh(self, *, update_active_turn: bool = False) -> None:
        """Refresh runtime skills and optionally update the active reminder snapshot."""
        session_id = self._session.session_id
        staged = await self.stage_refresh()
        if staged is None:
            return
        committed = self.commit_staged_refresh(staged, suppress_provider_errors=True)
        if committed is None:
            return
        await self.publish_committed_refresh(committed, session_id=session_id)
        if update_active_turn and self._current.loaded is not None:
            self._current.loaded.reminder_middleware.update_skill_catalog_for_active_turn()

    async def stage_refresh(self) -> StagedRuntimeSkillRefresh | None:
        """Discover runtime skills without mutating live provider or host runtime state."""
        provider = self._current.loaded.skills_provider if self._current.loaded is not None else None
        if provider is None:
            return None
        try:
            staged = await provider.stage_context_refresh()
        except Exception:
            logger.exception("Failed to stage runtime skill refresh")
            return None
        return StagedRuntimeSkillRefresh(
            provider=provider,
            context=staged,
            warnings=list(staged.warnings),
            skill_names=staged.skill_names(),
            skill_sources=staged.skill_sources(),
            skill_details=staged.skill_details(),
            skill_catalog=staged.render_catalog_reminder(),
        )

    def commit_staged_refresh(
        self,
        staged: StagedRuntimeSkillRefresh | None,
        *,
        suppress_provider_errors: bool = False,
    ) -> CommittedSkillRefresh | None:
        """Commit a staged runtime skill refresh synchronously to the captured owner."""
        if staged is None:
            return None
        try:
            warnings = staged.provider.commit_context_refresh(staged.context)
        except Exception:
            if not suppress_provider_errors:
                raise
            logger.exception("Failed to refresh runtime skills")
            return None
        manifest = self._loader.apply_skill_refresh(
            skill_names=list(staged.skill_names),
            skill_sources={source: list(names) for source, names in staged.skill_sources.items()},
            skill_details=list(staged.skill_details),
        )
        return CommittedSkillRefresh(warnings=warnings, manifest=manifest)

    async def publish_committed_refresh(
        self,
        committed: CommittedSkillRefresh | None,
        *,
        session_id: str | None,
    ) -> None:
        """Publish captured warnings and manifest values for one committed refresh."""
        if committed is None:
            return
        for warning in committed.warnings:
            await self._bus.publish(Warning(code=warning.code, message=warning.message, session_id=session_id))
        await self._publish_runtime_update(committed.manifest, session_id=session_id)

    async def _publish_runtime_update(self, manifest: AgentManifest, *, session_id: str | None) -> None:
        """Publish the current runtime catalog."""
        await self._bus.publish(
            AgentRuntimeUpdated(
                session_id=session_id,
                model_profile_id=manifest.runtime_details.model.profile_id,
                max_context_tokens=manifest.runtime_details.model.max_context_tokens,
                tool_names=list(manifest.tool_names),
                skill_names=list(manifest.skill_names),
                memory_files=list(manifest.memory_files),
                runtime_details=copy.deepcopy(manifest.runtime_details),
            )
        )
