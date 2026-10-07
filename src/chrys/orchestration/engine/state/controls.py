# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Engine-internal profile, settings, workspace, and approval controls."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chrys.foundation.config.context import EvalContext
from chrys.foundation.config.process_settings import reattribute_command_line, route_restart_settings
from chrys.foundation.config.settings_store import LoadedSettings, SettingsHandle, load_settings
from chrys.foundation.config.spec import Source
from chrys.foundation.config.warnings import settings_warning_events
from chrys.foundation.events.types import (
    AgentProfileSwitch,
    ApprovalModeUpdated,
    Error,
    ModelProfileSwitched,
    ProfileSwitched,
    SetApprovalMode,
    SetModelProfile,
    SettingsReload,
    SettingsReloaded,
    Warning,
    WorkspaceChange,
    WorkspaceUpdated,
)
from chrys.foundation.i18n import DisplayPath, DisplaySequence, msg
from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.models.workspace import Workspace
from chrys.foundation.platform import safe_getcwd
from chrys.foundation.platform.files import surrogate_safe_text
from chrys.orchestration.engine.state.lifecycle_permits import RebuildControlToken, RebuildPermit, RebuildPermitDenied
from chrys.service.approval.policy import ApprovalMode

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.orchestration.engine.session_lifecycle import SessionLifecycle
    from chrys.orchestration.engine.state.active_session import ActiveSession
    from chrys.orchestration.engine.state.current_agent import CurrentAgent
    from chrys.orchestration.engine.state.lifecycle_permits import LifecyclePermits
    from chrys.orchestration.workflows.coordinator import WorkflowCoordinator
    from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
    from chrys.service.profiles.agents.registry import AgentProfileRegistry
    from chrys.service.profiles.agents.schema import AgentProfile


logger = logging.getLogger(__name__)

_CONTROLS_NO_REGISTRY = msg(
    "controls.no_registry",
    fallback="No profile registry configured — cannot switch profiles",
)
_CONTROLS_PROFILE_NOT_FOUND = msg(
    "controls.profile_not_found",
    fallback="Profile '{profile_name}' not found",
)
_CONTROLS_MODEL_SWITCH_NOT_READY = msg(
    "controls.model_switch_not_ready",
    fallback="No active agent — cannot switch model",
)
_CONTROLS_WORKSPACE_SWITCH_NOT_READY = msg(
    "controls.workspace_switch_not_ready",
    fallback="No active agent — cannot change workspace",
)
_CONTROLS_WORKSPACE_MISSING = msg(
    "controls.workspace_missing",
    fallback="Cannot change the working directory: {path} does not exist.",
)
_CONTROLS_SETTINGS_RESTART_REQUIRED = msg(
    "controls.settings_restart_required",
    fallback="Saved; takes effect after restart: {keys}.",
)

PersistApprovalModeFn = Callable[[str], None]


def _denial_allows_already_satisfied_success(denied: RebuildPermitDenied) -> bool:
    """Return whether a denial can be converted to an already-satisfied success."""
    return denied.reason == "superseded"


def _workspace_primary_matches(workspace: Workspace | None, primary_cwd: str) -> bool:
    """Return whether *primary_cwd* names the live workspace primary cwd."""
    if workspace is None:
        return False
    if workspace.primary_cwd == primary_cwd:
        return True
    return workspace.primary_cwd == Workspace.from_cwd(primary_cwd).primary_cwd


class RuntimeControls:
    """Applies profile, model, settings, workspace, and approval changes."""

    def __init__(
        self,
        *,
        session: ActiveSession,
        current: CurrentAgent,
        permits: LifecyclePermits,
        lifecycle: SessionLifecycle,
        bus: EventBus,
        settings_handle: SettingsHandle,
        agent_registry: AgentProfileRegistry | None,
        workspace_change_tracker: WorkspaceChangeTracker,
        workflows: WorkflowCoordinator,
    ) -> None:
        self._session = session
        self._current = current
        self._permits = permits
        self._lifecycle = lifecycle
        self._bus = bus
        self._settings_handle = settings_handle
        self._agent_registry = agent_registry
        self._workspace_change_tracker = workspace_change_tracker
        self._workflows = workflows

    def current_profile_snapshot(self) -> ProfileSwitched:
        """Build a no-op ``ProfileSwitched`` reflecting the live runtime.

        A frontend reselecting the already-active agent performs no backend
        switch and emits no event, so callers that still owe the client the
        standard runtime envelope read it here instead of fabricating blank
        fields. ``from``/``to`` are identical because nothing changed. Field
        sourcing mirrors the real switch event so the two cannot drift.
        """
        profile = self._session.agent_profile
        name = profile.name if profile else ""
        display = (profile.display_name or profile.name) if profile else ""
        messages = (
            self._current.loaded.bindings.backend.history_state.get("messages", [])
            if self._current.loaded is not None
            else []
        )
        return ProfileSwitched(
            from_profile=name,
            to_profile=name,
            from_display_name=display,
            to_display_name=display,
            message_count=len(messages),
            model_profile_id=self._current.manifest.active_profile.id if self._current.manifest.active_profile else "",
            max_context_tokens=self._current.manifest.active_profile.max_context_tokens
            if self._current.manifest.active_profile
            else 0,
            session_id=self._session.session_id,
            tool_names=list(self._current.manifest.tool_names),
            skill_names=list(self._current.manifest.skill_names),
            sub_agent_tool_names=self._current.loaded.sub_agent_tools.tool_names()
            if self._current.loaded is not None and self._current.loaded.sub_agent_tools
            else [],
            memory_files=list(self._current.manifest.memory_files),
            runtime_details=copy.deepcopy(self._current.manifest.runtime_details),
        )

    async def on_set_approval_mode(
        self,
        event: SetApprovalMode,
        *,
        persist_approval_mode_fn: PersistApprovalModeFn,
    ) -> None:
        """Update the active approval mode on the running middleware."""
        try:
            mode = ApprovalMode(event.mode)
        except ValueError:
            logger.warning("Unknown approval mode: %r", event.mode)
            return
        self._session.approval_mode = mode
        if self._current.loaded is not None:
            self._current.loaded.bindings.approval.set_approval_mode(mode)
        # Propagate to sub-agent tools so each fresh sub-agent invocation
        # constructs its ApprovalMiddleware with the live mode.
        if self._current.loaded is not None and self._current.loaded.sub_agent_tools is not None:
            self._current.loaded.sub_agent_tools.set_approval_mode(mode)
        self._workflows.set_approval_mode(mode)
        if event.persist:
            # Persist the user's choice so it survives restart.  BYPASS is
            # downgraded to AUTO inside ``persist_approval_mode`` to avoid
            # booting into unattended auto-approval on next launch — mirror
            # that same downgrade onto the in-memory Settings so a later
            # ``SettingsReload`` stays consistent with what's on disk.
            #
            # Through the overlay rather than onto the field: this key is
            # ``Risk.DANGEROUS``, so a rejected value seals it at ``manual``, and
            # an in-place write would leave that seal on top of the mode the user
            # just chose.
            self._settings_handle.persist_approval_default(mode.value, persist=persist_approval_mode_fn)
        await self._bus.publish(ApprovalModeUpdated(mode=mode.value, session_id=self._session.session_id))

    async def on_profile_switch(self, event: AgentProfileSwitch) -> None:
        """Handle agent profile switch — preserves conversation history."""
        token = self._permits.capture_control_token()
        if self._agent_registry is None:
            await self._bus.publish(
                Error(
                    code="no_registry",
                    message="No profile registry configured — cannot switch profiles",
                    display_message=_CONTROLS_NO_REGISTRY.bind(),
                    session_id=self._session.session_id,
                )
            )
            return

        new_profile = self._agent_registry.get(event.profile_name)
        if new_profile is None:
            await self._bus.publish(
                Error(
                    code="profile_not_found",
                    message=f"Profile '{event.profile_name}' not found",
                    display_message=_CONTROLS_PROFILE_NOT_FOUND.bind(profile_name=event.profile_name),
                    session_id=self._session.session_id,
                )
            )
            return

        permit = await self._permits.acquire_rebuild_permit(token)
        if isinstance(permit, RebuildPermitDenied):
            if await self._publish_profile_satisfied_or_denied(event.profile_name, permit, token.session_id):
                return
            return
        try:
            if (
                self._current.loaded is not None
                and self._session.agent_profile
                and new_profile.name == self._session.agent_profile.name
            ):
                await self._bus.publish(self._profile_switched_snapshot(permit.token.session_id))
                return
            if self._current.loaded is None:
                await self._lifecycle.start_with_rebuild_permit(permit, new_profile, operation="switch")
                await self._bus.publish(self._profile_switched_snapshot(permit.token.session_id))
                return
            await self._lifecycle.reload_with_rebuild_permit(permit, new_profile, operation="switch")
        finally:
            self._permits.release_rebuild_permit(permit)

    async def _publish_profile_satisfied_or_denied(
        self,
        profile_name: str,
        denied: RebuildPermitDenied,
        session_id: str | None,
    ) -> bool:
        if _denial_allows_already_satisfied_success(denied) and (
            self._session.agent_profile is not None and self._session.agent_profile.name == profile_name
        ):
            await self._bus.publish(self._profile_switched_snapshot(session_id))
            return True
        await self._publish_rebuild_denied(denied, session_id)
        return False

    def _profile_switched_snapshot(self, session_id: str | None) -> ProfileSwitched:
        return dataclasses.replace(self.current_profile_snapshot(), session_id=session_id)

    async def _publish_rebuild_denied(
        self,
        denied: RebuildPermitDenied,
        session_id: str | None,
    ) -> None:
        await self._bus.publish(Error(code=denied.code, message=denied.message, session_id=session_id))

    async def _publish_model_switched_snapshot(
        self,
        profile_id: str,
        session_id: str | None,
    ) -> None:
        active = self._current.manifest.active_profile
        await self._bus.publish(
            ModelProfileSwitched(
                model_profile_id=active.id if active else profile_id,
                max_context_tokens=active.max_context_tokens if active else 0,
                runtime_details=copy.deepcopy(self._current.manifest.runtime_details),
                session_id=session_id,
            )
        )

    def _pin_session_model_profile(self, profile_id: str) -> None:
        self._settings_handle.install(
            self._settings_handle.loaded.overlay(
                Source.SESSION,
                model_profile=profile_id,
                model_profile_override=profile_id,
                model_profile_override_sub_agents=False,
            ),
        )
        self._session.model_profile_pinned = True

    async def _publish_workspace_updated_snapshot(
        self,
        session_id: str | None,
        *,
        primary_cwd: str | None = None,
    ) -> None:
        workspace = self._session.workspace
        resolved_primary_cwd = primary_cwd if primary_cwd is not None else (workspace.primary_cwd if workspace else "")
        await self._bus.publish(
            WorkspaceUpdated(
                primary_cwd=resolved_primary_cwd,
                working_dirs=[d.path for d in workspace.working_dirs] if workspace is not None else [],
                reference_files=list(workspace.reference_files) if workspace is not None else [],
                session_id=session_id,
            )
        )

    async def _publish_model_satisfied_or_denied(
        self,
        profile_id: str,
        denied: RebuildPermitDenied,
        session_id: str | None,
    ) -> None:
        if _denial_allows_already_satisfied_success(denied) and (
            self._current.manifest.active_profile is not None and self._current.manifest.active_profile.id == profile_id
        ):
            self._pin_session_model_profile(profile_id)
            await self._publish_model_switched_snapshot(profile_id, session_id)
            return
        await self._publish_rebuild_denied(denied, session_id)

    async def _publish_workspace_satisfied_or_denied(
        self,
        primary_cwd: str,
        denied: RebuildPermitDenied,
        session_id: str | None,
    ) -> None:
        if _denial_allows_already_satisfied_success(denied) and _workspace_primary_matches(
            self._session.workspace, primary_cwd
        ):
            await self._publish_workspace_updated_snapshot(session_id, primary_cwd=primary_cwd)
            return
        await self._publish_rebuild_denied(denied, session_id)

    async def _acquire_rebuild_permit_or_publish_error(
        self,
        token: RebuildControlToken,
    ) -> RebuildPermit | None:
        permit = await self._permits.acquire_rebuild_permit(token)
        if isinstance(permit, RebuildPermitDenied):
            await self._publish_rebuild_denied(permit, token.session_id)
            return None
        return permit

    async def _start_or_restart_with_permit(
        self,
        permit: RebuildPermit,
        profile: AgentProfile,
        *,
        operation: str,
        workspace: Workspace | None = None,
        staged_loaded: LoadedSettings | None = None,
    ) -> None:
        if self._current.loaded is None:
            # The workspace rides as staged input, exactly like the settings: a
            # build that fails before committing must leave the live root — the
            # one the live settings and hooks were derived from — untouched.
            await self._lifecycle.start_with_rebuild_permit(
                permit, profile, operation=operation, staged_loaded=staged_loaded, workspace=workspace
            )
            return
        await self._lifecycle.reload_with_rebuild_permit(
            permit, profile, workspace=workspace, operation=operation, staged_loaded=staged_loaded
        )

    async def on_set_model_profile(self, event: SetModelProfile) -> None:
        """Switch the active model profile for this session only (no global .env write).

        Swaps the in-memory model selector and explicit override, then rebuilds;
        the build path resolves the model from ``settings`` against the registry,
        and credentials come from the profile itself, so sessions stay isolated.
        """
        if self._session.agent_profile is None:
            await self._bus.publish(
                Error(
                    code="runtime_mutation_not_ready",
                    message="No active agent — cannot switch model",
                    display_message=_CONTROLS_MODEL_SWITCH_NOT_READY.bind(),
                    session_id=self._session.session_id,
                )
            )
            return
        token = self._permits.capture_control_token()
        permit = await self._permits.acquire_rebuild_permit(token)
        if isinstance(permit, RebuildPermitDenied):
            await self._publish_model_satisfied_or_denied(event.profile_id, permit, token.session_id)
            return

        old_loaded = self._settings_handle.loaded
        old_pinned = self._session.model_profile_pinned

        try:
            try:
                self._pin_session_model_profile(event.profile_id)
                await self._start_or_restart_with_permit(permit, self._session.agent_profile, operation="model_switch")
            except Exception:
                # Rebuild failed (AgentLoadFailed already published). Restore the pin/settings
                # so the still-running executor stays consistent with session state.
                self._settings_handle.install(old_loaded)
                self._session.model_profile_pinned = old_pinned
                raise
            await self._publish_model_switched_snapshot(event.profile_id, token.session_id)
        finally:
            self._permits.release_rebuild_permit(permit)

    def _session_pin_overrides(self) -> dict[str, Any]:
        """Snapshot the per-session pins every re-load must carry.

        Per-session overrides injected into ``Settings`` (not the environment) are
        carried across only when pinned, so unpinned sessions still pick up a
        changed env value on the next load:

        * ``model_profile`` / ``model_profile_override`` — pinned by a per-session
          model switch.
        * ``ask_user_timeout_seconds`` — pinned when ACP owns the ask_user
          lifetime (set via dataclasses.replace at launch — see app/cli/acp.py).
        * ``frontend_default_max_transient_retries`` — launch-mode policy; travels
          as the :func:`_reload_eval_context` instead, and the separate
          env-derived override is deliberately re-read.
        """
        overrides: dict[str, Any] = {}
        if self._session.ask_user_timeout_pinned:
            overrides["ask_user_timeout_seconds"] = self._settings_handle.settings.ask_user_timeout_seconds
        if self._session.model_profile_pinned:
            overrides["model_profile"] = self._settings_handle.settings.model_profile
            overrides["model_profile_override"] = self._settings_handle.settings.model_profile_override
            overrides["model_profile_override_sub_agents"] = (
                self._settings_handle.settings.model_profile_override_sub_agents
            )
        return overrides

    def _reload_eval_context(self) -> EvalContext:
        """The launch mode's retry policy, passed *into* the load.

        An input to the load rather than an override: the project layer's
        tighten/loosen verdicts are evaluated against it, and every re-load must
        use the same one the initial load did.
        """
        return EvalContext(
            frontend_default_max_transient_retries=self._settings_handle.settings.frontend_default_max_transient_retries
        )

    def _session_project_root(self) -> Path:
        """The workspace root whose project trust domain this session lives under."""
        workspace = self._session.workspace
        return Path(workspace.primary_cwd) if workspace is not None else Path(safe_getcwd())

    async def on_settings_reload(self, _event: SettingsReload) -> None:
        """Handle settings reload — reload Settings from disk and env, then rebuild."""
        token = self._permits.capture_control_token()
        # Taken even with nothing built yet. A host that has not built — startup
        # before the first build, a first build that failed, a host that only
        # subscribed — must still genuinely reload; echoing completion without
        # reloading would report success and then hand the first build the old
        # configuration. The permit is also what serializes the load against a
        # build, since ``start`` takes this same boundary: the values installed
        # below cannot land halfway through one.
        permit = await self._acquire_rebuild_permit_or_publish_error(token)
        if permit is None:
            return

        old_loaded = self._settings_handle.loaded
        try:
            try:
                # Re-read settings from env (matches TUI behavior, which persists
                # changes before triggering reload), for this session's workspace
                # root — the project trust domain is root-derived, so the reload
                # must re-derive it from where the session actually lives.
                #
                # Off-thread because the load reads config files and waits on their
                # lock, and this handler runs inline on the bus: a synchronous load
                # here would stall every other event for as long as the disk takes.
                # The pins are snapshotted before the hand-off, so the load works
                # from the state that asked for it.
                candidate = await asyncio.to_thread(
                    load_settings,
                    project_root=self._session_project_root(),
                    eval_context=self._reload_eval_context(),
                    **self._session_pin_overrides(),
                )
                # Routed after the load, not excluded from it: the reload still has
                # to report a RESTART value the user typed wrong, it just must not
                # claim the good ones took effect. The snapshot fields keep the
                # bootstrap values their readers hold; the rest of the RESTART tier
                # is held at the values already in force, so the rebuild below
                # cannot apply what only a restart may.
                loaded, deferred_keys = route_restart_settings(
                    reattribute_command_line(candidate, old_loaded),
                    old_loaded,
                )
                # A reload is the moment a user finds out their edit did not take.
                # Startup already reports these; dropping them here would make the
                # same bad value silent from the second read onwards.
                for warning in settings_warning_events(loaded):
                    await self._bus.publish(dataclasses.replace(warning, session_id=self._session.session_id))
                if deferred_keys:
                    display_message = _CONTROLS_SETTINGS_RESTART_REQUIRED.bind(keys=DisplaySequence(deferred_keys))
                    await self._bus.publish(
                        Warning(
                            code="settings_restart_required",
                            message=format_message(display_message),
                            display_message=display_message,
                            session_id=self._session.session_id,
                        )
                    )

                # Pull the latest active profile instance from the registry so edits
                # saved to user YAML (e.g. MCP enabled/disabled) take effect immediately.
                profile_for_restart = self._session.agent_profile
                if profile_for_restart is not None and self._agent_registry is not None:
                    refreshed = self._agent_registry.get(profile_for_restart.name)
                    if refreshed is not None:
                        profile_for_restart = refreshed
                    else:
                        logger.warning(
                            "Active profile '%s' missing from registry during settings reload; reusing current profile.",
                            profile_for_restart.name,
                        )
            except Exception as exc:
                # A failing load (an unreadable settings file, say) aborts *before*
                # the rebuild publishes any completion event. The bus swallows
                # handler exceptions, so without an explicit failure here a caller
                # awaiting the reload would hang until its timeout. Nothing was
                # installed yet — the candidate stays staged until the rebuild
                # commits it — so surfacing the error is all there is to do.
                #
                # Individually invalid values no longer land here: they are rejected
                # by their coercer and reported as warnings, so one bad variable can
                # no longer take a whole reload down.
                await self._bus.publish(
                    Error(code="settings_reload_failed", message=str(exc), session_id=self._session.session_id)
                )
                raise

            # Only the rebuild is conditional: with nothing built there is no
            # runtime to replace, so the candidate is installed directly and the
            # first build will read it. With a runtime, the candidate travels as
            # the rebuild's staged input and is committed by the build itself,
            # together with the executor it configured: a rebuild that fails
            # before installing leaves the live settings untouched, and one that
            # fails after keeps the settings its executor was actually built from
            # — restoring the old ones there would desynchronize the two.
            if profile_for_restart is not None:
                await self._start_or_restart_with_permit(
                    permit, profile_for_restart, operation="settings_reload", staged_loaded=loaded
                )
                if (
                    old_loaded.settings.workspace_change_notice
                    and not self._settings_handle.settings.workspace_change_notice
                ):
                    # Drop the baseline only once the off-state rebuild has committed —
                    # a failed rebuild keeps the old enabled settings AND the baseline.
                    self._workspace_change_tracker.invalidate()
            else:
                self._settings_handle.install(loaded)
            await self._publish_settings_reloaded()
        finally:
            self._permits.release_rebuild_permit(permit)

    async def _publish_settings_reloaded(self) -> None:
        await self._bus.publish(
            SettingsReloaded(
                runtime_details=copy.deepcopy(self._current.manifest.runtime_details),
                session_id=self._session.session_id,
            )
        )

    async def on_workspace_change(self, event: WorkspaceChange) -> None:
        """Handle workspace/cwd change — rebuild agent with new workspace."""
        if self._session.agent_profile is None:
            await self._bus.publish(
                Error(
                    code="runtime_mutation_not_ready",
                    message="No active agent — cannot change workspace",
                    display_message=_CONTROLS_WORKSPACE_SWITCH_NOT_READY.bind(),
                    session_id=self._session.session_id,
                )
            )
            return
        if event.primary_cwd and (missing := Workspace.from_cwd(event.primary_cwd).missing_primary()) is not None:
            await self._bus.publish(
                Error(
                    code="workspace_change_failed",
                    message=f"Working directory does not exist: {surrogate_safe_text(missing)}",
                    display_message=_CONTROLS_WORKSPACE_MISSING.bind(path=DisplayPath(missing)),
                    session_id=self._session.session_id,
                )
            )
            return
        token = self._permits.capture_control_token()
        permit = await self._permits.acquire_rebuild_permit(token)
        if isinstance(permit, RebuildPermitDenied):
            await self._publish_workspace_satisfied_or_denied(event.primary_cwd, permit, token.session_id)
            return

        new_workspace = Workspace.from_cwd(event.primary_cwd)
        starting_without_executor = self._current.loaded is None
        try:
            # A workspace change is a settings reload in disguise: the project
            # trust domain is root-derived, so the new root's layers must be
            # audited and applied — and the old root's dropped — by the same
            # rebuild that installs the new workspace. Same routing as a reload
            # (RESTART values stay in force, the command line keeps its credit),
            # minus the restart-required warning: nothing was edited here, so any
            # deferred value was already reported by the reload that deferred it.
            try:
                old_loaded = self._settings_handle.loaded
                candidate = await asyncio.to_thread(
                    load_settings,
                    project_root=Path(new_workspace.primary_cwd),
                    eval_context=self._reload_eval_context(),
                    **self._session_pin_overrides(),
                )
                staged_loaded, _ = route_restart_settings(reattribute_command_line(candidate, old_loaded), old_loaded)
            except Exception as exc:
                # Same shape as a reload's load failure: it aborts before the
                # rebuild publishes any completion event, and the bus swallows
                # handler exceptions — without an explicit terminal event here a
                # caller awaiting the change would hang until its timeout.
                await self._bus.publish(
                    Error(code="workspace_change_failed", message=str(exc), session_id=self._session.session_id)
                )
                raise
            await self._start_or_restart_with_permit(
                permit,
                self._session.agent_profile,
                operation="workspace_change",
                workspace=new_workspace,
                staged_loaded=staged_loaded,
            )
            # After the commit, not before: these verdicts describe the new
            # root's files, and a failed rebuild keeps the old root's settings.
            for warning in settings_warning_events(staged_loaded):
                await self._bus.publish(dataclasses.replace(warning, session_id=self._session.session_id))
            if starting_without_executor:
                await self._publish_workspace_updated_snapshot(token.session_id)
        finally:
            self._permits.release_rebuild_permit(permit)
