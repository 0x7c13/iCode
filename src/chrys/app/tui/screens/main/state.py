# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Explicit state containers for the main TUI screen."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from chrys.foundation.config.settings import DEFAULT_WORKSPACE_MRU_MAX_ENTRIES
from chrys.foundation.events.types import AgentRuntimeDetails
from chrys.foundation.models.execution import ExecutionSnapshot
from chrys.foundation.platform import safe_getcwd
from chrys.service.approval.policy import ApprovalMode

if TYPE_CHECKING:
    from chrys.foundation.config.settings_store import SettingsHandle
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import (
        BackgroundTasksChanged,
        Error,
        HookExecutionsChanged,
        InvocationContentAbandoned,
        InvocationContentDelta,
        InvocationContentFinished,
        InvocationMessage,
    )
    from chrys.orchestration.engine.engine import AgentEngine
    from chrys.service.profiles.agents.registry import AgentProfileRegistry
    from chrys.service.profiles.models.registry import ModelProfileRegistry
    from chrys.service.state.store import StateStore

    DeferredTurnEvent = (
        InvocationMessage | InvocationContentDelta | InvocationContentFinished | InvocationContentAbandoned | Error
    )


@dataclass
class MainScreenServices:
    """Service handles used by the main-screen controllers."""

    bus: EventBus
    settings_handle: SettingsHandle | None = None
    state_store: StateStore | None = None
    agent_registry: AgentProfileRegistry | None = None
    model_registry: ModelProfileRegistry | None = None
    active_model_profile_id: str = ""
    apply_saved_model_on_restore: bool = True
    engine_provider: Callable[[], AgentEngine] | None = None
    workspace_mru_max_entries: int = DEFAULT_WORKSPACE_MRU_MAX_ENTRIES

    def execution(self) -> ExecutionSnapshot:
        """Read the lease owner, including workflow admission and finalization."""
        return self.engine_provider().execution() if self.engine_provider is not None else ExecutionSnapshot("idle")

    def execution_busy(self) -> bool:
        return self.engine_provider().execution_busy() if self.engine_provider is not None else False

    def session_generation(self) -> int:
        """The engine's session generation; 0 without an engine."""
        return self.engine_provider().session_generation if self.engine_provider is not None else 0

    def turn_lifecycle_task(self) -> asyncio.Task[None] | None:
        """The live turn's lifecycle task (run, save, after-turn hooks), if any."""
        return self.engine_provider().turn_lifecycle_task if self.engine_provider is not None else None

    def was_turn_lifecycle_saved(self, task: asyncio.Task[None]) -> bool:
        """Whether *task*'s turn completed its final session save successfully; False without an engine."""
        return self.engine_provider().was_turn_lifecycle_saved(task) if self.engine_provider is not None else False


@dataclass
class RunState:
    """Live run and loading flags."""

    agent_running: bool = False
    agent_loading: bool = False
    has_messages: bool = False
    generation: int = 0
    started_at: datetime | None = None
    # Only the screen writes this: set when a run starts, consumed when it
    # stops. Backend handlers clear ``agent_running`` before the screen sees
    # the stop, so that flag cannot tell the screen a run just ended.
    turn_end_check_pending: bool = False


@dataclass
class RuntimeState:
    """Active runtime metadata shown by the main screen."""

    profile: str = ""
    main_usage_source_id: str = ""
    details: AgentRuntimeDetails = field(default_factory=AgentRuntimeDetails)
    details_confirmed: bool = False
    approval_mode: ApprovalMode = ApprovalMode.MANUAL
    background_tasks: BackgroundTasksChanged | None = None
    """The last list of the main agent's background commands; shown while its session is the current one."""
    hook_snapshots: dict[str, HookExecutionsChanged] = field(default_factory=dict)
    """The last snapshot of hook runs the user waits on, per session; the current session's shows.

    Keyed by session so neither a closing session's late snapshot nor a new
    session's early one (before it becomes current) displaces the other.
    """
    hook_render_pending: bool = False
    """A render of the last snapshot is queued: later snapshots in the same pump only replace what it reads."""
    # Set when a Save inside the agent config modal renames the active profile.
    # The engine switch waits for the modal to close (``on_agent_config_result``):
    # switching sooner would tear down the live agent while the modal's panels
    # still reference the old profile object.
    pending_active_switch: str | None = None

    def keep_hook_snapshot_of(self, session_id: str) -> None:
        """Forget every session's hook snapshot but *session_id*'s."""
        stale = [key for key in self.hook_snapshots if key != session_id]
        for key in stale:
            del self.hook_snapshots[key]


@dataclass
class SessionViewState:
    """Session lifecycle flags owned by the UI."""

    restoring_session: bool = False
    creating_new_session: bool = False


@dataclass
class UsageViewState:
    """Latest usage values rendered by the UI."""

    last_usage_tokens: int = 0
    last_total_session_tokens: int = 0


@dataclass
class WorkspaceViewState:
    """Workspace cwd shown by the main screen."""

    current_cwd: str = field(default_factory=safe_getcwd)
    current_git_branch: str = ""
    roots: list[str] = field(default_factory=lambda: [safe_getcwd()])


@dataclass
class OverlayState:
    """Overlay and modal confirmation flags."""

    interrupt_confirm_active: bool = False


@dataclass
class ShellModeState:
    """Shell-mode layout state."""

    active: bool = False
    fullscreen_terminal: bool = False


@dataclass
class ProfileSwitchMarkerState:
    """State for consecutive profile-switch system messages."""

    from_profile: str | None = None
    to_profile: str | None = None
    seq: int = 0


@dataclass
class WorkspaceMarkerState:
    """State for consecutive workspace-change system messages."""

    original_cwd: str | None = None
    current_cwd: str = field(default_factory=safe_getcwd)


# What a submit asks for: a new turn, a message for the running turn, or a
# retry. A refusal shows by it, and never ends a run: a new turn or retry
# starts running only after its publish returns, and the turn a mid-run
# message was meant for goes on without it.
type SubmitKind = Literal["turn", "injection", "retry"]


@dataclass
class SubmitCoordinator:
    """State for a user submit while backend validation is in progress."""

    active: bool = False
    text: str = ""
    blocked: bool = False
    kind: SubmitKind = "turn"
    # The task publishing the submit, named just before it publishes.
    # ``EventBus.publish`` awaits its handlers in order, so the backend's
    # refusal reaches the frontend on this task before the publish returns;
    # what a running turn publishes meanwhile comes from that turn's own task,
    # and is the turn's.
    publisher: asyncio.Task[Any] | None = field(default=None, repr=False)

    def begin(self, text: str, *, kind: SubmitKind = "turn") -> None:
        """Mark a submit as active and clear any previous block decision."""
        self.active = True
        self.text = text
        self.blocked = False
        self.kind = kind
        self.publisher = None

    def publish_from_current_task(self) -> None:
        """Name the current task as the one about to publish the submit."""
        self.publisher = asyncio.current_task()

    def in_publish(self) -> bool:
        """Return whether the code running now is inside this submit's publish."""
        return self.active and self.publisher is not None and asyncio.current_task() is self.publisher

    def block(self) -> None:
        """Record that the backend refused the submit."""
        self.blocked = True

    def clear(self) -> None:
        """Reset submit state after the publish phase is complete."""
        self.active = False
        self.text = ""
        self.blocked = False
        self.kind = "turn"
        self.publisher = None


@dataclass
class PendingInjectionState:
    """Tracks the mid-run injection queued behind the locked input bar.

    At most one injection is pending at a time — the input stays locked
    until the backend reports a ``UserInjectResult`` for it or the user
    cancels it with Esc.
    """

    injection_id: str | None = None
    text: str = ""

    @property
    def active(self) -> bool:
        """Return whether an injection is queued and awaiting its outcome."""
        return self.injection_id is not None

    def begin(self, injection_id: str, text: str) -> None:
        """Track a newly queued injection."""
        self.injection_id = injection_id
        self.text = text

    def matches(self, injection_id: str | None) -> bool:
        """Return whether *injection_id* identifies the tracked injection."""
        return self.injection_id is not None and self.injection_id == injection_id

    def clear(self) -> None:
        """Stop tracking after the outcome arrived or the user cancelled."""
        self.injection_id = None
        self.text = ""


@dataclass
class TurnRenderGate:
    """Defers fast backend events while the accepted user bubble mounts, then replays them in order."""

    active: bool = False
    _deferred: deque[DeferredTurnEvent] = field(default_factory=deque)
    _replaying: DeferredTurnEvent | None = None

    def begin(self) -> None:
        """Start deferring backend events for the current user render."""
        self.active = True
        self._deferred.clear()

    def defers(self, event: DeferredTurnEvent) -> bool:
        """Whether *event* waits its turn: the gate is up and it is not the event being replayed."""
        return self.active and event is not self._replaying

    def defer(self, event: DeferredTurnEvent) -> None:
        """Record an event to flush after user rendering completes."""
        self._deferred.append(event)

    def finish(self, *, rendered: bool) -> None:
        """End the user render: a failed one stops deferring and drops what it deferred.

        A rendered one keeps deferring until :meth:`drain` has replayed the backlog.
        """
        if not rendered:
            self.active = False
            self._deferred.clear()

    async def drain(self, replay: Callable[[DeferredTurnEvent], Awaitable[None]]) -> None:
        """Replay the deferred events in arrival order, then stop deferring.

        A replay can await a mount: events arriving meanwhile queue behind the
        backlog, so streamed text, which only appends, keeps its order.
        """
        try:
            while self._deferred:
                self._replaying = self._deferred.popleft()
                await replay(self._replaying)
        finally:
            self._replaying = None
            self.active = False


@dataclass
class MainScreenState:
    """Composite state for the main-screen shell and controllers."""

    run: RunState = field(default_factory=RunState)
    runtime: RuntimeState = field(default_factory=RuntimeState)
    session: SessionViewState = field(default_factory=SessionViewState)
    usage: UsageViewState = field(default_factory=UsageViewState)
    workspace: WorkspaceViewState = field(default_factory=WorkspaceViewState)
    overlays: OverlayState = field(default_factory=OverlayState)
    shell: ShellModeState = field(default_factory=ShellModeState)
    profile_marker: ProfileSwitchMarkerState = field(default_factory=ProfileSwitchMarkerState)
    workspace_marker: WorkspaceMarkerState = field(default_factory=WorkspaceMarkerState)
    submit: SubmitCoordinator = field(default_factory=SubmitCoordinator)
    render_gate: TurnRenderGate = field(default_factory=TurnRenderGate)
    pending_injection: PendingInjectionState = field(default_factory=PendingInjectionState)
