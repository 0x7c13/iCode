# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""On-demand activation records and a live agent transcript in a screen-local modal."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual import on, work
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.widgets import Button, Static, Tab, TabbedContent, TabPane, Tabs

from chrys.app.tui.binding_display import CLOSE_BINDING, localized_binding
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.screens.dialogs.tool_view import ToolDetailModal
from chrys.app.tui.util.message_gate import messages_disabled
from chrys.app.tui.util.removal import remove_children_shielded
from chrys.app.tui.widgets import DialogButtonRow, DialogButtonSpec
from chrys.app.tui.widgets.chat.agent_transcript_surface import (
    AgentTranscriptJournal,
    AgentTranscriptOp,
    AgentTranscriptSurface,
    TranscriptErrorOp,
    TranscriptInterruptedOp,
)
from chrys.app.tui.widgets.chat.tool_call import ToolViewRequested
from chrys.app.tui.widgets.workflow import text
from chrys.app.tui.widgets.workflow.values import (
    ShownValue,
    ValueDocument,
    WorkflowValueView,
    record_values,
    shown_value,
)
from chrys.service.workflows.artifacts import latest_node_output, read_node_records
from chrys.service.workflows.graph import AgentSpec
from chrys.service.workflows.store import DATA_DROPPED_KEY
from chrys.service.workflows.transcript import read_node_transcript

if TYPE_CHECKING:
    from pathlib import Path

    from textual.app import ComposeResult
    from textual.widget import Widget

    from chrys.app.tui.i18n import LocaleController
    from chrys.app.tui.widgets.workflow.projector import ObservedRun
    from chrys.foundation.events.types import WorkflowNodeStateChanged
    from chrys.service.workflows.artifacts import NodeRecords, WorkflowRunRecord


class WorkflowNodeDialog(BaseDialog[None]):
    CSS_PATH = "workflow_node.tcss"
    BINDINGS: ClassVar[list] = [localized_binding("escape", "close", CLOSE_BINDING, show=False, priority=True)]

    def __init__(
        self,
        node: dict[str, Any],
        *,
        run: ObservedRun | None,
        directory: Path | None,
        history: Callable[[], list[WorkflowRunRecord]],
        retry: Callable[[WorkflowNodeStateChanged], bool],
        retry_pending: Callable[[WorkflowNodeStateChanged], bool],
        locale_controller: LocaleController | None = None,
    ) -> None:
        super().__init__()
        self.node = node
        self.agent_spec = AgentSpec.from_manifest(node["agent"]) if node["kind"] == "agent" else None
        self.run = run
        self.directory = directory
        self.history = history
        self.retry = retry
        self.retry_pending = retry_pending
        self.locale_controller = locale_controller
        self.attempts: list[WorkflowNodeStateChanged] = []
        self.selected: WorkflowNodeStateChanged | None = None
        self._revision: tuple[int, bool] | None = None
        self._generation = 0
        self._transcript_id: tuple[str, int | None] | None = None
        self._transcript_lock = asyncio.Lock()
        self._iteration_keys: tuple[str, ...] = ()
        self._attempt_keys: tuple[tuple[str, int], ...] = ()
        self._tabs_sync_pending = False
        # Exclusive async workers can overlap in their uncancellable disk reads.
        # Cache even an absent/error result, once per dialog's archived history.
        self._previous_lock = Lock()
        self._previous_result: tuple[ShownValue | None, str] | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="workflow-node") as container:
            container.border_title = Text(self.node["id"])
            container.border_subtitle = Text(text.state_label("pending", self.locale_controller))
            yield Tabs(id="workflow-iteration-tabs")
            yield Tabs(id="workflow-attempt-tabs")
            with TabbedContent():
                with TabPane(
                    Content.from_text(text.render(text.INPUT.bind(), self.locale_controller), markup=False),
                    id="workflow-input-tab",
                ):
                    yield WorkflowValueView(self.locale_controller, scroll=True, id="workflow-node-input")
                with TabPane(
                    Content.from_text(text.render(text.OUTPUT.bind(), self.locale_controller), markup=False),
                    id="workflow-output-tab",
                ):
                    trailing: list[Widget] = []
                    if self.node["kind"] != "agent":
                        trailing.append(Vertical(Static(id="workflow-node-error"), id="workflow-node-errors"))
                    trailing.append(Static(id="workflow-node-diagnostics"))
                    previous = WorkflowValueView(self.locale_controller, id="workflow-node-previous")
                    previous.border_title = Text(text.render(text.PREVIOUS_RUN.bind(), self.locale_controller))
                    trailing.append(previous)
                    # The run's errors, diagnostics and previous output scroll below the output's content.
                    yield WorkflowValueView(self.locale_controller, *trailing, scroll=True, id="workflow-node-output")
                if self.node["kind"] == "agent":
                    with TabPane(
                        Content.from_text(text.render(text.TRANSCRIPT.bind(), self.locale_controller), markup=False),
                        id="workflow-transcript-tab",
                    ):
                        with VerticalScroll(id="workflow-node-errors"):
                            yield Static(id="workflow-node-error")
                        yield Vertical(id="workflow-node-transcript")
            yield DialogButtonRow(
                DialogButtonSpec(
                    Text(text.render(text.RETRY.bind(), self.locale_controller)),
                    id="workflow-node-retry",
                    variant="warning",
                ),
                id="workflow-node-actions",
            )

    def on_mount(self) -> None:
        self.call_after_refresh(self.refresh_run)

    def refresh_run(self) -> None:
        if not self.is_mounted or self._torn_down() or self.app.screen is not self:
            return
        revision = (
            (self.run.revisions.get(self.node["id"], 0), self.run.finished is not None) if self.run else (0, True)
        )
        if revision == self._revision:
            return
        self._revision = revision
        old = self.selected
        was_latest = old is None or (self.attempts and old == self.attempts[-1])
        self.attempts = (
            [attempt for attempt in self.run.attempts.values() if attempt.node_id == self.node["id"]]
            if self.run
            else []
        )
        index = len(self.attempts) - 1 if self.attempts else 0
        if old is not None and not was_latest:
            index = next(
                (
                    i
                    for i, attempt in enumerate(self.attempts)
                    if (attempt.activation_id, attempt.attempt) == (old.activation_id, old.attempt)
                ),
                index,
            )
        self.select_attempt(index)

    def on_screen_resume(self) -> None:
        self.refresh_run()
        self._queue_record_tabs()

    @on(Tabs.TabActivated, "#workflow-iteration-tabs")
    def iteration_changed(self, event: Tabs.TabActivated) -> None:
        event.stop()
        if event.tab.id != event.tabs.active or not self.selected:
            return
        index = int(event.tabs.active.removeprefix("workflow-iteration-"))
        if index >= len(self._iteration_keys):
            return
        activation = self._iteration_keys[index]
        if activation != self.selected.activation_id:
            self.select_attempt(
                max(i for i, attempt in enumerate(self.attempts) if attempt.activation_id == activation)
            )

    @on(Tabs.TabActivated, "#workflow-attempt-tabs")
    def attempt_changed(self, event: Tabs.TabActivated) -> None:
        event.stop()
        if event.tab.id != event.tabs.active or not self.selected:
            return
        index = int(event.tabs.active.removeprefix("workflow-attempt-"))
        if index >= len(self._attempt_keys):
            return
        key = self._attempt_keys[index]
        if key != (self.selected.activation_id, self.selected.attempt):
            self.select_attempt(
                next(i for i, attempt in enumerate(self.attempts) if (attempt.activation_id, attempt.attempt) == key)
            )

    async def _sync_record_tabs(self) -> None:
        """Coalesce live revisions without rebuilding the shared detail panes or transcript."""
        generation, selected, records = self._generation, self.selected, self.attempts
        try:
            if not self.is_mounted or self._torn_down() or self.app.screen is not self:
                return
            iterations = self.query_one("#workflow-iteration-tabs", Tabs)
            attempts = self.query_one("#workflow-attempt-tabs", Tabs)
            # Adding a bar's first tab activates it once the tab is mounted, before add_tab() returns.
            with messages_disabled(Tabs.TabActivated, iterations, attempts):
                iteration_keys = tuple(dict.fromkeys(attempt.activation_id for attempt in records))
                if iteration_keys != self._iteration_keys:
                    self._iteration_keys = iteration_keys
                    await iterations.clear()
                    if self._torn_down():
                        return
                    for index, activation in enumerate(iteration_keys):
                        number = next(attempt.iteration for attempt in records if attempt.activation_id == activation)
                        await iterations.add_tab(
                            Tab(
                                Text(text.render(text.ITERATION.bind(iteration=number), self.locale_controller)),
                                id=f"workflow-iteration-{index}",
                            )
                        )
                        if self._torn_down():
                            return
                if selected:
                    iterations.active = f"workflow-iteration-{iteration_keys.index(selected.activation_id)}"
                iterations.display = len(iteration_keys) > 1

                attempt_keys = tuple(
                    (attempt.activation_id, attempt.attempt)
                    for attempt in records
                    if selected and attempt.activation_id == selected.activation_id
                )
                if attempt_keys != self._attempt_keys:
                    self._attempt_keys = attempt_keys
                    await attempts.clear()
                    if self._torn_down():
                        return
                    for index, (_, number) in enumerate(attempt_keys):
                        await attempts.add_tab(
                            Tab(
                                Text(text.render(text.ATTEMPT.bind(attempt=number), self.locale_controller)),
                                id=f"workflow-attempt-{index}",
                            )
                        )
                        if self._torn_down():
                            return
                if selected:
                    key = (selected.activation_id, selected.attempt)
                    attempts.active = f"workflow-attempt-{attempt_keys.index(key)}"
                attempts.display = len(attempt_keys) > 1
        except ValueError:
            # Tabs.add_tab validates its active ID after awaiting mount; screen
            # teardown may already have rejected or removed that tab.
            if not self._torn_down():
                raise
        finally:
            self._tabs_sync_pending = False
            if generation != self._generation and self.is_mounted and not self._torn_down():
                self._queue_record_tabs()

    def _torn_down(self) -> bool:
        """App shutdown and removal both start, and drop new mounts, before the screen is detached."""
        return not self.is_attached or not self.app.is_running or self._closing or self._pruning

    def _queue_record_tabs(self) -> None:
        if not self._tabs_sync_pending:
            self._tabs_sync_pending = self.call_later(self._sync_record_tabs)

    def select_attempt(self, index: int) -> None:
        self.selected = self.attempts[index] if self.attempts else None
        state = self.selected.state if self.selected else "pending"
        self.query_one("#workflow-node").border_subtitle = Text(text.state_label(state, self.locale_controller))
        self._generation += 1
        self._queue_record_tabs()
        self._update_controls()
        self.load_records(self._generation, self.selected)

    def _awaiting_retry(self) -> bool:
        return bool(
            self.run
            and not self.run.finished
            and self.selected
            and self.selected.state == "awaiting_retry"
            and self.run.nodes.get(self.node["id"]) == self.selected
        )

    def _update_controls(self) -> None:
        self.query_one("#workflow-node-actions").display = self._awaiting_retry()
        self.query_one("#workflow-node-retry", Button).disabled = bool(
            self.selected and self.retry_pending(self.selected)
        )

    def _read(self, attempt: WorkflowNodeStateChanged | None) -> tuple[NodeRecords | None, ShownValue | None, str]:
        records = (
            read_node_records(self.directory, attempt.activation_id, attempt.attempt, node_kind=self.node["kind"])
            if self.directory and attempt
            else None
        )
        with self._previous_lock:
            if self._previous_result is None:
                try:
                    self._previous_result = (self._read_previous_output(), "")
                except (OSError, ValueError, KeyError) as exc:
                    self._previous_result = (None, str(exc))
            previous, error = self._previous_result
        return records, previous, error

    def _read_previous_output(self) -> ShownValue | None:
        if self.run is None:
            return None
        history = self.history()
        current_id = self.run.started.run_id
        index = next((i for i, record in enumerate(history) if record.directory.name == current_id), -1)
        previous = history[index + 1] if 0 <= index < len(history) - 1 else None
        if previous is not None:
            output = latest_node_output(previous.directory, self.node["id"], node_kind=self.node["kind"])
            if output is not None:
                return shown_value(output.get("value"))
        return None

    @work(exclusive=True, group="workflow-node-records")
    async def load_records(self, generation: int, attempt: WorkflowNodeStateChanged | None) -> None:
        read_error = ""
        try:
            records, previous, read_error = await asyncio.to_thread(self._read, attempt)
        except (OSError, ValueError, KeyError) as exc:
            # Clear the previous attempt's values and switch its transcript even
            # when this attempt's artifacts cannot be read.
            records, previous, read_error = None, None, str(exc)
        if generation != self._generation or not self.is_mounted or self._torn_down():
            return
        empty = text.render(text.NO_RECORD.bind(), self.locale_controller)
        input_document = output_document = ValueDocument(placeholder=empty)
        if records is not None:
            if records.input is not None:
                notice = (
                    text.render(text.DATA_DROPPED.bind(), self.locale_controller)
                    if records.input.get(DATA_DROPPED_KEY)
                    else ""
                )
                input_document = ValueDocument(record_values(records.input), notice=notice, placeholder=empty)
            output_document = ValueDocument(
                record_values(records.output) if records.output is not None else (),
                emits=tuple(records.emits),
                placeholder=empty,
            )
        previous_output = self.query_one("#workflow-node-previous", WorkflowValueView)
        previous_output.display = previous is not None
        for view, document in (
            (self.query_one("#workflow-node-input", WorkflowValueView), input_document),
            (self.query_one("#workflow-node-output", WorkflowValueView), output_document),
            (previous_output, ValueDocument((previous,) if previous is not None else ())),
        ):
            await view.show(document)
            if generation != self._generation or not self.is_mounted or self._torn_down():
                return
        error = attempt.error if attempt else ""
        if self.node["kind"] == "agent":
            # Agent failures already appear in their transcript; this area retains
            # record errors and failures with no transcript to display them.
            error = await self._load_transcript(generation, attempt)
        diagnostics = records.diagnostics or {} if records else {}
        sections: list[str] = []
        for entry in diagnostics.get("phases", []):
            label = text.phase_label(entry["phase"], self.locale_controller)
            if entry["iteration"]:
                label += " · " + text.render(text.ITERATION.bind(iteration=entry["iteration"]), self.locale_controller)
            parts = [label, entry["stdout"]["text"], entry["traceback"]]
            if entry["stdout"]["truncated"]:
                parts.append(text.render(text.STDOUT_TRUNCATED.bind(), self.locale_controller))
            sections.append("\n".join(part for part in parts if part))
        if records is not None and records.stderr_path is not None:
            sections.append(text.render(text.ACP_STDERR.bind(path=str(records.stderr_path)), self.locale_controller))
        diagnostics_text = "\n\n".join(sections)
        diagnostics_widget = self.query_one("#workflow-node-diagnostics", Static)
        diagnostics_widget.update(Text(diagnostics_text))
        diagnostics_widget.display = bool(diagnostics_text)
        if attempt is not None and attempt.failure_phase:
            phase = text.phase_label(attempt.failure_phase, self.locale_controller)
            error = text.render(text.FAILED_PHASE.bind(phase=phase), self.locale_controller) + "\n" + error
        if generation == self._generation and self.is_mounted and not self._torn_down():
            errors = "\n\n".join(part for part in (error, read_error) if part)
            self.query_one("#workflow-node-error", Static).update(Text(errors))
            self.query_one("#workflow-node-errors").display = bool(errors)

    async def _load_transcript(self, generation: int, attempt: WorkflowNodeStateChanged | None) -> str:
        invocation = attempt.invocation_id if attempt else ""
        journal = self.run.journals.get(invocation) if self.run else None
        # Only the current attempt follows live events. Older attempts share an
        # invocation identity, but must display their own archived boundary.
        live = bool(journal and self.run and self.run.nodes.get(self.node["id"]) == attempt)
        identity = (invocation, None if live else attempt.attempt if attempt else 0)
        if self._transcript_id == identity:
            return ""
        archived = None
        transcript_error = ""
        if not live and self.directory is not None and attempt is not None:
            try:
                archived = await asyncio.to_thread(
                    read_node_transcript, self.directory, attempt.activation_id, attempt.attempt
                )
            except (OSError, ValueError, KeyError) as exc:
                transcript_error = str(exc)
        async with self._transcript_lock:
            if generation != self._generation or not self.is_mounted or self._torn_down():
                return ""
            container = self.query_one("#workflow-node-transcript", Vertical)
            # A cancelled load must not leave the removed transcript's identity behind.
            self._transcript_id = None
            await remove_children_shielded(container)
            if generation != self._generation or not self.is_mounted or self._torn_down():
                return ""
            if live and journal is not None:
                if self.agent_spec is None:
                    raise RuntimeError("A live agent transcript requires an agent specification.")
                surface = AgentTranscriptSurface(journal, profile_name=self.agent_spec.profile)
                await container.mount(surface)
            elif archived is not None and attempt is not None:
                tail: list[AgentTranscriptOp] = []
                if attempt.state in {"failed", "retrying", "awaiting_retry"}:
                    tail.append(TranscriptErrorOp(archived.error or attempt.error or text.FAILED.bind()))
                elif attempt.state == "cancelled" or archived.status in {"cancelled", "orphaned"}:
                    tail.append(TranscriptInterruptedOp(text.CANCELLED.bind()))
                await container.mount(
                    AgentTranscriptSurface(
                        AgentTranscriptJournal(),
                        persisted_replay=archived.replay,
                        replay_tail=tuple(tail),
                    )
                )
            else:
                transcript_error = "\n\n".join(
                    part for part in (attempt.error if attempt else "", transcript_error) if part
                )
                await container.mount(Static(Text(text.render(text.NO_TRANSCRIPT.bind(), self.locale_controller))))
            self._transcript_id = identity if live or archived is not None else None
            return transcript_error

    @on(Button.Pressed, "#workflow-node-retry")
    def retry_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        attempt = self.selected
        if self._awaiting_retry() and attempt is not None and self.retry(attempt):
            self._update_controls()

    def _before_dismiss(self, _result: object | None = None) -> None:
        """Cancel record continuations before the asynchronous pop removes their widgets."""
        if self.is_attached:
            self.workers.cancel_group(self, "workflow-node-records")
        super()._before_dismiss(_result)

    def action_close(self) -> None:
        if self.app.screen is self:
            self.dismiss(None)

    @on(ToolViewRequested)
    def tool_view_requested(self, event: ToolViewRequested) -> None:
        event.stop()
        self.app.push_screen(ToolDetailModal.from_request(event))
