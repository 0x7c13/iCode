# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow tab content loading, fenced to the session and view that requested it."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chrys.app.tui.widgets.workflow import text
from chrys.app.tui.widgets.workflow.output import WorkflowOutputText
from chrys.app.tui.widgets.workflow.records import stored_output_value
from chrys.service.workflows.artifacts import read_node_output, read_run_source
from chrys.service.workflows.discovery import read_entry_bytes
from chrys.service.workflows.layout import run_dir
from chrys.service.workflows.store import read_run_header, read_run_output, read_run_spec

if TYPE_CHECKING:
    from chrys.app.tui.i18n import LocaleController
    from chrys.app.tui.widgets.workflow.info import WorkflowInfoData
    from chrys.app.tui.widgets.workflow.panel import WorkflowPanel
    from chrys.app.tui.widgets.workflow.projector import ObservedRun
    from chrys.orchestration.workflows.preview import WorkflowPreview


@dataclass(frozen=True, slots=True)
class _View:
    directory: Path | None
    run_id: str
    preview: WorkflowPreview | None


class WorkflowContent:
    def __init__(
        self,
        *,
        panel: WorkflowPanel,
        session_dir: Callable[[], Path | None],
        spawn: Callable[[Coroutine[Any, Any, None]], asyncio.Task[None]],
        refresh: Callable[[], None],
        show_error: Callable[[str], None],
        check_preview: Callable[[], object],
        locale: LocaleController | None,
    ) -> None:
        self._panel = panel
        self._session_dir = session_dir
        self._spawn = spawn
        self._refresh = refresh
        self._show_error = show_error
        self._check_preview = check_preview
        self._locale = locale
        self._source_generation = 0
        self._source_error: tuple[_View, str] | None = None
        self._output_run: ObservedRun | None = None
        self._outputs: tuple[WorkflowOutputText, ...] | None = None
        self._info_view: _View | None = None
        self._info_data: WorkflowInfoData | None = None
        self._closed = False

    def reset(self) -> None:
        """Release cached content and invalidate reads from the previous view."""
        self._source_generation += 1
        self._source_error = None
        self._output_run = None
        self._outputs = None
        self._info_view = None
        self._info_data = None

    def close(self) -> None:
        self._closed = True
        self.reset()

    def invalidate_source(self) -> None:
        self._source_generation += 1

    def _view(self) -> _View:
        panel = self._panel
        # Saved artifacts belong to the selected run, regardless of the next-run preview.
        return _View(self._session_dir(), panel.run_id, None if panel.run_id else panel.preview)

    def _is_current(self, view: _View) -> bool:
        panel, current = self._panel, self._view()
        return (
            not self._closed
            and panel.is_attached
            and view.directory == self._session_dir()
            and view.run_id == panel.run_id
            and view.preview is current.preview
        )

    def view_changed(self) -> None:
        self.invalidate_source()
        panel = self._panel
        if panel.code_visible and not (panel.empty or panel.previewing):
            self._spawn(self._load_source(self._view(), self._source_generation))
        self._show_info()

    def _show_info(self) -> None:
        panel = self._panel
        if not panel.info_visible or panel.empty or panel.previewing or not panel.run_id:
            return
        view = self._view()
        data = panel.info_data
        if data is None or data.run_id != view.run_id:
            return
        if view != self._info_view:
            self._info_view = view
            self._info_data = None
            self._spawn(self._load_info(view, data))
        elif self._info_data is not None:
            panel.info_data = self._info_data

    async def _load_info(self, view: _View, data: WorkflowInfoData) -> None:
        """Supplement the run's manifest with its saved interpreter and source digest."""

        def read() -> WorkflowInfoData:
            if view.directory is None:
                raise ValueError(text.render(text.NO_RECORD.bind(), self._locale))
            directory = run_dir(view.directory, view.run_id)
            header, spec = read_run_header(directory), read_run_spec(directory)
            environment = spec.get("environment", {})
            if not isinstance(environment, dict):
                raise ValueError("Invalid workflow environment record.")
            fields = {
                "source_digest": header.get("entry_digest", ""),
                "python_version": environment.get("python_version", ""),
                "environment_mode": environment.get("mode", ""),
                "interpreter": environment.get("executable", ""),
            }
            if not all(isinstance(value, str) for value in fields.values()):
                raise ValueError("Invalid workflow environment or source digest record.")
            return replace(data, **fields)

        try:
            result = await asyncio.to_thread(read)
        except (OSError, ValueError) as exc:
            if self._info_view == view and self._is_current(view) and self._panel.info_visible:
                self._show_error(str(exc))
            return
        if self._info_view == view and self._is_current(view):
            self._info_data = result
            self._refresh()

    async def _load_source(self, view: _View, generation: int) -> None:
        if view.preview is None and not view.run_id:
            return

        def read() -> bytes:
            if view.run_id:
                if view.directory is None:
                    raise ValueError(text.render(text.NO_RECORD.bind(), self._locale))
                return read_run_source(run_dir(view.directory, view.run_id))
            if view.preview is None:
                raise RuntimeError("Reading workflow source requires a run or a preview.")
            source = view.preview.source
            return read_entry_bytes(Path(source.canonical_path), source.source_kind)

        def current() -> bool:
            panel = self._panel
            return (
                generation == self._source_generation
                and self._is_current(view)
                and not (panel.empty or panel.previewing)
                and panel.code_visible
            )

        try:
            source = await asyncio.to_thread(read)
        except (OSError, ValueError) as exc:
            if current() and (view, str(exc)) != self._source_error:
                self._source_error = (view, str(exc))
                self._show_error(str(exc))
            return
        if current():
            self._check_preview()
            self._source_error = None
            self._panel.show_code(source, differs=view.preview is not None and source != view.preview.source.source)
            self._refresh()

    def project(self, run: ObservedRun | None) -> None:
        """Load a terminal result once and paint only the matching run's cached content."""
        self._show_info()
        if run is None or run.finished is None:
            return
        if self._output_run is not run:
            self._output_run = run
            self._outputs = None
            self._spawn(self._load_outputs(run, self._view()))
        elif self._outputs is not None:
            self._panel.show_outputs(self._outputs)

    async def _load_outputs(self, run: ObservedRun, view: _View) -> None:
        if run.finished is None:
            raise RuntimeError("Loading workflow outputs requires a finished run.")
        # Snapshot identities before crossing into a worker thread; live dictionaries
        # belong to the main pump even when this particular run has finished.
        outputs = tuple(run.finished.outputs)
        kinds = {node["id"]: node["kind"] for node in run.started.manifest.get("nodes", [])}

        def read() -> tuple[WorkflowOutputText, ...]:
            directory = run_dir(view.directory, view.run_id) if view.directory else None
            parts: list[WorkflowOutputText] = []
            for output in outputs:
                try:
                    value = (
                        read_node_output(
                            directory, output.activation_id, output.attempt, node_kind=kinds.get(output.node_id, "")
                        )
                        if directory
                        else None
                    )
                    full_text = output.summary_text if value is None else stored_output_value(value)["text"]
                except (OSError, ValueError) as exc:
                    # A damaged result must not hide the other output nodes.
                    parts.append(WorkflowOutputText(output.node_id, str(exc)))
                else:
                    parts.append(WorkflowOutputText(output.node_id, full_text, summary_only=value is None))
            if directory is not None:
                try:
                    diagnostics = read_run_output(directory) or {}
                    for key, label in (("load", text.LOAD_OUTPUT), ("native", text.NATIVE_OUTPUT)):
                        captured = diagnostics.get(key, {})
                        captured_text = captured.get("text", "")
                        dropped = captured.get("dropped_bytes", 0)
                        if captured.get("truncated") or dropped:
                            captured_text += "\n" + text.render(text.OUTPUT_TRUNCATED.bind(), self._locale)
                        if captured_text:
                            parts.append(WorkflowOutputText(text.render(label.bind(), self._locale), captured_text))
                    if diagnostics.get("error"):
                        parts.append(
                            WorkflowOutputText(
                                text.render(text.NATIVE_OUTPUT.bind(), self._locale), diagnostics["error"]
                            )
                        )
                except (OSError, ValueError) as exc:
                    parts.append(WorkflowOutputText(text.render(text.NATIVE_OUTPUT.bind(), self._locale), str(exc)))
            return tuple(parts)

        try:
            results = await asyncio.to_thread(read)
        except (OSError, ValueError) as exc:
            results = (WorkflowOutputText("", str(exc)),)
        if self._output_run is run and self._is_current(view):
            self._outputs = results
            self._refresh()
