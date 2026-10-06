# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared workflow information for trust review, loaded previews and recorded runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich import box
from rich.table import Table
from rich.text import Text
from textual import on
from textual.containers import Grid, Horizontal, VerticalGroup
from textual.reactive import reactive
from textual.widgets import Collapsible, Static

from chrys.app.tui.widgets.workflow import text
from chrys.app.tui.widgets.workflow.graph import node_detail
from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.i18n import MessageDef, msg
from chrys.service.workflows.environment import DefaultPlan
from chrys.service.workflows.graph import manifest_warnings

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController
    from chrys.foundation.events.types import WorkflowRunStarted
    from chrys.orchestration.workflows.preview import WorkflowInspection, WorkflowPreview
    from chrys.service.workflows.discovery import WorkflowSource

INFO_TAB = msg("tui.workflow.confirm.info_tab", fallback="Info")
_ENVIRONMENT = msg("tui.workflow.confirm.environment", fallback="Execution Environment")
_PYTHON = msg("tui.workflow.confirm.python", fallback="Python")
_MODE = msg("tui.workflow.confirm.mode", fallback="Environment")
_DEFAULT_ENVIRONMENT = msg("tui.workflow.confirm.default_environment", fallback="{app} environment")
_CUSTOM_ENVIRONMENT = msg("tui.workflow.confirm.custom_environment", fallback="Custom interpreter")
_INTERPRETER = msg("tui.workflow.confirm.interpreter", fallback="Interpreter Path")
_NODES = msg("tui.workflow.confirm.nodes", fallback="Nodes · {total}")
_NODE = msg("tui.workflow.confirm.node", fallback="Node")
_KIND = msg("tui.workflow.confirm.kind", fallback="Type")
_CONFIGURATION = msg("tui.workflow.confirm.configuration", fallback="Configuration")
_AGENT = msg("tui.workflow.confirm.agent", fallback="Agent")
_JOIN = msg("tui.workflow.confirm.join", fallback="Merge")
_LOOP = msg("tui.workflow.confirm.loop", fallback="Loop")
_LOOP_LIMIT = msg(
    "tui.workflow.confirm.loop_limit", fallback="Up to {count} iteration", plural_fallback="Up to {count} iterations"
)
_WARNINGS = msg("tui.workflow.confirm.warnings", fallback="Warnings")
_FINGERPRINTS = msg("tui.workflow.confirm.fingerprints", fallback="Verification Details")
_FINGERPRINT_HINT = msg(
    "tui.workflow.confirm.fingerprint_hint",
    fallback="These fingerprints identify the exact source and workflow definition being trusted.",
)
_ENTRY_DIGEST = msg("tui.workflow.confirm.entry_digest", fallback="Source SHA-256")
_SPEC_DIGEST = msg("tui.workflow.confirm.spec_digest", fallback="Workflow SHA-256")
_PACKAGE_FILES = msg("tui.workflow.info.package_files", fallback="Files")
_EXECUTION_CONSENT = msg(
    "tui.workflow.confirm.execution_consent",
    fallback="Trust allows this file and its selected interpreter to run with your permissions, including during preview. "
    "Review Source before continuing. Cancel does not load the workflow.",
)
_PACKAGE_EXECUTION_CONSENT = msg(
    "tui.workflow.confirm.package_execution_consent",
    fallback="Trust allows this workflow's folder and its selected interpreter to run with your permissions, "
    "including during preview. Source shows only the entry file: review the folder's other files before continuing. "
    "Cancel does not load the workflow.",
)
_INSPECTION_HINT = msg(
    "tui.workflow.confirm.inspection_hint",
    fallback="The source has not been executed. Nodes and runtime details are available after trusting and loading it.",
)

_NODE_KINDS = {"python": _PYTHON, "agent": _AGENT, "join": _JOIN, "loop": _LOOP}


@dataclass(frozen=True, slots=True)
class WorkflowInfoData:
    """Display facts only; constructing or rendering this never executes a workflow."""

    title: str
    source_kind: str
    canonical_path: str
    manifest: dict[str, Any] = field(default_factory=dict)
    resolved_nodes: tuple[dict[str, Any], ...] = ()
    python_version: str = ""
    environment_mode: str = ""
    interpreter: str = ""
    source_digest: str = ""
    spec_digest: str = ""
    requires_trust: bool = False
    run_id: str = ""
    package_files: int | None = None
    """How many files a workflow folder's confirmation covers; ``None`` for a single file."""

    @classmethod
    def from_inspection(cls, inspection: WorkflowInspection) -> WorkflowInfoData:
        prepared = inspection.prepared_environment
        return cls(
            title=inspection.source.workflow_id,
            source_kind=inspection.source.source_kind,
            canonical_path=inspection.source.canonical_path,
            python_version=prepared.python_version if prepared else "",
            environment_mode=(
                prepared.mode if prepared else "default" if isinstance(inspection.environment, DefaultPlan) else "byo"
            ),
            interpreter=prepared.executable if prepared else inspection.environment.interpreter,
            source_digest=inspection.source.source_digest,
            requires_trust=True,
            package_files=_package_files(inspection.source),
        )

    @classmethod
    def from_preview(cls, preview: WorkflowPreview, resolved_nodes: Sequence[dict[str, Any]] = ()) -> WorkflowInfoData:
        return cls(
            title=preview.title or preview.source.workflow_id,
            source_kind=preview.source.source_kind,
            canonical_path=preview.source.canonical_path,
            manifest=preview.manifest,
            resolved_nodes=tuple(resolved_nodes),
            python_version=preview.environment.python_version,
            environment_mode=preview.environment.mode,
            interpreter=preview.environment.executable,
            source_digest=preview.source.source_digest,
            spec_digest=preview.spec_digest,
            package_files=_package_files(preview.source),
        )

    @classmethod
    def from_run(cls, started: WorkflowRunStarted) -> WorkflowInfoData:
        return cls(
            run_id=started.run_id,
            title=started.title or started.workflow_id,
            source_kind=started.source_kind,
            canonical_path=started.canonical_path,
            manifest=started.manifest,
            resolved_nodes=tuple(started.resolved_nodes),
            spec_digest=started.spec_digest,
        )


class WorkflowInfo(VerticalGroup):
    """The same readable, localized information in the modal and the main Info tab."""

    DEFAULT_CSS = """
    WorkflowInfo { padding: 1 1 0 1; }
    WorkflowInfo Static { height: auto; }
    WorkflowInfo #workflow-info-identity { height: auto; }
    WorkflowInfo #workflow-info-title {
        width: auto; max-width: 70%; color: $primary; text-style: bold;
    }
    WorkflowInfo #workflow-info-origin {
        width: auto; margin-left: 1; padding: 0 1;
        color: $secondary; background: $secondary 15%;
    }
    WorkflowInfo #workflow-info-description { margin-top: 1; }
    WorkflowInfo #workflow-info-path { margin-top: 1; color: $text-muted; }
    WorkflowInfo .workflow-info-heading {
        margin-top: 1; margin-bottom: 1; color: $secondary; text-style: bold;
    }
    WorkflowInfo .workflow-info-fields {
        height: auto; grid-size: 2; grid-columns: 18 1fr; grid-rows: auto; grid-gutter: 0 2;
    }
    WorkflowInfo .workflow-info-field-label { color: $text-muted; }
    WorkflowInfo .workflow-info-field-value { width: 1fr; }
    WorkflowInfo .workflow-manifest-warning { color: $warning; }
    WorkflowInfo #workflow-info-nodes { width: 1fr; }
    WorkflowInfo #workflow-info-nodes-heading { margin: 1 0 0 0; }
    WorkflowInfo #workflow-info-verification {
        height: auto; margin-top: 1; padding: 0; border: none; background: transparent;
    }
    WorkflowInfo #workflow-info-verification > CollapsibleTitle {
        color: $text-muted; padding: 0;
    }
    WorkflowInfo #workflow-info-verification > Contents { padding: 1 0 1 2; }
    WorkflowInfo #workflow-info-fingerprint-hint { margin-bottom: 1; color: $text-muted; }
    """

    data: reactive[WorkflowInfoData | None] = reactive(None, recompose=True)

    def __init__(
        self, data: WorkflowInfoData | None = None, *, locale_controller: LocaleController | None = None
    ) -> None:
        super().__init__()
        self._verification_collapsed = False
        self.data = data
        self.locale_controller = locale_controller

    def _label(self, message: MessageDef) -> str:
        return text.render(message.bind(), self.locale_controller)

    def refresh_localization(self) -> None:
        self.refresh(recompose=True)

    def watch_data(self, previous: WorkflowInfoData | None, current: WorkflowInfoData | None) -> None:
        if (
            previous is None
            or current is None
            or (previous.canonical_path, previous.run_id)
            != (
                current.canonical_path,
                current.run_id,
            )
        ):
            self._verification_collapsed = False

    @on(Collapsible.Toggled, "#workflow-info-verification")
    def _verification_toggled(self, event: Collapsible.Toggled) -> None:
        self._verification_collapsed = event.collapsible.collapsed

    def _fields(self, fields: Sequence[tuple[MessageDef, str]], *, widget_id: str | None = None) -> Grid:
        cells = []
        for label, value in fields:
            cells.extend(
                (
                    Static(Text(self._label(label)), classes="workflow-info-field-label"),
                    Static(Text(value), classes="workflow-info-field-value"),
                )
            )
        return Grid(*cells, classes="workflow-info-fields", id=widget_id)

    def _node_configuration(self, node: dict[str, Any], resolved: dict[str, Any] | None) -> str:
        if node["kind"] == "loop":
            return text.render(_LOOP_LIMIT.bind(count=node["loop"]["max_iterations"]), self.locale_controller)
        if node["kind"] in {"python", "join"}:
            return (node.get("callable") or {}).get("name", "")
        return node_detail(node, resolved, locale=self.locale_controller)

    def _node_table(self, data: WorkflowInfoData) -> Table:
        table = Table(box=box.SIMPLE_HEAD, padding=(0, 1), pad_edge=False, header_style="bold", border_style="dim")
        table.add_column(Text(self._label(_NODE)), overflow="fold")
        table.add_column(Text(self._label(_KIND)), no_wrap=True)
        table.add_column(Text(self._label(_CONFIGURATION)), overflow="fold")
        resolved = {node["node_id"]: node for node in data.resolved_nodes}
        for node in data.manifest.get("nodes", []):
            kind = _NODE_KINDS.get(node["kind"])
            table.add_row(
                Text(node["id"], style="bold"),
                Text(self._label(kind) if kind is not None else node["kind"], style="dim"),
                Text(self._node_configuration(node, resolved.get(node["id"]))),
            )
        return table

    def compose(self) -> ComposeResult:
        data = self.data
        if data is None:
            return
        with Horizontal(id="workflow-info-identity"):
            yield Static(Text(text.shown(data.title)), id="workflow-info-title")
            source = text.SOURCES.get(data.source_kind)
            yield Static(
                Text(self._label(source) if source is not None else data.source_kind), id="workflow-info-origin"
            )
        consent = _EXECUTION_CONSENT if data.package_files is None else _PACKAGE_EXECUTION_CONSENT
        description = self._label(consent) if data.requires_trust else data.manifest.get("description")
        if description:
            yield Static(Text(description), id="workflow-info-description")
        yield Static(Text(text.shown(data.canonical_path)), id="workflow-info-path")
        if data.package_files is not None:
            yield self._fields([(_PACKAGE_FILES, str(data.package_files))], widget_id="workflow-info-package")
        warnings = manifest_warnings(data.manifest)
        if warnings:
            yield Static(Text(self._label(_WARNINGS)), classes="workflow-info-heading")
            for warning in warnings:
                yield Static(Text(f"⚠ {warning.message}"), classes="workflow-manifest-warning")
        fields = []
        if data.python_version:
            fields.append((_PYTHON, data.python_version))
        if data.environment_mode:
            fields.append(
                (
                    _MODE,
                    text.render(_DEFAULT_ENVIRONMENT.bind(app=APP_DISPLAY_NAME), self.locale_controller)
                    if data.environment_mode == "default"
                    else self._label(_CUSTOM_ENVIRONMENT),
                )
            )
        if data.interpreter:
            fields.append((_INTERPRETER, data.interpreter))
        if fields:
            yield Static(Text(self._label(_ENVIRONMENT)), classes="workflow-info-heading")
            yield self._fields(fields)
        if data.requires_trust:
            yield Static(Text(self._label(_INSPECTION_HINT)), classes="workflow-info-heading")
        else:
            yield Static(
                Text(text.render(_NODES.bind(total=len(data.manifest.get("nodes", []))), self.locale_controller)),
                classes="workflow-info-heading",
                id="workflow-info-nodes-heading",
            )
            yield Static(self._node_table(data), id="workflow-info-nodes")
        fingerprints = []
        if data.source_digest:
            fingerprints.append((_ENTRY_DIGEST, data.source_digest))
        if data.spec_digest:
            fingerprints.append((_SPEC_DIGEST, data.spec_digest))
        if fingerprints:
            with Collapsible(
                title=self._label(_FINGERPRINTS),
                id="workflow-info-verification",
                collapsed=self._verification_collapsed,
            ):
                yield Static(Text(self._label(_FINGERPRINT_HINT)), id="workflow-info-fingerprint-hint")
                yield self._fields(fingerprints)


def _package_files(source: WorkflowSource) -> int | None:
    return source.package.file_count if source.package is not None else None
