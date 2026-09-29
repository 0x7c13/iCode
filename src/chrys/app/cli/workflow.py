# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Headless ``chrys workflow`` commands: list the workflows a session would find, and run one to its outputs."""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rich.console import Console
from rich.table import Table
from rich.text import Text

from chrys.app.cli import headless
from chrys.app.cli.headless import PreparedRuntime
from chrys.app.cli.progress import ProgressWriter, WorkflowProgress, guarded, progress_console
from chrys.app.parsing import SanitizingArgumentParser
from chrys.foundation.branding import APP_COMMAND, APP_DISPLAY_NAME
from chrys.foundation.config.settings import DEFAULT_AGENT_PROFILE
from chrys.foundation.events.types import WorkflowRunAccepted
from chrys.foundation.i18n.formatting import sanitize_terminal_block
from chrys.foundation.models.session_surface import SessionSurface
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import surrogate_safe_text
from chrys.orchestration.session_host import (
    AgentProfileNotFoundError,
    AmbiguousSessionIdError,
    ChrysSessionHost,
    SessionNotFoundError,
    WorkflowRunRejectedError,
    WorkflowRunTimeoutError,
)
from chrys.orchestration.workflows.catalog import WorkflowCatalog, WorkflowNotFoundError
from chrys.orchestration.workflows.coordinator import REJECT_NOT_CONFIRMED
from chrys.orchestration.workflows.preview import WorkflowPreviewError
from chrys.orchestration.workflows.runner import WorkflowRunResult
from chrys.service.approval.policy import ApprovalMode
from chrys.service.workflows.discovery import (
    SOURCE_KIND_BUILTIN,
    Discovery,
)
from chrys.service.workflows.layout import run_dir
from chrys.service.workflows.outcomes import REASON_DEADLINE_EXCEEDED, RunOutcome
from chrys.service.workflows.store import read_run_output

EXIT_TIMEOUT: Final = 124
EXIT_INTERRUPTED: Final = 130


def build_parser() -> argparse.ArgumentParser:
    """Build the ``chrys workflow`` parser: ``list`` and ``run`` sub-commands."""
    parser = SanitizingArgumentParser(
        prog=f"{APP_COMMAND} workflow",
        description=f"List {APP_DISPLAY_NAME} workflows and run one headlessly until its outputs are ready.",
        add_help=False,
    )
    _add_help(parser)
    commands = parser.add_subparsers(
        dest="command", metavar="command", required=True, parser_class=SanitizingArgumentParser
    )
    list_parser = commands.add_parser(
        "list",
        help="List the workflows found for the current directory",
        description="List the workflows found for the current directory: builtin templates, the user's global "
        "directory, and the project's .chrys/workflows directory (a project file shadows a global one, which "
        "shadows a builtin).",
        add_help=False,
    )
    _add_help(list_parser)
    list_parser.add_argument("--json", action="store_true", help="Emit JSON output")
    run_parser = commands.add_parser(
        "run",
        help="Run one workflow until it finishes and print its outputs",
        description="Run one workflow until it finishes and print its outputs. A user workflow file must have "
        "been confirmed before; --trust confirms the file as it is now.",
        add_help=False,
    )
    _add_help(run_parser)
    run_parser.add_argument(
        "workflow_id", help=f"Workflow id (the file name without .py), as listed by '{APP_COMMAND} workflow list'"
    )
    run_parser.add_argument(
        "--input", metavar="TEXT", default="", help="Input text handed to the workflow's start node (default: empty)"
    )
    run_parser.add_argument("-s", "--session", default=None, help="Optional session id to restore before running")
    run_parser.add_argument(
        "--trust",
        action="store_true",
        help="Confirm the workflow file as it is now (source, topology and environment) instead of requiring "
        "an earlier confirmation; builtin templates need no confirmation",
    )
    run_parser.add_argument(
        "--timeout",
        metavar="SECONDS",
        type=float,
        default=None,
        help="Limit workflow preview, loading and execution to this many seconds (exit code 124)",
    )
    run_parser.add_argument("--json", action="store_true", help="Emit JSON output")
    run_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help=(
            "Do not show progress on stderr; only warnings, errors, captured load output and output outside nodes,"
            " and the outputs are printed"
        ),
    )
    return parser


def _add_help(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit"
    )


@dataclass(frozen=True, slots=True)
class _WorkflowRow:
    workflow_id: str
    source_kind: str
    path: str
    title: str


def _rows(catalog: WorkflowCatalog, discovery: Discovery) -> list[_WorkflowRow]:
    """List builtin and confirmed user titles without executing workflow code."""
    ledger = catalog.ledger()
    return [
        _WorkflowRow(
            source.workflow_id, source.source_kind, source.canonical_path, catalog.title(source, ledger=ledger) or ""
        )
        for source in discovery.sources
    ]


def _list_command(args: argparse.Namespace) -> int:
    prepared = headless.prepare_runtime()
    headless.write_pending_warnings(prepared, as_json=args.json)
    catalog = WorkflowCatalog(config_dir=get_platform().config_dir, project_cwd=Path.cwd())
    discovery = catalog.discover()
    for skipped in discovery.skipped:
        headless.write_warning(f"Skipped {skipped.path}: {skipped.reason}", as_json=args.json, code="workflow_skipped")
    rows = _rows(catalog, discovery)
    if args.json:
        headless.write_json(
            {
                "workflows": [
                    {"id": row.workflow_id, "source": row.source_kind, "title": row.title, "path": row.path}
                    for row in rows
                ]
            }
        )
        return 0
    _print_workflows(rows)
    return 0


def _print_workflows(rows: list[_WorkflowRow]) -> None:
    console = Console(file=sys.stdout, highlight=False)
    if not rows:
        console.print(Text("No workflows found."))
        return
    table = Table(box=None, show_edge=False, pad_edge=False)
    table.add_column("ID", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Title", overflow="fold")
    table.add_column("Path", overflow="fold")
    for row in rows:
        table.add_row(
            Text(row.workflow_id),
            Text(row.source_kind),
            Text(row.title or "-"),
            Text(surrogate_safe_text(row.path)),
        )
    console.print(table)


def _workflow_progress(args: argparse.Namespace, runtime: PreparedRuntime) -> WorkflowProgress | None:
    """Progress on stderr (only its warnings for quiet output); nothing for JSON."""
    if args.json:
        return None
    writer = ProgressWriter(
        progress_console(), reported_warnings=headless.reported_warning_keys(runtime), quiet=args.quiet
    )
    return WorkflowProgress(writer, render=runtime.localizer.render)


async def _run_command(args: argparse.Namespace) -> int:
    session_id = (args.session or "").strip()
    runtime = headless.prepare_runtime(restoring_session=bool(session_id))
    headless.write_pending_warnings(runtime, as_json=args.json)
    progress = _workflow_progress(args, runtime)
    loaded = runtime.loaded
    host = ChrysSessionHost(
        profile_name=loaded.settings.default_agent.strip() or DEFAULT_AGENT_PROFILE,
        loaded_settings=loaded,
        approval_mode=ApprovalMode.BYPASS,
        surface=SessionSurface.CLI,
    )
    started = time.monotonic()
    try:
        if session_id:
            await host.load_workflow_session(session_id)
        target = host.workflow_target(args.workflow_id)
        timeout = args.timeout or 0.0
        if progress is not None:
            progress.starting(args.workflow_id, checking=args.trust)
        if args.trust:
            preview_started = time.monotonic()
            preview_deadline = asyncio.timeout(timeout or None)
            try:
                async with preview_deadline:
                    prepared = await host.preview_workflow(target, trust=True)
            except TimeoutError as exc:
                if not preview_deadline.expired():
                    raise
                raise WorkflowRunTimeoutError("Workflow run timed out during preview.") from exc
            if prepared.preview.source.source_kind != SOURCE_KIND_BUILTIN:
                host.confirm_workflow(prepared)
            for warning in prepared.preview.warnings:
                headless.write_warning(warning.message, as_json=args.json, code=warning.code)
                if progress is not None:
                    progress.reported(warning.code, warning.message)
            target = prepared
            if timeout:
                timeout -= time.monotonic() - preview_started
                if timeout <= 0:
                    raise WorkflowRunTimeoutError("Workflow run timed out during preview.")
        run_id = ""
        on_event = None if progress is None else guarded(progress.handle)
        async for event in host.iter_workflow_events(
            target,
            input_text=args.input,
            timeout=timeout,
            include_node_activity=progress is not None,
        ):
            if isinstance(event, WorkflowRunAccepted):
                run_id = event.run_id
            if on_event is not None:
                on_event(event)
        result = host.engine.workflows.result(run_id)
        if result is None:
            error_message = "Workflow run ended without a recorded result."
            raise RuntimeError(error_message)
        diagnostics = None
        if host.workflow_session_dir is not None:
            try:
                diagnostics = await asyncio.to_thread(read_run_output, run_dir(host.workflow_session_dir, run_id))
            except (OSError, ValueError) as exc:
                diagnostics = {"error": str(exc)}
        duration = time.monotonic() - started
        if progress is not None and diagnostics:
            progress.captured(diagnostics)
        if progress is not None and result.outcome is RunOutcome.COMPLETED:
            progress.succeeded(duration=duration)
        return _report(
            result,
            session_id=host.workflow_session_id,
            as_json=args.json,
            duration=duration,
            diagnostics=diagnostics,
        )
    finally:
        await host.shutdown()


def _report(
    result: WorkflowRunResult,
    *,
    session_id: str | None,
    as_json: bool,
    duration: float,
    diagnostics: dict[str, Any] | None = None,
) -> int:
    """Write the outputs (text or JSON), then the outcome when the run did not complete.

    Text mode's ``diagnostics`` were already written to stderr (``WorkflowProgress.captured``); JSON carries them.
    """
    outcome = result.outcome.value
    if as_json:
        headless.write_json(
            {
                "session_id": session_id,
                "run_id": result.run_id,
                "outcome": outcome,
                "reason": result.reason,
                "node_id": result.node_id,
                "error": result.error,
                "duration": round(duration, 3),
                "diagnostics": diagnostics,
                "outputs": [
                    {
                        "node_id": output.node_id,
                        "activation_id": output.activation_id,
                        "text": output.value.text,
                        "data": output.value.data,
                    }
                    for output in result.outputs
                ],
            }
        )
    else:
        text = "\n\n".join(output.value.text for output in result.outputs)
        # Node output can carry model text: neutralize terminal control
        # sequences on a terminal, keep redirected output byte-for-byte.
        if sys.stdout.isatty():
            text = sanitize_terminal_block(text)
        if text:
            sys.stdout.write(text if text.endswith("\n") else f"{text}\n")
    if result.outcome is not RunOutcome.COMPLETED:
        headless.write_error(_outcome_message(result), as_json=as_json, code=outcome, session_id=session_id)
    return _exit_code(result)


def _outcome_message(result: WorkflowRunResult) -> str:
    match result.outcome:
        case RunOutcome.NODE_FAILED:
            message = f"Workflow run failed at node {result.node_id!r}"
        case RunOutcome.LOOP_EXHAUSTED:
            message = f"Workflow loop {result.node_id!r} exhausted its iterations"
        case RunOutcome.CANCELLED if result.reason == REASON_DEADLINE_EXCEEDED:
            message = "Workflow run timed out"
        case RunOutcome.CANCELLED:
            message = f"Workflow run was cancelled ({result.reason})" if result.reason else "Workflow run was cancelled"
        case RunOutcome.WORKER_LOST:
            message = "The workflow worker was lost"
        case RunOutcome.STORAGE_FAILED:
            message = "The workflow run record could not be written"
        case _:
            message = f"Workflow run ended with outcome {result.outcome.value}"
    return f"{message}: {result.error}" if result.error else f"{message}."


def _exit_code(result: WorkflowRunResult) -> int:
    if result.outcome is RunOutcome.COMPLETED:
        return 0
    if result.outcome is RunOutcome.CANCELLED and result.reason == REASON_DEADLINE_EXCEEDED:
        return EXIT_TIMEOUT
    return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``chrys workflow``."""
    headless.configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return _list_command(args)
    if args.timeout is not None and not (math.isfinite(args.timeout) and args.timeout > 0):
        parser.error("--timeout must be a positive number of seconds")
    try:
        return asyncio.run(_run_command(args))
    except WorkflowRunTimeoutError as exc:
        headless.write_error(str(exc), as_json=args.json, code=REASON_DEADLINE_EXCEEDED)
        return EXIT_TIMEOUT
    except WorkflowRunRejectedError as exc:
        message = exc.event.message
        if exc.event.error == REJECT_NOT_CONFIRMED:
            message = f"{message} Pass --trust to confirm it."
        headless.write_error(message, as_json=args.json, code=exc.event.error or "workflow_rejected")
        return 1
    except WorkflowNotFoundError as exc:
        headless.write_error(headless.exception_message(exc), as_json=args.json, code="workflow_not_found")
        return 1
    except WorkflowPreviewError as exc:
        headless.write_error(exc.message, as_json=args.json, code=exc.code)
        return 1
    except SessionNotFoundError as exc:
        headless.write_error(headless.exception_message(exc), as_json=args.json, code="session_not_found")
        return 1
    except AmbiguousSessionIdError as exc:
        headless.write_error(headless.exception_message(exc), as_json=args.json, code="session_ambiguous")
        return 1
    except AgentProfileNotFoundError as exc:
        headless.write_error(headless.exception_message(exc), as_json=args.json, code="profile_not_found")
        return 1
    except KeyboardInterrupt:
        headless.write_error("Interrupted by user.", as_json=args.json, code="interrupted")
        return EXIT_INTERRUPTED
    except Exception as exc:
        headless.write_error(headless.exception_message(exc), as_json=args.json)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
