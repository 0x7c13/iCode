# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Small, read-only workflow summaries for the session browser."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from chrys.foundation.util.time import parse_created_at
from chrys.service.workflows.layout import HEADER_FILE
from chrys.service.workflows.orphans import read_run_terminal
from chrys.service.workflows.outcomes import RunStatus, run_status
from chrys.service.workflows.store import INPUT_EXCERPT_CHARS, read_run_header

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkflowRunMeta:
    run_id: str
    workflow_id: str
    title: str
    status: RunStatus
    updated_at: datetime
    input_text: str = ""


def read_workflow_meta(directory: Path, *, active: bool) -> WorkflowRunMeta | None:
    """Read one run summary, using the log terminal and its timestamp."""
    try:
        header = read_run_header(directory)
        workflow_id = header.get("workflow_id")
        if not isinstance(workflow_id, str) or not workflow_id:
            return None
        title = header.get("title")
        try:
            terminal = read_run_terminal(directory)
        except OSError, ValueError:
            terminal = None
            logger.debug("Unable to read workflow terminal under %s", directory, exc_info=True)
        status = run_status(
            terminal.outcome if terminal else None, terminal.reason if terminal else None, active=active
        )
        modified = datetime.fromtimestamp((directory / HEADER_FILE).stat().st_mtime, UTC)
        started = parse_created_at(terminal.finished_at if terminal else header.get("started_at"))
        if started is not None and started.tzinfo is not None:
            modified = started
        prompt = header.get("input_excerpt")
        return WorkflowRunMeta(
            directory.name,
            workflow_id,
            title if isinstance(title, str) and title else workflow_id,
            status,
            modified,
            prompt[:INPUT_EXCERPT_CHARS] if isinstance(prompt, str) else "",
        )
    except OSError, ValueError, TypeError:
        return None
