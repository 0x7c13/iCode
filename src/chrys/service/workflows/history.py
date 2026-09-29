# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Small, read-only workflow summaries for the session browser."""

from __future__ import annotations

import errno
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from chrys.foundation.util.time import parse_created_at
from chrys.service.workflows.layout import HEADER_FILE
from chrys.service.workflows.orphans import read_run_terminal
from chrys.service.workflows.outcomes import RunStatus, run_status
from chrys.service.workflows.store import INPUT_EXCERPT_CHARS, read_run_header

logger = logging.getLogger(__name__)

_MISSING = (errno.ENOENT, errno.ENOTDIR)


@dataclass(frozen=True, slots=True)
class WorkflowRunMeta:
    run_id: str
    workflow_id: str
    title: str
    status: RunStatus
    updated_at: datetime
    input_text: str = ""


class WorkflowRunRead(NamedTuple):
    meta: WorkflowRunMeta | None
    settled: bool
    """False when an I/O error other than a missing file shaped ``meta``: the same files may read differently later."""


def read_workflow_meta(directory: Path, *, active: bool) -> WorkflowRunMeta | None:
    """Read one run summary, using the log terminal and its timestamp."""
    return read_workflow_run(directory, active=active).meta


def read_workflow_run(directory: Path, *, active: bool) -> WorkflowRunRead:
    """``read_workflow_meta``, telling a summary final for the files as they are from one an I/O error shaped."""
    settled = True
    try:
        header = read_run_header(directory)
        workflow_id = header.get("workflow_id")
        if not isinstance(workflow_id, str) or not workflow_id:
            return WorkflowRunRead(None, settled=True)
        title = header.get("title")
        try:
            terminal = read_run_terminal(directory)
        except (OSError, ValueError) as exc:
            terminal = None
            # A missing log already reads as no terminal: an I/O error only hides it for now.
            settled = not isinstance(exc, OSError)
            logger.debug("Unable to read workflow terminal under %s", directory, exc_info=True)
        status = run_status(
            terminal.outcome if terminal else None, terminal.reason if terminal else None, active=active
        )
        modified = datetime.fromtimestamp((directory / HEADER_FILE).stat().st_mtime, UTC)
        started = parse_created_at(terminal.finished_at if terminal else header.get("started_at"))
        if started is not None and started.tzinfo is not None:
            modified = started
        prompt = header.get("input_excerpt")
        meta = WorkflowRunMeta(
            directory.name,
            workflow_id,
            title if isinstance(title, str) and title else workflow_id,
            status,
            modified,
            prompt[:INPUT_EXCERPT_CHARS] if isinstance(prompt, str) else "",
        )
        return WorkflowRunRead(meta, settled=settled)
    except OSError as exc:
        return WorkflowRunRead(None, settled=isinstance(exc, FileNotFoundError) or exc.errno in _MISSING)
    except ValueError, TypeError:
        return WorkflowRunRead(None, settled=True)
