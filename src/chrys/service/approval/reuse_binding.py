# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bind approval reuse only to trusted built-in command and file-write tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chrys.kernel import FunctionTool
from chrys.service.approval.command_identity import normalize_simple_command
from chrys.service.approval.grant_store import GRANTS_FILE, ApprovalGrantStore, session_grants_path
from chrys.service.approval.reuse import (
    ApprovalReuseService,
    Candidate,
    CommandCandidate,
    CommandKey,
    FileCandidate,
    FileKey,
    ReuseContext,
    canonical,
    project_path,
    simple_argv,
)
from chrys.service.tools.builtins.filesystem import FilesystemTools
from chrys.service.tools.builtins.shell import ShellTools

if TYPE_CHECKING:
    from chrys.foundation.models.session_env import SessionEnvironment
    from chrys.kernel.middleware import FunctionInvocationContext
    from chrys.service.tools.file_approval import FileWriteTarget


class ApprovalReuseBinding:
    def __init__(self, runtime: SessionEnvironment, tools: list, *, session_id: str | None = None) -> None:
        self.runtime = runtime
        self.session_id = session_id if session_id is not None else runtime.session_id
        self.tools = {tool.name: tool for tool in tools if isinstance(tool, FunctionTool)}
        config_dir = runtime.platform.config_dir
        self.service = ApprovalReuseService(
            ApprovalGrantStore(config_dir / GRANTS_FILE),
            ApprovalGrantStore(session_grants_path(config_dir, self.session_id)),
        )

    def supports(self, context: FunctionInvocationContext) -> bool:
        tool = context.function
        owner = tool.bound_instance
        return (
            self.tools.get(tool.name) is tool
            and context.kwargs.get("session", context.session) is context.session
            and isinstance(context.arguments, dict)
            and (
                (isinstance(owner, ShellTools) and tool.func is ShellTools.execute.func)
                or (
                    isinstance(owner, FilesystemTools)
                    and tool.func in (FilesystemTools.write_file.func, FilesystemTools.edit_file.func)
                )
            )
        )

    def file_targets(self, context: FunctionInvocationContext) -> tuple[FileWriteTarget, ...] | None:
        owner = context.function.bound_instance
        if self.supports(context) and isinstance(owner, FilesystemTools) and isinstance(context.arguments, dict):
            return owner.approval_targets(context.arguments)
        return None

    def candidate(self, context: FunctionInvocationContext, *, non_reusable: bool = False) -> Candidate | None:
        if non_reusable or not self.supports(context):
            return None
        args = context.arguments
        if not isinstance(args, dict):
            return None
        reuse = ReuseContext(self.session_id, project_path(self.runtime.cwd))
        if not reuse.eligible:
            return None
        owner = context.function.bound_instance
        if isinstance(owner, FilesystemTools):
            paths = owner.affected_paths(args)
            return FileCandidate(reuse, frozenset(FileKey(path=path) for path in paths)) if paths else None
        if not isinstance(owner, ShellTools) or "working_dir" in args or not isinstance(args.get("command"), str):
            return None
        shell, command = owner.shell, args["command"]
        try:
            options = canonical(
                {key: value for key, value in args.items() if key not in {"command", "reason", "timeout", "max_tokens"}}
            )
            tokens = normalize_simple_command(command, shell.name)
            key = CommandKey(
                cwd=reuse.project_id,
                shell=shell.name,
                executable=shell.path,
                shell_args=tuple(shell.args),
                command=tokens if tokens is not None else command,
                options=options,
            )
        except ValueError, TypeError, RecursionError:
            return None
        return CommandCandidate(reuse, key, simple_argv(command, shell.name))
