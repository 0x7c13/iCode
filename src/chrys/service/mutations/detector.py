# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ShellMutationDetector — heuristic command parser for file mutation targets.

Best-effort detection of files that a shell command will modify.
Used as a fast supplementary hint alongside :class:`WorkspaceScanner`
for targeted pre-snapshotting.

Detection coverage::

    rm, del, Remove-Item, unlink          -> DELETE
    mv, move, Move-Item, rename           -> MOVE
    cp, copy, Copy-Item                   -> CREATE (destination)
    touch, New-Item                       -> CREATE
    sed -i, chmod, chown, chgrp, tee      -> MODIFY
    Set-Content, Add-Content, Out-File    -> MODIFY
    Output redirections: > file, >> file  -> CREATE / MODIFY

Inherent limitations (not detected)::

    - File writes inside scripts (``python script.py``)
    - Commands behind aliases or shell functions
    - Complex glob expansions (``rm *.log``)
    - Here-documents (``cat <<EOF > file``)

These limitations are why WorkspaceScanner is the primary detection
mechanism.  ShellMutationDetector serves as a fast supplementary hint.
"""

from __future__ import annotations

import os
import shlex

from chrys.service.mutations.types import MutationOp


class ShellMutationDetector:
    """Heuristic detection of file mutation targets in shell commands."""

    _DELETE_CMDS = frozenset({"rm", "del", "remove-item", "unlink"})
    _MOVE_CMDS = frozenset({"mv", "move", "move-item", "rename"})
    _COPY_CMDS = frozenset({"cp", "copy", "copy-item"})
    _CREATE_CMDS = frozenset({"touch", "new-item"})
    _MODIFY_CMDS = frozenset({"sed", "chmod", "chown", "chgrp", "tee", "truncate"})
    _POWERSHELL_CONTENT_CMDS = frozenset({"set-content", "add-content", "out-file"})
    _POWERSHELL_DELETE_CMDS = frozenset({"remove-item"})
    _POWERSHELL_MOVE_CMDS = frozenset({"move-item"})
    _POWERSHELL_COPY_CMDS = frozenset({"copy-item"})
    _POWERSHELL_CREATE_CMDS = frozenset({"new-item"})
    _POWERSHELL_CMDS = (
        _POWERSHELL_CONTENT_CMDS
        | _POWERSHELL_DELETE_CMDS
        | _POWERSHELL_MOVE_CMDS
        | _POWERSHELL_COPY_CMDS
        | _POWERSHELL_CREATE_CMDS
    )
    _POWERSHELL_PATH_PARAMS = frozenset({"-path", "-literalpath", "-filepath"})
    _POWERSHELL_DESTINATION_PARAMS = frozenset({"-destination"})
    _POWERSHELL_NAME_PARAMS = frozenset({"-name"})
    _POWERSHELL_VALUE_PARAMS = frozenset(
        {
            "-credential",
            "-encoding",
            "-erroraction",
            "-errorvariable",
            "-filter",
            "-exclude",
            "-include",
            "-informationaction",
            "-informationvariable",
            "-inputobject",
            "-itemtype",
            "-newerthan",
            "-olderthan",
            "-outbuffer",
            "-outvariable",
            "-pipelinevariable",
            "-progressaction",
            "-stream",
            "-value",
            "-warningaction",
            "-warningvariable",
            "-width",
        }
    )

    @classmethod
    def detect_targets(
        cls,
        command: str,
        cwd: str | None = None,
    ) -> list[tuple[str, MutationOp]]:
        """Parse a shell command and return likely mutation targets.

        Leverages ``shell_tokens.split_on_operators()`` for command
        splitting, then applies per-command heuristics to extract
        file arguments.

        Args:
            command: The raw shell command string.
            cwd: Working directory for resolving relative paths.
                 Defaults to ``os.getcwd()``.

        Returns:
            List of ``(absolute_path, expected_operation)`` tuples.
        """
        from chrys.foundation.util.shell_tokens import split_on_operators

        resolve_cwd = cwd or os.getcwd()
        targets: list[tuple[str, MutationOp]] = []

        segments = split_on_operators(command)
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Check for output redirections: ... > file or ... >> file
            targets.extend(cls._detect_redirections(segment, resolve_cwd))

            # Parse tokens. PowerShell uses backslash as a path separator,
            # so keep a non-POSIX parse alongside the normal POSIX one for
            # cmdlet-specific extraction.
            try:
                tokens = shlex.split(segment)
            except ValueError:
                tokens = segment.split()
            try:
                powershell_tokens = [cls._strip_powershell_quotes(token) for token in shlex.split(segment, posix=False)]
            except ValueError:
                powershell_tokens = [cls._strip_powershell_quotes(token) for token in segment.split()]
            if not tokens:
                continue

            # Skip env var assignments
            idx = 0
            while idx < len(tokens) and "=" in tokens[idx] and not tokens[idx].startswith("="):
                idx += 1
            if idx >= len(tokens):
                continue

            cmd_name = tokens[idx]
            args = tokens[idx + 1 :]
            # Strip path prefix (e.g. /usr/bin/rm -> rm)
            cmd_base = os.path.basename(cmd_name).lower()

            powershell_targets = cls._detect_from_powershell(powershell_tokens, resolve_cwd)
            if powershell_targets is not None:
                targets.extend(powershell_targets)
                continue
            targets.extend(cls._detect_from_command(cmd_base, args, resolve_cwd))

        return targets

    @classmethod
    def _detect_from_powershell(cls, tokens: list[str], cwd: str) -> list[tuple[str, MutationOp]] | None:
        """Extract targets from PowerShell file-writing cmdlets.

        The generic POSIX token parse corrupts unquoted Windows paths such
        as ``D:\\repo\\file.txt`` because backslash is an escape there.
        This PowerShell-specific pass uses ``shlex``'s non-POSIX mode for
        cmdlets whose names identify the command as PowerShell.
        """
        if not tokens:
            return None

        tokens = cls._drop_powershell_call_operator(tokens)
        if not tokens:
            return None

        cmd = os.path.basename(tokens[0]).lower()
        if cmd not in cls._POWERSHELL_CMDS:
            return None

        args = tokens[1:]
        if cmd in cls._POWERSHELL_CONTENT_CMDS:
            return cls._detect_powershell_content_targets(args, cwd)
        if cmd in cls._POWERSHELL_DELETE_CMDS:
            paths = cls._powershell_named_values(args, cls._POWERSHELL_PATH_PARAMS)
            if not paths:
                paths = cls._powershell_positional_args(args, cls._POWERSHELL_VALUE_PARAMS)
            return cls._powershell_path_targets(paths, cwd, MutationOp.DELETE)
        if cmd in cls._POWERSHELL_MOVE_CMDS:
            return cls._detect_powershell_move_targets(args, cwd)
        if cmd in cls._POWERSHELL_COPY_CMDS:
            return cls._detect_powershell_copy_targets(args, cwd)
        if cmd in cls._POWERSHELL_CREATE_CMDS:
            return cls._detect_powershell_create_targets(args, cwd)
        return []

    @classmethod
    def _drop_powershell_call_operator(cls, tokens: list[str]) -> list[str]:
        """Remove a leading PowerShell call operator from parsed tokens."""
        first = tokens[0]
        if first == "&":
            return tokens[1:]
        if first.startswith("&") and len(first) > 1:
            return [cls._strip_powershell_quotes(first[1:]), *tokens[1:]]
        return tokens

    @classmethod
    def _detect_powershell_content_targets(cls, args: list[str], cwd: str) -> list[tuple[str, MutationOp]]:
        """Extract targets from Set-Content/Add-Content/Out-File."""
        paths = cls._powershell_named_values(args, cls._POWERSHELL_PATH_PARAMS)
        if not paths:
            paths = cls._powershell_positional_args(args, cls._POWERSHELL_VALUE_PARAMS)[:1]
        return cls._powershell_path_targets(paths, cwd, MutationOp.MODIFY)

    @classmethod
    def _detect_powershell_move_targets(cls, args: list[str], cwd: str) -> list[tuple[str, MutationOp]]:
        """Extract src/dst targets from Move-Item."""
        src_paths = cls._powershell_named_values(args, cls._POWERSHELL_PATH_PARAMS)
        dst_paths = cls._powershell_named_values(args, cls._POWERSHELL_DESTINATION_PARAMS)
        positionals = cls._powershell_positional_args(args, cls._POWERSHELL_VALUE_PARAMS)
        src_raw, dst_raw = cls._powershell_source_destination(src_paths, dst_paths, positionals)
        if not src_raw or not dst_raw:
            return []

        src = cls._resolve_shell_path(cwd, src_raw)
        dst = cls._resolve_shell_path(cwd, dst_raw)
        results = [(dst, MutationOp.MOVE), (src, MutationOp.DELETE)]
        synth = cls._synth_dir_target(src, dst, dst_raw)
        if synth is not None:
            results.append((synth, MutationOp.MOVE))
        return results

    @classmethod
    def _detect_powershell_copy_targets(cls, args: list[str], cwd: str) -> list[tuple[str, MutationOp]]:
        """Extract destination targets from Copy-Item."""
        src_paths = cls._powershell_named_values(args, cls._POWERSHELL_PATH_PARAMS)
        dst_paths = cls._powershell_named_values(args, cls._POWERSHELL_DESTINATION_PARAMS)
        positionals = cls._powershell_positional_args(args, cls._POWERSHELL_VALUE_PARAMS)
        src_raw, dst_raw = cls._powershell_source_destination(src_paths, dst_paths, positionals)
        if not dst_raw:
            return []

        dst = cls._resolve_shell_path(cwd, dst_raw)
        results = [(dst, MutationOp.CREATE)]
        if src_raw:
            src = cls._resolve_shell_path(cwd, src_raw)
            synth = cls._synth_dir_target(src, dst, dst_raw)
            if synth is not None:
                results.append((synth, MutationOp.CREATE))
        return results

    @classmethod
    def _detect_powershell_create_targets(cls, args: list[str], cwd: str) -> list[tuple[str, MutationOp]]:
        """Extract targets from New-Item."""
        paths = cls._powershell_named_values(args, cls._POWERSHELL_PATH_PARAMS)
        names = cls._powershell_named_values(args, cls._POWERSHELL_NAME_PARAMS)
        if paths and names:
            paths = [os.path.join(path, name) for path in paths for name in names if path and name]
        if not paths:
            paths = names[:1] or cls._powershell_positional_args(args, cls._POWERSHELL_VALUE_PARAMS)[:1]
        return cls._powershell_path_targets(paths, cwd, MutationOp.CREATE)

    @staticmethod
    def _powershell_source_destination(
        src_paths: list[str],
        dst_paths: list[str],
        positionals: list[str],
    ) -> tuple[str, str]:
        """Resolve PowerShell source/destination from named and positional args."""
        src_raw = src_paths[0] if src_paths else positionals[0] if positionals else ""
        if dst_paths:
            dst_raw = dst_paths[0]
        elif src_paths:
            dst_raw = positionals[0] if positionals else ""
        else:
            dst_raw = positionals[1] if len(positionals) >= 2 else ""
        return src_raw, dst_raw

    @classmethod
    def _powershell_named_values(cls, args: list[str], param_names: frozenset[str]) -> list[str]:
        """Return values supplied to the named PowerShell parameters."""
        values: list[str] = []
        i = 0
        while i < len(args):
            token = args[i]
            param, inline_value = cls._powershell_param(token)
            if param in param_names:
                if inline_value is not None:
                    values.append(inline_value)
                elif i + 1 < len(args) and not cls._looks_like_powershell_param(args[i + 1]):
                    values.append(args[i + 1])
                    i += 1
            i += 1
        return values

    @classmethod
    def _powershell_positional_args(cls, args: list[str], value_params: frozenset[str]) -> list[str]:
        """Return positional args, skipping values of known value-taking params."""
        consuming_params = (
            value_params
            | cls._POWERSHELL_PATH_PARAMS
            | cls._POWERSHELL_DESTINATION_PARAMS
            | cls._POWERSHELL_NAME_PARAMS
        )
        positionals: list[str] = []
        i = 0
        while i < len(args):
            token = args[i]
            param, inline_value = cls._powershell_param(token)
            if param:
                if (
                    inline_value is None
                    and param in consuming_params
                    and i + 1 < len(args)
                    and not cls._looks_like_powershell_param(args[i + 1])
                ):
                    i += 1
                i += 1
                continue
            positionals.append(token)
            i += 1
        return positionals

    @classmethod
    def _powershell_path_targets(cls, paths: list[str], cwd: str, op: MutationOp) -> list[tuple[str, MutationOp]]:
        """Resolve non-empty PowerShell path tokens to mutation targets."""
        return [(cls._resolve_shell_path(cwd, path), op) for path in paths if path]

    @staticmethod
    def _resolve_shell_path(cwd: str, path: str) -> str:
        """Resolve a parsed shell path relative to the command cwd."""
        return os.path.normpath(os.path.join(cwd, path))

    @staticmethod
    def _strip_powershell_quotes(token: str) -> str:
        """Remove one layer of PowerShell-style single/double quotes."""
        if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
            return token[1:-1]
        return token

    @staticmethod
    def _looks_like_powershell_param(token: str) -> bool:
        return token.startswith("-") and len(token) > 1

    @classmethod
    def _powershell_param(cls, token: str) -> tuple[str, str | None]:
        """Split a PowerShell parameter token into name and inline value."""
        if not cls._looks_like_powershell_param(token):
            return "", None
        for sep in (":", "="):
            name, found, value = token.partition(sep)
            if found:
                return name.lower(), cls._strip_powershell_quotes(value)
        return token.lower(), None

    @classmethod
    def _detect_from_command(
        cls,
        cmd: str,
        args: list[str],
        cwd: str,
    ) -> list[tuple[str, MutationOp]]:
        """Extract mutation targets from a parsed command + args."""
        results: list[tuple[str, MutationOp]] = []
        # Filter out flags (tokens starting with -)
        file_args = [a for a in args if not a.startswith("-")]

        if cmd in cls._DELETE_CMDS:
            results.extend((os.path.normpath(os.path.join(cwd, f)), MutationOp.DELETE) for f in file_args)

        elif cmd in cls._MOVE_CMDS:
            if len(file_args) >= 2:
                src = os.path.normpath(os.path.join(cwd, file_args[-2]))
                dst = os.path.normpath(os.path.join(cwd, file_args[-1]))
                results.append((dst, MutationOp.MOVE))
                results.append((src, MutationOp.DELETE))
                synth = cls._synth_dir_target(src, dst, file_args[-1])
                if synth is not None:
                    results.append((synth, MutationOp.MOVE))

        elif cmd in cls._COPY_CMDS:
            if len(file_args) >= 2:
                src = os.path.normpath(os.path.join(cwd, file_args[-2]))
                dst = os.path.normpath(os.path.join(cwd, file_args[-1]))
                results.append((dst, MutationOp.CREATE))
                synth = cls._synth_dir_target(src, dst, file_args[-1])
                if synth is not None:
                    results.append((synth, MutationOp.CREATE))

        elif cmd in cls._CREATE_CMDS:
            results.extend((os.path.normpath(os.path.join(cwd, f)), MutationOp.CREATE) for f in file_args)

        elif cmd in cls._MODIFY_CMDS:
            results.extend((os.path.normpath(os.path.join(cwd, f)), MutationOp.MODIFY) for f in file_args)

        return results

    @staticmethod
    def _synth_dir_target(src: str, dst: str, dst_raw: str) -> str | None:
        """If ``dst`` is a directory destination, return ``dst/basename(src)``.

        Handles ``mv A B/`` / ``cp A B/`` (trailing slash) and bare
        ``mv A B`` / ``cp A B`` where ``B`` exists as a directory.  The
        primary dst target that callers already emit points at the
        directory, so the scanner's post-exec walk finds the file — but
        without this synthesised inner path the tracker has no
        pre-snapshot for its final location and falls back to the
        post-exec content (fix #1 in ``MutationTracker.record``).
        Returns ``None`` when ``dst`` is just a file rename.
        """
        if not (dst_raw.endswith(("/", os.sep)) or os.path.isdir(dst)):
            return None
        synth = os.path.normpath(os.path.join(dst, os.path.basename(src)))
        return synth if synth != dst else None

    @classmethod
    def _detect_redirections(
        cls,
        segment: str,
        cwd: str,
    ) -> list[tuple[str, MutationOp]]:
        """Detect output redirections (> file, >> file) in a command segment."""
        results: list[tuple[str, MutationOp]] = []
        i = 0
        in_single = False
        in_double = False

        while i < len(segment):
            ch = segment[i]
            if ch == "\\" and not in_single and i + 1 < len(segment):
                i += 2
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                i += 1
                continue

            if not in_single and not in_double and ch == ">":
                # Skip >> (append) or > (overwrite) — both are mutations
                j = i + 1
                if j < len(segment) and segment[j] == ">":
                    j += 1  # >>
                # Skip whitespace after > or >>
                while j < len(segment) and segment[j] == " ":
                    j += 1
                # Collect the filename token
                fname_chars: list[str] = []
                while j < len(segment) and segment[j] not in (" ", "\t", "|", "&", ";", ">", "<"):
                    fname_chars.append(segment[j])
                    j += 1
                if fname_chars:
                    fname = "".join(fname_chars)
                    results.append((os.path.normpath(os.path.join(cwd, fname)), MutationOp.MODIFY))
                i = j
                continue

            i += 1

        return results
