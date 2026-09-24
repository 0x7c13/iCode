# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Getting a shell to report what it is doing.

Each supported shell is started with hooks of its own kind that print two private OSC reports,
which the terminal emulator turns into events: the working directory ahead of every prompt
(OSC 2025) and the command line about to run (OSC 2026). Payloads travel as base64, so no path or
command can end the sequence early. The hooks load with the shell's start-up files and echo nothing.
"""

from __future__ import annotations

import os
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePath

from chrys.app.tui.terminal._pty_backend import IS_WINDOWS

_POSIX_REPORTS = r"""
__chrys_b64() { printf "%s" "$1" | base64 | tr -d "\n"; }
__chrys_report() { printf "\e]%s;b64:%s\e\\" "$1" "$(__chrys_b64 "$2")"; }
"""

# zsh reads its start-up files from ZDOTDIR, so ours stand in for the user's and read those first.
# The user's .zshenv may itself move ZDOTDIR; whatever it leaves is where their .zshrc is.
_ZSH_ENV = """\
__chrys_init_dir="$ZDOTDIR"
ZDOTDIR={user_zdotdir}
[[ -f "$ZDOTDIR/.zshenv" ]] && source "$ZDOTDIR/.zshenv"
__chrys_user_zdotdir="$ZDOTDIR"
ZDOTDIR="$__chrys_init_dir"
unset __chrys_init_dir
"""

_ZSH_RC = (
    """\
ZDOTDIR="$__chrys_user_zdotdir"
unset __chrys_user_zdotdir
[[ -f "$ZDOTDIR/.zshrc" ]] && source "$ZDOTDIR/.zshrc"
"""
    + _POSIX_REPORTS
    + """\
__chrys_preexec() { __chrys_report 2026 "$1"; }
__chrys_precmd() { __chrys_report 2025 "$PWD"; }
precmd_functions+=(__chrys_precmd)
preexec_functions+=(__chrys_preexec)
"""
)

# bash has no preexec. Its DEBUG trap fires before every simple command, so the first one after a
# prompt stands for the command line, whose full text is the history entry it just became. The
# user's own prompt commands are simple commands too; ours bracket them, so that the trap can tell
# them from a command line and they still see the exit status of the last one.
_BASH_RC = (
    """\
[[ -f ~/.bashrc ]] && source ~/.bashrc
"""
    + _POSIX_REPORTS
    + """\
__chrys_at_prompt=false
__chrys_read_history() {
  local entry pattern='^[[:space:]]*([0-9]+)[* ] (.*)$'
  entry="$(HISTTIMEFORMAT= builtin history 1)"
  if [[ $entry =~ $pattern ]]; then
    __chrys_entry_number="${BASH_REMATCH[1]}" __chrys_entry_text="${BASH_REMATCH[2]}"
  else
    __chrys_entry_number="" __chrys_entry_text=""
  fi
}
__chrys_preexec() {
  [[ $BASH_COMMAND == __chrys_prompt_begin ]] && __chrys_at_prompt=false
  [[ $__chrys_at_prompt == true ]] || return
  __chrys_at_prompt=false
  __chrys_read_history
  if [[ $__chrys_entry_number != "$__chrys_prompt_number" || $__chrys_entry_text == "$BASH_COMMAND"* ]]; then
    # A new history entry is the line just entered. So is an old one that begins with the command
    # now starting: the line was a repeat, which the history keeps once.
    __chrys_report 2026 "$__chrys_entry_text"
  else
    # Kept out of the history on purpose, or there is no history. This much of it is certain.
    __chrys_report 2026 "$BASH_COMMAND"
  fi
}
__chrys_prompt_begin() { return "$?"; }
__chrys_precmd() {
  __chrys_report 2025 "$PWD"
  __chrys_read_history
  __chrys_prompt_number="$__chrys_entry_number"
  __chrys_at_prompt=true
}
if (( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )); then
  PROMPT_COMMAND=(__chrys_prompt_begin "${PROMPT_COMMAND[@]}" __chrys_precmd)
else
  __chrys_theirs="${PROMPT_COMMAND%"${PROMPT_COMMAND##*[![:space:];]}"}"
  PROMPT_COMMAND="__chrys_prompt_begin;${__chrys_theirs:+$__chrys_theirs;}__chrys_precmd"
  unset __chrys_theirs
fi
trap '__chrys_preexec' DEBUG
"""
)

_FISH_HOOKS = (
    'function __chrys_report; printf "\\e]%s;b64:%s\\e\\\\" $argv[1] (printf "%s" "$argv[2]" | base64 | tr -d "\\n"); end; '
    'function __chrys_preexec --on-event fish_preexec; __chrys_report 2026 "$argv"; end; '
    "function __chrys_prompt --on-event fish_prompt; __chrys_report 2025 (pwd); end"
)

_POWERSHELL_HOOKS = (
    "function global:__chrys_report($code, $text) {"
    " $e = [char]27; $st = [char]92;"
    " $payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($text));"
    ' return "$e]$code;b64:$payload$e$st"'
    "};"
    "function prompt {"
    " Write-Host -NoNewline (__chrys_report 2025 (Get-Location).Path);"
    ' return "PS $($executionContext.SessionState.Path.CurrentLocation)> "'
    "};"
    "if (Get-Module PSReadLine) {"
    " Set-PSReadLineOption -AddToHistoryHandler {"
    "  param($command)"
    "  [Console]::Write((__chrys_report 2026 $command));"
    "  return $true"
    " }"
    "}"
)


@dataclass(frozen=True, slots=True)
class ShellLaunch:
    """How to start a shell so that it reports to us."""

    argv: list[str]
    scratch_directory: str | None = None
    """Holds start-up files written for this shell. The shell reads them once; whoever started it
    removes the directory when the shell is gone."""

    def clean_up(self) -> None:
        if self.scratch_directory is not None:
            shutil.rmtree(self.scratch_directory, ignore_errors=True)


def prepare_shell_launch(command: str, env: dict[str, str]) -> ShellLaunch:
    """Work out the argv that starts ``command`` with reporting hooks, adjusting ``env`` to match.

    A shell we have no hooks for starts as it is; it works, and reports nothing.
    """
    argv = _split_command(command)
    program, arguments = argv[0], argv[1:]
    name = PurePath(program).stem.lower()
    if name.startswith("zsh"):
        return _launch_zsh(argv, env)
    if name.startswith("bash"):
        scratch = _write_scratch("chrys_bash_", {"init.bash": _BASH_RC})
        rcfile = Path(scratch, "init.bash")
        # bash wants its long options ahead of any short ones.
        return ShellLaunch([program, "--rcfile", str(rcfile), *arguments], scratch)
    if name.startswith("fish"):
        return ShellLaunch([*argv, "-C", _FISH_HOOKS])
    if name.startswith(("pwsh", "powershell")):
        return ShellLaunch([*argv, "-NoExit", "-Command", _POWERSHELL_HOOKS])
    return ShellLaunch(argv)


def _split_command(command: str) -> list[str]:
    if os.path.exists(command):
        # A path, spaces and all (C:\Program Files\...), not a command line.
        return [command]
    try:
        argv = shlex.split(command, posix=not IS_WINDOWS)
    except ValueError:
        argv = []
    return argv or [command]


def _write_scratch(prefix: str, files: dict[str, str]) -> str:
    """Write shell startup files, removing their directory if any write fails."""
    scratch = tempfile.mkdtemp(prefix=prefix)
    try:
        for name, content in files.items():
            Path(scratch, name).write_text(content, encoding="utf-8")
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return scratch


def _launch_zsh(argv: list[str], env: dict[str, str]) -> ShellLaunch:
    # The home the shell will see, which is where it would have looked.
    user_zdotdir = env.get("ZDOTDIR") or env.get("HOME") or os.path.expanduser("~")
    scratch = _write_scratch(
        "chrys_zsh_",
        {".zshenv": _ZSH_ENV.format(user_zdotdir=shlex.quote(user_zdotdir)), ".zshrc": _ZSH_RC},
    )
    env["ZDOTDIR"] = scratch
    return ShellLaunch(argv, scratch)
