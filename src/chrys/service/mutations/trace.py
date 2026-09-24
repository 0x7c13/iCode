# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""fsatrace write tracing for the shell child tree (provenance Layer 3).

Wraps shell subprocess launches in `fsatrace <https://github.com/jacereda/
fsatrace>`_ so the actual write set of the child process tree is observed
(``LD_PRELOAD`` on Linux, DLL injection on Windows, DYLD insertion on
macOS — where SIP/hardened runtime usually block it; the probe below
auto-skips).

The evidence is strictly one-directional:

- a path in the trace write-set was written by our child tree →
  the matching window-detected mutation upgrades to ``PROVEN``;
- a path *absent* from the trace proves nothing (SIP-stripped injection,
  statically linked binaries, ``env -i`` children) → the row stays
  ``ASSUMED``.

The trace never *adds* mutations: out-of-scope writes (gitignored build
output etc.) are excluded from tracking by policy, and matching is done
by intersecting trace lines against the already-detected row set — the
full write set of a build command is never materialized.

Availability is probed once per process: ``CHRYS_MUTATION_TRACE=off``
disables outright; otherwise ``CHRYS_FSATRACE_PATH`` (authoritative when
set — no fallback past a broken override) or ``shutil.which`` locates the
binary, and a one-shot self-test (trace a trivial write, expect it in the
log) validates it.  Any launch- or parse-time failure degrades to
unwrapped execution / no upgrades — shell behavior must be identical with
tracing on, off, or broken.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

from chrys.foundation.platform import get_platform
from chrys.service.mutations.coordination import canonical_key, fold_case

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = logging.getLogger(__name__)

# writes, moves, deletes, touches — reads deliberately omitted (cheaper
# and irrelevant to mutation provenance).
TRACE_OPS = "wmdt"

# Past this size the whole trace for the call is discarded — degrade to
# no upgrades, never to slow parsing or memory blowups.
TRACE_MAX_BYTES = 32 * 1024 * 1024

TRACE_MODE_ENV = "CHRYS_MUTATION_TRACE"
FSATRACE_PATH_ENV = "CHRYS_FSATRACE_PATH"

_PROBE_TIMEOUT_SECONDS = 15.0

# Handoff from the mutation pipeline to the shell tool: the argv prefix
# ``[fsatrace, ops, trace_file, "--"]`` to prepend, or None to run
# unwrapped.  Set by ``prepare_mutation_tracking`` and reset by finalize
# or the sync abort — always in the setting context (each tool call runs
# in its own task, so parallel calls are isolated).
shell_trace_argv: ContextVar[tuple[str, ...] | None] = ContextVar("shell_trace_argv", default=None)

_probe_lock = threading.Lock()
_probe_done = False
_probe_result: str | None = None


def _trace_mode_off() -> bool:
    from chrys.foundation.config.process_settings import process_settings

    return process_settings().mutation_trace_mode == "off"


def resolve_fsatrace(*, force_probe: bool = False) -> str | None:
    """Locate and self-test the fsatrace binary, once per process.

    Returns the validated binary path, or None when tracing is switched
    off, no binary is found, or the self-test fails.  The result —
    including failure — is cached; ``force_probe`` re-runs the probe
    (tests, or after installing the binary mid-process).
    """
    global _probe_done, _probe_result
    with _probe_lock:
        if _probe_done and not force_probe:
            return _probe_result
        _probe_result = _probe()
        _probe_done = True
        return _probe_result


def _probe() -> str | None:
    from chrys.foundation.config.process_settings import process_settings

    if _trace_mode_off():
        return None
    override = process_settings().mutation_trace_fsatrace_path
    candidate = override or shutil.which("fsatrace")
    if not candidate:
        return None
    if _self_test(candidate):
        return candidate
    logger.debug("fsatrace self-test failed for %s; tracing unavailable", candidate)
    return None


def _self_test(binary: str) -> bool:
    """Trace a trivial write with the current interpreter; expect a hit.

    The venv CPython is dynamically linked and injection-friendly on
    Linux and most macOS dev setups; on hardened macOS the injection is
    silently stripped and the trace stays empty — probe fails, tracing
    auto-skips (by design, not an error).
    """
    try:
        with tempfile.TemporaryDirectory(prefix="chrys-fsatrace-probe-") as tmp:
            target = os.path.join(tmp, "probe-target.txt")
            trace = os.path.join(tmp, "probe.trace")
            argv = [
                binary,
                TRACE_OPS,
                trace,
                "--",
                sys.executable,
                "-c",
                f"open({target!r}, 'w').write('chrys fsatrace probe')",
            ]
            kwargs: dict = {}
            if get_platform().is_windows:
                # python -c needs no console; avoid a window flash.
                subprocess_module = cast(Any, subprocess)
                kwargs["creationflags"] = subprocess_module.CREATE_NO_WINDOW
            proc = subprocess.run(  # noqa: S603 — probe of an operator-supplied tracer binary
                argv,
                # Non-interactive probe: the process stdin (the ACP protocol
                # pipe under ACP) must not reach the tracer or its child.
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=_PROBE_TIMEOUT_SECONDS,
                check=False,
                **kwargs,
            )
            if proc.returncode != 0:
                return False
            return bool(parse_trace_write_keys(trace, {canonical_key(target)}))
    except Exception:
        logger.debug("fsatrace self-test errored", exc_info=True)
        return False


def allocate_trace_file(mutations_dir: Path) -> tuple[list[str], str] | None:
    """Reserve a trace path and build the wrapper argv prefix.

    Returns ``(argv_prefix, trace_file)`` or None when tracing is
    unavailable.  The file is nonce-named: provider call IDs are not
    unique across replay/reuse and must not key on-disk artifacts.
    fsatrace creates/truncates the file itself; if the wrapped launch
    never happens the path simply never exists.
    """
    binary = resolve_fsatrace()
    if binary is None:
        return None
    try:
        mutations_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("Could not create trace dir %s", mutations_dir, exc_info=True)
        return None
    trace_file = str(mutations_dir / f"trace-{uuid.uuid4().hex}.log")
    return [binary, TRACE_OPS, trace_file, "--"], trace_file


def discard_trace_file(trace_file: str | None) -> None:
    """Best-effort removal of a per-call trace artifact."""
    if not trace_file:
        return
    try:
        os.unlink(trace_file)
    except FileNotFoundError:
        pass
    except OSError:
        logger.debug("Could not remove trace file %s", trace_file, exc_info=True)


def parse_trace_write_keys(trace_file: str, candidate_keys: Iterable[str]) -> set[str]:
    """Canonical keys from ``candidate_keys`` positively written per the trace.

    Streaming intersection: each ``w|path`` / ``d|path`` / ``t|path`` /
    ``m|destination|source`` line (that upstream-documented move order is
    pinned by parser tests; both move sides count) is matched against the
    small candidate set and never accumulated.  A case-folded basename
    pre-filter skips the per-line ``realpath`` canonicalization for the
    vast majority of lines; a symlinked final component whose target has
    a different basename is missed by the pre-filter — acceptable, the
    evidence is one-directional and the row just stays ``ASSUMED``.

    Oversized traces are discarded whole; a missing file or any I/O
    error returns the empty set (no upgrades).
    """
    candidates = set(candidate_keys)
    if not candidates:
        return set()
    basenames = {os.path.basename(key) for key in candidates}
    hits: set[str] = set()
    try:
        if os.path.getsize(trace_file) > TRACE_MAX_BYTES:
            logger.debug("Trace %s exceeds %d bytes; discarding whole", trace_file, TRACE_MAX_BYTES)
            return set()
        with open(trace_file, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\r\n")
                if len(line) < 3 or line[1] != "|":
                    continue
                # Upstream emits the op UPPERCASE when it could not
                # realpath one side (``M|dst`` — source unknown, the
                # destination write is still positive evidence).
                op = line[0].lower()
                if op not in TRACE_OPS:
                    continue
                rest = line[2:]
                if op == "m":
                    destination, _, source = rest.partition("|")
                    paths = (destination, source) if source else (destination,)
                else:
                    paths = (rest,)
                for path in paths:
                    # ``<unknown>`` is upstream's placeholder for an
                    # unresolvable path.
                    if path == "<unknown>" or fold_case(os.path.basename(path)) not in basenames:
                        continue
                    key = canonical_key(path)
                    if key in candidates:
                        hits.add(key)
                        if len(hits) == len(candidates):
                            return hits
    except OSError:
        logger.debug("Trace %s unreadable; no upgrades", trace_file, exc_info=True)
        return set()
    return hits
