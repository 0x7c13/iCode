# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Wire contract between the main process and a workflow worker host.

Newline-framed JSON over the worker's stdio. The main process owns stdin/stdout
of the child: every frame the child writes is protocol, and the host moves the
user's ``fd 1``/``fd 2`` elsewhere before any user code runs. The two sides use
disjoint request-id spaces (main odd from 1, worker even from 2), so one reader
per side can demultiplex responses by id without agreement on ordering.

The host is a standalone stdlib file that cannot import this module; it carries
its own copy of the names below, and ``tests/service/workflows/test_protocol.py``
pins the two copies to each other.

Frames:

* request      ``{"id": int, "method": str, "params": {...}}``
* response     ``{"id": int, "result": {...}}`` or ``{"id": int, "error": {"code", "message", "data"}}``
* notification ``{"method": str, "params": {...}}``
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from chrys.service.workflows.scheduler import AttemptRef

PROTOCOL_VERSION: Final = 1
PYTHON_FLOOR: Final = (3, 9)
"""Oldest interpreter that can run the host and the injected SDK."""


class Method(StrEnum):
    """Every method name on the wire."""

    # worker → main, first frame
    HELLO = "hello"
    # main → worker requests
    LOAD = "load"
    RUN_PYTHON = "run_python"
    EVAL_OUTGOING = "eval_outgoing"
    EVAL_LOOP_UNTIL = "eval_loop_until"
    COMBINE = "combine"
    CANCEL = "cancel"
    NATIVE_OUTPUT = "native_output"
    SHUTDOWN = "shutdown"
    # worker → main during an attempt
    ASK = "ask"
    EMIT = "emit"


class ErrorCode(StrEnum):
    """Worker failure codes; the runner maps attempt failures onto scheduler retry classes."""

    USER_EXCEPTION = "user_exception"
    VALUE_NOT_SERIALIZABLE = "value_not_serializable"
    VALUE_TOO_LARGE = "value_too_large"
    PROTOCOL_LIMIT = "protocol_limit"
    ASK_UNAVAILABLE = "ask_unavailable"
    ATTEMPT_TERMINATED = "attempt_terminated"
    LOAD_FAILED = "load_failed"
    INVALID_PARAMS = "invalid_params"
    UNKNOWN_METHOD = "unknown_method"
    INTERNAL = "internal"


_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ProtocolLimits:
    """Resource bounds and deadlines shared with the standalone worker."""

    # one frame, either direction; a larger value is value_too_large, any other frame protocol_limit
    max_frame_bytes: int = 16 * _MIB
    # WorkflowValue.data shape: nesting depth, total containers + scalars, one string's length
    max_payload_depth: int = 64
    max_payload_items: int = 200_000
    max_string_chars: int = 4 * _MIB
    # main → worker requests in flight
    max_pending_requests: int = 256
    # both directions, per run; exceeding either is a worker_lost-class failure
    max_run_frames: int = 1_000_000
    max_run_bytes: int = 4 * 1024 * _MIB
    # join sources kept in memory for one combine call
    max_join_retained_bytes: int = 12 * _MIB
    # ctx.emit per attempt: sliding one-second rate, count, and bytes
    emit_rate_per_s: int = 200
    max_emits_per_attempt: int = 10_000
    max_emit_bytes_per_attempt: int = 8 * _MIB
    # main-side deadline in seconds for eval_outgoing / eval_loop_until / combine
    eval_deadline: float = 30.0
    # sync bodies and default-executor offloads; evaluations have their own pool
    worker_thread_pool_size: int = 8
    eval_thread_pool_size: int = 2
    # cumulative abandoned work, attributed by the worker to the pool that actually owns it
    worker_leak_budget: int = 8
    eval_leak_budget: int = 2
    load_timeout: float = 60.0  # seconds; includes executing the module and building its manifest
    hello_timeout: float = 15.0  # seconds
    shutdown_grace: float = 3.0  # seconds
    # after cancelling an async body the host waits this many seconds for it to unwind before acknowledging;
    # a body that keeps swallowing the cancellation is reported like a leaked thread
    cancel_drain: float = 1.0
    # captured sys.stdout/sys.stderr per attempt, and the run-level native tail
    captured_output_bytes: int = 64 * 1024
    native_output_tail_bytes: int = 64 * 1024


LIMITS: Final = ProtocolLimits()


class ProtocolError(ValueError):
    """A frame that does not fit the contract; the connection is not recoverable."""


def encode_frame(frame: Mapping[str, Any]) -> bytes:
    """Serialize one frame; raises :class:`ProtocolError` above the frame cap.

    Frames carry filesystem identity (the entry path and the workspace, which on
    POSIX may hold surrogateescaped bytes), so every str must survive the byte
    form unchanged: lone surrogates pass through. Whether a *value* is valid
    Unicode is the value layer's rule, checked where the value is produced.
    """
    line = json.dumps(frame, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    data = line.encode("utf-8", "surrogatepass") + b"\n"
    if len(data) > LIMITS.max_frame_bytes:
        raise ProtocolError(f"frame of {len(data)} bytes exceeds max_frame_bytes.")
    return data


def decode_frame(line: bytes) -> dict[str, Any]:
    """Parse one line into a frame dict, validating the envelope shape only."""
    if len(line) > LIMITS.max_frame_bytes:
        raise ProtocolError(f"frame of {len(line)} bytes exceeds max_frame_bytes.")
    try:
        frame = json.loads(line.decode("utf-8", "surrogatepass"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(f"frame is not UTF-8 JSON: {exc}") from exc
    if not isinstance(frame, dict):
        raise ProtocolError("frame is not a JSON object.")
    if "id" in frame:
        if not isinstance(frame["id"], int) or isinstance(frame["id"], bool) or frame["id"] < 1:
            raise ProtocolError("frame id must be a positive int.")
        if "method" in frame:
            _check_request(frame)
        elif ("result" in frame) == ("error" in frame):
            raise ProtocolError("response carries exactly one of result/error.")
        elif "error" in frame:
            _check_error(frame["error"])
        elif not isinstance(frame["result"], dict):
            raise ProtocolError("response result must be an object.")
    else:
        _check_request(frame)
    return frame


def _check_request(frame: dict[str, Any]) -> None:
    if not isinstance(frame.get("method"), str) or not frame["method"]:
        raise ProtocolError("request method must be a non-empty string.")
    if not isinstance(frame.get("params"), dict):
        raise ProtocolError("request params must be an object.")


def _check_error(error: Any) -> None:
    if (
        not isinstance(error, dict)
        or not isinstance(error.get("code"), str)
        or not isinstance(error.get("message"), str)
    ):
        raise ProtocolError("error must carry a string code and message.")
    if not isinstance(error.get("data", {}), dict):
        raise ProtocolError("error data must be an object.")


def ref_to_wire(ref: AttemptRef) -> dict[str, Any]:
    return {
        "run_id": ref.run_id,
        "node_id": ref.node_id,
        "activation_id": ref.activation_id,
        "attempt": ref.attempt,
    }


def ref_from_wire(payload: Any) -> AttemptRef:
    if not isinstance(payload, dict):
        raise ProtocolError("ref must be an object.")
    try:
        run_id, node_id, activation_id, attempt = (
            payload["run_id"],
            payload["node_id"],
            payload["activation_id"],
            payload["attempt"],
        )
    except KeyError as exc:
        raise ProtocolError(f"ref is missing {exc.args[0]!r}.") from exc
    if not all(isinstance(part, str) and part for part in (run_id, node_id, activation_id)):
        raise ProtocolError("ref ids must be non-empty strings.")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ProtocolError("ref attempt must be an int >= 1.")
    return AttemptRef(run_id=run_id, node_id=node_id, activation_id=activation_id, attempt=attempt)


def ref_key(ref: AttemptRef) -> str:
    """The wire identity of an attempt: what the fence on both sides keys on."""
    return f"{ref.run_id}|{ref.node_id}|{ref.activation_id}|{ref.attempt}"
