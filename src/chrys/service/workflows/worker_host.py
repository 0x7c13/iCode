# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow worker host: the standalone process that runs a user's workflow file.

Started as ``<interpreter> worker_host.py <sdk_dir>`` by the main process, on
any Python >= 3.9. This file is deliberately self-contained: standard library
only, no import of chrys, 3.9 syntax and runtime APIs only, because it has to
run inside a user's own interpreter. Its protocol constants are a copy of
``chrys.service.workflows.protocol``; a contract test keeps the two in step.

Bootstrap order matters and is fixed:

1. ``protocol_fd = dup(1)``: the protocol writer holds a private descriptor.
2. ``dup2`` fd 1 and fd 2 onto a capture pipe drained into a bounded tail, so
   ``os.write(1, ...)``, C extensions and grandchildren can never reach the
   protocol channel.
3. Replace ``sys.stdout``/``sys.stderr`` with a context-local capture that
   attributes Python-level output to the running attempt.
4. Prepend the injected SDK artifact to ``sys.path`` and import
   ``chrys.workflows``; the hello frame reports whether that import resolved
   to the artifact (a stale ``chrys`` in the user's environment fails here).
5. Serve requests: one reader thread, one writer thread, an asyncio loop for
   async bodies, a pool for sync bodies/offloads and a separate evaluation pool. Every attempt-bound message
   carries an ``AttemptRef``; a message for a terminal attempt is dropped.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import contextvars
import inspect
import io
import json
import linecache
import math
import os
import platform
import queue
import re
import sys
import threading
import time
import traceback
import types
import unicodedata
import warnings
from collections import deque
from collections.abc import Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, ClassVar, Deque, Dict, List, Optional, Set, Tuple, Union

PROTOCOL_VERSION = 2

METHOD_HELLO = "hello"
METHOD_LOAD = "load"
METHOD_RUN_PYTHON = "run_python"
METHOD_EVAL_OUTGOING = "eval_outgoing"
METHOD_EVAL_LOOP_UNTIL = "eval_loop_until"
METHOD_COMBINE = "combine"
METHOD_CANCEL = "cancel"
METHOD_NATIVE_OUTPUT = "native_output"
METHOD_SHUTDOWN = "shutdown"
METHOD_ASK = "ask"
METHOD_EMIT = "emit"

ERROR_USER_EXCEPTION = "user_exception"
ERROR_VALUE_NOT_SERIALIZABLE = "value_not_serializable"
ERROR_VALUE_TOO_LARGE = "value_too_large"
ERROR_PROTOCOL_LIMIT = "protocol_limit"
ERROR_ASK_UNAVAILABLE = "ask_unavailable"
ERROR_ATTEMPT_TERMINATED = "attempt_terminated"
ERROR_LOAD_FAILED = "load_failed"
ERROR_INVALID_PARAMS = "invalid_params"
ERROR_UNKNOWN_METHOD = "unknown_method"
ERROR_INTERNAL = "internal"

_MIB = 1024 * 1024

LIMITS = {
    "max_frame_bytes": 16 * _MIB,
    "max_payload_depth": 64,
    "max_payload_items": 200_000,
    "max_string_chars": 4 * _MIB,
    "max_pending_requests": 256,
    "max_run_frames": 1_000_000,
    "max_run_bytes": 4 * 1024 * _MIB,
    "max_join_retained_bytes": 12 * _MIB,
    "emit_rate_per_s": 200,
    "max_emits_per_attempt": 10_000,
    "max_emit_bytes_per_attempt": 8 * _MIB,
    "eval_deadline": 30.0,  # seconds
    "worker_thread_pool_size": 8,
    "eval_thread_pool_size": 2,
    "worker_leak_budget": 8,
    "eval_leak_budget": 2,
    "load_timeout": 60.0,  # seconds
    "hello_timeout": 15.0,  # seconds
    "shutdown_grace": 3.0,  # seconds
    "cancel_drain": 1.0,  # seconds
    "captured_output_bytes": 64 * 1024,
    "native_output_tail_bytes": 64 * 1024,
}

SDK_EXPORTS = (
    "Answer",
    "BuilderScope",
    "NodeContext",
    "NodeHandle",
    "Option",
    "Question",
    "Retry",
    "SourceValue",
    "Workflow",
    "WorkflowBuilder",
    "WorkflowValue",
)

MODULE_NAME = "__chrys_workflow__"
ENTRY_ENV = "CHRYS_WORKFLOW_ENTRY"  # the entry path, for multiprocessing children of the host
_LOAD_PHASE = "load"
_CURRENT_ATTEMPT: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("chrys_workflow_attempt", default=None)


def _canonical(payload: Any) -> bytes:
    # surrogatepass mirrors the protocol's frame codec: every str crosses byte for byte, values are checked upstream.
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return text.encode("utf-8", "surrogatepass")


def _dumps(payload: Any) -> bytes:
    return _canonical(payload) + b"\n"


def _clean(text: str) -> str:
    """Diagnostic text (messages, tracebacks) must always fit a UTF-8 frame."""
    return text.encode("utf-8", "replace").decode("utf-8")


def _answers_fit(answers: List[Any], count: int) -> bool:
    """Whether an ask reply's ``answers`` has the shape the SDK builds ``Answer`` objects from."""
    return len(answers) == count and all(
        isinstance(answer, dict)
        and isinstance(answer.get("selected"), list)
        and all(isinstance(label, str) for label in answer["selected"])
        and isinstance(answer.get("text"), str)
        for answer in answers
    )


def _diagnostic(text: str, *, tail: bool = False) -> str:
    """Cap diagnostic text at captured_output_bytes characters so an envelope always fits a frame."""
    text = _clean(text)
    limit = int(LIMITS["captured_output_bytes"])
    if len(text) <= limit:
        return text
    note = "... [" + str(len(text) - limit) + " chars dropped]"
    return (note + "\n" + text[-limit:]) if tail else (text[:limit] + note)  # a traceback keeps its end


def _verdict(result: Any, what: str) -> bool:
    """Conditions are sync: an awaitable is a bug in the workflow file, never a truthy value."""
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError(what + " returned an awaitable; conditions must be sync functions.")
    return bool(result)


class _Invalid(Exception):
    """A request the host cannot act on; answered with ``invalid_params``."""


class _NotSerializable(Exception):
    pass


class _TooLarge(Exception):
    pass


class _AskError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _Terminated(Exception):
    """The attempt was cancelled earlier; answered with ``attempt_terminated``."""


class _EmitLimit(Exception):
    pass


class _LoopExit(Exception):
    """SystemExit or KeyboardInterrupt raised inside a body task.

    asyncio lets those two escape the event loop itself, which would take the whole worker down with one
    node; the body task re-raises them as this ordinary exception so the attempt fails like any other.
    """

    def __init__(self, exc: BaseException) -> None:
        super().__init__(type(exc).__name__ + ": " + str(exc))
        self.formatted = traceback.format_exc()


async def _awaited(awaitable: Any) -> Any:
    return await awaitable


class _GuardedCoroutine(Coroutine[Any, Any, Any]):
    """The coroutine a task drives, with SystemExit/KeyboardInterrupt turned into _LoopExit at every step.

    A wrapper object rather than a wrapping coroutine: Task.throw() on a task cancelled before its first step
    must reach the original coroutine and close it, exactly as the default task factory would.
    """

    def __init__(self, coro: Any) -> None:
        self._coro = coro

    def send(self, value: Any) -> Any:
        return self._step(self._coro.send, value)

    def throw(self, typ: Any, val: Any = None, tb: Any = None) -> Any:
        args = (typ,) if val is None and tb is None else (typ, val, tb)
        return self._step(self._coro.throw, *args)

    def close(self) -> None:
        self._coro.close()

    def __await__(self) -> Any:
        return self._coro.__await__()

    def __getattr__(self, name: str) -> Any:
        # cr_frame, cr_code and the names: asyncio reads them for task reprs and get_stack().
        return getattr(self._coro, name)

    @staticmethod
    def _step(advance: Callable[..., Any], *args: Any) -> Any:
        try:
            return advance(*args)
        except (SystemExit, KeyboardInterrupt) as exc:
            raise _LoopExit(exc) from None


def _task_factory(loop: "asyncio.AbstractEventLoop", coro: Any, **kwargs: Any) -> "asyncio.Task[Any]":
    """Wrap every task the loop creates in _GuardedCoroutine.

    A child task (asyncio.gather, create_task, ensure_future) that raises SystemExit re-raises it out of
    Task.__step, past run_forever, killing the worker; wrapping the coroutine turns it into an ordinary failure.
    """
    return asyncio.Task(_GuardedCoroutine(coro), loop=loop, **kwargs)


def _is_finalizer(task: "asyncio.Task[Any]") -> bool:
    """Whether the task is an async generator's aclose(), scheduled by the loop when the generator was dropped."""
    coro = task.get_coro()
    inner = getattr(coro, "_coro", coro)  # through the _GuardedCoroutine wrapper
    return type(inner).__name__ == "async_generator_athrow"


# ---------------------------------------------------------------- values


def _value_from_wire(payload: Any, sdk: Any) -> Any:
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise _Invalid("value must be an object with a text field.")
    return sdk.WorkflowValue(text=payload["text"], data=payload.get("data"))


def _value_to_wire(value: Any) -> Dict[str, Any]:
    return {"text": value.text, "data": value.data}


def _sources_from_wire(payload: Any, sdk: Any) -> List[Any]:
    if not isinstance(payload, list):
        raise _Invalid("sources must be a list.")
    sources = []
    for item in payload:
        if not isinstance(item, dict):
            raise _Invalid("source must be an object.")
        sources.append(
            sdk.SourceValue(
                node_id=item["node_id"],
                activation_id=item["activation_id"],
                value=_value_from_wire(item["value"], sdk),
            )
        )
    return sources


class _ShapeCounter:
    def __init__(self) -> None:
        self.items = 0


def _check_json(payload: Any, depth: int, counter: _ShapeCounter) -> None:
    """Strict JSON plus the payload limits; raises _NotSerializable / _TooLarge."""
    counter.items += 1
    if counter.items > LIMITS["max_payload_items"]:
        raise _TooLarge("data has more than max_payload_items members.")
    if depth > LIMITS["max_payload_depth"]:
        raise _TooLarge("data is nested deeper than max_payload_depth.")
    if payload is None or isinstance(payload, (bool, int)):
        return
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise _NotSerializable("data contains a non-finite float.")
        return
    if isinstance(payload, str):
        if len(payload) > LIMITS["max_string_chars"]:
            raise _TooLarge("data contains a string longer than max_string_chars.")
        _require_utf8(payload, "data contains a string")
        return
    if isinstance(payload, list):
        for item in payload:
            _check_json(item, depth + 1, counter)
        return
    if isinstance(payload, dict):
        for key, item in payload.items():
            if not isinstance(key, str):
                raise _NotSerializable("data keys must be strings.")
            _require_utf8(key, "data contains a key")
            _check_json(item, depth + 1, counter)
        return
    raise _NotSerializable("data contains a non-JSON value: " + type(payload).__name__)


def _require_utf8(text: str, what: str) -> None:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise _NotSerializable(what + " that is not valid Unicode (lone surrogate).") from None


def _coerce_value(result: Any, sdk: Any, what: str) -> Any:
    """Apply the return-value ABI: ``str`` is sugar for ``WorkflowValue(text=str)``."""
    if isinstance(result, str):
        result = sdk.WorkflowValue(text=result)
    elif not isinstance(result, sdk.WorkflowValue):
        raise _NotSerializable(what + " must return str or WorkflowValue, got " + type(result).__name__ + ".")
    if not isinstance(result.text, str):
        raise _NotSerializable(what + " returned a WorkflowValue whose text is not a str.")
    if len(result.text) > LIMITS["max_string_chars"]:
        raise _TooLarge(what + " returned text longer than max_string_chars.")
    _require_utf8(result.text, what + " returned text")
    _check_json(result.data, 0, _ShapeCounter())
    return result


# ---------------------------------------------------------------- output capture


class _BoundedText:
    """Text sink that keeps the first ``limit`` bytes and remembers it overflowed.

    Bytes are decoded here, per sink, so a character split across two byte writes of one attempt
    is reassembled by that attempt's own decoder and never completed with another attempt's bytes.
    A byte sequence still open when text is written, or when the snapshot is taken, shows as the
    replacement character where it was written; it is never completed by bytes written later.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._chunks: List[str] = []
        self._size = 0
        self.truncated = False
        self._lock = threading.Lock()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def write(self, chunk: Union[str, bytes]) -> None:
        with self._lock:
            if self.truncated:
                return
            if isinstance(chunk, bytes):
                self._append(self._decoder.decode(chunk))
            else:
                self._append(self._decoder.decode(b"", final=True) + chunk)

    def _append(self, text: str) -> None:
        if not text:
            return
        data = text.encode("utf-8", "replace")
        if self._size + len(data) > self._limit:
            self.truncated = True
            data = data[: self._limit - self._size]
        self._chunks.append(data.decode("utf-8", "ignore"))
        self._size += len(data)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if not self.truncated:
                self._append(self._decoder.decode(b"", final=True))
            return {"text": "".join(self._chunks), "truncated": self.truncated}


class _NativeTail:
    """Drains the fd 1/2 capture pipe into a bounded tail; never blocks a writer for long."""

    def __init__(self, read_fd: int, limit: int) -> None:
        self._fd = read_fd
        self._limit = limit
        self._tail = bytearray()
        self._barrier = b""
        self._drained: Optional[Callable[[], None]] = None
        self.dropped = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="chrys-workflow-native-tail", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                chunk = os.read(self._fd, 65536)
            except OSError:
                return
            if not chunk:
                return
            with self._lock:
                self._tail.extend(chunk)
                if self._barrier:
                    offset = self._tail.find(self._barrier)
                    if offset >= 0:
                        del self._tail[offset : offset + len(self._barrier)]
                        self._barrier = b""
                        if self._drained is not None:
                            self._drained()
                retained = self._limit + len(self._barrier)
                if len(self._tail) > retained:
                    self.dropped += len(self._tail) - retained
                    del self._tail[: len(self._tail) - retained]

    async def snapshot(self) -> Dict[str, Any]:
        # A pipe barrier proves all previously written native bytes reached the reader,
        # even when inherited write descriptors keep the pipe open in grandchildren.
        loop = asyncio.get_running_loop()
        drained = asyncio.Event()

        def notify_drained() -> None:
            loop.call_soon_threadsafe(drained.set)

        barrier = b"\x00chrys-drain-" + os.urandom(32) + b"\x00"
        with self._lock:
            self._barrier = barrier
            self._drained = notify_drained
        view = memoryview(barrier)
        while view:
            view = view[os.write(2, view) :]
        try:
            await asyncio.wait_for(drained.wait(), 1.0)
        finally:
            with self._lock:
                self._drained = None
        with self._lock:
            return {"text": bytes(self._tail).decode("utf-8", "replace"), "dropped_bytes": self.dropped}


class _CaptureStream(io.TextIOBase):
    """``sys.stdout``/``sys.stderr`` replacement attributing writes to the current attempt.

    ``fileno()`` is the descriptor the stream stands in for (already redirected to the native tail), so a
    child given ``stdout=sys.stdout`` inherits the tail; ``buffer`` takes bytes into the same attribution.
    """

    def __init__(self, route: Callable[[Union[str, bytes]], None], fd: int) -> None:
        super().__init__()
        self._route = route
        self._fd = fd
        self.buffer = _CaptureBuffer(route, fd)

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return "utf-8"

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._fd

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("write() argument must be str")
        self._route(text)
        return len(text)

    def flush(self) -> None:
        return None


class _CaptureBuffer(io.BufferedIOBase):
    """The ``buffer`` of a _CaptureStream: bytes routed to the same owner, decoded by that owner's sink."""

    def __init__(self, route: Callable[[Union[str, bytes]], None], fd: int) -> None:
        super().__init__()
        self._route = route
        self._fd = fd

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._fd

    def write(self, data: Any) -> int:
        view = memoryview(data)
        self._route(view.tobytes())
        return view.nbytes

    def flush(self) -> None:
        return None


# ---------------------------------------------------------------- attempts


class _Attempt:
    def __init__(self, key: str, ref: Dict[str, Any], ref_key: str, request_id: int) -> None:
        self.key = key
        self.ref = ref
        self.ref_key = ref_key
        self.request_id = request_id
        self.terminal = False
        self.task: Optional[asyncio.Task] = None
        self.work: Optional["Future[Any]"] = None  # the pool entry; withdrawable while still queued
        self.work_pool = "body"
        self.offloads: List[Tuple["Future[Any]", str]] = []  # future and its actual executor pool
        self.children: Set["asyncio.Task[Any]"] = set()  # tasks created in the attempt's context, still pending
        self.unwound: Optional["asyncio.Future[int]"] = None  # the one unwind of those tasks (see _unwind)
        self.leaked_by_pool = {"body": 0, "eval": 0}
        self.stdout = _BoundedText(int(LIMITS["captured_output_bytes"]))
        self.last_emit_ordinal = 0
        self.emit_bytes = 0
        self.emit_times: Deque[float] = deque()
        self.limit_failure: Optional[str] = None


def _failure_of(exc: BaseException) -> Tuple[str, str, Optional[str]]:
    if isinstance(exc, _NotSerializable):
        return ERROR_VALUE_NOT_SERIALIZABLE, str(exc), None
    if isinstance(exc, _TooLarge):
        return ERROR_VALUE_TOO_LARGE, str(exc), None
    if isinstance(exc, _LoopExit):
        return ERROR_USER_EXCEPTION, str(exc), exc.formatted
    return ERROR_USER_EXCEPTION, type(exc).__name__ + ": " + str(exc), traceback.format_exc()


def _ref_of(params: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    ref = params.get("ref")
    if not isinstance(ref, dict):
        raise _Invalid("ref must be an object.")
    return ref, _ref_key(ref)


def _ref_key(ref: Any) -> str:
    if not isinstance(ref, dict):
        raise _Invalid("ref must be an object.")
    try:
        parts = (ref["run_id"], ref["node_id"], ref["activation_id"], ref["attempt"])
    except KeyError as exc:
        raise _Invalid("ref is missing " + repr(exc.args[0]) + ".") from exc
    if not all(isinstance(part, str) and part for part in parts[:3]):
        raise _Invalid("ref ids must be non-empty strings.")
    if not isinstance(parts[3], int) or isinstance(parts[3], bool) or parts[3] < 1:
        raise _Invalid("ref attempt must be an int >= 1.")
    return "|".join((parts[0], parts[1], parts[2], str(parts[3])))


# ---------------------------------------------------------------- host


class _Pool(ThreadPoolExecutor):
    """One bounded pool, with submission ownership retained for leak accounting.

    The body pool is also the default executor for ``asyncio.to_thread``. Evaluation work uses a
    separate pool, but any offloads it creates still belong to the executor they were submitted to.
    """

    def __init__(
        self, max_workers: int, loop: asyncio.AbstractEventLoop, charge: Callable[["Future[Any]", str], None], kind: str
    ) -> None:
        super().__init__(max_workers=max_workers)
        self._loop = loop
        self._charge = charge
        self.kind = kind

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> "Future[Any]":
        future = super().submit(self._on_thread, fn, *args, **kwargs)
        self._charge(future, self.kind)
        return future

    def _on_thread(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        # Python 3.9 binds an asyncio primitive to the thread's current loop when it is built, so a workflow
        # file's module-level Semaphore or Queue (built on a pool thread by the load) must see the loop that
        # will run the async bodies. Set per task: a body's own asyncio.run() clears the thread's loop.
        asyncio.set_event_loop(self._loop)
        return fn(*args, **kwargs)


class Host:
    def __init__(self, sdk_dir: str, input_fd: int, protocol_fd: int, diag_fd: int, native: _NativeTail) -> None:
        self.sdk_dir = sdk_dir
        self.pid = os.getpid()
        self.input_fd = input_fd
        self.protocol_fd = protocol_fd
        self.diag_fd = diag_fd
        self.native = native
        self.loop = asyncio.new_event_loop()
        self.loop.set_task_factory(self._new_task)  # SystemExit inside any task fails the attempt, not the worker
        self.pool = _Pool(int(LIMITS["worker_thread_pool_size"]), self.loop, self._charge, "body")
        self.eval_pool = _Pool(int(LIMITS["eval_thread_pool_size"]), self.loop, self._charge, "eval")
        self.loop.set_default_executor(self.pool)  # asyncio.to_thread shares (and is bounded by) the budget
        self.outbox: queue.Queue = queue.Queue()
        self.writer = threading.Thread(target=self._write_loop, name="chrys-workflow-writer", daemon=True)
        self.reader = threading.Thread(target=self._read_loop, name="chrys-workflow-reader", daemon=True)
        self.sdk: Any = None
        self.definition: Any = None
        self.load_stdout = _BoundedText(int(LIMITS["captured_output_bytes"]))
        self.attempts: Dict[str, _Attempt] = {}
        self.charging: Dict[str, _Attempt] = {}  # attempts whose leak verdict is not settled yet
        self.settled_keys: Set[str] = set()
        self.cancelled_refs: Set[str] = set()
        self.cancelling: Dict[str, "asyncio.Task[Dict[str, Any]]"] = {}  # ref key -> the drain under way
        self.emit_lock = threading.Lock()
        self.next_id = 2
        self.pending_asks: Dict[int, asyncio.Future] = {}

    # -- wiring --------------------------------------------------------------

    def send(self, frame: Dict[str, Any]) -> None:
        self.outbox.put(_dumps(frame))

    def _write_loop(self) -> None:
        while True:
            data = self.outbox.get()
            if data is None:
                return
            view = memoryview(data)
            try:
                while view:
                    written = os.write(self.protocol_fd, view)
                    view = view[written:]
            except OSError:
                os._exit(1)

    def _read_loop(self) -> None:
        buffer = bytearray()
        limit = LIMITS["max_frame_bytes"]
        while True:
            try:
                chunk = os.read(self.input_fd, 65536)
            except OSError:
                chunk = b""
            if not chunk:
                self.loop.call_soon_threadsafe(self._on_eof)
                return
            buffer.extend(chunk)
            while True:
                cut = buffer.find(b"\n")
                if cut < 0:
                    break
                line = bytes(buffer[:cut])
                del buffer[: cut + 1]
                self.loop.call_soon_threadsafe(self._on_line, line)
            if len(buffer) > limit:
                self.loop.call_soon_threadsafe(self._fatal, "incoming frame exceeds max_frame_bytes")
                return

    def _fatal(self, message: str) -> None:
        with contextlib.suppress(OSError):
            os.write(self.diag_fd, ("chrys workflow host: " + message + "\n").encode("utf-8", "replace"))
        os._exit(2)

    def _on_eof(self) -> None:
        # The main process is gone or closed the channel: nothing left to answer.
        os._exit(0)

    def _on_line(self, line: bytes) -> None:
        try:
            frame = json.loads(line.decode("utf-8", "surrogatepass"))
        except (UnicodeDecodeError, ValueError):
            self._fatal("incoming frame is not UTF-8 JSON")
            return
        if not isinstance(frame, dict):
            self._fatal("incoming frame is not an object")
            return
        if "method" in frame:
            request_id = frame.get("id")
            if not isinstance(request_id, int) or isinstance(request_id, bool):
                self._fatal("incoming request has no id")
                return
            params = frame.get("params")
            if not isinstance(params, dict):
                self.send(_error(request_id, ERROR_INVALID_PARAMS, "params must be an object."))
                return
            self.loop.create_task(self._serve(request_id, frame["method"], params))
            return
        future = self.pending_asks.pop(frame.get("id"), None)
        if future is not None and not future.done():
            future.set_result(frame)

    async def _serve(self, request_id: int, method: str, params: Dict[str, Any]) -> None:
        handler = self._handlers.get(method)
        if handler is None:
            self.send(_error(request_id, ERROR_UNKNOWN_METHOD, "unknown method " + repr(method) + "."))
            return
        try:
            result = await handler(self, request_id, params)
        except _Invalid as exc:
            self.send(_error(request_id, ERROR_INVALID_PARAMS, str(exc)))
        except _Terminated as exc:
            self.send(_error(request_id, ERROR_ATTEMPT_TERMINATED, str(exc)))
        except Exception as exc:
            data = {"traceback": _diagnostic(traceback.format_exc(), tail=True)}
            self.send(_error(request_id, ERROR_INTERNAL, _diagnostic("host failure: " + repr(exc)), data))
        else:
            if result is not None:
                self._respond(request_id, result)

    def _respond(self, request_id: int, result: Dict[str, Any]) -> None:
        data = _dumps({"id": request_id, "result": result})
        if len(data) > LIMITS["max_frame_bytes"]:
            self.send(_error(request_id, ERROR_VALUE_TOO_LARGE, "result frame exceeds max_frame_bytes."))
            return
        self.outbox.put(data)

    # -- output routing --------------------------------------------------------

    def _route_output(self, chunk: Union[str, bytes]) -> None:
        if os.getpid() != self.pid:
            # A forked child writes into its own copy of the capture sinks, which nobody reads.
            self._write_native(chunk)
            return
        key = _CURRENT_ATTEMPT.get()
        if key == _LOAD_PHASE:
            self.load_stdout.write(chunk)
            return
        attempt = self.attempts.get(key) if key is not None else None
        if attempt is not None and not attempt.terminal:
            attempt.stdout.write(chunk)
            return
        if key is not None:
            # A terminal attempt (cancelled, timed out) keeps writing: dropped.
            return
        self._write_native(chunk)  # nobody owns this write

    @staticmethod
    def _write_native(chunk: Union[str, bytes]) -> None:
        """Write through fd 1, which joins the run-level native tail."""
        with contextlib.suppress(OSError):
            os.write(1, chunk if isinstance(chunk, bytes) else chunk.encode("utf-8", "replace"))

    # -- handlers ------------------------------------------------------------------

    async def _load(self, request_id: int, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.definition is not None:
            raise _Invalid("workflow already loaded.")
        source, filename, workspace = params.get("source"), params.get("filename"), params.get("workspace")
        if not isinstance(source, str) or not isinstance(filename, str) or not isinstance(workspace, str):
            raise _Invalid("load needs source, filename and workspace strings.")
        bytecode_cache, package_dir = params.get("bytecode_cache"), params.get("package_dir")
        if not isinstance(bytecode_cache, str) or not isinstance(package_dir, (str, type(None))):
            raise _Invalid("load needs a bytecode_cache string and an optional package_dir string.")
        diagnose, precompile = params.get("diagnose", False), params.get("precompile", [])
        if not isinstance(diagnose, bool) or not (
            isinstance(precompile, list) and all(isinstance(path, str) for path in precompile)
        ):
            raise _Invalid("load takes an optional diagnose boolean and an optional precompile list of strings.")
        problem = _bytecode_cache_problem(bytecode_cache, package_dir)
        if problem is not None:
            self.send(_error(request_id, ERROR_LOAD_FAILED, problem, {"stdout": self.load_stdout.snapshot()}))
            return None
        context = contextvars.copy_context()
        context.run(_CURRENT_ATTEMPT.set, _LOAD_PHASE)
        try:
            os.chdir(workspace)
        except OSError as exc:
            raise _Invalid("workspace is not usable: " + str(exc)) from exc
        # Before the workflow's folder joins sys.path: a file of it named like a stdlib module must not shadow it here.
        places = _Places(filename, self.sdk_dir) if diagnose else None
        sys.path.insert(0, os.path.dirname(filename))
        os.environ[ENTRY_ENV] = filename
        if places is not None:
            broken = await self.loop.run_in_executor(self.pool, _precompile, source, precompile, places)
            if broken:
                data = {"traceback": "", "stdout": self.load_stdout.snapshot()}
                data.update(_diagnostics_data(broken))
                self.send(_error(request_id, ERROR_LOAD_FAILED, broken[0]["message"], data))
                return None
        module = _entry_module(filename)

        def run_module() -> None:
            _exec_entry(source, filename, module)

        try:
            await self.loop.run_in_executor(self.pool, context.run, run_module)
        except BaseException as exc:
            data: Dict[str, Any] = {
                "traceback": _diagnostic(traceback.format_exc(), tail=True),
                "stdout": self.load_stdout.snapshot(),
            }
            to_dict = getattr(exc, "to_dict", None)
            if isinstance(exc, self.sdk.WorkflowValidationError) and callable(to_dict):
                data["validation"] = {
                    field: _diagnostic(text) if isinstance(text, str) else text
                    for field, text in to_dict().items()
                    if field in ("message", "location")
                }
            message = _diagnostic(type(exc).__name__ + ": " + str(exc))
            if places is None:
                self.send(_error(request_id, ERROR_LOAD_FAILED, message, data))
                return None
            data["traceback"] = _diagnostic(_user_traceback(exc, places), tail=True)
            data.update(_diagnostics_data([_failure_diagnostic(exc, places, self.sdk.WorkflowValidationError)]))
            self.send(_error(request_id, ERROR_LOAD_FAILED, message, data))
            return None
        found = module.__dict__.get("workflow", _MISSING)
        if not isinstance(found, self.sdk.Workflow):
            message = "the workflow file must bind a module-level `workflow` to the result of build()."
            data = {"stdout": self.load_stdout.snapshot()}
            if diagnose:
                data.update(_diagnostics_data([_missing_workflow_diagnostic(found, filename)]))
            self.send(_error(request_id, ERROR_LOAD_FAILED, message, data))
            return None
        self.definition = found.definition
        result = {"manifest": found.manifest(), "stdout": self.load_stdout.snapshot()}
        if diagnose:
            return _with_sites(request_id, result, _declaration_sites(found.definition), int(LIMITS["max_frame_bytes"]))
        return result

    async def _run_python(self, request_id: int, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ref, ref_key = _ref_of(params)
        node = self._node(ref, "python")
        attempt = self._begin(ref, ref_key, request_id, "body")
        value = _value_from_wire(params.get("value"), self.sdk)
        ctx = self.sdk.NodeContext(
            emit=lambda text: self._emit(attempt, text),
            ask=lambda questions: self._ask(attempt, questions),
        )
        args = (value,) if node.fn_arity == 1 else (value, ctx)
        context = contextvars.copy_context()
        context.run(_CURRENT_ATTEMPT.set, attempt.key)
        result: Any = None
        failure: Optional[Tuple[str, str, Optional[str]]] = None
        try:
            if node.fn_is_async:
                attempt.task = context.run(self.loop.create_task, node.fn(*args))  # wrapped by _task_factory
                result = await attempt.task
            else:
                result = await self._submit(attempt, context.run, node.fn, *args)
                if inspect.isawaitable(result):
                    if attempt.terminal:
                        # Cancelled while the thread ran: its late awaitable must not start now.
                        if inspect.iscoroutine(result):
                            result.close()
                        return None
                    # A sync wrapper around an async body: the awaitable is awaited on the loop.
                    attempt.task = context.run(self.loop.create_task, _awaited(result))
                    result = await attempt.task
            result = _coerce_value(result, self.sdk, "python node " + repr(node.node_id))
        except asyncio.CancelledError:
            failure = (ERROR_USER_EXCEPTION, "body was cancelled from inside user code.", None)
        except _AskError as exc:
            failure = (exc.code, str(exc), None)
        except _EmitLimit as exc:
            failure = (ERROR_PROTOCOL_LIMIT, str(exc), None)
        except BaseException as exc:
            failure = _failure_of(exc)
        if attempt.terminal:
            return None
        stuck = await self._unwind(attempt)  # the body's tasks end with it, charged while they unwind
        if attempt.terminal:  # a cancel landed meanwhile: its drain settles the attempt and counts them
            return None
        attempt.leaked_by_pool["body"] += stuck
        if failure is not None:
            self._finish(attempt, *failure)
        elif attempt.limit_failure is not None:
            self._finish(attempt, ERROR_PROTOCOL_LIMIT, attempt.limit_failure)
        else:
            self._finish_ok(attempt, {"value": _value_to_wire(result)}, barrier=True)
        return None

    async def _eval_outgoing(self, request_id: int, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._require_loaded()
        ref, ref_key = _ref_of(params)
        edge_ids = params.get("edge_ids")
        if not isinstance(edge_ids, list) or not all(isinstance(edge_id, str) for edge_id in edge_ids):
            raise _Invalid("edge_ids must be a list of strings.")
        edges = []
        for edge_id in edge_ids:
            edge = self.definition.edges.get(edge_id)
            if edge is None:
                raise _Invalid("unknown edge " + repr(edge_id) + ".")
            edges.append(edge)
        value = _value_from_wire(params.get("value"), self.sdk)

        def evaluate() -> Dict[str, bool]:
            decisions: Dict[str, bool] = {}
            cases: Dict[str, List[Any]] = {}
            for edge in edges:
                if edge.switch_group is None:
                    decisions[edge.edge_id] = (
                        _verdict(edge.predicate(value), "edge condition") if edge.predicate is not None else False
                    )
                elif edge.switch_default:
                    decisions[edge.edge_id] = False  # the scheduler opens the default when no case hit
                else:
                    cases.setdefault(edge.switch_group, []).append(edge)
            for group in cases.values():
                # Switch cases are evaluated in declared order and stop at the first hit.
                matched = False
                for edge in sorted(group, key=lambda edge: edge.switch_position or 0):
                    hit = not matched and _verdict(edge.predicate(value), "switch case")
                    decisions[edge.edge_id] = hit
                    matched = matched or hit
            return decisions

        return await self._evaluate(
            ref, ref_key, request_id, "outgoing", evaluate, lambda decisions: {"decisions": decisions}
        )

    async def _eval_loop_until(self, request_id: int, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ref, ref_key = _ref_of(params)
        node = self._node(ref, "loop")
        iteration = params.get("iteration")
        if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 1:
            raise _Invalid("iteration must be an int >= 1.")
        value = _value_from_wire(params.get("value"), self.sdk)
        phase = "until#" + str(iteration)  # the loop keeps its ref across iterations
        return await self._evaluate(
            ref,
            ref_key,
            request_id,
            phase,
            lambda: _verdict(node.loop.until(value), "loop until"),
            lambda verdict: {"verdict": verdict},
        )

    async def _combine(self, request_id: int, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ref, ref_key = _ref_of(params)
        node = self._node(ref, "join")
        if node.fn is None:
            raise _Invalid("join " + repr(node.node_id) + " has no user combine.")
        sources = _sources_from_wire(params.get("sources"), self.sdk)
        what = "join combine " + repr(node.node_id)
        return await self._evaluate(
            ref,
            ref_key,
            request_id,
            "combine",
            lambda: _coerce_value(node.fn(sources), self.sdk, what),
            lambda value: {"value": _value_to_wire(value)},
        )

    async def _evaluate(
        self,
        ref: Dict[str, Any],
        ref_key: str,
        request_id: int,
        phase: str,
        compute: Callable[[], Any],
        shape: Callable[[Any], Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        attempt = self._begin(ref, ref_key, request_id, phase)
        context = contextvars.copy_context()
        context.run(_CURRENT_ATTEMPT.set, attempt.key)
        try:
            result = await self._submit(attempt, context.run, compute, pool=self.eval_pool)
        except BaseException as exc:
            self._finish(attempt, *_failure_of(exc))
            return None
        if attempt.terminal:
            return None
        self._finish_ok(attempt, shape(result))
        return None

    async def _cancel(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        key = _ref_key(params.get("ref"))
        self.cancelled_refs.add(key)
        drain = self.cancelling.get(key)
        if drain is None:
            # Overlapping cancels of one ref (a direct cancel and the node timeout) share one drain and one verdict.
            drain = self.loop.create_task(self._drain(key))
            self.cancelling[key] = drain
            drain.add_done_callback(lambda _task: self.cancelling.pop(key, None))
        return await drain

    async def _drain(self, key: str) -> Dict[str, Any]:
        live = [attempt for attempt in self.attempts.values() if attempt.ref_key == key]
        message = "attempt cancelled by the main process."
        draining: List[Tuple[_Attempt, "asyncio.Task[Any]", int]] = []
        for attempt in live:
            if attempt.task is not None:
                # Fenced now, answered later: the body's envelope and the ack both wait for its
                # cleanup to unwind, so nobody moves on while the finally blocks still run.
                draining.append((attempt, attempt.task, self._settle(attempt)))
                attempt.task.cancel()
                continue
            self._finish(attempt, ERROR_ATTEMPT_TERMINATED, message)  # withdraws a queued body, counts a running one
        if draining:
            _done, pending = await asyncio.wait([task for _, task, _ in draining], timeout=LIMITS["cancel_drain"])
            for attempt, task, ordinal in draining:
                # A coroutine that keeps swallowing the cancellation is as leaked as a stuck thread. The body's
                # tasks are unwound after it, so a finally block that awaits its own child still can.
                stuck = int(task in pending) + await self._unwind(attempt)
                attempt.leaked_by_pool["body"] += stuck
                self._send_failure(attempt, ordinal, ERROR_ATTEMPT_TERMINATED, message, None)
        return {
            "cancelled": bool(live),
            "leaked_threads_by_pool": {
                pool: sum(attempt.leaked_by_pool[pool] for attempt in live) for pool in ("body", "eval")
            },
        }

    async def _native_output(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        return await self.native.snapshot()

    async def _shutdown(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        self.loop.call_soon(self.loop.stop)
        return {"ok": True}

    _handlers: ClassVar[Dict[str, Callable[..., Any]]] = {}

    # -- attempt lifecycle -----------------------------------------------------

    def _require_loaded(self) -> None:
        if self.definition is None:
            raise _Invalid("no workflow loaded.")

    def _node(self, ref: Dict[str, Any], kind: str) -> Any:
        self._require_loaded()
        node = self.definition.nodes.get(ref["node_id"])
        if node is None or node.kind != kind:
            raise _Invalid("ref does not name a " + kind + " node.")
        return node

    def _begin(self, ref: Dict[str, Any], ref_key: str, request_id: int, phase: str) -> _Attempt:
        if ref_key in self.cancelled_refs:
            raise _Terminated("attempt " + ref_key + " was cancelled.")
        key = phase + "|" + ref_key
        if key in self.attempts or key in self.settled_keys:
            raise _Invalid("attempt " + key + " was already started.")
        attempt = _Attempt(key, ref, ref_key, request_id)
        self.attempts[key] = attempt
        self.charging[key] = attempt
        return attempt

    def _submit(
        self, attempt: _Attempt, fn: Callable[..., Any], *args: Any, pool: Optional[_Pool] = None
    ) -> "asyncio.Future[Any]":
        executor = self.pool if pool is None else pool
        attempt.work_pool = executor.kind
        attempt.work = executor.submit(fn, *args)
        return asyncio.wrap_future(attempt.work, loop=self.loop)

    def _new_task(self, loop: "asyncio.AbstractEventLoop", coro: Any, **kwargs: Any) -> "asyncio.Task[Any]":
        """A task created while an attempt is current is its child until it finishes (see _withdraw)."""
        task = _task_factory(loop, coro, **kwargs)
        key = _CURRENT_ATTEMPT.get()
        attempt = self.charging.get(key) if key is not None else None
        if attempt is not None:
            attempt.children.add(task)
            task.add_done_callback(attempt.children.discard)
        return task

    def _charge(self, future: "Future[Any]", pool: str) -> None:
        """Pool work submitted while an attempt is current (an async body offloading) is that attempt's thread.

        Charged until the verdict is settled, so an offload made while a cancelled body unwinds still counts.
        """
        key = _CURRENT_ATTEMPT.get()
        attempt = self.charging.get(key) if key is not None else None
        if attempt is not None:
            attempt.offloads.append((future, pool))

    async def _unwind(self, attempt: _Attempt) -> int:
        """Cancel the attempt's surviving child tasks (a failed gather's siblings, a fire-and-forget task) and wait
        for them to unwind, bounded like the body's own drain; returns how many would not.

        The attempt is still charged meanwhile, so a child's cleanup offload (``finally: await to_thread(...)``)
        is counted by the verdict like the body's own would be. A cancel that lands during the unwind joins it:
        a second cancellation would cut the child's finally block short. Tasks a finally block starts meanwhile
        are unwound in turn, within the same allowance. An async generator's aclose() is cleanup already under
        way: it gets the allowance instead of a cancel, as it would under plain asyncio.
        """
        if attempt.unwound is not None:
            return await attempt.unwound
        attempt.unwound = self.loop.create_future()
        deadline = self.loop.time() + float(LIMITS["cancel_drain"])
        stuck = 0
        try:
            while True:
                children = [task for task in attempt.children if task is not attempt.task and not task.done()]
                if not children:
                    break
                for task in children:
                    if not _is_finalizer(task):
                        task.cancel()
                _done, pending = await asyncio.wait(children, timeout=max(deadline - self.loop.time(), 0.0))
                if pending:
                    stuck = len(pending)
                    break
        finally:
            attempt.unwound.set_result(stuck)
        return stuck

    def _withdraw(self, attempt: _Attempt) -> None:
        """Withdraw the attempt's queued pool work; the entries already on a thread stay there and are counted.

        A child task never unwound (an evaluation's, see _unwind) is cancelled here: nothing of a settled
        attempt runs on. The ones an unwind gave up on are left alone; a second cancellation would only cut
        their cleanup short.
        """
        self.charging.pop(attempt.key, None)
        if attempt.unwound is None:
            for task in list(attempt.children):
                if task is not attempt.task and not _is_finalizer(task):
                    task.cancel()
        futures = list(attempt.offloads)
        if attempt.work is not None:
            futures.append((attempt.work, attempt.work_pool))
        attempt.work, attempt.offloads = None, []
        for future, pool in futures:
            if not future.cancel() and not future.done():
                attempt.leaked_by_pool[pool] += 1

    def _settle(self, attempt: _Attempt) -> int:
        """Close the attempt to new emits; returns the ordinal of its last queued emit."""
        # Under the emit lock: an emit that took an ordinal has queued its frame, and every
        # later emit sees the attempt terminal, so the envelope queued next covers exactly this ordinal.
        with self.emit_lock:
            attempt.terminal = True
            ordinal = attempt.last_emit_ordinal
        self.attempts.pop(attempt.key, None)
        self.settled_keys.add(attempt.key)
        return ordinal

    def _finish(self, attempt: _Attempt, code: str, message: str, tb: Optional[str] = None) -> None:
        """Send the terminal error envelope once; later completions of this attempt are dropped."""
        if attempt.terminal:
            return
        self._send_failure(attempt, self._settle(attempt), code, message, tb)

    def _send_failure(self, attempt: _Attempt, ordinal: int, code: str, message: str, tb: Optional[str]) -> None:
        self._withdraw(attempt)
        data: Dict[str, Any] = {
            "stdout": attempt.stdout.snapshot(),
            "last_emit_ordinal": ordinal,
            "leaked_threads_by_pool": dict(attempt.leaked_by_pool),
        }
        if tb is not None:
            data["traceback"] = _diagnostic(tb, tail=True)
        self.send(_error(attempt.request_id, code, _diagnostic(message), data))

    def _finish_ok(self, attempt: _Attempt, result: Dict[str, Any], *, barrier: bool = False) -> None:
        """Send the success envelope, or the same-shaped failure when it cannot be framed.

        Every envelope carries the attempt's leak verdict (a predicate or combine can leave an offload on a
        thread too); every phase retains its captured stdout. ``barrier`` adds the body's emit high-water mark.
        """
        ordinal = self._settle(attempt)
        self._withdraw(attempt)
        result = {
            **result,
            "leaked_threads_by_pool": dict(attempt.leaked_by_pool),
            "stdout": attempt.stdout.snapshot(),
        }
        if barrier:
            result = {**result, "last_emit_ordinal": ordinal}
        try:
            data = _dumps({"id": attempt.request_id, "result": result})
        except (TypeError, ValueError, OverflowError) as exc:
            self._send_failure(
                attempt, ordinal, ERROR_VALUE_NOT_SERIALIZABLE, "result cannot be serialized: " + str(exc), None
            )
            return
        if len(data) > LIMITS["max_frame_bytes"]:
            self._send_failure(attempt, ordinal, ERROR_VALUE_TOO_LARGE, "result frame exceeds max_frame_bytes.", None)
            return
        self.outbox.put(data)

    def _emit(self, attempt: _Attempt, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("ctx.emit() takes a str.")
        with self.emit_lock:
            if attempt.terminal:
                return
            now = time.monotonic()
            while attempt.emit_times and now - attempt.emit_times[0] >= 1.0:
                attempt.emit_times.popleft()
            attempt.emit_bytes += len(text.encode("utf-8", "replace"))
            if len(attempt.emit_times) >= LIMITS["emit_rate_per_s"]:
                attempt.limit_failure = "ctx.emit() exceeded emit_rate_per_s."
            elif attempt.last_emit_ordinal >= LIMITS["max_emits_per_attempt"]:
                attempt.limit_failure = "ctx.emit() exceeded max_emits_per_attempt."
            elif attempt.emit_bytes > LIMITS["max_emit_bytes_per_attempt"]:
                attempt.limit_failure = "ctx.emit() exceeded max_emit_bytes_per_attempt."
            if attempt.limit_failure is not None:
                raise _EmitLimit(attempt.limit_failure)
            ordinal = attempt.last_emit_ordinal + 1
            try:
                text.encode("utf-8")  # the value rule, not the frame's: what leaves the worker is valid Unicode
            except UnicodeEncodeError:
                raise ValueError("ctx.emit() text is not valid Unicode (lone surrogate).") from None
            frame = {"method": METHOD_EMIT, "params": {"ref": attempt.ref, "ordinal": ordinal, "text": text}}
            data = _dumps(frame)  # before the ordinal is taken: the barrier counts only queued emits
            if len(data) > LIMITS["max_frame_bytes"]:
                attempt.limit_failure = "ctx.emit() frame exceeds max_frame_bytes."
                raise _EmitLimit(attempt.limit_failure)
            attempt.emit_times.append(now)
            attempt.last_emit_ordinal = ordinal
            self.outbox.put(data)

    async def _ask(self, attempt: _Attempt, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transport one SDK-serialized ``ctx.ask``; the SDK owns the question schema."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not self.loop:
            raise _AskError(ERROR_ASK_UNAVAILABLE, "ctx.ask() can only be awaited inside an async def node body.")
        if attempt.terminal:
            raise _AskError(ERROR_ATTEMPT_TERMINATED, "attempt is already terminal.")
        request_id = self.next_id
        self.next_id += 2
        frame = {"id": request_id, "method": METHOD_ASK, "params": {"ref": attempt.ref, "questions": questions}}
        data = _dumps(frame)  # checked before anything is registered or queued
        if len(data) > LIMITS["max_frame_bytes"]:
            raise _AskError(ERROR_PROTOCOL_LIMIT, "ctx.ask() questions exceed max_frame_bytes.")
        future: asyncio.Future = self.loop.create_future()
        self.pending_asks[request_id] = future
        self.outbox.put(data)
        try:
            reply = await future
        finally:
            self.pending_asks.pop(request_id, None)
        error = reply.get("error")
        if isinstance(error, dict):
            raise _AskError(str(error.get("code", ERROR_ASK_UNAVAILABLE)), str(error.get("message", "")))
        result = reply.get("result")
        answers = result.get("answers") if isinstance(result, dict) else None
        if not isinstance(answers, list) or not _answers_fit(answers, len(questions)):
            raise _AskError(ERROR_ASK_UNAVAILABLE, "ask reply carried no answers.")
        return answers

    # -- bootstrap ------------------------------------------------------------------

    def import_sdk(self) -> Tuple[bool, Optional[str]]:
        sys.path.insert(0, self.sdk_dir)
        try:
            import chrys.workflows as sdk
        except BaseException as exc:
            return False, "import chrys.workflows failed: " + type(exc).__name__ + ": " + str(exc)
        origin = os.path.realpath(getattr(sdk, "__file__", None) or "")
        expected = os.path.realpath(self.sdk_dir)
        if not origin.startswith(expected + os.sep):
            return False, "chrys.workflows resolved to " + repr(origin) + " instead of the injected SDK."
        missing = [name for name in SDK_EXPORTS if not hasattr(sdk, name)]
        if missing:
            return False, "injected SDK lacks exports: " + ", ".join(missing)
        self.sdk = sdk
        return True, None

    def hello(self, sdk_origin_ok: bool, sdk_origin_error: Optional[str]) -> Dict[str, Any]:
        return {
            "method": METHOD_HELLO,
            "params": {
                "protocol_version": PROTOCOL_VERSION,
                "python_version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": sys.platform,
                "sdk_origin_ok": sdk_origin_ok,
                "sdk_origin_error": sdk_origin_error,
            },
        }

    def serve_forever(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.writer.start()
        self.reader.start()
        self.loop.run_forever()
        # A stuck sync body keeps its pool thread forever; 3.9's asyncio.run
        # would wait for it. Drain the writer, then leave without joining.
        self.outbox.put(None)
        self.writer.join(timeout=LIMITS["shutdown_grace"])
        os._exit(0)


Host._handlers = {
    METHOD_LOAD: Host._load,
    METHOD_RUN_PYTHON: Host._run_python,
    METHOD_EVAL_OUTGOING: Host._eval_outgoing,
    METHOD_EVAL_LOOP_UNTIL: Host._eval_loop_until,
    METHOD_COMBINE: Host._combine,
    METHOD_CANCEL: Host._cancel,
    METHOD_NATIVE_OUTPUT: Host._native_output,
    METHOD_SHUTDOWN: Host._shutdown,
}


def _error(request_id: int, code: str, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"id": request_id, "error": {"code": code, "message": message, "data": data or {}}}


# -- load diagnostics ----------------------------------------------------------------
# A load with ``diagnose`` reports failures as a compiler would: a structured
# diagnostic with the file, line and character column at fault, the source line,
# notes saying how the entry led there, and a hint for common mistakes. Every
# field and the number of diagnostics and notes are capped, so the error frame
# stays far below the frame limit whatever user code raised.

_MISSING = object()
_MAX_DIAGNOSTICS = 20
_MAX_NOTES = 5
_MESSAGE_CHARS = 4096
_SOURCE_LINE_CHARS = 1024
_NOTE_CHARS = 512
_HINT_CHARS = 512
_PATH_CHARS = 4096
_FILE_TOO_LONG = " (file name too long to report)"
_RELATIVE_IMPORT = re.compile(r"\s*from\s+\.+([\w.]*)\s+import\b")


def _capped(text: str, limit: int) -> str:
    """*text* made frame-safe and cut at *limit* characters, saying how many were dropped."""
    text = _clean(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "... [" + str(len(text) - limit) + " chars dropped]"


class _Places:
    """Tells the workflow author's own files from the SDK, this host, installed packages and made-up names."""

    def __init__(self, entry: str, sdk_dir: str) -> None:
        import sysconfig

        self.entry = entry
        paths = sysconfig.get_paths()
        roots = [sdk_dir] + [paths[key] for key in ("stdlib", "platstdlib", "purelib", "platlib") if key in paths]
        # Real paths on both sides: an interpreter may name its stdlib through a symlink its frames don't use.
        self.roots = tuple(os.path.join(os.path.normcase(os.path.realpath(root)), "") for root in roots)
        self.host = os.path.normcase(os.path.realpath(__file__))

    def is_file(self, name: Any) -> bool:
        """Whether *name* is the entry or an existing file; anything else, such as ``<generated>``, is a label."""
        return isinstance(name, str) and (name == self.entry or (os.path.isabs(name) and os.path.isfile(name)))

    def is_user(self, name: Any) -> bool:
        if name == self.entry:
            return True
        if not self.is_file(name):
            return False
        folded = os.path.normcase(os.path.realpath(name))
        parts = folded.split(os.sep)
        installed = "site-packages" in parts or "dist-packages" in parts or folded.startswith(self.roots)
        return folded != self.host and not installed


class _Spot:
    """A place in a file: 1-based line, 1-based character columns (``end_column`` is just past the end)."""

    def __init__(
        self,
        file: str,
        line: Optional[int],
        column: Optional[int] = None,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
        module: bool = False,
    ) -> None:
        self.file = file
        self.line = line
        self.column = column
        self.end_line = end_line if end_column is not None else None
        self.end_column = end_column
        self.module = module  # the code running here is a module's top level, so reaching it was an import


def _line_number(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _line_text(file: str, line: Optional[int]) -> str:
    return linecache.getline(file, line) if line is not None else ""


def _byte_column(text: str, offset: Any) -> Optional[int]:
    """The 1-based character column of 0-based UTF-8 byte *offset* into *text*, if it lands on a character."""
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        return None
    raw = text.encode("utf-8", "surrogatepass")
    if offset > len(raw.rstrip(b"\r\n")):
        return None
    try:
        return len(raw[:offset].decode("utf-8", "surrogatepass")) + 1
    except UnicodeDecodeError:
        return None


def _syntax_column(offset: Any) -> Optional[int]:
    """A ``SyntaxError`` offset as a character column; before Python 3.11 some count bytes, so none is given."""
    if sys.version_info < (3, 11) or not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
        return None
    return offset


def _frame_spot(frame: traceback.FrameSummary) -> _Spot:
    line = _line_number(frame.lineno)
    text = _line_text(frame.filename, line)
    column = _byte_column(text, getattr(frame, "colno", None)) if text else None  # Python 3.11+
    end_line = _line_number(getattr(frame, "end_lineno", None))
    end_column = None
    if column is not None and line is not None and end_line is not None and end_line >= line:
        end_text = text if end_line == line else _line_text(frame.filename, end_line)
        end_column = _byte_column(end_text, getattr(frame, "end_colno", None)) if end_text else None
    return _Spot(frame.filename, line, column, end_line, end_column, frame.name == "<module>")


def _syntax_spot(exc: SyntaxError) -> _Spot:
    line = _line_number(exc.lineno)
    column = _syntax_column(exc.offset) if line is not None else None
    end_line = _line_number(getattr(exc, "end_lineno", None))
    end_column = None
    if column is not None and line is not None and end_line is not None and end_line >= line:
        end_column = _syntax_column(getattr(exc, "end_offset", None))
        if end_line == line and end_column is not None and end_column <= column:
            end_column = None
    return _Spot(str(exc.filename), line, column, end_line, end_column, module=True)


def _describe(exc: BaseException) -> str:
    try:
        text = str(exc)
    except Exception:
        text = "<str() of the exception failed>"
    name = type(exc).__name__
    return name + ": " + text if text else name


def _label(name: Any) -> str:
    """A made-up file name, kept on one line for a message."""
    text = name if isinstance(name, str) else repr(name)
    return _capped("".join(char if char.isprintable() else "�" for char in text), _NOTE_CHARS)


def _reported_file(file: Optional[str]) -> Tuple[Optional[str], str]:
    """The file name to report, raw, and the message suffix that says when it was too long to."""
    if file is not None and len(file) > _PATH_CHARS:
        return None, _FILE_TOO_LONG
    return file, ""


def _record(
    code: str,
    message: str,
    spot: Optional[_Spot],
    *,
    node: Any = None,
    notes: Optional[List[Dict[str, Any]]] = None,
    hint: Optional[str] = None,
) -> Dict[str, Any]:
    file, suffix = _reported_file(spot.file if spot is not None else None)
    placed = spot if file is not None else None
    source_line = None
    if placed is not None:
        text = _line_text(placed.file, placed.line)
        text = text.removesuffix("\n")
        if text and len(text) <= _SOURCE_LINE_CHARS:  # a longer line is not quoted: a cut excerpt misplaces the caret
            source_line = _clean(text)
    return {
        "code": code,
        "message": _capped(message + suffix, _MESSAGE_CHARS),
        "file": file,
        "line": placed.line if placed is not None else None,
        "column": placed.column if placed is not None else None,
        "end_line": placed.end_line if placed is not None else None,
        "end_column": placed.end_column if placed is not None else None,
        "node": node if isinstance(node, str) and len(node) <= _MESSAGE_CHARS else None,
        "source_line": source_line,
        "notes": notes or [],
        "hint": _capped(hint, _HINT_CHARS) if hint is not None else None,
    }


def _note(message: str, spot: _Spot) -> Dict[str, Any]:
    file, suffix = _reported_file(spot.file)
    return {
        "message": _capped(message + suffix, _NOTE_CHARS),
        "file": file,
        "line": spot.line if file is not None else None,
    }


def _notes(chain: List[_Spot], lead: Optional[str]) -> List[Dict[str, Any]]:
    """How the entry led to ``chain[0]``, one note per outer place; a long chain keeps its innermost and outermost."""
    notes = []
    for index in range(1, len(chain)):
        if index == 1 and lead is not None:
            message = lead
        else:
            message = "imported from" if chain[index - 1].module else "called from"
        notes.append(_note(message, chain[index]))
    if len(notes) <= _MAX_NOTES:
        return notes
    gap = {"message": str(len(notes) - _MAX_NOTES + 1) + " more frames not shown", "file": None, "line": None}
    return [*notes[: _MAX_NOTES - 2], gap, *notes[-1:]]


def _failure_diagnostic(exc: BaseException, places: _Places, validation_error: Any) -> Dict[str, Any]:
    """The diagnostic of a load that raised *exc*: placed at the innermost line of the author's own code."""
    try:
        return _placed_failure(exc, places, validation_error)
    except Exception:  # a diagnostic that can't be placed still reports the failure
        return _record("load_error", _describe(exc), None)


def _user_traceback(exc: BaseException, places: _Places) -> str:
    """*exc*'s traceback, chained exceptions included, with only the frames of the author's own code."""
    try:
        shown = traceback.TracebackException.from_exception(exc, limit=sys.maxsize)
        pending = [shown]
        while pending:
            current = pending.pop()
            current.stack = traceback.StackSummary.from_list(
                [frame for frame in current.stack if places.is_user(frame.filename)]
            )
            pending.extend(chained for chained in (current.__cause__, current.__context__) if chained is not None)
            pending.extend(getattr(current, "exceptions", None) or ())  # an exception group's members, Python 3.11+
        return "".join(shown.format())
    except Exception:  # the whole traceback still beats none
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _placed_failure(exc: BaseException, places: _Places, validation_error: Any) -> Dict[str, Any]:
    # extract_tb, not walk_tb: only it carries the column ranges of Python 3.11+; an explicit limit ignores
    # whatever sys.tracebacklimit the workflow set.
    frames = traceback.extract_tb(exc.__traceback__, limit=sys.maxsize)
    chain: List[_Spot] = []
    for frame in reversed(frames):
        if places.is_user(frame.filename):
            spot = _frame_spot(frame)
            if not chain or (chain[-1].file, chain[-1].line) != (spot.file, spot.line):  # recursion repeats a line
                chain.append(spot)
    message = _describe(exc)
    node: Any = None
    lead: Optional[str] = None
    hint: Optional[str] = None
    if isinstance(exc, SyntaxError):
        code = "syntax_error"
        message = type(exc).__name__ + ": " + str(exc.msg)
        if places.is_file(exc.filename):
            chain.insert(0, _syntax_spot(exc))
        else:  # code compiled under a made-up name is placed at the line that compiled it
            line = _line_number(exc.lineno)
            message += " (in " + _label(exc.filename) + (", line " + str(line) if line is not None else "") + ")"
    elif validation_error is not None and isinstance(exc, validation_error):
        code = "sdk_validation_error"
        error: Any = exc  # the SDK's WorkflowValidationError, loaded by the user's interpreter
        node = error.location
        site = error.site
        if isinstance(site, tuple) and len(site) == 2 and places.is_file(site[0]) and _line_number(site[1]) is not None:
            chain.insert(0, _Spot(site[0], site[1]))
            lead = "build() was called at"
    else:
        code = "load_error"
        # Python's own wording for a relative import outside a package; inside a real subpackage one is fine.
        if isinstance(exc, ImportError) and chain and str(exc).startswith("attempted relative import"):
            match = _RELATIVE_IMPORT.match(_line_text(chain[0].file, chain[0].line))
            if match is not None:
                written = "from " + match.group(1) + " import ..." if match.group(1) else "import ..."
                hint = "files of a workflow import each other by name: write '" + written + "'"
    return _record(code, message, chain[0] if chain else None, node=node, notes=_notes(chain, lead), hint=hint)


def _missing_workflow_diagnostic(found: Any, entry: str) -> Dict[str, Any]:
    if found is _MISSING:
        message = "the workflow file defines no module-level `workflow`"
    else:
        message = "module-level `workflow` is a " + type(found).__name__ + ", not the Workflow that build() returns"
    return _record("missing_workflow", message, _Spot(entry, None), hint="assign workflow = wf.build() at module level")


def _precompile(source: str, paths: List[str], places: _Places) -> List[Dict[str, Any]]:
    """The syntax errors of a folder's Python files, the entry's first, found without running any of them.

    Each file compiles from its bytes as an import would (honouring its
    coding declaration). When any is broken, the entry is compiled too, so
    the report lists every syntax error at once.
    """
    import importlib.machinery

    suffixes = tuple(_fold(suffix) for suffix in importlib.machinery.SOURCE_SUFFIXES)
    found: List[Dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the import reports a file's warnings; compiling it here must not twice
        for path in paths:
            if path == places.entry or not _fold(path).endswith(suffixes):
                continue
            try:
                with open(path, "rb") as handle:
                    payload = handle.read()
            except OSError:
                continue  # a file that can't be read now fails its import, if anything imports it
            try:
                compile(payload, path, "exec", dont_inherit=True)
            except (SyntaxError, ValueError) as exc:  # ValueError: a null byte, before Python 3.12
                found.append(_compile_failure(exc, path, places))
        if found:
            try:
                _compile_entry(source, places.entry)
            except (SyntaxError, ValueError) as exc:
                found.insert(0, _compile_failure(exc, places.entry, places))
    return found


def _compile_failure(exc: Exception, path: str, places: _Places) -> Dict[str, Any]:
    if isinstance(exc, SyntaxError):
        if exc.filename is None:  # a null byte, such as a UTF-16 file has: Python names no file
            exc.filename = path
        return _failure_diagnostic(exc, places, None)
    return _record("syntax_error", _describe(exc), _Spot(path, None))


def _diagnostics_data(found: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The ``diagnostics`` of a failed load, at most ``_MAX_DIAGNOSTICS``, and whether any were dropped."""
    return {"diagnostics": found[:_MAX_DIAGNOSTICS], "diagnostics_truncated": len(found) > _MAX_DIAGNOSTICS}


def _declaration_sites(definition: Any) -> Dict[str, Any]:
    """Where each node was declared: ``{"files": [file, ...], "nodes": {node_id: [file_index, line]}}``."""
    files: List[Optional[str]] = []
    index: Dict[str, int] = {}
    nodes: Dict[str, List[int]] = {}
    for node_id in definition.node_order:
        site = definition.nodes[node_id].site
        if site is None:
            continue
        file, line = site
        if file not in index:
            index[file] = len(files)
            files.append(_reported_file(file)[0])
        nodes[node_id] = [index[file], line]
    return {"files": files, "nodes": nodes}


def _with_sites(request_id: int, result: Dict[str, Any], sites: Dict[str, Any], limit: int) -> Dict[str, Any]:
    """*result* with the declaration sites, unless they would push the frame past *limit* bytes."""
    with_sites = dict(result, sites=sites)
    if len(_dumps({"id": request_id, "result": with_sites})) <= limit:
        return with_sites
    return dict(result, sites_truncated=True)


def _take_stdio() -> Tuple[int, int, int, _NativeTail]:
    """Move fd 0/1/2 off the protocol channel; return the private protocol in/out fds, a diagnostics fd and the tail."""
    input_fd = os.dup(0)
    os.set_inheritable(input_fd, False)
    protocol_fd = os.dup(1)
    os.set_inheritable(protocol_fd, False)
    diag_fd = os.dup(2)
    os.set_inheritable(diag_fd, False)
    if sys.platform == "win32":
        # CRT fds default to text mode: newline translation and Ctrl-Z EOF would corrupt frames.
        import msvcrt

        msvcrt.setmode(input_fd, os.O_BINARY)
        msvcrt.setmode(protocol_fd, os.O_BINARY)
    # fd 0 becomes DEVNULL: user code that reads stdin, and grandchildren that inherit it, get EOF, never
    # the protocol's own request bytes off the shared channel.
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    os.close(devnull)
    native = _NativeTail(read_fd, int(LIMITS["native_output_tail_bytes"]))
    native.start()
    return input_fd, protocol_fd, diag_fd, native


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        os.write(2, b"usage: worker_host.py <sdk_dir>\n")
        return 2
    input_fd, protocol_fd, diag_fd, native = _take_stdio()
    host = Host(argv[1], input_fd, protocol_fd, diag_fd, native)
    sys.stdout = _CaptureStream(host._route_output, 1)
    sys.stderr = _CaptureStream(host._route_output, 2)
    ok, error = host.import_sdk()
    host.send(host.hello(ok, error))
    if not ok:
        host.outbox.put(None)
        host.writer.start()
        host.writer.join(timeout=LIMITS["shutdown_grace"])
        return 3
    host.serve_forever()
    return 0


def _entry_module(filename: str) -> types.ModuleType:
    """The module the entry runs in, registered under ``MODULE_NAME`` so its functions pickle by reference."""
    module = types.ModuleType(MODULE_NAME)
    module.__file__ = filename
    sys.modules[MODULE_NAME] = module
    return module


def _exec_entry(source: str, filename: str, module: types.ModuleType) -> None:
    """Run the entry's text as the program: the host and its multiprocessing children share this one rule."""
    exec(_compile_entry(source, filename), module.__dict__)  # noqa: S102 - running the user's workflow file is this process's purpose


def _compile_entry(source: str, filename: str) -> types.CodeType:
    # A BOM belongs to the file bytes (and the entry digest) but not to the program.
    program = source.removeprefix("\ufeff")
    # Tracebacks quote the text that ran, not whatever the file holds by then; no mtime keeps checkcache off it.
    # Lines split as Python reads a file: str.splitlines() would also break at a U+2028 inside a string.
    lines = io.StringIO(program, newline=None).readlines()
    linecache.cache[filename] = (len(program), None, lines, filename)
    try:
        # dont_inherit: the file gets its own future flags, not this module's (PEP 563 annotations).
        return compile(program, filename, "exec", dont_inherit=True)
    except SyntaxError as exc:
        if exc.filename is None:  # a null byte: Python names no file
            exc.filename = filename
        raise


def _fold(name: str) -> str:
    """One spelling per file name on a case-insensitive or normalizing file system."""
    return unicodedata.normalize("NFC", name).casefold()


def _bytecode_cache_problem(bytecode_cache: str, package_dir: Optional[str]) -> Optional[str]:
    """Why loading must not go ahead with this cache, after clearing what *package_dir* left in it.

    The main process starts this interpreter with ``PYTHONPYCACHEPREFIX`` so
    no bytecode cached next to a workflow's source (which nobody confirmed)
    can run instead of the source. That cache is still keyed by whole-second
    mtimes and sizes, so a folder's own entries are cleared on every load and
    its modules compile from the source that was confirmed.
    """
    prefix = sys.pycache_prefix
    if prefix is None or os.path.normcase(os.path.abspath(prefix)) != os.path.normcase(os.path.abspath(bytecode_cache)):
        return (
            "the workflow interpreter ignores PYTHONPYCACHEPREFIX (is it started with -E or -I?), so bytecode "
            "cached next to the workflow source could run instead of the source."
        )
    if package_dir is None:
        return None
    try:
        _clear_cached_bytecode(package_dir)
    except OSError as exc:
        return "could not clear the cached bytecode of the workflow folder: " + _clean(str(exc))
    return None


def _clear_cached_bytecode(package_dir: str) -> None:
    """Delete every cached variant (any optimization level, any interpreter tag) of the folder's source files.

    Hidden entries are skipped, as the confirmation skips them; a
    ``__pycache__`` folder is searched like any other, since it can hold
    sources. A name matches by its stem whatever its case or normalization,
    and deleting a cache too many only costs one compilation.
    """
    import importlib.machinery
    import importlib.util

    suffixes = tuple(_fold(suffix) for suffix in importlib.machinery.SOURCE_SUFFIXES)
    stems: Dict[str, Set[str]] = {}
    for current, folders, files in os.walk(package_dir):
        folders[:] = [name for name in folders if not name.startswith(".")]
        for name in files:
            folded = _fold(name)
            suffix = next((suffix for suffix in suffixes if folded.endswith(suffix)), None)
            if name.startswith(".") or suffix is None:
                continue
            try:
                cached = importlib.util.cache_from_source(os.path.join(current, name))
            except NotImplementedError:  # no cache tag: this interpreter never caches bytecode
                return
            stems.setdefault(os.path.dirname(cached), set()).add(folded[: len(folded) - len(suffix)] + ".")
    for mirror, prefixes in stems.items():
        try:
            with os.scandir(mirror) as listing:
                entries = list(listing)
        except (FileNotFoundError, NotADirectoryError):
            continue
        for entry in entries:
            folded = _fold(entry.name)
            if folded.endswith(".pyc") and folded.startswith(tuple(prefixes)) and entry.is_file(follow_symlinks=False):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(entry.path)


def _import_entry_in_child() -> None:
    """Re-import the workflow module in a multiprocessing child of the host.

    A spawned or forkserver child re-imports the parent's main module, which here is this file, not the
    user's; functions pickled by reference from the workflow file need its module importable by name. The
    entry runs from its UTF-8 text exactly as the host ran it, whatever coding declaration it carries.
    """
    entry = os.environ.get(ENTRY_ENV)
    if not entry:
        return
    with open(entry, "rb") as handle:
        source = handle.read().decode("utf-8")
    _exec_entry(source, entry, _entry_module(entry))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
elif __name__ == "__mp_main__":
    _import_entry_in_child()
