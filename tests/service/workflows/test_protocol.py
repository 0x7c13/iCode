# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Wire contract: golden frames, envelope validation, and the host's mirrored constants."""

from __future__ import annotations

import importlib.util
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from chrys.service.workflows import protocol, sdk
from chrys.service.workflows.protocol import (
    LIMITS,
    PROTOCOL_VERSION,
    ErrorCode,
    Method,
    ProtocolError,
    decode_frame,
    encode_frame,
    ref_from_wire,
    ref_key,
    ref_to_wire,
)
from chrys.service.workflows.scheduler import AttemptRef
from chrys.service.workflows.sdk import _builder as sdk_builder

HOST_PATH = Path(protocol.__file__).with_name("worker_host.py")
REF = AttemptRef(run_id="r", node_id="n", activation_id="n@iter#1", attempt=2)
REF_WIRE = {"run_id": "r", "node_id": "n", "activation_id": "n@iter#1", "attempt": 2}


@pytest.fixture(scope="module")
def host() -> ModuleType:
    spec = importlib.util.spec_from_file_location("worker_host_under_test", HOST_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- golden frames


@pytest.mark.parametrize(
    ("frame", "line"),
    [
        (
            {"id": 1, "method": "run_python", "params": {"ref": REF_WIRE, "value": {"text": "é", "data": None}}},
            (
                b'{"id":1,"method":"run_python","params":{"ref":{"activation_id":"n@iter#1","attempt":2,'
                b'"node_id":"n","run_id":"r"},"value":{"data":null,"text":"\xc3\xa9"}}}\n'
            ),
        ),
        (
            {
                "id": 3,
                "method": "eval_loop_until",
                "params": {"ref": REF_WIRE, "iteration": 1, "value": {"text": "", "data": {"k": [1, 2.5, True]}}},
            },
            (
                b'{"id":3,"method":"eval_loop_until","params":{"iteration":1,"ref":{"activation_id":"n@iter#1",'
                b'"attempt":2,"node_id":"n","run_id":"r"},"value":{"data":{"k":[1,2.5,true]},"text":""}}}\n'
            ),
        ),
        (
            {"method": "emit", "params": {"ref": REF_WIRE, "ordinal": 7, "text": "line"}},
            (
                b'{"method":"emit","params":{"ordinal":7,"ref":{"activation_id":"n@iter#1","attempt":2,"node_id":"n",'
                b'"run_id":"r"},"text":"line"}}\n'
            ),
        ),
        (
            {
                "id": 2,
                "method": "ask",
                "params": {
                    "ref": REF_WIRE,
                    "questions": [
                        {
                            "question": "?",
                            "header": "",
                            "options": [{"label": "yes", "description": ""}],
                            "multi_select": False,
                        }
                    ],
                },
            },
            (
                b'{"id":2,"method":"ask","params":{"questions":[{"header":"","multi_select":false,"options":'
                b'[{"description":"","label":"yes"}],"question":"?"}],"ref":{"activation_id":"n@iter#1",'
                b'"attempt":2,"node_id":"n","run_id":"r"}}}\n'
            ),
        ),
        (
            {"id": 2, "result": {"answers": [{"selected": ["yes"], "text": ""}]}},
            b'{"id":2,"result":{"answers":[{"selected":["yes"],"text":""}]}}\n',
        ),
        (
            {"id": 5, "error": {"code": "user_exception", "message": "boom", "data": {"last_emit_ordinal": 0}}},
            b'{"error":{"code":"user_exception","data":{"last_emit_ordinal":0},"message":"boom"},"id":5}\n',
        ),
    ],
    ids=["run_python", "eval_loop_until", "emit", "ask", "ask-reply", "error"],
)
def test_golden_frames_encode_and_decode(frame: dict, line: bytes) -> None:
    assert encode_frame(frame) == line
    assert decode_frame(line.rstrip(b"\n")) == frame


def test_frames_carry_lone_surrogates_byte_for_byte() -> None:
    """A surrogateescaped path (POSIX bytes that are not UTF-8) is identity: the frame must not alter or reject it."""
    frame = {"id": 1, "method": "load", "params": {"source": "", "filename": "raw-\udcff.py", "workspace": "工作"}}
    line = encode_frame(frame)
    assert b"raw-\xed\xb3\xbf.py" in line and "工作".encode() in line
    assert decode_frame(line.rstrip(b"\n")) == frame


def test_encode_rejects_nan_and_oversized_frames() -> None:
    with pytest.raises(ValueError, match="not JSON compliant"):
        encode_frame({"method": "emit", "params": {"x": float("nan")}})
    huge = {"method": "emit", "params": {"text": "x" * LIMITS.max_frame_bytes}}
    with pytest.raises(ProtocolError, match="max_frame_bytes"):
        encode_frame(huge)
    with pytest.raises(ProtocolError, match="max_frame_bytes"):
        decode_frame(b"x" * (LIMITS.max_frame_bytes + 1))


@pytest.mark.parametrize(
    "line",
    [
        b"not json",
        b"\xff\xfe",
        b"[1, 2]",
        b'{"id": 0, "method": "load", "params": {}}',
        b'{"id": true, "method": "load", "params": {}}',
        b'{"id": 1, "method": "", "params": {}}',
        b'{"id": 1, "method": "load"}',
        b'{"id": 1, "method": "load", "params": []}',
        b'{"id": 1}',
        b'{"id": 1, "result": {}, "error": {"code": "x", "message": "y"}}',
        b'{"id": 1, "result": 3}',
        b'{"id": 1, "error": {"code": 1, "message": "y"}}',
        b'{"id": 1, "error": {"code": "x", "message": "y", "data": 1}}',
        b'{"method": "emit"}',
        b'{"params": {}}',
    ],
    ids=[
        "not-json",
        "not-utf8",
        "not-object",
        "id-zero",
        "id-bool",
        "empty-method",
        "no-params",
        "params-not-object",
        "id-only",
        "result-and-error",
        "result-not-object",
        "error-code-not-str",
        "error-data-not-object",
        "notification-no-params",
        "no-method",
    ],
)
def test_decode_rejects_malformed_envelopes(line: bytes) -> None:
    with pytest.raises(ProtocolError):
        decode_frame(line)


def test_ref_wire_roundtrip_and_key() -> None:
    assert ref_to_wire(REF) == REF_WIRE
    assert ref_from_wire(REF_WIRE) == REF
    assert ref_key(REF) == "r|n|n@iter#1|2"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {**REF_WIRE, "attempt": 0},
        {**REF_WIRE, "attempt": True},
        {**REF_WIRE, "node_id": ""},
        {**REF_WIRE, "run_id": 1},
    ],
)
def test_ref_from_wire_rejects_bad_refs(payload: object) -> None:
    with pytest.raises(ProtocolError):
        ref_from_wire(payload)


# --------------------------------------------------------------------------- host parity


def test_host_mirrors_the_protocol_constants(host: ModuleType) -> None:
    """The host cannot import protocol.py, so its copies are pinned here."""
    assert host.PROTOCOL_VERSION == PROTOCOL_VERSION
    host_methods = {value for name, value in vars(host).items() if name.startswith("METHOD_")}
    assert host_methods == set(Method)
    host_errors = {value for name, value in vars(host).items() if name.startswith("ERROR_")}
    assert host_errors == set(ErrorCode)
    assert asdict(LIMITS) == host.LIMITS
    assert tuple(host.SDK_EXPORTS) == tuple(sdk.__all__)
    assert host.ENTRY_ENV == sdk_builder._ENTRY_ENV  # the SDK reads it to place declarations in the entry


@pytest.mark.parametrize(
    ("answers", "fits"),
    [
        ([{"selected": ["a"], "text": ""}, {"selected": [], "text": "note"}], True),
        ([{"selected": ["a"], "text": ""}], False),
        ([{"selected": "a", "text": ""}, {"selected": [], "text": ""}], False),
        ([{"selected": [1], "text": ""}, {"selected": [], "text": ""}], False),
        ([{"selected": [], "text": None}, {"selected": [], "text": ""}], False),
        ([{"selected": []}, {"selected": [], "text": ""}], False),
        (["a", "b"], False),
    ],
    ids=["fits", "short", "selected-str", "selected-int", "text-none", "text-missing", "not-dicts"],
)
def test_host_checks_the_ask_reply_shape(host: ModuleType, answers: list[Any], fits: bool) -> None:
    """The SDK builds Answers from the reply unchecked; the host refuses a reply it could not build them from."""
    assert host._answers_fit(answers, 2) is fits


class _ReplyingOutbox(queue.Queue):
    """Answers each ask frame with a canned reply the moment the host queues it."""

    def __init__(self, worker: Any, reply: dict[str, Any]) -> None:
        super().__init__()
        self.worker = worker
        self.reply = reply

    def put(self, item: object, block: bool = True, timeout: float | None = None) -> None:
        super().put(item, block, timeout)
        assert isinstance(item, bytes)
        self.worker.pending_asks[decode_frame(item.rstrip(b"\n"))["id"]].set_result(self.reply)


@pytest.mark.parametrize(
    "reply",
    [
        {"result": {"answers": [{"selected": [], "text": "x"}]}},
        {"result": {"answers": []}},
        {"result": {"answers": [{"selected": "x", "text": ""}]}},
        {"result": None},
    ],
    ids=["fits", "no-answers", "bad-answer", "no-result"],
)
def test_the_host_fails_an_ask_whose_reply_does_not_fit_as_unavailable(host: ModuleType, reply: dict[str, Any]) -> None:
    worker = host.Host(sdk_dir="", input_fd=-1, protocol_fd=-1, diag_fd=-1, native=None)
    outbox = worker.outbox = _ReplyingOutbox(worker, reply)
    attempt = host._Attempt("body|" + ref_key(REF), REF_WIRE, ref_key(REF), 7)
    question = {"question": "q", "header": "", "options": [], "multi_select": False}
    answers: Any = None
    error: Any = None
    try:
        answers = worker.loop.run_until_complete(worker._ask(attempt, [question]))
    except host._AskError as exc:
        error = exc
    finally:
        worker.pool.shutdown(wait=False)
        worker.eval_pool.shutdown(wait=False)
        worker.loop.close()

    frame = decode_frame(outbox.get_nowait().rstrip(b"\n"))
    assert (frame["method"], frame["params"]["questions"]) == ("ask", [question])
    assert worker.pending_asks == {}
    if reply == {"result": {"answers": [{"selected": [], "text": "x"}]}}:
        assert (answers, error) == ([{"selected": [], "text": "x"}], None)
    else:
        assert answers is None and error is not None
        assert (error.code, str(error)) == (host.ERROR_ASK_UNAVAILABLE, "ask reply carried no answers.")


def test_host_serves_every_forward_method(host: ModuleType) -> None:
    assert set(host.Host._handlers) == set(Method) - {Method.HELLO, Method.EMIT, Method.ASK}


# --------------------------------------------------------------------------- completion ordering


class _WatchedLock:
    """Stands in for the emit lock; reports when a second thread comes to wait on it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def __enter__(self) -> None:
        if self._lock.locked():
            self.contended.set()
        self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


class _GatedQueue(queue.Queue):
    """Holds each emit frame between taking its ordinal and queueing it until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.in_flight = threading.Event()
        self.release = threading.Event()
        self.timed_out = False

    def put(self, item: object, block: bool = True, timeout: float | None = None) -> None:
        if isinstance(item, bytes) and b'"method":"emit"' in item:
            self.in_flight.set()
            self.timed_out = not self.release.wait(timeout=5.0)
        super().put(item, block, timeout)


@pytest.mark.parametrize("finish", ["ok", "error"])
def test_terminal_envelope_follows_every_emit_it_covers(host: ModuleType, finish: str) -> None:
    worker = host.Host(sdk_dir="", input_fd=-1, protocol_fd=-1, diag_fd=-1, native=None)
    outbox = worker.outbox = _GatedQueue()
    lock = worker.emit_lock = _WatchedLock()
    attempt = host._Attempt("body|" + ref_key(REF), REF_WIRE, ref_key(REF), 7)
    worker.attempts[attempt.key] = attempt

    def complete() -> None:
        if finish == "ok":
            worker._finish_ok(attempt, {"value": {"text": "x", "data": None}}, barrier=True)
        else:
            worker._finish(attempt, host.ERROR_USER_EXCEPTION, "boom")

    emitter = threading.Thread(target=worker._emit, args=(attempt, "one"))
    finisher = threading.Thread(target=complete)
    try:
        emitter.start()
        assert outbox.in_flight.wait(timeout=5.0)  # the emit holds the lock: ordinal taken, frame not queued
        finisher.start()
        deadline = time.monotonic() + 5.0
        while finisher.is_alive() and not lock.contended.is_set() and time.monotonic() < deadline:
            time.sleep(0.001)  # completion either waits on the emit lock or has already run past it
        assert lock.contended.is_set() or not finisher.is_alive(), "completion neither waited nor finished"
    finally:
        outbox.release.set()
        emitter.join(timeout=5.0)
        finisher.join(timeout=5.0)
        worker.pool.shutdown(wait=False)
        worker.eval_pool.shutdown(wait=False)
        worker.loop.close()
    assert not outbox.timed_out, "the emit was released by its timeout, not by the test"

    frames = [decode_frame(outbox.get_nowait().rstrip(b"\n")) for _ in range(2)]
    assert frames[0].get("method") == "emit"
    envelope = frames[1]["result"] if finish == "ok" else frames[1]["error"]["data"]
    assert envelope["last_emit_ordinal"] == 1
