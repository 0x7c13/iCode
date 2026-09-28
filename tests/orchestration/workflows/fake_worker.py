# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A scripted worker host for client tests: the node id of a ``run_python`` picks the misbehaviour.

Started exactly like the real host (``<python> fake_worker.py <sdk_dir>``);
single-threaded, one frame in, scripted frames out. The ``probe`` method
returns every frame received so far, so a test can check what the client
sent back for a reverse request.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from typing import Any

_LIMIT = 16 * 1024 * 1024


def _send(frame: dict[str, Any]) -> None:
    sys.stdout.buffer.write(json.dumps(frame, sort_keys=True).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


def _result(request_id: int, result: dict[str, Any]) -> None:
    _send({"id": request_id, "result": result})


def _python_result(request_id: int, text: str, last_emit_ordinal: int = 0) -> None:
    _result(
        request_id,
        {
            "value": {"text": text, "data": None},
            "stdout": {"text": "", "truncated": False},
            "last_emit_ordinal": last_emit_ordinal,
            "leaked_threads_by_pool": {"body": 0, "eval": 0},
        },
    )


def _emit(ref: dict[str, Any], ordinal: int) -> None:
    _send({"method": "emit", "params": {"ref": ref, "ordinal": ordinal, "text": f"emit {ordinal}"}})


class _Worker:
    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []
        self.hung: dict[str, int] = {}
        self.next_reverse_id = 2

    def serve(self) -> None:
        _send(
            {
                "method": "hello",
                "params": {
                    "protocol_version": 2,
                    "python_version": platform.python_version(),
                    "implementation": platform.python_implementation(),
                    "platform": sys.platform,
                    "sdk_origin_ok": True,
                    "sdk_origin_error": None,
                },
            }
        )
        for raw in sys.stdin.buffer:
            frame = json.loads(raw)
            self.received.append(frame)
            if "method" in frame:
                self.handle(frame)

    def handle(self, frame: dict[str, Any]) -> None:
        request_id, method, params = frame["id"], frame["method"], frame["params"]
        if method == "load":
            _result(
                request_id,
                {
                    "manifest": {},
                    "stdout": {"text": "", "truncated": False},
                },
            )
        elif method == "run_python":
            self.run_python(request_id, params)
        elif method == "eval_outgoing":
            # Capacity tests drive the same request lifecycle on the independent evaluation lane.
            self.run_python(request_id, params)
        elif method == "cancel":
            key = _key(params["ref"])
            hung = self.hung.pop(key, None)
            if hung is not None:
                _send(
                    {
                        "id": hung,
                        "error": {
                            "code": "attempt_terminated",
                            "message": "cancelled",
                            "data": {
                                "stdout": {"text": "", "truncated": False},
                                "last_emit_ordinal": 0,
                                "leaked_threads_by_pool": {"body": int(params["ref"]["node_id"] == "leak"), "eval": 0},
                            },
                        },
                    }
                )
            _result(
                request_id,
                {
                    "cancelled": hung is not None,
                    "leaked_threads_by_pool": {"body": int(params["ref"]["node_id"] == "leak"), "eval": 0},
                },
            )
        elif method == "native_output":
            _result(request_id, {"text": "", "dropped_bytes": 0})
        elif method == "probe":
            _result(request_id, {"received": self.received[:-1]})
        elif method == "shutdown":
            _result(request_id, {"ok": True})
            sys.stdout.buffer.flush()
            os._exit(0)
        else:
            _send({"id": request_id, "error": {"code": "unknown_method", "message": method, "data": {}}})

    def run_python(self, request_id: int, params: dict[str, Any]) -> None:
        ref = params["ref"]
        node = ref["node_id"]
        if node == "ok":
            _python_result(request_id, "ok")
        elif node == "leaky":
            # A body that ended with one of its offloads still on a thread: the envelope carries the verdict.
            _result(
                request_id,
                {
                    "value": {"text": "leaky", "data": None},
                    "stdout": {"text": "", "truncated": False},
                    "last_emit_ordinal": 0,
                    "leaked_threads_by_pool": {"body": 1, "eval": 0},
                },
            )
        elif node == "late_emit":
            _python_result(request_id, "late", 0)
            _emit(ref, 1)
            _emit(ref, 2)
        elif node == "late_ask":
            _python_result(request_id, "late", 0)
            self.ask(ref)
        elif node == "emits":
            for ordinal in (1, 2, 3):
                _emit(ref, ordinal)
            _python_result(request_id, "emits", 3)
        elif node == "emits_error":
            _emit(ref, 1)
            _emit(ref, 2)
            _send(
                {
                    "id": request_id,
                    "error": {
                        "code": "user_exception",
                        "message": "boom",
                        "data": {"stdout": {"text": "", "truncated": False}, "last_emit_ordinal": 2, "traceback": "tb"},
                    },
                }
            )
        elif node == "ask":
            reply = self.ask(ref)
            if "result" in reply:
                _python_result(request_id, "answer=" + reply["result"]["answers"][0]["text"])
            else:
                _send({"id": request_id, "error": {**reply["error"], "data": {"last_emit_ordinal": 0}}})
        elif node == "malformed_ask":
            # An option without its description: the client must treat the frame as a protocol error.
            question = {"question": "q", "header": "", "options": [{"label": "a"}], "multi_select": False}
            _send({"id": self.next_reverse_id, "method": "ask", "params": {"ref": ref, "questions": [question]}})
            self.next_reverse_id += 2
        elif node == "ask_then_hang":
            # Hung first, so a cancel that arrives while the ask is open errors this request.
            self.hung[_key(ref)] = request_id
            self.ask(ref)
        elif node in {"hang", "leak"}:
            self.hung[_key(ref)] = request_id
        elif node == "garbage":
            sys.stdout.buffer.write(b"this is not json\n")
            sys.stdout.buffer.flush()
        elif node == "bogus_reverse":
            _send({"id": self.next_reverse_id, "method": "bogus", "params": {}})
            self.next_reverse_id += 2
            self.received.append(json.loads(next(sys.stdin.buffer)))
            _python_result(request_id, "bogus")
        elif node == "exit":
            sys.stdout.buffer.flush()
            os._exit(0)
        else:
            _send(
                {"id": request_id, "error": {"code": "invalid_params", "message": f"unknown script {node}", "data": {}}}
            )

    def ask(self, ref: dict[str, Any]) -> dict[str, Any]:
        """Send one ask for *ref* and return the client's reply frame (also recorded)."""
        request_id = self.next_reverse_id
        self.next_reverse_id += 2
        question = {"question": "q", "header": "", "options": [], "multi_select": False}
        _send({"id": request_id, "method": "ask", "params": {"ref": ref, "questions": [question]}})
        while True:
            reply = json.loads(next(sys.stdin.buffer))
            self.received.append(reply)
            if reply.get("id") == request_id and "method" not in reply:
                return reply
            if "method" in reply:
                self.handle(reply)


def _key(ref: dict[str, Any]) -> str:
    return f"{ref['run_id']}|{ref['node_id']}|{ref['activation_id']}|{ref['attempt']}"


if __name__ == "__main__":
    _Worker().serve()
