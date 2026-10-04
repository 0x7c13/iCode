# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chrys-owned per-invocation log lines and the per-tool span emitted by the loop."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import pytest

from chrys.foundation.observability.gate import TELEMETRY_GATE
from chrys.kernel import SKIP_PARSING, FunctionTool, tool
from chrys.kernel.middleware import (
    FunctionInvocationContext,
    FunctionMiddleware,
    MiddlewareTermination,
)
from chrys.kernel.types import (
    Content,
)
from tests.kernel._fakes import (
    _call_response,
    _make_tool,
    _result_contents,
    _stack,
    _TerminateFunction,
    _text_response,
    _user,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


# ---------------------------------------------------------------------------
# Per-invocation logging — FunctionInvocationLayer's tool call log lines
# ---------------------------------------------------------------------------


_LOOP_LOGGER = "chrys.kernel.loop"


def _loop_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _LOOP_LOGGER]


class TestInvocationLogging:
    @pytest.mark.asyncio
    async def test_gate_off_logs_summaries_under_chrys_logger(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", False)
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "hello"})), _text_response()])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            await layer.get_response([_user()], options={"tools": [_make_tool()]})

        records = _loop_records(caplog)
        by_level = {(r.levelno, r.getMessage()) for r in records}
        assert (logging.INFO, "Function name: echo") in by_level
        assert (logging.DEBUG, "Function arguments: 1 key(s) (text)") in by_level
        assert any(
            lvl == logging.INFO and msg.startswith("Function echo succeeded in") and msg.endswith("s.")
            for lvl, msg in by_level
        )
        assert (logging.DEBUG, "Function result: 1 item(s) (text)") in by_level
        # Raw argument / result text must not leak with the gate off.
        assert all("hello" not in r.getMessage() for r in records)

    @pytest.mark.asyncio
    async def test_gate_off_does_not_disclose_unrecognized_argument_keys(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Schemas rarely pin ``additionalProperties: false``, so model-supplied
        keys survive validation — a key name carrying secret text must not
        reach the gate-off log line."""
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", False)

        async def lookup(**kwargs: Any) -> str:
            return "ok"

        schema_tool = FunctionTool(
            name="lookup",
            description="schema-supplied tool",
            func=lookup,
            input_model={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        layer, _wire = _stack(
            [_call_response(("c1", "lookup", {"query": "x", "hunter2-secret": "y"})), _text_response()]
        )
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            await layer.get_response([_user()], options={"tools": [schema_tool]})

        messages = [r.getMessage() for r in _loop_records(caplog)]
        assert "Function arguments: 2 key(s) (query, +1 unrecognized)" in messages
        assert all("hunter2" not in message for message in messages)

    @pytest.mark.asyncio
    async def test_gate_off_handles_basemodel_argument_rewrite(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The middleware contract admits ``BaseModel`` rewrites of
        ``context.arguments`` — the describe helper must not crash on one
        (the invoke site has no try/except, so a describe error would be
        converted into a tool error result)."""
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", False)
        echo_tool = _make_tool()
        input_model = echo_tool.input_model
        assert input_model is not None

        class _RewriteToModel(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                context.arguments = input_model(text="hello")
                await call_next()

        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "hi"})), _text_response()])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            response = await layer.get_response(
                [_user()], options={"tools": [echo_tool]}, middleware=[_RewriteToModel()]
            )

        messages = [r.getMessage() for r in _loop_records(caplog)]
        assert "Function arguments: 1 key(s) (text)" in messages
        assert all("hello" not in message for message in messages)
        assert str(_result_contents(response)[0].result) == "echo:hello"

    @pytest.mark.asyncio
    async def test_gate_on_logs_raw_arguments_and_result(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", True)
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "hello"})), _text_response()])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            await layer.get_response([_user()], options={"tools": [_make_tool()]})

        messages = [r.getMessage() for r in _loop_records(caplog)]
        assert "Function arguments: {'text': 'hello'}" in messages
        assert "Function result: echo:hello" in messages

    @pytest.mark.parametrize(("sensitive", "expected"), [(False, "list"), (True, "[1, 2, 3]")])
    @pytest.mark.asyncio
    async def test_gate_off_skip_parsing_raw_list_result_does_not_crash_describe(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        *,
        sensitive: bool,
        expected: str,
    ) -> None:
        """A skip-parsing tool returns the raw value — possibly a non-Content
        ``list`` — so the result describe helper must render it scalar-style
        (framework ``:697`` / ``:756``) instead of iterating ``.type``.  The
        invoke site has no try/except, so a describe crash would convert a
        successful call into a tool error result."""
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", sensitive)

        @tool(name="rawlist", result_parser=SKIP_PARSING)
        async def rawlist(n: int) -> list[int]:
            return [1, 2, 3]

        layer, _wire = _stack([_call_response(("c1", "rawlist", {"n": 1})), _text_response()])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            response = await layer.get_response([_user()], options={"tools": [rawlist]})

        results = _result_contents(response)
        assert results and results[0].exception is None
        assert str(results[0].result) == "[1, 2, 3]"
        assert f"Function result: {expected}" in [r.getMessage() for r in _loop_records(caplog)]

    @pytest.mark.asyncio
    async def test_failure_logs_single_warning_no_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Failure reporting is owned by the loop's outer except (one WARNING).
        FunctionTool.invoke is logging-free and the loop-owned per-tool span
        already records the exception, so an extra ERROR at the invoke
        site would only duplicate that single WARNING."""

        @tool(name="boom")
        async def boom(text: str) -> str:
            raise RuntimeError("kaput")

        layer, _wire = _stack([_call_response(("c1", "boom", {"text": "x"})), _text_response()])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            response = await layer.get_response([_user()], options={"tools": [boom]})

        records = _loop_records(caplog)
        assert [r for r in records if r.levelno >= logging.ERROR] == []
        warnings = [r for r in records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "Function 'boom' raised an exception" in message
        assert "kaput" in message
        assert not any("succeeded" in r.getMessage() for r in records)
        # The loop still converts the failure into an error result for the model.
        assert str(_result_contents(response)[0].result) == "Error: Function failed."

    @pytest.mark.asyncio
    async def test_termination_through_invoke_logs_nothing_above_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """A termination raised inside the tool (e.g. an interrupt propagating
        out of a nested sub-agent run) is control flow, not a failure — no
        WARNING/ERROR record may be emitted for it."""
        canned = Content.from_function_result(call_id="c1", result="interrupted")

        @tool(name="interruptible")
        async def interruptible(text: str) -> str:
            raise MiddlewareTermination("stop", result=canned)

        layer, _wire = _stack([_call_response(("c1", "interruptible", {"text": "x"}))])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            await layer.get_response([_user()], options={"tools": [interruptible]})

        records = _loop_records(caplog)
        assert all(r.levelno <= logging.INFO for r in records)
        assert any(r.getMessage() == "Function name: interruptible" for r in records)
        assert not any("succeeded" in r.getMessage() for r in records)

    @pytest.mark.asyncio
    async def test_blocked_invocation_emits_no_log_lines(self, caplog: pytest.LogCaptureFixture) -> None:
        """Middleware termination before the final handler must not log an invocation."""
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "a"}))])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            await layer.get_response(
                [_user()],
                options={"tools": [_make_tool()]},
                middleware=[_TerminateFunction(result="interrupted")],
            )

        assert _loop_records(caplog) == []


# ---------------------------------------------------------------------------
# Per-tool spans — _invoke_with_function_span owns invocation telemetry
# ---------------------------------------------------------------------------


class _SpanProbe:
    """Recording stand-ins for the telemetry seams the span path touches."""

    class _Span:
        def __init__(self, name: str, attributes: dict[Any, Any]) -> None:
            self.name = name
            self.attributes = dict(attributes)
            self.set_attrs: dict[Any, Any] = {}
            self.exceptions: list[BaseException] = []
            self.status: Any = None

        def set_attribute(self, key: Any, value: Any) -> None:
            self.set_attrs[key] = value

        def record_exception(self, exception: BaseException, timestamp: int | None = None) -> None:
            self.exceptions.append(exception)

        def set_status(self, status: Any, description: str | None = None) -> None:
            self.status = (status, description)

    class _Histogram:
        def __init__(self) -> None:
            self.records: list[tuple[float, dict[Any, Any]]] = []

        def record(self, value: float, attributes: Any = None) -> None:
            self.records.append((value, dict(attributes or {})))

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import contextlib

        from chrys.kernel import _tool_execution as tool_execution_module

        self.spans: list[_SpanProbe._Span] = []
        self.histogram = self._Histogram()
        probe = self

        @contextlib.contextmanager
        def fake_span(attributes: dict[Any, Any]) -> Any:
            span = probe._Span(
                f"{attributes[tool_execution_module.OtelAttr.OPERATION]} {attributes[tool_execution_module.OtelAttr.TOOL_NAME]}",
                attributes,
            )
            probe.spans.append(span)
            yield span

        monkeypatch.setattr(tool_execution_module, "get_function_span", lambda *, attributes: fake_span(attributes))
        monkeypatch.setattr(tool_execution_module, "_FUNCTION_DURATION_HISTOGRAM", self.histogram)


@pytest.fixture
def span_probe(monkeypatch: pytest.MonkeyPatch) -> _SpanProbe:
    return _SpanProbe(monkeypatch)


class TestPerToolSpan:
    @pytest.mark.asyncio
    async def test_gate_off_emits_no_span(self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(TELEMETRY_GATE, "enabled", False)
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "hello"})), _text_response()])
        await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert span_probe.spans == []
        assert span_probe.histogram.records == []

    @pytest.mark.asyncio
    async def test_gate_on_emits_framework_shaped_span(
        self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chrys.kernel.instrumentation import OtelAttr

        monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", False)
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "hello"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert response.messages[-1].contents[0].text == "done"

        assert len(span_probe.spans) == 1, "exactly one span per invocation — no double emission"
        span = span_probe.spans[0]
        assert span.name == "execute_tool echo"
        assert span.attributes[OtelAttr.OPERATION] == OtelAttr.TOOL_EXECUTION_OPERATION
        assert span.attributes[OtelAttr.TOOL_NAME] == "echo"
        assert span.attributes[OtelAttr.TOOL_CALL_ID] == "c1"
        assert span.attributes[OtelAttr.TOOL_TYPE] == "function"
        # Gate-off-sensitive: raw arguments/results must not reach the span.
        assert OtelAttr.TOOL_ARGUMENTS not in span.attributes
        assert OtelAttr.TOOL_RESULT not in span.set_attrs
        assert OtelAttr.MEASUREMENT_FUNCTION_INVOCATION_DURATION in span.set_attrs

        assert len(span_probe.histogram.records) == 1
        duration, attrs = span_probe.histogram.records[0]
        assert duration >= 0
        assert attrs[OtelAttr.MEASUREMENT_FUNCTION_TAG_NAME] == "echo"

    @pytest.mark.asyncio
    async def test_sensitive_on_captures_arguments_and_result(
        self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chrys.kernel.instrumentation import OtelAttr

        monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", True)
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "hello"})), _text_response()])
        await layer.get_response([_user()], options={"tools": [_make_tool()]})

        span = span_probe.spans[0]
        assert json.loads(span.attributes[OtelAttr.TOOL_ARGUMENTS]) == {"text": "hello"}
        assert span.set_attrs[OtelAttr.TOOL_RESULT] == "echo:hello"

    @pytest.mark.asyncio
    async def test_sensitive_basemodel_argument_rewrite_is_captured(
        self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Middleware may rewrite ``context.arguments`` to a ``BaseModel`` (the
        same shape ``invoke`` and the logging path accept).  The sensitive span
        capture must normalize it via ``model_dump`` instead of recording
        ``"None"`` for the whole argument set."""
        from chrys.kernel.instrumentation import OtelAttr

        monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", True)

        echo_tool = _make_tool()
        input_model = echo_tool.input_model
        assert input_model is not None

        class _RewriteToModel(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                context.arguments = input_model(text="rewritten")
                await call_next()

        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "orig"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [echo_tool]}, middleware=[_RewriteToModel()])

        assert str(_result_contents(response)[0].result) == "echo:rewritten"
        span = span_probe.spans[0]
        assert json.loads(span.attributes[OtelAttr.TOOL_ARGUMENTS]) == {"text": "rewritten"}
        assert span.set_attrs[OtelAttr.TOOL_RESULT] == "echo:rewritten"

    @pytest.mark.asyncio
    async def test_failure_records_error_type_and_exception(
        self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chrys.kernel.instrumentation import OtelAttr

        monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", False)

        @tool(name="boom")
        async def boom(text: str) -> str:
            raise RuntimeError("tool broke")

        layer, _wire = _stack([_call_response(("c1", "boom", {"text": "x"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [boom]})
        # The loop still converts the failure into an error function_result.
        results = _result_contents(response)
        assert results and "Error" in str(results[0].result)

        span = span_probe.spans[0]
        assert len(span.exceptions) == 1
        assert isinstance(span.exceptions[0], RuntimeError)
        assert span.set_attrs[OtelAttr.ERROR_TYPE] == "RuntimeError"
        (_duration, attrs) = span_probe.histogram.records[0]
        assert attrs[OtelAttr.ERROR_TYPE] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_skip_parsing_tool_sensitive_capture_uses_str(
        self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skip-parsing tools return the raw value, not ``list[Content]`` —
        ``_invoke_with_function_span`` must capture ``str(result)`` on that
        branch because the raw value has no parsed contents to text-join."""
        from chrys.kernel.instrumentation import OtelAttr

        monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", True)

        @tool(name="raw", result_parser=SKIP_PARSING)
        async def raw(text: str) -> str:
            return f"raw:{text}"

        layer, _wire = _stack([_call_response(("c1", "raw", {"text": "x"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [raw]})

        # The call must succeed — telemetry-on must not turn a successful
        # invocation into an error function_result.
        results = _result_contents(response)
        assert results and results[0].exception is None
        assert str(results[0].result) == "raw:x"

        span = span_probe.spans[0]
        assert span.exceptions == []
        assert span.set_attrs[OtelAttr.TOOL_RESULT] == "raw:x"

    @pytest.mark.asyncio
    async def test_compatible_skip_parsing_sentinel_set_late_sensitive_capture_uses_str(
        self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chrys.kernel.instrumentation import OtelAttr

        class _CompatibleSkipParsing:
            def __repr__(self) -> str:
                return "SKIP_PARSING"

        monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", True)

        @tool(name="raw")
        async def raw(text: str) -> str:
            return f"raw:{text}"

        raw.result_parser = _CompatibleSkipParsing()

        layer, _wire = _stack([_call_response(("c1", "raw", {"text": "x"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [raw]})

        results = _result_contents(response)
        assert results and results[0].exception is None
        assert str(results[0].result) == "raw:x"

        span = span_probe.spans[0]
        assert span.exceptions == []
        assert span.set_attrs[OtelAttr.TOOL_RESULT] == "raw:x"
