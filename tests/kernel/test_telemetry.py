# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Acceptance tests for the chrys-owned telemetry layers.

Pins the span/metric/event semantics of
``ChatTelemetryLayer``/``AgentTelemetryLayer`` against the chrys gate, plus
the deliberate divergences: the INNER ContextVar lifecycle fix (H group) and
the getattr stream-error probe. The E group pins the vendored constants and
the private stream surface the telemetry layer depends on
(``ResponseStream._stream_error``).

OTel plumbing is faked at the module seam (``get_tracer``/``get_meter``)
instead of installing global SDK providers: the OTel API only allows one
global provider per process and xdist workers share test order, while the
fakes give per-test span/metric capture with zero global state.
"""

from __future__ import annotations

import asyncio
import gc
import importlib.metadata
import json
import logging
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

import chrys.kernel as kernel
import chrys.kernel.instrumentation as instrumentation
from chrys.foundation.observability.gate import TELEMETRY_GATE, TelemetryGate, configure_telemetry
from chrys.kernel.instrumentation import (
    FINISH_REASON_MAP,
    GEN_AI_METRIC_ATTRIBUTES,
    INNER_ACCUMULATED_USAGE,
    INNER_RESPONSE_ID_CAPTURED_FIELD,
    INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS,
    INNER_USAGE_CAPTURED_FIELD,
    OPERATION_DURATION_BUCKET_BOUNDARIES,
    ROLE_EVENT_MAP,
    TOKEN_USAGE_BUCKET_BOUNDARIES,
    AgentTelemetryLayer,
    ChatTelemetryLayer,
    MessageListTimestampFilter,
    OtelAttr,
    _apply_accumulated_usage,
    _capture_current_agent_system_instructions,
    _get_data_bytes_as_str,
    _get_instructions_from_options,
    _get_response_attributes,
    _get_span,
    _get_span_attributes,
    _mark_inner_response_telemetry_captured,
    _serialize_tool_definitions,
    _stream_error_of,
    _to_otel_part,
    capture_exception,
)
from chrys.kernel.types import (
    AgentResponse,
    AgentResponseUpdate,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
    tool,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

TELEMETRY_LOGGER = "chrys.kernel.instrumentation"

# ---------------------------------------------------------------------------
# Fake OTel plumbing (duck-typed; use_span/_activate_span only need attach,
# detach, end, and the attribute setters)
# ---------------------------------------------------------------------------


class _FakeSpanContext:
    def __init__(self, trace_id: int, span_id: int) -> None:
        self.trace_id = trace_id
        self.span_id = span_id


class _FakeSpan:
    def __init__(
        self,
        name: str,
        *,
        context: _FakeSpanContext | None = None,
        parent: _FakeSpanContext | None = None,
        attributes: Mapping[Any, Any] | None = None,
    ) -> None:
        self.name = name
        self.context = context or _FakeSpanContext(trace_id=1, span_id=1)
        self.parent = parent
        self.attributes: dict[Any, Any] = dict(attributes or {})
        self.exceptions: list[BaseException] = []
        self.status: tuple[Any, str | None] | None = None
        self.end_count = 0

    @property
    def ended(self) -> bool:
        return self.end_count > 0

    def set_attributes(self, attributes: Mapping[Any, Any]) -> None:
        self.attributes.update(attributes)

    def set_attribute(self, key: Any, value: Any) -> None:
        self.attributes[key] = value

    def record_exception(self, exception: BaseException, timestamp: int | None = None) -> None:
        self.exceptions.append(exception)

    def set_status(self, status: Any, description: str | None = None) -> None:
        self.status = (status, description)

    def update_name(self, name: str) -> None:
        self.name = name

    def is_recording(self) -> bool:
        return True

    def get_span_context(self) -> _FakeSpanContext:
        return self.context

    def end(self) -> None:
        self.end_count += 1


class _FakeNonRecordingSpan(_FakeSpan):
    def is_recording(self) -> bool:
        return False


class _FakeTracer:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []
        self._next_span_id = 1

    def start_span(self, name: str, **kwargs: Any) -> _FakeSpan:
        parent_span = next((span for span in reversed(self.spans) if not span.ended), None)
        parent = parent_span.get_span_context() if parent_span is not None else None
        span = _FakeSpan(
            name,
            context=_FakeSpanContext(trace_id=1, span_id=self._next_span_id),
            parent=parent,
            attributes=kwargs.get("attributes"),
        )
        self._next_span_id += 1
        self.spans.append(span)
        return span


class _FakeHistogram:
    def __init__(self, name: str, unit: Any, description: Any, boundaries: Any) -> None:
        self.name = name
        self.unit = unit
        self.description = description
        self.boundaries = boundaries
        self.records: list[tuple[Any, dict[Any, Any]]] = []

    def record(self, value: Any, attributes: Mapping[Any, Any] | None = None) -> None:
        self.records.append((value, dict(attributes or {})))


class _FakeMeter:
    def __init__(self) -> None:
        self.histograms: list[_FakeHistogram] = []

    def create_histogram(
        self,
        name: str,
        unit: Any = None,
        description: Any = None,
        explicit_bucket_boundaries_advisory: Any = None,
    ) -> _FakeHistogram:
        histogram = _FakeHistogram(name, unit, description, explicit_bucket_boundaries_advisory)
        self.histograms.append(histogram)
        return histogram


class _FakeOtel:
    def __init__(self) -> None:
        self.tracer = _FakeTracer()
        self.meter = _FakeMeter()

    @property
    def spans(self) -> list[_FakeSpan]:
        return self.tracer.spans


@pytest.fixture
def fake_otel(monkeypatch: pytest.MonkeyPatch) -> _FakeOtel:
    state = _FakeOtel()
    monkeypatch.setattr(instrumentation, "get_tracer", lambda *a, **k: state.tracer)
    monkeypatch.setattr(instrumentation, "get_meter", lambda *a, **k: state.meter)
    monkeypatch.setattr(
        instrumentation.trace,
        "get_current_span",
        lambda: next((span for span in reversed(state.spans) if not span.ended), _FakeNonRecordingSpan("invalid")),
    )
    return state


@pytest.fixture
def gate_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TELEMETRY_GATE, "enabled", False)
    monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", False)


@pytest.fixture
def gate_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
    monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", False)


@pytest.fixture
def gate_sensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
    monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", True)


# ---------------------------------------------------------------------------
# Chat client doubles
# ---------------------------------------------------------------------------


def _text_update(text: str, **kwargs: Any) -> ChatResponseUpdate:
    return ChatResponseUpdate(contents=[Content.from_text(text)], role="assistant", **kwargs)


def _chat_response(text: str = "done", **kwargs: Any) -> ChatResponse:
    return ChatResponse(messages=[Message(role="assistant", contents=[text])], **kwargs)


class _WireChat:
    """Innermost fake chat client recording the exact call surface.

    Non-stream turns are ``ChatResponse`` (an ``Exception`` raises on await);
    stream turns are lists of ``ChatResponseUpdate`` (an ``Exception`` item
    raises mid-stream). ``sync_error`` raises synchronously from
    ``get_response`` itself.
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "testchat"

    def __init__(
        self,
        turns: list[Any] | None = None,
        *,
        model: str | None = "wire-model",
        url: str | None = None,
        sync_error: Exception | None = None,
    ) -> None:
        self.turns = list(turns) if turns is not None else None
        self.calls: list[dict[str, Any]] = []
        self.returned: list[Any] = []
        self.sync_error = sync_error
        if model is not None:
            self.model = model
        if url is not None:

            def _service_url() -> str:
                return url

            self.service_url = _service_url

    def get_response(
        self,
        messages: Any = None,
        *,
        stream: bool = False,
        options: Any = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Any = None,
        client_kwargs: Any = None,
    ) -> Any:
        self.calls.append(
            {
                "messages": list(messages or []),
                "stream": stream,
                "options": dict(options or {}),
                "compaction_strategy": compaction_strategy,
                "tokenizer": tokenizer,
                "function_invocation_kwargs": function_invocation_kwargs,
                "client_kwargs": client_kwargs,
            }
        )
        if self.sync_error is not None:
            raise self.sync_error
        default_turn: Any = [_text_update("done")] if stream else _chat_response()
        turn = self.turns.pop(0) if self.turns is not None else default_turn
        if not stream:

            async def _resolve() -> ChatResponse:
                if isinstance(turn, Exception):
                    raise turn
                return turn

            result = _resolve()
            self.returned.append(result)
            return result

        async def _gen() -> Any:
            for update in turn:
                if isinstance(update, Exception):
                    raise update
                yield update

        result = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)
        self.returned.append(result)
        return result


class _TelChat(ChatTelemetryLayer, _WireChat):
    pass


class _InstanceProvChat(_WireChat):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.OTEL_PROVIDER_NAME = "instance-prov"


class _TelInstanceProvChat(ChatTelemetryLayer, _InstanceProvChat):
    pass


# ---------------------------------------------------------------------------
# Agent core doubles
# ---------------------------------------------------------------------------


def _agent_update(text: str = "done", **kwargs: Any) -> AgentResponseUpdate:
    return AgentResponseUpdate(contents=[Content.from_text(text)], role="assistant", **kwargs)


def _agent_response(text: str = "done", **kwargs: Any) -> AgentResponse:
    return AgentResponse(messages=[Message(role="assistant", contents=[text])], **kwargs)


class _CoreAgent:
    """Fake agent core recording the exact ``run`` kwarg surface."""

    AGENT_PROVIDER_NAME: ClassVar[str] = "stub-provider"

    def __init__(
        self,
        turns: list[Any] | None = None,
        *,
        id: str = "agent-1",
        name: str | None = "A",
        description: str | None = None,
        default_options: dict[str, Any] | None = None,
        stream_awaitable: bool = False,
    ) -> None:
        self.id = id
        self.name = name
        self.description = description
        self.default_options = dict(default_options or {})
        self.turns = list(turns) if turns is not None else None
        self.calls: list[dict[str, Any]] = []
        self.returned: list[Any] = []
        self.stream_awaitable = stream_awaitable

    def run(self, messages: Any = None, **kwargs: Any) -> Any:
        self.calls.append({"messages": messages, **kwargs})
        stream = kwargs.get("stream", False)
        default_turn: Any = [_agent_update()] if stream else _agent_response()
        turn = self.turns.pop(0) if self.turns is not None else default_turn
        if isinstance(turn, Exception):
            raise turn
        if not stream:

            async def _resolve() -> AgentResponse:
                return turn

            result = _resolve()
            self.returned.append(result)
            return result

        async def _gen() -> Any:
            for update in turn:
                if isinstance(update, Exception):
                    raise update
                yield update

        inner = ResponseStream(_gen(), finalizer=AgentResponse.from_updates)
        if self.stream_awaitable:

            async def _later() -> Any:
                return inner

            result = _later()
        else:
            result = inner
        self.returned.append(result)
        return result


class _TelAgent(AgentTelemetryLayer, _CoreAgent):
    pass


class _LateProvCore(_CoreAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.AGENT_PROVIDER_NAME = "instance-late"


class _TelLateProvAgent(AgentTelemetryLayer, _LateProvCore):
    pass


class _ChatDrivenCore:
    """Agent core that drives a real (telemetry-wrapped) chat client."""

    AGENT_PROVIDER_NAME: ClassVar[str] = "stub-provider"

    def __init__(self, chat: _TelChat, *, chat_calls: int = 2) -> None:
        self.chat = chat
        self.chat_calls = chat_calls
        self.id = "agent-1"
        self.name = "A"
        self.description = None
        self.default_options: dict[str, Any] = {}

    def run(self, messages: Any = None, *, stream: bool = False, **kwargs: Any) -> Any:
        if not stream:

            async def _go() -> AgentResponse:
                collected: list[Message] = []
                for index in range(self.chat_calls):
                    response = await self.chat.get_response([Message(role="user", contents=[f"q{index}"])])
                    collected.extend(response.messages)
                return AgentResponse(
                    messages=collected,
                    response_id="agent-rid",
                    usage_details={"input_token_count": 999, "output_token_count": 999},
                )

            return _go()

        async def _gen() -> Any:
            response = await self.chat.get_response([Message(role="user", contents=["q0"])])
            yield AgentResponseUpdate(
                contents=[Content.from_text(response.text or "x")], role="assistant", response_id="agent-rid"
            )

        return ResponseStream(_gen(), finalizer=AgentResponse.from_updates)


class _PipelineAgent(AgentTelemetryLayer, _ChatDrivenCore):
    pass


def _chat_usage_turns() -> list[ChatResponse]:
    return [
        _chat_response(
            "r1",
            response_id="chat-1",
            model="m1",
            usage_details={
                "input_token_count": 3,
                "output_token_count": 5,
                "cache_read_input_token_count": 2,
                "reasoning_output_token_count": 1,
            },
        ),
        _chat_response(
            "r2",
            response_id="chat-2",
            model="m1",
            usage_details={
                "input_token_count": 7,
                "output_token_count": 11,
                "cache_read_input_token_count": 4,
                "reasoning_output_token_count": 8,
            },
        ),
    ]


# ---------------------------------------------------------------------------
# A. Gate and re-export surface
# ---------------------------------------------------------------------------


class TestGate:
    def test_defaults_off(self) -> None:
        gate = TelemetryGate()
        assert gate.enabled is False
        assert gate.sensitive_data is False
        assert gate.sensitive_enabled is False
        assert isinstance(TELEMETRY_GATE, TelemetryGate)

    def test_configure_telemetry_flips_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(TELEMETRY_GATE, "enabled", False)
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", False)
        configure_telemetry(enabled=True, sensitive_data=True)
        assert TELEMETRY_GATE.enabled is True
        assert TELEMETRY_GATE.sensitive_enabled is True
        configure_telemetry(enabled=True)
        assert TELEMETRY_GATE.sensitive_data is False, "sensitive_data defaults back to False"
        configure_telemetry(enabled=False, sensitive_data=True)
        assert TELEMETRY_GATE.enabled is False
        assert TELEMETRY_GATE.sensitive_enabled is False, "sensitive requires enabled"

    def test_package_reexports_are_telemetry_objects(self) -> None:
        assert kernel.ChatTelemetryLayer is ChatTelemetryLayer
        assert kernel.AgentTelemetryLayer is AgentTelemetryLayer
        assert kernel.TELEMETRY_GATE is TELEMETRY_GATE
        assert kernel.configure_telemetry is configure_telemetry


# ---------------------------------------------------------------------------
# B. ChatTelemetryLayer — gate off
# ---------------------------------------------------------------------------


class TestChatLayerDisabled:
    @pytest.mark.asyncio
    async def test_non_stream_identity_passthrough(self, fake_otel: _FakeOtel, gate_off: None) -> None:
        client = _TelChat()
        result = client.get_response([Message(role="user", contents=["hi"])])
        assert result is client.returned[0], "disabled path must return the core awaitable untouched"
        response = await result
        assert response.text == "done"
        assert fake_otel.spans == []

    @pytest.mark.asyncio
    async def test_stream_identity_passthrough_no_hooks(self, fake_otel: _FakeOtel, gate_off: None) -> None:
        client = _TelChat()
        stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
        assert stream is client.returned[0]
        assert stream._cleanup_hooks == []
        assert await stream.get_final_response()
        assert fake_otel.spans == []

    @pytest.mark.asyncio
    async def test_client_kwargs_none_normalized_to_empty_dict(self, fake_otel: _FakeOtel, gate_off: None) -> None:
        client = _TelChat()
        await client.get_response([Message(role="user", contents=["hi"])], client_kwargs=None)
        assert client.calls[0]["client_kwargs"] == {}
        given = {"model": "ck-model"}
        await client.get_response([Message(role="user", contents=["hi"])], client_kwargs=given)
        assert client.calls[1]["client_kwargs"] == given
        assert client.calls[1]["client_kwargs"] is not given, "copied, not aliased"


# ---------------------------------------------------------------------------
# C. ChatTelemetryLayer — non-streaming, gate on
# ---------------------------------------------------------------------------


class TestChatLayerNonStreaming:
    @pytest.mark.asyncio
    async def test_span_name_and_base_attributes(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        client = _TelChat(url="https://api.test")
        response = await client.get_response([Message(role="user", contents=["hi"])])
        assert response.text == "done"
        (span,) = fake_otel.spans
        assert span.name == "chat wire-model"
        assert span.ended
        assert span.attributes[OtelAttr.OPERATION] == "chat"
        assert span.attributes[OtelAttr.PROVIDER_NAME] == "testchat"
        assert span.attributes[OtelAttr.REQUEST_MODEL] == "wire-model"
        assert span.attributes[OtelAttr.ADDRESS] == "https://api.test"
        assert span.attributes[OtelAttr.CHOICE_COUNT] == 1

    @pytest.mark.asyncio
    async def test_service_url_absent_falls_back_to_unknown(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        client = _TelChat()
        await client.get_response([Message(role="user", contents=["hi"])])
        assert fake_otel.spans[0].attributes[OtelAttr.ADDRESS] == "unknown"

    @pytest.mark.asyncio
    async def test_model_resolution_precedence(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        client = _TelChat()
        await client.get_response([Message(role="user", contents=["hi"])], options={"model": "opt-model"})
        assert fake_otel.spans[0].name == "chat opt-model", "options beat the client attribute"
        await client.get_response([Message(role="user", contents=["hi"])])
        assert fake_otel.spans[1].name == "chat wire-model"
        no_model = _TelChat(model=None)
        await no_model.get_response([Message(role="user", contents=["hi"])])
        assert fake_otel.spans[2].name == "chat unknown"
        # ChatTelemetryLayer passes model explicitly to _get_span_attributes;
        # a "model" key in client_kwargs therefore supplies it twice and raises.
        # Chrys never routes the model via client_kwargs — pin the rejection
        # so accepting that input requires an explicit precedence decision.
        with pytest.raises(TypeError, match="multiple values"):
            client.get_response([Message(role="user", contents=["hi"])], client_kwargs={"model": "ck-model"})

    @pytest.mark.asyncio
    async def test_client_kwargs_request_attributes(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        client = _TelChat()
        await client.get_response(
            [Message(role="user", contents=["hi"])],
            client_kwargs={
                "temperature": 0.5,
                "max_tokens": 128,
                "seed": 42,
                "top_p": 0.9,
                "top_k": 5,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
                "stop": ["x"],
                "conversation_id": "conv-9",
            },
        )
        attrs = fake_otel.spans[0].attributes
        assert attrs[OtelAttr.REQUEST_TEMPERATURE] == 0.5
        assert attrs[OtelAttr.REQUEST_MAX_TOKENS] == 128
        assert attrs[OtelAttr.SEED] == 42
        assert attrs[OtelAttr.REQUEST_TOP_P] == 0.9
        assert attrs[OtelAttr.TOP_K] == 5
        assert attrs[OtelAttr.FREQUENCY_PENALTY] == 0.1
        assert attrs[OtelAttr.PRESENCE_PENALTY] == 0.2
        assert attrs[OtelAttr.STOP_SEQUENCES] == ["x"]
        assert attrs[OtelAttr.CONVERSATION_ID] == "conv-9"

    @pytest.mark.asyncio
    async def test_provider_name_fallbacks(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        # Explicit ctor kwarg wins over the ClassVar.
        client = _TelChat(otel_provider_name="explicit")
        await client.get_response([Message(role="user", contents=["hi"])])
        assert fake_otel.spans[0].attributes[OtelAttr.PROVIDER_NAME] == "explicit"
        # Instance attribute set during the core __init__ is visible: the CTL
        # ctor reads OTEL_PROVIDER_NAME *after* super().__init__ (:1357-1362).
        instance_prov = _TelInstanceProvChat()
        await instance_prov.get_response([Message(role="user", contents=["hi"])])
        assert fake_otel.spans[1].attributes[OtelAttr.PROVIDER_NAME] == "instance-prov"

    @pytest.mark.asyncio
    async def test_exception_captured_and_raised(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        error = RuntimeError("wire down")
        client = _TelChat(turns=[error])
        with pytest.raises(RuntimeError, match="wire down"):
            await client.get_response([Message(role="user", contents=["hi"])])
        (span,) = fake_otel.spans
        assert span.ended
        assert span.attributes[OtelAttr.ERROR_TYPE] == "RuntimeError"
        assert span.exceptions == [error]
        assert span.status is not None and span.status[1] == repr(error)
        ((duration, duration_attrs),) = client.duration_histogram.records
        assert duration >= 0
        assert duration_attrs[OtelAttr.ERROR_TYPE] == "RuntimeError"
        assert client.token_usage_histogram.records == []

    @pytest.mark.asyncio
    async def test_response_attributes_and_metrics(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        client = _TelChat(
            turns=[
                _chat_response(
                    response_id="resp-1",
                    model="served-model",
                    finish_reason="stop",
                    usage_details={"input_token_count": 3, "output_token_count": 5},
                )
            ],
            url="https://api.test",
        )
        await client.get_response([Message(role="user", contents=["hi"])])
        (span,) = fake_otel.spans
        assert span.attributes[OtelAttr.RESPONSE_ID] == "resp-1"
        assert span.attributes[OtelAttr.RESPONSE_MODEL] == "served-model"
        assert span.attributes[OtelAttr.FINISH_REASONS] == json.dumps(["stop"])
        assert span.attributes[OtelAttr.INPUT_TOKENS] == 3
        assert span.attributes[OtelAttr.OUTPUT_TOKENS] == 5
        token_records = client.token_usage_histogram.records
        assert [(value, attrs[OtelAttr.T_TYPE]) for value, attrs in token_records] == [(3, "input"), (5, "output")]
        for _, attrs in token_records:
            assert set(attrs) <= {*GEN_AI_METRIC_ATTRIBUTES, OtelAttr.T_TYPE}
            assert attrs[OtelAttr.PROVIDER_NAME] == "testchat"
            assert attrs[OtelAttr.RESPONSE_MODEL] == "served-model"
        ((duration, duration_attrs),) = client.duration_histogram.records
        assert duration >= 0
        assert OtelAttr.ERROR_TYPE not in duration_attrs

    @pytest.mark.asyncio
    async def test_usage_details_map_cache_reasoning_and_record_zero_tokens(
        self, fake_otel: _FakeOtel, gate_on: None
    ) -> None:
        client = _TelChat(
            turns=[
                _chat_response(
                    usage_details={
                        "input_token_count": 0,
                        "output_token_count": 0,
                        "cache_read_input_token_count": 12,
                        "reasoning_output_token_count": 4,
                    }
                )
            ]
        )
        await client.get_response([Message(role="user", contents=["hi"])])

        attrs = fake_otel.spans[0].attributes
        assert attrs[OtelAttr.INPUT_TOKENS] == 0
        assert attrs[OtelAttr.OUTPUT_TOKENS] == 0
        assert attrs[OtelAttr.CACHE_READ_INPUT_TOKENS] == 12
        assert attrs[OtelAttr.REASONING_OUTPUT_TOKENS] == 4
        assert [(value, rec_attrs[OtelAttr.T_TYPE]) for value, rec_attrs in client.token_usage_histogram.records] == [
            (0, "input"),
            (0, "output"),
        ]

    @pytest.mark.asyncio
    async def test_request_model_backfilled_from_response(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        client = _TelChat(turns=[_chat_response(model="served-model")], model=None)
        await client.get_response([Message(role="user", contents=["hi"])])
        (span,) = fake_otel.spans
        assert span.name == "chat served-model"
        assert span.attributes[OtelAttr.REQUEST_MODEL] == "served-model"

    @pytest.mark.asyncio
    async def test_finish_reason_falls_back_to_raw_representation(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        raw = SimpleNamespace(finish_reason="length")
        client = _TelChat(turns=[_chat_response(raw_representation=raw)])
        await client.get_response([Message(role="user", contents=["hi"])])
        assert fake_otel.spans[0].attributes[OtelAttr.FINISH_REASONS] == json.dumps(["length"])

    @pytest.mark.asyncio
    async def test_sensitive_off_captures_no_messages(
        self, fake_otel: _FakeOtel, gate_on: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger=TELEMETRY_LOGGER)
        client = _TelChat(turns=[_chat_response(finish_reason="stop")])
        await client.get_response([Message(role="user", contents=["hi"])], options={"instructions": "be nice"})
        attrs = fake_otel.spans[0].attributes
        assert OtelAttr.INPUT_MESSAGES not in attrs
        assert OtelAttr.OUTPUT_MESSAGES not in attrs
        assert OtelAttr.SYSTEM_INSTRUCTIONS not in attrs
        assert [record for record in caplog.records if isinstance(record.msg, dict)] == []

    @pytest.mark.asyncio
    async def test_sensitive_on_captures_input_output_and_events(
        self, fake_otel: _FakeOtel, gate_sensitive: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger=TELEMETRY_LOGGER)
        client = _TelChat(turns=[_chat_response("answer", finish_reason="tool_calls")])
        await client.get_response([Message(role="user", contents=["hi"])], options={"instructions": "be nice"})
        attrs = fake_otel.spans[0].attributes
        input_messages = json.loads(attrs[OtelAttr.INPUT_MESSAGES])
        assert input_messages[0]["parts"] == [{"type": "text", "content": "hi"}]
        assert json.loads(attrs[OtelAttr.SYSTEM_INSTRUCTIONS]) == [{"type": "text", "content": "be nice"}]
        output_messages = json.loads(attrs[OtelAttr.OUTPUT_MESSAGES])
        assert output_messages[-1]["finish_reason"] == "tool_call", "FINISH_REASON_MAP maps tool_calls -> tool_call"
        events = [record for record in caplog.records if isinstance(record.msg, dict)]
        assert len(events) == 2
        assert events[0].__dict__[OtelAttr.EVENT_NAME] == OtelAttr.USER_MESSAGE
        assert events[1].__dict__[OtelAttr.EVENT_NAME] == OtelAttr.CHOICE
        assert events[0].__dict__[MessageListTimestampFilter.INDEX_KEY] == 0

    @pytest.mark.asyncio
    async def test_sensitive_non_streaming_keeps_unmapped_finish_reason(
        self, fake_otel: _FakeOtel, gate_sensitive: None
    ) -> None:
        # Streaming and non-streaming report the same provider reason: one no map
        # entry covers is emitted verbatim on both paths, never dropped.
        client = _TelChat(turns=[_chat_response(finish_reason="max_tokens")])
        await client.get_response([Message(role="user", contents=["hi"])])
        (span,) = fake_otel.spans
        output_messages = json.loads(span.attributes[OtelAttr.OUTPUT_MESSAGES])
        assert output_messages[-1]["finish_reason"] == "max_tokens"
        assert span.exceptions == []
        assert span.status is None


# ---------------------------------------------------------------------------
# D. ChatTelemetryLayer — streaming, gate on
# ---------------------------------------------------------------------------


class TestChatLayerStreaming:
    @pytest.mark.asyncio
    async def test_streaming_setup_runs_under_chat_span(
        self, fake_otel: _FakeOtel, gate_on: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        active_spans: list[Any] = []

        @contextmanager
        def mark_active(span: Any) -> Any:
            active_spans.append(span)
            try:
                yield
            finally:
                active_spans.pop()

        class _SetupProbeWireChat(_WireChat):
            def __init__(self) -> None:
                super().__init__()
                self.setup_span: Any = None

            def get_response(self, *args: Any, **kwargs: Any) -> Any:
                self.setup_span = active_spans[-1] if active_spans else None
                return super().get_response(*args, **kwargs)

        class _SetupProbeChat(ChatTelemetryLayer, _SetupProbeWireChat):
            pass

        monkeypatch.setattr(instrumentation, "_activate_span", mark_active)
        client = _SetupProbeChat()

        stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
        _ = [update async for update in stream]
        await stream.get_final_response()

        assert client.setup_span is fake_otel.spans[0]

    @pytest.mark.asyncio
    async def test_hooks_registered_in_place(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        client = _TelChat()
        stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
        assert stream is client.returned[0], "wrapping mutates the core stream in place"
        assert len(stream._cleanup_hooks) == 2, "_record_duration + _finalize_stream"
        assert len(stream._pull_context_manager_factories) == 1
        assert await stream.get_final_response()

    @pytest.mark.asyncio
    async def test_full_consumption_finalizes_span_and_metrics(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        client = _TelChat(turns=[[_text_update("a"), _text_update("b", response_id="resp-s", finish_reason="stop")]])
        stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
        (span,) = fake_otel.spans
        assert not span.ended, "streaming span stays open until finalized"
        updates = [update async for update in stream]
        assert len(updates) == 2
        final = await stream.get_final_response()
        assert final.text == "ab"
        assert span.end_count == 1
        assert span.attributes[OtelAttr.RESPONSE_ID] == "resp-s"
        ((duration, _),) = client.duration_histogram.records
        assert duration >= 0, "_record_duration hook ran before _finalize_stream"

    @pytest.mark.asyncio
    async def test_mid_stream_error_skips_final_response(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        error = RuntimeError("mid")
        client = _TelChat(turns=[[_text_update("a"), error]])
        stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
        with pytest.raises(RuntimeError, match="mid"):
            _ = [update async for update in stream]
        await stream._run_cleanup_hooks()
        (span,) = fake_otel.spans
        assert span.end_count == 1
        assert span.attributes[OtelAttr.ERROR_TYPE] == "RuntimeError"
        assert span.exceptions == [error]
        assert client.token_usage_histogram.records == [], "get_final_response skipped on stream error"
        ((duration, duration_attrs),) = client.duration_histogram.records
        assert duration >= 0
        assert duration_attrs[OtelAttr.ERROR_TYPE] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_failing_result_hook_fails_the_stream_once(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        client = _TelChat()
        stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
        hook_calls: list[str] = []

        def first(response: ChatResponse) -> None:
            hook_calls.append("first")

        def failing(response: ChatResponse) -> None:
            hook_calls.append("failing")
            raise RuntimeError("hook failed")

        stream.with_result_hook(first).with_result_hook(failing)
        with pytest.raises(RuntimeError, match="hook failed"):
            _ = [update async for update in stream]

        assert hook_calls == ["first", "failing"], "no hook runs a second time"
        (span,) = fake_otel.spans
        assert span.end_count == 1
        assert span.attributes[OtelAttr.ERROR_TYPE] == "RuntimeError"
        ((_, duration_attrs),) = client.duration_histogram.records
        assert duration_attrs[OtelAttr.ERROR_TYPE] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_telemetry_capture_failure_never_fails_the_stream(
        self, fake_otel: _FakeOtel, gate_on: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken_capture(**_kwargs: Any) -> None:
            raise ValueError("telemetry bug")

        monkeypatch.setattr(instrumentation, "_capture_response", broken_capture)
        client = _TelChat()
        stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
        _ = [update async for update in stream]

        assert (await stream.get_final_response()).text == "done"
        assert fake_otel.spans[0].end_count == 1

    def test_sync_raise_closes_span_and_propagates(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        error = RuntimeError("sync boom")
        client = _TelChat(sync_error=error)
        with pytest.raises(RuntimeError, match="sync boom"):
            client.get_response([Message(role="user", contents=["hi"])], stream=True)
        (span,) = fake_otel.spans
        assert span.ended
        assert span.exceptions == [error]

    @pytest.mark.asyncio
    async def test_weakref_finalizer_closes_unconsumed_span_once(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        client = _TelChat()
        stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
        (span,) = fake_otel.spans
        client.returned.clear()  # the test double must not keep the stream alive
        del stream
        gc.collect()
        assert span.end_count == 1
        await asyncio.sleep(0)  # let any loop-scheduled GC callbacks settle before teardown

    @pytest.mark.asyncio
    async def test_close_once_after_consumption_and_gc(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        client = _TelChat()
        stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
        _ = [update async for update in stream]
        await stream.get_final_response()
        await stream._run_cleanup_hooks()
        del stream
        gc.collect()
        assert fake_otel.spans[0].end_count == 1

    @pytest.mark.asyncio
    async def test_sensitive_streaming_captures_input_then_output(
        self, fake_otel: _FakeOtel, gate_sensitive: None
    ) -> None:
        client = _TelChat(turns=[[_text_update("a", finish_reason="stop")]])
        stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
        (span,) = fake_otel.spans
        assert OtelAttr.INPUT_MESSAGES in span.attributes, "input captured synchronously at span start"
        assert OtelAttr.OUTPUT_MESSAGES not in span.attributes
        _ = [update async for update in stream]
        await stream.get_final_response()
        output_messages = json.loads(span.attributes[OtelAttr.OUTPUT_MESSAGES])
        assert output_messages[-1]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_sensitive_streaming_survives_unmapped_finish_reason(
        self, fake_otel: _FakeOtel, gate_sensitive: None
    ) -> None:
        # The streaming capture runs inside the span's own except/finally, so a failed
        # lookup of the provider's raw finish_reason would mark an otherwise successful
        # chat span as errored.
        client = _TelChat(turns=[[_text_update("a", finish_reason="max_tokens")]])
        stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
        _ = [update async for update in stream]
        await stream.get_final_response()
        (span,) = fake_otel.spans
        output_messages = json.loads(span.attributes[OtelAttr.OUTPUT_MESSAGES])
        assert output_messages[-1]["finish_reason"] == "max_tokens", "unmapped reason emitted verbatim"
        assert span.exceptions == []
        assert span.status is None, "capture must not turn a successful turn into an error span"


# ---------------------------------------------------------------------------
# E. Vendored telemetry contracts
# ---------------------------------------------------------------------------


class TestDriftPins:
    def test_vendored_otel_attr_values_stay_stable(self) -> None:
        expected = {
            "OPERATION": "gen_ai.operation.name",
            "PROVIDER_NAME": "gen_ai.provider.name",
            "ERROR_TYPE": "error.type",
            "REQUEST_MODEL": "gen_ai.request.model",
            "RESPONSE_MODEL": "gen_ai.response.model",
            "CACHE_CREATION_INPUT_TOKENS": "gen_ai.usage.cache_creation.input_tokens",
            "CACHE_READ_INPUT_TOKENS": "gen_ai.usage.cache_read.input_tokens",
            "REASONING_OUTPUT_TOKENS": "gen_ai.usage.reasoning.output_tokens",
            "TOOL_CALL_ID": "gen_ai.tool.call.id",
            "TOOL_NAME": "gen_ai.tool.name",
            "TOOL_TYPE": "gen_ai.tool.type",
            "TOOL_RESULT": "gen_ai.tool.call.result",
            "LLM_OPERATION_DURATION": "gen_ai.client.operation.duration",
            "LLM_TOKEN_USAGE": "gen_ai.client.token.usage",
            "MCP_METHOD_NAME": "mcp.method.name",
            "MEASUREMENT_FUNCTION_TAG_NAME": "chrys.function.name",
            "MEASUREMENT_FUNCTION_INVOCATION_DURATION": "chrys.function.invocation.duration",
        }
        for name, value in expected.items():
            assert OtelAttr[name].value == value

    def test_function_span_attributes_contract(self) -> None:
        t = tool(lambda text: text, name="echo", description="Echo it.")
        assert {
            str(k): str(v) for k, v in instrumentation.get_function_span_attributes(t, tool_call_id="c1").items()
        } == {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "echo",
            "gen_ai.tool.call.id": "c1",
            "gen_ai.tool.type": "function",
            "gen_ai.tool.description": "Echo it.",
        }
        assert instrumentation.get_function_span_attributes(t, tool_call_id=None)[OtelAttr.TOOL_CALL_ID] == "unknown"
        # No-description variant drops TOOL_DESCRIPTION on both sides.
        bare = tool(lambda text: text, name="bare", description="")
        assert str(OtelAttr.TOOL_DESCRIPTION) not in {
            str(k) for k in instrumentation.get_function_span_attributes(bare)
        }

    def test_function_duration_histogram_definition_literals(self, fake_otel: _FakeOtel) -> None:
        """Keep get_function_duration_histogram's name/unit/description/buckets
        byte-identical to the published metric definition so dashboards stay compatible."""
        histogram = instrumentation.get_function_duration_histogram()
        assert histogram.name == "chrys.function.invocation.duration"
        assert histogram.unit == "s"
        assert histogram.description == "Measures the duration of a function's execution"
        assert histogram.boundaries == OPERATION_DURATION_BUCKET_BOUNDARIES

    def test_bucket_boundaries_stay_stable(self) -> None:
        assert TOKEN_USAGE_BUCKET_BOUNDARIES == (
            1,
            4,
            16,
            64,
            256,
            1024,
            4096,
            16384,
            65536,
            262144,
            1048576,
            4194304,
            16777216,
            67108864,
        )
        assert OPERATION_DURATION_BUCKET_BOUNDARIES == (
            0.01,
            0.02,
            0.04,
            0.08,
            0.16,
            0.32,
            0.64,
            1.28,
            2.56,
            5.12,
            10.24,
            20.48,
            40.96,
            81.92,
        )

    def test_event_and_finish_reason_maps_stay_stable(self) -> None:
        assert {key: str(value) for key, value in ROLE_EVENT_MAP.items()} == {
            "system": "gen_ai.system.message",
            "user": "gen_ai.user.message",
            "assistant": "gen_ai.assistant.message",
            "tool": "gen_ai.tool.message",
        }
        assert FINISH_REASON_MAP == {
            "stop": "stop",
            "content_filter": "content_filter",
            "tool_calls": "tool_call",
            "length": "length",
        }

    def test_inner_dedup_field_names_stay_stable(self) -> None:
        assert INNER_RESPONSE_ID_CAPTURED_FIELD == "response_id"
        assert INNER_USAGE_CAPTURED_FIELD == "usage"

    def test_metric_attribute_allowlist_stays_stable(self) -> None:
        assert [str(attr) for attr in GEN_AI_METRIC_ATTRIBUTES] == [
            "gen_ai.operation.name",
            "gen_ai.provider.name",
            "gen_ai.request.model",
            "gen_ai.response.model",
            "server.address",
            "server.port",
        ]

    def test_timestamp_filter_index_key_stays_stable(self) -> None:
        assert MessageListTimestampFilter.INDEX_KEY == "chat_message_index"

    @pytest.mark.asyncio
    async def test_response_stream_private_stream_error_surface(self) -> None:
        # The telemetry finalizers read this private attribute DURING
        # cleanup-hook execution: the stream sets it just before running the
        # cleanup hooks on an iteration error and clears it right after
        # (_types.py:3116-3121). Pin the attribute name, the None default,
        # and the hook-time visibility window the finalizers depend on.
        async def _gen() -> Any:
            yield _text_update("a")
            raise RuntimeError("boom")

        stream = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)
        assert hasattr(stream, "_stream_error")
        assert stream._stream_error is None
        assert _stream_error_of(stream) is None
        seen_at_hook_time: list[BaseException | None] = []

        async def _probe() -> None:
            seen_at_hook_time.append(_stream_error_of(stream))

        stream.with_cleanup_hook(_probe)
        with pytest.raises(RuntimeError, match="boom"):
            _ = [update async for update in stream]
        assert len(seen_at_hook_time) == 1
        assert isinstance(seen_at_hook_time[0], RuntimeError)
        assert _stream_error_of(stream) is None, "cleared again after the cleanup hooks ran"

    def test_histogram_definitions_match_framework_literals(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        client = _TelChat()
        assert client.token_usage_histogram.name == "gen_ai.client.token.usage"
        assert client.token_usage_histogram.unit == "tokens"
        assert client.token_usage_histogram.description == "Captures the token usage of chat clients"
        assert client.token_usage_histogram.boundaries == TOKEN_USAGE_BUCKET_BOUNDARIES
        assert client.duration_histogram.name == "gen_ai.client.operation.duration"
        assert client.duration_histogram.unit == "s"
        assert (
            client.duration_histogram.description
            == "Captures the duration of operations of function-invoking chat clients"
        )
        assert client.duration_histogram.boundaries == OPERATION_DURATION_BUCKET_BOUNDARIES

    def test_scope_version_resolves_installed_chrys_version(self) -> None:
        assert instrumentation._SCOPE_NAME == "chrys"
        assert importlib.metadata.version("chrys") == instrumentation._SCOPE_VERSION


# ---------------------------------------------------------------------------
# F. AgentTelemetryLayer
# ---------------------------------------------------------------------------


class TestAgentLayer:
    @pytest.mark.asyncio
    async def test_streaming_setup_runs_under_agent_span(
        self, fake_otel: _FakeOtel, gate_on: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        active_spans: list[Any] = []

        @contextmanager
        def mark_active(span: Any) -> Any:
            active_spans.append(span)
            try:
                yield
            finally:
                active_spans.pop()

        class _SetupProbeCore(_CoreAgent):
            def __init__(self) -> None:
                super().__init__()
                self.setup_span: Any = None

            def run(self, *args: Any, **kwargs: Any) -> Any:
                self.setup_span = active_spans[-1] if active_spans else None
                return super().run(*args, **kwargs)

        class _SetupProbeAgent(AgentTelemetryLayer, _SetupProbeCore):
            pass

        monkeypatch.setattr(instrumentation, "_activate_span", mark_active)
        agent = _SetupProbeAgent()

        stream = agent.run("hi", stream=True)
        _ = [update async for update in stream]
        await stream.get_final_response()

        assert agent.setup_span is fake_otel.spans[0]

    def test_ctor_swallows_both_kwargs_with_precedence(self, fake_otel: _FakeOtel) -> None:
        # _CoreAgent has a strict signature: construction succeeding proves
        # the layer swallowed both telemetry kwargs.
        agent = _TelAgent(otel_agent_provider_name="agent-prov", otel_provider_name="chat-prov")
        assert agent.otel_provider_name == "agent-prov"
        assert _TelAgent(otel_provider_name="chat-prov").otel_provider_name == "chat-prov"
        assert _TelAgent().otel_provider_name == "stub-provider"

    def test_ctor_reads_provider_classvar_before_super(self, fake_otel: _FakeOtel) -> None:
        # AgentTelemetryLayer reads AGENT_PROVIDER_NAME
        # before the core __init__, so an instance attribute set there is
        # invisible — the ClassVar wins. The CTL pin asserts the opposite.
        agent = _TelLateProvAgent()
        assert agent.otel_provider_name == "stub-provider"
        assert agent.AGENT_PROVIDER_NAME == "instance-late"

    def test_ctor_builds_histograms(self, fake_otel: _FakeOtel) -> None:
        agent = _TelAgent()
        # str() the names: OtelAttr members hash by member name, not value, so
        # a set of members never equals a set of their value strings.
        assert {str(h.name) for h in fake_otel.meter.histograms} == {
            "gen_ai.client.token.usage",
            "gen_ai.client.operation.duration",
        }
        assert agent.token_usage_histogram in fake_otel.meter.histograms

    @pytest.mark.asyncio
    async def test_disabled_identity_passthrough(self, fake_otel: _FakeOtel, gate_off: None) -> None:
        agent = _TelAgent()
        coro = agent.run("hi")
        assert coro is agent.returned[0], "disabled path returns the core coroutine untouched"
        assert (await coro).text == "done"
        stream = agent.run("hi", stream=True)
        assert stream is agent.returned[1]
        assert stream._cleanup_hooks == []
        assert await stream.get_final_response()
        assert fake_otel.spans == []

    @pytest.mark.asyncio
    async def test_non_stream_span_name_and_agent_attributes(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        agent = _TelAgent(description="desc")
        session = SimpleNamespace(service_session_id="svc-7")
        response = await agent.run("hi", session=session)  # type: ignore[arg-type]
        assert response.text == "done"
        (span,) = fake_otel.spans
        assert span.name == "invoke_agent A"
        assert span.ended
        assert span.attributes[OtelAttr.OPERATION] == "invoke_agent"
        assert span.attributes[OtelAttr.PROVIDER_NAME] == "stub-provider"
        assert span.attributes[OtelAttr.AGENT_ID] == "agent-1"
        assert span.attributes[OtelAttr.AGENT_NAME] == "A"
        assert span.attributes[OtelAttr.AGENT_DESCRIPTION] == "desc"
        assert span.attributes[OtelAttr.CONVERSATION_ID] == "svc-7"

    @pytest.mark.asyncio
    async def test_agent_name_falls_back_to_id(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        agent = _TelAgent(name=None)
        await agent.run("hi")
        assert fake_otel.spans[0].name == "invoke_agent agent-1"
        assert fake_otel.spans[0].attributes[OtelAttr.AGENT_NAME] == "agent-1"

    @pytest.mark.asyncio
    async def test_thread_id_overrides_options_conversation_id(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        agent = _TelAgent(default_options={"conversation_id": "opt-conv"})
        await agent.run("hi")
        assert fake_otel.spans[0].attributes[OtelAttr.CONVERSATION_ID] == "opt-conv"
        session = SimpleNamespace(service_session_id="svc-7")
        await agent.run("hi", session=session)  # type: ignore[arg-type]
        assert fake_otel.spans[1].attributes[OtelAttr.CONVERSATION_ID] == "svc-7"

    @pytest.mark.asyncio
    async def test_merged_options_drive_span_attributes(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        agent = _TelAgent(default_options={"temperature": 0.3})
        await agent.run("hi", options={"max_tokens": 50})
        attrs = fake_otel.spans[0].attributes
        assert attrs[OtelAttr.REQUEST_TEMPERATURE] == 0.3, "default_options merged in"
        assert attrs[OtelAttr.REQUEST_MAX_TOKENS] == 50
        await agent.run("hi", options={"temperature": 0.9})
        assert fake_otel.spans[1].attributes[OtelAttr.REQUEST_TEMPERATURE] == 0.9, "run options win the merge"

    @pytest.mark.asyncio
    async def test_tool_definition_failure_is_isolated_at_live_boundary(
        self, fake_otel: _FakeOtel, gate_on: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A poisoned tool riding a live ``run`` must neither raise out of the
        telemetry layer nor suppress the valid definitions."""

        @tool(name="echo", description="Echo text.")
        async def echo(text: str) -> str:
            return text

        class _ModelDumpFailure:
            def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
                raise RuntimeError("model dump failed")

        agent = _TelAgent()
        with caplog.at_level(logging.WARNING, logger=TELEMETRY_LOGGER):
            response = await agent.run("hi", options={"tools": [echo, _ModelDumpFailure()]})

        assert response.text == "done"
        attrs = fake_otel.spans[0].attributes
        assert json.loads(attrs[OtelAttr.TOOL_DEFINITIONS]) == [
            {
                "type": "function",
                "name": "echo",
                "description": "Echo text.",
                "parameters": echo.parameters(),
            }
        ]
        assert "_ModelDumpFailure" in caplog.text

    def test_run_forwards_kwargs_middleware_only_if_given(self, fake_otel: _FakeOtel, gate_off: None) -> None:
        agent = _TelAgent()
        session = SimpleNamespace(service_session_id=None)
        sentinel_tools = [object()]
        agent.run(
            "hi",
            session=session,  # type: ignore[arg-type]
            tools=sentinel_tools,
            options={"a": 1},
            compaction_strategy="cs",
            tokenizer="tk",
            function_invocation_kwargs={"f": 1},
            client_kwargs={"c": 2},
        ).close()
        call = agent.calls[0]
        assert call["messages"] == "hi"
        assert call["session"] is session
        assert call["tools"] is sentinel_tools
        assert call["options"] == {"a": 1}
        assert call["compaction_strategy"] == "cs"
        assert call["tokenizer"] == "tk"
        assert call["function_invocation_kwargs"] == {"f": 1}
        assert call["client_kwargs"] == {"c": 2}
        assert "middleware" not in call, "middleware forwarded only when not None"
        middleware = [object()]
        agent.run("hi", middleware=middleware).close()
        assert agent.calls[1]["middleware"] is middleware

    @pytest.mark.asyncio
    async def test_non_stream_error_captured_and_vars_reset(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        error = RuntimeError("agent boom")
        agent = _TelAgent(turns=[error])
        with pytest.raises(RuntimeError, match="agent boom"):
            await agent.run("hi")
        (span,) = fake_otel.spans
        assert span.ended
        assert span.attributes[OtelAttr.ERROR_TYPE] == "RuntimeError"
        assert INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.get() is None
        assert INNER_ACCUMULATED_USAGE.get() is None

    @pytest.mark.asyncio
    async def test_non_stream_response_attributes(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        agent = _TelAgent(
            turns=[
                _agent_response(
                    response_id="rid-1",
                    usage_details={"input_token_count": 2, "output_token_count": 4},
                )
            ]
        )
        await agent.run("hi")
        attrs = fake_otel.spans[0].attributes
        assert attrs[OtelAttr.RESPONSE_ID] == "rid-1"
        assert attrs[OtelAttr.INPUT_TOKENS] == 2
        assert attrs[OtelAttr.OUTPUT_TOKENS] == 4

    @pytest.mark.asyncio
    async def test_streaming_hooks_and_built_but_unused_histograms(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        agent = _TelAgent(
            turns=[[_agent_update(response_id="rid-s")]],
        )
        stream = agent.run("hi", stream=True)
        assert stream is agent.returned[0], "wrapping mutates the core stream in place"
        assert len(stream._cleanup_hooks) == 2
        assert len(stream._pull_context_manager_factories) == 1
        _ = [update async for update in stream]
        final = await stream.get_final_response()
        assert final.response_id == "rid-s"
        (span,) = fake_otel.spans
        assert span.name == "invoke_agent A"
        assert span.end_count == 1
        assert span.attributes[OtelAttr.RESPONSE_ID] == "rid-s"
        # Mirror of :1807/:1868 — the agent layer builds histograms in the
        # ctor but never passes them to _capture_response.
        assert agent.token_usage_histogram.records == []
        assert agent.duration_histogram.records == []

    @pytest.mark.asyncio
    async def test_streaming_awaitable_result_wrapped_via_from_awaitable(
        self, fake_otel: _FakeOtel, gate_on: None
    ) -> None:
        agent = _TelAgent(turns=[[_agent_update("late")]], stream_awaitable=True)
        stream = agent.run("hi", stream=True)
        assert isinstance(stream, ResponseStream)
        assert stream is not agent.returned[0], "awaitable result is wrapped, not returned as-is"
        updates = [update async for update in stream]
        assert updates[0].text == "late"
        await stream.get_final_response()
        assert fake_otel.spans[0].end_count == 1

    def test_streaming_sync_raise_resets_vars_and_closes_span(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        error = RuntimeError("stream ctor boom")
        agent = _TelAgent(turns=[error])
        with pytest.raises(RuntimeError, match="stream ctor boom"):
            agent.run("hi", stream=True)
        (span,) = fake_otel.spans
        assert span.ended
        assert span.exceptions == [error]
        assert INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.get() is None, "same-context reset succeeded"
        assert INNER_ACCUMULATED_USAGE.get() is None

    @pytest.mark.asyncio
    async def test_streaming_mid_error_captured(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        error = RuntimeError("agent mid")
        agent = _TelAgent(turns=[[_agent_update("a"), error]])
        stream = agent.run("hi", stream=True)
        with pytest.raises(RuntimeError, match="agent mid"):
            _ = [update async for update in stream]
        await stream._run_cleanup_hooks()
        (span,) = fake_otel.spans
        assert span.end_count == 1
        assert span.attributes[OtelAttr.ERROR_TYPE] == "RuntimeError"


# ---------------------------------------------------------------------------
# G. INNER dedup between a real chat layer and a real agent layer
# ---------------------------------------------------------------------------


class TestInnerDedupIntegration:
    @pytest.mark.asyncio
    async def test_non_stream_agent_span_dedups_and_accumulates(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        agent = _PipelineAgent(chat=_TelChat(turns=_chat_usage_turns()))
        response = await agent.run("hi")
        assert response.response_id == "agent-rid"
        agent_span, chat_span_1, chat_span_2 = fake_otel.spans
        assert agent_span.name == "invoke_agent A"
        assert chat_span_1.attributes[OtelAttr.RESPONSE_ID] == "chat-1"
        assert chat_span_2.attributes[OtelAttr.RESPONSE_ID] == "chat-2"
        assert OtelAttr.RESPONSE_ID not in agent_span.attributes, "inner chat already captured a response_id"
        assert agent_span.attributes[OtelAttr.INPUT_TOKENS] == 10, "accumulated 3+7, not the agent's own 999"
        assert agent_span.attributes[OtelAttr.OUTPUT_TOKENS] == 16
        assert agent_span.attributes[OtelAttr.CACHE_READ_INPUT_TOKENS] == 6
        assert agent_span.attributes[OtelAttr.REASONING_OUTPUT_TOKENS] == 9

    @pytest.mark.asyncio
    async def test_streaming_agent_span_dedups_via_pull_context(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        agent = _PipelineAgent(chat=_TelChat(turns=_chat_usage_turns()))
        stream = agent.run("hi", stream=True)
        _ = [update async for update in stream]
        final = await stream.get_final_response()
        assert final.response_id == "agent-rid"
        agent_span = fake_otel.spans[0]
        assert agent_span.end_count == 1
        assert OtelAttr.RESPONSE_ID not in agent_span.attributes
        assert agent_span.attributes[OtelAttr.INPUT_TOKENS] == 3, "single inner chat call accumulated"
        assert agent_span.attributes[OtelAttr.OUTPUT_TOKENS] == 5
        assert agent_span.attributes[OtelAttr.CACHE_READ_INPUT_TOKENS] == 2
        assert agent_span.attributes[OtelAttr.REASONING_OUTPUT_TOKENS] == 1

    @pytest.mark.asyncio
    async def test_context_provider_instructions_are_copied_to_agent_span(
        self, fake_otel: _FakeOtel, gate_sensitive: None
    ) -> None:
        chat = _TelChat(turns=[_chat_response("answer")])

        class _InstructionCore(_ChatDrivenCore):
            def __init__(self, chat: _TelChat) -> None:
                super().__init__(chat=chat, chat_calls=1)
                self.default_options = {"instructions": "static"}

            def run(self, messages: Any = None, *, stream: bool = False, **kwargs: Any) -> Any:
                async def _go() -> AgentResponse:
                    response = await self.chat.get_response(
                        [Message(role="user", contents=["q0"])],
                        options={"instructions": "static\nprovider"},
                    )
                    return AgentResponse(messages=response.messages, response_id="agent-rid")

                return _go()

        class _InstructionAgent(AgentTelemetryLayer, _InstructionCore):
            pass

        agent = _InstructionAgent(chat=chat)
        await agent.run("hi")

        agent_span, chat_span = fake_otel.spans
        assert chat_span.parent == agent_span.get_span_context()
        assert json.loads(agent_span.attributes[OtelAttr.SYSTEM_INSTRUCTIONS]) == [
            {"type": "text", "content": "static\nprovider"}
        ]

    @pytest.mark.asyncio
    async def test_chat_layer_alone_leaves_inner_vars_untouched(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        client = _TelChat(turns=_chat_usage_turns())
        await client.get_response([Message(role="user", contents=["hi"])])
        assert INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.get() is None
        assert INNER_ACCUMULATED_USAGE.get() is None
        assert fake_otel.spans[0].attributes[OtelAttr.RESPONSE_ID] == "chat-1"


# ---------------------------------------------------------------------------
# H. INNER ContextVar lifecycle across task boundaries
# ---------------------------------------------------------------------------


class TestContextVarLifecycle:
    @pytest.mark.asyncio
    async def test_non_stream_coroutine_driven_in_foreign_task(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        # Call AgentTelemetryLayer.run() here and drive its coroutine inside
        # asyncio.create_task. Setting INNER vars in the synchronous caller
        # and resetting them in _run() would raise "token was created in a
        # different Context". Confining BOTH operations to _run() must keep
        # cleanup and inner-chat telemetry dedup working in the child task.
        agent = _PipelineAgent(chat=_TelChat(turns=_chat_usage_turns()))
        coroutine = agent.run("hi")
        assert INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.get() is None, "nothing set in the calling context"
        response = await asyncio.create_task(coroutine)  # type: ignore[arg-type]
        assert response.response_id == "agent-rid"
        assert INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.get() is None
        assert INNER_ACCUMULATED_USAGE.get() is None
        agent_span = fake_otel.spans[0]
        assert agent_span.ended
        assert agent_span.attributes[OtelAttr.INPUT_TOKENS] == 10, "dedup channel intact across the task boundary"

    @pytest.mark.asyncio
    async def test_streaming_same_task_resets_vars(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        agent = _TelAgent()
        stream = agent.run("hi", stream=True)
        assert INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.get() is not None, "streaming sets vars in the sync call"
        _ = [update async for update in stream]
        await stream.get_final_response()
        assert INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.get() is None, "finalizer reset in the same context"
        assert INNER_ACCUMULATED_USAGE.get() is None
        assert fake_otel.spans[0].end_count == 1

    @pytest.mark.asyncio
    async def test_streaming_cross_task_finalize_suppresses_reset(
        self, fake_otel: _FakeOtel, gate_on: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Hand the un-consumed stream to another task: the finalizer's reset
        # fires in a foreign context. AgentTelemetryLayer must close the span
        # before resetting ContextVars and downgrade the
        # cross-context reset to a debug log. The leaked set in this (dying)
        # context is inert.
        caplog.set_level(logging.DEBUG, logger=TELEMETRY_LOGGER)
        agent = _TelAgent(turns=[[_agent_update(response_id="rid-x")]])
        stream = agent.run("hi", stream=True)

        async def _consume() -> AgentResponse:
            _ = [update async for update in stream]
            return await stream.get_final_response()

        final = await asyncio.create_task(_consume())
        assert final.response_id == "rid-x"
        (span,) = fake_otel.spans
        assert span.end_count == 1, "span closed before the reset attempt"
        assert span.attributes[OtelAttr.RESPONSE_ID] == "rid-x"
        assert any("cross-context reset" in record.getMessage() for record in caplog.records)
        assert INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.get() is not None, "creator-context value left in place"

    @pytest.mark.asyncio
    async def test_nested_agent_runs_keep_outer_accumulation(self, fake_otel: _FakeOtel, gate_on: None) -> None:
        inner_agent = _TelAgent(
            turns=[
                _agent_response(
                    response_id="inner-rid",
                    usage_details={"input_token_count": 50, "output_token_count": 60},
                )
            ],
            id="inner",
            name="Inner",
        )
        chat = _TelChat(turns=_chat_usage_turns())

        class _NestingCore(_ChatDrivenCore):
            def run(self, messages: Any = None, *, stream: bool = False, **kwargs: Any) -> Any:
                async def _go() -> AgentResponse:
                    await inner_agent.run("sub")
                    response = await self.chat.get_response([Message(role="user", contents=["q0"])])
                    return AgentResponse(messages=response.messages, response_id="outer-rid")

                return _go()

        class _NestingAgent(AgentTelemetryLayer, _NestingCore):
            pass

        outer = _NestingAgent(chat=chat)
        await outer.run("hi")
        outer_span, inner_span, chat_span = fake_otel.spans
        assert inner_span.attributes[OtelAttr.RESPONSE_ID] == "inner-rid", "inner agent captures its own response"
        assert inner_span.attributes[OtelAttr.INPUT_TOKENS] == 50
        assert chat_span.attributes[OtelAttr.RESPONSE_ID] == "chat-1"
        assert OtelAttr.RESPONSE_ID not in outer_span.attributes, "outer dedups against the chat call"
        assert outer_span.attributes[OtelAttr.INPUT_TOKENS] == 3, "outer accumulation survives the nested reset"
        assert outer_span.attributes[OtelAttr.OUTPUT_TOKENS] == 5
        assert INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.get() is None


# ---------------------------------------------------------------------------
# I. Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_span_attributes_options_vs_kwargs_precedence(self) -> None:
        attrs = _get_span_attributes(
            operation_name="chat",
            all_options={"temperature": 0.7, "conversation_id": "opt-conv"},
            temperature=0.1,
        )
        assert attrs[OtelAttr.REQUEST_TEMPERATURE] == 0.7, "options consulted before kwargs"
        assert attrs[OtelAttr.CONVERSATION_ID] == "opt-conv"
        assert attrs[OtelAttr.CHOICE_COUNT] == 1, "default applies when absent"
        assert _get_span_attributes(choice_count=3)[OtelAttr.CHOICE_COUNT] == 3

    def test_span_attributes_thread_id_overrides_conversation_id(self) -> None:
        attrs = _get_span_attributes(all_options={"conversation_id": "C"}, thread_id="T")
        assert attrs[OtelAttr.CONVERSATION_ID] == "T"
        attrs = _get_span_attributes(all_options={"conversation_id": "C"}, thread_id=None)
        assert attrs[OtelAttr.CONVERSATION_ID] == "C"

    def test_span_attributes_error_and_encoding_transforms(self) -> None:
        attrs = _get_span_attributes(error=ValueError("x"))
        assert attrs[OtelAttr.ERROR_TYPE] == "ValueError"
        assert _get_span_attributes(encoding_formats="float")[OtelAttr.ENCODING_FORMATS] == json.dumps(["float"])
        assert _get_span_attributes(encoding_formats=["a", "b"])[OtelAttr.ENCODING_FORMATS] == json.dumps(["a", "b"])

    def test_span_attributes_flatten_function_tool_definition(self) -> None:
        @tool(name="echo", description="Echo text.")
        async def echo(text: str) -> str:
            return text

        attrs = _get_span_attributes(all_options={"tools": [echo]})
        assert json.loads(attrs[OtelAttr.TOOL_DEFINITIONS]) == [
            {
                "type": "function",
                "name": "echo",
                "description": "Echo text.",
                "parameters": echo.parameters(),
            }
        ]
        assert OtelAttr.TOOL_DEFINITIONS not in _get_span_attributes(all_options={"tools": []})

    def test_span_attributes_narrow_hosted_mcp_definition(self) -> None:
        hosted_mcp = {
            "type": "mcp",
            "server_label": "my_server",
            "server_description": "d",
            "server_url": "https://example.test/mcp",
            "headers": {"Authorization": "Bearer secret"},
        }

        serialized = _get_span_attributes(all_options={"tools": [hosted_mcp]})[OtelAttr.TOOL_DEFINITIONS]

        assert json.loads(serialized) == [{"type": "mcp", "name": "my_server", "description": "d"}]
        assert "Authorization" not in serialized
        assert "Bearer secret" not in serialized
        assert "server_url" not in serialized

    def test_span_attributes_maps_anthropic_input_schema_to_parameters(self) -> None:
        schema = {"type": "object", "properties": {"query": {"type": "string"}}}

        serialized = _get_span_attributes(
            all_options={"tools": [{"type": "custom", "name": "t", "input_schema": schema}]}
        )[OtelAttr.TOOL_DEFINITIONS]

        assert json.loads(serialized) == [{"type": "custom", "name": "t", "parameters": schema}]

    def test_span_attributes_falls_back_to_type_for_hosted_tool_name(self) -> None:
        serialized = _get_span_attributes(all_options={"tools": [{"type": "code_interpreter"}]})[
            OtelAttr.TOOL_DEFINITIONS
        ]

        assert json.loads(serialized) == [{"type": "code_interpreter", "name": "code_interpreter"}]

    def test_span_attributes_isolate_conversion_and_serialization_failures(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        @tool(name="echo", description="Echo text.")
        async def echo(text: str) -> str:
            return text

        class _ModelDumpFailure:
            def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
                raise RuntimeError("model dump failed")

        unserializable = {
            "type": "custom",
            "name": "bad-json",
            "parameters": {"default": object()},
        }
        with caplog.at_level(logging.WARNING, logger=TELEMETRY_LOGGER):
            attrs = _get_span_attributes(all_options={"tools": [echo, _ModelDumpFailure(), unserializable]})

        assert json.loads(attrs[OtelAttr.TOOL_DEFINITIONS]) == [
            {
                "type": "function",
                "name": "echo",
                "description": "Echo text.",
                "parameters": echo.parameters(),
            }
        ]
        assert "_ModelDumpFailure" in caplog.text
        assert "bad-json" in caplog.text

    def test_tool_definition_normalization_failure_skips_attribute(self, caplog: pytest.LogCaptureFixture) -> None:
        class _BrokenTools:
            def __bool__(self) -> bool:
                raise RuntimeError("normalization failed")

        with caplog.at_level(logging.WARNING, logger=TELEMETRY_LOGGER):
            assert _serialize_tool_definitions(_BrokenTools()) is None

        assert "Failed to normalize tool definitions" in caplog.text

    def test_get_data_bytes_as_str_matrix(self) -> None:
        assert _get_data_bytes_as_str(SimpleNamespace(type="text", uri=None)) is None  # type: ignore[arg-type]
        assert _get_data_bytes_as_str(SimpleNamespace(type="data", uri=None)) is None  # type: ignore[arg-type]
        assert _get_data_bytes_as_str(SimpleNamespace(type="data", uri="https://x")) is None  # type: ignore[arg-type]
        assert (
            _get_data_bytes_as_str(SimpleNamespace(type="data", uri="data:text/plain;base64,aGk="))  # type: ignore[arg-type]
            == "aGk="
        )
        with pytest.raises(ValueError, match="base64"):
            _get_data_bytes_as_str(SimpleNamespace(type="data", uri="data:text/plain,raw"))  # type: ignore[arg-type]

    def test_to_otel_part_matrix(self) -> None:
        assert _to_otel_part(Content.from_text("hi")) == {"type": "text", "content": "hi"}
        uri_part = _to_otel_part(Content.from_uri(uri="https://x/img.png", media_type="image/png"))
        assert uri_part == {
            "type": "uri",
            "uri": "https://x/img.png",
            "mime_type": "image/png",
            "modality": "image",
        }
        blob_part = _to_otel_part(
            SimpleNamespace(type="data", uri="data:image/png;base64,aGk=", media_type="image/png")  # type: ignore[arg-type]
        )
        assert blob_part == {"type": "blob", "content": "aGk=", "mime_type": "image/png", "modality": "image"}
        call_part = _to_otel_part(Content.from_function_call(call_id="c1", name="echo", arguments={"a": 1}))
        assert call_part == {"type": "tool_call", "id": "c1", "name": "echo", "arguments": {"a": 1}}
        result_part = _to_otel_part(Content.from_function_result(call_id="c1", result=None))
        assert result_part == {"type": "tool_call_response", "id": "c1", "response": ""}
        generic = SimpleNamespace(type="custom", to_dict=lambda exclude_none=True: {"type": "custom", "k": "v"})
        assert _to_otel_part(generic) == {"type": "custom", "k": "v"}  # type: ignore[arg-type]

    def test_get_instructions_from_options_matrix(self) -> None:
        assert _get_instructions_from_options(None) is None
        assert _get_instructions_from_options({"instructions": "sys"}) == "sys"
        assert _get_instructions_from_options({"instructions": ["a", "b"]}) == ["a", "b"]
        assert _get_instructions_from_options({"instructions": ["a", 1]}) is None
        assert _get_instructions_from_options({"instructions": 42}) is None
        assert _get_instructions_from_options("not-a-mapping") is None

    def test_mark_and_apply_accumulated_usage(self) -> None:
        fields_token = INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.set(set())
        usage_token = INNER_ACCUMULATED_USAGE.set({})
        try:
            _mark_inner_response_telemetry_captured(
                _chat_response(response_id="r1", usage_details={"input_token_count": 3, "output_token_count": 5})
            )
            _mark_inner_response_telemetry_captured(_chat_response(usage_details={"input_token_count": 7}))
            captured = INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.get()
            assert captured == {"response_id", "usage"}
            attrs: dict[str, Any] = {}
            _apply_accumulated_usage(attrs, captured)
            assert attrs[OtelAttr.INPUT_TOKENS] == 10
            assert attrs[OtelAttr.OUTPUT_TOKENS] == 5
            _apply_accumulated_usage(attrs2 := {}, set())
            assert attrs2 == {}, "not applied when usage was never captured"
        finally:
            INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.reset(fields_token)
            INNER_ACCUMULATED_USAGE.reset(usage_token)

    def test_mark_is_noop_without_agent_context(self) -> None:
        _mark_inner_response_telemetry_captured(_chat_response(response_id="r1"))
        assert INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.get() is None

    def test_get_response_attributes_capture_flags(self) -> None:
        response = _chat_response(
            response_id="rid",
            usage_details={"input_token_count": 3, "output_token_count": 5},
        )
        attrs = _get_response_attributes({}, response, capture_response_id=False, capture_usage=False)
        assert OtelAttr.RESPONSE_ID not in attrs
        assert OtelAttr.INPUT_TOKENS not in attrs
        attrs = _get_response_attributes({}, response)
        assert attrs[OtelAttr.RESPONSE_ID] == "rid"
        assert attrs[OtelAttr.INPUT_TOKENS] == 3

    def test_get_response_attributes_maps_legacy_usage_keys(self) -> None:
        attrs = _get_response_attributes(
            {},
            _chat_response(
                usage_details={
                    "anthropic.cache_creation_input_tokens": 1,
                    "prompt/cached_tokens": 2,
                    "completion/reasoning_tokens": 3,
                }
            ),
        )
        assert attrs[OtelAttr.CACHE_CREATION_INPUT_TOKENS] == 1
        assert attrs[OtelAttr.CACHE_READ_INPUT_TOKENS] == 2
        assert attrs[OtelAttr.REASONING_OUTPUT_TOKENS] == 3

    def test_get_response_attributes_prefers_provider_native_cache_key(self) -> None:
        attrs = _get_response_attributes(
            {},
            _chat_response(
                usage_details={
                    "cache_read_input_token_count": 1,
                    "prompt/cached_tokens": 10,
                    "deepseek.prompt_cache_hit_tokens": 5000,
                }
            ),
        )
        assert attrs[OtelAttr.CACHE_READ_INPUT_TOKENS] == 5000

    def test_agent_instruction_capture_ignores_unrelated_chat_span(self) -> None:
        agent_span = _FakeSpan(
            "invoke_agent A",
            context=_FakeSpanContext(trace_id=1, span_id=10),
            attributes={
                OtelAttr.OPERATION: OtelAttr.AGENT_INVOKE_OPERATION,
                OtelAttr.SYSTEM_INSTRUCTIONS: json.dumps([{"type": "text", "content": "static"}]),
            },
        )
        unrelated_chat_span = _FakeSpan(
            "chat m",
            context=_FakeSpanContext(trace_id=1, span_id=11),
            parent=_FakeSpanContext(trace_id=1, span_id=999),
        )

        _capture_current_agent_system_instructions(agent_span, unrelated_chat_span, "static\nprovider")

        assert json.loads(agent_span.attributes[OtelAttr.SYSTEM_INSTRUCTIONS]) == [
            {"type": "text", "content": "static"}
        ]

    def test_agent_instruction_capture_requires_invoke_agent_span(self) -> None:
        agent_span = _FakeSpan(
            "chat m",
            context=_FakeSpanContext(trace_id=1, span_id=10),
            attributes={
                OtelAttr.OPERATION: OtelAttr.CHAT_COMPLETION_OPERATION,
                OtelAttr.SYSTEM_INSTRUCTIONS: json.dumps([{"type": "text", "content": "static"}]),
            },
        )
        chat_span = _FakeSpan(
            "chat child",
            context=_FakeSpanContext(trace_id=1, span_id=11),
            parent=_FakeSpanContext(trace_id=1, span_id=10),
        )

        _capture_current_agent_system_instructions(agent_span, chat_span, "static\nprovider")

        assert json.loads(agent_span.attributes[OtelAttr.SYSTEM_INSTRUCTIONS]) == [
            {"type": "text", "content": "static"}
        ]

    def test_agent_instruction_capture_preserves_existing_prefix(self) -> None:
        agent_span = _FakeSpan(
            "invoke_agent A",
            context=_FakeSpanContext(trace_id=1, span_id=10),
            attributes={
                OtelAttr.OPERATION: OtelAttr.AGENT_INVOKE_OPERATION,
                OtelAttr.SYSTEM_INSTRUCTIONS: json.dumps([{"type": "text", "content": "static"}]),
            },
        )
        chat_span = _FakeSpan(
            "chat m",
            context=_FakeSpanContext(trace_id=1, span_id=11),
            parent=_FakeSpanContext(trace_id=1, span_id=10),
        )

        _capture_current_agent_system_instructions(agent_span, chat_span, "replacement\nprovider")

        assert json.loads(agent_span.attributes[OtelAttr.SYSTEM_INSTRUCTIONS]) == [
            {"type": "text", "content": "static"}
        ]

    def test_capture_exception_sets_type_event_and_status(self) -> None:
        span = _FakeSpan("x")
        error = ValueError("bad")
        capture_exception(span, error, timestamp=123)  # type: ignore[arg-type]
        assert span.attributes[OtelAttr.ERROR_TYPE] == "ValueError"
        assert span.exceptions == [error]
        assert span.status is not None and span.status[1] == repr(error)

    def test_get_span_name_fallbacks(self, fake_otel: _FakeOtel) -> None:
        with _get_span(attributes={}, span_name_attribute=OtelAttr.REQUEST_MODEL) as span:
            assert span.name == "operation unknown"  # type: ignore[union-attr]
        assert fake_otel.spans[0].end_count == 1, "use_span(end_on_exit=True) closes the span"

    def test_timestamp_filter_bumps_indexed_records(self) -> None:
        record = logging.LogRecord("n", logging.INFO, "p", 1, "m", None, None)
        baseline = record.created
        assert MessageListTimestampFilter().filter(record) is True
        assert record.created == baseline, "records without the index key are untouched"
        record.chat_message_index = 3  # type: ignore[attr-defined]
        MessageListTimestampFilter().filter(record)
        assert record.created == pytest.approx(baseline + 3e-6)

    def test_get_tracer_meter_chrys_scope_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_get_tracer(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "tracer"

        monkeypatch.setattr(instrumentation.trace, "get_tracer", fake_get_tracer)
        assert instrumentation.get_tracer() == "tracer"
        assert captured["instrumenting_module_name"] == "chrys"
        assert captured["instrumenting_library_version"] == importlib.metadata.version("chrys")

        meter_calls: list[dict[str, Any]] = []

        def fake_get_meter(**kwargs: Any) -> str:
            meter_calls.append(kwargs)
            if "attributes" in kwargs:
                raise TypeError("no attributes parameter")
            return "meter"

        monkeypatch.setattr(instrumentation.metrics, "get_meter", fake_get_meter)
        assert instrumentation.get_meter() == "meter", "TypeError falls back to the legacy signature"
        assert meter_calls[0]["name"] == "chrys"
        assert "attributes" not in meter_calls[1]

    def test_message_capture_event_roles_and_order(self, caplog: pytest.LogCaptureFixture) -> None:
        # _capture_messages itself is gate-agnostic: the sensitive gate lives
        # in its ChatTelemetryLayer and AgentTelemetryLayer callers.
        caplog.set_level(logging.INFO, logger=TELEMETRY_LOGGER)
        span = _FakeSpan("x")
        instrumentation._capture_messages(
            span=span,  # type: ignore[arg-type]
            provider_name="p",
            messages=[
                Message(role="system", contents=["s"]),
                Message(role="user", contents=["u"]),
                Message(role="assistant", contents=["a"]),
            ],
        )
        events = [record for record in caplog.records if isinstance(record.msg, dict)]
        names = [record.__dict__[OtelAttr.EVENT_NAME] for record in events]
        assert names == [OtelAttr.SYSTEM_MESSAGE, OtelAttr.USER_MESSAGE, OtelAttr.ASSISTANT_MESSAGE]
        indices = [record.__dict__[MessageListTimestampFilter.INDEX_KEY] for record in events]
        assert indices == [0, 1, 2]
        parsed = json.loads(span.attributes[OtelAttr.INPUT_MESSAGES])
        assert [message["role"] for message in parsed] == ["system", "user", "assistant"]

    def test_capture_messages_finish_reason_sink_is_total(self) -> None:
        span = _FakeSpan("x")
        instrumentation._capture_messages(
            span=span,  # type: ignore[arg-type]
            provider_name="p",
            messages=[Message(role="assistant", contents=["a"])],
            output=True,
            finish_reason="max_tokens",  # type: ignore[arg-type]
        )
        assert json.loads(span.attributes[OtelAttr.OUTPUT_MESSAGES])[-1]["finish_reason"] == "max_tokens"

        # Defensive half: every caller that passes a finish_reason is guarded by a
        # non-empty message list, so the empty tail is only reachable from here.
        empty = _FakeSpan("x")
        instrumentation._capture_messages(
            span=empty,  # type: ignore[arg-type]
            provider_name="p",
            messages=[],
            output=True,
            finish_reason="stop",
        )
        assert json.loads(empty.attributes[OtelAttr.OUTPUT_MESSAGES]) == []
