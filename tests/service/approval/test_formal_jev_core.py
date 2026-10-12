# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Minimal Formal/Jev runtime contracts using the real factory and mocked HTTP."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from types import ModuleType
from typing import Any
from unittest.mock import create_autospec

import httpx
import pytest

from chrys.service.approval import judge as judge_module
from chrys.service.approval.judge import _SYSTEM_PROMPT, ApprovalJudge, FormalEvaluationCancelled, JudgeVerdict
from chrys.service.approval.predicate import (
    PredicateAssetError,
    TruthValue,
    load_default_asset,
    parse_asset,
    validate_predicate_response,
)
from chrys.service.llm import clients
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.waiting import wait_for

_REASONING = ModelProfile(
    id="main-reasoning",
    name="Main reasoning",
    model_id="reasoning-test",
    base_url="https://reasoning.test/v1",
    api_key="reasoning-key",
    stream=False,
    chat_options='{"reasoning_effort":"high"}',
    max_output_tokens=222,
    http_max_retries=0,
    http_read_timeout=60,
)


def _values(value: TruthValue = False) -> dict[str, TruthValue]:
    return dict.fromkeys((p.id for p in load_default_asset().principles), value)


def _reply(values: dict[str, TruthValue], model: str) -> str | dict[str, Any]:
    if model == "test":
        return json.dumps(values)
    return {
        "answers": {key: {"type": "choice", "choice": str(value).lower()} for key, value in values.items()},
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }


@asynccontextmanager
async def _judge(
    monkeypatch: pytest.MonkeyPatch,
    replies: list[str | dict[str, Any] | httpx.Response],
    *,
    model: str = "test",
    formal: bool = True,
    reasoning: ModelProfile | None = _REASONING,
) -> AsyncIterator[tuple[ApprovalJudge, list[httpx.Request]]]:
    calls: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        reply = replies.pop(0)
        if isinstance(reply, httpx.Response):
            return reply
        if request.url.path.endswith("/chat/completions"):
            reply = {
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
        return httpx.Response(200, json=reply)

    http_clients: list[httpx.AsyncClient] = []

    def build_http(profile, timeout, *, raw_http_log_path=None, session_id=None):
        http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        http_clients.append(http)
        return http

    monkeypatch.setattr(
        clients,
        "_build_profile_http_client",
        create_autospec(clients._build_profile_http_client, side_effect=build_http),
    )
    judge = ApprovalJudge(
        ModelProfile(
            id="core-test",
            name="Core test",
            model_id=model,
            base_url="https://provider.test/v1",
            api_key="synthetic-key",
            stream=formal and model != "test",
            http_max_retries=0,
            http_read_timeout=5,
        ),
        reasoning_profile=reasoning,
    )
    try:
        yield judge, calls
    finally:
        await judge.aclose()
        assert all(http.is_closed for http in http_clients)


async def _evaluate(judge: ApprovalJudge, formal: bool = True) -> JudgeVerdict:
    return await judge.evaluate(
        "Inspect source", "read_file", "filesystem.read", {"path": "main.py"}, ["/workspace"], formal=formal
    )


@pytest.mark.parametrize("model", ["test", "typesafe/jev-test", "~typesafe/jev-test"])
@pytest.mark.parametrize("triggered", list(_values()))
@pytest.mark.parametrize("other", [False, "unknown"])
async def test_any_true_requires_review_without_reasoning(monkeypatch, model, triggered, other):
    values = _values(other) | {triggered: True}
    async with _judge(monkeypatch, [_reply(values, model)], model=model) as (judge, calls):
        verdict = await _evaluate(judge)
        assert verdict.approved is False
        assert triggered in verdict.reason
        assert verdict.audit["route"] == "predicate_reasoning_v1"
        assert verdict.audit["stages"] == ["predicate"]
        assert verdict.audit["application_attempts"] == verdict.audit["transport_attempts"] == len(calls) == 1
        assert verdict.audit["usage"]["total_token_count"] == 5
        assert verdict.audit["predicate_results"] == [{"id": key, "value": value} for key, value in values.items()]
        assert judge._reasoning_judge._client is None
        body = json.loads(calls[0].content)
        assert body["model"] == model
        definitions = json.loads(files("chrys.service.approval").joinpath("principles.json").read_text())
        if model == "test":
            assert calls[0].url.path == "/v1/chat/completions"
            messages = body["messages"]
        else:
            assert calls[0].url.path == "/v1/decisions"
            assert body["questions"] == {
                p["id"]: {
                    "type": "choice",
                    "instructions": p,
                    "criteria": {
                        "true": "The condition holds.",
                        "false": "The condition does not hold.",
                        "unknown": "The available input is insufficient to determine whether the condition holds.",
                    },
                }
                for p in definitions
            }
            assert "stream" not in body
            messages = body["state"]["messages"]
        rendered = messages[0]["content"].split("Fixed principles:\n", 1)[1].split("\n\nReturn JSON", 1)[0]
        assert json.loads(rendered) == definitions
        assert [message["role"] for message in messages] == ["system", "user"]
        assert "main.py" in messages[1]["content"]
        assert calls[0].headers["authorization"] == "Bearer synthetic-key"
        assert "synthetic-key" not in calls[0].content.decode()


@pytest.mark.parametrize("model", ["test", "typesafe/jev-test", "~typesafe/jev-test"])
@pytest.mark.parametrize(
    "reasoning", [_REASONING, None, ModelProfile(id="jev", name="Jev", model_id="typesafe/jev-test")]
)
async def test_all_false_approves_without_reasoning(monkeypatch, tmp_path, model, reasoning):
    values = _values()
    async with _judge(monkeypatch, [_reply(values, model)], model=model, reasoning=reasoning) as (judge, calls):
        verdict = await judge.evaluate(
            "Inspect source",
            "read_file",
            "filesystem.read",
            {"path": "main.py"},
            ["/workspace"],
            request_id="safe",
            log_dir=tmp_path,
            formal=True,
        )
        assert verdict.approved is True
        assert verdict.audit["approved"] is True
        assert verdict.audit["decision_version"] == "three-valued-or-v2"
        assert verdict.audit["stages"] == ["predicate"]
        assert verdict.audit["failure_reason"] is None
        assert verdict.audit["application_attempts"] == verdict.audit["transport_attempts"] == len(calls) == 1
        assert verdict.audit["usage"]["total_token_count"] == 5
        assert verdict.audit["predicate_results"] == [{"id": key, "value": False} for key in values]
        assert judge._reasoning_judge is None or judge._reasoning_judge._client is None
        saved = json.loads((tmp_path / "safe.formal.jsonl").read_text())
        assert saved == {
            **{key: value for key, value in verdict.audit.items() if key != "calls"},
            "calls_file": "safe.calls.jsonl",
        }


@pytest.mark.parametrize("model", ["test", "typesafe/jev-test", "~typesafe/jev-test"])
@pytest.mark.parametrize(
    "values",
    [_values() | {"external_action": "unknown"}, _values("unknown")],
    ids=["mixed", "all-unknown"],
)
@pytest.mark.parametrize("approved", [False, True])
async def test_unknown_without_true_uses_main_reasoning_profile_and_direct_verdict(
    monkeypatch, model, values, approved
):
    replies = [_reply(values, model), json.dumps({"approved": approved, "reason": "reasoning verdict"})]
    async with _judge(monkeypatch, replies, model=model) as (judge, calls):
        verdict = await judge.evaluate(
            "Inspect source",
            "read_file",
            "filesystem.read",
            {"path": "main.py"},
            ["/workspace"],
            user_messages=["Use this workspace", "Inspect source"],
            formal=True,
        )
        assert verdict.approved is approved
        assert verdict.reason == "reasoning verdict"
        assert verdict.audit["stages"] == ["predicate", "reasoning"]
        assert verdict.audit["application_attempts"] == verdict.audit["transport_attempts"] == len(calls) == 2
        assert verdict.audit["usage"]["total_token_count"] == 10
        assert [c["model_profile_id"] for c in verdict.audit["calls"]] == ["core-test", _REASONING.id]
        assert verdict.audit["predicate_results"] == [{"id": k, "value": v} for k, v in values.items()]
        body = json.loads(calls[1].content)
        assert calls[1].url == "https://reasoning.test/v1/chat/completions"
        assert calls[1].headers["authorization"] == "Bearer reasoning-key"
        assert body["model"] == _REASONING.model_id
        assert body["reasoning_effort"] == "high"
        assert body["max_completion_tokens"] == 222
        assert body["stream"] is False
        assert body["messages"][0]["content"] == _SYSTEM_PROMPT
        for expected in (
            "Use this workspace",
            "Inspect source",
            "read_file",
            "filesystem.read",
            "main.py",
            "/workspace",
        ):
            assert expected in body["messages"][1]["content"]


@pytest.mark.parametrize("model", ["test", "typesafe/jev-test"])
@pytest.mark.parametrize("recovers", [False, True])
async def test_invalid_response_retries_before_reasoning(monkeypatch, model, recovers):
    invalid = "{}" if model == "test" else {"answers": {}}
    replies = (
        [invalid, _reply(_values("unknown"), model), '{"approved":true,"reason":"ok"}'] if recovers else [invalid] * 3
    )
    async with _judge(monkeypatch, replies, model=model) as (judge, calls):
        verdict = await _evaluate(judge)
        assert verdict.approved is recovers
        assert len(calls) == verdict.audit["application_attempts"] == 3
        assert verdict.audit["stages"] == (["predicate", "reasoning"] if recovers else ["predicate"])
        assert verdict.audit["failure_reason"] == (None if recovers else "invalid_shape")
        body = json.loads(calls[1].content)
        messages = body["messages"] if model == "test" else body["state"]["messages"]
        assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
        assert "Invalid response" in messages[-1]["content"]


async def test_disabled_formal_keeps_direct_even_with_jev_model_name(monkeypatch):
    async with _judge(
        monkeypatch, ['{"approved":true,"reason":"direct"}'], model="typesafe/jev-test", formal=False
    ) as (judge, calls):
        verdict = await _evaluate(judge, formal=False)
        assert verdict.approved is True
        assert verdict.reason == "direct"
        assert verdict.audit is None
        assert len(calls) == 1
        assert calls[0].url.path == "/v1/chat/completions"


@pytest.mark.parametrize("message,tool_name", [("Inspect source", ""), ("", "read_file")])
async def test_missing_context_retains_manual_review_policy(monkeypatch, tmp_path, message, tool_name):
    async with _judge(monkeypatch, []) as (judge, calls):
        verdict = await judge.evaluate(
            message,
            tool_name,
            "filesystem.read",
            {"path": "main.py"},
            ["/workspace"],
            request_id="invalid",
            log_dir=tmp_path,
            formal=True,
        )
        assert verdict.approved is False
        assert verdict.reason == "Approval judge input is invalid"
        assert calls == []
        assert verdict.audit is not None
        assert verdict.audit["failure_reason"] == "invalid_input"
        assert verdict.audit["stages"] == verdict.audit["calls"] == []
        assert verdict.audit["application_attempts"] == 0
        saved = json.loads((tmp_path / "invalid.formal.jsonl").read_text())
        assert saved["failure_reason"] == "invalid_input"
        assert saved["request_id"] == "invalid"


@pytest.mark.parametrize("model", ["test", "typesafe/jev-test"])
@pytest.mark.parametrize("triggered", [False, True])
@pytest.mark.parametrize("kind", ["", "   "])
async def test_missing_tool_kind_still_uses_predicate_review(monkeypatch, model, triggered, kind):
    values = _values() | {"external_action": triggered}
    async with _judge(monkeypatch, [_reply(values, model)], model=model) as (judge, calls):
        verdict = await judge.evaluate(
            "Inspect source", "read_file", kind, {"path": "main.py"}, ["/workspace"], formal=True
        )
        assert verdict.approved is (not triggered)
        assert len(calls) == 1
        assert verdict.audit["stages"] == ["predicate"]
        assert verdict.audit["failure_reason"] is None
        assert verdict.audit["predicate_results"] == [{"id": key, "value": value} for key, value in values.items()]
        body = json.loads(calls[0].content)
        messages = body["messages"] if model == "test" else body["state"]["messages"]
        assert f"Tool: read_file (kind: {kind})" in messages[1]["content"]


@pytest.mark.parametrize("timeout", [0, -1, None])
async def test_unusable_time_budget_records_audit_without_model_calls(monkeypatch, tmp_path, timeout):
    async with _judge(monkeypatch, []) as (judge, calls):
        judge.profile.http_read_timeout = timeout
        verdict = await judge.evaluate(
            "Inspect source", "read_file", "filesystem.read", {}, [], request_id="budget", log_dir=tmp_path, formal=True
        )
        assert verdict.approved is False
        assert verdict.reason == "Approval judge time budget exhausted"
        assert calls == []
        assert verdict.audit is not None
        assert verdict.audit["failure_reason"] == "timeout"
        assert verdict.audit["stages"] == verdict.audit["calls"] == []
        saved = json.loads((tmp_path / "budget.formal.jsonl").read_text())
        assert saved["failure_reason"] == "timeout"


@pytest.mark.parametrize("formal", [False, True], ids=["direct", "formal"])
@pytest.mark.parametrize("transport", ["missing", "wrong-type"])
async def test_unavailable_sdk_transport_does_not_block_judging(monkeypatch, formal, transport):
    from chrys.kernel import ChatResponse, Message

    class ClientWithoutTransport:
        def __init__(self):
            self.sdk_client = object() if transport == "missing" else type("SDK", (), {"_client": object()})()
            self.closed = False

        async def aclose(self):
            self.closed = True

    client = ClientWithoutTransport()
    monkeypatch.setattr(clients, "create_client", create_autospec(clients.create_client, return_value=client))
    responses = [ChatResponse(messages=[Message("assistant", ['{"approved":true,"reason":"ok"}'])])]
    if formal:
        responses.insert(0, ChatResponse(messages=[Message("assistant", [json.dumps(_values("unknown"))])]))
    respond = create_autospec(judge_module.get_final_response, side_effect=responses)
    monkeypatch.setattr(judge_module, "get_final_response", respond)
    async with _judge(monkeypatch, [], formal=formal) as (judge, _):
        # The reasoning stage uses Direct with transport auditing enabled, too.
        verdict = await _evaluate(judge, formal=formal)
        assert verdict.approved is True
        assert respond.await_count == (2 if formal else 1)
        if formal:
            assert verdict.audit["transport_count_available"] is False
            assert verdict.audit["application_attempts"] == 2
            assert all(call["transport_count_available"] is False for call in verdict.audit["calls"])
    assert client.closed


async def test_openai_sdk_transport_contract(monkeypatch):
    async with _judge(monkeypatch, [_reply(_values(), "test")]) as (judge, _):
        client = await judge._get_client()
        transport = getattr(client.sdk_client, "_client", None)
        assert isinstance(transport, httpx.AsyncClient), "Revisit transport auditing after an OpenAI SDK change"
        assert transport.event_hooks["request"].count(judge_module._count_transport_request) == 0
        assert (await _evaluate(judge)).approved is True
        assert transport.event_hooks["request"].count(judge_module._count_transport_request) == 1


async def test_one_judge_switches_direct_and_formal_without_model_configuration(monkeypatch):
    replies = ['{"approved":true,"reason":"direct"}', _reply(_values() | {"external_action": True}, "test")]
    async with _judge(monkeypatch, replies) as (judge, calls):
        direct = await _evaluate(judge, formal=False)
        formal = await _evaluate(judge, formal=True)
        assert direct.approved is True and direct.audit is None
        assert formal.approved is False and formal.audit["stages"] == ["predicate"]
        first, second = [json.loads(c.content)["messages"][0]["content"] for c in calls]
        assert first == judge_module._SYSTEM_PROMPT
        assert "Fixed principles:" in second


async def test_direct_mode_with_a_jev_judge_uses_the_main_model(monkeypatch):
    async with _judge(monkeypatch, ['{"approved":true,"reason":"direct"}'], model="typesafe/jev-test") as (
        judge,
        calls,
    ):
        verdict = await _evaluate(judge, formal=False)
        assert verdict.approved is True and verdict.audit is None
        assert len(calls) == 1
        body = json.loads(calls[0].content)
        assert str(calls[0].url) == "https://reasoning.test/v1/chat/completions"
        assert body["model"] == _REASONING.model_id
        assert body["messages"][0]["content"] == judge_module._SYSTEM_PROMPT


@pytest.mark.parametrize("model", ["test", "typesafe/jev-test"])
async def test_empty_latest_message_uses_shared_user_context(monkeypatch, model):
    async with _judge(
        monkeypatch, [_reply(_values("unknown"), model), '{"approved":true,"reason":"ok"}'], model=model
    ) as (
        judge,
        calls,
    ):
        verdict = await judge.evaluate(
            "",
            "read_file",
            "filesystem.read",
            {"path": "main.py"},
            ["/workspace"],
            user_messages=["Inspect the parent session source"],
            formal=True,
        )
        assert verdict.approved is True
        assert verdict.audit["application_attempts"] == len(calls) == 2
        body = json.loads(calls[0].content)
        messages = body["messages"] if model == "test" else body["state"]["messages"]
        assert "Inspect the parent session source" in messages[1]["content"]


@pytest.mark.parametrize(
    "response",
    [
        "{}",
        "invalid JSON",
        *[json.dumps(_values() | {"external_action": value}) for value in (None, 0, 1, "true", "UNKNOWN", [], {})],
        json.dumps(_values() | {"external_action": "false"}),
        json.dumps(_values() | {"extra": False}),
        json.dumps(_values()).replace('"external_action": false', '"external_action": true, "external_action": false'),
    ],
)
def test_predicates_reject_missing_duplicate_extra_or_illegal_values(response):
    with pytest.raises(PredicateAssetError):
        validate_predicate_response(response, load_default_asset())


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(f"```json\n{json.dumps(_values())}\n```", id="json"),
        pytest.param(f"```\n{json.dumps(_values())}\n```", id="plain"),
        pytest.param(f" \n```JSON\n{json.dumps(_values())}\n```\n ", id="uppercase-with-whitespace"),
    ],
)
def test_predicates_accept_complete_fence(response):
    values = validate_predicate_response(response, load_default_asset())
    assert {value.id: value.value for value in values} == _values()


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(f"Here you go:\n```json\n{json.dumps(_values())}\n```", id="leading-prose"),
        pytest.param(f"```json\n{json.dumps(_values())}\n```\n```json\n{{}}\n```", id="two-blocks"),
        pytest.param(
            "```json\n" + json.dumps(_values()).replace(", ", ",\n```\n```json\n", 1) + "\n```",
            id="object-split-across-blocks",
        ),
        pytest.param(f"```json\n{json.dumps(_values())}", id="unterminated"),
        pytest.param(
            "```json\n"
            + json.dumps(_values()).replace(
                '"external_action": false', '"external_action": true, "external_action": false'
            )
            + "\n```",
            id="duplicate-key",
        ),
    ],
)
def test_predicates_reject_invalid_fenced_response(response):
    with pytest.raises(PredicateAssetError):
        validate_predicate_response(response, load_default_asset())


@pytest.mark.parametrize("value", [False, True, "unknown"], ids=["approve", "human-review", "reasoning"])
async def test_fenced_formal_response_routes_without_parse_retry(monkeypatch, value):
    values = _values() | {"external_action": value}
    replies = [f"```json\n{json.dumps(values)}\n```"]
    if value == "unknown":
        replies.append('{"approved":true,"reason":"ok"}')
    async with _judge(monkeypatch, replies) as (judge, calls):
        verdict = await _evaluate(judge)
        assert verdict.approved is (value is not True)
        assert (
            verdict.audit["application_attempts"]
            == verdict.audit["transport_attempts"]
            == len(calls)
            == (2 if value == "unknown" else 1)
        )
        assert verdict.audit["predicate_results"] == [{"id": key, "value": value} for key, value in values.items()]


@pytest.mark.parametrize("reasoning", [None, ModelProfile(id="jev", name="Jev", model_id="typesafe/jev-test")])
async def test_missing_or_jev_reasoning_profile_requires_human(monkeypatch, reasoning):
    async with _judge(monkeypatch, [json.dumps(_values("unknown"))], reasoning=reasoning) as (judge, calls):
        verdict = await _evaluate(judge)
        assert verdict.approved is False
        assert verdict.audit["failure_reason"] == "reasoning_profile_unavailable"
        assert len(calls) == 1


@pytest.mark.parametrize("recovers", [False, True])
async def test_reasoning_reuses_direct_retry_and_parser(monkeypatch, recovers):
    replies = [json.dumps(_values("unknown")), "invalid"]
    replies += ['{"approved":true,"reason":"repaired'] if recovers else ["invalid", "invalid"]
    async with _judge(monkeypatch, replies) as (judge, calls):
        verdict = await _evaluate(judge)
        assert verdict.approved is recovers
        assert verdict.audit["failure_reason"] == (None if recovers else "invalid_verdict")
        assert [c["stage"] for c in verdict.audit["calls"]] == ["predicate"] + ["reasoning"] * (2 if recovers else 3)
        body = json.loads(calls[2].content)
        assert body["model"] == _REASONING.model_id
        assert "Invalid response" in body["messages"][-1]["content"]


@pytest.mark.parametrize("choice", [None, True, False, 1, "TRUE", "undefined", {}, []])
async def test_illegal_jev_choices_are_not_unknown(monkeypatch, choice):
    reply = _reply(_values(), "typesafe/jev-test")
    reply["answers"]["external_action"]["choice"] = choice
    async with _judge(monkeypatch, [reply] * 3, model="typesafe/jev-test") as (judge, calls):
        verdict = await _evaluate(judge)
        assert verdict.approved is False
        assert verdict.audit["failure_reason"] == "invalid_shape"
        assert verdict.audit["predicate_results"] is None
        assert verdict.audit["stages"] == ["predicate"]
        assert len(calls) == 3


@pytest.mark.parametrize("mutation", ["legacy", "duplicate", "missing", "bad-role", "bad-example", "bad-label"])
def test_invalid_assets_are_rejected(mutation):
    definitions = json.loads(files("chrys.service.approval").joinpath("principles.json").read_text())
    if mutation == "legacy":
        definitions = {"principles": definitions}
    elif mutation == "duplicate":
        definitions.append(definitions[0])
    elif mutation == "missing":
        del definitions[0]["exceptions"]
    elif mutation == "bad-role":
        definitions[0]["role"] = []
    elif mutation == "bad-example":
        del definitions[0]["examples"][0]["reason"]
    else:
        definitions[0]["examples"][0]["result"] = True
    with pytest.raises(PredicateAssetError):
        parse_asset(json.dumps(definitions))


async def test_asset_failure_never_calls_either_model(monkeypatch):
    monkeypatch.setattr(
        judge_module, "load_default_asset", create_autospec(load_default_asset, side_effect=PredicateAssetError)
    )
    async with _judge(monkeypatch, []) as (judge, calls):
        verdict = await _evaluate(judge)
        assert verdict.approved is False
        assert verdict.audit["failure_reason"] == "asset_unavailable"
        assert calls == []


@pytest.mark.parametrize("stage", ["predicate", "reasoning"])
@pytest.mark.parametrize("error", [RuntimeError("provider failed"), TimeoutError("provider timed out")])
async def test_stage_failure_requires_human(monkeypatch, stage, error):
    from chrys.kernel import ChatResponse, Message

    replies = (
        [error]
        if stage == "predicate"
        else [ChatResponse(messages=[Message("assistant", [json.dumps(_values("unknown"))])]), error]
    )
    monkeypatch.setattr(
        judge_module, "get_final_response", create_autospec(judge_module.get_final_response, side_effect=replies)
    )
    async with _judge(monkeypatch, []) as (judge, _):
        verdict = await _evaluate(judge)
        assert verdict.approved is False
        assert verdict.audit["failure_reason"] == ("timeout" if isinstance(error, TimeoutError) else "extraction_error")
        assert verdict.audit["stages"][-1] == stage
        assert verdict.audit["calls"][-1]["status"] == "failed"


@pytest.mark.parametrize("stage", ["predicate", "reasoning"])
async def test_cancellation_stays_cancellation_at_either_stage(monkeypatch, stage):
    from chrys.kernel import ChatResponse, Message

    entered = asyncio.Event()
    release = asyncio.Event()

    async def response(client, messages, *, stream, options, timeout):
        if (messages[0].text == _SYSTEM_PROMPT) is (stage == "reasoning"):
            entered.set()
            await release.wait()
        return ChatResponse(messages=[Message("assistant", [json.dumps(_values("unknown"))])])

    monkeypatch.setattr(
        judge_module, "get_final_response", create_autospec(judge_module.get_final_response, side_effect=response)
    )
    async with _judge(monkeypatch, []) as (judge, _):
        task = asyncio.create_task(_evaluate(judge))
        try:
            await wait_for(lambda: entered.is_set() or task.done())
            assert entered.is_set() and not task.done()
            task.cancel()
            with pytest.raises(FormalEvaluationCancelled) as cancelled:
                await task
            audit = cancelled.value.verdict.audit
            assert audit["approved"] is False
            assert audit["stages"][-1] == stage
            assert audit["calls"][-1]["status"] == "cancelled"
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("reasoning_end", [106.0, 165.0])
async def test_stages_use_own_deadlines_and_reject_late_approval(monkeypatch, reasoning_end):
    from chrys.kernel import ChatResponse, Message

    clock = [100.0]
    budgets = []

    async def bounded(awaitable, timeout):
        budgets.append(timeout)
        return await asyncio.wait_for(awaitable, timeout)

    async def response(client, messages, *, stream, options, timeout):
        reasoning = messages[0].text == _SYSTEM_PROMPT
        clock[0] = reasoning_end if reasoning else 104.0
        result = {"approved": True, "reason": "late"} if reasoning else _values("unknown")
        return ChatResponse(messages=[Message("assistant", [json.dumps(result)])])

    fake_time = ModuleType("time")
    fake_time.monotonic = lambda: clock[0]
    fake_asyncio = ModuleType("asyncio")
    fake_asyncio.__dict__.update(vars(asyncio), wait_for=bounded)
    monkeypatch.setattr(judge_module, "time", fake_time)
    monkeypatch.setattr(judge_module, "asyncio", fake_asyncio)
    monkeypatch.setattr(
        judge_module, "get_final_response", create_autospec(judge_module.get_final_response, side_effect=response)
    )
    async with _judge(monkeypatch, []) as (judge, _):
        verdict = await _evaluate(judge)
        assert budgets == [5, 60]
        late = reasoning_end >= 164.0
        assert verdict.approved is not late
        assert verdict.audit["failure_reason"] == ("timeout" if late else None)
        if late:
            assert verdict.audit["calls"][-1]["error"] == "TimeoutError"


async def test_predicate_asset_is_loaded_once_per_judge(monkeypatch):
    load = create_autospec(judge_module.load_default_asset, wraps=judge_module.load_default_asset)
    load.return_value = load_default_asset()
    monkeypatch.setattr(judge_module, "load_default_asset", load)
    async with _judge(monkeypatch, [json.dumps(_values()), json.dumps(_values())]) as (judge, calls):
        assert (await _evaluate(judge)).approved is True
        assert (await _evaluate(judge)).approved is True
        assert len(calls) == 2
        load.assert_called_once_with()


async def test_concurrent_requests_keep_context_and_audits_separate(monkeypatch, tmp_path):
    from chrys.kernel import ChatResponse, Message

    entered, release = asyncio.Event(), asyncio.Event()

    async def response(client, messages, *, stream, options, timeout):
        if messages[0].text == _SYSTEM_PROMPT:
            assert "first.py" in messages[1].text and "second.py" not in messages[1].text
            result = {"approved": True, "reason": "first reviewed"}
        elif "first.py" in messages[1].text:
            entered.set()
            await release.wait()
            result = _values("unknown")
        else:
            result = _values() | {"external_action": True}
        return ChatResponse(messages=[Message("assistant", [json.dumps(result)])])

    monkeypatch.setattr(
        judge_module, "get_final_response", create_autospec(judge_module.get_final_response, side_effect=response)
    )
    async with _judge(monkeypatch, []) as (judge, _):
        args = {"path": "first.py"}
        first = asyncio.create_task(
            judge.evaluate(
                "Inspect first",
                "read_file",
                "filesystem.read",
                args,
                ["/first"],
                request_id="first",
                log_dir=tmp_path,
                formal=True,
            )
        )
        try:
            await wait_for(lambda: entered.is_set() or first.done())
            assert entered.is_set() and not first.done()
            args["path"] = "second.py"
            second = await judge.evaluate(
                "Push second", "shell", "shell", args, ["/second"], request_id="second", log_dir=tmp_path, formal=True
            )
            release.set()
            verdict = await first
            assert verdict.approved is True and second.approved is False
            for identity, expected_stages in (("first", ["predicate", "reasoning"]), ("second", ["predicate"])):
                audit = json.loads((tmp_path / f"{identity}.formal.jsonl").read_text())
                assert audit["request_id"] == identity
                assert audit["stages"] == expected_stages
                assert audit["application_attempts"] == len(expected_stages)
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)


async def test_cancelled_close_drains_both_clients_once(monkeypatch):
    async with _judge(monkeypatch, [json.dumps(_values("unknown")), '{"approved":true,"reason":"ok"}']) as (judge, _):
        assert (await _evaluate(judge)).approved is True
        entered, release = asyncio.Event(), asyncio.Event()
        client = judge._client
        reasoning_client = judge._reasoning_judge._client
        original_close = client.aclose

        async def close():
            entered.set()
            await release.wait()
            await original_close()

        close_spy = create_autospec(original_close, side_effect=close)
        monkeypatch.setattr(client, "aclose", close_spy)
        task = asyncio.create_task(judge.aclose())
        try:
            await wait_for(lambda: entered.is_set() or task.done())
            assert entered.is_set() and not task.done()
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await judge.aclose()
            assert close_spy.await_count == 1
            assert reasoning_client.sdk_client.is_closed()
            for stage_judge in (judge, judge._reasoning_judge):
                with pytest.raises(RuntimeError, match="closed"):
                    await stage_judge._get_client()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        '{"answers":{"external_action":{"type":"choice","choice":"true"},'
        + json.dumps(_reply(_values("unknown"), "typesafe/jev-test")["answers"])[1:]
        + "}",
    ],
)
async def test_jev_invalid_json_and_duplicate_ids_retry(monkeypatch, raw):
    replies = [httpx.Response(200, content=raw) for _ in range(3)]
    async with _judge(monkeypatch, replies, model="typesafe/jev-test") as (judge, calls):
        verdict = await _evaluate(judge)
        assert verdict.approved is False
        assert verdict.audit["failure_reason"] == "invalid_shape"
        assert verdict.audit["stages"] == ["predicate"]
        assert len(calls) == 3
