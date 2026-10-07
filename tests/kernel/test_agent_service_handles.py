# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Acceptance tests for service-conversation handle invalidation in the agent.

Pins what happens to the service handle (``conversation_id`` /
``previous_response_id`` / ``continuation_token`` / ``extra_body``) once the
tool loop invalidates it, and how the history fallback provider replays the
turns the service can no longer be trusted to hold.
"""

from __future__ import annotations

from typing import Any

from chrys.kernel import (
    Agent,
    AgentSession,
    ContextProvider,
    InMemoryHistoryProvider,
    ServiceFallbackHistoryProvider,
    SessionContext,
)
from chrys.kernel.types import ChatResponseUpdate
from tests.kernel._fakes import (
    _call_response,
    _call_update,
    _make_tool,
    _stack,
    _text_response,
    _text_update,
)


def _service_metadata_update(conversation_id: str, response_id: str | None = None) -> ChatResponseUpdate:
    """A content-free, metadata-only update carrying a service conversation handle."""
    return ChatResponseUpdate(contents=[], role="assistant", conversation_id=conversation_id, response_id=response_id)


class TestServiceHandleInvalidation:
    async def test_exhaustion_strip_invalidates_service_state_through_agent_path(self) -> None:
        # Exhaustion tail: the provider ignores tool_choice="none" and the
        # conversation id arrives on a separate metadata-only update. The
        # eager transform writes the session mid-stream, AgentResponse
        # restoration re-derives response_id from the mapped updates, and the
        # post-hook raw-scan would restore the session — the invalidation
        # must survive all three.
        metadata_update = _service_metadata_update("conv-svc", "resp-svc")
        layer, _wire = _stack(
            [
                [_call_update("c1", "echo", {"text": "a"})],
                [_call_update("c2", "echo", {"text": "b"}), metadata_update],
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])
        stream = agent.run("hi", stream=True, session=session, options={"store": True})
        [u async for u in stream]
        final = await stream.get_final_response()
        assert final.response_id is None
        assert final._chrys_service_state_invalidated is True
        assert session.service_session_id is None
        call_ids = [c.call_id for m in final.messages for c in m.contents if c.type == "function_call"]
        assert call_ids == ["c1"], "the stripped call must not reach the agent response"

    async def test_exhaustion_tail_abandonment_does_not_mirror_conversation_id(self) -> None:
        # A consumer that abandons the stream right after the exhaustion
        # tail's metadata update never reaches the tail verdict: entering the
        # service-stored tail must have cleared the previous round's mirrored
        # handle, and the eager transform must not have mirrored the tail's
        # own — either would leave the session pointing at a service
        # transcript whose stripped calls will never be answered.
        previous_round_update = _service_metadata_update("conv-old")
        metadata_update = _service_metadata_update("conv-svc", "resp-svc")
        layer, _wire = _stack(
            [
                [_call_update("c1", "echo", {"text": "a"}), previous_round_update],
                [_call_update("c2", "echo", {"text": "b"}), metadata_update],
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])
        stream = agent.run("hi", stream=True, session=session, options={"store": True})
        mirrored_mid_run = False
        async for update in stream:
            raw_conversation_id = getattr(update.raw_representation, "conversation_id", None)
            if raw_conversation_id == "conv-old":
                mirrored_mid_run = session.service_session_id == "conv-old"
            if raw_conversation_id == "conv-svc":
                break
        await stream.aclose()
        assert mirrored_mid_run, "the mid-run eager mirror must stay load-bearing"
        assert session.service_session_id is None

    async def test_last_iteration_result_abandonment_does_not_keep_stale_handle(self) -> None:
        # The synthesized tool-result update of the LAST iteration is the
        # final suspension point before the exhaustion tail. If the consumer
        # closes there, this batch's results are never posted to the service,
        # so the previously mirrored handle must already be gone by the time
        # that update is yielded.
        previous_round_update = _service_metadata_update("conv-old")
        layer, _wire = _stack(
            [
                [_call_update("c1", "echo", {"text": "a"}), previous_round_update],
                [_call_update("c2", "echo", {"text": "b"})],
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])
        stream = agent.run("hi", stream=True, session=session, options={"store": True})
        async for update in stream:
            raw = update.raw_representation
            if any(c.type == "function_result" for c in getattr(raw, "contents", None) or []):
                break
        await stream.aclose()
        assert session.service_session_id is None

    async def test_tool_limit_result_abandonment_does_not_keep_stale_handle(self) -> None:
        # A hit tool limit goes to the exhaustion tail too, so the batch that
        # hit it is the last one, well before max_iterations.
        previous_round_update = _service_metadata_update("conv-old")
        layer, _wire = _stack(
            [
                [_call_update("c1", "echo", {"text": "a"}), previous_round_update],
                [_call_update("c2", "echo", {"text": "b"})],
            ],
            max_iterations=5,
            max_function_calls=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])
        stream = agent.run("hi", stream=True, session=session, options={"store": True})
        async for update in stream:
            raw = update.raw_representation
            if any(c.type == "function_result" for c in getattr(raw, "contents", None) or []):
                break
        await stream.aclose()
        assert session.service_session_id is None

    async def test_invalidation_installs_history_fallback_preserving_next_run_context(self) -> None:
        # store=True suppresses the auto-injected plain local history
        # provider, so the discarded service transcript would otherwise be
        # the only history store. The invalidated run must leave the shadow
        # fallback in place with this run's input/output persisted, so the
        # next run replays them instead of sending only the new user message.
        metadata_update = _service_metadata_update("conv-svc", "resp-svc")
        layer, wire = _stack(
            [
                [_call_update("c1", "echo", {"text": "a"})],
                [_call_update("c2", "echo", {"text": "b"}), metadata_update],
                [_text_update("second answer")],
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        stream = agent.run("hi", stream=True, session=session, options={"store": True})
        [u async for u in stream]
        await stream.get_final_response()
        assert session.service_session_id is None
        assert [type(p) for p in agent.context_providers] == [ServiceFallbackHistoryProvider]

        stream = agent.run("and now?", stream=True, session=session, options={"store": True})
        [u async for u in stream]
        await stream.get_final_response()

        run2_request = wire.calls[2]
        messages = run2_request["messages"]
        assert messages[0].role == "user"
        assert messages[0].text == "hi"
        kinds = [c.type for m in messages for c in m.contents]
        assert "function_call" in kinds, "run-1 transcript must ride the request"
        assert "function_result" in kinds
        assert messages[-1].role == "user"
        assert messages[-1].text == "and now?"
        # No handle survived the invalidation to accompany the replay.
        assert "conversation_id" not in run2_request["options"]

    async def test_history_fallback_stops_replaying_once_new_handle_established(self) -> None:
        # Once a later run establishes a fresh service conversation (which
        # received the replayed history), the fallback must stop loading:
        # replaying alongside a live handle would duplicate the transcript
        # server-side.
        exhaustion_metadata = _service_metadata_update("conv-svc", "resp-svc")
        new_handle_metadata = _service_metadata_update("conv-new", "resp-new")
        layer, wire = _stack(
            [
                [_call_update("c1", "echo", {"text": "a"})],
                [_call_update("c2", "echo", {"text": "b"}), exhaustion_metadata],
                [_text_update("second answer"), new_handle_metadata],
                [_text_update("third answer")],
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        for prompt in ("hi", "and now?", "one more"):
            stream = agent.run(prompt, stream=True, session=session, options={"store": True})
            [u async for u in stream]
            await stream.get_final_response()

        assert session.service_session_id == "conv-new"
        run3_request = wire.calls[3]
        assert [m.text for m in run3_request["messages"]] == ["one more"]
        assert run3_request["options"].get("conversation_id") == "conv-new"
        # Still exactly one fallback: the run-2/run-3 prepare passes must not
        # stack additional providers.
        assert [type(p) for p in agent.context_providers] == [ServiceFallbackHistoryProvider]

    async def test_service_stored_turns_before_invalidation_survive_via_eager_shadow(self) -> None:
        # Turn 1 succeeds purely through service storage; turn 2's exhaustion
        # discards the handle. A fallback installed only at invalidation time
        # could never recover turn 1 — the shadow must be in place from the
        # first service-stored run so every turn is persisted locally.
        first_metadata = _service_metadata_update("conv-svc", "resp-1")
        exhaustion_metadata = _service_metadata_update("conv-svc", "resp-2")
        layer, wire = _stack(
            [
                [_text_update("first answer"), first_metadata],
                [_call_update("c1", "echo", {"text": "a"})],
                [_call_update("c2", "echo", {"text": "b"}), exhaustion_metadata],
                [_text_update("third answer")],
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        stream = agent.run("hi", stream=True, session=session, options={"store": True})
        [u async for u in stream]
        await stream.get_final_response()
        # The shadow provider exists before any invalidation happened.
        assert [type(p) for p in agent.context_providers] == [ServiceFallbackHistoryProvider]
        assert session.service_session_id == "conv-svc"

        stream = agent.run("and now?", stream=True, session=session, options={"store": True})
        [u async for u in stream]
        await stream.get_final_response()
        assert session.service_session_id is None
        # While the handle was live the shadow stayed silent on the wire.
        run2_request = wire.calls[1]
        assert [m.text for m in run2_request["messages"]] == ["and now?"]
        assert run2_request["options"].get("conversation_id") == "conv-svc"

        stream = agent.run("one more", stream=True, session=session, options={"store": True})
        [u async for u in stream]
        await stream.get_final_response()
        run3_request = wire.calls[3]
        texts = [c.text for m in run3_request["messages"] for c in m.contents if c.type == "text"]
        assert "hi" in texts, "turn-1 input must survive the invalidation"
        assert "first answer" in texts, "turn-1 answer must survive the invalidation"
        assert "and now?" in texts
        assert texts[-1] == "one more"
        assert "conversation_id" not in run3_request["options"]

    async def test_invalidated_explicit_conversation_id_suppressed_on_next_run(self) -> None:
        # A caller-supplied conversation_id skips the eager shadow and rides
        # every run via the options spread. Once the exhaustion tail withheld
        # that handle, repeating the option must not re-send it: the service
        # transcript behind it still holds stripped unanswered calls, and it
        # would arrive combined with the local fallback replay. A different,
        # never-invalidated explicit handle must still ride.
        exhaustion_metadata = _service_metadata_update("conv-ext", "resp-svc")
        layer, wire = _stack(
            [
                [_call_update("c1", "echo", {"text": "a"})],
                [_call_update("c2", "echo", {"text": "b"}), exhaustion_metadata],
                [_text_update("second answer")],
                [_text_update("third answer")],
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        stream = agent.run("hi", stream=True, session=session, options={"store": True, "conversation_id": "conv-ext"})
        [u async for u in stream]
        await stream.get_final_response()
        assert wire.calls[0]["options"].get("conversation_id") == "conv-ext"
        assert session.service_session_id is None
        assert "conv-ext" in session.invalidated_service_session_ids

        stream = agent.run(
            "and now?", stream=True, session=session, options={"store": True, "conversation_id": "conv-ext"}
        )
        [u async for u in stream]
        await stream.get_final_response()
        run2_request = wire.calls[2]
        assert "conversation_id" not in run2_request["options"], "the withheld handle must not ride again"
        texts = [c.text for m in run2_request["messages"] for c in m.contents if c.type == "text"]
        assert "hi" in texts, "local fallback replay owns continuity"
        assert texts[-1] == "and now?"

        stream = agent.run(
            "one more", stream=True, session=session, options={"store": True, "conversation_id": "conv-other"}
        )
        [u async for u in stream]
        await stream.get_final_response()
        assert wire.calls[3]["options"].get("conversation_id") == "conv-other", "suppression is per-handle"
        assert [m.text for m in wire.calls[3]["messages"]] == ["one more"], (
            "the fallback replay must not ride along and contaminate the other live conversation"
        )

    async def test_repeated_invalidated_handle_falls_back_to_newer_live_handle(self) -> None:
        # Once the replay run established a fresh service conversation, the
        # fallback provider stays silent on seeing the live handle. Repeating
        # the old invalidated option then must ride that live handle instead
        # of being bare-removed — otherwise the request carries neither a
        # handle nor local history, only the new user message.
        exhaustion_metadata = _service_metadata_update("conv-old", "resp-1")
        new_handle_metadata = _service_metadata_update("conv-new", "resp-2")
        layer, wire = _stack(
            [
                [_call_update("c1", "echo", {"text": "a"})],
                [_call_update("c2", "echo", {"text": "b"}), exhaustion_metadata],
                [_text_update("second answer"), new_handle_metadata],
                [_text_update("third answer")],
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        for prompt in ("hi", "and now?", "one more"):
            stream = agent.run(
                prompt, stream=True, session=session, options={"store": True, "conversation_id": "conv-old"}
            )
            [u async for u in stream]
            await stream.get_final_response()

        assert session.service_session_id == "conv-new"
        run3_request = wire.calls[3]
        assert run3_request["options"].get("conversation_id") == "conv-new", "live handle must replace the stale one"
        assert [m.text for m in run3_request["messages"]] == ["one more"], "no replay may ride the live handle"

    async def test_suppressed_handle_unlocks_eager_shadow_for_fresh_agent(self) -> None:
        # A fresh agent joining a session whose handle was already withheld
        # has no fallback provider yet; the suppressed explicit handle must
        # not also block the eager shadow install, or the new agent's turns
        # would go unshadowed until its own first invalidation.
        layer, wire = _stack([[_text_update("answer")]], max_iterations=1)
        session = AgentSession()
        session.invalidated_service_session_ids.add("conv-ext")
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        stream = agent.run("hi", stream=True, session=session, options={"store": True, "conversation_id": "conv-ext"})
        [u async for u in stream]
        await stream.get_final_response()
        assert [type(p) for p in agent.context_providers] == [ServiceFallbackHistoryProvider]
        assert "conversation_id" not in wire.calls[0]["options"]

    async def test_suppressed_native_spelling_unlocks_eager_shadow_for_fresh_agent(self) -> None:
        # The eager-shadow decision runs on the same normalized handle view
        # as the wire choke point: an invalidated previous_response_id counts
        # as absent there too, and is withheld from the wire.
        layer, wire = _stack([[_text_update("answer")]], max_iterations=1)
        session = AgentSession()
        session.invalidated_service_session_ids.add("resp-old")
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        stream = agent.run(
            "hi", stream=True, session=session, options={"store": True, "previous_response_id": "resp-old"}
        )
        [u async for u in stream]
        await stream.get_final_response()
        assert [type(p) for p in agent.context_providers] == [ServiceFallbackHistoryProvider]
        assert "previous_response_id" not in wire.calls[0]["options"]

    async def test_blocking_invalidation_suppresses_explicit_handle(self) -> None:
        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "echo", {"text": "b"}), conversation_id="conv-ext", response_id="resp-svc"),
                _text_response("second answer"),
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        first = await agent.run("hi", session=session, options={"store": True, "conversation_id": "conv-ext"})
        assert first._chrys_service_state_invalidated is True
        assert "conv-ext" in session.invalidated_service_session_ids

        await agent.run("and now?", session=session, options={"store": True, "conversation_id": "conv-ext"})
        run2_request = wire.calls[2]
        assert "conversation_id" not in run2_request["options"], "the withheld handle must not ride again"
        assert run2_request["messages"][0].text == "hi", "local fallback replay owns continuity"

    async def test_invalidated_previous_response_id_suppressed_on_next_run(self) -> None:
        # The withheld continuation state includes the stripped response's
        # own id — a stateful provider accepts it back as
        # previous_response_id, so repeating it under that spelling must be
        # suppressed exactly like the conversation_id spelling.
        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "echo", {"text": "b"}), conversation_id="conv-svc", response_id="resp-svc"),
                _text_response("second answer"),
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        first = await agent.run("hi", session=session, options={"store": True})
        assert first._chrys_service_state_invalidated is True
        assert "resp-svc" in session.invalidated_service_session_ids

        await agent.run("and now?", session=session, options={"store": True, "previous_response_id": "resp-svc"})
        run2_request = wire.calls[2]
        assert "previous_response_id" not in run2_request["options"], "the withheld response id must not ride again"
        assert run2_request["messages"][0].text == "hi", "local fallback replay owns continuity"

    async def test_invalidated_continuation_token_suppressed_on_next_run(self) -> None:
        # A reused token would short-circuit the Responses client into
        # retrieving the poisoned completed response outright — resurfacing
        # the stripped call the invalidation withheld — while the request
        # messages are ignored.
        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "echo", {"text": "b"}), conversation_id="conv-svc", response_id="resp-svc"),
                _text_response("second answer"),
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        first = await agent.run("hi", session=session, options={"store": True})
        assert first._chrys_service_state_invalidated is True
        assert "resp-svc" in session.invalidated_service_session_ids

        await agent.run(
            "and now?",
            session=session,
            options={"store": True, "continuation_token": {"response_id": "resp-svc"}},
        )
        run2_request = wire.calls[2]
        assert "continuation_token" not in run2_request["options"], "the poisoned token must not ride again"
        assert run2_request["messages"][0].text == "hi", "local fallback replay owns continuity"

    async def test_valid_continuation_token_rides_and_gates_fallback_replay(self) -> None:
        # A live token is continuation state like any other handle: it rides
        # untouched, and the fallback replay must not load under it — the
        # retrieve short-circuit ignores request messages anyway.
        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "echo", {"text": "b"}), conversation_id="conv-svc", response_id="resp-svc"),
                _text_response("second answer"),
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        await agent.run("hi", session=session, options={"store": True})
        await agent.run(
            "and now?",
            session=session,
            options={"store": True, "continuation_token": {"response_id": "resp-live"}},
        )
        run2_request = wire.calls[2]
        assert run2_request["options"].get("continuation_token") == {"response_id": "resp-live"}
        assert [m.text for m in run2_request["messages"]] == ["and now?"], (
            "the fallback replay must not ride alongside a live token"
        )

    async def test_invalidated_conversation_mapping_form_suppressed(self) -> None:
        # The ``conversation`` spelling admits a mapping form carrying the id
        # under "id"; a poisoned one is dropped whole (removal-only — never
        # rewritten) and the caller-owned mapping stays untouched.
        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "echo", {"text": "b"}), conversation_id="conv-svc", response_id="resp-svc"),
                _text_response("second answer"),
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        await agent.run("hi", session=session, options={"store": True})
        conversation = {"id": "conv-svc"}
        await agent.run("and now?", session=session, options={"store": True, "conversation": conversation})
        run2_request = wire.calls[2]
        assert "conversation" not in run2_request["options"], "the poisoned mapping spelling must be dropped whole"
        assert conversation == {"id": "conv-svc"}, "the caller-owned mapping must stay untouched"
        assert run2_request["messages"][0].text == "hi", "local fallback replay owns continuity"

    async def test_invalidated_extra_body_handle_removed_copy_on_write(self) -> None:
        # Provider SDKs merge extra_body over the named parameters, so a
        # nested handle spelling reaches the wire all the same. A poisoned
        # nested key is removed from a copy — sibling keys survive and the
        # caller-owned mapping stays untouched.
        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "echo", {"text": "b"}), conversation_id="conv-svc", response_id="resp-svc"),
                _text_response("second answer"),
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        await agent.run("hi", session=session, options={"store": True})
        extra_body = {"previous_response_id": "resp-svc", "keep": "k"}
        await agent.run("and now?", session=session, options={"store": True, "extra_body": extra_body})
        run2_request = wire.calls[2]
        assert run2_request["options"]["extra_body"] == {"keep": "k"}
        assert extra_body == {"previous_response_id": "resp-svc", "keep": "k"}, (
            "the caller-owned extra_body must stay untouched"
        )
        assert run2_request["messages"][0].text == "hi", "local fallback replay owns continuity"

    async def test_agent_default_handle_gates_fallback_replay(self) -> None:
        # Agent-default continuation handles ride the wire through the
        # options merge exactly like run-level ones, so the fallback load
        # gate must see the same merged view: a live default handle means
        # the service owns the conversation and the local replay must not
        # ride alongside it.
        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "echo", {"text": "b"}), conversation_id="conv-svc", response_id="resp-svc"),
                _text_response("second answer"),
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])
        # The ctor exposes no default_options parameter; the attribute is a
        # plain dict re-read every run.
        agent.default_options["store"] = True
        agent.default_options["previous_response_id"] = "resp-live"

        first = await agent.run("hi", session=session)
        assert first._chrys_service_state_invalidated is True
        assert [type(p) for p in agent.context_providers] == [ServiceFallbackHistoryProvider]

        await agent.run("and now?", session=session)
        run2_request = wire.calls[2]
        assert run2_request["options"].get("previous_response_id") == "resp-live"
        assert [m.text for m in run2_request["messages"]] == ["and now?"], (
            "the fallback replay must not ride alongside the live default handle"
        )

    async def test_history_providers_see_sanitized_options_view(self) -> None:
        # Every history provider reflects on the provider-facing options to
        # decide whether the service owns this run's history — not only the
        # kernel fallback provider with its own invalidation check. An
        # invalidated handle must therefore already be gone from that view:
        # a provider that trusted it would skip local replay while the wire
        # choke point strips the handle, sending the request with neither
        # remote nor local history. Live options survive untouched.
        class _OptionsProbe(ContextProvider):
            def __init__(self) -> None:
                super().__init__("probe")
                self.seen_options: list[dict[str, Any]] = []

            async def before_run(
                self, *, agent: Any, session: AgentSession, context: SessionContext, state: dict
            ) -> None:
                self.seen_options.append(dict(context.options))

        layer, _wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "echo", {"text": "b"}), conversation_id="conv-svc", response_id="resp-svc"),
                _text_response("second answer"),
                _text_response("third answer"),
            ],
            max_iterations=1,
        )
        session = AgentSession()
        probe = _OptionsProbe()
        agent = Agent(client=layer, name="T", tools=[_make_tool()], context_providers=[probe])

        first = await agent.run("hi", session=session, options={"store": True})
        assert first._chrys_service_state_invalidated is True
        assert "resp-svc" in session.invalidated_service_session_ids

        await agent.run(
            "and now?",
            session=session,
            options={"store": True, "previous_response_id": "resp-svc", "metadata": {"k": "v"}},
        )
        run2_options = probe.seen_options[-1]
        assert "previous_response_id" not in run2_options, "providers must not see the withheld handle as live"
        assert run2_options.get("store") is True
        assert run2_options.get("metadata") == {"k": "v"}

        await agent.run(
            "third",
            session=session,
            options={"store": True, "previous_response_id": "resp-live"},
        )
        assert probe.seen_options[-1].get("previous_response_id") == "resp-live", "live handles stay visible"

    async def test_blocking_invalidation_installs_history_fallback(self) -> None:
        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "echo", {"text": "b"}), conversation_id="conv-svc", response_id="resp-svc"),
                _text_response("second answer"),
            ],
            max_iterations=1,
        )
        session = AgentSession()
        agent = Agent(client=layer, name="T", tools=[_make_tool()])

        first = await agent.run("hi", session=session, options={"store": True})
        assert first._chrys_service_state_invalidated is True
        assert session.service_session_id is None
        assert [type(p) for p in agent.context_providers] == [ServiceFallbackHistoryProvider]

        await agent.run("and now?", session=session, options={"store": True})
        messages = wire.calls[2]["messages"]
        assert messages[0].role == "user"
        assert messages[0].text == "hi"
        kinds = [c.type for m in messages for c in m.contents]
        assert "function_call" in kinds
        assert "function_result" in kinds
        assert messages[-1].text == "and now?"

    async def test_invalidation_keeps_existing_loading_history_provider(self) -> None:
        # A session that already has a loading HistoryProvider keeps it as
        # the sole history owner: it persisted the turn through the standard
        # after_run pass, so installing the fallback would double history.
        metadata_update = _service_metadata_update("conv-svc", "resp-svc")
        layer, _wire = _stack(
            [
                [_call_update("c1", "echo", {"text": "a"})],
                [_call_update("c2", "echo", {"text": "b"}), metadata_update],
            ],
            max_iterations=1,
        )
        session = AgentSession()
        provider = InMemoryHistoryProvider()
        agent = Agent(client=layer, name="T", tools=[_make_tool()], context_providers=[provider])
        stream = agent.run("hi", stream=True, session=session, options={"store": True})
        [u async for u in stream]
        await stream.get_final_response()
        assert agent.context_providers == [provider]
