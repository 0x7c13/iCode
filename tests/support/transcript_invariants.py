# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Structural invariants of a kernel tool-loop transcript.

``assert_transcript_invariants`` is the guard oracle for the kernel loop's
final responses. It states, as one mechanical check, what the loop's landing
normalization (``_land_response_contents``), dispatch filter, and result
folding jointly guarantee about the transcript they build — independent of
HOW any particular test drove the loop. ``InvariantCheckedToolLoopLayer``
(below) is how tests apply it: a drop-in ``ToolLoopLayer`` that checks every
final response, blocking and streaming. A hygiene rule in
``tests/architecture/test_hygiene_test_source_rules.py`` bans bare ``ToolLoopLayer``
construction in tests outside a small allowlist, so oracle coverage is
enforced mechanically rather than by convention. A fuzzing harness in
``tests/kernel/test_loop_echo_normalization.py`` additionally drives adversarial (echoing,
id-reusing, object-duplicating) clients against it, and
``tests/kernel/test_transcript_invariants.py`` pins the failure paths of the
oracle itself so no sub-check can silently rot into a no-op.

Why this exists: the echo-normalization work showed that counterexample-shaped
tests only pin the failure shapes someone has already imagined (a dangling
duplicate call, a twice-recorded result, a call wiped out of a shared message
by in-place mutation, a stale ordinal collision). Each of those historical
defects violates one of the invariants below, so a single checker applied
broadly catches the *next* unimagined shape too. When extending loop logic,
run the whole loop suite — if this checker fires, the transcript contract is
broken even if every targeted assertion still passes.

The invariants deliberately do NOT include "every call is answered": the loop
legitimately returns unanswered calls (a no-tools batch, the max-function-call
cap, duplicate-id calls dropped by dispatch, the tool_choice="none" tail), so
that property belongs to individual tests that know their scenario.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from chrys.foundation.tool_invocation_order import read_tool_invocation_order
from chrys.foundation.trajectory.context import TRAJECTORY_CONTEXT_KWARG, TrajectoryContext
from chrys.foundation.trajectory.envelope import EventDraft
from chrys.foundation.trajectory.metadata import read_operation_id
from chrys.kernel.loop import ToolLoopLayer
from chrys.kernel.types import ChatResponse, Content, Message
from tests.support.trajectory_invariants import assert_trajectory_operation_settlement

__all__ = ["InvariantCheckedToolLoopLayer", "assert_transcript_invariants"]


@runtime_checkable
class _DraftRecordingSink(Protocol):
    """Test sink surface needed by the operation-settlement oracle."""

    drafts: list[EventDraft]


def _recording_scope(kwargs: Mapping[str, Any]) -> tuple[_DraftRecordingSink, int] | None:
    client_kwargs = kwargs.get("client_kwargs")
    if not isinstance(client_kwargs, Mapping):
        return None
    context = client_kwargs.get(TRAJECTORY_CONTEXT_KWARG)
    if not isinstance(context, TrajectoryContext) or not isinstance(context.sink, _DraftRecordingSink):
        return None
    return context.sink, len(context.sink.drafts)


# LITERAL, test-owned copies of the production content-type sets: the oracle
# must not inherit a production edit silently, so it never imports them. A
# parity pin in tests/kernel/test_exchanges.py compares these against
# ``chrys.kernel.exchanges`` — any divergence is a conscious, visible change.
TOOL_CALL_CONTENT_TYPES = frozenset(
    {
        "function_call",
        "hosted_tool_call",
        "code_interpreter_tool_call",
        "image_generation_tool_call",
        "mcp_server_tool_call",
        "search_tool_call",
        "shell_tool_call",
    }
)
TOOL_RESULT_CONTENT_TYPES = frozenset(
    {
        "function_result",
        "hosted_tool_result",
        "code_interpreter_tool_result",
        "image_generation_tool_result",
        "mcp_server_tool_result",
        "search_tool_result",
        "shell_tool_result",
    }
)

# LITERAL, test-owned copy of ``HistoryMarkerKind.KEY`` for the same
# independence reason as the type sets above.
_MARKER_KEY = "_chrys_kind"


def _is_marker(message: Message) -> bool:
    # Key presence, not value truthiness — a falsy marker kind still bounds
    # the exchange.
    return _MARKER_KEY in message.additional_properties


def _is_actionable_function_call(content: Content) -> bool:
    # Mirror of the kernel loop's predicate: informational (hosted) calls are
    # never dispatched and never stamped, so the invariants skip them too.
    return content.type == "function_call" and not content.informational_only


def assert_transcript_invariants(response: ChatResponse) -> None:
    """Assert the structural contract of a kernel-loop final transcript.

    I1 — ordinal integrity: the stamped ``_chrys_tool_invocation_order``
    values over actionable function calls, read in transcript order, form
    one consecutive strictly-increasing run (``k, k+1, …``). The kernel loop
    is the sole producer and stamps at response landing in emission order, so
    a gap, duplicate, or inversion means a stale foreign stamp survived, an
    echo was re-numbered, or two numbering schemes diverged again — the
    exact defect class that mis-attached result metadata across tools. The
    run may start above zero: the middleware-termination exit returns the
    current response WITHOUT the accumulated-transcript prepend, so its
    window legitimately begins mid-run.

    I2 — object identity uniqueness: no non-usage Content and no Message
    object appears twice. Landing removes echoed conversation objects (a
    client re-sending contents or whole messages the run already landed);
    usage is request-scoped accounting and may legitimately reuse one object
    across calls. Any other identity duplicate in the final transcript means
    an echo escaped normalization — the shape that produced dangling
    duplicate calls and double-recorded results.

    I3 — exchange-scoped result pairing: within each exchange (a run of
    assistant messages plus the tool output that follows), every id-bearing
    result — of ANY result content type, hosted MCP included — answers a call
    of that exchange, and no id is answered twice. The pairing id is
    type-aware (``_pairing_key``): image generation pairs on ``image_id``
    (its ``call_id`` is always None), everything else on ``call_id``, and the
    two namespaces never collide. Tool output rides
    tool-role messages OR result-only assistant messages — hosted MCP results
    arrive assistant-role (see ``_is_tool_output_continuation``) — and both
    placements are held to the same pairing rule; results embedded in a
    call-carrying assistant message pair against that (new) exchange, whose
    calls register first, so hosted same-message pairs pass while embedded
    orphans fire. An orphaned result means a call was lost (e.g. a shared
    message mutated in place); a doubly-answered id means an echoed result
    survived. Pairing is scoped to the exchange because providers
    legitimately reuse call ids across iterations — a global check would
    reject valid ``call_00``-style transcripts. Results without their pairing
    id are exempt: they are unpairable by construction (the kernel copies the
    call's empty id onto its result) and are covered by I2 instead.
    """
    messages = list(response.messages)
    _assert_ordinal_integrity(messages)
    _assert_object_identity_uniqueness(messages)
    _assert_exchange_scoped_result_pairing(messages)


def _assert_ordinal_integrity(messages: Sequence[Message]) -> None:
    ordinals: list[int] = []
    for message in messages:
        for content in message.contents:
            if not _is_actionable_function_call(content):
                continue
            stamped = read_tool_invocation_order(content.additional_properties)
            assert stamped is not None, (
                f"I1: actionable call {content.call_id!r} ({content.name!r}) has no invocation-order stamp; "
                "the kernel loop stamps every actionable call at landing"
            )
            ordinals.append(stamped)
    # One consecutive run, not necessarily 0-based: the middleware-termination
    # exit returns the current response without the accumulated prepend, so a
    # final transcript can be a trailing window of the run's numbering.
    expected = list(range(ordinals[0], ordinals[0] + len(ordinals))) if ordinals else []
    assert ordinals == expected, (
        f"I1: invocation orders must form one consecutive increasing run in transcript order, got {ordinals}; "
        "a gap/duplicate/inversion means stale stamps or echo re-numbering leaked through"
    )


def _assert_object_identity_uniqueness(messages: Sequence[Message]) -> None:
    seen_message_ids: set[int] = set()
    seen_content_ids: set[int] = set()
    for message in messages:
        assert id(message) not in seen_message_ids, (
            f"I2: Message object ({message.role!r}) appears twice in the transcript; "
            "an echoed message escaped landing normalization"
        )
        seen_message_ids.add(id(message))
        for content in message.contents:
            if content.type == "usage":
                continue
            assert id(content) not in seen_content_ids, (
                f"I2: Content object ({content.type!r}, call_id={content.call_id!r}) "
                "appears twice in the transcript; an echoed content escaped landing normalization"
            )
            seen_content_ids.add(id(content))


def _is_tool_output_continuation(message: Message) -> bool:
    """A message carrying an exchange's tool output without starting new work.

    Mirror of the shared exchange grammar's role-gated result-only rule
    (``is_result_only_message`` in ``chrys.kernel.exchanges``): tool output
    rides tool-role messages or result-only assistant messages — hosted MCP
    tools report their results in assistant-role messages. Such a message
    CONTINUES the current exchange's output run; only an assistant message
    carrying new calls (or a non-assistant, non-tool role) closes the
    exchange. Marker handling lives in the scan loop, which checks markers
    before this predicate ever sees one.
    """
    if message.role == "tool":
        return True
    if message.role != "assistant":
        return False
    return any(content.type in TOOL_RESULT_CONTENT_TYPES for content in message.contents) and not any(
        content.type in TOOL_CALL_CONTENT_TYPES for content in message.contents
    )


def _pairing_key(content: Content) -> tuple[str, str] | None:
    """Type-aware pairing id of a call/result content, or None if unpairable.

    Image generation pairs on ``image_id`` — its ``call_id`` is always None —
    while every other call/result type pairs on ``call_id``. The field name
    is part of the key so ids never collide across the two namespaces. A
    falsy id yields None: such contents are unpairable by construction and
    covered by I2 instead.
    """
    if content.type in ("image_generation_tool_call", "image_generation_tool_result"):
        return ("image_id", content.image_id) if content.image_id else None
    return ("call_id", content.call_id) if content.call_id else None


def _assert_exchange_scoped_result_pairing(messages: Sequence[Message]) -> None:
    # An exchange is a maximal run of assistant messages followed by the tool
    # output answering them (tool-role or result-only assistant messages, see
    # _is_tool_output_continuation); the next call-carrying assistant (or
    # other-role) message after tool output starts a new exchange. Pairing
    # spans ALL call/result content types of the shared sets, not just
    # function_call/function_result — hosted MCP results obey the same
    # one-answer-per-id rule, and image generation pairs on its own field
    # (see _pairing_key). Duplicate pairing ids WITHIN one exchange's calls
    # are legal (dispatch executes one and keeps both contents, mirroring the
    # upstream framework), so only results are required to be unique per id.
    # A call-carrying assistant message registers its calls BEFORE its
    # embedded results are checked, so a hosted same-message pair passes as
    # self-paired while an embedded orphan (an id with no call in the — new —
    # exchange) still fires.
    exchange_call_keys: set[tuple[str, str]] = set()
    answered_call_keys: set[tuple[str, str]] = set()
    in_output_run = False

    def _check_results(message: Message) -> None:
        for content in message.contents:
            if content.type not in TOOL_RESULT_CONTENT_TYPES:
                continue
            key = _pairing_key(content)
            if key is None:
                continue
            assert key in exchange_call_keys, (
                f"I3: result {content.type!r} with {key[0]}={key[1]!r} answers no call of its exchange; "
                "its call was lost (e.g. a shared message was mutated) or the result was misplaced"
            )
            assert key not in answered_call_keys, (
                f"I3: {key[0]}={key[1]!r} is answered twice within one exchange ({content.type!r}); "
                "an echoed result escaped landing normalization"
            )
            answered_call_keys.add(key)

    for message in messages:
        if _is_marker(message):
            # A history marker is exchange chrome, never a member: it closes
            # the exchange in both walks. Its own result contents are checked
            # against the cleared state, so a result riding a marker always
            # fires — the kernel never emits markers inside a response.
            exchange_call_keys.clear()
            answered_call_keys.clear()
            in_output_run = False
            _check_results(message)
        elif _is_tool_output_continuation(message):
            in_output_run = True
            _check_results(message)
        elif message.role == "assistant":
            if in_output_run:
                exchange_call_keys.clear()
                answered_call_keys.clear()
                in_output_run = False
            for content in message.contents:
                if content.type in TOOL_CALL_CONTENT_TYPES:
                    key = _pairing_key(content)
                    if key is not None:
                        exchange_call_keys.add(key)
            _check_results(message)
        else:
            exchange_call_keys.clear()
            answered_call_keys.clear()
            in_output_run = False


class InvariantCheckedToolLoopLayer(ToolLoopLayer):
    """ToolLoopLayer that runs the transcript-invariant oracle on every final response.

    Tests build this instead of the raw layer so every loop-driving test —
    present and future — mechanically verifies the structural transcript
    contract in addition to its own targeted assertions. A hygiene rule in
    ``tests/architecture/test_hygiene_test_source_rules.py`` bans bare ``ToolLoopLayer``
    construction in tests outside an explicit allowlist, so the coverage
    claim holds mechanically rather than by convention. Do not switch a test
    back to the raw ``ToolLoopLayer`` to silence a failure here: a firing
    invariant means the landing/dispatch/fold contract is broken even if the
    test's own assertions still pass.
    """

    def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
        recording = _recording_scope(kwargs)

        def _assert_operation_settlement(response: ChatResponse) -> None:
            if recording is None:
                return
            sink, start = recording
            answered_call_ids = {
                content.call_id
                for message in response.messages
                for content in message.contents
                if content.type == "function_result" and content.call_id
            }
            # Only the calls the loop still owed a terminal for. An executed
            # call's operation is opened and closed by the tool-events
            # middleware, which a kernel-only stack does not install, so
            # expecting it here would assert on a layer the test never built.
            # The full-stack version of this check runs in the trajectory
            # integration suite, where that middleware is present.
            expected_tool_operations = {
                operation_id
                for message in response.messages
                for content in message.contents
                if _is_actionable_function_call(content) and content.call_id not in answered_call_ids
                if (operation_id := read_operation_id(content.additional_properties)) is not None
            }
            assert_trajectory_operation_settlement(
                sink.drafts[start:],
                expected_tool_operation_ids=expected_tool_operations,
            )

        if not stream:
            pending = super().get_response(messages, stream=False, **kwargs)

            async def _checked() -> ChatResponse:
                response = await pending
                assert_transcript_invariants(response)
                _assert_operation_settlement(response)
                return response

            return _checked()

        response_stream = super().get_response(messages, stream=True, **kwargs)
        inner_final = response_stream.get_final_response

        async def _checked_final() -> ChatResponse:
            response = await inner_final()
            assert_transcript_invariants(response)
            _assert_operation_settlement(response)
            return response

        response_stream.get_final_response = _checked_final
        return response_stream
