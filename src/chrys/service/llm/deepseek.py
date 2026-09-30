# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""DeepSeek OpenAI-compatible Chat Completions and Responses clients.

DeepSeek's API is OpenAI-compatible at the wire level, so both clients
subclass Chrys' owned OpenAI wire clients instead of writing a transport
from scratch.

``DeepSeekChatCompletionClient`` subclasses
``RawOpenAIChatCompletionClient``. The base client owns reasoning capture
and replay for the ``reasoning_content`` / ``reasoning_details`` dialects,
plus vLLM's fallback ``reasoning`` field; the subclass keeps only
DeepSeek-specific policy:

- ``TOKEN_LIMIT_PARAM``: DeepSeek's Chat Completions API uses the
  OpenAI-legacy ``max_tokens`` field, not ``max_completion_tokens``
  (https://api-docs.deepseek.com/api/create-chat-completion).
- ``_should_replay_reasoning``: DeepSeek thinking mode emits an extra
  ``reasoning_content`` field on each assistant message.  Its docs
  (https://api-docs.deepseek.com/guides/thinking_mode) say a request
  without ``tools`` ignores the field, while one that sends ``tools``
  must replay every historical ``reasoning_content`` — turns without a
  tool call included — or get HTTP 400.  The live API has been seen
  accepting such requests anyway; the client follows the documented
  contract, not observed rejections.  A request after a tool interaction
  replays too.  An assistant message with no reasoning of its own
  (another model's history) then goes out with an empty
  ``reasoning_content``.
- A strict message assembler: DeepSeek's schema wants consecutive
  same-role fragments merged into one wire message, plain-string
  content, and an explicit ``content: ""`` on assistant messages that
  carry ``tool_calls``.
- ``_parse_usage_from_openai``: surfaces DeepSeek's non-standard
  ``prompt_cache_hit_tokens`` usage field.

``DeepSeekResponsesClient`` subclasses ``RawOpenAIChatClient`` (the
Responses wire client).  DeepSeek's Responses endpoint is a stateless
chat-mapping compatibility layer: ``store`` / ``previous_response_id`` /
``conversation`` are unsupported, reasoning is plaintext (no
``encrypted_content``), and every response reports ``store: false``.
The subclass forces client-side statelessness (``FORCES_STATELESS``),
replays reasoning as plaintext ``reasoning_text`` items per
``REASONING_REPLAY_MODE``, strips conversation handles from the wire,
rejects ``continuation_token`` / ``background``, and never learns
service conversation ids.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Any, ClassVar, Literal, cast

from chrys.kernel.exceptions import ChatClientInvalidRequestException
from chrys.kernel.types import Content, Message
from chrys.service.llm.openai_chat_completion import (
    _PENDING_IMAGE_PARTS_KEY,
    RawOpenAIChatCompletionClient,
    _sanitize_author_name,
)
from chrys.service.llm.openai_chat_completion import (
    REASONING_CONTENT_FIELD as REASONING_CONTENT_FIELD,
)
from chrys.service.llm.openai_chat_completion import (
    REASONING_DETAILS_FIELD as REASONING_DETAILS_FIELD,
)
from chrys.service.llm.openai_chat_completion import (
    REASONING_FIELD as REASONING_FIELD,
)
from chrys.service.llm.openai_chat_completion import (
    REASONING_FORMAT_KEY as REASONING_FORMAT_KEY,
)
from chrys.service.llm.openai_responses import RawOpenAIChatClient
from chrys.service.llm.token_limit_params import CHAT_COMPLETIONS_TOKEN_LIMIT_PARAMS

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DeepSeekResponsesReasoningReplayMode = Literal["encrypted", "plaintext-replay", "plaintext-valid-drop"]


def _add_deepseek_cache_usage(details: Any, usage: Any) -> Any:
    """Add DeepSeek's top-level cache-hit count to normalized usage."""
    if details is None:
        return None
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    if hit is None:
        extra = getattr(usage, "model_extra", None) or {}
        hit = extra.get("prompt_cache_hit_tokens")
    if hit is not None:
        details["deepseek.prompt_cache_hit_tokens"] = int(hit)
    return details


class DeepSeekChatCompletionClient(RawOpenAIChatCompletionClient):
    """OpenAI-compatible Chat Completions client for DeepSeek thinking mode.

    Subclasses the loop-free ``Raw`` wire client: every override here is
    a ``_prepare_*``/``_parse_*`` wire hook defined on the Raw class; the tool
    loop and chat middleware are chrys-owned layers composed around it by
    ``chrys.service.llm.instrumented``.
    """

    TOKEN_LIMIT_PARAM: ClassVar[str] = CHAT_COMPLETIONS_TOKEN_LIMIT_PARAMS["deepseek-openai"]

    def _parse_usage_from_openai(self, usage: Any) -> Any:
        """Inject DeepSeek's non-standard ``prompt_cache_hit_tokens`` field.

        DeepSeek reports cache reads as top-level ``prompt_cache_hit_tokens``
        on ``usage`` instead of OpenAI's ``prompt_tokens_details.cached_tokens``,
        so the base extractor drops it. We stash it under a namespaced key the
        chrys usage middleware whitelists.
        """
        return _add_deepseek_cache_usage(super()._parse_usage_from_openai(usage), usage)

    def _should_replay_reasoning(self, chat_messages: Sequence[Message], *, request_has_tools: bool = False) -> bool:
        """Replay historical reasoning only once tools are in play.

        DeepSeek's thinking-mode docs require every historical assistant
        message to carry its ``reasoning_content`` on a request that sends
        ``tools`` (even one without a tool call) and on an assistant message
        carrying ``tool_calls``, and say a request with neither ignores the
        field. Such a request stays lean instead of replaying
        unconditionally like the base client.
        """
        return request_has_tools or self._has_tool_interaction(chat_messages)

    def _prepare_messages_for_openai(
        self,
        chat_messages: Sequence[Message],
        role_key: str = "role",
        content_key: str = "content",
        *,
        request_has_tools: bool = False,
    ) -> list[dict[str, Any]]:
        chat_messages = self._degrade_cross_provider_hosted_history(chat_messages)
        replay_reasoning = self._should_replay_reasoning(chat_messages, request_has_tools=request_has_tools)
        list_of_list = [
            self._prepare_message_for_openai(message, replay_reasoning=replay_reasoning) for message in chat_messages
        ]
        wire_messages = list(chain.from_iterable(list_of_list))
        if replay_reasoning:
            # Once tools are in play, the thinking-mode docs require
            # ``reasoning_content`` on every assistant message. History
            # another model wrote (or DeepSeek with thinking off) has none:
            # send it empty on the wire only, leaving the stored history as
            # it is.
            for wire_message in wire_messages:
                if wire_message.get("role") == "assistant" and "reasoning_content" not in wire_message:
                    wire_message["reasoning_content"] = ""
        return self._insert_synthetic_image_messages(wire_messages)

    def _prepare_message_for_openai(
        self,
        message: Message,
        *,
        replay_reasoning: bool = True,
    ) -> list[dict[str, Any]]:
        if message.role in ("system", "developer"):
            texts = [content.text for content in message.contents if content.type == "text" and content.text]
            if texts:
                sys_args: dict[str, Any] = {"role": message.role, "content": "\n".join(texts)}
                if author_name := _sanitize_author_name(message.author_name):
                    sys_args["name"] = author_name
                return [sys_args]
            return []

        # Reasoning-bearing assistant messages delegate to the base run
        # coalescer: it emits string-content aggregates (DeepSeek's schema
        # rejects assistant content arrays), keeps non-text fragments as
        # standalone messages instead of merging them into the tool_calls
        # carrier, and owns per-run reasoning placement around tool results.
        if replay_reasoning and message.role == "assistant" and self._replayable_reasoning_fields(message):
            return self._prepare_reasoning_assistant_message(message)

        all_messages: list[dict[str, Any]] = []
        for content in message.contents:
            args: dict[str, Any] = {"role": message.role}
            if message.role != "tool" and (author_name := _sanitize_author_name(message.author_name)):
                args["name"] = author_name

            match content.type:
                case "function_call":
                    if all_messages and "tool_calls" in all_messages[-1]:
                        all_messages[-1]["tool_calls"].append(self._prepare_content_for_openai(content))
                    elif self._can_extend_last_assistant_message(all_messages, message):
                        all_messages[-1]["tool_calls"] = [self._prepare_content_for_openai(content)]
                    else:
                        args["tool_calls"] = [self._prepare_content_for_openai(content)]
                case "function_result":
                    args["tool_call_id"] = content.call_id
                    args["content"], pending_image_parts = self._lower_function_result_to_openai(content)
                    if pending_image_parts:
                        args[_PENDING_IMAGE_PARTS_KEY] = pending_image_parts
                    all_messages.append(args)
                    continue
                case "text_reasoning":
                    # Never emits inline: replayable assistant reasoning was
                    # delegated above; anything else (non-assistant roles,
                    # replay off, non-contributing fragments) drops.
                    continue
                case _:
                    if self._can_extend_last_content_message(all_messages, message):
                        last_content = all_messages[-1].setdefault("content", [])
                        if not isinstance(last_content, list):
                            last_content = [Content.from_text(text=str(last_content)).to_dict(exclude_none=True)]
                            all_messages[-1]["content"] = last_content
                        cast(list[dict[str, Any]], last_content).append(self._prepare_content_for_openai(content))
                    else:
                        if "content" not in args:
                            args["content"] = []
                        args["content"].append(self._prepare_content_for_openai(content))
            if "content" in args or "tool_calls" in args:
                self._ensure_assistant_tool_call_content(args)
                all_messages.append(args)

        for msg in all_messages:
            msg_content: Any = msg.get("content")
            if isinstance(msg_content, list):
                typed_msg_content = cast(list[object], msg_content)
                text_items: list[Mapping[str, Any]] = []
                for item in typed_msg_content:
                    if not isinstance(item, Mapping):
                        break
                    text_item = cast(Mapping[str, Any], item)
                    if text_item.get("type") != "text":
                        break
                    text_items.append(text_item)
                else:
                    msg["content"] = "\n".join(
                        text_item.get("text", "") if isinstance(text_item.get("text", ""), str) else ""
                        for text_item in text_items
                    )

        return all_messages

    def _ensure_assistant_tool_call_content(self, message: dict[str, Any]) -> None:
        if message.get("role") == "assistant" and "tool_calls" in message and "content" not in message:
            message["content"] = ""

    def _can_extend_last_content_message(
        self,
        all_messages: list[dict[str, Any]],
        message: Message,
    ) -> bool:
        return bool(
            all_messages
            and message.role != "tool"
            and all_messages[-1].get("role") == message.role
            and "tool_call_id" not in all_messages[-1]
        )

    def _has_tool_interaction(self, chat_messages: Sequence[Message]) -> bool:
        return any(
            message.role == "tool"
            or any(content.type in ("function_call", "function_result") for content in message.contents)
            for message in chat_messages
        )

    def _can_extend_last_assistant_message(
        self,
        all_messages: list[dict[str, Any]],
        message: Message,
    ) -> bool:
        return bool(
            all_messages
            and message.role == "assistant"
            and all_messages[-1].get("role") == "assistant"
            and "tool_call_id" not in all_messages[-1]
        )


class DeepSeekResponsesClient(RawOpenAIChatClient):
    """DeepSeek's stateless OpenAI-compatible Responses wire dialect."""

    STORES_BY_DEFAULT: ClassVar[bool] = False
    FORCES_STATELESS: ClassVar[bool] = True
    INJECT_ENCRYPTED_REASONING_INCLUDE: ClassVar[bool] = False
    MINTS_CONTINUATION_TOKENS: ClassVar[bool] = False
    HOSTED_PROVIDER: ClassVar[str] = "deepseek-openai"
    REASONING_REPLAY_MODE: ClassVar[DeepSeekResponsesReasoningReplayMode] = "plaintext-replay"

    @staticmethod
    def _occurrence_marker_values(contents: Sequence[Content]) -> list[Any]:
        return [
            content.additional_properties[REASONING_FORMAT_KEY]
            for content in contents
            if REASONING_FORMAT_KEY in content.additional_properties
        ]

    @classmethod
    def _occurrence_is_valid(cls, contents: Sequence[Content]) -> bool:
        markers = cls._occurrence_marker_values(contents)
        if markers:
            return all(
                marker in (REASONING_CONTENT_FIELD, REASONING_DETAILS_FIELD, REASONING_FIELD) for marker in markers
            )
        payload = any(
            content.protected_data or content.additional_properties.get("encrypted_content") for content in contents
        )
        return not (payload and not contents[0].id)

    def _prepare_reasoning_items_for_openai(
        self,
        occurrences: Sequence[Sequence[Content]],
    ) -> dict[int, dict[str, Any]]:
        if type(self).REASONING_REPLAY_MODE == "encrypted":
            return super()._prepare_reasoning_items_for_openai(occurrences)

        reasoning_items: dict[int, dict[str, Any]] = {}
        if type(self).REASONING_REPLAY_MODE == "plaintext-valid-drop":
            return reasoning_items

        for contents in occurrences:
            if not contents or not self._occurrence_is_valid(contents) or self._occurrence_marker_values(contents):
                continue
            reasoning_id = contents[0].id
            reasoning_texts: list[dict[str, str]] = []
            status: Any = None
            for content in contents:
                properties = content.additional_properties
                if properties.get("status") is not None:
                    status = properties["status"]
                marker = properties.get("reasoning_text")
                if marker:
                    reasoning_text = content.text if marker is True else marker
                    if isinstance(reasoning_text, str) and reasoning_text:
                        reasoning_texts.append({"type": "reasoning_text", "text": reasoning_text})
            if not reasoning_texts:
                continue
            item: dict[str, Any] = {
                "type": "reasoning",
                "content": reasoning_texts,
            }
            if reasoning_id:
                item["id"] = reasoning_id
            if status is not None:
                item["status"] = status
            # Composite occurrences replay only this plaintext channel;
            # encrypted-only occurrences never reach this assignment.
            reasoning_items[id(contents[0])] = item
        return reasoning_items

    def _reasoning_occurrences_are_valid(
        self,
        occurrences: Sequence[Sequence[Content]],
        reasoning_items: Mapping[int, dict[str, Any]],
    ) -> bool:
        if type(self).REASONING_REPLAY_MODE == "encrypted":
            return all(
                self._occurrence_is_valid(contents) for contents in occurrences
            ) and super()._reasoning_occurrences_are_valid(
                occurrences,
                reasoning_items,
            )
        return all(self._occurrence_is_valid(contents) for contents in occurrences)

    @staticmethod
    def _stateless_option_view(options: Mapping[str, Any]) -> dict[str, Any]:
        stateless = dict(options)
        extra_body = stateless.get("extra_body")
        clean_extra_body = dict(extra_body) if isinstance(extra_body, Mapping) else None
        had_store = "store" in stateless or (clean_extra_body is not None and "store" in clean_extra_body)
        for key in ("conversation_id", "previous_response_id", "conversation", "continuation_token"):
            stateless.pop(key, None)
            if clean_extra_body is not None:
                clean_extra_body.pop(key, None)
        if had_store:
            stateless["store"] = False
            if clean_extra_body is not None:
                clean_extra_body.pop("store", None)
        if clean_extra_body is not None:
            stateless["extra_body"] = clean_extra_body
        return stateless

    @staticmethod
    def _reject_stateful_execution_options(options: Mapping[str, Any]) -> None:
        extra_body = options.get("extra_body")
        nested = extra_body if isinstance(extra_body, Mapping) else {}
        if options.get("continuation_token") is not None or nested.get("continuation_token") is not None:
            raise ChatClientInvalidRequestException("DeepSeek Responses does not support continuation_token")
        if options.get("background") is not None or nested.get("background") is not None:
            raise ChatClientInvalidRequestException(
                "DeepSeek Responses does not support background responses because the dialect is stateless"
            )

    async def _prepare_options(
        self,
        messages: Sequence[Message],
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        return await super()._prepare_options(messages, self._stateless_option_view(options))

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        options: Mapping[str, Any],
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        self._reject_stateful_execution_options(options)
        return super()._inner_get_response(messages=messages, options=options, stream=stream, **kwargs)

    def _get_conversation_id(self, response: Any, store: bool | None) -> None:
        del response, store

    def _parse_usage_from_openai(self, usage: Any) -> Any:
        return _add_deepseek_cache_usage(super()._parse_usage_from_openai(usage), usage)
