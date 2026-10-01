# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""Stream events read back as chat response updates.

One :class:`StreamState` reads one stream. Events about an output item name
it by its output index, so the state keeps one :class:`OutputSlot` per index:
the function call it announced, the hosted call and result contents later
events update in place, and the text contents its message envelope belongs
to. An update carries what the event added; a hosted content already sent
is sent again whenever an event changes it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from chrys.foundation.hosted_tools import PRESENTATION_TEXT_SEGMENT_ID_KEY, HostedRetrySafety, HostedToolPhase
from chrys.kernel import (
    OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY,
    Annotation,
    ChatResponseUpdate,
    Content,
    TextSpanRegion,
)
from chrys.kernel.exchanges import TOOL_RESULT_CONTENT_TYPES
from chrys.service.profiles.models.options import effective_store_option

from .decode import (
    RUNNING_STATUSES,
    OpenAIContinuationToken,
    continuation_token,
    conversation_handle,
    decode_client_tool_call,
    decode_reasoning_item,
    decode_usage,
    finish_reason,
    logprobs_metadata,
    output_message_envelope,
    timestamp,
)
from .hosted import decode_hosted_item, image_data_uri, item_properties, refresh_in_place

if TYPE_CHECKING:
    from .client import ResponsesVariant

logger = logging.getLogger(__name__)

# Progress events of hosted tools; the last segment is the new status.
_STATUS_EVENTS = (
    "response.web_search_call.in_progress",
    "response.web_search_call.searching",
    "response.web_search_call.completed",
    "response.file_search_call.in_progress",
    "response.file_search_call.searching",
    "response.file_search_call.completed",
    "response.mcp_call.in_progress",
    "response.mcp_call.completed",
    "response.mcp_call.failed",
    "response.code_interpreter_call.in_progress",
    "response.code_interpreter_call.interpreting",
    "response.code_interpreter_call.completed",
    "response.image_generation_call.in_progress",
    "response.image_generation_call.generating",
    "response.image_generation_call.completed",
)


@dataclass(slots=True)
class OutputSlot:
    """What the stream has seen of one output item."""

    function_call: tuple[str, str] | None = None
    call: Content | None = None
    result: Content | None = None
    envelope: dict[str, str] | None = None
    message_contents: list[Content] = field(default_factory=list)

    def merge(self, snapshot: Content) -> Content:
        """Fold a whole-item snapshot into the content already sent for its side.

        The first snapshot becomes the carrier; later ones refresh it in
        place, so everything holding the carrier sees the newest state.
        """
        if snapshot.type in TOOL_RESULT_CONTENT_TYPES:
            self.result = _fold(self.result, snapshot)
            return self.result
        self.call = _fold(self.call, snapshot)
        return self.call


def _fold(carrier: Content | None, snapshot: Content) -> Content:
    if carrier is None:
        return snapshot
    refresh_in_place(carrier, snapshot)
    return carrier


@dataclass(slots=True)
class _Update:
    model: str
    contents: list[Content] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    conversation_id: str | None = None
    response_id: str | None = None
    created_at: str | None = None
    continuation_token: OpenAIContinuationToken | None = None
    finish_reason: Literal["length"] | None = None


class StreamState:
    """Reads the events of one stream, in order."""

    def __init__(self, options: Mapping[str, Any], *, model: str, variant: ResponsesVariant) -> None:
        self._options = options
        self._model = model
        self._variant = variant
        self._slots: dict[Any, OutputSlot] = {}
        # Reasoning items whose text came as deltas: their done event repeats it.
        self._reasoning_with_deltas: set[str] = set()

    def update_for(self, event: Any) -> ChatResponseUpdate:
        """The update one event makes."""
        update = _Update(model=self._model)
        handler = _HANDLERS.get(event.type)
        if handler is None:
            logger.debug("Unparsed event of type: %s: %s", event.type, event)
        else:
            handler(self, event, update)
        return ChatResponseUpdate(
            contents=update.contents,
            conversation_id=update.conversation_id,
            response_id=update.response_id,
            role="assistant",
            model=update.model,
            created_at=update.created_at,
            continuation_token=update.continuation_token,
            finish_reason=update.finish_reason,
            additional_properties=update.metadata,
            raw_representation=event,
        )

    def _slot(self, index: Any) -> OutputSlot:
        return self._slots.setdefault(index, OutputSlot())

    def _store(self) -> Any:
        return effective_store_option(self._options)

    # Text

    def _part_added(self, event: Any, update: _Update) -> None:
        part = event.part
        if part.type == "output_text":
            update.contents.append(self._message_text(event, part.text))
            update.metadata.update(logprobs_metadata(part))
        elif part.type == "refusal":
            update.contents.append(self._message_text(event, part.refusal))

    def _text_delta(self, event: Any, update: _Update) -> None:
        update.contents.append(self._message_text(event, event.delta))
        update.metadata.update(logprobs_metadata(event))

    def _annotation_added(self, event: Any, update: _Update) -> None:
        if (citation := streamed_citation(event)) is not None:
            update.contents.append(self._message_text(event, "", annotations=[citation]))

    def _message_text(self, event: Any, text: str, annotations: list[Annotation] | None = None) -> Content:
        """A text content of an output message, kept to receive its envelope when the item is done."""
        properties: dict[str, Any] = {}
        if segment_id := _text_segment_id(event):
            properties[PRESENTATION_TEXT_SEGMENT_ID_KEY] = segment_id
        index = getattr(event, "output_index", None)
        tracked = isinstance(index, int) and not isinstance(index, bool)
        if tracked and (envelope := self._slot(index).envelope):
            properties[OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY] = dict(envelope)
        content = Content.from_text(
            text=text, annotations=annotations, raw_representation=event, additional_properties=properties or None
        )
        if tracked:
            self._slot(index).message_contents.append(content)
        return content

    # Reasoning

    def _reasoning_delta(self, event: Any, update: _Update) -> None:
        self._reasoning_with_deltas.add(event.item_id)
        update.contents.append(_reasoning_text(event, event.delta))
        update.metadata.update(logprobs_metadata(event))

    def _reasoning_done(self, event: Any, update: _Update) -> None:
        if event.item_id not in self._reasoning_with_deltas:
            update.contents.append(_reasoning_text(event, event.text))
        update.metadata.update(logprobs_metadata(event))

    # Response lifecycle

    def _created(self, event: Any, update: _Update) -> None:
        response = event.response
        update.response_id = response.id
        update.conversation_id = conversation_handle(response, store=self._store(), variant=self._variant)
        if response.status in RUNNING_STATUSES:
            update.continuation_token = continuation_token(response.id, store=self._store(), variant=self._variant)

    def _in_progress(self, event: Any, update: _Update) -> None:
        response = event.response
        update.response_id = response.id
        update.conversation_id = conversation_handle(response, store=self._store(), variant=self._variant)
        update.continuation_token = continuation_token(response.id, store=self._store(), variant=self._variant)

    def _finished(self, event: Any, update: _Update) -> None:
        response = event.response
        update.response_id = response.id
        update.conversation_id = conversation_handle(response, store=self._store(), variant=self._variant)
        update.model = response.model
        update.created_at = timestamp(response.created_at)
        if response.usage and (usage := decode_usage(response.usage, variant=self._variant)):
            update.contents.append(Content.from_usage(usage_details=usage, raw_representation=event))
        update.finish_reason = finish_reason(response)

    # Output items

    def _item_added(self, event: Any, update: _Update) -> None:
        item = event.item
        index = getattr(event, "output_index", -1)
        match item.type:
            case "message":
                if envelope := output_message_envelope(item):
                    self._slot(index).envelope = envelope
            case "function_call":
                self._slot(index).function_call = (item.call_id, item.name)
            case "reasoning":
                update.contents.extend(decode_reasoning_item(item, streamed=True))
            case "shell_call_output" | "tool_search_output":
                # Results are decoded once, from their done item.
                pass
            case _:
                if decoded := decode_hosted_item(item, self._variant.hosted_provider, phase=HostedToolPhase.START):
                    self._slot(index).call = decoded[0]
                    update.contents.append(decoded[0])

    def _item_done(self, event: Any, update: _Update) -> None:
        item = event.item
        slot = self._slot(getattr(event, "output_index", -1))
        match getattr(item, "type", None):
            case "message":
                envelope = output_message_envelope(item)
                if envelope:
                    slot.envelope = envelope
                for content in slot.message_contents:
                    if envelope:
                        content.additional_properties[OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY] = dict(envelope)
                    else:
                        content.additional_properties.pop(OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY, None)
            case "reasoning":
                # The payload only arrives once the item is done.
                if payload := getattr(item, "encrypted_content", None):
                    update.contents.append(
                        Content.from_text_reasoning(
                            id=getattr(item, "id", None), text="", protected_data=payload, raw_representation=item
                        )
                    )
            case "custom_tool_call":
                name = getattr(item, "name", "") or ""
                update.contents.append(decode_client_tool_call(item, name=name, arguments=getattr(item, "input", None)))
            case "apply_patch_call":
                operation = getattr(item, "operation", None)
                update.contents.append(decode_client_tool_call(item, name="apply_patch", arguments=operation))
            case item_type:
                snapshots = decode_hosted_item(item, self._variant.hosted_provider) or []
                update.contents.extend(slot.merge(snapshot) for snapshot in snapshots)
                if item_type == "image_generation_call" and len(snapshots) == 1 and slot.result is not None:
                    # No final image: the result built from partial images ends here.
                    status = getattr(item, "status", None)
                    slot.result.status = status
                    slot.result.provider_status = status
                    slot.result.provider_phase = HostedToolPhase.TERMINAL
                    slot.result.additional_properties = item_properties(item, shadow=True)
                    slot.result.raw_representation = item
                    update.contents.append(slot.result)

    # Tool progress

    def _arguments_delta(self, event: Any, update: _Update) -> None:
        call_id, name = self._slot(event.output_index).function_call or (None, None)
        if call_id and name:
            update.contents.append(
                Content.from_function_call(
                    call_id=call_id,
                    name=name,
                    arguments=event.delta,
                    additional_properties={"output_index": event.output_index, "fc_id": event.item_id},
                    raw_representation=event,
                )
            )

    def _status_changed(self, event: Any, update: _Update) -> None:
        call = self._slot(getattr(event, "output_index", -1)).call
        if call is not None:
            status = event.type.rsplit(".", 1)[-1]
            call.status = status
            call.provider_status = status
            call.provider_phase = HostedToolPhase.SNAPSHOT
            call.raw_representation = event
            update.contents.append(call)

    def _code_delta(self, event: Any, update: _Update) -> None:
        properties = _code_properties(event)
        slot = self._slot(event.output_index)
        call = slot.call
        if call is None:
            call = slot.call = self._code_call(event)
            update.contents.append(call)
        if not call.inputs:
            call.inputs = [Content.from_text(text="")]
        code = call.inputs[0]
        code.text = (code.text or "") + event.delta
        code.raw_representation = event
        code.additional_properties = properties
        call.provider_phase = HostedToolPhase.DELTA
        call.raw_representation = event
        if call not in update.contents:
            update.contents.append(call)
        update.metadata.update(logprobs_metadata(event))

    def _code_done(self, event: Any, update: _Update) -> None:
        properties = _code_properties(event)
        slot = self._slot(event.output_index)
        call = slot.call
        if call is None:
            call = slot.call = self._code_call(event)
        call.inputs = [Content.from_text(text=event.code, raw_representation=event, additional_properties=properties)]
        call.provider_phase = HostedToolPhase.SNAPSHOT
        call.raw_representation = event
        update.contents.append(call)
        update.metadata.update(logprobs_metadata(event))

    def _code_call(self, event: Any) -> Content:
        """The code call for code that arrives before its item."""
        return Content.from_code_interpreter_tool_call(
            call_id=getattr(event, "call_id", None) or getattr(event, "id", None) or event.item_id,
            inputs=[],
            hosted_provider=self._variant.hosted_provider,
            provider_item_type="code_interpreter_call",
            provider_item_id=event.item_id,
            retry_safety=HostedRetrySafety.SANDBOXED,
        )

    def _partial_image(self, event: Any, update: _Update) -> None:
        """A partial image, added to a result that stays open until the item is done."""
        image = Content.from_uri(
            uri=image_data_uri(event.partial_image_b64),
            additional_properties={"partial_image_index": event.partial_image_index, "is_partial_image": True},
            raw_representation=event,
        )
        image_id = getattr(event, "item_id", None)
        slot = self._slot(getattr(event, "output_index", -1))
        provider = self._variant.hosted_provider
        if slot.call is None:
            slot.call = Content.from_image_generation_tool_call(
                image_id=image_id,
                hosted_provider=provider,
                provider_item_type="image_generation_call",
                provider_item_id=image_id,
                provider_phase=HostedToolPhase.START,
                retry_safety=HostedRetrySafety.SANDBOXED,
                raw_representation=event,
            )
            update.contents.append(slot.call)
        if slot.result is None:
            slot.result = Content.from_image_generation_tool_result(
                image_id=image_id,
                outputs=[],
                hosted_provider=provider,
                provider_item_type="image_generation_call",
                provider_item_id=image_id,
                provider_phase=HostedToolPhase.SNAPSHOT,
                provider_status="generating",
                retry_safety=HostedRetrySafety.SANDBOXED,
                raw_representation=event,
            )
        result = slot.result
        if not isinstance(result.outputs, list):
            result.outputs = []
        result.outputs.append(image)
        result.provider_phase = HostedToolPhase.SNAPSHOT
        result.provider_status = "generating"
        result.raw_representation = event
        update.contents.append(result)


def _text_segment_id(event: Any) -> str:
    """Which streamed text part *event* belongs to, unique within the response."""
    item_id = getattr(event, "item_id", None)
    index = getattr(event, "output_index", None)
    part = getattr(event, "content_index", None)
    if isinstance(item_id, str) and item_id:
        segment = f"item:{item_id}"
    elif isinstance(index, int) and not isinstance(index, bool):
        segment = f"output:{index}"
    else:
        return ""
    if isinstance(part, int) and not isinstance(part, bool):
        return f"{segment}:content:{part}"
    return segment


def _reasoning_text(event: Any, text: str) -> Content:
    properties = {"reasoning_text": True} if event.type.startswith("response.reasoning_text.") else None
    return Content.from_text_reasoning(
        id=event.item_id, text=text, raw_representation=event, additional_properties=properties
    )


def _code_properties(event: Any) -> dict[str, Any]:
    return {"output_index": event.output_index, "sequence_number": event.sequence_number, "item_id": event.item_id}


def streamed_citation(event: Any) -> Annotation | None:
    """The citation an ``output_text.annotation.added`` event adds, if it names its source.

    The annotation may be a mapping or an object. Blocking responses
    describe citations in another shape (:func:`.decode._citation`).
    """
    annotation = event.annotation

    def value(key: str) -> Any:
        if isinstance(annotation, dict):
            return annotation.get(key)
        return getattr(annotation, key, None)

    kind = value("type")
    file_id = value("file_id")
    if kind == "file_path":
        if not file_id:
            return None
        return Annotation(
            type="citation",
            file_id=str(file_id),
            additional_properties={"annotation_index": event.annotation_index, "index": value("index")},
            raw_representation=annotation,
        )
    if kind == "file_citation":
        if not file_id:
            return None
        return Annotation(
            type="citation",
            file_id=str(file_id),
            url=value("filename"),
            additional_properties={"annotation_index": event.annotation_index, "index": value("index")},
            raw_representation=annotation,
        )
    if kind == "container_file_citation":
        if not file_id:
            return None
        citation = Annotation(
            type="citation",
            file_id=str(file_id),
            url=value("filename"),
            additional_properties={"annotation_index": event.annotation_index, "container_id": value("container_id")},
            raw_representation=annotation,
        )
    elif kind == "url_citation":
        url = value("url")
        if not url:
            return None
        properties: dict[str, Any] = {"annotation_index": event.annotation_index}
        if (get_url := value("get_url")) is not None:
            properties["get_url"] = get_url
        citation = Annotation(
            type="citation",
            title=value("title") or "",
            url=str(url),
            additional_properties=properties,
            raw_representation=annotation,
        )
    else:
        logger.debug("Unparsed annotation type in streaming: %s", kind)
        return None
    start, end = value("start_index"), value("end_index")
    if start is not None and end is not None:
        citation["annotated_regions"] = [TextSpanRegion(type="text_span", start_index=start, end_index=end)]
    return citation


_HANDLERS: dict[str, Callable[[StreamState, Any, _Update], None]] = {
    "response.content_part.added": StreamState._part_added,
    "response.output_text.delta": StreamState._text_delta,
    "response.refusal.delta": StreamState._text_delta,
    "response.output_text.annotation.added": StreamState._annotation_added,
    "response.reasoning_text.delta": StreamState._reasoning_delta,
    "response.reasoning_summary_text.delta": StreamState._reasoning_delta,
    "response.reasoning_text.done": StreamState._reasoning_done,
    "response.reasoning_summary_text.done": StreamState._reasoning_done,
    "response.created": StreamState._created,
    "response.in_progress": StreamState._in_progress,
    "response.completed": StreamState._finished,
    "response.incomplete": StreamState._finished,
    "response.output_item.added": StreamState._item_added,
    "response.output_item.done": StreamState._item_done,
    "response.function_call_arguments.delta": StreamState._arguments_delta,
    "response.code_interpreter_call_code.delta": StreamState._code_delta,
    "response.code_interpreter_call_code.done": StreamState._code_done,
    "response.image_generation_call.partial_image": StreamState._partial_image,
    **dict.fromkeys(_STATUS_EVENTS, StreamState._status_changed),
}
