# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Anthropic citations decode into annotations with each SDK citation type's own fields."""

from __future__ import annotations

from typing import Any

import pytest
from anthropic.types.beta import (
    BetaCitationCharLocation,
    BetaCitationContentBlockLocation,
    BetaCitationPageLocation,
    BetaCitationSearchResultLocation,
    BetaCitationsWebSearchResultLocation,
    BetaTextBlock,
)

from chrys.kernel import Annotation
from chrys.service.llm.anthropic_messages.decode import decode_citations
from chrys.service.llm.clients import create_client
from tests.support.scripted_wire import ScriptedWire, pin_wire_inputs, route_clients_to
from tests.support.wire_cases._kit import Case, anth_message, anth_replies, weather_question


def _without_raw(annotation: Annotation) -> dict[str, Any]:
    return {key: value for key, value in annotation.items() if key != "raw_representation"}


def _region(start: int, end: int) -> list[dict[str, Any]]:
    return [{"type": "text_span", "start_index": start, "end_index": end}]


@pytest.mark.parametrize(
    ("citation", "expected"),
    [
        pytest.param(
            BetaCitationCharLocation(
                type="char_location",
                cited_text="the sky is blue",
                document_index=0,
                document_title="Sky facts",
                start_char_index=4,
                end_char_index=19,
                file_id="file_char",
            ),
            {
                "title": "Sky facts",
                "snippet": "the sky is blue",
                "file_id": "file_char",
                "annotated_regions": _region(4, 19),
            },
            id="char_location",
        ),
        pytest.param(
            BetaCitationPageLocation(
                type="page_location",
                cited_text="page text",
                document_index=1,
                document_title="Manual",
                start_page_number=2,
                end_page_number=3,
                file_id="file_page",
            ),
            {"title": "Manual", "snippet": "page text", "file_id": "file_page", "annotated_regions": _region(2, 3)},
            id="page_location",
        ),
        pytest.param(
            BetaCitationPageLocation(
                type="page_location",
                cited_text="untitled text",
                document_index=1,
                document_title=None,
                start_page_number=4,
                end_page_number=5,
            ),
            {"snippet": "untitled text", "annotated_regions": _region(4, 5)},
            id="untitled_page_location",
        ),
        pytest.param(
            BetaCitationContentBlockLocation(
                type="content_block_location",
                cited_text="block text",
                document_index=2,
                document_title="Notes",
                start_block_index=0,
                end_block_index=1,
            ),
            {"title": "Notes", "snippet": "block text", "annotated_regions": _region(0, 1)},
            id="content_block_location",
        ),
        pytest.param(
            BetaCitationsWebSearchResultLocation(
                type="web_search_result_location",
                cited_text="search snippet",
                encrypted_index="enc",
                title="Result page",
                url="https://example.com/result",
            ),
            {"title": "Result page", "snippet": "search snippet", "url": "https://example.com/result"},
            id="web_search_result_location",
        ),
        pytest.param(
            BetaCitationSearchResultLocation(
                type="search_result_location",
                cited_text="result text",
                search_result_index=0,
                source="https://example.com/source",
                title="Source",
                start_block_index=1,
                end_block_index=2,
            ),
            {
                "title": "Source",
                "snippet": "result text",
                "url": "https://example.com/source",
                "annotated_regions": _region(1, 2),
            },
            id="search_result_location",
        ),
    ],
)
def test_each_citation_type_decodes_its_own_fields(citation: Any, expected: dict[str, Any]) -> None:
    block = BetaTextBlock(type="text", text="cited answer", citations=[citation])

    annotations = decode_citations(block)

    assert annotations is not None
    assert [_without_raw(annotation) for annotation in annotations] == [{"type": "citation", **expected}]
    assert annotations[0]["raw_representation"] is citation


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streamed"])
async def test_a_response_carries_its_citations(stream: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stream sends each citation as a delta of the text block it cites."""
    pin_wire_inputs(monkeypatch)
    message = anth_message(
        message_id="msg_cited",
        content=[
            {
                "type": "text",
                "text": "The sky is blue.",
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": "the sky is blue",
                        "document_index": 0,
                        "document_title": "Sky facts",
                        "start_char_index": 0,
                        "end_char_index": 15,
                    }
                ],
            }
        ],
    )
    case = Case(
        provider="anthropic",
        replies=anth_replies([message], stream=stream),
        messages=weather_question,
        options=dict,
        stream=stream,
    )
    wire = ScriptedWire(case.replies)
    route_clients_to(wire.transport, monkeypatch)
    stack = await create_client(case.profile(), session_id="citations")
    try:
        result = stack.inner.inner.get_response(case.messages(), options={}, stream=stream)
        response = await (result.get_final_response() if stream else result)
    finally:
        await stack.aclose()

    assert response.text == "The sky is blue."

    [text] = [content for content in response.messages[0].contents if content.type == "text"]
    assert [_without_raw(annotation) for annotation in text.annotations or []] == [
        {
            "type": "citation",
            "title": "Sky facts",
            "snippet": "the sky is blue",
            "annotated_regions": _region(0, 15),
        }
    ]
