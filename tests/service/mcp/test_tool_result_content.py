# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Media and links an MCP server sends, as the model gets them.

One bad media item becomes a placeholder and the rest of the result survives;
a link reaches the model as text, never as the linked item itself.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from mcp import types

from chrys.service.mcp._http_transport import _HTTPMCPTool
from chrys.service.mcp.content import (
    INVALID_AUDIO_TEXT,
    INVALID_IMAGE_TEXT,
    INVALID_RESOURCE_TEXT,
    decode_media_base64,
)
from tests.support.images import image_bytes

_PNG = image_bytes("PNG")
_PNG_BASE64 = base64.b64encode(_PNG).decode("ascii")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (_PNG_BASE64, _PNG),
        (f"data:image/png;base64,{_PNG_BASE64}", _PNG),
        ("\n".join([_PNG_BASE64[:20], _PNG_BASE64[20:]]), _PNG),
        ("", b""),
        ("not base64!", None),
        ("aGVsbG8", None),
        ("data:text/plain,hello", None),
    ],
    ids=["bare", "data-uri", "line-wrapped", "empty", "invalid", "unpadded", "data-uri-not-base64"],
)
def test_decode_media_base64(value: str, expected: bytes | None) -> None:
    assert decode_media_base64(value) == expected


def _items(blob: str) -> list[Any]:
    """Text, then one each of a bad image, an empty audio, a good image and a blob resource."""
    return [
        types.TextContent(type="text", text="chart:"),
        types.ImageContent(type="image", data="not base64!", mimeType="image/png"),
        types.AudioContent(type="audio", data="", mimeType="audio/wav"),
        types.ImageContent(type="image", data=_PNG_BASE64, mimeType="image/png"),
        types.EmbeddedResource(
            type="resource",
            resource=types.BlobResourceContents(uri="file:///r.bin", blob=blob, mimeType="application/octet-stream"),
            annotations=types.Annotations(priority=0.5),
        ),
    ]


def _parse_tool_result(items: list[Any]) -> list[Any]:
    tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
    return tool._parse_tool_result_from_mcp(types.CallToolResult(content=items))


def test_bad_media_becomes_a_placeholder_and_the_rest_is_kept() -> None:
    out = _parse_tool_result(_items(blob=""))

    assert [content.type for content in out] == ["text", "text", "text", "data", "data"]
    assert [content.text for content in out[:3]] == ["chart:", INVALID_IMAGE_TEXT, INVALID_AUDIO_TEXT]
    assert out[3].uri == f"data:image/png;base64,{_PNG_BASE64}"
    # An empty binary resource is valid, unlike an empty image or audio.
    assert out[4].uri == "data:application/octet-stream;base64,"


def test_a_blob_resource_reads_bare_base64_and_says_when_it_is_not() -> None:
    assert _parse_tool_result(_items(blob=_PNG_BASE64))[4].uri == f"data:application/octet-stream;base64,{_PNG_BASE64}"
    assert _parse_tool_result(_items(blob="not base64!"))[4].text == INVALID_RESOURCE_TEXT


def _link(uri: str, mime_type: str | None) -> Any:
    return types.ResourceLink(
        type="resource_link", uri=uri, name="report", mimeType=mime_type, description="Q3 numbers"
    )


@pytest.mark.parametrize(
    ("uri", "mime_type"),
    [
        ("https://example.com/report", None),
        ("file:///tmp/report.pdf", "application/pdf"),
        ("https://example.com/report.pdf", "application/pdf"),
        ("file:///tmp/chart.png", "image/png"),
        ("https://example.com/chart.svg", "image/svg+xml"),
        ("https://example.com/chart.png", "image/png"),
    ],
    ids=["no-type", "local-file", "not-an-image", "local-image", "unread-image-format", "web-image"],
)
def test_a_link_reaches_the_model_as_text(uri: str, mime_type: str | None) -> None:
    (out,) = _parse_tool_result([_link(uri, mime_type)])

    assert out.type == "text"
    assert out.text.splitlines()[:2] == ["Resource link: report", f"URI: {uri}"]
    assert "Description: Q3 numbers" in out.text
