# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for ACP prompt content conversion."""

from __future__ import annotations

import base64

import pytest
from acp.helpers import audio_block, image_block, resource_block, resource_link_block, text_block
from acp.schema import BlobResourceContents, TextResourceContents

from chrys.app.acp.content import AcpContentError, convert_prompt_blocks
from chrys.orchestration.engine.run.attachments import MAX_IMAGE_BYTES
from tests.support.images import image_bytes


def test_convert_prompt_blocks_keeps_plain_text_as_normal_user_text() -> None:
    converted = convert_prompt_blocks([text_block("hello")], vision_enabled=False)

    assert converted.message.text == "hello"
    assert converted.message.prepared_contents is None
    assert converted.has_images is False


def test_convert_prompt_blocks_attaches_images_when_vision_enabled() -> None:
    data = base64.b64encode(image_bytes("PNG")).decode("ascii")

    converted = convert_prompt_blocks([text_block("describe"), image_block(data, "image/png")], vision_enabled=True)

    assert converted.message.text == "describe"
    assert converted.message.prepared_contents is not None
    assert converted.message.prepared_contents[0] == "describe"
    assert converted.message.prepared_contents[1].media_type == "image/png"
    assert converted.has_images is True


def test_convert_prompt_blocks_types_an_image_by_its_bytes() -> None:
    data = base64.b64encode(image_bytes("JPEG")).decode("ascii")

    converted = convert_prompt_blocks([image_block(data, "image/png")], vision_enabled=True)

    assert converted.message.prepared_contents is not None
    assert converted.message.prepared_contents[-1].media_type == "image/jpeg"


def test_convert_prompt_blocks_rejects_images_the_model_apis_cannot_read() -> None:
    data = base64.b64encode(image_bytes("BMP")).decode("ascii")

    with pytest.raises(AcpContentError, match="Unsupported image format"):
        convert_prompt_blocks([image_block(data, "image/png")], vision_enabled=True)


def test_convert_prompt_blocks_rejects_images_when_vision_disabled() -> None:
    data = base64.b64encode(b"image-bytes").decode("ascii")

    with pytest.raises(AcpContentError, match="does not support image input"):
        convert_prompt_blocks([image_block(data, "image/png")], vision_enabled=False)


def test_convert_prompt_blocks_rejects_oversized_images() -> None:
    data = base64.b64encode(b"x" * (MAX_IMAGE_BYTES + 1)).decode("ascii")

    with pytest.raises(AcpContentError, match="exceeds"):
        convert_prompt_blocks([image_block(data, "image/png")], vision_enabled=True)


def test_convert_prompt_blocks_rejects_audio() -> None:
    with pytest.raises(AcpContentError, match="Audio content is not supported"):
        convert_prompt_blocks([audio_block("AAAA", "audio/wav")], vision_enabled=True)


def test_convert_prompt_blocks_preserves_resource_context_as_text() -> None:
    converted = convert_prompt_blocks(
        [
            text_block("use this"),
            resource_link_block("readme", "file:///repo/README.md", mime_type="text/markdown", size=12),
            resource_block(TextResourceContents(uri="file:///repo/a.txt", text="embedded", mimeType="text/plain")),
        ],
        vision_enabled=False,
    )

    assert "use this" in converted.message.text
    assert "Resource link: readme" in converted.message.text
    assert "URI: file:///repo/README.md" in converted.message.text
    assert "Embedded resource: file:///repo/a.txt (text/plain)" in converted.message.text
    assert "embedded" in converted.message.text


def test_convert_prompt_blocks_rejects_embedded_blobs() -> None:
    with pytest.raises(AcpContentError, match="Embedded blob resources are not supported"):
        convert_prompt_blocks(
            [resource_block(BlobResourceContents(uri="file:///repo/blob.bin", blob="AAAA"))],
            vision_enabled=True,
        )
