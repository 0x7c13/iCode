# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cell-based image previews for chat messages."""

from __future__ import annotations

import base64
import binascii
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any, cast

from PIL import Image, ImageOps, UnidentifiedImageError
from rich.color import Color
from rich.console import Console, ConsoleOptions, RenderResult
from rich.style import Style
from rich.text import Text

from chrys.foundation.text.images import IMAGE_DECODE_FORMATS

_HALF_BLOCK = "\u2580"
_TRANSPARENT_BACKGROUND = (15, 18, 24)
_DEFAULT_MAX_ROWS = 16
_IMAGE_GAP = 2
_MIN_IMAGE_WIDTH = 8
_MAX_PREVIEW_INPUT_PIXELS = 80_000_000
_MAX_PREVIEW_SOURCE_SIZE = (512, 512)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatImagePreview:
    """Decoded image data ready for cell rendering."""

    image: Image.Image
    media_type: str


def extract_image_previews(contents: Sequence[Any] | None) -> list[ChatImagePreview]:
    """Decode image content blocks from a framework/user-message content list."""
    if not contents:
        return []
    previews: list[ChatImagePreview] = []
    for content in contents:
        extracted = _extract_content_data_uri(content)
        if extracted is None:
            continue
        media_type, data_uri = extracted
        if not media_type.startswith("image/"):
            continue
        try:
            data = _decode_data_uri(data_uri)
            image = _load_preview_image(data)
        except binascii.Error, Image.DecompressionBombError, OSError, UnidentifiedImageError, ValueError:
            logger.debug("Skipping invalid chat image preview for media type %s", media_type, exc_info=True)
            continue
        previews.append(ChatImagePreview(image=image, media_type=media_type))
    return previews


class ImagePreviewGrid:
    """Rich renderable that shows one row of image previews."""

    def __init__(
        self,
        images: Sequence[ChatImagePreview],
        *,
        max_width: int,
        max_rows: int = _DEFAULT_MAX_ROWS,
        indent: str = "",
        allow_upscale: bool = False,
    ) -> None:
        self.images = list(images)
        self.max_width = max(1, max_width)
        self.max_rows = max(1, max_rows)
        self.indent = indent
        self.allow_upscale = allow_upscale

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        _ = console, options
        yield from self.render_lines()

    def render_lines(self) -> list[Text]:
        """Render images into terminal-cell rows."""
        if not self.images:
            return []

        available = max(1, self.max_width - len(self.indent))
        gap = _IMAGE_GAP if len(self.images) > 1 else 0
        total_gap = gap * (len(self.images) - 1)
        per_image_width = max(1, (available - total_gap) // len(self.images))
        if per_image_width < _MIN_IMAGE_WIDTH and len(self.images) > 1:
            gap = 1
            total_gap = gap * (len(self.images) - 1)
            per_image_width = max(1, (available - total_gap) // len(self.images))

        rendered_images = [
            _render_single_image(
                preview.image,
                max_width=per_image_width,
                max_rows=self.max_rows,
                allow_upscale=self.allow_upscale,
            )
            for preview in self.images
        ]
        max_line_count = max((len(lines) for lines in rendered_images), default=0)
        output: list[Text] = []

        for row_index in range(max_line_count):
            line = Text(self.indent)
            for image_index, image_lines in enumerate(rendered_images):
                if row_index < len(image_lines):
                    line.append(image_lines[row_index])
                else:
                    image_width = _image_render_width(image_lines)
                    line.append(" " * image_width)
                if image_index < len(rendered_images) - 1:
                    line.append(" " * gap)
            if line.cell_len > self.max_width:
                line.truncate(self.max_width, overflow="crop")
            output.append(line)
        return output


def _extract_content_data_uri(content: Any) -> tuple[str, str] | None:
    if isinstance(content, dict):
        media_type = content.get("media_type") or content.get("mime_type") or ""
        if not media_type:
            extra = content.get("additional_properties")
            media_type = extra.get("media_type") if isinstance(extra, dict) else ""
        uri = content.get("uri") or ""
    else:
        # Kernel Content is an external type; rich content fields
        # vary by content kind, so this boundary intentionally uses getattr.
        media_type = getattr(content, "media_type", "") or ""
        uri = getattr(content, "uri", "") or ""
    if not isinstance(media_type, str) or not isinstance(uri, str):
        return None
    if not uri.startswith("data:"):
        return None
    if not media_type:
        media_type = uri[5:].split(",", 1)[0].split(";", 1)[0]
    return media_type, uri


def _decode_data_uri(uri: str) -> bytes:
    header, encoded = uri.split(",", 1)
    if ";base64" not in header:
        msg = "image preview data URI is not base64 encoded"
        raise ValueError(msg)
    return base64.b64decode(encoded, validate=True)


def _load_preview_image(data: bytes) -> Image.Image:
    with Image.open(BytesIO(data), formats=IMAGE_DECODE_FORMATS) as opened:
        if opened.width * opened.height > _MAX_PREVIEW_INPUT_PIXELS:
            msg = "image preview input exceeds the supported pixel limit"
            raise ValueError(msg)
        opened.draft("RGB", _MAX_PREVIEW_SOURCE_SIZE)
        image = ImageOps.exif_transpose(opened)
        image.thumbnail(_MAX_PREVIEW_SOURCE_SIZE, Image.Resampling.LANCZOS)
        if image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, _TRANSPARENT_BACKGROUND)
            background.paste(rgba, mask=rgba.getchannel("A"))
            return background
        return image.convert("RGB")


def _render_single_image(
    image: Image.Image, *, max_width: int, max_rows: int, allow_upscale: bool = False
) -> list[Text]:
    fitted = _fit_image(image, max_width=max_width, max_height=max_rows * 2, allow_upscale=allow_upscale)
    if fitted.height % 2:
        padded = Image.new("RGB", (fitted.width, fitted.height + 1), _TRANSPARENT_BACKGROUND)
        padded.paste(fitted, (0, 0))
        fitted = padded

    pixels = fitted.load()
    if pixels is None:
        return []
    lines: list[Text] = []
    for y in range(0, fitted.height, 2):
        line = Text()
        for x in range(fitted.width):
            # Runtime guarantee: preview loading normalizes every image to RGB.
            upper = cast(tuple[int, int, int], pixels[x, y])
            lower = cast(tuple[int, int, int], pixels[x, y + 1])
            line.append(
                _HALF_BLOCK,
                style=Style(
                    color=Color.from_rgb(*upper),
                    bgcolor=Color.from_rgb(*lower),
                ),
            )
        lines.append(line)
    return lines


def _fit_image(image: Image.Image, *, max_width: int, max_height: int, allow_upscale: bool = False) -> Image.Image:
    width, height = image.size
    scale = min(max_width / width, max_height / height)
    if not allow_upscale:
        scale = min(scale, 1.0)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    if new_size == image.size:
        return image.copy()
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _image_render_width(lines: list[Text]) -> int:
    return max((line.cell_len for line in lines), default=0)
