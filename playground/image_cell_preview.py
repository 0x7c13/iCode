# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Playground: preview raster images with terminal cell colors.

Usage:
    uv run python playground/image_cell_preview.py
    uv run python playground/image_cell_preview.py /absolute/path/to/image.png
    uv run python playground/image_cell_preview.py --print --width 80 --height 28 /absolute/path/to/image.png

Keybindings:
    q       - Quit
    r       - Reload image from disk
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from rich.color import Color
from rich.console import Console, ConsoleOptions, RenderResult
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.events import Resize
from textual.widgets import Footer, Header, Static

HALF_BLOCK = "\u2580"
DEFAULT_WIDTH = 100
DEFAULT_HEIGHT = 50
BACKGROUND = (15, 18, 24)


class CellImage:
    """Rich renderable that maps two image rows to one terminal cell row."""

    def __init__(
        self,
        image: Image.Image,
        *,
        max_width: int = DEFAULT_WIDTH,
        max_height: int = DEFAULT_HEIGHT,
        background: tuple[int, int, int] = BACKGROUND,
    ) -> None:
        self.image = image
        self.max_width = max(1, max_width)
        self.max_height = max(1, max_height)
        self.background = background

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        max_width: int = DEFAULT_WIDTH,
        max_height: int = DEFAULT_HEIGHT,
        background: tuple[int, int, int] = BACKGROUND,
    ) -> CellImage:
        """Load an image from disk and return a cell preview renderable."""
        return cls(_load_image(path), max_width=max_width, max_height=max_height, background=background)

    def __rich_console__(self, _console: Console, _options: ConsoleOptions) -> RenderResult:
        image = _fit_image(self.image, max_width=self.max_width, max_height=self.max_height * 2)
        if image.height % 2:
            padded = Image.new("RGB", (image.width, image.height + 1), self.background)
            padded.paste(image, (0, 0))
            image = padded

        pixels = image.load()
        for y in range(0, image.height, 2):
            line = Text()
            for x in range(image.width):
                upper = pixels[x, y]
                lower = pixels[x, y + 1]
                line.append(
                    HALF_BLOCK,
                    style=Style(
                        color=Color.from_rgb(*upper),
                        bgcolor=Color.from_rgb(*lower),
                    ),
                )
            yield line


class ImagePreviewWidget(Static):
    """Textual widget that displays a cell-based image preview."""

    DEFAULT_CSS = """
    ImagePreviewWidget {
        height: 1fr;
        width: 100%;
        border: round $accent;
        padding: 1;
        overflow: hidden hidden;
    }
    """

    def __init__(
        self,
        image_path: Path | None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__("", name=name, id=id, classes=classes, markup=False)
        self.image_path = image_path
        self._image = _load_image(image_path) if image_path is not None else _make_sample_image()
        self.border_title = str(image_path) if image_path is not None else "Generated sample"

    def on_mount(self) -> None:
        self._render_preview()

    def on_resize(self, _event: Resize) -> None:
        self._render_preview()

    def reload(self) -> None:
        """Reload the source image when this widget was created from a path."""
        if self.image_path is not None:
            self._image = _load_image(self.image_path)
        self._render_preview()

    def _render_preview(self) -> None:
        size = self.content_size
        width = size.width or DEFAULT_WIDTH
        height = size.height or DEFAULT_HEIGHT
        self.update(CellImage(self._image, max_width=width, max_height=height))


class ImageCellPreviewApp(App[None]):
    """Small Textual app for inspecting the cell image renderer."""

    CSS = """
    Screen {
        background: $surface;
    }

    #preview-help {
        dock: top;
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reload", "Reload"),
    ]

    def __init__(self, image_path: Path | None) -> None:
        self.image_path = image_path
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(
            "Cell preview uses colored upper-half block glyphs. It is approximate, but works in plain terminals.",
            id="preview-help",
        )
        yield ImagePreviewWidget(self.image_path, id="preview")
        yield Footer()

    def action_reload(self) -> None:
        self.query_one(ImagePreviewWidget).reload()


def _load_image(path: Path) -> Image.Image:
    """Load and normalize an image for preview rendering."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, BACKGROUND)
            background.paste(rgba, mask=rgba.getchannel("A"))
            return background
        return image.convert("RGB")


def _fit_image(image: Image.Image, *, max_width: int, max_height: int) -> Image.Image:
    """Resize an image to fit inside a cell preview area."""
    width, height = image.size
    scale = min(max_width / width, max_height / height, 1.0)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    if new_size == image.size:
        return image.copy()
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _make_sample_image() -> Image.Image:
    """Generate a sample image when no file path is supplied."""
    width, height = 960, 540
    image = Image.new("RGB", (width, height), BACKGROUND)
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            r = int(42 + 150 * x / width)
            g = int(48 + 130 * y / height)
            b = int(78 + 115 * (x + y) / (width + height))
            pixels[x, y] = (r, g, b)

    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((52, 54, 908, 486), outline=(255, 255, 255, 70), width=3)
    draw.rectangle((90, 92, 420, 228), fill=(28, 33, 45, 190), outline=(255, 255, 255, 90), width=2)
    draw.rectangle((464, 92, 870, 228), fill=(12, 19, 31, 120), outline=(255, 255, 255, 70), width=2)
    draw.ellipse((110, 270, 310, 470), fill=(236, 107, 94, 210))
    draw.ellipse((248, 260, 490, 502), fill=(86, 190, 150, 190))
    draw.ellipse((610, 274, 820, 484), fill=(248, 198, 92, 185))
    draw.line((540, 290, 830, 126), fill=(255, 255, 255, 120), width=7)
    draw.line((540, 330, 830, 166), fill=(0, 0, 0, 100), width=4)
    draw.text((118, 126), "Chrys", fill=(255, 255, 255, 235))
    draw.text((118, 160), "cell image preview", fill=(217, 226, 238, 230))
    draw.text((500, 132), "Pillow -> Rich Text", fill=(255, 255, 255, 225))
    draw.text((500, 166), "two pixels per terminal cell", fill=(217, 226, 238, 220))
    return image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview an image using terminal cell colors.")
    parser.add_argument("image", nargs="?", type=Path, help="PNG, JPEG, or WebP image to preview")
    parser.add_argument("--print", action="store_true", help="Print the Rich renderable instead of starting Textual")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Preview width in terminal cells")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Preview height in terminal cells")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image: Image.Image
    image_path = args.image.expanduser().resolve() if args.image is not None else None
    if args.print:
        image = _load_image(image_path) if image_path is not None else _make_sample_image()
        Console().print(CellImage(image, max_width=args.width, max_height=args.height))
        return
    ImageCellPreviewApp(image_path).run()


if __name__ == "__main__":
    main()
