#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Profile the editor's stock-Textual whole-document Markdown highlighting.

This is a manual calibration tool, not a CI benchmark. Run it on each target
platform when reconsidering the editor's character/line degradation cutoffs.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from textual.highlight import highlight

from chrys.app.tui.widgets.editor.highlighter import MarkdownHighlightTheme

_MARKDOWN_SAMPLE = """# Editor profiling sample

Paragraph with **strong**, *emphasis*, `inline code`, and [a link](https://example.com).

```python
def example(value: int) -> str:
    return f"value={value}"
```

- one list item
- another list item

> quoted text with `code`

"""


def _representative_markdown(character_count: int) -> str:
    repeats = (character_count // len(_MARKDOWN_SAMPLE)) + 1
    return (_MARKDOWN_SAMPLE * repeats)[:character_count]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-kib", type=int, nargs="+", default=(8, 16, 32, 64, 128))
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.iterations <= 0 or any(size <= 0 for size in args.sizes_kib):
        raise SystemExit("sizes and iterations must be positive")

    sys.stdout.write("size_kib\tcharacters\tlines\tmedian_ms\tp95_ms\n")
    for size_kib in args.sizes_kib:
        source = _representative_markdown(size_kib * 1024)
        samples: list[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            highlight(f"{source}\n```", language="markdown", theme=MarkdownHighlightTheme)
            samples.append((time.perf_counter() - started) * 1000)
        sys.stdout.write(
            f"{size_kib}\t{len(source)}\t{source.count(chr(10)) + 1}\t"
            f"{statistics.median(samples):.2f}\t{_percentile(samples, 0.95):.2f}\n"
        )


if __name__ == "__main__":
    main()
