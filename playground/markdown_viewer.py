# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Playground: live test for VirtualizedMarkdownViewer with all markdown formats.

Usage:
    uv run python playground/markdown_viewer.py

Keybindings:
    s — Switch to streaming demo
    r — Reset to the full static document
    c — Toggle copy buttons on code fences
    q — Quit
"""

import asyncio

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer

from chrys.app.tui.widgets.markdown.copyable import CopyableMarkdown
from chrys.app.tui.widgets.markdown.stream import MarkdownStream
from chrys.app.tui.widgets.markdown.toc import MarkdownTableOfContents
from chrys.app.tui.widgets.markdown.viewer import VirtualizedMarkdownViewer
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown

MARKDOWN = """\
# The Developer's Handbook

> A comprehensive guide to modern software development practices, tools, and philosophies.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Version Control](#version-control)
3. [Code Quality](#code-quality)
4. [Testing Strategies](#testing-strategies)
5. [Performance](#performance)
6. [Security](#security)
7. [Deployment](#deployment)

---

## Introduction

Software development is as much an art as it is a science. The best engineers balance
**pragmatism** with **craftsmanship**, shipping working software while maintaining a
codebase that can evolve gracefully over time.

This handbook distills lessons learned across thousands of codebases, countless
production incidents, and decades of collective engineering wisdom.

---

## Version Control

### The Golden Rules of Git

Never rewrite shared history. Once a commit is on `main`, treat it as immutable.
Force-pushing to a shared branch is the quickest way to lose colleagues' trust.

**Commit message format:**

```
<type>(<scope>): <short summary>

<body — wrap at 72 chars>

<footer — breaking changes, issue refs>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

### Branching Strategy

A simple trunk-based workflow for small teams:

```bash
# Create a feature branch
git switch -c feat/add-user-auth

# Keep it up to date
git fetch origin
git rebase origin/main

# Merge via PR — never directly to main
gh pr create --fill
```

For larger teams, consider **GitFlow** or **GitHub Flow** depending on your release cadence.

### Useful Aliases

```gitconfig
[alias]
    lg = log --oneline --graph --decorate --all
    undo = reset HEAD~1 --mixed
    wip = commit -am "WIP: checkpoint"
    st = status -sb
```

---

## Code Quality

### The Boy Scout Rule

> Leave the code cleaner than you found it.

This doesn't mean refactoring entire modules on every PR. It means fixing the obvious
typo in a comment, extracting a magic number into a named constant, or renaming a
confusing variable while you're already in the file.

### Static Analysis

Automate as much quality enforcement as possible. Humans should review logic and
architecture — not style.

**Python stack:**

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
addopts = "--tb=short -q"
```

**Run everything with one command:**

```bash
uvx ruff check . && uvx mypy . && pytest
```

### Design Principles

| Principle | Description |
|-----------|-------------|
| **SRP** | A class should have one reason to change |
| **OCP** | Open for extension, closed for modification |
| **DIP** | Depend on abstractions, not concretions |
| **YAGNI** | Don't build what you don't need yet |
| **DRY** | Every piece of knowledge has one authoritative source |

---

## Testing Strategies

### The Testing Pyramid

```
        /\\
       /  \\
      / E2E\\        <- Few, slow, expensive
     /------\\
    /  Integ. \\     <- Some, moderate speed
   /------------\\
  /     Unit     \\  <- Many, fast, cheap
 /----------------\\
```

Write the bulk of your tests at the unit level. They're fast, focused, and give precise
failure messages. Integration tests verify that components work together. E2E tests
validate critical user journeys.

### Writing Good Tests

A test should be **FIRST**:

- **Fast** — milliseconds, not seconds
- **Isolated** — no shared mutable state between tests
- **Repeatable** — same result every time, no flakiness
- **Self-validating** — pass or fail, never "check the output"
- **Timely** — written alongside (or before) the code

**Example — testing a pure function:**

```python
import pytest
from myapp.utils import slugify

@pytest.mark.parametrize("input,expected", [
    ("Hello World", "hello-world"),
    ("  leading spaces  ", "leading-spaces"),
    ("Special $#@ chars!", "special-chars"),
    ("", ""),
])
def test_slugify(input: str, expected: str) -> None:
    assert slugify(input) == expected
```

### Mocking External Dependencies

```python
from unittest.mock import AsyncMock, patch

async def test_fetch_user_returns_none_on_404(http_client):
    with patch.object(http_client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value.status_code = 404
        result = await fetch_user(http_client, user_id=42)
    assert result is None
```

---

## Performance

### Measure First, Optimize Second

> Premature optimization is the root of all evil. — Donald Knuth

Profile before assuming where the bottleneck is. Use `py-spy`, `cProfile`, or
`austin` for CPU profiling. Use `memory_profiler` or `memray` for memory.

```bash
# Flame graph with py-spy
py-spy record -o profile.svg -- python myapp/server.py

# Memory snapshot with memray
python -m memray run -o output.bin myapp/server.py
python -m memray flamegraph output.bin
```

### Common Python Performance Patterns

```python
# Prefer list comprehensions over map/filter for clarity + speed
squares = [x * x for x in range(1000)]

# Use generators for large sequences — avoid materializing the whole list
def read_chunks(path: str, size: int = 8192):
    with open(path, "rb") as f:
        while chunk := f.read(size):
            yield chunk

# Cache expensive pure computations
from functools import lru_cache

@lru_cache(maxsize=256)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

---

## Security

### The OWASP Top 10 (Abbreviated)

1. **Broken Access Control** — enforce authz on every request, server-side
2. **Cryptographic Failures** — use `bcrypt`/`argon2` for passwords, TLS everywhere
3. **Injection** — parameterize queries, never interpolate user input into SQL
4. **Insecure Design** — threat-model features before building them
5. **Security Misconfiguration** — disable debug mode in production, rotate secrets

### Secrets Management

```bash
# Never hardcode secrets. Use environment variables or a secrets manager.
export DATABASE_URL="postgresql://user:pass@localhost/db"

# Scan your repo for accidentally committed secrets
pip install detect-secrets
detect-secrets scan > .secrets.baseline
```

```python
# Load config from environment with validation
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    debug: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Deployment

### The Twelve-Factor App

Build apps that are **stateless**, **config-from-env**, and **disposable**. The
[12factor.net](https://12factor.net) methodology has stood the test of time.

### Docker Best Practices

```dockerfile
# Multi-stage build — keep the final image lean
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv .venv
COPY src/ src/
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["python", "-m", "myapp"]
```

### Health Checks

Every service should expose a `/healthz` endpoint. Load balancers and orchestrators
use it to know whether to send traffic.

```python
@app.get("/healthz")
async def healthz(db: AsyncSession = Depends(get_db)) -> dict:
    await db.execute(text("SELECT 1"))
    return {"status": "ok"}
```

---

## Closing Thoughts

The best code is code that solves the problem simply, is easy to delete when the
requirements change, and makes the next engineer's life easier — not harder.

_Write code for humans first, computers second._

---

*Last updated: February 2026 — contributions welcome via pull request.*
"""

STREAM_CHUNKS = [
    "# Streaming Demo\n\n",
    "This content is being **streamed** in real-time ",
    "using `MarkdownStream`.\n\n",
    "## Features\n\n",
    "- Fragments accumulate if rendering can't keep up\n",
    "- Uses `asyncio` under the hood\n",
    "- Backed by `VirtualizedMarkdown`\n\n",
    "## Code Example\n\n",
    "```python\nstream = VirtualizedMarkdown.get_stream(widget)\n",
    "await stream.write('Hello, **world**!')\n",
    "await stream.stop()\n```\n\n",
    "### A Table Built Incrementally\n\n",
    "| Step | Action         | Status   |\n",
    "|------|----------------|----------|\n",
    "| 1    | Parse tokens   | Done     |\n",
    "| 2    | Layout blocks  | Done     |\n",
    "| 3    | Render strips  | Done     |\n\n",
    "> Streaming complete!\n",
]


class MarkdownPlayground(App):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "stream_demo", "Stream Demo"),
        Binding("r", "reset", "Reset"),
        Binding("c", "toggle_copy", "Toggle Copy Buttons"),
    ]

    _copyable: bool = False

    def compose(self) -> ComposeResult:
        yield VirtualizedMarkdownViewer(MARKDOWN, show_table_of_contents=True)
        yield Footer()

    async def action_stream_demo(self) -> None:
        """Replace content with a streaming demo."""
        md_widget = self.query_one(VirtualizedMarkdown)
        await md_widget.update("")
        stream: MarkdownStream = VirtualizedMarkdown.get_stream(md_widget)
        for chunk in STREAM_CHUNKS:
            await stream.write(chunk)
            await asyncio.sleep(0.15)
        await stream.stop()

    async def action_reset(self) -> None:
        """Restore the original document."""
        md_widget = self.query_one(VirtualizedMarkdown)
        await md_widget.update(MARKDOWN)

    async def action_toggle_copy(self) -> None:
        """Swap the inner markdown widget between standard and copyable."""
        self._copyable = not self._copyable
        viewer = self.query_one(VirtualizedMarkdownViewer)

        # Remove the current inner markdown widget
        old_md = viewer.query_one(VirtualizedMarkdown)
        await old_md.remove()

        # Mount the replacement
        toc = viewer.query_one(MarkdownTableOfContents)
        new_md = CopyableMarkdown(MARKDOWN) if self._copyable else VirtualizedMarkdown()
        await viewer.mount(new_md, before=toc)
        await new_md.update(MARKDOWN)
        self.notify("Copy buttons " + ("ON" if self._copyable else "OFF"))


if __name__ == "__main__":
    MarkdownPlayground().run()
