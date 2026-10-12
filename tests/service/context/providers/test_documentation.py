# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Document routing, capability fallbacks and stable provider snapshots."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from chrys.kernel import AgentSession, Message, SessionContext
from chrys.service.context.providers import documentation as documentation_module
from chrys.service.context.providers.documentation import DocumentationProvider
from chrys.service.llm.chat_completions.client import OPENAI
from chrys.service.llm.chat_completions.request import build_request


@pytest.fixture
def product_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    docs = tmp_path / "product docs"
    docs.mkdir()
    (docs / "index.yaml").write_text("locales: [en]\ntopics: []\n", encoding="utf-8")
    (docs / "sentinel.md").write_text("DOCUMENT-BODY-SENTINEL", encoding="utf-8")
    monkeypatch.setenv("CHRYS_DOCS_ROOT", str(docs))
    return docs


async def test_exposes_absolute_routes_without_preloading_docs_or_adding_tools(product_docs: Path) -> None:
    provider = DocumentationProvider(file_read_available=True)
    context = SessionContext(input_messages=[])
    await provider.before_run(agent=None, session=AgentSession(), context=context, state={})

    text = context.instructions[0]
    assert str(product_docs) in text
    assert str(product_docs / "index.yaml") in text
    assert "read_file" in text
    assert "line_range" in text
    assert "DOCUMENT-BODY-SENTINEL" not in text
    assert context.tools == []


async def test_no_read_tool_keeps_routes_for_delegation(product_docs: Path) -> None:
    provider = DocumentationProvider(file_read_available=False)
    context = SessionContext(input_messages=[])
    await provider.before_run(agent=None, session=AgentSession(), context=context, state={})
    text = context.instructions[0]
    assert str(product_docs / "index.yaml") in text
    assert "no read_file tool" in text
    assert "delegate the lookup" in text
    assert "Otherwise explain" in text
    assert "DOCUMENT-BODY-SENTINEL" not in text
    assert context.tools == []


@pytest.mark.parametrize("file_read_available", [False, True])
async def test_surrogate_root_is_unavailable_and_wire_request_remains_encodable(
    monkeypatch: pytest.MonkeyPatch, file_read_available: bool
) -> None:
    root = Path("/installed/icode-\udcfc/docs")
    monkeypatch.setattr(documentation_module, "resolve_docs_root", lambda: root)
    provider = DocumentationProvider(file_read_available=file_read_available)
    context = SessionContext(input_messages=[])
    await provider.before_run(agent=None, session=AgentSession(), context=context, state={})
    text = context.instructions[0]
    assert "Product documentation could not be found" in text
    assert str(root) not in text
    assert "Topic index:" not in text
    request = build_request([Message("user", ["What is iCode?"])], {"instructions": text}, model="test", variant=OPENAI)
    wire = httpx.Request("POST", "https://example.invalid/chat/completions", json=request)
    assert "Product documentation could not be found" in wire.content.decode("utf-8")


@pytest.mark.parametrize("file_read_available", [False, True])
async def test_missing_capabilities_do_not_advertise_unusable_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_read_available: bool
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv("CHRYS_DOCS_ROOT", str(missing))
    provider = DocumentationProvider(file_read_available=file_read_available)
    context = SessionContext(input_messages=[])
    await provider.before_run(agent=None, session=AgentSession(), context=context, state={})

    text = context.instructions[0]
    assert str(missing) not in text
    assert "Product documentation could not be found" in text
    assert "/help" in text


async def test_documentation_route_stays_stable_until_runtime_rebuild(
    product_docs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = DocumentationProvider(file_read_available=True)
    monkeypatch.setenv("CHRYS_DOCS_ROOT", str(tmp_path / "missing"))
    contexts = [SessionContext(input_messages=[]), SessionContext(input_messages=[])]
    for context in contexts:
        await provider.before_run(agent=None, session=AgentSession(), context=context, state={})
    assert contexts[0].instructions == contexts[1].instructions
    assert str(product_docs) in contexts[0].instructions[0]
