# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Playground: demo for DiffView widget with multiple test cases.

Usage:
    uv run python playground/diff_view.py

Keybindings:
    n / Shift+n      — cycle through diff cases
    Space            — toggle split / unified view
    a                — toggle line annotations
    q                — quit
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Label

from chrys.app.tui.theme import ANSI_THEMES, CHRYS_ANSI_THEME, CHRYS_THEME, CHRYS_THEMES, TuiVariableDefaultsMixin
from chrys.app.tui.widgets.diff_view import DiffView

# ── Case 1: Python function refactoring ──────────────────────────────────────

CASE_PY_BEFORE = """\
def loop_first(values):
    iter_values = iter(values)
    try:
        value = next(iter_values)
    except StopIteration:
        return
    yield True, value
    for value in iter_values:
        yield False, value


def loop_first_last(values):
    \"\"\"Iterate and generate a tuple with a flag for first and last value.\"\"\"
    iter_values = iter(values)
    try:
        previous_value = next(iter_values)
    except StopIteration:
        return
    first = True
    for value in iter_values:
        yield first, False, previous_value
        first = False
        previous_value = value
    yield first, True, previous_value
"""

CASE_PY_AFTER = """\
from typing import Iterable, TypeVar

T = TypeVar("T")


def loop_first(values: Iterable[T]) -> Iterable[tuple[bool, T]]:
    \"\"\"Iterate and generate a tuple with a flag for first value.

    Args:
        values: Iterable of values.

    Returns:
        Iterable of (is_first, value) tuples.
    \"\"\"
    iter_values = iter(values)
    try:
        value = next(iter_values)
    except StopIteration:
        return
    yield True, value
    for value in iter_values:
        yield False, value


def loop_last(values: Iterable[T]) -> Iterable[tuple[bool, T]]:
    \"\"\"Iterate and generate a tuple with a flag for last value.\"\"\"
    iter_values = iter(values)
    try:
        previous_value = next(iter_values)
    except StopIteration:
        return
    for value in iter_values:
        yield False, previous_value
        previous_value = value
    yield True, previous_value


def loop_first_last(values: Iterable[T]) -> Iterable[tuple[bool, bool, T]]:
    \"\"\"Iterate and generate a tuple with a flag for first and last value.\"\"\"
    iter_values = iter(values)
    try:
        previous_value = next(iter_values)
    except StopIteration:
        return
    first = True
    for value in iter_values:
        yield first, False, previous_value
        first = False
        previous_value = value
    yield first, True, previous_value
"""

# ── Case 2: Config file change (pyproject.toml) ──────────────────────────────

CASE_CFG_BEFORE = """\
[project]
name = "my-app"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "httpx>=0.24",
    "pydantic>=1.10",
]

[tool.ruff]
line-length = 79
select = ["E", "F"]

[tool.mypy]
ignore_missing_imports = true
"""

CASE_CFG_AFTER = """\
[project]
name = "my-app"
version = "2.0.0"
description = "A production-ready web service"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.6",
    "anyio>=4.3",
]

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
addopts = "--tb=short -q"
"""

# ── Case 3: New file (all additions) ─────────────────────────────────────────

CASE_NEW_BEFORE = ""

CASE_NEW_AFTER = """\
\"\"\"Health check endpoint.\"\"\"
from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

router = APIRouter()


@router.get("/healthz", tags=["ops"])
async def healthz(db: AsyncSession) -> dict:
    \"\"\"Return service health status.\"\"\"
    await db.execute(text("SELECT 1"))
    return {"status": "ok", "db": "ok"}
"""

# ── Case 4: Deleted file (all removals) ──────────────────────────────────────

CASE_DEL_BEFORE = """\
# legacy_auth.py — DEPRECATED, use auth.oauth2 instead

import hashlib
import base64


def md5_password(password: str) -> str:
    \"\"\"Hash password with MD5. DO NOT USE in production.\"\"\"
    return hashlib.md5(password.encode()).hexdigest()


def basic_auth_header(username: str, password: str) -> str:
    \"\"\"Build a Basic Auth header value.\"\"\"
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"
"""

CASE_DEL_AFTER = ""

# ── Case 5: Inline rename + signature change ─────────────────────────────────

CASE_RENAME_BEFORE = """\
class UserManager:
    def get_user(self, id: int):
        return self.db.query(User).filter(User.id == id).first()

    def create_user(self, name: str, email: str, pwd: str):
        user = User(name=name, email=email, password=self._hash(pwd))
        self.db.add(user)
        self.db.commit()
        return user

    def delete_user(self, id: int) -> bool:
        user = self.get_user(id)
        if not user:
            return False
        self.db.delete(user)
        self.db.commit()
        return True

    def _hash(self, pwd: str) -> str:
        import hashlib
        return hashlib.sha256(pwd.encode()).hexdigest()
"""

CASE_RENAME_AFTER = """\
class UserService:
    async def get_user(self, user_id: int) -> User | None:
        return await self.db.get(User, user_id)

    async def create_user(
        self, *, name: str, email: str, password: str
    ) -> User:
        user = User(
            name=name,
            email=email,
            hashed_password=self._hash_password(password),
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def delete_user(self, user_id: int) -> bool:
        user = await self.get_user(user_id)
        if not user:
            return False
        await self.db.delete(user)
        await self.db.commit()
        return True

    def _hash_password(self, password: str) -> str:
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest()
"""

# ── Case 6: CJK & double-width emoji ─────────────────────────────────────────

CASE_CJK_BEFORE = """\
# 用户管理模块
# User management module

GREETING = "你好世界 👋"
FAREWELL = "再见 🌍"

class 用户管理:
    \"\"\"管理系统中的用户 📋\"\"\"

    def 获取用户(self, 用户ID: int) -> dict:
        \"\"\"根据ID获取用户信息\"\"\"
        return {"名前": "田中太郎", "メール": "tanaka@example.com"}

    def 创建用户(self, 名前: str, メール: str) -> dict:
        \"\"\"创建新用户 🆕\"\"\"
        用户 = {
            "名前": 名前,
            "メール": メール,
            "状态": "活跃 ✅",
        }
        return 用户

    def 显示表格(self) -> str:
        \"\"\"显示用户表格\"\"\"
        header = "| 姓名     | 邮箱              | 状态  |"
        sep    = "|----------|-------------------|-------|"
        row1   = "| 田中太郎 | tanaka@test.com   | 活跃 ✅ |"
        row2   = "| 鈴木花子 | suzuki@test.com   | 停用 ❌ |"
        return "\\n".join([header, sep, row1, row2])
"""

CASE_CJK_AFTER = """\
# 用户服务模块（重构版）
# User service module (refactored)

GREETING = "你好世界 👋🎉"
FAREWELL = "再见朋友们 🌍✨"
VERSION = "2.0 🚀"

class 用户服务:
    \"\"\"管理系统中的用户 📋 — 异步重构版\"\"\"

    async def 获取用户(self, 用户ID: int) -> dict | None:
        \"\"\"根据ID获取用户信息（异步）\"\"\"
        return {"名前": "田中太郎", "メール": "tanaka@example.com", "角色": "管理者 👑"}

    async def 创建用户(self, 名前: str, メール: str, 角色: str = "普通用户") -> dict:
        \"\"\"创建新用户 🆕✨\"\"\"
        用户 = {
            "名前": 名前,
            "メール": メール,
            "角色": 角色,
            "状态": "活跃 ✅",
            "创建时间": "2024-01-01 🕐",
        }
        return 用户

    async def 删除用户(self, 用户ID: int) -> bool:
        \"\"\"删除用户 🗑\"\"\"
        return True

    def 显示表格(self) -> str:
        \"\"\"显示用户表格 📊\"\"\"
        header = "| 姓名     | 邮箱              | 角色       | 状态  |"
        sep    = "|----------|-------------------|------------|-------|"
        row1   = "| 田中太郎 | tanaka@test.com   | 管理者 👑  | 活跃 ✅ |"
        row2   = "| 鈴木花子 | suzuki@test.com   | 编辑者 ✏  | 活跃 ✅ |"
        row3   = "| 佐藤健   | sato@test.com     | 普通用户 👤 | 停用 ❌ |"
        return "\\n".join([header, sep, row1, row2, row3])
"""

# ── Case 7: Large file — stress test (~600 lines, many scattered changes) ────


def _gen_large_case() -> tuple[str, str]:
    """Generate a large before/after pair for scroll performance testing."""
    before: list[str] = []
    after: list[str] = []

    # --- header ---
    before.append('"""API handlers for the task management service."""')
    before.append("import logging")
    before.append("from datetime import datetime")
    before.append("from typing import Optional, List")
    before.append("")
    before.append("from flask import Flask, request, jsonify, abort")
    before.append("from sqlalchemy.orm import Session")
    before.append("")
    before.append("from .models import Task, User, Project, Tag, Comment")
    before.append("from .database import get_db")
    before.append("from .auth import require_auth")
    before.append("")
    before.append("logger = logging.getLogger(__name__)")
    before.append("app = Flask(__name__)")
    before.append("")

    after.append('"""API handlers for the task management service (async rewrite)."""')
    after.append("import logging")
    after.append("import uuid")
    after.append("from datetime import datetime, timezone")
    after.append("from typing import Annotated")
    after.append("")
    after.append("from fastapi import FastAPI, Depends, HTTPException, Query, Path, status")
    after.append("from sqlalchemy.ext.asyncio import AsyncSession")
    after.append("")
    after.append("from .models import Task, User, Project, Tag, Comment, AuditLog")
    after.append("from .database import get_async_db")
    after.append("from .auth import require_auth, get_current_user")
    after.append("from .schemas import (")
    after.append("    TaskCreate, TaskUpdate, TaskResponse, TaskListResponse,")
    after.append("    ProjectCreate, ProjectResponse,")
    after.append("    CommentCreate, CommentResponse,")
    after.append(")")
    after.append("")
    after.append("logger = logging.getLogger(__name__)")
    after.append("app = FastAPI(title='Task Service', version='2.0.0')")
    after.append("")

    # --- generate many endpoint pairs ---
    resources = [
        ("task", "Task"),
        ("project", "Project"),
        ("user", "User"),
        ("tag", "Tag"),
        ("comment", "Comment"),
        ("milestone", "Milestone"),
        ("attachment", "Attachment"),
        ("notification", "Notification"),
        ("webhook", "Webhook"),
        ("label", "Label"),
    ]

    for res_name, res_class in resources:
        plural = res_name + "s"

        # ---- LIST endpoint (before) ----
        before.append(f"# \u2500\u2500 {res_class} endpoints \u2500\u2500")
        before.append("")
        before.append(f'@app.route("/{plural}", methods=["GET"])')
        before.append("@require_auth")
        before.append(f"def list_{plural}():")
        before.append(f'    """List all {plural}."""')
        before.append("    db = get_db()")
        before.append("    page = request.args.get('page', 1, type=int)")
        before.append("    per_page = request.args.get('per_page', 20, type=int)")
        before.append(f"    query = db.query({res_class})")
        before.append("    if request.args.get('search'):")
        before.append("        search = request.args['search']")
        before.append(f"        query = query.filter({res_class}.name.ilike(f'%{{search}}%'))")
        before.append("    total = query.count()")
        before.append("    items = query.offset((page - 1) * per_page).limit(per_page).all()")
        before.append("    return jsonify({")
        before.append('        "items": [item.to_dict() for item in items],')
        before.append('        "total": total,')
        before.append('        "page": page,')
        before.append("    })")
        before.append("")

        # ---- LIST endpoint (after) ----
        after.append(f"# \u2500\u2500 {res_class} endpoints \u2500\u2500")
        after.append("")
        after.append(f'@app.get("/{plural}", response_model={res_class}ListResponse)')
        after.append(f"async def list_{plural}(")
        after.append("    db: Annotated[AsyncSession, Depends(get_async_db)],")
        after.append("    current_user: Annotated[User, Depends(get_current_user)],")
        after.append("    page: int = Query(default=1, ge=1),")
        after.append("    per_page: int = Query(default=20, ge=1, le=100),")
        after.append("    search: str | None = Query(default=None, max_length=200),")
        after.append(f") -> {res_class}ListResponse:")
        after.append(f'    """List all {plural} with pagination and search."""')
        after.append(f"    query = select({res_class}).where({res_class}.org_id == current_user.org_id)")
        after.append("    if search:")
        after.append(f"        query = query.where({res_class}.name.ilike(f'%{{search}}%'))")
        after.append("    total = await db.scalar(select(func.count()).select_from(query.subquery()))")
        after.append("    result = await db.execute(query.offset((page - 1) * per_page).limit(per_page))")
        after.append("    items = result.scalars().all()")
        after.append(f"    logger.info('Listed %d {plural} for user %s', len(items), current_user.id)")
        after.append(f"    return {res_class}ListResponse(items=items, total=total, page=page)")
        after.append("")

        # ---- GET endpoint (before) ----
        before.append(f'@app.route("/{plural}/<int:{res_name}_id>", methods=["GET"])')
        before.append("@require_auth")
        before.append(f"def get_{res_name}({res_name}_id):")
        before.append(f'    """Get a single {res_name}."""')
        before.append("    db = get_db()")
        before.append(f"    item = db.query({res_class}).get({res_name}_id)")
        before.append("    if not item:")
        before.append("        abort(404)")
        before.append("    return jsonify(item.to_dict())")
        before.append("")

        # ---- GET endpoint (after) ----
        after.append(f'@app.get("/{plural}/{{{res_name}_id}}", response_model={res_class}Response)')
        after.append(f"async def get_{res_name}(")
        after.append(f"    {res_name}_id: Annotated[int, Path(ge=1)],")
        after.append("    db: Annotated[AsyncSession, Depends(get_async_db)],")
        after.append("    current_user: Annotated[User, Depends(get_current_user)],")
        after.append(f") -> {res_class}Response:")
        after.append(f'    """Get a single {res_name} by ID."""')
        after.append(f"    item = await db.get({res_class}, {res_name}_id)")
        after.append("    if not item or item.org_id != current_user.org_id:")
        after.append(f"        raise HTTPException(status_code=404, detail='{res_class} not found')")
        after.append("    return item")
        after.append("")

        # ---- CREATE endpoint (before) ----
        before.append(f'@app.route("/{plural}", methods=["POST"])')
        before.append("@require_auth")
        before.append(f"def create_{res_name}():")
        before.append(f'    """Create a new {res_name}."""')
        before.append("    db = get_db()")
        before.append("    data = request.get_json()")
        before.append("    if not data or 'name' not in data:")
        before.append("        abort(400)")
        before.append(f"    item = {res_class}(")
        before.append("        name=data['name'],")
        before.append("        description=data.get('description', ''),")
        before.append("        created_at=datetime.utcnow(),")
        before.append("    )")
        before.append("    db.add(item)")
        before.append("    db.commit()")
        before.append("    return jsonify(item.to_dict()), 201")
        before.append("")

        # ---- CREATE endpoint (after) ----
        after.append(f'@app.post("/{plural}", response_model={res_class}Response, status_code=status.HTTP_201_CREATED)')
        after.append(f"async def create_{res_name}(")
        after.append(f"    body: {res_class}Create,")
        after.append("    db: Annotated[AsyncSession, Depends(get_async_db)],")
        after.append("    current_user: Annotated[User, Depends(get_current_user)],")
        after.append(f") -> {res_class}Response:")
        after.append(f'    """Create a new {res_name}."""')
        after.append(f"    item = {res_class}(")
        after.append("        **body.model_dump(),")
        after.append("        org_id=current_user.org_id,")
        after.append("        created_by=current_user.id,")
        after.append("        created_at=datetime.now(timezone.utc),")
        after.append("    )")
        after.append("    db.add(item)")
        after.append("    await db.commit()")
        after.append("    await db.refresh(item)")
        after.append(f"    logger.info('Created {res_name} %s by user %s', item.id, current_user.id)")
        after.append("    return item")
        after.append("")

        # ---- UPDATE endpoint (before) ----
        before.append(f'@app.route("/{plural}/<int:{res_name}_id>", methods=["PUT"])')
        before.append("@require_auth")
        before.append(f"def update_{res_name}({res_name}_id):")
        before.append(f'    """Update an existing {res_name}."""')
        before.append("    db = get_db()")
        before.append(f"    item = db.query({res_class}).get({res_name}_id)")
        before.append("    if not item:")
        before.append("        abort(404)")
        before.append("    data = request.get_json()")
        before.append("    if 'name' in data:")
        before.append("        item.name = data['name']")
        before.append("    if 'description' in data:")
        before.append("        item.description = data['description']")
        before.append("    item.updated_at = datetime.utcnow()")
        before.append("    db.commit()")
        before.append("    return jsonify(item.to_dict())")
        before.append("")

        # ---- UPDATE endpoint (after) ----
        after.append(f'@app.put("/{plural}/{{{res_name}_id}}", response_model={res_class}Response)')
        after.append(f"async def update_{res_name}(")
        after.append(f"    {res_name}_id: Annotated[int, Path(ge=1)],")
        after.append(f"    body: {res_class}Update,")
        after.append("    db: Annotated[AsyncSession, Depends(get_async_db)],")
        after.append("    current_user: Annotated[User, Depends(get_current_user)],")
        after.append(f") -> {res_class}Response:")
        after.append(f'    """Update an existing {res_name}."""')
        after.append(f"    item = await db.get({res_class}, {res_name}_id)")
        after.append("    if not item or item.org_id != current_user.org_id:")
        after.append(f"        raise HTTPException(status_code=404, detail='{res_class} not found')")
        after.append("    update_data = body.model_dump(exclude_unset=True)")
        after.append("    for field, value in update_data.items():")
        after.append("        setattr(item, field, value)")
        after.append("    item.updated_at = datetime.now(timezone.utc)")
        after.append("    item.updated_by = current_user.id")
        after.append("    await db.commit()")
        after.append("    await db.refresh(item)")
        after.append("    return item")
        after.append("")

        # ---- DELETE endpoint (before) ----
        before.append(f'@app.route("/{plural}/<int:{res_name}_id>", methods=["DELETE"])')
        before.append("@require_auth")
        before.append(f"def delete_{res_name}({res_name}_id):")
        before.append(f'    """Delete a {res_name}."""')
        before.append("    db = get_db()")
        before.append(f"    item = db.query({res_class}).get({res_name}_id)")
        before.append("    if not item:")
        before.append("        abort(404)")
        before.append("    db.delete(item)")
        before.append("    db.commit()")
        before.append('    return "", 204')
        before.append("")

        # ---- DELETE endpoint (after) ----
        after.append(f'@app.delete("/{plural}/{{{res_name}_id}}", status_code=status.HTTP_204_NO_CONTENT)')
        after.append(f"async def delete_{res_name}(")
        after.append(f"    {res_name}_id: Annotated[int, Path(ge=1)],")
        after.append("    db: Annotated[AsyncSession, Depends(get_async_db)],")
        after.append("    current_user: Annotated[User, Depends(get_current_user)],")
        after.append(") -> None:")
        after.append(f'    """Soft-delete a {res_name}."""')
        after.append(f"    item = await db.get({res_class}, {res_name}_id)")
        after.append("    if not item or item.org_id != current_user.org_id:")
        after.append(f"        raise HTTPException(status_code=404, detail='{res_class} not found')")
        after.append("    item.deleted_at = datetime.now(timezone.utc)")
        after.append("    item.deleted_by = current_user.id")
        after.append("    await db.commit()")
        after.append(f"    logger.info('Deleted {res_name} %s by user %s', {res_name}_id, current_user.id)")
        after.append("")

        # Add identical "unchanged" padding between resource blocks so that
        # difflib produces multiple groups separated by ellipsis lines.
        pad = [
            "",
            f"# {'\u2500' * 70}",
            f"# Internal helpers for {res_class}",
            f"# {'\u2500' * 70}",
            "",
            f"def _validate_{res_name}(data: dict) -> bool:",
            f'    """Validate {res_name} data before persistence."""',
            "    if not data.get('name'):",
            "        return False",
            "    if len(data['name']) > 255:",
            "        return False",
            "    return True",
            "",
            "",
        ]
        before.extend(pad)
        after.extend(pad)

    # --- footer ---
    before.append("")
    before.append("# \u2500\u2500 Error handlers \u2500\u2500")
    before.append("")
    before.append("@app.errorhandler(404)")
    before.append("def not_found(e):")
    before.append('    return jsonify({"error": "Not found"}), 404')
    before.append("")
    before.append("@app.errorhandler(400)")
    before.append("def bad_request(e):")
    before.append('    return jsonify({"error": "Bad request"}), 400')
    before.append("")

    after.append("")
    after.append("# \u2500\u2500 Exception handlers \u2500\u2500")
    after.append("")
    after.append("@app.exception_handler(Exception)")
    after.append("async def global_exception_handler(request, exc):")
    after.append("    logger.exception('Unhandled error: %s', exc)")
    after.append("    return JSONResponse(")
    after.append("        status_code=500,")
    after.append('        content={"detail": "Internal server error"},')
    after.append("    )")
    after.append("")
    after.append("")
    after.append("@app.on_event('startup')")
    after.append("async def startup():")
    after.append("    logger.info('Task service v2.0 starting up')")
    after.append("")

    return "\n".join(before), "\n".join(after)


CASE_LARGE_BEFORE, CASE_LARGE_AFTER = _gen_large_case()

# ─────────────────────────────────────────────────────────────────────────────

CASES: list[tuple[str, str, str, str, str]] = [
    (
        "1  Python \u2014 type hints & refactor",
        "_loop.py",
        "_loop.py",
        CASE_PY_BEFORE,
        CASE_PY_AFTER,
    ),
    (
        "2  Config \u2014 pyproject.toml upgrade",
        "pyproject.toml",
        "pyproject.toml",
        CASE_CFG_BEFORE,
        CASE_CFG_AFTER,
    ),
    (
        "3  New file \u2014 all additions",
        "/dev/null",
        "health.py",
        CASE_NEW_BEFORE,
        CASE_NEW_AFTER,
    ),
    (
        "4  Deleted file \u2014 all removals",
        "legacy_auth.py",
        "/dev/null",
        CASE_DEL_BEFORE,
        CASE_DEL_AFTER,
    ),
    (
        "5  Rename + async migration",
        "user_manager.py",
        "user_service.py",
        CASE_RENAME_BEFORE,
        CASE_RENAME_AFTER,
    ),
    (
        "6  CJK & double-width emoji",
        "user_mgmt.py",
        "user_service.py",
        CASE_CJK_BEFORE,
        CASE_CJK_AFTER,
    ),
    (
        "7  Large file \u2014 Flask\u2192FastAPI migration (~600 lines)",
        "handlers.py",
        "handlers.py",
        CASE_LARGE_BEFORE,
        CASE_LARGE_AFTER,
    ),
]


class DiffDemoApp(TuiVariableDefaultsMixin, App):
    CSS = """
    Screen {
        background: $background;
        layout: vertical;
    }
    #case-label {
        background: $primary 20%;
        color: $text;
        padding: 0 2;
        height: 1;
        dock: top;
    }
    #diff {
        height: 1fr;
    }
    #diff .diff-group {
        height: 1fr;
    }
    .-chrys Screen {
        background: transparent;
    }
    .-chrys Header {
        background: transparent;
    }
    .-chrys DiffView {
        background: transparent;
    }
    .-chrys .diff-group {
        background: transparent;
    }
    .-chrys CodeColumn {
        background: transparent;
    }
    .-chrys GutterColumn {
        background: transparent;
    }
    .-chrys Footer {
        background: #21222C;
        color: #F8F8F2;
    }
    .-chrys .footer-key--key {
        background: transparent;
        color: #BD93F9;
    }
    .-chrys .footer-key--description {
        background: transparent;
        color: #CFCFCF;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("n", "next_case", "Next case"),
        Binding("shift+n", "prev_case", "Prev case"),
        Binding("space", "toggle_split", "Split/Unified"),
        Binding("a", "toggle_annotations", "Annotations"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.register_theme(CHRYS_THEME)
        self.register_theme(CHRYS_ANSI_THEME)
        self.theme = "chrys"
        self._case_index = 0

    def _watch_theme(self, theme_name: str) -> None:
        """Enable ANSI color passthrough for transparent-background themes."""
        super()._watch_theme(theme_name)
        is_ansi = theme_name in ANSI_THEMES
        if is_ansi:
            self.ansi_color = True
        self.set_class(theme_name in CHRYS_THEMES, "-chrys")
        self.set_class(is_ansi and theme_name in CHRYS_THEMES, "-chrys-ansi")

    def _current_case(self) -> tuple[str, str, str, str, str]:
        return CASES[self._case_index]

    def compose(self) -> ComposeResult:
        label, path_before, path_after, before, after = self._current_case()
        yield Header()
        yield Label(f"  Case {label}", id="case-label")
        yield DiffView(path_before, path_after, before, after, id="diff")
        yield Footer()

    async def on_mount(self) -> None:
        diff = self.query_one("#diff", DiffView)
        await diff.prepare()

    async def _reload_case(self) -> None:
        label, path_before, path_after, before, after = self._current_case()
        self.query_one("#case-label", Label).update(f"  Case {label}")
        await self.query_one("#diff").remove()
        new_diff = DiffView(path_before, path_after, before, after, id="diff")
        await new_diff.prepare()
        await self.mount(new_diff, before=self.query_one(Footer))

    async def action_next_case(self) -> None:
        self._case_index = (self._case_index + 1) % len(CASES)
        await self._reload_case()

    async def action_prev_case(self) -> None:
        self._case_index = (self._case_index - 1) % len(CASES)
        await self._reload_case()

    def action_toggle_split(self) -> None:
        diff = self.query_one("#diff", DiffView)
        diff.split = not diff.split

    def action_toggle_annotations(self) -> None:
        diff = self.query_one("#diff", DiffView)
        diff.annotations = not diff.annotations


if __name__ == "__main__":
    DiffDemoApp().run()
