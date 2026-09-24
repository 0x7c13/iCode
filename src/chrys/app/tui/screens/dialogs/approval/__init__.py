# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Approval dialog package."""

from chrys.app.tui.screens.dialogs.approval.body import ApprovalBody, ApprovalBypass, create_approval_body
from chrys.app.tui.screens.dialogs.approval.dialog import _DETAIL_KEYS, ApprovalDialog, _build_detail

__all__ = [
    "_DETAIL_KEYS",
    "ApprovalBody",
    "ApprovalBypass",
    "ApprovalDialog",
    "_build_detail",
    "create_approval_body",
]
