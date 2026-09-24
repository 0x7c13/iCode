# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Append-only trajectory event log: envelope, writer, reader, and helpers.

``session.json`` holds the conversation truth; ``<session>/trajectory/events.jsonl``
holds the execution truth — one content-minimized JSON event per line, written
through a single per-session writer with written-ack semantics. This package is
the foundation-tier contract: every module here depends on the standard library
and sibling foundation helpers only.
"""
