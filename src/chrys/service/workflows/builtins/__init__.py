# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow templates shipped with chrys.

Every ``*.py`` file here is a workflow the discovery layer lists under the
``builtin`` source kind, shadowed by a user file of the same id. Templates are
standard-library only and run on the default interpreter: a read-only install
location cannot depend on anything the user's machine happens to provide. A
``<id>.manifest.json`` sibling is the template's pre-generated manifest, so
listings show its metadata without executing the file. Templates carry no
copyright header: users copy them as the start of their own workflows.
"""
