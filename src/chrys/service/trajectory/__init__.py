# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session-level trajectory recording: activation, lifecycle, deletion.

The foundation package owns the file format and the writer; this package
binds one writer to one live session (lease, recovery, coverage markers) and
provides the engine-facing helpers (turn registry, fork prelude, tombstone
deletion). Modules here are imported lazily by their callers — keep this
``__init__`` free of imports so ``service.state.store`` and this package can
depend on each other without a cycle.
"""
