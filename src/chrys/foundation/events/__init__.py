# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Event-driven communication system between frontend and backend."""

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Event

__all__ = ["Event", "EventBus"]
