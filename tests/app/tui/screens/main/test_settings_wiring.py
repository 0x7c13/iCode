# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""How MainScreen wires the settings queue to the panel coordinator."""

from __future__ import annotations

from types import SimpleNamespace

from chrys.app.tui.screens.main import screen as screen_module
from chrys.app.tui.screens.main.settings_coordinator import SettingsCoordinator
from chrys.foundation.config.settings_store import PersistResult


class _Coordinator(SettingsCoordinator):
    """A coordinator double: records the feedback the screen forwards."""

    def __init__(self) -> None:  # bypasses the real ctor on purpose
        self.written: list[PersistResult] = []
        self.reloaded = 0
        self.failed = 0
        self.projected: dict[str, object] = {}

    def on_written(self, result: PersistResult) -> None:
        self.written.append(result)

    def on_reloaded(self) -> None:
        self.reloaded += 1

    def on_write_failed(self) -> None:
        self.failed += 1

    def projected_value(self, key: str) -> object:
        return self.projected[key]


def test_write_and_reload_feedback_reach_only_an_existing_coordinator() -> None:
    """Every queue write originates from the panel, so feedback arriving before
    the panel was ever opened has nothing to update — and must not build the
    coordinator (which needs a running app) as a side effect."""
    screen = object.__new__(screen_module.MainScreen)
    result = PersistResult(written={"session.title.auto": False}, rejected={})

    assert screen._existing_settings_coordinator() is None
    screen._on_settings_written(result)
    screen._on_settings_reloaded_for_panel()
    assert screen._existing_settings_coordinator() is None

    coordinator = _Coordinator()
    screen.__dict__["_settings_coordinator_instance"] = coordinator
    screen._on_settings_written(result)
    screen._on_settings_reloaded_for_panel()

    assert coordinator.written == [result]
    assert coordinator.reloaded == 1


def test_save_failures_toast_and_snap_the_panel_back() -> None:
    notifications: list[tuple[str, str]] = []
    screen = object.__new__(screen_module.MainScreen)
    screen.__dict__["_view_adapter"] = SimpleNamespace(
        notify=lambda message, *, title, severity="information", timeout=3: notifications.append(
            (str(message), severity)
        )
    )
    coordinator = _Coordinator()
    screen.__dict__["_settings_coordinator_instance"] = coordinator

    screen._notify_settings_failure(RuntimeError("disk full"))
    screen._notify_settings_rejected(PersistResult(written={}, rejected={"ui.theme": object()}))  # type: ignore[dict-item]

    assert [severity for _message, severity in notifications] == ["error", "error"]
    assert notifications[0][0] == "disk full"
    assert coordinator.failed == 2


def test_inline_question_preference_follows_the_panel_before_the_reload_lands(monkeypatch) -> None:
    """A ticked checkbox is written at once but reaches the in-force settings only
    after the pending reload; a question arriving in between honours the tick."""
    screen = object.__new__(screen_module.MainScreen)
    fake_app = SimpleNamespace(settings_handle=SimpleNamespace(settings=SimpleNamespace(ask_user_inline=False)))
    monkeypatch.setattr(screen_module.MainScreen, "app", property(lambda self: fake_app))

    assert screen._question_inline_preferred() is False, "no panel yet: the in-force value"

    coordinator = _Coordinator()
    coordinator.projected = {"tools.ask_user.inline": True}
    screen.__dict__["_settings_coordinator_instance"] = coordinator

    assert screen._question_inline_preferred() is True


def test_settings_reload_reprojects_the_verify_command_word_list(monkeypatch) -> None:
    """The screen and the dashboard hold the word list as a projection taken at
    construction. A reload re-projects the value now in force — and must not
    record it as a runtime override, which would credit the reload to the
    layer nothing loads from and pin it over every later reload."""
    screen = object.__new__(screen_module.MainScreen)
    overridden: list[dict[str, object]] = []
    fake_app = SimpleNamespace(
        settings_handle=SimpleNamespace(
            settings=SimpleNamespace(trajectory_verify_commands="pytest,ruff check"),
            override=lambda **values: overridden.append(values),
        )
    )
    monkeypatch.setattr(screen_module.MainScreen, "app", property(lambda self: fake_app))
    applied: list[str] = []
    dashboard = SimpleNamespace(set_verify_commands=applied.append)
    monkeypatch.setattr(screen_module.MainScreen, "query_one", lambda self, *args, **kwargs: dashboard)
    screen._trajectory_verify_commands = "stale words"

    screen._refresh_trajectory_verify_commands()

    assert screen._trajectory_verify_commands == "pytest,ruff check"
    assert applied == ["pytest,ruff check"]
    assert overridden == []

    screen._refresh_trajectory_verify_commands()

    assert applied == ["pytest,ruff check"], "an unchanged value must not rebuild the projection"


def test_the_panel_locale_row_uses_the_shared_language_action() -> None:
    """Same path as the picker and ``/language``: a bundle that fails to load
    is reported through the navigation controller, not silently ignored."""
    screen = object.__new__(screen_module.MainScreen)
    requested: list[str] = []
    screen.__dict__["_navigation"] = SimpleNamespace(set_language=requested.append)

    screen._switch_locale_for_panel("zh-Hans")

    assert requested == ["zh-Hans"]


def test_the_settings_queue_is_built_once_and_shared_by_notifications() -> None:
    screen = object.__new__(screen_module.MainScreen)

    queue = screen._settings_persistence()

    assert screen._settings_persistence() is queue
    assert screen.__dict__["_settings_persistence_queue"] is queue
