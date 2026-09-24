# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Playground: inspect Buddy species sprites and animation frames.

Usage:
    uv run python playground/buddy.py

Keybindings:
    Space       — play / pause animation
    n / p       — next / previous animation tick
    x           — toggle pet-mode animation
    s           — toggle shiny badge animation
    t           — cycle rarity tier
    b           — toggle blink override
    r           — reset all controls
    q           — quit
"""

from __future__ import annotations

from itertools import batched

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.timer import Timer
from textual.widgets import Button, Footer, Header, Static

from chrys.app.features.buddy.animation import FRAME_COUNT, get_idle_frame, get_pet_frame
from chrys.app.features.buddy.hatchery import SPECIES_RARITY
from chrys.app.features.buddy.model import Appearance, Rarity, Species
from chrys.app.features.buddy.portrait import PORTRAIT_WIDTH, RARITY_COLORS, SHINY_FPS, render_portrait

COLUMNS = 4
TICK_SECONDS = 0.35


RARITY_OPTIONS = [None, *Rarity]


def _look_for(species: Species, shiny: bool = False, rarity_override: Rarity | None = None) -> Appearance:
    return Appearance(species, rarity_override if rarity_override is not None else SPECIES_RARITY[species], shiny)


class SpeciesTile(Static):
    """One rendered Buddy species."""

    def __init__(self, species: Species) -> None:
        super().__init__("", classes="species-tile")
        self.species = species

    def render_species(
        self,
        tick: int,
        blink_override: bool,
        pet_mode: bool,
        shiny: bool = False,
        rarity_override: Rarity | None = None,
        *,
        effect_tick: int = 0,
    ) -> None:
        if pet_mode:
            frame = get_pet_frame(tick)
            blink = False
        else:
            frame, should_blink = get_idle_frame(self.species, tick)
            blink = blink_override or should_blink
        look = _look_for(self.species, shiny, rarity_override)
        _base_background, background = self.background_colors
        portrait_lines = render_portrait(
            look,
            self.species.value,
            frame=frame,
            blink=blink,
            width=self.content_size.width or PORTRAIT_WIDTH,
            effect_tick=effect_tick,
            bg_rgb=background.rgb,
        )
        rarity = look.rarity
        color = RARITY_COLORS[rarity]

        text = Text()
        text.append(f"{self.species.value}  ", style="bold")
        text.append(f"[{rarity.value}]", style=f"bold {color}")
        text.append(f"  frame {frame + 1}/{FRAME_COUNT}")
        if shiny:
            text.append("  ✨ SHINY", style="bold gold1")
        if blink:
            text.append("  blink", style="italic")
        if pet_mode:
            text.append("  pet", style="italic")
        text.append("\n")
        text.append(Text("\n").join(portrait_lines))
        self.update(text)


class BuddyPlayground(App):
    """Buddy sprite debugging playground."""

    BINDINGS = [
        Binding("space", "toggle_play", "Play/Pause"),
        Binding("n", "next_tick", "Next"),
        Binding("p", "previous_tick", "Previous"),
        Binding("x", "toggle_pet", "Pet Mode"),
        Binding("s", "toggle_shiny", "Shiny"),
        Binding("t", "next_rarity", "Rarity Tier"),
        Binding("b", "toggle_blink", "Blink"),
        Binding("r", "reset", "Reset"),
        Binding("q", "quit", "Quit"),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    #controls {
        dock: top;
        height: 3;
        padding: 0 1;
        background: $panel;
        align: left middle;
    }

    #status {
        width: 1fr;
        content-align: right middle;
        padding: 0 1;
        color: $text-muted;
    }

    #species-scroll {
        height: 1fr;
    }

    .species-row {
        height: auto;
    }

    .species-tile {
        width: 1fr;
        min-height: 16;
        margin: 0 1 1 0;
        padding: 1;
        border: solid $surface;
        background: $surface 20%;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.tick = 0
        self.effect_tick = 0
        self.playing = True
        self.pet_mode = False
        self.shiny_mode = False
        self.rarity_index = 0
        self.blink_override = False
        self._timer: Timer | None = None
        self._shiny_timer: Timer | None = None

    @property
    def rarity_override(self) -> Rarity | None:
        return RARITY_OPTIONS[self.rarity_index]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="controls"):
            yield Button("Play / Pause", id="play-pause")
            yield Button("Prev", id="previous")
            yield Button("Next", id="next")
            yield Button("Pet", id="pet")
            yield Button("Shiny", id="shiny")
            yield Button("Rarity", id="rarity")
            yield Button("Blink", id="blink")
            yield Button("Reset", id="reset")
            yield Static("", id="status")
        with VerticalScroll(id="species-scroll"):
            for row in batched(Species, COLUMNS, strict=False):
                with Horizontal(classes="species-row"):
                    for species in row:
                        yield SpeciesTile(species)
        yield Footer()

    def on_mount(self) -> None:
        self._timer = self.set_interval(TICK_SECONDS, self._advance)
        self._shiny_timer = self.set_interval(1 / SHINY_FPS, self._advance_shiny)
        self._refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "play-pause":
                self.action_toggle_play()
            case "previous":
                self.action_previous_tick()
            case "next":
                self.action_next_tick()
            case "pet":
                self.action_toggle_pet()
            case "shiny":
                self.action_toggle_shiny()
            case "rarity":
                self.action_next_rarity()
            case "blink":
                self.action_toggle_blink()
            case "reset":
                self.action_reset()

    def action_toggle_play(self) -> None:
        self.playing = not self.playing
        for timer in (self._timer, self._shiny_timer):
            if timer is not None:
                if self.playing:
                    timer.resume()
                else:
                    timer.pause()
        self._refresh()

    def action_next_tick(self) -> None:
        self.tick += 1
        self.effect_tick += 1
        self._refresh()

    def action_previous_tick(self) -> None:
        self.tick = max(0, self.tick - 1)
        self.effect_tick = max(0, self.effect_tick - 1)
        self._refresh()

    def action_toggle_blink(self) -> None:
        self.blink_override = not self.blink_override
        self._refresh()

    def action_toggle_pet(self) -> None:
        self.pet_mode = not self.pet_mode
        self._reset_timer()
        self._refresh()

    def _reset_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            interval = 0.12 if self.pet_mode else TICK_SECONDS
            self._timer = self.set_interval(interval, self._advance, pause=not self.playing)

    def action_toggle_shiny(self) -> None:
        self.shiny_mode = not self.shiny_mode
        self._refresh()

    def action_next_rarity(self) -> None:
        self.rarity_index = (self.rarity_index + 1) % len(RARITY_OPTIONS)
        self._refresh()

    def action_reset(self) -> None:
        self.tick = 0
        self.effect_tick = 0
        self.pet_mode = False
        self._reset_timer()
        self.shiny_mode = False
        self.rarity_index = 0
        self.blink_override = False
        self._refresh()

    def _advance(self) -> None:
        self.tick += 1
        self._refresh()

    def _advance_shiny(self) -> None:
        if self.shiny_mode:
            self.effect_tick += 1
            self._refresh()

    def _refresh(self) -> None:
        for tile in self.query(SpeciesTile):
            tile.render_species(
                self.tick,
                self.blink_override,
                self.pet_mode,
                self.shiny_mode,
                self.rarity_override,
                effect_tick=self.effect_tick,
            )

        state = "playing" if self.playing else "paused"
        blink = "forced blink" if self.blink_override else "automatic blink"
        mode = "pet mode" if self.pet_mode else "idle mode"
        shiny_label = "✨ shiny" if self.shiny_mode else "normal"
        rarity_label = f"rarity {self.rarity_override.value}" if self.rarity_override else "rarity default"
        self.query_one("#status", Static).update(
            Text(f"tick {self.tick} | {state} | {mode} | {shiny_label} | {rarity_label} | {blink}")
        )


if __name__ == "__main__":
    BuddyPlayground().run()
