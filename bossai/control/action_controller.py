"""Turns policy intents into the minimum set of key events.

* Movement: the wanted key set is compared with the keys already held; only
  the difference is sent. Repeating the same direction sends nothing.
* Dash: direction keys first, then a single Q tap.
* Macro: delegated to MacroStateManager, which decides whether R is tapped.
* Every tap is press + timed release; nothing is ever held by accident, and
  `release_all` is called on stop, errors, and focus loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..core.game_state import DIRECTION_KEYS
from .input_backend import InputBackend
from .macro_state import MacroEvidence, MacroStateManager

MOVE_KEYS = ("w", "a", "s", "d")


@dataclass
class AgentIntent:
    move: int = 0  # 0 idle, 1..8 direction (camera frame)
    dash: bool = False  # dash in the move direction
    macro: int = 0  # 0 keep, 1 want ON, 2 want OFF


@dataclass(frozen=True)
class ControllerConfig:
    tap_hold: float = 0.05  # seconds between key down and key up for R/Q taps
    dash_dir_lead: float = 0.03  # direction key held at least this long before Q


@dataclass
class ControllerReport:
    r_pressed: bool = False
    dash_sent: bool = False
    direction_changed: bool = False
    key_events: int = 0


@dataclass
class ActionController:
    backend: InputBackend
    macro: MacroStateManager
    cfg: ControllerConfig = field(default_factory=ControllerConfig)
    enabled: bool = True

    def __post_init__(self) -> None:
        self.held: set[str] = set()
        self.current_dir = 0
        self._scheduled: list[tuple[float, str, str]] = []  # (time, "up"/"tap", key)
        self._tapping: set[str] = set()
        self.shield_held = False  # E, held separately from the movement keys

    # ------------------------------------------------------------------ public
    def apply(self, now: float, intent: AgentIntent, evidence: MacroEvidence) -> ControllerReport:
        rep = ControllerReport()
        if not self.enabled:
            self.release_all()
            return rep
        self.service(now)

        if intent.macro == 1:
            self.macro.request(True)
        elif intent.macro == 2:
            self.macro.request(False)
        moving = intent.move != 0 or intent.dash
        if moving and not evidence.moving:
            evidence = replace(evidence, moving=True)
        if self.macro.update(now, evidence):
            rep.r_pressed = self._tap(now, "r")
            rep.key_events += 1

        want = set(DIRECTION_KEYS[intent.move])
        changed_now = False
        for k in sorted(self.held - want):
            self.backend.key_up(k)
            self.held.discard(k)
            rep.key_events += 1
            changed_now = True
        for k in sorted(want - self.held):
            self.backend.key_down(k)
            self.held.add(k)
            rep.key_events += 1
            changed_now = True
        if intent.move != self.current_dir:
            rep.direction_changed = self.current_dir != 0 and intent.move != 0
            self.current_dir = intent.move

        if intent.dash and intent.move != 0 and "q" not in self._tapping:
            if changed_now and self.cfg.dash_dir_lead > 0:
                self._scheduled.append((now + self.cfg.dash_dir_lead, "tap", "q"))
                self._tapping.add("q")
            else:
                self._tap(now, "q")
            rep.dash_sent = True
            rep.key_events += 1
        return rep

    def service(self, now: float) -> None:
        """Process due releases and delayed taps. Call often (every frame) in real time."""
        due = [e for e in self._scheduled if e[0] <= now]
        if not due:
            return
        self._scheduled = [e for e in self._scheduled if e[0] > now]
        for _, kind, key in sorted(due):
            if kind == "up":
                self.backend.key_up(key)
                self._tapping.discard(key)
            else:  # delayed tap
                self._tapping.discard(key)
                self._tap(now, key)

    def set_shield(self, on: bool) -> bool:
        """Hold or release E. Returns True if a key event was sent."""
        if not self.enabled or on == self.shield_held:
            return False
        (self.backend.key_down if on else self.backend.key_up)("e")
        self.shield_held = on
        return True

    def release_all(self) -> None:
        if self.shield_held:
            self.backend.key_up("e")
            self.shield_held = False
        for k in sorted(self.held):
            self.backend.key_up(k)
        self.held.clear()
        for _, kind, key in self._scheduled:
            if kind == "up":
                self.backend.key_up(key)
        self._scheduled.clear()
        self._tapping.clear()
        self.current_dir = 0

    # ----------------------------------------------------------------- helpers
    def _tap(self, now: float, key: str) -> bool:
        if key in self._tapping:
            return False
        self.backend.key_down(key)
        self._tapping.add(key)
        if self.cfg.tap_hold <= 0:
            self.backend.key_up(key)
            self._tapping.discard(key)
        else:
            self._scheduled.append((now + self.cfg.tap_hold, "up", key))
        return True
