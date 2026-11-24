"""State machine that owns the R (auto-attack macro) key.

The policy only says which macro state it *wants*. This class decides if and
when R is pressed, verifies the result from screen evidence, and refuses to
press while the current state is not confirmed.

Rules:
* R is pressed only when the state is a confirmed ON/OFF and differs from the
  wanted state, never while UNKNOWN or PENDING (except a controlled probe).
* Every press is followed by a confirmation window. A press that changes
  nothing is retried a limited number of times, then the manager locks itself.
* Rate limits: minimum gap between presses and a per-minute cap.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ..core.game_state import MacroState


@dataclass(frozen=True)
class MacroConfig:
    confirm_timeout: float = 0.8  # seconds to wait for evidence after a press
    min_press_interval: float = 0.6
    max_presses_per_min: int = 20
    max_retries: int = 2
    lockout_after_failures: float = 10.0
    on_threshold: float = 0.9
    off_threshold: float = 0.1
    decay: float = 0.5  # weight kept from older evidence on each new observation
    indicator_llr: float = 3.0  # log-odds from a confident UI indicator reading
    damage_llr: float = 1.2  # boss lost HP (steady state alone reaches ~0.92 P(ON))
    no_damage_llr: float = 0.25  # in range, standing, but no damage
    probe_after: float = 3.0  # seconds UNKNOWN before a probe press is allowed
    probe_enabled: bool = True


@dataclass
class MacroEvidence:
    indicator_on: float | None = None  # P(ON) from the UI reader, None if unreadable
    damage_dealt: bool | None = None  # boss HP dropped since last update
    attack_expected: bool = False  # in range + boss targeted + not moving
    invalidate: bool = False  # death, loading screen, focus loss, target lost
    safe_to_probe: bool = False  # no hazard threatens the player
    can_turn_on: bool = True  # e.g. boss is targeted


@dataclass
class MacroEvent:
    t: float
    kind: str
    detail: str = ""


@dataclass
class MacroStateManager:
    cfg: MacroConfig = field(default_factory=MacroConfig)
    state: MacroState = MacroState.UNKNOWN
    log_odds: float = 0.0
    desired: bool | None = None

    def __post_init__(self) -> None:
        self._press_times: deque[float] = deque()
        self._pending_since = 0.0
        self._pending_prev: MacroState = MacroState.UNKNOWN
        self._probing = False
        self._retries = 0
        self._locked_until = -math.inf
        self._unknown_since = 0.0
        self._last_t = 0.0
        self.events: deque[MacroEvent] = deque(maxlen=500)

    # ------------------------------------------------------------------ public
    @property
    def p_on(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.log_odds))

    @property
    def is_confirmed(self) -> bool:
        return self.state in (MacroState.ON, MacroState.OFF)

    def request(self, want_on: bool | None) -> None:
        """Set the wanted state (None = no preference)."""
        self.desired = want_on

    def reset(self, now: float = 0.0, known: MacroState | None = None) -> None:
        self.state = known if known is not None else MacroState.UNKNOWN
        self.log_odds = {MacroState.ON: 4.0, MacroState.OFF: -4.0}.get(self.state, 0.0)
        self.desired = None
        self._press_times.clear()
        self._probing = False
        self._retries = 0
        self._locked_until = -math.inf
        self._unknown_since = now
        self._last_t = now

    def update(self, now: float, ev: MacroEvidence) -> bool:
        """Feed new evidence. Returns True when R must be pressed exactly once now."""
        self._last_t = now
        if ev.invalidate:
            self._set_unknown(now, "invalidated")
            self.log_odds = 0.0
            self._probing = False
            return False

        self._absorb(ev)
        belief = self._belief()

        if self.state in (MacroState.PENDING_ON, MacroState.PENDING_OFF) or self._probing:
            self._resolve_pending(now, belief)
            if self.state in (MacroState.PENDING_ON, MacroState.PENDING_OFF) or self._probing:
                return False
        elif belief is not None:
            if belief != self.state:
                self._log(now, "state", f"{self.state.name}->{belief.name} (p_on={self.p_on:.2f})")
            self.state = belief
        elif self.state != MacroState.UNKNOWN and abs(self.log_odds) < 1.0:
            self._set_unknown(now, "evidence ambiguous")

        return self._maybe_press(now, ev)

    # ----------------------------------------------------------------- helpers
    def _absorb(self, ev: MacroEvidence) -> None:
        e = 0.0
        if ev.indicator_on is not None:
            e += self.cfg.indicator_llr * (2.0 * float(ev.indicator_on) - 1.0)
        if ev.damage_dealt is True:
            e += self.cfg.damage_llr
        elif ev.damage_dealt is False and ev.attack_expected:
            e -= self.cfg.no_damage_llr
        if e != 0.0:
            self.log_odds = max(-8.0, min(8.0, self.cfg.decay * self.log_odds + e))

    def _belief(self) -> MacroState | None:
        p = self.p_on
        if p >= self.cfg.on_threshold:
            return MacroState.ON
        if p <= self.cfg.off_threshold:
            return MacroState.OFF
        return None

    def _resolve_pending(self, now: float, belief: MacroState | None) -> None:
        if self._probing:
            if belief is not None and now - self._pending_since >= 0.2:
                self._probing = False
                self.state = belief
                self._log(now, "probe_result", belief.name)
            elif now - self._pending_since > self.cfg.confirm_timeout * 2:
                self._probing = False
                self._set_unknown(now, "probe inconclusive")
            return
        target = MacroState.ON if self.state == MacroState.PENDING_ON else MacroState.OFF
        if belief == target:
            self.state = target
            self._retries = 0
            self._log(now, "confirmed", target.name)
            return
        if now - self._pending_since <= self.cfg.confirm_timeout:
            return
        if belief == self._pending_prev:
            self._retries += 1
            self.state = self._pending_prev
            self._log(now, "press_lost", f"retry {self._retries}")
            if self._retries > self.cfg.max_retries:
                self._locked_until = now + self.cfg.lockout_after_failures
                self._retries = 0
                self._set_unknown(now, "too many failed presses; locked")
        else:
            self._set_unknown(now, "no confirmation after press")

    def _rate_ok(self, now: float) -> bool:
        while self._press_times and now - self._press_times[0] > 60.0:
            self._press_times.popleft()
        if now < self._locked_until:
            return False
        if self._press_times and now - self._press_times[-1] < self.cfg.min_press_interval:
            return False
        return len(self._press_times) < self.cfg.max_presses_per_min

    def _maybe_press(self, now: float, ev: MacroEvidence) -> bool:
        if self.desired is None or not self._rate_ok(now):
            return False
        if self.state == MacroState.UNKNOWN:
            if (
                self.cfg.probe_enabled
                and ev.safe_to_probe
                and ev.can_turn_on
                and now - self._unknown_since >= self.cfg.probe_after
            ):
                self._press(now)
                self._probing = True
                self._pending_since = now
                self.log_odds = 0.0  # judge the probe on fresh evidence only
                self._log(now, "probe_press")
                return True
            return False
        want = MacroState.ON if self.desired else MacroState.OFF
        if self.state == want:
            return False
        if want == MacroState.ON and not ev.can_turn_on:
            return False
        self._pending_prev = self.state
        self.state = MacroState.PENDING_ON if want == MacroState.ON else MacroState.PENDING_OFF
        self._pending_since = now
        self._press(now)
        self._log(now, "press", f"want {want.name}")
        return True

    def _press(self, now: float) -> None:
        self._press_times.append(now)

    def _set_unknown(self, now: float, why: str) -> None:
        if self.state != MacroState.UNKNOWN:
            self._unknown_since = now
            self._log(now, "unknown", why)
        self.state = MacroState.UNKNOWN

    def _log(self, now: float, kind: str, detail: str = "") -> None:
        self.events.append(MacroEvent(now, kind, detail))

    @property
    def presses_last_minute(self) -> int:
        return sum(1 for t in self._press_times if self._last_t - t <= 60.0)
