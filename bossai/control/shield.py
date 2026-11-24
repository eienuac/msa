"""E shield reflex.

The spider's following ring cannot be walked out of: it tracks the player until
it lands. The only answer (user) is the E shield, which stays up while E is
held, about 1 s at most. Timing is a rule, not something for the policy to
learn: hold E when a following zone around the player is about to land,
release it once the zone has landed.

Like the macro, this sits outside the policy's action space, so policies
trained before it existed keep working.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.game_state import GameState, HazardPhase


@dataclass(frozen=True)
class ShieldConfig:
    enabled: bool = True
    lead: float = 0.5  # press this long before the expected impact (covers latency and timing noise)
    max_hold: float = 1.0  # the game drops the shield after ~1 s anyway
    cooldown: float = 6.0  # do not try again sooner (the real cooldown is not known yet)


class ShieldReflex:
    def __init__(self, cfg: ShieldConfig | None = None):
        self.cfg = cfg or ShieldConfig()
        self.reset()

    def reset(self) -> None:
        self._since: float | None = None
        self._ready_at = 0.0
        self.presses = 0

    @property
    def holding(self) -> bool:
        return self._since is not None

    def _threat(self, s: GameState, horizon: float) -> bool:
        pos = s.player.pos[None]
        return any(
            h.following and h.phase == HazardPhase.TELEGRAPH and h.time_to_impact <= horizon and h.contains(pos, margin=0.3)[0]
            for h in s.hazards
        )

    def update(self, t: float, s: GameState) -> bool:
        """True while E should be held."""
        c = self.cfg
        if not c.enabled:
            return False
        if self._since is not None:
            if t - self._since >= c.max_hold or not self._threat(s, c.lead + c.max_hold):
                self._since = None
                self._ready_at = t + c.cooldown
            return self._since is not None
        if t >= self._ready_at and self._threat(s, c.lead):
            self._since = t
            self.presses += 1
            return True
        return False
