"""Reward function, shared by training and evaluation.

Dominant terms: damage dealt, damage taken, kill, death. Danger is shaped
with a potential function (gamma * phi(s') - phi(s)), which does not change
the optimal policy. Small costs discourage wasted dashes, R presses and
jittery movement. Every component is returned separately for logging.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..core.game_state import GameState, HazardPhase


@dataclass(frozen=True)
class RewardWeights:
    boss_damage: float = 10.0  # whole boss HP bar = +10
    kill: float = 10.0
    damage_taken: float = 15.0  # whole player HP bar = -15
    per_hit: float = 0.5
    death: float = 15.0
    time: float = 0.003  # per 0.1 s tick (a 240 s timeout costs -7.2)
    danger_potential: float = 1.0
    danger_horizon: float = 2.0  # seconds
    dodge: float = 0.1  # left a threatening zone with time to spare (keep small)
    late_dodge: float = 0.3  # survived, but left the zone at the last moment
    inside_telegraph: float = 0.04  # per 0.1 s tick spent inside a visible warning zone
    dash: float = 0.05
    dash_unneeded: float = 0.15
    r_press: float = 0.04  # every R press costs; the indicator tells whether one is needed
    macro_on_in_danger: float = 0.3
    direction_change: float = 0.01
    reversal: float = 0.05
    invalid_action: float = 0.02


@dataclass
class StepEvents:
    boss_damage: float = 0.0  # fraction of boss HP removed this step
    player_damage: float = 0.0  # fraction of player HP lost this step
    hits: int = 0
    dodges: int = 0
    late_dodges: int = 0
    inside_telegraph: int = 0
    dashes: int = 0
    dash_unneeded: int = 0
    r_presses: int = 0
    macro_on_in_danger: int = 0
    direction_changes: int = 0
    reversals: int = 0
    invalid_actions: int = 0
    boss_killed: bool = False
    player_dead: bool = False


class RewardSystem:
    def __init__(self, weights: RewardWeights | None = None, gamma: float = 0.99):
        self.w = weights or RewardWeights()
        self.gamma = gamma

    def potential(self, s: GameState) -> float:
        """Negative urgency of the visible hazards currently covering the player."""
        pos = s.player.pos[None]
        total = 0.0
        for h in s.hazards:
            if h.phase != HazardPhase.TELEGRAPH:
                continue
            if h.contains(pos, margin=0.4)[0]:
                total += 1.0 - min(max(h.time_to_impact, 0.0) / self.w.danger_horizon, 1.0)
        return -self.w.danger_potential * total

    def step(self, prev: GameState, cur: GameState, ev: StepEvents, terminal: bool) -> tuple[float, dict[str, float]]:
        w = self.w
        phi_prev = self.potential(prev)
        phi_cur = 0.0 if terminal else self.potential(cur)
        c = {
            "boss_damage": w.boss_damage * ev.boss_damage,
            "kill": w.kill if ev.boss_killed else 0.0,
            "damage_taken": -w.damage_taken * ev.player_damage - w.per_hit * ev.hits,
            "death": -w.death if ev.player_dead else 0.0,
            "time": -w.time,
            "danger_shaping": self.gamma * phi_cur - phi_prev,
            "dodge": w.dodge * ev.dodges - w.late_dodge * ev.late_dodges,
            "inside_warning": -w.inside_telegraph * ev.inside_telegraph,
            "dash": -w.dash * ev.dashes - w.dash_unneeded * ev.dash_unneeded,
            "macro": -w.r_press * ev.r_presses - w.macro_on_in_danger * ev.macro_on_in_danger,
            "jitter": -w.direction_change * ev.direction_changes - w.reversal * ev.reversals,
            "invalid": -w.invalid_action * ev.invalid_actions,
        }
        return float(sum(c.values())), c

    def describe(self) -> dict:
        return asdict(self.w)
