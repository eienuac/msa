"""GameState -> policy observation.

Output is a dict:
  grid: (5, 32, 32) egocentric danger grid in [0, 1] (shape-agnostic hazards)
  vec : (D,) features in [-1, 1], normalised with fixed formulas so no
        running statistics file is needed at deployment time.

The feature list is versioned; `schema_hash()` changes whenever the layout
changes, and saved models refuse to run with a different hash.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
from gymnasium import spaces

from ..core.game_state import GameState, MacroState
from ..core.geometry import N_CHANNELS, DangerGrid, FieldOfView, GridSpec
from ..core.safety import SLACK_CAP, EscapeParams, ray_features

SCHEMA_VERSION = 1


@dataclass
class AgentContext:
    """Agent-side memory that the game screen does not show."""

    move_dir: int = 0
    time_in_dir: float = 0.0
    last_r_time: float = -99.0
    dir_change_times: deque = field(default_factory=lambda: deque(maxlen=32))

    def update(self, now: float, move_dir: int, r_pressed: bool, dt: float) -> tuple[bool, bool]:
        """Returns (direction_changed, reversal)."""
        changed = reversal = False
        if move_dir != self.move_dir:
            if self.move_dir != 0 and move_dir != 0:
                changed = True
                d = abs(move_dir - self.move_dir) % 8
                if min(d, 8 - d) == 4 and self.time_in_dir < 0.3:
                    reversal = True
                self.dir_change_times.append(now)
            self.move_dir = move_dir
            self.time_in_dir = 0.0
        else:
            self.time_in_dir += dt
        if r_pressed:
            self.last_r_time = now
        return changed, reversal

    def changes_last(self, now: float, window: float = 1.0) -> int:
        return sum(1 for t in self.dir_change_times if now - t <= window)


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(min(max(x, lo), hi))


class ObservationEncoder:
    N_RAYS = 16

    def __init__(
        self,
        grid: GridSpec | None = None,
        fov: FieldOfView | None = None,
        escape: EscapeParams | None = None,
    ):
        self.grid_spec = grid or GridSpec()
        self.fov = fov or FieldOfView()
        self.escape = escape or EscapeParams()
        self._grid = DangerGrid(self.grid_spec, self.fov)
        self.feature_names = self._names()
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, (N_CHANNELS, self.grid_spec.size, self.grid_spec.size), np.float32),
                "vec": spaces.Box(-1.0, 1.0, (len(self.feature_names),), np.float32),
            }
        )

    # ---------------------------------------------------------------- schema
    def _names(self) -> list[str]:
        n = [
            "hp",
            "dmg_taken_1s",
            "dash_charges",
            "can_dash",
            "dash_lockout",
            "dash_recharge",
            "dashing",
        ]
        n += [f"move_{i}" for i in range(9)]
        n += ["time_in_dir", "dir_changes_1s"]
        n += [f"macro_{m.name}" for m in MacroState]
        n += ["time_since_r", "boss_targeted"]
        n += ["boss_dx", "boss_dy", "boss_dist", "boss_in_range", "boss_hp", "dmg_dealt_1s", "boss_visible", "boss_windup"]
        n += ["in_danger", "earliest_threat", "n_hazards", "ping"]
        for k in range(self.N_RAYS):
            n += [f"ray{k}_free", f"ray{k}_walk_slack", f"ray{k}_dash_slack", f"ray{k}_wall"]
        return n

    def schema(self) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "features": self.feature_names,
            "grid": asdict(self.grid_spec),
            "grid_channels": ["urgency", "occupied", "boss", "wall", "unseen"],
            "fov": asdict(self.fov),
            "escape": asdict(self.escape),
        }

    def schema_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.schema(), sort_keys=True).encode()).hexdigest()[:16]

    # ---------------------------------------------------------------- encode
    def encode(self, s: GameState, macro: MacroState, ctx: AgentContext) -> dict[str, np.ndarray]:
        p = s.player
        ext = self.grid_spec.extent
        v: list[float] = [
            _clip(p.hp_frac * 2 - 1),
            _clip(s.recent_damage_taken * 4),
            p.dash_charges / max(p.dash_max_charges, 1) * 2 - 1,
            1.0 if p.can_dash else -1.0,
            _clip(p.dash_lockout / 9.0 * 2 - 1),
            _clip(p.dash_recharge / 5.0 * 2 - 1),
            1.0 if p.dashing else -1.0,
        ]
        mv = [-1.0] * 9
        mv[ctx.move_dir] = 1.0
        v += mv
        v += [_clip(ctx.time_in_dir / 2.0 * 2 - 1), _clip(ctx.changes_last(s.t) / 5.0 * 2 - 1)]
        mo = [-1.0] * len(MacroState)
        mo[int(macro)] = 1.0
        v += mo
        v += [_clip((s.t - ctx.last_r_time) / 5.0 * 2 - 1), 1.0 if s.boss.targeted else -1.0]
        b = s.boss
        if b.pos is not None:
            rel = b.pos - p.pos
            dist = float(np.linalg.norm(rel))
            v += [_clip(rel[0] / ext), _clip(rel[1] / ext), _clip(dist / 20.0 * 2 - 1), 1.0 if s.in_melee_range() else -1.0]
        else:
            v += [0.0, 0.0, 1.0, -1.0]
        v += [
            _clip(b.hp_frac * 2 - 1),
            _clip(s.recent_damage_dealt * 20 * 2 - 1),
            1.0 if b.visible else -1.0,
            _clip(b.windup * 2 - 1),
        ]
        rf = ray_features(s, self.escape)
        v += [
            1.0 if rf.in_danger else -1.0,
            _clip(rf.earliest_threat / SLACK_CAP * 2 - 1),
            _clip(len(s.hazards) / 8.0 * 2 - 1),
            _clip(s.ping_ms / 200.0 * 2 - 1),
        ]
        rays = np.stack(
            [
                rf.free_dist / self.escape.max_range * 2 - 1,
                rf.walk_slack / SLACK_CAP,
                rf.dash_slack / SLACK_CAP if p.can_dash else np.full(self.N_RAYS, -1.0),
                rf.wall_dist / self.escape.max_range * 2 - 1,
            ],
            1,
        ).ravel()
        vec = np.concatenate([np.asarray(v, np.float32), rays.astype(np.float32)])
        vec = np.clip(vec, -1.0, 1.0)
        return {"grid": self._grid.render(s), "vec": vec}
