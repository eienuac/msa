"""Scripted reference policies.

They are not the agent; they bound the expected performance and catch reward
bugs (an expert must score far above "never dodge" and "never attack").
"""

from __future__ import annotations

import numpy as np

from ..core.game_state import DIRECTION_VECTORS, GameState, MacroState
from ..core.safety import EscapeParams, SafetyPlanner, ray_features


def direction_towards(vec: np.ndarray) -> int:
    if np.linalg.norm(vec) < 1e-6:
        return 0
    return int(np.argmax(DIRECTION_VECTORS[1:] @ (vec / np.linalg.norm(vec)))) + 1


class BasePolicy:
    name = "base"

    def act(self, env) -> np.ndarray:  # env: BossFightEnv
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env) -> np.ndarray:
        m = env.action_masks()
        out = []
        o = 0
        for n in env.action_space.nvec:
            valid = np.flatnonzero(m[o : o + n])
            out.append(int(self.rng.choice(valid)))
            o += n
        return np.array(out)


def _approach(s: GameState) -> int:
    if s.boss.pos is None or s.in_melee_range():
        return 0
    return direction_towards(s.boss.pos - s.player.pos)


class NeverDodgePolicy(BasePolicy):
    """Walks to the boss, keeps the macro on, ignores every hazard."""

    name = "never_dodge"

    def act(self, env) -> np.ndarray:
        s = env.observed_state()
        mac = 1 if env.macro.state == MacroState.OFF and s.boss.targeted else 0
        return np.array([_approach(s), 0, mac])


class ExpertPolicy(BasePolicy):
    """Rule-based reference: dodge with the planner, otherwise melee with the macro."""

    name = "expert"

    def __init__(self, attack: bool = True, params: EscapeParams | None = None):
        self.attack = attack
        self.planner = SafetyPlanner(params or EscapeParams(), min_slack=0.5)
        if not attack:
            self.name = "pacifist"

    def _blocked(self, s: GameState, move: int) -> bool:
        """True if walking that way enters a warning zone before reaching the boss."""
        rf = ray_features(s, self.planner.p, DIRECTION_VECTORS[move : move + 1])
        reach = min(s.boss_distance(), 4.0) if s.boss.pos is not None else 4.0
        return bool(rf.free_dist[0] < reach)

    def act(self, env) -> np.ndarray:
        s = env.observed_state()
        prev = env.ctx.move_dir
        plan = self.planner.plan(s, prev)
        macro_state = env.macro.state
        if plan.direction:
            move = plan.direction
            dash = plan.dash
            mac = 0  # do not toggle R to dodge: walking usually leaves it on (user, recordings)
            return np.array([move, int(dash and s.player.can_dash), mac])
        if plan.wait:
            return np.array([0, 0, 0])
        move = _approach(s)
        if move and self._blocked(s, move):
            move = 0  # wait outside instead of walking into a warning zone
        mac = 0
        if self.attack and macro_state == MacroState.OFF and (s.boss.targeted or s.in_melee_range()) and move == 0:
            mac = 1
        if not self.attack and macro_state == MacroState.ON:
            mac = 2
        if not self.attack:
            # keep some distance from the boss
            if s.boss.pos is not None and s.boss_distance() < 6:
                move = direction_towards(s.player.pos - s.boss.pos)
            else:
                move = 0
        return np.array([move, 0, mac])


class ModelPolicy(BasePolicy):
    def __init__(self, model, name: str = "model", deterministic: bool = True):
        self.model = model
        self.name = name
        self.deterministic = deterministic
        self._obs = None

    def set_obs(self, obs) -> None:
        self._obs = obs

    def act(self, env) -> np.ndarray:
        action, _ = self.model.predict(self._obs, action_masks=env.action_masks(), deterministic=self.deterministic)
        return np.asarray(action)

