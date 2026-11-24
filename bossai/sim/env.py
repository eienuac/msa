"""Gymnasium environment wrapping the simulated boss fight.

The environment runs the *same* MacroStateManager, ActionController and
ObservationEncoder as the live runtime; only the input backend (recording
instead of SendInput) and the state source (simulator instead of vision)
differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from ..agent.action_space import NVEC, action_masks, decode
from ..agent.observation_encoder import AgentContext, ObservationEncoder
from ..control.action_controller import ActionController, AgentIntent, ControllerConfig
from ..control.input_backend import RecordingBackend
from ..control.macro_state import MacroEvidence, MacroStateManager
from ..core.game_state import GameState, HazardPhase
from ..core.safety import EscapeParams, SafetyPlanner
from ..rewards.reward_system import RewardSystem, RewardWeights, StepEvents
from .config import SimConfig, level_config
from .perception_noise import Observation, PerceptionModel
from .world import BossWorld


@dataclass
class EpisodeStats:
    steps: int = 0
    reward: float = 0.0
    boss_damage: float = 0.0
    player_damage: float = 0.0
    hits: int = 0
    threats: int = 0
    dodges: int = 0
    late_dodges: int = 0
    dashes: int = 0
    r_presses: int = 0
    direction_changes: int = 0
    reversals: int = 0
    safe_moves: int = 0
    outcome: str = "running"


class BossFightEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        cfg: SimConfig | None = None,
        level: int | None = None,
        reward_weights: RewardWeights | None = None,
        encoder: ObservationEncoder | None = None,
        render_mode: str | None = None,
        gamma: float = 0.99,
    ):
        super().__init__()
        cfg = cfg or SimConfig()
        if level is not None:
            cfg = level_config(level, cfg)
        self.cfg = cfg
        self.encoder = encoder or ObservationEncoder()
        self.observation_space = self.encoder.observation_space
        self.action_space = gym.spaces.MultiDiscrete(NVEC)
        self.rewards = RewardSystem(reward_weights, gamma)
        # the "move to safe zone" action aims for exits with >= 0.5 s to spare, not the last moment
        self.planner = SafetyPlanner(EscapeParams(), min_slack=0.5)
        self.render_mode = render_mode
        self._rng = np.random.default_rng()
        self.world: BossWorld | None = None

    # --------------------------------------------------------------- control
    def set_level(self, level: int) -> None:
        self.cfg = level_config(level, self.cfg)

    def get_level(self) -> int:
        return self.cfg.level

    def action_masks(self) -> np.ndarray:
        return action_masks(self._obs.state, self.macro.state)

    # ------------------------------------------------------------------ gym
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "level" in options:
            self.set_level(int(options["level"]))
        self.world = BossWorld(self.cfg, self._rng)
        self.perception = PerceptionModel(self.cfg.noise, self._rng, self.encoder.fov, self.cfg.decision_dt)
        self.backend = RecordingBackend()
        self.macro = MacroStateManager()
        self.macro.reset(0.0)
        self.controller = ActionController(self.backend, self.macro, ControllerConfig(tap_hold=0.0, dash_dir_lead=0.0))
        self.ctx = AgentContext()
        self.stats = EpisodeStats()
        self._prev_boss_hp_obs = 1.0
        self._obs: Observation = self.perception.observe(self.world)
        self._truth = self.world.true_state()
        self._last_intent = AgentIntent()
        return self.encoder.encode(self._obs.state, self.macro.state, self.ctx), {}

    def _evidence(self) -> MacroEvidence:
        s = self._obs.state
        boss_drop = self._prev_boss_hp_obs - s.boss.hp
        self._prev_boss_hp_obs = s.boss.hp
        in_danger = any(h.contains(s.player.pos[None], margin=0.8)[0] for h in s.hazards) or s.boss.windup > 0
        return MacroEvidence(
            indicator_on=self._obs.indicator_on,
            damage_dealt=bool(boss_drop > 1e-4),
            attack_expected=s.in_melee_range() and s.boss.targeted and self.ctx.move_dir == 0 and not s.player.dashing,
            safe_to_probe=not in_danger,
            can_turn_on=s.boss.targeted or s.in_melee_range(),
        )

    def step(self, action):
        assert self.world is not None, "call reset() first"
        w = self.world
        obs_state = self._obs.state
        intent, invalid = decode(np.asarray(action), obs_state, self.planner, self.ctx.move_dir)
        if int(action[0]) == 9:
            self.stats.safe_moves += 1
        evidence = self._evidence()
        self.backend.clear_events()
        report = self.controller.apply(w.t, intent, evidence)
        taps = [k for kind, k in self.backend.events if kind == "down" and k in ("r", "q")]
        held = {k for k in self.backend.pressed if k in "wasd"}

        prev_truth = self._truth
        in_danger_true = any(
            h.contains(prev_truth.player.pos[None], margin=0.4)[0] and h.phase == HazardPhase.TELEGRAPH
            for h in prev_truth.hazards
        )
        urgent_true = any(
            h.contains(prev_truth.player.pos[None], margin=0.4)[0] and h.time_to_impact < 0.5 and h.phase == HazardPhase.TELEGRAPH
            for h in prev_truth.hazards
        )
        wev = w.step(held, taps, self.cfg.decision_dt)
        changed, reversal = self.ctx.update(w.t, self.controller.current_dir, report.r_pressed, self.cfg.decision_dt)

        ev = StepEvents(
            boss_damage=wev.boss_damage,
            player_damage=wev.player_damage,
            hits=wev.hits,
            dodges=wev.dodges,
            late_dodges=wev.late_dodges,
            inside_telegraph=int(in_danger_true),
            dashes=wev.dashes,
            dash_unneeded=int(wev.dashes > 0 and not in_danger_true and prev_truth.boss.windup <= 0),
            r_presses=wev.r_taps,
            macro_on_in_danger=int(intent.macro == 1 and urgent_true),
            direction_changes=int(changed),
            reversals=int(reversal),
            invalid_actions=int(invalid) + wev.wasted_dash_taps,
            boss_killed=wev.boss_killed,
            player_dead=wev.player_dead,
        )
        self._truth = w.true_state()
        self._obs = self.perception.observe(w)
        terminated = wev.boss_killed or wev.player_dead
        truncated = (not terminated) and w.t >= self.cfg.max_time
        reward, comps = self.rewards.step(prev_truth, self._truth, ev, terminated)

        st = self.stats
        st.steps += 1
        st.reward += reward
        st.boss_damage += wev.boss_damage
        st.player_damage += wev.player_damage
        st.hits += wev.hits
        st.threats += wev.threats
        st.dodges += wev.dodges
        st.late_dodges += wev.late_dodges
        st.dashes += wev.dashes
        st.r_presses += wev.r_taps
        st.direction_changes += int(changed)
        st.reversals += int(reversal)
        info: dict = {"reward_components": comps, "hit_skills": wev.hit_skills}
        if terminated or truncated:
            st.outcome = "kill" if wev.boss_killed else "death" if wev.player_dead else "timeout"
            info["episode_stats"] = dict(st.__dict__, time=w.t, level=self.cfg.level)
        self._last_intent = intent
        obs = self.encoder.encode(self._obs.state, self.macro.state, self.ctx)
        return obs, reward, terminated, truncated, info

    # -------------------------------------------------------------- helpers
    def observed_state(self) -> GameState:
        return self._obs.state

    def true_state(self) -> GameState:
        return self._truth

    def render(self):
        from .render import render_topdown

        return render_topdown(self.world, self._obs.state, self.macro.state, self._last_intent, self.stats)
