"""Turns ground truth into what the vision pipeline would plausibly report.

Models: position/radius/timing errors, missed and late detections, false
positives, field of view, macro indicator misreads and observation latency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..core.game_state import GameState, Hazard, HazardPhase, HazardShape
from ..core.geometry import FieldOfView
from .config import NoiseSpec, sample
from .world import BossWorld, SimHazard


@dataclass
class Observation:
    state: GameState
    indicator_on: float | None  # macro indicator reading (P(ON))


class PerceptionModel:
    def __init__(self, noise: NoiseSpec, rng: np.random.Generator, fov: FieldOfView | None = None, decision_dt: float = 0.1):
        self.n = noise
        self.rng = rng
        self.fov = fov or FieldOfView()
        self.decision_dt = decision_dt
        self.reset()

    def reset(self) -> None:
        self.latency_ticks = int(round(sample(self.rng, self.n.latency) / self.decision_dt))
        self._buf: deque[Observation] = deque(maxlen=self.latency_ticks + 1)
        self._fps: list[tuple[Hazard, float]] = []
        self._fp_id = 10_000_000
        self._seen: set[int] = set()

    def _init_hazard(self, sh: SimHazard, t: float) -> None:
        n, r = self.n, self.rng
        sh.missed = r.random() < n.miss_prob
        sh.visible_from = t + sample(r, n.late_detect)
        sh.center_err = r.normal(0, n.center_sigma, 2) if n.center_sigma > 0 else np.zeros(2)
        sh.radius_scale = float(1.0 + r.normal(0, n.radius_rel_sigma)) if n.radius_rel_sigma > 0 else 1.0

    def _in_fov(self, h: Hazard, player: np.ndarray) -> bool:
        if not self.n.use_fov:
            return True
        poly = h.polygon(n_arc=12) - player[None]
        return bool(self.fov.visible(poly).any() or self.fov.visible((h.center - player)[None])[0])

    def observe(self, world: BossWorld) -> Observation:
        n, r, t = self.n, self.rng, world.t
        truth = world.true_state(include_hidden=False)
        hazards: list[Hazard] = []
        for sh in world.hazards:
            if sh.hidden:
                continue
            if sh.hz.id not in self._seen:
                self._seen.add(sh.hz.id)
                self._init_hazard(sh, t)
            if sh.missed or t < sh.visible_from:
                continue
            h = sh.hz.copy()
            h.center = h.center + sh.center_err + (r.normal(0, n.center_sigma * 0.3, 2) if n.center_sigma else 0)
            h.radius *= sh.radius_scale
            h.inner_radius *= sh.radius_scale
            if h.phase == HazardPhase.TELEGRAPH and n.tti_sigma > 0:
                h.time_to_impact = max(0.0, h.time_to_impact + r.normal(0, n.tti_sigma))
            if self._in_fov(h, truth.player.pos):
                hazards.append(h)
        # false positives: short-lived small circles near the player
        if n.false_pos_rate > 0 and r.random() < n.false_pos_rate * self.decision_dt:
            fp = Hazard(
                id=self._fp_id,
                shape=HazardShape.CIRCLE,
                center=truth.player.pos + r.uniform(-6, 6, 2),
                radius=float(r.uniform(1.0, 2.5)),
                time_to_impact=float(r.uniform(0.5, 1.5)),
                confidence=0.6,
            )
            self._fp_id += 1
            self._fps.append((fp, t + 0.3))
        self._fps = [(h, until) for h, until in self._fps if until > t]
        hazards += [h.copy() for h, _ in self._fps]

        obs = truth
        obs.hazards = hazards
        if n.boss_pos_sigma > 0 and obs.boss.pos is not None:
            obs.boss.pos = obs.boss.pos + r.normal(0, n.boss_pos_sigma, 2)
        if n.use_fov and obs.boss.pos is not None:
            obs.boss.visible = bool(self.fov.visible((obs.boss.pos - obs.player.pos)[None])[0])
        if n.hp_sigma > 0:
            obs.player.hp = float(np.clip(obs.player.hp + r.normal(0, n.hp_sigma), 0, 1))
            obs.boss.hp = float(np.clip(obs.boss.hp + r.normal(0, n.hp_sigma * 0.3), 0, 1))
        # wind-up cue detector is imperfect
        if obs.boss.windup > 0 and n.center_sigma > 0 and r.random() < 0.15:
            obs.boss.windup = 0.0
        obs.arena_radius = world.arena_r  # assumed known from the minimap/calibration
        ind = 1.0 if world.macro_on else 0.0
        if n.indicator_flip > 0 and r.random() < n.indicator_flip:
            ind = 1.0 - ind
        current = Observation(obs, ind)
        self._buf.append(current)
        return self._buf[0]
