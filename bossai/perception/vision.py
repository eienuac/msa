"""Frame -> GameState for the live game (and for recorded videos).

v0 scope (validated on the Kor Bekçi recordings):
  * HUD: macro indicator, player/boss HP bars, dash icon readiness
  * telegraphs: detection, tracking, time to impact, projection to the ground
Not yet: boss position on the ground (needs the boss distance OCR or a
detector), exact dash charges (needs digit templates), wind-up cue detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.game_state import BossState, GameState, Hazard, HazardPhase, HazardShape, MacroState, PlayerState
from .boss_reader import BossInfoReader
from .calibration import CameraModel, GroundProjector, fit_circle
from .dash_reader import DashIconReader, DashStateFilter
from .hazard_detection import DetectorConfig, HazardDetector
from .tracking import HazardTracker, TrackerConfig
from .ui_reader import HudReader, HudReading, UILayout


@dataclass
class VisionConfig:
    layout: UILayout = field(default_factory=UILayout)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    camera: CameraModel = field(default_factory=CameraModel)
    circle_fit_max_residual: float = 0.12  # relative residual to accept a ground circle
    boss_keywords: tuple[str, ...] = ("kor be",)  # target name fragment that identifies the boss
    melee_range_m: float = 2.5  # HUD metres at which the macro lands hits (recordings: "2 m")
    read_boss_text: bool = True


@dataclass
class VisionOutput:
    state: GameState
    hud: HudReading
    tracks: list
    dash: object = None
    boss_info: object = None


class VisionPipeline:
    def __init__(self, cfg: VisionConfig | None = None):
        self.cfg = cfg or VisionConfig()
        self.hud = HudReader(self.cfg.layout)
        self.detector = HazardDetector(self.cfg.detector, self.cfg.layout)
        self.tracker = HazardTracker(self.cfg.tracker)
        self.dash_reader = DashIconReader()
        self.dash_filter = DashStateFilter()
        self.boss_reader = BossInfoReader(self.cfg.boss_keywords) if self.cfg.read_boss_text else None
        self._proj: GroundProjector | None = None
        self._last_player_hp = 1.0
        self._last_boss_hp = 1.0
        self._hp_hist: list[tuple[float, float, float]] = []  # (t, player, boss)

    def _projector(self, w: int, h: int) -> GroundProjector:
        if self._proj is None or (self._proj.cam.width, self._proj.cam.height) != (w, h):
            self._proj = GroundProjector(self.cfg.camera.scaled(w, h))
        return self._proj

    def track_to_hazard(self, tr, t: float, proj: GroundProjector) -> Hazard | None:
        g = proj.to_ground(tr.det.contour)
        g = g[~np.isnan(g).any(1)]
        if len(g) < 6:
            return None
        tti = self.tracker.time_to_impact(tr, t)
        phase = tr.phase
        center, radius = fit_circle(g)
        resid = float(np.abs(np.linalg.norm(g - center, axis=1) - radius).mean() / max(radius, 1e-6))
        if tr.det.kind == "ellipse" and resid <= self.cfg.circle_fit_max_residual:
            return Hazard(tr.id, HazardShape.CIRCLE, center, radius, tti, phase, confidence=1.0)
        step = max(1, len(g) // 48)
        return Hazard(tr.id, HazardShape.POLYGON, g.mean(0), float(np.ptp(g, 0).max() / 2), tti, phase, vertices=g[::step])

    def process(self, frame: np.ndarray, t: float) -> VisionOutput:
        h, w = frame.shape[:2]
        proj = self._projector(w, h)
        hud = self.hud.read(frame)
        tracks = self.tracker.update(t, self.detector.detect(frame))
        hazards = [hz for tr in tracks if (hz := self.track_to_hazard(tr, t, proj)) is not None]

        php = hud.player_hp if hud.player_hp is not None else self._last_player_hp
        bhp = hud.boss_hp if hud.boss_hp is not None else self._last_boss_hp
        self._last_player_hp, self._last_boss_hp = php, bhp
        self._hp_hist.append((t, php, bhp))
        while self._hp_hist and t - self._hp_hist[0][0] > 1.0:
            self._hp_hist.pop(0)
        t0, p0, b0 = self._hp_hist[0]
        dash = self.dash_reader.read(frame)
        charges, can_dash, lockout = self.dash_filter.update(dash, t)
        info = self.boss_reader.read(frame, t) if self.boss_reader is not None else None
        targeted = hud.boss_hp is not None and (info is None or info.is_boss)
        boss_pos = None
        if info is not None and info.distance_m is not None:
            # bearing is not read yet: assume the boss is ahead (the camera usually faces it)
            boss_pos = np.array([0.0, info.distance_m])
        state = GameState(
            t=t,
            player=PlayerState(
                pos=np.zeros(2),
                hp=php,
                max_hp=1.0,
                dash_charges=charges,
                dash_lockout=0.0 if can_dash else max(lockout, 0.05),
            ),
            boss=BossState(pos=boss_pos, hp=bhp, max_hp=1.0, targeted=targeted, visible=False, hitbox_radius=0.5),
            hazards=hazards,
            macro=MacroState.UNKNOWN,  # owned by MacroStateManager, fed with hud.macro_p_on
            melee_range=self.cfg.melee_range_m,
            recent_damage_dealt=max(0.0, b0 - bhp),
            recent_damage_taken=max(0.0, p0 - php),
        )
        return VisionOutput(state, hud, tracks, dash, info)


__all__ = ["VisionPipeline", "VisionConfig", "VisionOutput", "HazardPhase"]
