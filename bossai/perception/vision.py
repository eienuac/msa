"""Frame -> GameState for the live game (and for recorded videos).

v0 scope (validated on the Kor Bekçi recordings):
  * HUD: macro indicator, player/boss HP bars, dash icon readiness
  * telegraphs: detection, tracking, time to impact, projection to the ground
Not yet: boss position on the ground (needs the boss distance OCR or a
detector), exact dash charges (needs digit templates), wind-up cue detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from ..core.game_state import BossState, GameState, Hazard, HazardPhase, HazardShape, MacroState, PlayerState
from .boss_reader import BossInfoReader
from .calibration import CameraModel, GroundProjector, fit_circle, fit_circles_ransac
from .dash_reader import DashIconReader, DashStateFilter
from .hazard_detection import DetectorConfig, HazardDetector, ScreenHazard
from .pitch_estimate import PitchConfig, PitchEstimator
from .player_anchor import AnchorConfig, PlayerAnchor
from .profile import load_profile
from .rim_detection import RimDetector, RimStyle
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
    split_radius_range: tuple[float, float] = (1.0, 9.0)  # ground units
    split_tol: float = 0.4
    split_min_coverage: float = 0.8
    ocr_in_background: bool = False  # True for live use: OCR never blocks the frame loop
    telegraph_style: str = "filled"  # "filled" (HazardDetector) or "rim" (RimDetector)
    rim: RimStyle = field(default_factory=RimStyle)
    auto_camera: bool = True  # follow camera lag (overhead bar) and camera tilt (shape of ground circles)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)

    @classmethod
    def for_boss(cls, boss: str, **kw) -> "VisionConfig":
        """Vision settings from configs/bosses/<boss>.yaml."""
        p = load_profile(boss)
        return cls(
            detector=replace(DetectorConfig(), color=p.telegraph_color),
            tracker=replace(TrackerConfig(), default_telegraph=p.default_telegraph, min_confirm_age=p.min_confirm_age),
            boss_keywords=p.ocr_keywords or ("kor be",),
            telegraph_style=p.telegraph_style,
            rim=p.rim_style,
            **kw,
        )


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
        if self.cfg.telegraph_style == "rim":
            self.detector = RimDetector(self.cfg.rim, self.cfg.layout)
        else:
            self.detector = HazardDetector(self.cfg.detector, self.cfg.layout)
        self.tracker = HazardTracker(self.cfg.tracker)
        self.dash_reader = DashIconReader()
        self.dash_filter = DashStateFilter()
        self.boss_reader = (
            BossInfoReader(self.cfg.boss_keywords, background=self.cfg.ocr_in_background) if self.cfg.read_boss_text else None
        )
        self._proj: GroundProjector | None = None
        self.anchor = PlayerAnchor(self.cfg.anchor)
        self.pitch = PitchEstimator(self.cfg.pitch)
        self.camera = self.cfg.camera  # current camera at 1920x1080 (auto_camera adjusts it)
        self._gm: dict[tuple[int, int], np.ndarray] = {}
        self._split_cache: dict[int, tuple[float, list, int, np.ndarray, float]] = {}
        self._last_player_hp = 1.0
        self._last_boss_hp = 1.0
        self._hp_hist: list[tuple[float, float, float]] = []  # (t, player, boss)

    def _projector(self, w: int, h: int) -> GroundProjector:
        cam = self.camera.scaled(w, h)
        if self._proj is None or self._proj.cam != cam:
            self._proj = GroundProjector(cam)
        return self._proj

    def _update_camera(self, frame: np.ndarray, t: float) -> None:
        """Tilt from the circles seen so far, foot point from the overhead bar."""
        cam = self.cfg.camera
        if self.cfg.auto_camera:
            p = self.pitch.estimate(t)
            if p is not None:
                cam = cam.with_pitch(p)
            off = self.anchor.update(frame, t)
            if off is not None:
                cam = cam.with_foot(cam.foot_px[0] + off[0], cam.foot_px[1] + off[1])
        self.camera = cam

    def _ground_mask(self, w: int, h: int) -> np.ndarray:
        if (w, h) not in self._gm:
            self._gm[(w, h)] = self.cfg.layout.ground_mask(w, h)
        return self._gm[(w, h)]

    def _circle_px(self, det: ScreenHazard, w: int, h: int) -> np.ndarray | None:
        """Screen pixels of a whole ground-circle outline, or None if unsuitable."""
        if det.rim_px is not None:
            return det.rim_px
        if det.kind != "ellipse" or det.ellipse_iou < 0.95 or det.ellipse is None:
            return None
        # the outline must not be cut by the screen edge or a UI panel
        c = np.array(det.ellipse[0], float)
        pts = det.contour.reshape(-1, 2).astype(float)
        d = pts - c
        out = pts + 8.0 * d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        if (out[:, 0] < 0).any() or (out[:, 1] < 0).any() or (out[:, 0] >= w).any() or (out[:, 1] >= h).any():
            return None
        gm = self._ground_mask(w, h)
        if (gm[out[:, 1].astype(int), out[:, 0].astype(int)] == 0).any():
            return None
        return pts

    def track_to_hazards(self, tr, t: float, proj: GroundProjector) -> list[Hazard]:
        """Ground hazards for one tracked screen region.

        One circle if the outline is circular; several circles if the outline is
        made of circular arcs (touching/overlapping zones); otherwise a polygon.
        """
        g = proj.to_ground(tr.det.contour)
        g = g[~np.isnan(g).any(1)]
        if len(g) < 6:
            return []
        tti = self.tracker.time_to_impact(tr, t)
        phase = tr.phase
        center, radius = fit_circle(g)
        resid = float(np.abs(np.linalg.norm(g - center, axis=1) - radius).mean() / max(radius, 1e-6))
        if tr.det.kind == "ellipse" and resid <= self.cfg.circle_fit_max_residual:
            return [Hazard(tr.id * 10, HazardShape.CIRCLE, center, radius, tti, phase)]
        circles = self._cached_split(tr, g, proj)
        if circles:
            return [Hazard(tr.id * 10 + k, HazardShape.CIRCLE, c, r, tti, phase) for k, (c, r) in enumerate(circles)]
        step = max(1, len(g) // 48)
        return [Hazard(tr.id * 10, HazardShape.POLYGON, g.mean(0), float(np.ptp(g, 0).max() / 2), tti, phase, vertices=g[::step])]

    def _cached_split(self, tr, hull: np.ndarray, proj: GroundProjector) -> list[tuple[np.ndarray, float]]:
        """Re-run the circle split only when the region changed noticeably (it is costly)."""
        cache = self._split_cache.get(tr.id)
        area = tr.det.area
        centroid = hull.mean(0)
        if (
            cache is not None
            and abs(area - cache[0]) <= 0.12 * cache[0]
            and tr.frames - cache[2] < 8
            and abs(proj.cam.pitch_deg - cache[4]) < 0.5
        ):
            # the player (and the camera) moved since: shift the zones with the outline
            return [(c + centroid - cache[3], r) for c, r in cache[1]]
        circles: list[tuple[np.ndarray, float]] = []
        if tr.det.raw_contour is not None and len(tr.det.raw_contour) >= 12:
            graw = proj.to_ground(tr.det.raw_contour)
            graw = graw[~np.isnan(graw).any(1)]
            if len(graw) >= 12:
                circles = self._split_circles(graw[:: max(1, len(graw) // 400)], hull)
        self._split_cache[tr.id] = (area, circles, tr.frames, centroid, proj.cam.pitch_deg)
        return circles

    def _split_circles(self, g: np.ndarray, hull: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Accept a multi-circle explanation only if it reproduces the outline.

        g: hull-free outline (arcs of the individual circles), hull: the region's
        convex-hull outline. The union of circles must lie inside the hull region
        and fill most of it.
        """
        circles = fit_circles_ransac(g, r_range=self.cfg.split_radius_range, tol=self.cfg.split_tol, region=hull[:: max(1, len(hull) // 200)])
        if len(circles) < 2:
            return []
        lo, hi = hull.min(0) - 1, hull.max(0) + 1
        res = 0.1
        size = np.ceil((hi - lo) / res).astype(int) + 1
        if size.prod() > 4_000_000:
            return []
        region = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillPoly(region, [((hull - lo) / res).astype(np.int32)], 1)

        def disc(c, r):
            m = np.zeros_like(region)
            cc = ((c - lo) / res).astype(int)
            cv2.circle(m, (int(cc[0]), int(cc[1])), int(r / res), 1, -1)
            return m

        # drop circles that stick out of the region (a wrong arc combination)
        kept, union = [], np.zeros_like(region)
        for c, r in circles:
            m = disc(c, r)
            if (m & region).sum() / max(m.sum(), 1) >= 0.85:
                kept.append((c, r))
                union |= m
        if len(kept) < 2:
            return []
        d = np.min([np.abs(np.linalg.norm(g - c, axis=1) - r) for c, r in kept], axis=0)
        if np.mean(d < 2.5 * self.cfg.split_tol) < self.cfg.split_min_coverage:
            return []
        if union.sum() / max(region.sum(), 1) < 0.55:  # must explain most of the region
            return []
        return kept

    def process(self, frame: np.ndarray, t: float) -> VisionOutput:
        h, w = frame.shape[:2]
        self._update_camera(frame, t)
        proj = self._projector(w, h)
        if isinstance(self.detector, RimDetector):
            self.detector.projector = proj
        hud = self.hud.read(frame)
        tracks = self.tracker.update(t, self.detector.detect(frame))
        alive = {tr.id for tr in self.tracker.tracks}
        self._split_cache = {k: v for k, v in self._split_cache.items() if k in alive}
        hazards = [hz for tr in tracks for hz in self.track_to_hazards(tr, t, proj)]
        if self.cfg.auto_camera:
            for tr in tracks:
                px = self._circle_px(tr.det, w, h)
                if px is not None:
                    self.pitch.observe(px, proj.cam, t, key=tr.id)

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
