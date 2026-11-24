"""Tracks telegraphs across frames and estimates time to impact.

* Detections are matched to tracks by bounding-box overlap / centre distance.
* A new track records the unfilled tint level (baseline) of its region.
* Fill progress (0..1) is sampled every frame; a line fitted to recent
  samples predicts when the fill reaches the rim, i.e. the impact time.
* Without a usable fill trend the estimate falls back to the typical
  telegraph duration minus the track's age.
* A sudden jump of colour strength over the whole region is the impact flash.
* Tracks that live far longer than any telegraph are background (decals,
  scenery) and are suppressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.game_state import HazardPhase
from .hazard_detection import ScreenHazard, fill_from_area


@dataclass
class Track:
    id: int
    first_t: float
    last_t: float
    det: ScreenHazard
    baseline: float
    frames: int = 1
    missed: int = 0
    fills: list[tuple[float, float]] = field(default_factory=list)
    phase: HazardPhase = HazardPhase.TELEGRAPH
    impact_t: float | None = None
    background: bool = False

    @property
    def age(self) -> float:
        return self.last_t - self.first_t


@dataclass(frozen=True)
class TrackerConfig:
    match_iou: float = 0.2
    match_center_frac: float = 0.4  # centre distance / max(bbox w, h)
    max_missed: int = 3
    confirm_frames: int = 2
    default_telegraph: float = 1.25  # seconds (video: 1.1-1.5)
    max_age: float = 4.0  # longer-lived "telegraphs" are scenery
    min_fill_span: float = 0.15  # seconds of fill samples before trusting the trend
    flash_jump: float = 18.0  # median colour strength jump that signals impact
    active_time: float = 0.3
    fill_delta: float = 7.0


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1)


class HazardTracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.tracks: list[Track] = []
        self._next = 1

    def update(self, t: float, dets: list[ScreenHazard]) -> list[Track]:
        c = self.cfg
        unmatched = list(range(len(dets)))
        for tr in self.tracks:
            best, best_score = None, 0.0
            for i in unmatched:
                d = dets[i]
                iou = _iou(tr.det.bbox, d.bbox)
                size = max(tr.det.bbox[2], tr.det.bbox[3], 1)
                dist = float(np.linalg.norm(tr.det.center - d.center)) / size
                score = iou if iou >= c.match_iou else (0.5 - dist if dist <= c.match_center_frac else 0.0)
                if score > best_score:
                    best, best_score = i, score
            if best is None:
                tr.missed += 1
                continue
            unmatched.remove(best)
            self._advance(tr, t, dets[best])
        for i in unmatched:
            d = dets[i]
            tr = Track(self._next, t, t, d, baseline=d.mean_strength)
            tr.fills.append((t, 0.0))
            self._next += 1
            self.tracks.append(tr)
        # retire tracks
        alive = []
        for tr in self.tracks:
            if tr.missed > c.max_missed:
                continue
            if tr.phase == HazardPhase.ACTIVE and tr.impact_t is not None and t - tr.impact_t > c.active_time:
                continue
            if tr.age > c.max_age:
                tr.background = True
            alive.append(tr)
        self.tracks = alive
        return [tr for tr in self.tracks if tr.frames >= c.confirm_frames and not tr.background and tr.missed == 0]

    def _advance(self, tr: Track, t: float, d: ScreenHazard) -> None:
        c = self.cfg
        prev = tr.det
        tr.det, tr.last_t, tr.frames, tr.missed = d, t, tr.frames + 1, 0
        if tr.phase == HazardPhase.ACTIVE:
            return
        if d.mean_strength - prev.mean_strength > c.flash_jump and tr.frames > 2:
            tr.phase = HazardPhase.ACTIVE
            tr.impact_t = t
            return
        tr.fills.append((t, fill_from_area(d.strength_sample, tr.baseline, c.fill_delta)))
        if len(tr.fills) > 30:
            tr.fills.pop(0)

    def time_to_impact(self, tr: Track, t: float) -> float:
        c = self.cfg
        if tr.phase == HazardPhase.ACTIVE:
            return 0.0
        fallback = max(0.0, c.default_telegraph - (t - tr.first_t))
        pts = np.array(tr.fills[-10:])
        if len(pts) >= 3 and pts[-1, 0] - pts[0, 0] >= c.min_fill_span and pts[:, 1].max() > 0.05:
            slope, icpt = np.polyfit(pts[:, 0], pts[:, 1], 1)
            if slope > 0.2:
                t_full = (1.0 - icpt) / slope
                est = max(0.0, t_full - t)
                # blend with the prior while the trend is short
                w = min(1.0, (pts[-1, 0] - pts[0, 0]) / 0.5)
                return float(w * est + (1 - w) * fallback)
        return float(fallback)

    def current_fill(self, tr: Track) -> float:
        return tr.fills[-1][1] if tr.fills else 0.0
