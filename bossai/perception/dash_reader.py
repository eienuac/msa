"""Dash (Q) icon reader.

Icon states seen in the Kor Bekçi recordings (1920x1080):
* ready, 3 charges : bright warm arrows, no small digit
* ready, 2/1 charges: bright arrows with a small "2"/"1" bottom-right
* short lock       : dark icon with "0,x" countdown and the small charge digit
* empty lockout    : dark icon with a large countdown, no small digit (~8 s)
* unlock flash     : very bright glow for ~0.25 s, counts as ready

Measured features (warm = share of warm-yellow pixels in the icon):
ready 0.26-0.34, dark 0.00-0.04, flash 0.68-0.75.
The small digit is matched against grayscale templates cut from the videos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TEMPLATE_DIR = Path(__file__).parent / "templates"
ICON_ORIGIN = (604, 960)  # top-left of the 50x55 icon crop at 1920x1080
SUB_BOX = (32, 28, 43, 43)  # x0, y0, x1, y1 of the small digit inside the crop


@dataclass
class DashReading:
    ready: bool
    charges: int | None  # None when unreadable
    locked: bool
    empty: bool  # all charges used, long lockout
    warm: float
    sub_scores: dict


class DashIconReader:
    def __init__(self, width: int = 1920, height: int = 1080):
        self.sx, self.sy = width / 1920, height / 1080
        self.templates: dict[str, np.ndarray] = {}
        for name in ("sub_b2", "sub_bnone", "sub_d1", "sub_d2"):
            img = cv2.imread(str(TEMPLATE_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.templates[name] = img.astype(np.float32)

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] != 1920 or frame.shape[0] != 1080:  # HUD positions are in 1080p pixels
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        x0, y0 = ICON_ORIGIN
        return frame[y0 : y0 + 55, x0 : x0 + 50]

    def _score(self, patch: np.ndarray, name: str) -> float:
        t = self.templates.get(name)
        if t is None:
            return -1.0
        # allow +-2 px misalignment
        r = cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED)
        return float(r.max())

    def read(self, frame: np.ndarray) -> DashReading:
        c = self._crop(frame)
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        h, s, v = (hsv[..., i].astype(np.int16) for i in range(3))
        warm = float(((h >= 10) & (h <= 32) & (s >= 50) & (v >= 150))[12:40, 10:38].mean())
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
        x0, y0, x1, y1 = SUB_BOX
        patch = gray[y0 - 2 : y1 + 2, x0 - 2 : x1 + 2]
        ready = warm >= 0.15
        if ready:
            scores = {"3": self._score(patch, "sub_bnone"), "2": self._score(patch, "sub_b2")}
            if warm >= 0.55:  # unlock flash: digit hidden by the glow
                return DashReading(True, None, False, False, warm, scores)
            best = max(scores, key=scores.get)
            charges = int(best) if scores[best] >= 0.5 else None
            if charges is None and self._score(patch, "sub_d1") >= 0.85:
                charges = 1
            return DashReading(True, charges, False, False, warm, scores)
        scores = {"2": self._score(patch, "sub_d2"), "1": self._score(patch, "sub_d1")}
        best = max(scores, key=scores.get)
        if scores[best] >= 0.85:  # true small digits score ~0.95; the big countdown <= 0.75
            return DashReading(False, int(best), True, False, warm, scores)
        return DashReading(False, 0, True, True, warm, scores)


class DashStateFilter:
    """Smooths icon readings over time and estimates the remaining lockout.

    The first frame after a press shows a white "pressed" border that hides
    the small digit; a single "empty" reading is therefore not trusted until
    it repeats. Timings measured in the recordings: short lock 1.0 s ("1,0"
    countdown), empty lockout ~8 s, one charge back every ~3.3 s.
    """

    def __init__(self, max_charges: int = 3, short_lock: float = 1.0, empty_lock: float = 8.0):
        self.max = max_charges
        self.short_lock = short_lock
        self.empty_lock = empty_lock
        self.charges = max_charges
        self.lock_start: float | None = None
        self.empty_since: float | None = None
        self._empty_votes = 0

    def update(self, r: DashReading, t: float) -> tuple[int, bool, float]:
        """Returns (charges, can_dash_now, estimated_lockout_seconds)."""
        if r.ready:
            self._empty_votes = 0
            self.lock_start = self.empty_since = None
            if r.charges is not None:
                self.charges = r.charges
            elif self.charges == 0:
                self.charges = self.max  # unlock flash after the long lockout
            return self.charges, True, 0.0
        if r.empty:
            self._empty_votes += 1
            if self._empty_votes >= 2:
                if self.empty_since is None:
                    self.empty_since = self.lock_start if self.lock_start is not None else t
                self.charges = 0
                return 0, False, max(0.0, self.empty_lock - (t - self.empty_since))
        else:
            self._empty_votes = 0
            if r.charges is not None:
                self.charges = r.charges
        if self.lock_start is None:
            self.lock_start = t
        return self.charges, False, max(0.0, self.short_lock - (t - self.lock_start))
