"""Where the character is on screen, from its overhead bars.

The follow camera lags behind the character: while walking or dashing the
character drifts 30-80 px away from its resting screen point (Kor Bekçi
fight 1: bar x 908-1003, y 458-527 for 90% of the frames). The ground
projection must be anchored on the character, not on a fixed pixel.

The character's overhead plate has a red HP bar with a blue MP bar of the same
width right below it. Teammates have the same plate, so the candidate nearest
to the last position (at the start: nearest to the resting point) is taken,
and a jump to another plate must be seen twice before it is believed.

Resting point of the bar centre at 1920x1080: (960, 512) in all six fights
(Kor Bekçi and spider). The foot point is shifted by the bar's displacement
from that point.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AnchorConfig:
    search: tuple[int, int, int, int] = (660, 250, 1260, 800)  # x0, y0, x1, y1 at 1920x1080
    rest: tuple[float, float] = (960.0, 512.0)  # bar centre when the camera has caught up
    min_width: int = 35
    max_width: int = 140
    smooth: int = 3  # frames in the median
    max_jump: float = 90.0  # px; a bigger jump needs a second frame to confirm
    max_offset: float = 220.0  # px from the resting point; further is someone else
    hold: float = 0.5  # seconds to keep the last offset when the bar is not seen


class PlayerAnchor:
    def __init__(self, cfg: AnchorConfig | None = None):
        self.cfg = cfg or AnchorConfig()
        self._hist: deque[tuple[float, float]] = deque(maxlen=self.cfg.smooth)
        self._pending: tuple[float, float] | None = None
        self._seen_t = -1e9

    def reset(self) -> None:
        self._hist.clear()
        self._pending = None
        self._seen_t = -1e9

    def find_bars(self, frame: np.ndarray) -> list[tuple[float, float, int]]:
        """All red-over-blue bar pairs: (centre x, centre y, width) in 1080p pixels."""
        h, w = frame.shape[:2]
        sx, sy = w / 1920, h / 1080
        x0, y0, x1, y1 = self.cfg.search
        roi = frame[int(y0 * sy) : int(y1 * sy), int(x0 * sx) : int(x1 * sx)]
        if (w, h) != (1920, 1080):
            roi = cv2.resize(roi, (x1 - x0, y1 - y0))
        b, g, r = cv2.split(roi)

        def above(img: np.ndarray, thr: int) -> np.ndarray:
            return cv2.threshold(img, thr, 1, cv2.THRESH_BINARY)[1]

        # saturating uint8 differences: r - g > 70 etc.
        red = above(r, 140) & above(cv2.subtract(r, g), 70) & above(cv2.subtract(r, b), 50)
        blue = above(b, 140) & above(cv2.subtract(b, r), 50)
        n, _, st, _ = cv2.connectedComponentsWithStats(red)
        nb, _, stb, _ = cv2.connectedComponentsWithStats(blue)
        blues = [stb[j] for j in range(1, nb) if 2 <= stb[j][3] <= 10 and stb[j][2] >= self.cfg.min_width // 2]
        out = []
        for i in range(1, n):
            x, y, ww, hh, a = st[i]
            if not (self.cfg.min_width <= ww <= self.cfg.max_width and 2 <= hh <= 10 and a >= 0.5 * ww * hh):
                continue
            for bx, by, bw, bh, _ in blues:
                dy = by - (y + hh)
                overlap = min(x + ww, bx + bw) - max(x, bx)
                if -1 <= dy <= 8 and overlap >= 0.6 * min(ww, bw):
                    out.append((float(x + ww / 2 + x0), float(y + hh / 2 + y0), int(ww)))
                    break
        return out

    def update(self, frame: np.ndarray, t: float) -> tuple[float, float] | None:
        """Displacement (dx, dy) of the character from its resting screen point,
        in 1080p pixels, or None if it has not been seen recently."""
        c = self.cfg
        bars = [b for b in self.find_bars(frame) if np.hypot(b[0] - c.rest[0], b[1] - c.rest[1]) <= c.max_offset]
        if bars:
            ref = np.median(np.array(self._hist), axis=0) if self._hist else np.array(c.rest)
            bx, by, _ = min(bars, key=lambda b: (b[0] - ref[0]) ** 2 + (b[1] - ref[1]) ** 2)
            if self._hist and np.hypot(bx - ref[0], by - ref[1]) > c.max_jump:
                p = self._pending
                if p is not None and np.hypot(bx - p[0], by - p[1]) <= c.max_jump / 3:
                    self._hist.clear()
                    self._hist.append(p)
                    self._hist.append((bx, by))
                    self._pending = None
                    self._seen_t = t
                else:
                    self._pending = (bx, by)
            else:
                self._hist.append((bx, by))
                self._pending = None
                self._seen_t = t
        if not self._hist or t - self._seen_t > c.hold:
            return None
        mx, my = np.median(np.array(self._hist), axis=0)
        return float(mx - c.rest[0]), float(my - c.rest[1])
