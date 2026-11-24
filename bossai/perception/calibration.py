"""Screen pixels <-> ground plane (metres), via a pinhole camera model.

Defaults were fitted to the Kor Bekçi recordings: ground circles near the
player appear with a minor/major axis ratio of ~0.40-0.49, which gives a
viewing depression of ~25-27 degrees. Scale (camera height) is provisional
until a calibration clip (walk W for 2 s) or the boss distance readout
("5 m") fixes it.

Ground frame: origin at the player's feet, +y = camera forward (W), +x = right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    focal_px: float = 1100.0
    cx: float = 960.0
    cy: float = 540.0
    pitch_deg: float = 19.0  # camera tilt below the horizon
    height_m: float = 6.0
    foot_px: tuple[float, float] = (958.0, 668.0)  # player's ground contact point
    width: int = 1920
    height: int = 1080

    def scaled(self, w: int, h: int) -> "CameraModel":
        if (w, h) == (self.width, self.height):
            return self
        s = w / self.width
        return CameraModel(
            self.focal_px * s, self.cx * s, self.cy * h / self.height, self.pitch_deg, self.height_m,
            (self.foot_px[0] * s, self.foot_px[1] * h / self.height), w, h,
        )


class GroundProjector:
    def __init__(self, cam: CameraModel | None = None):
        self.cam = cam or CameraModel()
        th = math.radians(self.cam.pitch_deg)
        self._s, self._c = math.sin(th), math.cos(th)
        self._foot = self._raw_ground(np.array([self.cam.foot_px], float))[0]

    def _raw_ground(self, px: np.ndarray) -> np.ndarray:
        c = self.cam
        dx = (px[:, 0] - c.cx) / c.focal_px
        dy = (px[:, 1] - c.cy) / c.focal_px
        denom = dy * self._c + self._s
        out = np.full((len(px), 2), np.nan)
        ok = denom > 1e-4  # below the horizon
        t = c.height_m / denom[ok]
        out[ok, 0] = t * dx[ok]
        out[ok, 1] = t * (self._c - dy[ok] * self._s)
        return out

    def to_ground(self, px: np.ndarray) -> np.ndarray:
        """(N,2) pixels -> (N,2) metres relative to the player's feet (NaN above horizon)."""
        return self._raw_ground(np.asarray(px, float).reshape(-1, 2)) - self._foot[None]

    def to_pixels(self, ground: np.ndarray) -> np.ndarray:
        c = self.cam
        g = np.asarray(ground, float).reshape(-1, 2) + self._foot[None]
        X, Y = g[:, 0], g[:, 1]
        yc = -Y * self._s + c.height_m * self._c
        zc = Y * self._c + c.height_m * self._s
        return np.stack([c.cx + c.focal_px * X / zc, c.cy + c.focal_px * yc / zc], 1)


def fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic (Kasa) least-squares circle fit. Returns (centre, radius)."""
    pts = np.asarray(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = math.sqrt(max(sol[2] + cx**2 + cy**2, 0.0))
    return np.array([cx, cy]), r
