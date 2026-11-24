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


def _circle_from_3(p: np.ndarray) -> tuple[np.ndarray, float] | None:
    (x1, y1), (x2, y2), (x3, y3) = p
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return None
    ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
    c = np.array([ux, uy])
    return c, float(np.linalg.norm(p[0] - c))


def fit_circles_ransac(
    pts: np.ndarray,
    r_range: tuple[float, float] = (1.0, 9.0),
    tol: float = 0.25,
    max_circles: int = 6,
    min_inlier_frac: float = 0.08,
    iters: int = 300,
    seed: int = 0,
    region: np.ndarray | None = None,
    region_margin: float = 0.3,
) -> list[tuple[np.ndarray, float]]:
    """Find the circles whose arcs make up a closed outline (ground coordinates).

    The outline of several overlapping circles consists of one arc per circle.
    Circles are extracted greedily: the best-supported circle (most outline
    points within `tol` of it) is kept, its points removed, and the search
    repeats on the rest. If `region` (an outline polygon) is given, candidate
    circles must lie inside it, which rejects big circles that only happen to
    pass through arcs of different zones.
    """
    from ..core.game_state import _polygon_signed_distance

    ring = np.stack([np.cos(np.linspace(0, 2 * np.pi, 24, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False))], 1)
    rng = np.random.default_rng(seed)
    pts = np.asarray(pts, float)
    remaining = pts.copy()
    found: list[tuple[np.ndarray, float]] = []
    n_total = len(pts)
    while len(remaining) >= max(8, int(min_inlier_frac * n_total)) and len(found) < max_circles:
        best, best_in = None, None
        m = len(remaining)
        for _ in range(iters):
            # sample three points that are near each other along the outline (same arc)
            i = rng.integers(m)
            span = max(3, m // 8)
            j, k = (i + rng.integers(1, span)) % m, (i + rng.integers(span // 2, span * 2)) % m
            fit = _circle_from_3(remaining[[i, j, k]])
            if fit is None:
                continue
            c, r = fit
            if not (r_range[0] <= r <= r_range[1]):
                continue
            if region is not None and np.mean(_polygon_signed_distance(c + r * ring, region) <= region_margin) < 0.9:
                continue
            inl = np.abs(np.linalg.norm(remaining - c, axis=1) - r) < tol
            if best_in is None or inl.sum() > best_in.sum():
                best, best_in = (c, r), inl
        if best is None or best_in.sum() < max(8, min_inlier_frac * n_total):
            break
        c, r = fit_circle(remaining[best_in])  # refine on the inliers
        found.append((c, r))
        remaining = remaining[~best_in]
    return merge_duplicate_circles(found)


def merge_duplicate_circles(circles: list[tuple[np.ndarray, float]], center_tol: float = 1.2, radius_tol: float = 1.2):
    """A painted rim has an inner and an outer edge; both fit the same circle. Keep the larger."""
    out: list[tuple[np.ndarray, float]] = []
    for c, r in sorted(circles, key=lambda x: -x[1]):
        if any(np.linalg.norm(c - c2) < center_tol and abs(r - r2) < radius_tol for c2, r2 in out):
            continue
        if any(np.linalg.norm(c - c2) + r <= r2 + 0.3 for c2, r2 in out):
            continue  # fully inside a kept circle (inner fill or rim edge)
        out.append((c, r))
    return out
