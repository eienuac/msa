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
    pass through arcs of different zones. Vectorised: all candidates of one
    round are evaluated at once.
    """
    rng = np.random.default_rng(seed)
    pts = np.asarray(pts, float)
    inside_mask = lo = None
    res = 0.1
    if region is not None:
        import cv2

        lo = region.min(0) - region_margin - 1
        size = np.ceil((region.max(0) + region_margin + 1 - lo) / res).astype(int) + 1
        inside_mask = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillPoly(inside_mask, [((region - lo) / res).astype(np.int32)], 1)
        k = max(1, int(round(region_margin / res)))
        inside_mask = cv2.dilate(inside_mask, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], 1)
    remaining = pts.copy()
    found: list[tuple[np.ndarray, float]] = []
    n_total = len(pts)
    while len(remaining) >= max(8, int(min_inlier_frac * n_total)) and len(found) < max_circles:
        m = len(remaining)
        span = max(3, m // 8)
        i = rng.integers(m, size=iters)
        j = (i + rng.integers(1, span, size=iters)) % m
        kk = (i + rng.integers(max(1, span // 2), span * 2, size=iters)) % m
        p1, p2, p3 = remaining[i], remaining[j], remaining[kk]
        d = 2 * (p1[:, 0] * (p2[:, 1] - p3[:, 1]) + p2[:, 0] * (p3[:, 1] - p1[:, 1]) + p3[:, 0] * (p1[:, 1] - p2[:, 1]))
        ok = np.abs(d) > 1e-9
        d = np.where(ok, d, 1.0)
        s1, s2, s3 = (p1**2).sum(1), (p2**2).sum(1), (p3**2).sum(1)
        cx = (s1 * (p2[:, 1] - p3[:, 1]) + s2 * (p3[:, 1] - p1[:, 1]) + s3 * (p1[:, 1] - p2[:, 1])) / d
        cy = (s1 * (p3[:, 0] - p2[:, 0]) + s2 * (p1[:, 0] - p3[:, 0]) + s3 * (p2[:, 0] - p1[:, 0])) / d
        cen = np.stack([cx, cy], 1)
        rad = np.linalg.norm(p1 - cen, axis=1)
        ok &= (rad >= r_range[0]) & (rad <= r_range[1])
        if inside_mask is not None and ok.any():
            circ = cen[:, None, :] + rad[:, None, None] * ring[None]  # (iters,24,2)
            idx = np.floor((circ - lo) / res).astype(int)
            h, w = inside_mask.shape
            valid = (idx[..., 0] >= 0) & (idx[..., 0] < w) & (idx[..., 1] >= 0) & (idx[..., 1] < h)
            vals = np.zeros(valid.shape, bool)
            vals[valid] = inside_mask[idx[..., 1][valid], idx[..., 0][valid]] > 0
            ok &= vals.mean(1) >= 0.9
        if not ok.any():
            break
        cand = np.flatnonzero(ok)
        dist = np.abs(np.linalg.norm(remaining[None] - cen[cand, None, :], axis=2) - rad[cand, None])  # (C,m)
        counts = (dist < tol).sum(1)
        b = int(np.argmax(counts))
        if counts[b] < max(8, min_inlier_frac * n_total):
            break
        inl = dist[b] < tol
        c, r = fit_circle(remaining[inl])  # refine on the inliers
        found.append((c, r))
        remaining = remaining[~inl]
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
