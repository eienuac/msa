"""Camera pitch from the telegraph circles themselves.

A circle on the ground only back-projects to a circle when the assumed camera
pitch is right; with a wrong pitch it becomes an egg shape (too long or too
short in the forward direction). For every confirmed circular telegraph the
residual of a circle fit is computed over a grid of pitches; these curves are
summed with exponential forgetting and the minimum is the estimate. The
forgetting lets the estimate follow the player when they tilt the camera with
right-click drag in the middle of a fight.

Measured on the recordings: Kor Bekçi fight 1 ~19-20 deg, fight 2 ~23 deg,
the spider fights ~20-22 deg (single circles scatter by ~2 deg).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import CameraModel


@dataclass(frozen=True)
class PitchConfig:
    lo: float = 8.0
    hi: float = 45.0
    step: float = 0.5
    max_resid: float = 0.05  # relative circle residual at the best pitch
    min_sharpness: float = 2.0  # residual 4 degrees away / residual at the best pitch
    half_life: float = 30.0  # seconds; old evidence fades so a camera tilt is followed
    min_weight: float = 0.8  # evidence needed before the first estimate (later ones keep the last)
    obs_weight: float = 0.2  # one observation; a telegraph gives ~5-15 of them
    min_interval: float = 0.1  # seconds between observations of the same telegraph
    min_points: int = 12


def circle_residual_curve(px: np.ndarray, cam: CameraModel, pitches: np.ndarray) -> np.ndarray:
    """Relative circle-fit residual of the back-projected pixels, per pitch."""
    px = np.asarray(px, float).reshape(-1, 2)
    dx = (px[:, 0] - cam.cx) / cam.focal_px
    dy = (px[:, 1] - cam.cy) / cam.focal_px
    th = np.radians(pitches)[:, None]
    s, c = np.sin(th), np.cos(th)
    denom = dy[None] * c + s
    valid = (denom > 1e-3).all(1)
    t = 1.0 / np.where(denom > 1e-3, denom, 1.0)  # camera height is only a scale here
    x = t * dx[None]
    y = t * (c - dy[None] * s)
    # Kasa fit per pitch: [x y 1] @ sol = x^2 + y^2
    b = x * x + y * y
    a = np.stack([x, y, np.ones_like(x)], 2)
    ata = np.einsum("pni,pnj->pij", a, a)
    atb = np.einsum("pni,pn->pi", a, b)
    ok = valid & (np.abs(np.linalg.det(ata)) > 1e-12)
    sol = np.zeros((len(pitches), 3))
    sol[ok] = np.linalg.solve(ata[ok], atb[ok][..., None])[..., 0]
    cx, cy = sol[:, 0] / 2, sol[:, 1] / 2
    r = np.sqrt(np.maximum(sol[:, 2] + cx**2 + cy**2, 1e-12))
    d = np.hypot(x - cx[:, None], y - cy[:, None]) - r[:, None]
    res = np.sqrt((d * d).mean(1)) / r
    res[~ok] = np.inf
    return res


class PitchEstimator:
    def __init__(self, cfg: PitchConfig | None = None):
        self.cfg = cfg or PitchConfig()
        c = self.cfg
        self.pitches = np.arange(c.lo, c.hi + 1e-9, c.step)
        self._score = np.zeros(len(self.pitches))
        self._weight = 0.0
        self._t: float | None = None
        self._ready = False
        self._last_obs: dict[int, float] = {}

    def _decay(self, t: float) -> None:
        if self._t is not None and t > self._t:
            f = 0.5 ** ((t - self._t) / self.cfg.half_life)
            self._score *= f
            self._weight *= f
        self._t = t if self._t is None else max(self._t, t)

    def observe(self, px: np.ndarray, cam: CameraModel, t: float, key: int | None = None) -> float | None:
        """Add one circle outline (screen pixels). Returns its own best pitch if it was used."""
        c = self.cfg
        if key is not None:
            if t - self._last_obs.get(key, -1e9) < c.min_interval:
                return None
            self._last_obs[key] = t
            if len(self._last_obs) > 256:
                self._last_obs = {k: v for k, v in self._last_obs.items() if t - v < 10.0}
        if px is None or len(px) < c.min_points:
            return None
        res = circle_residual_curve(px, cam, self.pitches)
        i = int(np.argmin(res))
        k = int(round(4.0 / c.step))
        if not np.isfinite(res[i]) or res[i] > c.max_resid or i < k or i >= len(res) - k:
            return None
        if min(res[i - k], res[i + k]) < c.min_sharpness * max(res[i], 1e-4):
            return None
        self._decay(t)
        curve = np.log(np.minimum(res, 1.0) / max(res[i], 1e-4))
        self._score += c.obs_weight * curve
        self._weight += c.obs_weight
        self._ready = self._ready or self._weight >= c.min_weight
        return float(self.pitches[i])

    def estimate(self, t: float | None = None) -> float | None:
        if t is not None:
            self._decay(t)
        if not self._ready:  # fading evidence does not reset it: the camera rarely moves
            return None
        s = self._score
        i = int(np.argmin(s))
        if 0 < i < len(s) - 1:  # parabolic refinement
            den = s[i - 1] - 2 * s[i] + s[i + 1]
            if den > 1e-12:
                return float(self.pitches[i] + 0.5 * self.cfg.step * (s[i - 1] - s[i + 1]) / den)
        return float(self.pitches[i])

    @property
    def weight(self) -> float:
        return self._weight
