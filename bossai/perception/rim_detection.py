"""Telegraphs drawn as a thin bright ring on top of the ground ("rim" style).

Seen on the spider boss (Örümcek Ana): the warning is a translucent pale
green-yellow disc whose only strong cue is its rim; the arena floor is orange
sand, so a fixed hue window does not work. Instead:

1. "Green shift": (G - R) minus its local median (large blur). The rim is
   greener than the sand around it.
2. Keep bright pixels on bright ground (drops spider legs, shadows, UI).
3. Project the candidate pixels to the ground and find circles with RANSAC on
   the unordered point cloud.
4. Accept a circle only if its visible part is well covered by rim pixels
   (angular coverage), which rejects circles fitted through scattered noise.

Output uses the same ScreenHazard type as the filled-zone detector, so the
tracker and the rest of the pipeline are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import GroundProjector, fit_circle
from .hazard_detection import ScreenHazard
from .ui_reader import UILayout


@dataclass(frozen=True)
class RimStyle:
    min_green_shift: int = 14  # (G-R) above the local median
    min_value: int = 160  # rim pixels are bright (measured: V 173-221)
    max_value: int = 235  # ...but never white-hot: the player's own fire ring is V ~255
    min_sat: int = 115  # rim S 126-154; the fire ring is washed out (S 54-99)
    max_sat: int = 180
    min_bg_value: int = 110  # ...and lie on bright ground
    bg_scale: int = 8  # downscale factor for the local median
    tol: float = 0.35  # ground units, distance of a rim pixel to the circle
    r_range: tuple[float, float] = (1.5, 16.0)
    min_visible_coverage: float = 0.6  # share of the visible circumference with rim pixels
    min_visible_bins: int = 12  # of 48; at least a quarter of the circle must be on screen
    max_circles: int = 4
    iters: int = 700
    max_points: int = 2500
    eval_points: int = 700
    candidates_checked: int = 12
    max_ground_dist: float = 30.0


class RimDetector:
    def __init__(self, style: RimStyle | None = None, layout: UILayout | None = None, projector: GroundProjector | None = None):
        self.style = style or RimStyle()
        self.layout = layout or UILayout()
        self.projector = projector
        self._gm: dict[tuple[int, int], np.ndarray] = {}
        self._rng = np.random.default_rng(0)

    def _ground_mask(self, w: int, h: int) -> np.ndarray:
        if (w, h) not in self._gm:
            self._gm[(w, h)] = self.layout.ground_mask(w, h)
        return self._gm[(w, h)]

    def candidate_mask(self, frame: np.ndarray) -> np.ndarray:
        """Rim pixels: greener than the local ground, bright, moderately saturated.

        Written with saturating uint8 OpenCV operations (a few ms at 1080p;
        the same rule with numpy int16 arrays took ~50 ms).
        """
        st = self.style
        H, W = frame.shape[:2]
        sw, sh = max(8, W // st.bg_scale), max(8, H // st.bg_scale)
        _, g, r = cv2.split(frame)
        gr = cv2.addWeighted(g, 1.0, r, -1.0, 128.0)  # G - R + 128, saturated to 0..255
        bg = cv2.medianBlur(cv2.resize(gr, (sw, sh), interpolation=cv2.INTER_AREA), 9)
        shift = cv2.subtract(gr, cv2.resize(bg, (W, H)))  # (G-R) above its local median, >= 0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat, v = hsv[..., 1], hsv[..., 2]  # V = max(B, G, R)
        vbg = cv2.resize(cv2.medianBlur(cv2.resize(v, (sw, sh), interpolation=cv2.INTER_AREA), 9), (W, H))
        m = cv2.threshold(shift, st.min_green_shift, 255, cv2.THRESH_BINARY)[1]
        m = cv2.bitwise_and(m, cv2.inRange(v, st.min_value, st.max_value))
        m = cv2.bitwise_and(m, cv2.inRange(sat, st.min_sat, st.max_sat))
        m = cv2.bitwise_and(m, cv2.threshold(vbg, st.min_bg_value - 1, 255, cv2.THRESH_BINARY)[1])
        m = cv2.bitwise_and(m, self._ground_mask(W, H))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    def detect(self, frame: np.ndarray) -> list[ScreenHazard]:
        st = self.style
        H, W = frame.shape[:2]
        if self.projector is None or (self.projector.cam.width, self.projector.cam.height) != (W, H):
            from .calibration import CameraModel

            self.projector = GroundProjector(CameraModel().scaled(W, H))
        m = self.candidate_mask(frame)
        nz = cv2.findNonZero(m)
        if nz is None or len(nz) < 40:
            return []
        px = nz.reshape(-1, 2).astype(float)
        if len(px) > st.max_points:
            px = px[self._rng.choice(len(px), st.max_points, replace=False)]
        g = self.projector.to_ground(px)
        ok = ~np.isnan(g).any(1)
        ok[ok] = np.linalg.norm(g[ok], axis=1) < st.max_ground_dist
        g, px = g[ok], px[ok]
        if len(g) < 40:
            return []
        out = []
        for c, rad in self._find_circles(g, m):
            on_rim = np.abs(np.linalg.norm(g - c, axis=1) - rad) < 2 * st.tol
            ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
            poly = self.projector.to_pixels(c + rad * np.stack([np.cos(ang), np.sin(ang)], 1)).astype(np.float32)
            x0, y0 = poly.min(0)
            x1, y1 = poly.max(0)
            area = float(cv2.contourArea(poly))
            out.append(
                ScreenHazard(
                    contour=poly,
                    area=area,
                    bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                    ellipse=cv2.fitEllipse(poly) if len(poly) >= 5 else None,
                    ellipse_iou=1.0,
                    kind="ellipse",
                    rim_px=px[on_rim].astype(np.float32),
                )
            )
        return out

    def _visible_bins(self, c: np.ndarray, r: float, mask: np.ndarray, n: int = 48) -> np.ndarray:
        H, W = mask.shape
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        px = self.projector.to_pixels(c + r * np.stack([np.cos(ang), np.sin(ang)], 1))
        gm = self._ground_mask(W, H)
        vis = np.zeros(n, bool)
        inside = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H) & ~np.isnan(px).any(1)
        idx = np.flatnonzero(inside)
        vis[idx] = gm[px[idx, 1].astype(int), px[idx, 0].astype(int)] > 0
        return vis

    def _find_circles(self, pts: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, float]]:
        st = self.style
        rng = self._rng
        rem = pts.copy()
        found: list[tuple[np.ndarray, float]] = []
        n_bins = 48
        for _ in range(st.max_circles):
            m = len(rem)
            if m < 40:
                break
            idx = rng.integers(m, size=(st.iters, 3))
            p1, p2, p3 = rem[idx[:, 0]], rem[idx[:, 1]], rem[idx[:, 2]]
            d = 2 * (p1[:, 0] * (p2[:, 1] - p3[:, 1]) + p2[:, 0] * (p3[:, 1] - p1[:, 1]) + p3[:, 0] * (p1[:, 1] - p2[:, 1]))
            ok = np.abs(d) > 1e-6
            d = np.where(ok, d, 1.0)
            s1, s2, s3 = (p1**2).sum(1), (p2**2).sum(1), (p3**2).sum(1)
            cx = (s1 * (p2[:, 1] - p3[:, 1]) + s2 * (p3[:, 1] - p1[:, 1]) + s3 * (p1[:, 1] - p2[:, 1])) / d
            cy = (s1 * (p3[:, 0] - p2[:, 0]) + s2 * (p1[:, 0] - p3[:, 0]) + s3 * (p2[:, 0] - p1[:, 0])) / d
            cen = np.stack([cx, cy], 1)
            rad = np.linalg.norm(p1 - cen, axis=1)
            ok &= (rad >= st.r_range[0]) & (rad <= st.r_range[1])
            cand = np.flatnonzero(ok)
            if not len(cand):
                break
            sub = rem[:: max(1, m // st.eval_points)]
            dist = np.abs(np.linalg.norm(sub[None] - cen[cand, None, :], axis=2) - rad[cand, None])
            counts = (dist < st.tol).sum(1)
            best = None
            for b in np.argsort(-counts)[: st.candidates_checked]:
                c, r = cen[cand[b]], rad[cand[b]]
                inl = dist[b] < st.tol
                if inl.sum() < 12:
                    break
                vis = self._visible_bins(c, r, mask, n_bins)
                if vis.sum() < st.min_visible_bins:
                    continue
                a = np.arctan2(sub[inl, 1] - c[1], sub[inl, 0] - c[0])
                hit = np.zeros(n_bins, bool)
                hit[((a + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins] = True
                cov = (hit & vis).sum() / vis.sum()
                if cov >= st.min_visible_coverage:
                    score = inl.sum() * cov
                    if best is None or score > best[0]:
                        best = (score, c, r)
            if best is None:
                break
            _, c, r = best
            inl_all = np.abs(np.linalg.norm(rem - c, axis=1) - r) < st.tol
            c, r = fit_circle(rem[inl_all])
            if any(np.linalg.norm(c - c2) < 1.0 and abs(r - r2) < 1.0 for c2, r2 in found):
                rem = rem[~inl_all]
                continue
            found.append((c, r))
            rem = rem[~inl_all]
        return found
