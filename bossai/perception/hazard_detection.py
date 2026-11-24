"""Ground telegraph detection on game frames.

Pipeline (half resolution for speed):
  colour mask -> HUD mask -> morphology -> components -> filled outer contour
  -> ellipse fit + IoU (circle-like or generic polygon)
  -> radial colour profile (for circles: how far the inner fill has grown)

Telegraph colour differs per boss, so it lives in a small per-boss config.
Measured on Kor Bekçi: the outer area appears at full size at once, a darker
red fill grows from the centre and reaches the rim at impact (~1.0-1.5 s).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .ui_reader import UILayout


@dataclass(frozen=True)
class TelegraphColor:
    """Colour rule for one boss's telegraphs.

    A pixel belongs to a telegraph when its colour contrast is high enough
    AND its hue lies in `hue_window` (OpenCV hue, may be negative to wrap
    around red) AND its saturation is not above `max_sat`.
    Kor Bekçi (measured): telegraph hue -1..4, sat 85-150. False positives
    seen in recordings: boss lava/fire hue 7-19, tree bark hue 12-16 sat 150-200.

    contrast "redness": R - max(G, B); "saturation": S*V/255 (other colours).
    """

    contrast: str = "redness"
    min_contrast: int = 35  # "strong" pixels: seed a telegraph
    weak_contrast: int = 20  # "weak" pixels: the faint tint, kept only when attached to strong ones
    min_value: int = 90
    hue_window: tuple[int, int] = (-5, 6)
    max_sat: int = 200

    def mask(self, bgr: np.ndarray, min_contrast: int | None = None) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = hsv[..., 0].astype(np.int16)
        lo, hi = self.hue_window
        if lo < 0:
            hue_ok = (h >= 180 + lo) | (h <= hi)
        else:
            hue_ok = (h >= lo) & (h <= hi)
        ok = hue_ok & (hsv[..., 1] <= self.max_sat) & (hsv[..., 2] >= self.min_value)
        ok &= self.strength(bgr) >= (self.min_contrast if min_contrast is None else min_contrast)
        return ok.astype(np.uint8) * 255

    def strength(self, bgr: np.ndarray) -> np.ndarray:
        """Continuous colour strength (also used for the fill profile)."""
        if self.contrast == "redness":
            b, g, r = cv2.split(bgr.astype(np.float32))
            return r - np.maximum(g, b)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        return hsv[..., 1] * hsv[..., 2] / 255.0


@dataclass
class ScreenHazard:
    contour: np.ndarray  # (N,2) full-resolution pixels, outer boundary
    area: float  # full-resolution pixels
    bbox: tuple[int, int, int, int]  # x, y, w, h (full resolution)
    ellipse: tuple | None = None  # cv2 ellipse ((cx,cy),(w,h),angle), full resolution
    ellipse_iou: float = 0.0
    kind: str = "polygon"  # "ellipse" or "polygon"
    profile: np.ndarray | None = None  # colour strength per normalised radius bin (ellipses)
    mean_strength: float = 0.0
    strength_sample: np.ndarray | None = None  # colour strength of up to 4000 region pixels
    raw_contour: np.ndarray | None = None  # outline without convex hulls (keeps circle arcs)

    @property
    def center(self) -> np.ndarray:
        if self.ellipse is not None and self.kind == "ellipse":
            return np.array(self.ellipse[0])
        return self.contour.mean(0)


@dataclass
class DetectorConfig:
    color: TelegraphColor = field(default_factory=TelegraphColor)
    scale: float = 0.5
    min_area: float = 6000.0  # full-resolution pixels
    max_area_frac: float = 0.6
    ellipse_iou_min: float = 0.78
    close_kernel: int = 7
    profile_bins: int = 10
    min_strong_px: int = 400  # half-resolution pixels of strong colour a region must contain
    weak_min_solidity: float = 0.85  # faint-tint blobs must be this convex to be used


class HazardDetector:
    def __init__(self, cfg: DetectorConfig | None = None, layout: UILayout | None = None):
        self.cfg = cfg or DetectorConfig()
        self.layout = layout or UILayout()
        self._mask_cache: dict[tuple[int, int], np.ndarray] = {}

    def _ground_mask(self, w: int, h: int, sw: int, sh: int) -> np.ndarray:
        key = (sw, sh)
        if key not in self._mask_cache:
            self._mask_cache[key] = cv2.resize(self.layout.ground_mask(w, h), (sw, sh), interpolation=cv2.INTER_NEAREST)
        return self._mask_cache[key]

    def detect(self, frame: np.ndarray) -> list[ScreenHazard]:
        c = self.cfg
        H, W = frame.shape[:2]
        s = c.scale
        small = cv2.resize(frame, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        gm = self._ground_mask(W, H, sw, sh)
        regions = self._regions(small, gm)
        strength = None
        out: list[ScreenHazard] = []
        min_a = c.min_area * s * s
        max_a = c.max_area_frac * sw * sh
        for region in regions:
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < min_a or area > max_a:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 1.6 * w:  # ground decals look flattened; tall shapes are objects
                continue
            filled = region[y : y + h, x : x + w]
            ell, iou = None, 0.0
            if len(cnt) >= 5:
                ell = cv2.fitEllipse(cnt)
                em = np.zeros((h, w), np.uint8)
                (ex, ey), (ew, eh), ea = ell
                cv2.ellipse(em, ((ex - x, ey - y), (ew, eh), ea), 255, -1)
                inter = np.count_nonzero(filled & em)
                union = np.count_nonzero(filled | em)
                iou = inter / max(union, 1)
            kind = "ellipse" if ell is not None and iou >= c.ellipse_iou_min else "polygon"
            if strength is None:
                strength = c.color.strength(small)
            vals = strength[y : y + h, x : x + w][filled > 0]
            haz = ScreenHazard(
                contour=cnt.reshape(-1, 2).astype(np.float32) / s,
                area=area / (s * s),
                bbox=(int(x / s), int(y / s), int(w / s), int(h / s)),
                ellipse=None if ell is None else ((ell[0][0] / s, ell[0][1] / s), (ell[1][0] / s, ell[1][1] / s), ell[2]),
                ellipse_iou=float(iou),
                kind=kind,
                mean_strength=float(np.median(vals)) if vals.size else 0.0,
                strength_sample=vals[:: max(1, vals.size // 4000)].astype(np.float32) if vals.size else None,
            )
            if kind == "ellipse":
                haz.profile = self._radial_profile(strength, gm, ell, c.profile_bins)
            if self._raw is not None:
                rc, _ = cv2.findContours(cv2.bitwise_and(self._raw, region), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if rc:
                    haz.raw_contour = max(rc, key=cv2.contourArea).reshape(-1, 2).astype(np.float32) / s
            out.append(haz)
        return out

    _raw: np.ndarray | None = None

    def _regions(self, small: np.ndarray, gm: np.ndarray) -> list[np.ndarray]:
        """Build one filled region per telegraph.

        * Strong-colour blobs (rims, fills) are replaced by their convex hulls,
          which closes rims broken by the character standing on them.
        * Faint-tint blobs that contain strong pixels and are convex as a whole
          (a cone, a circle) contribute their hull too; this recovers the
          unfilled part of a cone. Non-convex tint blobs (several circles joined
          by reddish floor cracks) are ignored so safe gaps stay safe.
        * Overlapping hulls merge into one region.
        """
        c = self.cfg
        k = np.ones((c.close_kernel, c.close_kernel), np.uint8)
        strong = cv2.bitwise_and(c.color.mask(small), gm)
        strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, k)
        strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        min_part = c.min_area * c.scale * c.scale * 0.05
        canvas = np.zeros_like(strong)
        raw = np.zeros_like(strong)  # strong blobs with holes filled, no hulls (keeps arcs)
        contours, _ = cv2.findContours(strong, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_part:
                cv2.drawContours(canvas, [cv2.convexHull(cnt)], -1, 255, -1)
                cv2.drawContours(raw, [cnt], -1, 255, -1)
        self._raw = raw
        if not canvas.any():
            return []
        weak = cv2.bitwise_and(c.color.mask(small, c.color.weak_contrast), gm)
        weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, k)
        weak = cv2.morphologyEx(weak, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        wcont, _ = cv2.findContours(weak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in wcont:
            area = cv2.contourArea(cnt)
            if area < c.min_area * c.scale * c.scale:
                continue
            hull = cv2.convexHull(cnt)
            if area / max(cv2.contourArea(hull), 1.0) < c.weak_min_solidity:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if np.count_nonzero(strong[y : y + h, x : x + w]) < c.min_strong_px:
                continue
            cv2.drawContours(canvas, [hull], -1, 255, -1)
        n, labels = cv2.connectedComponents(canvas)
        return [((labels == i).astype(np.uint8) * 255) for i in range(1, n)]

    @staticmethod
    def _radial_profile(strength: np.ndarray, gm: np.ndarray, ell, bins: int) -> np.ndarray:
        (cx, cy), (w, h), ang = ell
        a, b = max(w, h) / 2, min(w, h) / 2
        x0, x1 = int(max(cx - a, 0)), int(min(cx + a + 1, strength.shape[1]))
        y0, y1 = int(max(cy - a, 0)), int(min(cy + a + 1, strength.shape[0]))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        t = np.deg2rad(ang)
        # rotate into the ellipse frame (OpenCV angle refers to the first axis)
        dx, dy = xx - cx, yy - cy
        u = dx * np.cos(t) + dy * np.sin(t)
        v = -dx * np.sin(t) + dy * np.cos(t)
        ax_u, ax_v = (w / 2, h / 2)
        rho = np.sqrt((u / max(ax_u, 1e-6)) ** 2 + (v / max(ax_v, 1e-6)) ** 2)
        sub = strength[y0:y1, x0:x1]
        ok = gm[y0:y1, x0:x1] > 0
        prof = np.zeros(bins, np.float32)
        edges = np.linspace(0, 1, bins + 1)
        for i in range(bins):
            sel = ok & (rho >= edges[i]) & (rho < edges[i + 1])
            prof[i] = float(np.median(sub[sel])) if sel.any() else np.nan
        return prof


def fill_from_profile(profile: np.ndarray, baseline: float | None, delta: float = 7.0) -> float:
    """Fraction of the radius covered by the growing inner fill.

    Counts consecutive bins from the centre whose colour strength exceeds the
    unfilled tint (`baseline`, measured when the telegraph first appeared).
    The outermost bin is the rim and is ignored.
    """
    p = np.asarray(profile, float)[:-1]
    if np.all(np.isnan(p)):
        return 0.0
    base = baseline if baseline is not None else float(np.nanmin(p))
    above = np.nan_to_num(p, nan=base) > base + delta
    n = 0
    for a in above:
        if not a:
            break
        n += 1
    return n / len(p)


def fill_from_area(strength_sample: np.ndarray | None, baseline: float, delta: float = 7.0) -> float:
    """Shape-agnostic fill estimate.

    The inner fill grows from the centre (circles) or apex (cones), so the
    filled area grows with the square of the front position. sqrt(filled
    area fraction) therefore approximates how far the front has travelled.
    """
    if strength_sample is None or strength_sample.size == 0:
        return 0.0
    frac = float(np.mean(strength_sample > baseline + delta))
    return float(np.sqrt(frac))

