"""Reads fixed HUD elements: R macro indicator, HP bars, dash icon.

Regions are measured on 1920x1080 recordings and scaled for other sizes.
Measured on the Kor Bekçi recordings:
* R icon border has a yellow frame when the macro is ON (yellow ratio
  ~0.09 ON vs 0.000 OFF) and the icon itself is brighter (V ~123 vs ~76).
* HP bars: bright red fill; the empty part is a dark red (R ~85, G ~0).
  The player bar animates towards the new value, lagging the number by
  ~0.3 s; the boss bar matched the HP numbers within ~1.5%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x0, y0, x1, y1


@dataclass(frozen=True)
class UILayout:
    width: int = 1920
    height: int = 1080
    r_icon: Box = (563, 966, 606, 1009)
    q_icon: Box = (611, 969, 647, 1006)
    player_hp_bar: Box = (113, 55, 292, 64)  # x0, y0, x1, y1 of the full bar
    boss_hp_bar: Box = (712, 65, 1140, 73)
    # areas never searched for ground hazards (HUD, minimap, chat, quest panel)
    hud_masks: tuple[Box, ...] = (
        (0, 0, 1920, 135),
        (0, 935, 1920, 1080),
        (1480, 0, 1920, 1080),
        (20, 460, 305, 620),
        (60, 760, 545, 945),
    )

    def scaled(self, w: int, h: int) -> "UILayout":
        if (w, h) == (self.width, self.height):
            return self
        sx, sy = w / self.width, h / self.height

        def s(b: Box) -> Box:
            return (int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy))

        return UILayout(
            w, h, s(self.r_icon), s(self.q_icon), s(self.player_hp_bar), s(self.boss_hp_bar), tuple(s(b) for b in self.hud_masks)
        )

    def ground_mask(self, w: int, h: int) -> np.ndarray:
        lay = self.scaled(w, h)
        m = np.full((h, w), 255, np.uint8)
        for x0, y0, x1, y1 in lay.hud_masks:
            m[y0:y1, x0:x1] = 0
        return m


@dataclass
class HudReading:
    macro_p_on: float
    macro_yellow: float
    macro_icon_value: float
    player_hp: float | None
    boss_hp: float | None
    dash_ready: float  # 0..1 brightness-based readiness score


def _yellow_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 15) & (h <= 40) & (s >= 110) & (v >= 140)


def macro_indicator(frame: np.ndarray, lay: UILayout, band: int = 5) -> tuple[float, float, float]:
    """Returns (P(ON), yellow border ratio, icon brightness)."""
    x0, y0, x1, y1 = lay.r_icon
    patch = frame[y0 - band : y1 + band, x0 - band : x1 + band]
    ring = np.ones(patch.shape[:2], bool)
    ring[2 * band : -2 * band, 2 * band : -2 * band] = False
    yellow = float(_yellow_mask(patch)[ring].mean())
    inner = frame[y0 + 6 : y1 - 6, x0 + 6 : x1 - 6]
    value = float(cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)[..., 2].mean())
    # logistic combination; yellow frame is the main cue, icon brightness secondary
    z = (yellow - 0.02) / 0.008 + (value - 100.0) / 8.0
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))), yellow, value


def bar_fraction(frame: np.ndarray, bar: Box, min_red: int = 100) -> float | None:
    """Fraction of a horizontal HP bar that is filled with bright red."""
    x0, y0, x1, y1 = bar
    roi = frame[y0:y1, x0 - 3 : x1 + 3].astype(np.int16)
    b, g, r = roi[..., 0], roi[..., 1], roi[..., 2]
    red = (r > min_red) & (r > g + 60) & (r > b + 40)
    col = red.mean(0)
    if col.max() < 0.3:
        return None  # bar not visible (e.g. no target)
    smooth = np.convolve(col > 0.4, np.ones(5) / 5, mode="same") > 0.5
    idx = np.flatnonzero(smooth)
    if len(idx) == 0:
        return 0.0
    end = idx.max() + x0 - 3
    return float(np.clip((end - x0) / max(x1 - x0, 1), 0.0, 1.0))


def dash_ready(frame: np.ndarray, lay: UILayout) -> float:
    """Ready icons are bright and saturated; cooldown icons are darkened."""
    x0, y0, x1, y1 = lay.q_icon
    v = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
    return float(np.clip((np.percentile(v, 90) - 120.0) / 80.0, 0.0, 1.0))


class HudReader:
    def __init__(self, layout: UILayout | None = None):
        self.base = layout or UILayout()
        self._lay: UILayout | None = None

    def read(self, frame: np.ndarray) -> HudReading:
        h, w = frame.shape[:2]
        if self._lay is None or (self._lay.width, self._lay.height) != (w, h):
            self._lay = self.base.scaled(w, h)
        lay = self._lay
        p, yel, val = macro_indicator(frame, lay)
        return HudReading(
            macro_p_on=p,
            macro_yellow=yel,
            macro_icon_value=val,
            player_hp=bar_fraction(frame, lay.player_hp_bar),
            boss_hp=bar_fraction(frame, lay.boss_hp_bar),
            dash_ready=dash_ready(frame, lay),
        )
