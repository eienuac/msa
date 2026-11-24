"""Reads the target frame text above the boss HP bar: "Sv. 22 Kor Bekçi | ▲ 5 m arkanda".

Uses RapidOCR (ONNX, models bundled with the pip package) on a fixed text
band. Measured on the recordings: 30/33 distances read correctly from single
frames; misreads are single-frame ("3" read as "5", the arrow read as "4"), so
a median over the last readings is used. OCR costs ~100 ms on a laptop CPU,
so it runs at most every `interval` seconds (distance changes slowly).

Not read yet: the arrow direction (needs a dedicated glyph detector).
"""

from __future__ import annotations

import re
import threading
import unicodedata
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

BAND = (700, 34, 1220, 62)  # x0, y0, x1, y1 at 1920x1080
_DIST_RE = re.compile(r"(\d{1,2})\s?m(?![a-z]{2})")


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


@dataclass
class BossInfo:
    text: str
    distance_m: float | None
    is_boss: bool  # target name matches the boss (not an add)
    behind: bool  # "arkanda" shown


def parse_target_text(text: str, boss_keywords: tuple[str, ...]) -> BossInfo:
    folded = _fold(text)
    m = _DIST_RE.findall(text.replace(",", " "))
    dist = None
    if m:
        d = int(m[-1])
        if d >= 30:  # an arrow glyph read as a leading digit ("42" for "<2"); arena is < 30 m
            d = d % 10
        dist = float(d)
    is_boss = any(k in folded for k in boss_keywords)
    return BossInfo(text, dist, is_boss, "arkand" in folded)


class BossInfoReader:
    def __init__(
        self,
        boss_keywords: tuple[str, ...] = ("kor be",),
        interval: float = 0.3,
        history: int = 3,
        background: bool = False,
    ):
        """background=True runs OCR in a worker thread so the frame loop never waits for it."""
        self.keywords = tuple(_fold(k) for k in boss_keywords)
        self.interval = interval
        self.background = background
        self._lock = threading.Lock()
        self._busy = False
        self._last_t = -1e9
        self._dists: deque[float] = deque(maxlen=history)
        self._last: BossInfo | None = None
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR()
        except Exception:  # optional dependency
            self._ocr = None

    @property
    def available(self) -> bool:
        return self._ocr is not None

    def read_text(self, frame: np.ndarray) -> str:
        h, w = frame.shape[:2]
        sx, sy = w / 1920, h / 1080
        x0, y0, x1, y1 = BAND
        band = frame[int(y0 * sy) : int(y1 * sy), int(x0 * sx) : int(x1 * sx)]
        band = cv2.resize(band, ((x1 - x0) * 2, (y1 - y0) * 2), interpolation=cv2.INTER_CUBIC)
        res, _ = self._ocr(band, use_det=False, use_cls=False, use_rec=True)
        if not res:
            return ""
        return " ".join(r[1] if len(r) == 3 else r[0] for r in res)

    def _update(self, frame: np.ndarray) -> None:
        info = parse_target_text(self.read_text(frame), self.keywords)
        with self._lock:
            if info.distance_m is not None:
                self._dists.append(info.distance_m)
            smoothed = float(np.median(self._dists)) if self._dists else None
            self._last = BossInfo(info.text, smoothed, info.is_boss, info.behind)
            self._busy = False

    def read(self, frame: np.ndarray, t: float) -> BossInfo | None:
        if self._ocr is None:
            return None
        due = t - self._last_t >= self.interval or self._last is None
        if due and not self._busy:
            self._last_t = t
            if self.background:
                self._busy = True
                band = frame.copy()
                threading.Thread(target=self._update, args=(band,), daemon=True).start()
            else:
                self._update(frame)
        with self._lock:
            return self._last
