"""Frame sources with one interface: `read() -> (frame_bgr, timestamp) | None`.

* VideoFileSource: recorded gameplay, replayed at its own timestamps
  (used for offline testing and shadow-mode demos).
* DXCamSource (Windows): Desktop Duplication API, low latency.
* MSSSource: portable fallback (slower; ~15-30 ms per 1080p grab on Windows).

Only the newest frame matters for control; sources never queue old frames.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


class VideoFileSource:
    def __init__(self, path: str, start: float = 0.0, step: int = 1):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        self.step = max(1, step)
        self.n = int(start * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.n)

    def read(self) -> tuple[np.ndarray, float] | None:
        frame = None
        for _ in range(self.step):
            ok, frame = self.cap.read()
            if not ok:
                return None
            self.n += 1
        return frame, self.n / self.fps

    def close(self) -> None:
        self.cap.release()


class MSSSource:  # pragma: no cover - needs a display
    def __init__(self, region: tuple[int, int, int, int] | None = None):
        import mss

        self.sct = mss.mss()
        mon = self.sct.monitors[1]
        if region:
            x, y, w, h = region
            mon = {"left": x, "top": y, "width": w, "height": h}
        self.mon = mon
        self.t0 = time.perf_counter()

    def read(self) -> tuple[np.ndarray, float] | None:
        img = np.asarray(self.sct.grab(self.mon))[:, :, :3]
        return np.ascontiguousarray(img), time.perf_counter() - self.t0

    def close(self) -> None:
        self.sct.close()


class DXCamSource:  # pragma: no cover - Windows only
    def __init__(self, region: tuple[int, int, int, int] | None = None, target_fps: int = 60):
        import dxcam

        self.cam = dxcam.create(output_color="BGR")
        reg = None
        if region:
            x, y, w, h = region
            reg = (x, y, x + w, y + h)
        self.cam.start(region=reg, target_fps=target_fps, video_mode=False)
        self.t0 = time.perf_counter()

    def read(self) -> tuple[np.ndarray, float] | None:
        frame = self.cam.get_latest_frame()
        if frame is None:
            return None
        return frame, time.perf_counter() - self.t0

    def close(self) -> None:
        self.cam.stop()


def open_source(spec: str, region: tuple[int, int, int, int] | None = None, step: int = 1):
    """spec: a video path, 'dxcam' or 'mss'."""
    if spec == "dxcam":
        return DXCamSource(region)
    if spec == "mss":
        return MSSSource(region)
    return VideoFileSource(spec, step=step)
