"""Setup check for the live test on the user's PC.

Run it with the game open, standing near the boss:

    python tools/check_setup.py --seconds 20

It checks the Python packages, grabs the screen, runs the vision pipeline and
prints what it reads every 0.5 s (macro, HP, dash, boss distance, warning
zones). A few annotated frames are saved to check_out/ so they can be sent
back for inspection. Nothing is sent to the game.

Offline (e.g. in the cloud) on a recording:

    python tools/check_setup.py --video recording.mp4 --seconds 10
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_packages() -> bool:
    ok = True
    print(f"python {platform.python_version()} on {platform.system()} {platform.release()}")
    if sys.version_info < (3, 10):
        print("  !! Python 3.10+ required")
        ok = False
    for mod, need in [
        ("numpy", True),
        ("cv2", True),
        ("torch", True),
        ("gymnasium", True),
        ("stable_baselines3", True),
        ("sb3_contrib", True),
        ("rapidocr_onnxruntime", False),
        ("dxcam", False),
        ("mss", False),
    ]:
        try:
            m = importlib.import_module(mod)
            print(f"  ok   {mod} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {'!!' if need else '--'}   {mod} missing ({type(e).__name__})")
            ok = ok and not need
    return ok


def open_capture(video: str | None):
    from bossai.capture.screen_capture import open_source

    if video:
        return open_source(video, step=2), "video"
    for name in ("dxcam", "mss"):
        try:
            src = open_source(name)
            for _ in range(20):
                got = src.read()
                if got is not None:
                    return src, name
                time.sleep(0.05)
            src.close()
        except Exception as e:  # noqa: BLE001
            print(f"  capture {name} failed: {e}")
    raise SystemExit("no screen capture backend works (install dxcam or mss)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default="check_out")
    args = ap.parse_args(argv)

    print("== packages")
    if not check_packages():
        raise SystemExit("install the missing packages: pip install -e .[windows,ocr]")

    from bossai.perception.vision import VisionConfig, VisionPipeline

    print("== capture")
    src, name = open_capture(args.video)
    frame, t0 = src.read()
    h, w = frame.shape[:2]
    print(f"  backend={name} frame={w}x{h}")
    if (w, h) != (1920, 1080):
        print("  !! HUD positions were measured at 1920x1080; other sizes are scaled and may be off")

    print("== vision (what the agent would see)")
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    pipe = VisionPipeline(VisionConfig(ocr_in_background=True))
    if pipe.boss_reader is None or not pipe.boss_reader.available:
        print("  -- OCR not available: boss distance will be missing (pip install rapidocr_onnxruntime)")
    times = []
    next_print = 0.0
    saved = 0
    start = time.perf_counter()
    while True:
        got = src.read()
        if got is None:
            if args.video:
                break
            continue
        frame, t = got
        rel = t - t0
        if rel > args.seconds:
            break
        t1 = time.perf_counter()
        o = pipe.process(frame, t)
        times.append(time.perf_counter() - t1)
        if rel >= next_print:
            next_print = rel + 0.5
            s = o.state
            dist = "?" if s.boss.pos is None else f"{s.boss_distance():.0f}m"
            print(
                f"  t={rel:5.1f}s macroP(ON)={o.hud.macro_p_on:.2f} HP={s.player.hp:.2f} bossHP={s.boss.hp:.2f} "
                f"dash={s.player.dash_charges}{'' if s.player.can_dash else '(locked)'} boss={dist} "
                f"target_is_boss={s.boss.targeted} zones={len(s.hazards)} "
                + " ".join(f"[{h.shape.value} r={h.radius:.1f} tti={h.time_to_impact:.1f}]" for h in s.hazards[:3])
            )
            if saved < 6 and (rel > saved * args.seconds / 6):
                vis = frame.copy()
                for tr in o.tracks:
                    cv2.polylines(vis, [tr.det.contour.astype(np.int32)], True, (0, 255, 0), 3)
                cv2.imwrite(str(out_dir / f"frame_{saved}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
                saved += 1
    src.close()
    if times:
        ms = np.array(times) * 1000
        print(f"== speed: vision {ms.mean():.1f} ms/frame (p95 {np.percentile(ms, 95):.1f} ms), "
              f"{len(times) / max(time.perf_counter() - start, 1e-6):.1f} frames/s processed")
        if ms.mean() > 50:
            print("  !! slower than 20 frames/s; decisions will lag")
    print(f"== saved {saved} frames to {out_dir}/")


if __name__ == "__main__":
    main()
