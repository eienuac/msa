"""Run the vision pipeline on a recorded video.

Writes an annotated video (detections, time to impact, HUD readings and a
top-down inset of the projected hazards) and a CSV timeline.

    python tools/analyze_video.py recording.mp4 --out out_dir [--start 0 --end 30 --step 2]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bossai.core.game_state import HazardPhase  # noqa: E402
from bossai.perception.vision import VisionPipeline  # noqa: E402


def draw_inset(state, size: int = 300, extent: float = 12.0) -> np.ndarray:
    img = np.full((size, size, 3), 25, np.uint8)
    s = size / (2 * extent)

    def px(p):
        return int(size / 2 + p[0] * s), int(size / 2 - p[1] * s)

    for r in (3, 6, 9, 12):
        cv2.circle(img, (size // 2, size // 2), int(r * s), (60, 60, 60), 1)
    for h in state.hazards:
        u = 1.0 - min(h.time_to_impact / 2.0, 1.0)
        col = (0, 0, 255) if h.phase == HazardPhase.ACTIVE else (40, 40, int(100 + 155 * u))
        poly = np.array([px(v) for v in h.polygon(32)], np.int32)
        cv2.fillPoly(img, [poly], col)
        cv2.putText(img, f"{h.time_to_impact:.1f}s", px(h.center), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.circle(img, (size // 2, size // 2), 4, (80, 220, 80), -1)
    cv2.putText(img, "top-down (m)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return img


def annotate(frame: np.ndarray, out, pipe: VisionPipeline, t: float) -> np.ndarray:
    vis = frame.copy()
    for tr in out.tracks:
        d = tr.det
        col = (0, 0, 255) if tr.phase == HazardPhase.ACTIVE else (0, 255, 0) if d.kind == "ellipse" else (0, 255, 255)
        cv2.polylines(vis, [d.contour.astype(np.int32)], True, col, 3)
        tti = pipe.tracker.time_to_impact(tr, t)
        x, y, w, h = d.bbox
        cv2.putText(vis, f"#{tr.id} {d.kind} fill={pipe.tracker.current_fill(tr):.2f} tti={tti:.2f}s",
                    (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    hud = out.hud
    lines = [
        f"t={t:6.2f}s",
        f"macro P(ON)={hud.macro_p_on:.2f}",
        f"player HP={hud.player_hp if hud.player_hp is None else round(hud.player_hp, 3)}",
        f"boss HP={hud.boss_hp if hud.boss_hp is None else round(hud.boss_hp, 3)}",
        f"dash ready={hud.dash_ready:.2f}",
    ]
    for i, s in enumerate(lines):
        cv2.putText(vis, s, (20, 180 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(vis, s, (20, 180 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    inset = draw_inset(out.state)
    H, W = vis.shape[:2]
    vis[H - 320 : H - 20, W - 320 : W - 20] = inset
    return vis


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--step", type=int, default=2, help="process every n-th frame")
    ap.add_argument("--scale", type=float, default=0.5, help="output video scale")
    ap.add_argument("--boss", default="kor_bekci", help="profile in configs/bosses/")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    from bossai.perception.vision import VisionConfig

    pipe = VisionPipeline(VisionConfig.for_boss(args.boss))
    stem = Path(args.video).stem
    writer = None
    rows = []
    n = int(args.start * fps)
    times = []
    while True:
        ok, frame = cap.read()
        if not ok or n / fps > args.end:
            break
        n += 1
        if n % args.step:
            continue
        t = n / fps
        t0 = time.perf_counter()
        out = pipe.process(frame, t)
        times.append(time.perf_counter() - t0)
        vis = annotate(frame, out, pipe, t)
        vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale)
        if writer is None:
            writer = cv2.VideoWriter(str(out_dir / f"{stem}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / args.step, (vis.shape[1], vis.shape[0]))
        writer.write(vis)
        rows.append({
            "t": round(t, 3),
            "macro_p_on": round(out.hud.macro_p_on, 3),
            "player_hp": out.hud.player_hp,
            "boss_hp": out.hud.boss_hp,
            "dash_ready": round(out.hud.dash_ready, 2),
            "n_hazards": len(out.state.hazards),
            "hazards": ";".join(
                f"{h.id}:{h.shape.value}:c=({h.center[0]:.1f},{h.center[1]:.1f}):r={h.radius:.1f}:tti={h.time_to_impact:.2f}:{h.phase.name}"
                for h in out.state.hazards
            ),
        })
    if writer is not None:
        writer.release()
    with open(out_dir / f"{stem}_timeline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["t"])
        w.writeheader()
        w.writerows(rows)
    if times:
        print(f"{stem}: {len(rows)} frames, vision {np.mean(times)*1000:.1f} ms/frame (p95 {np.percentile(times, 95)*1000:.1f} ms)")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
