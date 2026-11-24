"""Live / replay runtime: screen -> vision -> policy -> (optional) keys.

Modes
  shadow (default): nothing is sent to the game; the overlay shows what the
                    agent would do. Works on recorded videos too.
  control:          keys are sent with plain SendInput (Windows). Use only
                    where automation is allowed; there is no anti-cheat bypass.

Safety: F12 (Windows) or Ctrl+C stops the agent and releases every key.
Keys are only sent while the game window has focus.

Examples
  python -m bossai.runtime.live --model models/kor_bekci_v1 --source recording.mp4 --out shadow.mp4
  python -m bossai.runtime.live --model models/kor_bekci_v1 --source dxcam --show
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from ..agent.action_space import MOVE_NAMES, action_masks, decode
from ..agent.observation_encoder import AgentContext, ObservationEncoder
from ..capture.screen_capture import open_source
from ..control.action_controller import ActionController, AgentIntent, ControllerConfig
from ..control.input_backend import NullBackend
from ..control.macro_state import MacroEvidence, MacroStateManager
from ..control.shield import ShieldReflex
from ..core.safety import EscapeParams, SafetyPlanner
from ..perception.vision import VisionConfig, VisionPipeline
from ..training.registry import load_model

DECISION_DT = 0.1


def kill_switch_pressed() -> bool:
    if sys.platform != "win32":
        return False
    import ctypes

    return bool(ctypes.windll.user32.GetAsyncKeyState(0x7B) & 0x8000)  # F12


def overlay(
    frame: np.ndarray, intent, macro, out, foot=(958.0, 668.0), pitch: float | None = None, shield_on: bool = False
) -> np.ndarray:
    vis = frame.copy()
    lines = [
        f"agent: move={MOVE_NAMES[intent.move]} dash={int(intent.dash)} macro_want={['keep', 'ON', 'OFF'][intent.macro]}",
        f"macro state={macro.state.name} P(ON)={out.hud.macro_p_on:.2f}",
        f"boss dist={'?' if out.state.boss.pos is None else f'{out.state.boss_distance():.0f} m'} "
        f"target_is_boss={out.state.boss.targeted} dash charges={out.state.player.dash_charges} "
        f"ready={out.state.player.can_dash}",
        f"hazards={len(out.state.hazards)} " + " ".join(f"tti={h.time_to_impact:.1f}" for h in out.state.hazards[:4]),
        f"camera pitch={pitch:.1f}" if pitch is not None else "camera pitch=default",
        f"shield(E)={'HOLD' if shield_on else '-'}",
    ]
    for i, s in enumerate(lines):
        cv2.putText(vis, s, (520, 180 + 36 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(vis, s, (520, 180 + 36 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    if intent.move:
        from ..core.game_state import DIRECTION_VECTORS

        c = np.array(foot, int)
        d = DIRECTION_VECTORS[intent.move] * np.array([1, -1])
        cv2.arrowedLine(vis, tuple(c), tuple((c + d * 160).astype(int)), (0, 255, 255), 6, tipLength=0.3)
    for tr in out.tracks:
        cv2.polylines(vis, [tr.det.contour.astype(np.int32)], True, (0, 255, 0), 3)
    return vis


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", default="dxcam", help="video path, 'dxcam' or 'mss'")
    ap.add_argument("--control", action="store_true", help="send keys (Windows only)")
    ap.add_argument("--window-title", default=None, help="only send keys while this window has focus")
    ap.add_argument("--out", default=None, help="write the overlay video here")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=1e9)
    ap.add_argument("--boss", default="kor_bekci", help="profile in configs/bosses/")
    args = ap.parse_args(argv)

    model = load_model(args.model)
    encoder = ObservationEncoder()
    vision = VisionPipeline(VisionConfig.for_boss(args.boss, ocr_in_background=True))
    planner = SafetyPlanner(EscapeParams())
    macro = MacroStateManager()
    if args.control:
        from ..control.input_backend import WindowsSendInputBackend

        backend = WindowsSendInputBackend(args.window_title)
    else:
        backend = NullBackend()
    controller = ActionController(backend, macro, ControllerConfig())
    ctx = AgentContext()
    shield = ShieldReflex()  # E when a following ring is about to land (spider)
    is_file = args.source not in ("dxcam", "mss")
    src = open_source(args.source, step=2 if is_file else 1)  # recordings are 60 fps; 30 fps is plenty
    writer = None
    next_decision = 0.0
    prev_boss = 1.0
    intent = AgentIntent()
    try:
        while True:
            got = src.read()
            if got is None:
                if is_file:
                    break
                continue
            frame, t = got
            if t > args.max_seconds or kill_switch_pressed():
                break
            out = vision.process(frame, t)
            controller.service(t)
            if t >= next_decision:
                next_decision = t + DECISION_DT
                s = out.state
                ev = MacroEvidence(
                    indicator_on=out.hud.macro_p_on,
                    damage_dealt=bool(prev_boss - s.boss.hp > 1e-3),
                    attack_expected=s.in_melee_range() and s.boss.targeted and ctx.move_dir == 0,
                    safe_to_probe=not any(h.contains(np.zeros((1, 2)), margin=0.8)[0] for h in s.hazards),
                    can_turn_on=s.boss.targeted or s.in_melee_range(),
                )
                prev_boss = s.boss.hp
                obs = encoder.encode(s, macro.state, ctx)
                masks = action_masks(s, macro.state)
                action, _ = model.predict(obs, action_masks=masks, deterministic=True)
                intent, _ = decode(np.asarray(action), s, planner, ctx.move_dir)
                rep = controller.apply(t, intent, ev)
                controller.set_shield(shield.update(t, s))
                ctx.update(t, controller.current_dir, rep.r_pressed, DECISION_DT)
            if args.out or args.show:
                vis = overlay(frame, intent, macro, out, vision.camera.foot_px, vision.pitch.estimate(), controller.shield_held)
                if args.out:
                    small = cv2.resize(vis, None, fx=0.5, fy=0.5)
                    if writer is None:
                        out_fps = getattr(src, "fps", 60.0) / getattr(src, "step", 1) if is_file else 30
                        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (small.shape[1], small.shape[0]))
                    writer.write(small)
                if args.show:
                    cv2.imshow("boss agent", cv2.resize(vis, None, fx=0.5, fy=0.5))
                    if cv2.waitKey(1) == 27:
                        break
    except KeyboardInterrupt:
        pass
    finally:
        controller.release_all()
        src.close()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
