"""Top-down debug rendering of the simulator (RGB array, 720x720).

Legend
  green dot          player (yellow ring = macro ON)
  green thin circle  melee range
  orange disc        boss (pulsing outline while it winds up the summon burst)
  red filled shapes  warning zones; the redder, the closer the impact; number = seconds left
  bright red         zone exploding right now
  cyan outline       what the vision system would report (noise, misses, delays)
  yellow arrow       the agent's chosen direction (thick = dash)
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.game_state import DIRECTION_VECTORS, GameState, HazardPhase, MacroState

SIZE = 720
HUD = 150
MOVE_NAMES = ["-", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _px(p: np.ndarray, scale: float) -> tuple[int, int]:
    return int(SIZE / 2 + p[0] * scale), int(SIZE / 2 - p[1] * scale)


def render_topdown(world, observed: GameState, macro: MacroState, intent, stats) -> np.ndarray:
    scale = (SIZE / 2 - 10) / world.arena_r
    img = np.full((SIZE + HUD, SIZE, 3), 25, np.uint8)
    cv2.circle(img, (SIZE // 2, SIZE // 2), int(world.arena_r * scale), (62, 58, 52), -1)
    for r in range(5, int(world.arena_r), 5):  # 5 m grid rings
        cv2.circle(img, (SIZE // 2, SIZE // 2), int(r * scale), (72, 68, 62), 1)

    def poly(h):
        return np.array([_px(v, scale) for v in h.polygon(40)], np.int32)

    overlay = img.copy()
    labels = []
    for sh in world.hazards:
        h = sh.hz
        if sh.hidden:
            continue
        if h.phase == HazardPhase.ACTIVE:
            col = (40, 40, 255)
        else:
            u = 1.0 - min(h.time_to_impact / 2.0, 1.0)
            col = (50, 50, int(110 + 145 * u))
            labels.append((h, f"{h.time_to_impact:.1f}"))
        cv2.fillPoly(overlay, [poly(h)], col)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    for h in observed.hazards:
        cv2.polylines(img, [poly(h)], True, (255, 230, 0) if h.confidence >= 1 else (255, 0, 255), 1)
    for h, text in labels:
        c = _px(h.center if h.shape.value == "circle" else h.polygon(12).mean(0), scale)
        cv2.putText(img, text, (c[0] - 14, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    bp = _px(world.boss_pos, scale)
    br = max(4, int(world.boss_hitbox * scale))
    cv2.circle(img, bp, br, (40, 140, 240), -1)
    if world.windup > 0:
        cv2.circle(img, bp, br + 4 + int(6 * world.windup), (0, 220, 255), 3)
        cv2.putText(img, "kollar kalkti!", (bp[0] - 50, bp[1] - br - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "BOSS", (bp[0] - 18, bp[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    pp = _px(world.player_pos, scale)
    cv2.circle(img, pp, int((world.melee_range + world.boss_hitbox) * scale), (90, 200, 90), 1)
    if world.macro_on:
        cv2.circle(img, pp, max(6, int(0.9 * scale)), (0, 230, 255), 2)
    cv2.circle(img, pp, max(4, int(0.45 * scale)), (80, 230, 80), -1)
    if intent is not None and intent.move:
        d = DIRECTION_VECTORS[intent.move] * np.array([1, -1])
        tip = (int(pp[0] + d[0] * 45), int(pp[1] + d[1] * 45))
        cv2.arrowedLine(img, pp, tip, (0, 255, 255), 5 if intent.dash else 2, tipLength=0.35)

    y0 = SIZE + 8

    def txt(row, col, s, color=(235, 235, 235)):
        cv2.putText(img, s, (10 + col * 360, y0 + 20 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def bar(row, col, frac, color):
        x, y = 10 + col * 360 + 130, y0 + 8 + row * 24
        cv2.rectangle(img, (x, y), (x + 200, y + 14), (60, 60, 60), -1)
        cv2.rectangle(img, (x, y), (x + int(200 * max(frac, 0)), y + 14), color, -1)

    txt(0, 0, f"sure {world.t:5.1f} sn")
    txt(0, 1, f"seviye {world.cfg.level}")
    txt(1, 0, "oyuncu HP")
    bar(1, 0, world.player_hp, (60, 200, 60))
    txt(1, 1, "boss HP")
    bar(1, 1, world.boss_hp, (60, 60, 220))
    txt(2, 0, f"makro: oyunda {'ACIK' if world.macro_on else 'KAPALI'} / ajan {macro.name}")
    txt(2, 1, f"dash sarj: {world.charges}  kilit {world.dash_lockout:.1f} sn")
    if intent is not None:
        txt(3, 0, f"karar: hareket={MOVE_NAMES[intent.move]} dash={'EVET' if intent.dash else '-'} makro={['-', 'AC', 'KAPAT'][intent.macro]}",
            (0, 255, 255))
    txt(3, 1, f"isabet {stats.hits}  erken kacis {stats.dodges}  gec {getattr(stats, 'late_dodges', 0)}")
    txt(4, 0, f"dash {stats.dashes}  R basma {stats.r_presses}  yon degistirme {stats.direction_changes}")
    return img[:, :, ::-1].copy()  # RGB
