"""Top-down debug rendering of the simulator (RGB array, 640x640)."""

from __future__ import annotations

import cv2
import numpy as np

from ..core.game_state import GameState, HazardPhase, HazardShape, MacroState

SIZE = 640
MOVE_NAMES = ["-", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _to_px(p: np.ndarray, scale: float) -> tuple[int, int]:
    return int(SIZE / 2 + p[0] * scale), int(SIZE / 2 - p[1] * scale)


def render_topdown(world, observed: GameState, macro: MacroState, intent, stats) -> np.ndarray:
    scale = (SIZE / 2 - 10) / world.arena_r
    img = np.full((SIZE, SIZE, 3), 30, np.uint8)
    cv2.circle(img, (SIZE // 2, SIZE // 2), int(world.arena_r * scale), (70, 60, 50), -1)

    def poly(h):
        return np.array([_to_px(v, scale) for v in h.polygon(32)], np.int32)

    # ground truth hazards (filled), hidden ones dashed outline
    overlay = img.copy()
    for sh in world.hazards:
        h = sh.hz
        if sh.hidden:
            cv2.polylines(img, [poly(h)], True, (0, 140, 255), 1)
            continue
        if h.phase == HazardPhase.ACTIVE:
            col = (0, 0, 255)
        else:
            u = 1.0 - min(h.time_to_impact / 2.0, 1.0)
            col = (40, 40, int(120 + 135 * u))
        cv2.fillPoly(overlay, [poly(h)], col)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    # what the agent sees (outlines)
    for h in observed.hazards:
        cv2.polylines(img, [poly(h)], True, (255, 255, 0) if h.confidence >= 1 else (255, 0, 255), 1)
    bp = _to_px(world.boss_pos, scale)
    cv2.circle(img, bp, max(3, int(world.boss_hitbox * scale)), (60, 160, 230) if world.windup == 0 else (0, 200, 255), -1)
    pp = _to_px(world.player_pos, scale)
    cv2.circle(img, pp, max(3, int(0.5 * scale)), (80, 220, 80), -1)
    cv2.circle(img, pp, int((world.melee_range + world.boss_hitbox) * scale), (80, 220, 80), 1)

    def txt(y, s, col=(230, 230, 230)):
        cv2.putText(img, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    txt(18, f"t={world.t:5.1f}s  level={world.cfg.level}")
    txt(38, f"player HP {world.player_hp*100:5.1f}%   boss HP {world.boss_hp*100:5.1f}%")
    txt(58, f"macro true={'ON' if world.macro_on else 'OFF'}  agent={macro.name}")
    txt(78, f"dash charges={world.charges} lockout={world.dash_lockout:.1f}s")
    if intent is not None:
        txt(98, f"move={MOVE_NAMES[intent.move]} dash={int(intent.dash)} macro={['keep','ON','OFF'][intent.macro]}")
    txt(118, f"hits={stats.hits} dodges={stats.dodges}/{stats.threats} dashes={stats.dashes} R={stats.r_presses}")
    return img[:, :, ::-1].copy()  # RGB
