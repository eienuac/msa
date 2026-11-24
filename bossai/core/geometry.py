"""Egocentric danger grid: a shape-agnostic top-down view around the player.

Any hazard shape (circle, cone, line, or a future shape) is rasterised into
the same grid, so the policy does not need to know which shapes exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .game_state import GameState

# Grid channels
CH_URGENCY = 0  # 1 - time_to_impact / T_MAX for the most urgent hazard covering the cell
CH_OCCUPIED = 1  # any hazard covers the cell
CH_BOSS = 2  # boss body
CH_WALL = 3  # outside the arena (if known)
CH_UNSEEN = 4  # outside the camera's view (hazards there would be invisible)
N_CHANNELS = 5


@dataclass(frozen=True)
class GridSpec:
    size: int = 32  # cells per side
    extent: float = 12.0  # metres from the player to the grid edge
    t_max: float = 2.0  # seconds; urgency saturates beyond this horizon

    @property
    def cell(self) -> float:
        return 2 * self.extent / self.size

    def cell_offsets(self) -> np.ndarray:
        """(size*size, 2) cell-centre offsets from the player, row 0 = forward (+y)."""
        idx = (np.arange(self.size) + 0.5) * self.cell - self.extent
        xs = idx
        ys = -idx  # row 0 is the far forward row
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx.ravel(), gy.ravel()], 1)


@dataclass(frozen=True)
class FieldOfView:
    """Approximate visible ground region relative to the player (camera frame).

    Third-person camera: we see far ahead, a little behind, and the view widens
    with distance. Values are placeholders until calibrated from recordings.
    """

    behind: float = 4.0  # metres visible behind the player
    ahead: float = 20.0
    half_width_near: float = 9.0  # half width at the player's position
    widen_per_m: float = 0.6  # extra half width per metre ahead

    def visible(self, rel: np.ndarray) -> np.ndarray:
        rel = np.asarray(rel).reshape(-1, 2)
        x, y = rel[:, 0], rel[:, 1]
        hw = self.half_width_near + self.widen_per_m * np.maximum(y, 0.0)
        return (y >= -self.behind) & (y <= self.ahead) & (np.abs(x) <= hw)


class DangerGrid:
    def __init__(self, spec: GridSpec | None = None, fov: FieldOfView | None = None):
        self.spec = spec or GridSpec()
        self.fov = fov
        self._offsets = self.spec.cell_offsets()
        self._unseen = (
            (~self.fov.visible(self._offsets)).astype(np.float32)
            if self.fov is not None
            else np.zeros(len(self._offsets), np.float32)
        )

    def render(self, state: GameState) -> np.ndarray:
        s = self.spec
        pts = self._offsets + state.player.pos[None]
        out = np.zeros((N_CHANNELS, s.size * s.size), np.float32)
        for h in state.hazards:
            inside = h.contains(pts)
            if not inside.any():
                continue
            urg = 1.0 - min(max(h.time_to_impact, 0.0) / s.t_max, 1.0)
            urg = max(urg, 0.05) * h.confidence
            out[CH_URGENCY] = np.where(inside, np.maximum(out[CH_URGENCY], urg), out[CH_URGENCY])
            out[CH_OCCUPIED] = np.where(inside, 1.0, out[CH_OCCUPIED])
        if state.boss.pos is not None:
            d = np.linalg.norm(pts - state.boss.pos[None], axis=1)
            out[CH_BOSS] = (d <= state.boss.hitbox_radius + 0.5 * s.cell).astype(np.float32)
        if state.arena_radius is not None and state.arena_center is not None:
            d = np.linalg.norm(pts - state.arena_center[None], axis=1)
            out[CH_WALL] = (d > state.arena_radius).astype(np.float32)
        out[CH_UNSEEN] = self._unseen
        return out.reshape(N_CHANNELS, s.size, s.size)
