"""Time-aware escape geometry.

`ray_features` tells the policy, for each direction, whether walking or
dashing that way gets the player out of danger before the damage lands.
It does not pick an action. `SafetyPlanner` turns the same numbers into a
concrete choice; it backs the optional "move to safe zone" action and serves
as the expert baseline in evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .game_state import DIRECTION_VECTORS, GameState, HazardPhase

SLACK_CAP = 2.0  # seconds; slack values are clipped to [-SLACK_CAP, SLACK_CAP]


@dataclass(frozen=True)
class EscapeParams:
    walk_speed: float = 5.5  # m/s
    dash_distance: float = 5.0  # m
    dash_duration: float = 0.25  # s
    hitbox: float = 0.4  # player body radius, m
    margin: float = 0.3  # extra safety margin, m
    latency: float = 0.1  # reaction + input delay, s
    max_range: float = 12.0
    step: float = 0.25
    n_dirs: int = 16


def ray_directions(n: int) -> np.ndarray:
    """Unit vectors; index 0 = forward (+y), clockwise. n=8 matches action directions 1..8."""
    ang = math.pi / 2 - np.arange(n) * 2 * math.pi / n
    return np.stack([np.cos(ang), np.sin(ang)], 1)


@dataclass
class RayFeatures:
    free_dist: np.ndarray  # (K,) metres before entering any hazard (ignoring time)
    walk_slack: np.ndarray  # (K,) seconds to spare when walking out that way
    dash_slack: np.ndarray  # (K,) seconds to spare when dashing out that way
    wall_dist: np.ndarray  # (K,) metres to the arena wall
    exit_dist: np.ndarray  # (K,) metres until outside every hazard (max_range if never)
    in_danger: bool
    earliest_threat: float  # time to impact of the most urgent hazard covering the player


def _travel_times(s: np.ndarray, p: EscapeParams, dash: bool) -> np.ndarray:
    if not dash:
        return p.latency + s / p.walk_speed
    t_dash = np.where(
        s <= p.dash_distance,
        s / max(p.dash_distance, 1e-6) * p.dash_duration,
        p.dash_duration + (s - p.dash_distance) / p.walk_speed,
    )
    return p.latency + t_dash


def ray_features(state: GameState, p: EscapeParams, dirs: np.ndarray | None = None) -> RayFeatures:
    if dirs is None:
        dirs = ray_directions(p.n_dirs)
    K = len(dirs)
    s = np.arange(0.0, p.max_range + 1e-9, p.step)  # (S,)
    S = len(s)
    pos = state.player.pos
    pts = pos[None, None] + s[None, :, None] * dirs[:, None, :]  # (K,S,2)
    flat = pts.reshape(-1, 2)

    # arena wall
    if state.arena_radius is not None and state.arena_center is not None:
        r = np.linalg.norm(pts - state.arena_center[None, None], axis=2)  # (K,S)
        in_arena = r <= state.arena_radius - p.hitbox
        wall_idx = np.where((~in_arena).any(1), (~in_arena).argmax(1), S)
        wall_dist = np.minimum(np.where(wall_idx < S, s[np.minimum(wall_idx, S - 1)], p.max_range), p.max_range)
    else:
        in_arena = np.ones((K, S), bool)
        wall_dist = np.full(K, p.max_range)

    hz = [h for h in state.hazards]
    grow = p.hitbox + p.margin
    if hz:
        C = np.stack([h.contains(flat, margin=grow).reshape(K, S) for h in hz])  # (H,K,S)
        tti = np.array(
            [0.0 if h.phase == HazardPhase.ACTIVE else max(h.time_to_impact, 0.0) for h in hz]
        )
    else:
        C = np.zeros((0, K, S), bool)
        tti = np.zeros(0)

    any_c = C.any(0) if len(hz) else np.zeros((K, S), bool)
    in_danger = bool(any_c[:, 0].any())
    earliest = float(tti[C[:, 0, 0]].min()) if in_danger else SLACK_CAP

    first_hit = np.where(any_c.any(1), any_c.argmax(1), S)
    free_dist = np.where(first_hit < S, s[np.minimum(first_hit, S - 1)], p.max_range)

    safe_pt = (~any_c) & in_arena  # (K,S)
    # a point counts as an exit only if the arena is not crossed before it
    blocked_before = np.cumsum(~in_arena, axis=1) > 0
    exit_ok = safe_pt & ~blocked_before
    exit_idx = np.where(exit_ok.any(1), exit_ok.argmax(1), S)
    exit_dist = np.where(exit_idx < S, s[np.minimum(exit_idx, S - 1)], p.max_range)

    def slack_for(dash: bool) -> np.ndarray:
        t = _travel_times(s, p, dash)  # (S,)
        out = np.full(K, SLACK_CAP)
        for k in range(K):
            e = exit_idx[k]
            if e >= S:
                out[k] = -SLACK_CAP
                continue
            if e == 0 or not len(hz):
                continue
            # for each hazard covering part of the path before the exit, the time we leave it
            seg = C[:, k, :e]  # (H,e)
            touched = seg.any(1)
            if not touched.any():
                continue
            last = e - 1 - np.argmax(seg[:, ::-1], axis=1)  # last inside index
            leave_t = t[np.minimum(last + 1, S - 1)]
            sl = tti[touched] - leave_t[touched]
            out[k] = float(np.clip(sl.min(), -SLACK_CAP, SLACK_CAP))
        return out

    return RayFeatures(
        free_dist=free_dist.astype(np.float32),
        walk_slack=slack_for(False).astype(np.float32),
        dash_slack=slack_for(True).astype(np.float32),
        wall_dist=wall_dist.astype(np.float32),
        exit_dist=exit_dist.astype(np.float32),
        in_danger=in_danger,
        earliest_threat=earliest,
    )


@dataclass
class EscapePlan:
    direction: int  # 0 = stay, 1..8 = action direction
    dash: bool
    slack: float


class SafetyPlanner:
    """Picks the best of the 8 movement directions to leave danger in time."""

    def __init__(self, params: EscapeParams | None = None, min_slack: float = 0.05):
        self.p = params or EscapeParams()
        self.min_slack = min_slack
        self._dirs8 = DIRECTION_VECTORS[1:]

    def plan(self, state: GameState, prev_dir: int = 0) -> EscapePlan:
        rf = ray_features(state, self.p, self._dirs8)
        if not rf.in_danger:
            return EscapePlan(0, False, SLACK_CAP)
        boss = state.boss.pos
        best: tuple[float, int, bool, float] | None = None
        for dash in (False, True):
            if dash and not state.player.can_dash:
                continue
            slack = rf.dash_slack if dash else rf.walk_slack
            for k in range(8):
                if slack[k] < self.min_slack:
                    continue
                exit_pt = state.player.pos + rf.exit_dist[k] * self._dirs8[k]
                boss_term = 0.0
                if boss is not None:
                    boss_term = max(0.0, float(np.linalg.norm(exit_pt - boss)) - state.melee_range - state.boss.hitbox_radius)
                turn = 0.0
                if prev_dir:
                    d = abs((k + 1) - prev_dir) % 8
                    turn = min(d, 8 - d) / 4.0
                cost = (
                    rf.exit_dist[k] / self.p.walk_speed
                    + 0.08 * boss_term
                    + 0.15 * turn
                    + (0.6 if dash else 0.0)
                    - 0.1 * min(float(slack[k]), 1.0)
                )
                if best is None or cost < best[0]:
                    best = (cost, k + 1, dash, float(slack[k]))
            if best is not None and not dash:
                break  # walking works: never spend a dash
        if best is None:
            # nothing reaches safety in time: take the least bad option
            use_dash = state.player.can_dash and rf.dash_slack.max() > rf.walk_slack.max()
            arr = rf.dash_slack if use_dash else rf.walk_slack
            k = int(np.argmax(arr))
            return EscapePlan(k + 1, bool(use_dash), float(arr[k]))
        return EscapePlan(best[1], best[2], best[3])
