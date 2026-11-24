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
    wait_margin: float = 0.15  # s after a zone lands before walking through where it was
    # the player may also be faster than assumed (measured: walk 8.5-11, dash 4.5-6.5); arriving
    # early matters when one zone lands before another (poison wave: outside first)
    walk_speed_max: float = 11.0
    dash_distance_max: float = 6.5
    late_gap: float = 0.8  # s; a zone landing this much after the first one can be left after it


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
    walk_delay: np.ndarray | None = None  # (K,) seconds to wait before leaving (0 = go now)
    dash_delay: np.ndarray | None = None


def _travel_times(s: np.ndarray, p: EscapeParams, dash: bool, fast: bool = False) -> np.ndarray:
    walk = p.walk_speed_max if fast else p.walk_speed
    dash_d = p.dash_distance_max if fast else p.dash_distance
    latency = 0.0 if fast else p.latency
    if not dash:
        return latency + s / walk
    t_dash = np.where(
        s <= dash_d,
        s / max(dash_d, 1e-6) * p.dash_duration,
        p.dash_duration + (s - dash_d) / walk,
    )
    return latency + t_dash


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

    # a point counts as an exit only if the arena is not crossed before it
    reachable = in_arena & ~(np.cumsum(~in_arena, axis=1) > 0)
    active = np.array([h.phase == HazardPhase.ACTIVE for h in hz], bool)
    e_idx = np.arange(S)

    def escape(dash: bool, delay: float) -> tuple[np.ndarray, np.ndarray]:
        """(slack, exit distance) per ray when walking/dashing straight out after `delay` s.

        Time-aware: the player is hit by a zone only if standing in it when it
        lands. A zone that lands before the player gets there does not block
        the way, and one the player stands in must be left before it lands.
        """
        t = _travel_times(s, p, dash) + delay  # arrival time at s[i] (slow assumption)
        t_fast = _travel_times(s, p, dash, fast=True) + delay
        slack = np.full(K, SLACK_CAP)
        if not len(hz):
            return slack, np.zeros(K)
        # where the player is when each zone lands (anywhere between the slow and the fast
        # assumption); ACTIVE zones block wherever the player stops
        at = np.clip(np.searchsorted(t, tti, side="right") - 1, 0, S - 1)
        at_fast = np.clip(np.searchsorted(t_fast, tti + p.wait_margin, side="right") - 1, 0, S - 1)
        at = np.where(active, S - 1, at)
        at_fast = np.where(active, S - 1, at_fast)
        lo = np.minimum(at[:, None], e_idx[None])  # (H,S), stopping at e
        hi = np.minimum(at_fast[:, None], e_idx[None])
        cs = np.concatenate([np.zeros((len(hz), K, 1), np.int32), np.cumsum(C, axis=2, dtype=np.int32)], axis=2)
        upto = np.take_along_axis(cs, np.broadcast_to(hi[:, None, :] + 1, C.shape), axis=2)
        before = np.take_along_axis(cs, np.broadcast_to(lo[:, None, :], C.shape), axis=2)
        hit_h = (upto - before) > 0  # (H,K,S)
        if late.any():
            # a zone that lands well after the first one does not block a stop inside it if
            # there is time to walk out of it once the first one has landed (the poison wave:
            # step into the ring for the outer hit, leave before the inner one)
            ready = np.maximum(t[None, :], t_first)  # (1,S) earliest time to start leaving from e
            need = p.latency + np.maximum(grow - SD[late], 0.0) / p.walk_speed + p.wait_margin  # (L,K,S)
            ok_late = (tti[late][:, None, None] - ready[None]) >= need
            hit_h[late] &= ~ok_late
        hit = hit_h.any(0)  # (K,S)
        ok = reachable & ~hit
        exit_i = np.where(ok.any(1), ok.argmax(1), S)
        exit_d = np.where(exit_i < S, s[np.minimum(exit_i, S - 1)], p.max_range)
        for k in range(K):
            e = exit_i[k]
            if e >= S:
                slack[k] = -SLACK_CAP
                continue
            if e == 0:
                continue
            for h in np.flatnonzero(~active & ~late):
                last = min(e, max(at[h], at_fast[h]))
                inside = np.flatnonzero(C[h, k, : last + 1])
                if not len(inside):
                    continue  # never stood in it before it landed
                leave_t = t[min(inside[-1] + 1, S - 1)]
                slack[k] = min(slack[k], tti[h] - leave_t)
        return np.clip(slack, -SLACK_CAP, SLACK_CAP), exit_d

    tel = ~active
    t_first = float(tti[tel].min()) if tel.any() else 0.0
    late = tel & (tti >= t_first + p.late_gap)
    SD = np.stack([h.signed_distance(flat).reshape(K, S) for h in hz]) if late.any() else None

    # waiting can be the answer: the poison wave hits outside its ring first, then inside
    delays = [0.0]
    if in_danger:
        here = C[:, 0, 0]
        cutoff = tti[here & ~active].max() if (here & ~active).any() else 0.0
        later = sorted({float(x) + p.wait_margin for x in tti[~here & ~active] if 0 < x < cutoff})
        delays += later[:2]

    def best(dash: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        res = [escape(dash, d) for d in delays]
        sl = np.stack([r[0] for r in res])
        i = sl.argmax(0)
        ks = np.arange(K)
        return sl[i, ks], np.stack([r[1] for r in res])[i, ks], np.asarray(delays)[i]

    walk_slack, exit_dist, walk_delay = best(False)
    dash_slack, _, dash_delay = best(True)

    return RayFeatures(
        free_dist=free_dist.astype(np.float32),
        walk_slack=walk_slack.astype(np.float32),
        dash_slack=dash_slack.astype(np.float32),
        wall_dist=wall_dist.astype(np.float32),
        exit_dist=exit_dist.astype(np.float32),
        in_danger=in_danger,
        earliest_threat=earliest,
        walk_delay=walk_delay.astype(np.float32),
        dash_delay=dash_delay.astype(np.float32),
    )


@dataclass
class EscapePlan:
    direction: int  # 0 = stay, 1..8 = action direction
    dash: bool
    slack: float
    wait: bool = False  # in danger, but the right move is to stay until an earlier zone lands


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
        here = (rf.exit_dist <= 1e-6) & (rf.walk_slack >= self.min_slack) & (rf.walk_delay <= 0)
        if here.any():
            # safe where we stand for now (inside the poison ring before the outer hit)
            return EscapePlan(0, False, float(rf.walk_slack[here].max()), wait=True)
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
        if best is not None:
            delay = (rf.dash_delay if best[2] else rf.walk_delay)[best[1] - 1]
            if delay > 0:
                return EscapePlan(0, False, best[3], wait=True)  # stay until the first wave has landed
        if best is None:
            # nothing reaches safety in time: take the least bad option
            use_dash = state.player.can_dash and rf.dash_slack.max() > rf.walk_slack.max()
            arr = rf.dash_slack if use_dash else rf.walk_slack
            k = int(np.argmax(arr))
            return EscapePlan(k + 1, bool(use_dash), float(arr[k]))
        return EscapePlan(best[1], best[2], best[3])
