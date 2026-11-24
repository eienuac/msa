"""Shared contract between perception, simulation, encoder, reward and control.

Every module consumes or produces these types only, so simulation and the
real game are interchangeable producers of the same state.

Coordinate frame ("ground frame"): a top-down plane in metres, aligned with
the camera. +y is the camera's forward direction (the W key), +x is right
(the D key). The simulation uses this frame as its world frame; the vision
pipeline produces it through the ground-plane homography.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np


class MacroState(IntEnum):
    OFF = 0
    ON = 1
    UNKNOWN = 2
    PENDING_ON = 3
    PENDING_OFF = 4


class HazardShape(str, Enum):
    CIRCLE = "circle"  # centre + radius
    SECTOR = "sector"  # apex + direction + half_angle + radius (cones, fans)
    RECT = "rect"  # start point + direction + length (radius field) + width (lines)
    POLYGON = "polygon"  # any other shape seen by the camera (vertices field)


class HazardPhase(IntEnum):
    TELEGRAPH = 0  # warning shown, no damage yet
    ACTIVE = 1  # damage is being applied


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Hazard:
    """One danger zone on the ground."""

    id: int
    shape: HazardShape
    center: np.ndarray  # (2,) circle centre / sector apex / rect start
    radius: float  # circle radius / sector radius / rect length
    time_to_impact: float  # seconds until damage; 0 when ACTIVE
    phase: HazardPhase = HazardPhase.TELEGRAPH
    direction: float = 0.0  # radians, sector/rect axis (0 = +x, pi/2 = +y)
    half_angle: float = math.pi  # sector half opening angle
    width: float = 0.0  # rect width
    damage: float = 0.0  # fraction of max HP (sim ground truth, unknown in game)
    confidence: float = 1.0
    skill: str = ""
    vertices: np.ndarray | None = None  # POLYGON only, (M,2) ground coordinates
    following: bool = False  # still tracking the player (cannot be left yet)
    inner_radius: float = 0.0  # CIRCLE only: > 0 makes it a ring that spares the inside (poison wave, outer hit)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(2)
        if self.vertices is not None:
            self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 2)

    # ------------------------------------------------------------------ geometry
    def contains(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of points (N,2) inside the hazard grown by `margin`."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.shape is HazardShape.POLYGON:
            return _polygon_signed_distance(pts, self.vertices) <= margin
        d = pts - self.center
        if self.shape is HazardShape.CIRCLE:
            d2 = d[:, 0] ** 2 + d[:, 1] ** 2
            inside = d2 <= (self.radius + margin) ** 2
            if self.inner_radius > 0:
                inside &= d2 >= max(self.inner_radius - margin, 0.0) ** 2
            return inside
        ux, uy = math.cos(self.direction), math.sin(self.direction)
        along = d[:, 0] * ux + d[:, 1] * uy
        across = -d[:, 0] * uy + d[:, 1] * ux
        if self.shape is HazardShape.RECT:
            return (
                (along >= -margin)
                & (along <= self.radius + margin)
                & (np.abs(across) <= self.width / 2 + margin)
            )
        # SECTOR
        dist = np.hypot(d[:, 0], d[:, 1])
        if self.half_angle >= math.pi - 1e-6:
            return dist <= self.radius + margin
        ang = np.abs(np.arctan2(across, along))
        inside_angle = ang <= self.half_angle
        # margin handling: perpendicular distance to the nearest edge ray
        edge_dist = dist * np.sin(np.clip(ang - self.half_angle, 0.0, math.pi / 2))
        near_edge = (~inside_angle) & (edge_dist <= margin) & (ang - self.half_angle < math.pi / 2)
        return (dist <= self.radius + margin) & (inside_angle | near_edge)

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """Approximate signed distance (negative inside) for points (N,2)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        d = pts - self.center
        if self.shape is HazardShape.CIRCLE and self.inner_radius > 0:
            r = np.hypot(d[:, 0], d[:, 1])
            return np.maximum(r - self.radius, self.inner_radius - r)
        if self.shape is HazardShape.CIRCLE or (
            self.shape is HazardShape.SECTOR and self.half_angle >= math.pi - 1e-6
        ):
            return np.hypot(d[:, 0], d[:, 1]) - self.radius
        return _polygon_signed_distance(pts, self.polygon())

    def polygon(self, n_arc: int = 24) -> np.ndarray:
        """Polygon approximation (M,2) in ground coordinates."""
        c = self.center
        if self.shape is HazardShape.POLYGON:
            return self.vertices.copy()
        if self.shape is HazardShape.CIRCLE:
            t = np.linspace(0, 2 * math.pi, n_arc, endpoint=False)
            return c + self.radius * np.stack([np.cos(t), np.sin(t)], 1)
        ux, uy = math.cos(self.direction), math.sin(self.direction)
        if self.shape is HazardShape.RECT:
            u = np.array([ux, uy])
            n = np.array([-uy, ux])
            w = self.width / 2
            L = self.radius
            return np.array([c - w * n, c + L * u - w * n, c + L * u + w * n, c + w * n])
        t = np.linspace(self.direction - self.half_angle, self.direction + self.half_angle, n_arc)
        arc = c + self.radius * np.stack([np.cos(t), np.sin(t)], 1)
        return np.vstack([c[None], arc])

    def copy(self) -> "Hazard":
        return Hazard(
            id=self.id,
            shape=self.shape,
            center=self.center.copy(),
            radius=self.radius,
            time_to_impact=self.time_to_impact,
            phase=self.phase,
            direction=self.direction,
            half_angle=self.half_angle,
            width=self.width,
            damage=self.damage,
            confidence=self.confidence,
            skill=self.skill,
            vertices=None if self.vertices is None else self.vertices.copy(),
            following=self.following,
            inner_radius=self.inner_radius,
        )


def _polygon_signed_distance(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a  # (M,2)
    ap = pts[:, None, :] - a[None]  # (N,M,2)
    denom = np.maximum((ab**2).sum(-1), 1e-12)
    t = np.clip((ap * ab[None]).sum(-1) / denom, 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    dist = np.sqrt(((pts[:, None, :] - proj) ** 2).sum(-1)).min(1)
    # even-odd rule for inside test
    x, y = pts[:, 0:1], pts[:, 1:2]
    ay, by = a[None, :, 1], b[None, :, 1]
    ax, bx = a[None, :, 0], b[None, :, 0]
    cond = (ay > y) != (by > y)
    xint = ax + (y - ay) * (bx - ax) / np.where(by - ay == 0, 1e-12, by - ay)
    inside = (cond & (x < xint)).sum(1) % 2 == 1
    return np.where(inside, -dist, dist)


@dataclass
class PlayerState:
    pos: np.ndarray
    hp: float  # absolute HP
    max_hp: float
    move_dir: int = 0  # 0 = idle, 1..8 = N, NE, E, SE, S, SW, W, NW (camera frame)
    dash_charges: int = 3
    dash_max_charges: int = 3
    dash_lockout: float = 0.0  # seconds until a dash can be used again (0 = ready)
    dash_recharge: float = 0.0  # seconds until the next charge is restored
    dashing: bool = False

    @property
    def hp_frac(self) -> float:
        return float(self.hp / max(self.max_hp, 1e-9))

    @property
    def can_dash(self) -> bool:
        return self.dash_charges > 0 and self.dash_lockout <= 0.0 and not self.dashing


@dataclass
class BossState:
    pos: np.ndarray | None  # None when the boss position is unknown
    hp: float
    max_hp: float
    targeted: bool = True  # is the boss the currently selected target?
    visible: bool = True
    windup: float = 0.0  # 0..1 strength of a "big attack incoming" cue
    hitbox_radius: float = 1.5

    @property
    def hp_frac(self) -> float:
        return float(self.hp / max(self.max_hp, 1e-9))


@dataclass
class GameState:
    """Everything the agent may know at one instant."""

    t: float
    player: PlayerState
    boss: BossState
    hazards: list[Hazard] = field(default_factory=list)
    macro: MacroState = MacroState.UNKNOWN
    melee_range: float = 3.0
    arena_center: np.ndarray | None = None
    arena_radius: float | None = None  # None = unknown boundary
    ping_ms: float = 60.0
    time_since_r_press: float = 99.0
    recent_damage_dealt: float = 0.0  # boss HP fraction lost in the last second
    recent_damage_taken: float = 0.0  # player HP fraction lost in the last second

    def boss_distance(self) -> float:
        if self.boss.pos is None:
            return float("inf")
        return float(np.linalg.norm(self.boss.pos - self.player.pos))

    def in_melee_range(self) -> bool:
        return self.boss_distance() <= self.melee_range + self.boss.hitbox_radius


# Direction index -> unit vector in the camera-aligned ground frame.
DIRECTION_VECTORS = np.array(
    [
        [0.0, 0.0],  # 0 idle
        [0.0, 1.0],  # 1 N  (W)
        [math.sqrt(0.5), math.sqrt(0.5)],  # 2 NE (W+D)
        [1.0, 0.0],  # 3 E  (D)
        [math.sqrt(0.5), -math.sqrt(0.5)],  # 4 SE (S+D)
        [0.0, -1.0],  # 5 S  (S)
        [-math.sqrt(0.5), -math.sqrt(0.5)],  # 6 SW (S+A)
        [-1.0, 0.0],  # 7 W  (A)
        [-math.sqrt(0.5), math.sqrt(0.5)],  # 8 NW (W+A)
    ]
)

DIRECTION_KEYS: list[frozenset[str]] = [
    frozenset(),
    frozenset({"w"}),
    frozenset({"w", "d"}),
    frozenset({"d"}),
    frozenset({"s", "d"}),
    frozenset({"s"}),
    frozenset({"s", "a"}),
    frozenset({"a"}),
    frozenset({"w", "a"}),
]


def direction_from_keys(keys: set[str] | frozenset[str]) -> int:
    """Map a set of pressed movement keys to a direction index (0 if none/conflicting)."""
    k = frozenset(x for x in keys if x in "wasd")
    if "w" in k and "s" in k:
        k = k - {"w", "s"}
    if "a" in k and "d" in k:
        k = k - {"a", "d"}
    for i, ks in enumerate(DIRECTION_KEYS):
        if ks == k:
            return i
    return 0


def direction_angle_diff(a: int, b: int) -> int:
    """Number of 45-degree steps between two direction indices (0..4). Idle counts as 0."""
    if a == 0 or b == 0:
        return 0
    d = abs(a - b) % 8
    return min(d, 8 - d)
