"""Ground-truth 2D boss fight: player, dash charges, boss skills, R macro."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.game_state import (
    DIRECTION_VECTORS,
    BossState,
    GameState,
    Hazard,
    HazardPhase,
    HazardShape,
    MacroState,
    PlayerState,
    direction_from_keys,
)
from .config import SimConfig, sample

ACTIVE_FLASH = 0.25  # seconds a resolved hazard stays visible as ACTIVE


@dataclass
class SimHazard:
    hz: Hazard
    impact_t: float
    hidden: bool = False
    threatened: bool = False
    resolved: bool = False
    inside: bool = False
    exit_slack: float = -1.0  # time left when the player last left the zone (-1 = never left)
    follow_until: float = -1.0  # the zone tracks the player until this time
    stun: float = 0.0  # seconds of stun if it hits
    active_until: float = 0.0
    # fixed perception errors for this hazard (applied by the perception model)
    missed: bool = False
    visible_from: float = 0.0
    center_err: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius_scale: float = 1.0


@dataclass
class WorldEvents:
    boss_damage: float = 0.0
    player_damage: float = 0.0
    hits: int = 0
    dodges: int = 0  # threatened, not hit, left with at least safe_exit_slack to spare
    late_dodges: int = 0  # threatened, not hit, but left at the last moment
    shield_blocks: int = 0  # hits stopped by the E shield
    threats: int = 0
    dashes: int = 0
    wasted_dash_taps: int = 0
    r_taps: int = 0
    macro_toggles: int = 0
    boss_killed: bool = False
    player_dead: bool = False
    hit_skills: list[str] = field(default_factory=list)


class BossWorld:
    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        c, r = self.cfg, self.rng
        self.t = 0.0
        self.arena_r = sample(r, c.arena_radius)
        self.walk_speed = sample(r, c.walk_speed)
        self.accel_time = sample(r, c.accel_time)
        self.dash_distance = sample(r, c.dash_distance)
        self.dash_duration = sample(r, c.dash_duration)
        self.dash_gap = sample(r, c.dash_gap)
        self.dash_regen = sample(r, c.dash_charge_regen)
        self.dash_empty = sample(r, c.dash_empty_lockout)
        self.boss_speed = sample(r, c.boss_speed)
        self.boss_hitbox = sample(r, c.boss_hitbox)
        self.melee_range = sample(r, c.melee_range)
        self.dps = 1.0 / sample(r, c.kill_time)
        self.macro_latency = sample(r, c.macro_latency)
        self.move_cancel_chance = sample(r, c.macro_move_cancel_chance)
        self._bout_cancels = False  # does the current movement bout cancel the macro?
        self._was_moving = False
        self.attack_resume = sample(r, c.attack_resume)

        ang = r.uniform(0, 2 * math.pi)
        start_d = r.uniform(2.0, 10.0)
        self.player_pos = r.uniform(-4, 4, 2)
        self.boss_pos = self.player_pos + start_d * np.array([math.cos(ang), math.sin(ang)])
        self._clamp_arena()
        self.vel = np.zeros(2)
        self.player_hp = 1.0
        self.boss_hp = 1.0
        self.charges = c.dash_charges
        self.dash_lockout = 0.0
        self.dash_recharge = 0.0
        self.dash_left = 0.0
        self.dash_vel = np.zeros(2)
        self.macro_on = bool(r.random() < 0.3)  # unknown start state for the agent
        self.boss_targeted = True
        self.stunned_until = -1.0  # web cocoon: no movement, no dash
        self.shield_max = sample(r, c.shield_max)
        self.shield_cooldown = sample(r, c.shield_cooldown)
        self.shield_since = -1.0  # E shield up since (-1 = down)
        self.shield_ready_at = 0.0
        self._e_held = False
        self._target_lost = False  # an add took the target (after a summon burst)
        self._summon_at = float("inf")
        self._summon_spec = None
        self._macro_pending: list[float] = []
        self._move_time = 0.0
        self._stand_time = 0.0
        self.move_dir = 0
        self.hazards: list[SimHazard] = []
        self._next_id = 0
        self.windup = 0.0
        self._windup_until = -1.0
        self._windup_start = -1.0
        self.casting_until = -1.0
        self.next_skill_t = sample(r, (2.0, 4.0))
        self._dmg_dealt_hist: list[tuple[float, float]] = []
        self._dmg_taken_hist: list[tuple[float, float]] = []

    # ------------------------------------------------------------------- step
    def step(self, held: set[str], taps: list[str], duration: float) -> WorldEvents:
        ev = WorldEvents()
        self.move_dir = direction_from_keys(held)
        self._e_held = "e" in held
        for k in taps:
            if k == "r":
                ev.r_taps += 1
                if self.rng.random() >= self.cfg.macro_press_loss:
                    self._macro_pending.append(self.t + self.macro_latency)
            elif k == "q":
                if self.t >= self.stunned_until and self.charges > 0 and self.dash_lockout <= 0 and self.dash_left <= 0 and self.move_dir:
                    self._start_dash()
                    ev.dashes += 1
                else:
                    ev.wasted_dash_taps += 1
        n = max(1, int(round(duration / self.cfg.dt)))
        for _ in range(n):
            self._substep(self.cfg.dt, ev)
            if ev.player_dead or ev.boss_killed:
                break
        return ev

    def _start_dash(self) -> None:
        d = DIRECTION_VECTORS[self.move_dir]
        self.dash_left = self.dash_duration
        self.dash_vel = d * self.dash_distance / self.dash_duration
        self.charges -= 1
        if self.charges == 0:
            self.dash_lockout = self.dash_empty
            self.dash_recharge = 0.0
        else:
            self.dash_lockout = self.dash_gap
            if self.dash_recharge <= 0:
                self.dash_recharge = self.dash_regen

    @property
    def shield_up(self) -> bool:
        return self.shield_since >= 0

    def _update_shield(self) -> None:
        if self.shield_up and (not self._e_held or self.t - self.shield_since >= self.shield_max):
            self.shield_since = -1.0
            self.shield_ready_at = self.t + self.shield_cooldown
        elif not self.shield_up and self._e_held and self.t >= self.shield_ready_at:
            self.shield_since = self.t

    def _substep(self, dt: float, ev: WorldEvents) -> None:
        c = self.cfg
        self.t += dt
        self._update_shield()
        # --- dash charges
        if self.dash_lockout > 0:
            self.dash_lockout = max(0.0, self.dash_lockout - dt)
            if self.dash_lockout == 0 and self.charges == 0:
                self.charges = c.dash_charges
        if 0 < self.charges < c.dash_charges and self.dash_recharge > 0:
            self.dash_recharge -= dt
            if self.dash_recharge <= 0:
                self.charges += 1
                self.dash_recharge = self.dash_regen if self.charges < c.dash_charges else 0.0
        # --- player movement
        if self.dash_left > 0:
            self.player_pos = self.player_pos + self.dash_vel * dt
            self.dash_left -= dt
            self.vel = np.zeros(2)
        else:
            stunned = self.t < self.stunned_until
            target = np.zeros(2) if stunned else DIRECTION_VECTORS[self.move_dir] * self.walk_speed
            a = min(1.0, dt / max(self.accel_time, 1e-3))
            self.vel = self.vel + (target - self.vel) * a
            self.player_pos = self.player_pos + self.vel * dt
        self._clamp_arena()
        moving = self.dash_left > 0 or self.move_dir != 0
        dist = float(np.linalg.norm(self.boss_pos - self.player_pos))
        in_range = dist <= self.melee_range + self.boss_hitbox
        # --- macro toggles (delayed key effect)
        due = [x for x in self._macro_pending if x <= self.t]
        if due:
            self._macro_pending = [x for x in self._macro_pending if x > self.t]
            for _ in due:
                if self.macro_on:
                    self.macro_on = False
                    ev.macro_toggles += 1
                elif self.boss_targeted or in_range:
                    # R next to the boss also selects it (user)
                    self._target_lost = False
                    self.boss_targeted = True
                    self.macro_on = True
                    ev.macro_toggles += 1
        if moving and not self._was_moving:  # a new movement bout starts
            self._bout_cancels = bool(self.rng.random() < self.move_cancel_chance)
        self._was_moving = moving
        if self.macro_on and self._bout_cancels and moving:
            self._move_time += dt
            if self._move_time > self.cfg.macro_cancel_delay:
                self.macro_on = False
        else:
            self._move_time = 0.0
        # --- player attacks via macro (melee, standing still)
        if self.macro_on and self.boss_targeted and in_range and not moving:
            self._stand_time += dt
            if self._stand_time >= self.attack_resume:
                dmg = min(self.dps * dt, self.boss_hp)
                self.boss_hp -= dmg
                ev.boss_damage += dmg
                self._dmg_dealt_hist.append((self.t, dmg))
                if self.boss_hp <= 1e-9:
                    ev.boss_killed = True
        else:
            self._stand_time = 0.0
        self.boss_targeted = (not self._target_lost) and dist < 20.0
        # --- summon burst: push back + adds may take the target
        if self.t >= self._summon_at:
            spec, self._summon_at = self._summon_spec, float("inf")
            away = self.player_pos - self.boss_pos
            if dist < spec.knockback_radius:
                self.player_pos = self.player_pos + away / max(dist, 1e-6) * sample(self.rng, spec.knockback)
                self._clamp_arena()
            if self.rng.random() < spec.target_steal_prob:
                self._target_lost = True
                self.boss_targeted = False
        # --- boss movement
        if self.t >= self.casting_until and dist > self.melee_range * 0.7 + self.boss_hitbox:
            step = min(self.boss_speed * dt, dist - (self.melee_range * 0.7 + self.boss_hitbox))
            self.boss_pos = self.boss_pos + (self.player_pos - self.boss_pos) / max(dist, 1e-6) * step
        # --- boss skills
        if self.t >= self.next_skill_t:
            self._cast_skill()
        if self._windup_until > self.t:
            span = max(self._windup_until - self._windup_start, 1e-3)
            self.windup = min(1.0, (self.t - self._windup_start) / span)
        else:
            self.windup = 0.0
        # --- hazards
        keep = []
        for sh in self.hazards:
            h = sh.hz
            if not sh.resolved:
                h.following = self.t < sh.follow_until
                if h.following:
                    h.center = self.player_pos.copy()  # the ring tracks the player, then locks
                h.time_to_impact = sh.impact_t - self.t
                inside = bool(h.contains(self.player_pos[None], margin=c.hitbox)[0])
                if inside:
                    sh.threatened = True
                elif sh.inside:
                    sh.exit_slack = h.time_to_impact
                sh.inside = inside
                if h.time_to_impact <= 0:
                    sh.resolved = True
                    h.phase = HazardPhase.ACTIVE
                    h.time_to_impact = 0.0
                    sh.active_until = self.t + ACTIVE_FLASH
                    immune = c.dash_iframes and self.dash_left > 0
                    hit = h.contains(self.player_pos[None], margin=c.hitbox)[0] and not immune
                    if sh.threatened:
                        ev.threats += 1
                    if hit and self.shield_up:
                        hit = False
                        ev.shield_blocks += 1
                        ev.dodges += 1
                    elif hit:
                        dmg = min(h.damage, self.player_hp)
                        self.player_hp -= dmg
                        ev.player_damage += dmg
                        ev.hits += 1
                        ev.hit_skills.append(h.skill)
                        if sh.stun > 0:
                            self.stunned_until = max(self.stunned_until, self.t + sh.stun)
                        self._dmg_taken_hist.append((self.t, dmg))
                        if self.player_hp <= 1e-9:
                            ev.player_dead = True
                    elif sh.threatened:
                        # a following zone can only be left after it locks: judge the exit from then on
                        safe = c.safe_exit_slack
                        if sh.follow_until > 0:
                            safe = min(safe, 0.5 * (sh.impact_t - sh.follow_until))
                        if sh.exit_slack >= safe:
                            ev.dodges += 1
                        else:
                            ev.late_dodges += 1
            if not sh.resolved or self.t < sh.active_until:
                keep.append(sh)
        self.hazards = keep

    # ----------------------------------------------------------------- skills
    def _cast_skill(self) -> None:
        c, r = self.cfg, self.rng
        names = list(c.skills)
        w = np.array([c.skill(n).weight for n in names])
        name = names[int(r.choice(len(names), p=w / w.sum()))]
        spec = c.skill(name)
        tele = sample(r, spec.telegraph)
        dmg = sample(r, spec.damage)
        to_player = self.player_pos - self.boss_pos
        face = math.atan2(to_player[1], to_player[0])
        new: list[tuple[Hazard, float, bool]] = []

        def add(shape, center, radius, delay=0.0, **kw):
            h = Hazard(
                id=self._next_id,
                shape=shape,
                center=np.asarray(center, float),
                radius=radius,
                time_to_impact=tele + delay,
                damage=dmg,
                skill=name,
                **kw,
            )
            self._next_id += 1
            new.append((h, self.t + tele + delay, spec.hidden))

        if name == "circle_on_player":
            add(HazardShape.CIRCLE, self.player_pos + r.normal(0, 0.3, 2), sample(r, spec.radius))
        elif name == "cone":
            add(
                HazardShape.SECTOR,
                self.boss_pos,
                sample(r, spec.radius),
                direction=face,
                half_angle=math.radians(sample(r, spec.half_angle_deg)),
            )
            self.casting_until = self.t + tele
        elif name == "nova":
            add(HazardShape.CIRCLE, self.boss_pos, sample(r, spec.radius))
            self.casting_until = self.t + tele
        elif name == "rain":
            n = int(round(sample(r, spec.count)))
            for i in range(n):
                off = np.zeros(2) if i == 0 else r.uniform(-6, 6, 2)
                add(HazardShape.CIRCLE, self.player_pos + off, sample(r, spec.radius), delay=sample(r, spec.stagger))
        elif name == "line":
            add(HazardShape.RECT, self.boss_pos, sample(r, spec.radius), direction=face, width=sample(r, spec.width))
            self.casting_until = self.t + tele
        elif name == "follow_ring":
            add(HazardShape.CIRCLE, self.player_pos.copy(), sample(r, spec.radius))
        elif name == "boss_disc":
            add(HazardShape.CIRCLE, self.boss_pos.copy(), sample(r, spec.radius))
            self.casting_until = self.t + tele
        elif name == "poison_wave":
            # outside the ring first (stay in), then the inside (get out)
            ring = sample(r, spec.radius)
            gap = sample(r, spec.second_hit)
            add(HazardShape.CIRCLE, self.boss_pos.copy(), 2.0 * self.arena_r + ring, inner_radius=ring)
            add(HazardShape.CIRCLE, self.boss_pos.copy(), ring, delay=gap)
            self.casting_until = self.t + tele + gap
        elif name == "beams":
            n = int(round(sample(r, spec.count)))
            for _ in range(n):
                ang = r.uniform(0, 2 * math.pi)
                u = np.array([math.cos(ang), math.sin(ang)])
                perp = np.array([-u[1], u[0]])
                length = sample(r, spec.radius)
                start = self.player_pos + perp * r.uniform(-4.0, 4.0) - u * length / 2
                add(HazardShape.RECT, start, length, direction=ang, width=sample(r, spec.width))
        elif name in ("summon_burst", "spawn_adds"):
            self._summon_at = self.t + tele
            self._summon_spec = spec
            self._windup_start = self.t
            self._windup_until = self.t + tele
            self.casting_until = self.t + tele
        follow = sample(r, spec.follow)
        stun = sample(r, spec.stun)
        for h, impact, hidden in new:
            self.hazards.append(
                SimHazard(h, impact, hidden=hidden, follow_until=self.t + follow if follow > 0 else -1.0, stun=stun)
            )
        interval = sample(r, c.skill_interval)
        if self.boss_hp < c.enrage_below:
            interval *= c.enrage_speedup
        self.next_skill_t = self.t + interval

    # ------------------------------------------------------------------ state
    def _clamp_arena(self) -> None:
        lim = self.arena_r - self.cfg.hitbox
        for attr in ("player_pos", "boss_pos"):
            p = getattr(self, attr)
            d = float(np.linalg.norm(p))
            if d > lim:
                setattr(self, attr, p / d * lim)

    def _recent(self, hist: list[tuple[float, float]], window: float = 1.0) -> float:
        while hist and self.t - hist[0][0] > window:
            hist.pop(0)
        return float(sum(x for _, x in hist))

    @property
    def macro_state(self) -> MacroState:
        return MacroState.ON if self.macro_on else MacroState.OFF

    def true_state(self, include_hidden: bool = False) -> GameState:
        c = self.cfg
        return GameState(
            t=self.t,
            player=PlayerState(
                pos=self.player_pos.copy(),
                hp=self.player_hp,
                max_hp=1.0,
                move_dir=self.move_dir,
                dash_charges=self.charges,
                dash_max_charges=c.dash_charges,
                dash_lockout=self.dash_lockout,
                dash_recharge=self.dash_recharge,
                dashing=self.dash_left > 0,
            ),
            boss=BossState(
                pos=self.boss_pos.copy(),
                hp=self.boss_hp,
                max_hp=1.0,
                targeted=self.boss_targeted,
                windup=self.windup,
                hitbox_radius=self.boss_hitbox,
            ),
            hazards=[sh.hz.copy() for sh in self.hazards if include_hidden or not sh.hidden],
            macro=self.macro_state,
            melee_range=self.melee_range,
            arena_center=np.zeros(2),
            arena_radius=self.arena_r,
            recent_damage_dealt=self._recent(self._dmg_dealt_hist),
            recent_damage_taken=self._recent(self._dmg_taken_hist),
        )
