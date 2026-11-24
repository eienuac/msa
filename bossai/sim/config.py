"""Simulation parameters.

Values marked (video) come from the Kor Bekçi recordings; the others are
placeholders to be replaced by calibration recordings. Every (lo, hi) pair
is sampled once per episode (domain randomisation).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np

Rng = tuple[float, float]


def sample(rng: np.random.Generator, r: Rng | float) -> float:
    if isinstance(r, (int, float)):
        return float(r)
    lo, hi = r
    return float(lo if hi <= lo else rng.uniform(lo, hi))


@dataclass(frozen=True)
class SkillSpec:
    name: str
    weight: float
    telegraph: Rng = (1.0, 1.5)  # (video) 1.1-1.5 s for circles and cones
    damage: Rng = (0.12, 0.25)  # (video) 180-390 of 1570 HP per circle hit
    radius: Rng = (3.5, 5.0)
    count: Rng = (1, 1)
    half_angle_deg: Rng = (60, 90)  # (video) cones look like 120-180 degree fans
    width: Rng = (2.0, 3.0)
    stagger: Rng = (0.0, 0.0)
    hidden: bool = False  # no ground telegraph; only a boss wind-up cue
    knockback: Rng = (0.0, 0.0)  # metres the player is pushed away from the boss
    knockback_radius: float = 0.0  # only players closer than this to the boss are pushed
    target_steal_prob: float = 0.0  # chance the summoned adds take over the player's target
    follow: Rng = (0.0, 0.0)  # seconds the zone follows the player before it locks in place
    stun: Rng = (0.0, 0.0)  # seconds the player cannot move or dash after being hit


SKILLS_ALL: dict[str, SkillSpec] = {
    "circle_on_player": SkillSpec("circle_on_player", 1.0, radius=(3.5, 5.0)),
    "cone": SkillSpec("cone", 0.8, radius=(7.0, 10.0), telegraph=(1.0, 1.5)),
    "nova": SkillSpec("nova", 0.5, radius=(5.0, 8.0), telegraph=(1.2, 1.8)),
    "rain": SkillSpec(
        "rain", 0.6, radius=(2.0, 3.5), count=(3, 6), telegraph=(1.2, 1.8), damage=(0.08, 0.15), stagger=(0.0, 0.4)
    ),
    "line": SkillSpec("line", 0.5, radius=(12.0, 18.0), width=(2.0, 3.0), telegraph=(0.8, 1.2)),
    # (user) raises its arms (~1 s), fire bursts around it summon adds and push the player back; no damage
    "summon_burst": SkillSpec(
        "summon_burst", 0.4, telegraph=(0.8, 1.2), damage=(0.0, 0.0), hidden=True,
        knockback=(1.0, 2.0), knockback_radius=6.0, target_steal_prob=0.5,
    ),
    # ---- Örümcek Ana (spider). Green warnings deal damage (user); exact numbers unknown.
    # Ring that follows the player ~2 s, then the player is wrapped in a web cocoon
    # (recordings). Assumed: it locks shortly before impact and a hit stuns.
    "follow_ring": SkillSpec(
        "follow_ring", 0.6, radius=(2.5, 3.5), telegraph=(2.2, 2.6), follow=(1.3, 1.7),  # leaves 0.5-1.3 s to step out
        damage=(0.08, 0.15), stun=(1.0, 1.5),
    ),
    # filled pale-green disc at the boss (recording 11, ~9.5 s)
    "boss_disc": SkillSpec("boss_disc", 0.6, radius=(4.0, 5.5), telegraph=(1.0, 1.4), damage=(0.12, 0.2)),
    # huge ring around the boss, then a poison wave over it (~1.0-1.2 s, clips b2_3 / 11)
    "poison_wave": SkillSpec("poison_wave", 0.4, radius=(9.0, 13.0), telegraph=(1.0, 1.2), damage=(0.15, 0.25)),
    # long green beams across the arena (recording 33)
    "beams": SkillSpec(
        "beams", 0.5, radius=(25.0, 35.0), width=(1.5, 2.5), count=(2, 4), telegraph=(1.0, 1.4), damage=(0.1, 0.18)
    ),
    # "YAVRULAR YAĞIYOR": small spiders appear and may take the target
    "spawn_adds": SkillSpec(
        "spawn_adds", 0.3, telegraph=(0.8, 1.2), damage=(0.0, 0.0), hidden=True, target_steal_prob=0.4,
    ),
}


@dataclass(frozen=True)
class NoiseSpec:
    center_sigma: float = 0.0  # metres
    radius_rel_sigma: float = 0.0
    tti_sigma: float = 0.0  # seconds
    miss_prob: float = 0.0  # chance a hazard is never detected
    late_detect: Rng = (0.0, 0.0)  # seconds before a new hazard becomes visible
    false_pos_rate: float = 0.0  # per second
    boss_pos_sigma: float = 0.0
    hp_sigma: float = 0.0
    indicator_flip: float = 0.0  # chance per tick the macro indicator is misread
    latency: Rng = (0.0, 0.0)  # seconds between the game state and what the agent sees
    use_fov: bool = False


@dataclass(frozen=True)
class SimConfig:
    level: int = 0
    dt: float = 0.05
    decision_dt: float = 0.1
    max_time: float = 240.0
    arena_radius: Rng = (18.0, 25.0)
    # player
    # (calibration clip) units are the vision ground units: steady walk ~9-10.6/s, the
    # camera never rotated by itself, A/D strafe. A player-centred circle has radius ~3.8.
    walk_speed: Rng = (8.5, 11.0)
    accel_time: Rng = (0.03, 0.12)
    hitbox: float = 0.4
    dash_distance: Rng = (4.5, 6.5)  # (calibration clip) ~5-6 on top of walking
    dash_duration: Rng = (0.25, 0.35)
    dash_charges: int = 3  # (video)
    dash_gap: Rng = (0.9, 1.05)  # (video) "1,0" countdown between dashes
    dash_charge_regen: Rng = (3.0, 3.8)  # (video) 3.25-3.5 s per charge
    dash_empty_lockout: Rng = (7.5, 8.5)  # (video) ~8 s after the third dash
    dash_iframes: bool = False  # (user) dashing does not protect from damage
    # boss
    kill_time: Rng = (70.0, 110.0)  # seconds of uninterrupted macro damage (video: whole fight ~130 s)
    boss_speed: Rng = (1.5, 3.0)
    boss_hitbox: Rng = (1.2, 1.8)
    melee_range: Rng = (2.0, 3.0)  # (video) boss shows "2 m" while being hit
    skill_interval: Rng = (5.0, 8.0)  # (video) ~7 s between attacks
    enrage_below: float = 0.4  # boss HP fraction where attacks speed up
    enrage_speedup: float = 0.75
    skills: tuple[str, ...] = ("circle_on_player",)
    skill_overrides: dict = field(default_factory=dict)  # user/boss overrides, kept across levels
    level_overrides: dict = field(default_factory=dict)  # set by the curriculum preset only
    boss_skills: tuple[str, ...] | None = None  # this boss's skill list; levels use growing subsets of it
    # macro (R)
    macro_latency: Rng = (0.1, 0.35)
    macro_press_loss: float = 0.02
    # (user + recordings 33/44) walking does NOT reliably turn the macro off: it stayed ON
    # through many moves. "Sometimes" it does → each movement bout cancels it with this
    # chance, drawn per episode, so the agent must read the indicator, not assume.
    macro_move_cancel_chance: Rng = (0.0, 0.4)
    macro_cancel_delay: float = 0.05  # seconds of movement before the game cancels it
    safe_exit_slack: float = 0.5  # leaving a telegraph with less time than this is a "late" dodge
    attack_resume: Rng = (0.2, 0.5)  # standing time before the first hit lands
    noise: NoiseSpec = field(default_factory=NoiseSpec)

    def skill(self, name: str) -> SkillSpec:
        spec = SKILLS_ALL[name]
        for over in (self.level_overrides.get(name), self.skill_overrides.get(name)):
            if over:
                spec = replace(spec, **over)
        return spec

    def with_level(self, level: int) -> "SimConfig":
        return level_config(level, base=self)


def level_config(level: int, base: SimConfig | None = None) -> SimConfig:
    """Curriculum presets: 0 = one slow circle type, 4 = everything + full noise."""
    b = base or SimConfig()
    level = int(np.clip(level, 0, 4))
    bs = b.boss_skills

    def pick(n: int | None, default: tuple[str, ...]) -> tuple[str, ...]:
        if not bs:
            return default
        return tuple(bs) if n is None else tuple(bs[:n])

    if level == 0:
        return replace(
            b,
            level=0,
            skills=pick(1, ("circle_on_player",)),
            skill_interval=(5.0, 8.0),
            level_overrides={"circle_on_player": {"telegraph": (1.6, 2.0)}},
            noise=NoiseSpec(),
        )
    if level == 1:
        return replace(
            b,
            level=1,
            skills=pick(2, ("circle_on_player", "cone", "nova")),
            skill_interval=(4.5, 7.5),
            level_overrides={},
            noise=NoiseSpec(center_sigma=0.1, tti_sigma=0.03, latency=(0.0, 0.05)),
        )
    if level == 2:
        return replace(
            b,
            level=2,
            skills=pick(4, ("circle_on_player", "cone", "nova", "rain", "line")),
            skill_interval=(4.5, 7.5),
            level_overrides={},
            noise=NoiseSpec(
                center_sigma=0.2,
                radius_rel_sigma=0.04,
                tti_sigma=0.06,
                miss_prob=0.01,
                late_detect=(0.0, 0.1),
                boss_pos_sigma=0.3,
                hp_sigma=0.01,
                indicator_flip=0.005,
                latency=(0.0, 0.1),
                use_fov=True,
            ),
        )
    noise = NoiseSpec(
        center_sigma=0.3,
        radius_rel_sigma=0.06,
        tti_sigma=0.1,
        miss_prob=0.03,
        late_detect=(0.0, 0.15),
        false_pos_rate=0.05,
        boss_pos_sigma=0.4,
        hp_sigma=0.015,
        indicator_flip=0.01,
        latency=(0.05, 0.2),
        use_fov=True,
    )
    return replace(
        b,
        level=level,
        skills=pick(None, ("circle_on_player", "cone", "nova", "rain", "line", "summon_burst")),
        skill_interval=(4.5, 7.5) if level == 3 else (3.5, 6.0),
        level_overrides={},
        noise=noise,
    )


def load_yaml(path: str | Path) -> SimConfig:
    """Load overrides from YAML (keys = SimConfig fields; `level` applies a preset first)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    level = int(data.pop("level", 0))
    noise = data.pop("noise", None)
    valid = {f.name for f in fields(SimConfig)}
    kw = {}
    for k, v in data.items():
        if k not in valid:
            raise KeyError(f"unknown sim config key: {k}")
        kw[k] = tuple(v) if isinstance(v, list) else v
    # preset computed with the boss's own skill list, then the file's values on top
    cfg = level_config(level, replace(SimConfig(), **kw))
    cfg = replace(cfg, **{k: v for k, v in kw.items() if k != "skills"})
    if noise:
        cfg = replace(cfg, noise=replace(cfg.noise, **{k: tuple(v) if isinstance(v, list) else v for k, v in noise.items()}))
    return cfg
