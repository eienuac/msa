from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from bossai.agent.action_space import NVEC
from bossai.evaluation.baselines import ExpertPolicy, NeverDodgePolicy
from bossai.sim.config import NoiseSpec, SimConfig, level_config
from bossai.sim.env import BossFightEnv
from bossai.sim.world import BossWorld


def run_episode(env, policy, seed):
    env.reset(seed=seed)
    total, done, info = 0.0, False, {}
    while not done:
        _, r, term, trunc, info = env.step(policy.act(env))
        total += r
        done = term or trunc
    return total, info["episode_stats"]


def test_gymnasium_api():
    env = BossFightEnv(level=2)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert env.action_masks().shape == (sum(NVEC),)


def test_seeded_episodes_are_deterministic():
    a = BossFightEnv(level=3)
    b = BossFightEnv(level=3)
    pol = ExpertPolicy()
    ra, sa = run_episode(a, pol, 7)
    rb, sb = run_episode(b, pol, 7)
    assert ra == pytest.approx(rb)
    assert sa["steps"] == sb["steps"]


def test_dash_charges_and_lockout():
    cfg = level_config(0, SimConfig(dash_gap=(0.5, 0.5), dash_empty_lockout=(8.0, 8.0), dash_charge_regen=(4.0, 4.0)))
    w = BossWorld(cfg, np.random.default_rng(0))
    w.hazards.clear()
    w.next_skill_t = 1e9
    held = {"w"}
    ev = w.step(held, ["q"], 0.6)
    assert ev.dashes == 1 and w.charges == 2
    ev = w.step(held, ["q"], 0.6)
    assert ev.dashes == 1 and w.charges == 1
    ev = w.step(held, ["q"], 0.1)
    assert ev.dashes == 1 and w.charges == 0
    ev = w.step(held, ["q"], 0.1)  # empty: the tap is wasted
    assert ev.dashes == 0 and ev.wasted_dash_taps == 1
    w.step(held, [], 8.0)
    assert w.charges == 3


def test_dash_needs_direction():
    w = BossWorld(level_config(0), np.random.default_rng(0))
    w.next_skill_t = 1e9
    ev = w.step(set(), ["q"], 0.1)
    assert ev.dashes == 0 and ev.wasted_dash_taps == 1


def test_macro_toggle_has_latency_and_deals_damage():
    cfg = level_config(0, SimConfig(macro_latency=(0.3, 0.3), macro_press_loss=0.0))
    w = BossWorld(cfg, np.random.default_rng(1))
    w.next_skill_t = 1e9
    w.macro_on = False
    w.boss_pos = w.player_pos + np.array([0.0, 2.0])
    w.step(set(), ["r"], 0.1)
    assert not w.macro_on
    w.step(set(), [], 0.3)
    assert w.macro_on
    ev = w.step(set(), [], 2.0)
    assert ev.boss_damage > 0


def test_reward_ranks_expert_above_baselines():
    env = BossFightEnv(level=0)
    expert = np.mean([run_episode(env, ExpertPolicy(), s)[0] for s in range(2)])
    pacifist = np.mean([run_episode(env, ExpertPolicy(attack=False), s)[0] for s in range(2)])
    reckless = np.mean([run_episode(env, NeverDodgePolicy(), s)[0] for s in range(2)])
    assert expert > pacifist > reckless
    assert expert > 5.0 and reckless < -10.0


def test_boss_yaml_config_loads():
    from pathlib import Path

    from bossai.sim.config import load_yaml

    cfg = load_yaml(Path(__file__).parents[1] / "configs" / "kor_bekci_sim.yaml")
    assert cfg.dash_charges == 3 and cfg.skill_interval == (5.0, 8.0)
    env = BossFightEnv(cfg=cfg)
    env.reset(seed=0)
    env.step(np.array([0, 0, 0]))


def test_saved_model_refuses_other_schema(tmp_path):
    import json

    from bossai.agent.observation_encoder import ObservationEncoder
    from bossai.core.geometry import GridSpec
    from bossai.training.registry import SchemaMismatch, load_model, write_run_files

    write_run_files(tmp_path, ObservationEncoder(), SimConfig(), {}, meta={})
    saved = json.loads((tmp_path / "obs_schema.json").read_text())
    assert saved["hash"] == ObservationEncoder().schema_hash()
    with pytest.raises(SchemaMismatch):
        load_model(tmp_path, encoder=ObservationEncoder(grid=GridSpec(size=24)))


def _world_with_circle(telegraph: float):
    from bossai.core.game_state import Hazard, HazardShape
    from bossai.sim.world import SimHazard

    w = BossWorld(level_config(0, SimConfig(walk_speed=(5.0, 5.0), accel_time=(0.01, 0.01), dash_distance=(6.0, 6.0))), np.random.default_rng(0))
    w.next_skill_t = 1e9
    w.boss_pos = w.player_pos + np.array([0.0, 15.0])
    h = Hazard(0, HazardShape.CIRCLE, w.player_pos.copy(), 3.0, telegraph, damage=0.2)
    w.hazards = [SimHazard(h, w.t + telegraph)]
    return w


def test_early_exit_counts_as_dodge_late_exit_does_not():
    early = _world_with_circle(1.5)
    ev = early.step({"s"}, ["q"], 1.6)  # dash out right away
    assert ev.dodges == 1 and ev.late_dodges == 0 and ev.hits == 0
    late = _world_with_circle(1.5)
    late.step(set(), [], 0.65)  # wait, then walk out at the last moment
    ev = late.step({"s"}, [], 1.0)
    assert ev.hits == 0 and ev.late_dodges == 1 and ev.dodges == 0


def test_walking_cancels_macro_only_sometimes():
    # recordings 33/44: the macro stayed on through many moves; the user says it "sometimes" turns off
    always = BossWorld(level_config(0, SimConfig(macro_move_cancel_chance=(1.0, 1.0))), np.random.default_rng(3))
    never = BossWorld(level_config(0, SimConfig(macro_move_cancel_chance=(0.0, 0.0))), np.random.default_rng(3))
    for w in (always, never):
        w.next_skill_t = 1e9
        w.macro_on = True
        w.step({"w"}, [], 0.3)
    assert not always.macro_on
    assert never.macro_on


def test_dash_gives_no_invulnerability():
    w = _world_with_circle(0.05)
    ev = w.step({"s"}, ["q"], 0.1)  # mid-dash at the impact moment, still inside the circle
    assert ev.hits == 1


def test_summon_burst_pushes_back_and_can_steal_target():
    cfg = level_config(4, SimConfig(skill_overrides={"summon_burst": {"target_steal_prob": 1.0, "knockback": (1.5, 1.5)}}))
    w = BossWorld(cfg, np.random.default_rng(0))
    w.hazards.clear()
    w.boss_pos = w.player_pos + np.array([0.0, 3.0])
    w.boss_speed = 0.0
    w._summon_spec = cfg.skill("summon_burst")
    w._summon_at = w.t + 0.05
    w.next_skill_t = 1e9
    hp = w.player_hp
    before = np.linalg.norm(w.boss_pos - w.player_pos)
    w.step(set(), [], 0.1)
    assert np.linalg.norm(w.boss_pos - w.player_pos) == pytest.approx(before + 1.5, abs=0.05)
    assert w.player_hp == hp and not w.boss_targeted
    w.macro_on = False
    w.boss_pos = w.player_pos + np.array([0.0, 2.0])
    w.step(set(), ["r"], 1.0)  # R next to the boss re-selects it and starts the macro
    assert w.boss_targeted and w.macro_on


def test_level_presets_do_not_leak_between_levels():
    from bossai.sim.config import level_config

    l0 = level_config(0)
    assert l0.skill("circle_on_player").telegraph == (1.6, 2.0)
    l3 = level_config(3, l0)  # what the curriculum does
    assert l3.skill("circle_on_player").telegraph == (1.0, 1.5)
    user = level_config(3, SimConfig(skill_overrides={"cone": {"radius": (9.0, 9.0)}}))
    assert level_config(4, user).skill("cone").radius == (9.0, 9.0)


def _spider_world(skill: str, seed: int = 0):
    from bossai.sim.config import load_yaml

    cfg = level_config(4, load_yaml(Path(__file__).parents[1] / "configs" / "orumcek_ana_sim.yaml"))
    cfg = replace(cfg, skills=(skill,), noise=NoiseSpec())
    w = BossWorld(cfg, np.random.default_rng(seed))
    w.hazards.clear()
    w.next_skill_t = w.t  # cast on the next substep
    return w


def test_spider_curriculum_uses_its_own_skills():
    from bossai.sim.config import load_yaml

    cfg = load_yaml(Path(__file__).parents[1] / "configs" / "orumcek_ana_sim.yaml")
    assert level_config(0, cfg).skills == ("follow_ring",)
    assert set(level_config(4, cfg).skills) == {"follow_ring", "boss_disc", "poison_wave", "beams", "spawn_adds"}
    env = BossFightEnv(cfg=cfg, level=4)
    env.reset(seed=0)
    for _ in range(50):
        env.step(np.array([0, 0, 0]))


def test_follow_ring_cannot_be_walked_out_of_and_stuns():
    # (user) the ring follows until it lands; walking away does not help
    w = _spider_world("follow_ring")
    w.step(set(), [], 0.1)
    sh = next(s for s in w.hazards if s.hz.skill == "follow_ring")
    hp = w.player_hp
    for _ in range(30):
        w.step({"d"}, [], 0.1)
        if sh.resolved:
            break
        assert np.linalg.norm(sh.hz.center - w.player_pos) < 0.6
    assert sh.resolved and w.player_hp < hp and w.stunned_until > w.t - 0.2
    w.stunned_until = w.t + 1.0
    before = w.player_pos.copy()
    w.step({"w"}, ["q"], 0.5)  # stunned: no walking, no dash
    np.testing.assert_allclose(w.player_pos, before, atol=1e-6)


def test_shield_blocks_the_follow_ring_only_when_held_at_impact():
    w = _spider_world("follow_ring")
    w.step(set(), [], 0.1)
    sh = next(s for s in w.hazards if s.hz.skill == "follow_ring")
    w.step(set(), [], sh.impact_t - w.t - 0.4)
    hp = w.player_hp
    ev = w.step({"e"}, [], 0.6)  # E held across the impact
    assert ev.shield_blocks == 1 and w.player_hp == hp and w.stunned_until < 0
    # too early: the shield is down (max ~1 s) when the next ring lands, and it is on cooldown
    w.hazards.clear()
    w.next_skill_t = w.t
    w.step(set(), [], 0.1)
    sh = next(s for s in w.hazards if s.hz.skill == "follow_ring")
    w.step({"e"}, [], sh.impact_t - w.t + 0.1)
    assert w.player_hp < hp


def test_shield_cooldown_and_max_hold():
    w = _spider_world("beams")
    w.step({"e"}, [], 0.1)
    assert w.shield_up
    w.step({"e"}, [], 1.5)
    assert not w.shield_up  # dropped after ~1 s even while held
    w.step(set(), [], 0.1)
    w.step({"e"}, [], 0.1)
    assert not w.shield_up  # cooldown


def test_poison_wave_hits_outside_first_then_inside():
    # (user) stay inside the ring for the first hit, then get out
    w = _spider_world("poison_wave")
    w.player_pos = w.boss_pos + np.array([2.5, 0.0])
    w.step(set(), [], 0.05)
    outer, inner = sorted((s for s in w.hazards if s.hz.skill == "poison_wave"), key=lambda s: s.impact_t)
    assert outer.hz.inner_radius > 0 and inner.hz.inner_radius == 0
    assert inner.impact_t - outer.impact_t >= 0.9
    hp = w.player_hp
    w.step(set(), [], outer.impact_t - w.t + 0.1)  # waited inside: the outer hit misses
    assert w.player_hp == hp
    away = (w.player_pos - w.boss_pos) / np.linalg.norm(w.player_pos - w.boss_pos)
    key = "d" if away[0] > 0 else "a"
    w.step({key}, ["q"], inner.impact_t - w.t + 0.1)  # dash out before the inside is hit
    assert w.player_hp == hp
    # standing still through both hits: only the inside one lands
    w2 = _spider_world("poison_wave", seed=1)
    w2.player_pos = w2.boss_pos + np.array([2.5, 0.0])
    w2.step(set(), [], 0.05)
    end = max(s.impact_t for s in w2.hazards) + 0.1
    ev = w2.step(set(), [], end - w2.t)
    assert ev.hits == 1 and 1.0 - w2.player_hp <= 0.18 + 1e-9  # one hit never kills (user)


def test_shield_reflex_in_the_env():
    from bossai.sim.config import load_yaml

    cfg = load_yaml(Path(__file__).parents[1] / "configs" / "orumcek_ana_sim.yaml")
    cfg = replace(level_config(1, cfg), skills=("follow_ring",))
    env = BossFightEnv(cfg=cfg)
    env.reset(seed=0)
    blocks = hits = 0
    for _ in range(600):
        _, _, term, trunc, info = env.step(np.array([0, 0, 0]))
        hits += info["hit_skills"].count("follow_ring")
        if term or trunc:
            break
    assert env.stats.shield_blocks >= 3 and env.stats.shield_blocks > 2 * hits


def test_beams_cross_near_the_player():
    w = _spider_world("beams", seed=3)
    w.step(set(), [], 0.1)
    beams = [s.hz for s in w.hazards if s.hz.skill == "beams"]
    assert len(beams) >= 2
    from bossai.core.game_state import HazardShape

    assert all(b.shape is HazardShape.RECT and b.radius >= 25 for b in beams)


def test_replay_episodes_use_lower_levels():
    # a share of the episodes comes from easier levels so earlier skills are not forgotten
    from bossai.sim.env import BossFightEnv

    env = BossFightEnv(level=3, replay_prob=1.0)
    seen = set()
    for seed in range(12):
        env.reset(seed=seed)
        seen.add(env.ep_cfg.level)
    assert seen <= {0, 1, 2} and len(seen) >= 2
    assert env.get_level() == 3  # the curriculum level itself is unchanged
    env = BossFightEnv(level=3, replay_prob=0.0)
    env.reset(seed=0)
    assert env.ep_cfg.level == 3


def test_planner_waits_inside_the_poison_ring_then_leaves():
    from bossai.core.game_state import BossState, GameState, Hazard, HazardShape, PlayerState
    from bossai.core.safety import EscapeParams, SafetyPlanner

    boss = np.array([0.0, 0.0])

    def state(outer_tti: float | None, inner_tti: float, player=(2.5, 0.0)):
        hz = [Hazard(1, HazardShape.CIRCLE, boss, 7.5, inner_tti)]
        if outer_tti is not None:
            hz.append(Hazard(0, HazardShape.CIRCLE, boss, 60.0, outer_tti, inner_radius=7.5))
        return GameState(
            t=0.0,
            player=PlayerState(pos=np.array(player), hp=1.0, max_hp=1.0),
            boss=BossState(pos=boss, hp=1.0, max_hp=1.0),
            hazards=hz,
        )

    from bossai.core.game_state import DIRECTION_VECTORS
    from bossai.core.safety import ray_features

    planner = SafetyPlanner(EscapeParams(), min_slack=0.5)
    first = planner.plan(state(0.9, 2.1))
    # stay inside for the outer hit (at most a step towards the edge, never a dash out)
    assert not first.dash
    if first.direction:
        rf = ray_features(state(0.9, 2.1), planner.p, DIRECTION_VECTORS[first.direction : first.direction + 1])
        assert 2.5 + rf.exit_dist[0] < 7.5 - planner.p.hitbox - planner.p.margin
    assert planner.plan(state(0.9, 2.6)).wait  # plenty of time for the second hit: just wait
    after = planner.plan(state(None, 1.1))
    assert after.direction != 0  # then leave before the inside is hit
    outside = planner.plan(state(1.0, 2.2, player=(9.0, 0.0)))
    assert outside.direction != 0  # caught outside the ring: step in before the outer hit
