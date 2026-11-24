import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from bossai.agent.action_space import NVEC
from bossai.evaluation.baselines import ExpertPolicy, NeverDodgePolicy
from bossai.sim.config import SimConfig, level_config
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
