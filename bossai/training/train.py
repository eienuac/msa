"""Train the boss agent with MaskablePPO in the simulator.

Example:
    python -m bossai.training.train --timesteps 2000000 --n-envs 4 --run-name kor_bekci_v1
    tensorboard --logdir models/kor_bekci_v1/tb
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ..agent.observation_encoder import ObservationEncoder
from ..agent.policy_net import BossFeatureExtractor
from ..rewards.reward_system import RewardWeights
from ..sim.config import SimConfig, level_config, load_yaml
from ..sim.env import BossFightEnv
from .callbacks import CurriculumCallback, EpisodeLoggerCallback, RunCheckpointCallback
from .registry import write_run_files


def linear_schedule(start: float, end: float = 0.0):
    return lambda progress: end + (start - end) * progress


def build_model(env, seed: int, tb_dir: str | None, lr: float = 3e-4, n_steps: int = 512, batch_size: int = 512, ent_coef: float = 0.01):
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=linear_schedule(lr, lr * 0.1),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=6,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            features_extractor_class=BossFeatureExtractor,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        tensorboard_log=tb_dir,
        seed=seed,
        device="cpu",
        verbose=1,
    )


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--level-start", type=int, default=0)
    ap.add_argument("--level-max", type=int, default=4)
    ap.add_argument("--sim-config", type=str, default=None, help="YAML overrides for the simulator")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--models-dir", type=str, default="models")
    ap.add_argument("--resume", type=str, default=None, help="run folder to continue training from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=200_000)
    ap.add_argument("--resume-from", type=str, default=None, help="checkpoint inside --resume to start from, e.g. checkpoints/step_4306776")
    ap.add_argument("--replay-prob", type=float, default=0.25, help="share of episodes from a lower level than the current one")
    ap.add_argument("--lr", type=float, default=None, help="learning rate (start of a linear decay to 1/5); default 3e-4, or the saved schedule when resuming")
    ap.add_argument("--ent-coef", type=float, default=None, help="entropy bonus; default 0.01, or the saved value when resuming")
    ap.add_argument("--dummy-vec", action="store_true", help="single-process envs (debugging)")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.models_dir) / run_name
    base_cfg = load_yaml(args.sim_config) if args.sim_config else SimConfig()
    start_cfg = level_config(args.level_start, base_cfg)
    encoder = ObservationEncoder()
    weights = RewardWeights()

    vec_cls = DummyVecEnv if args.dummy_vec or args.n_envs == 1 else SubprocVecEnv
    env = make_vec_env(
        BossFightEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_cls,
        env_kwargs=dict(cfg=start_cfg, reward_weights=weights, replay_prob=args.replay_prob),
    )
    write_run_files(
        run_dir,
        encoder,
        start_cfg,
        weights,
        meta={"seed": args.seed, "timesteps": args.timesteps, "level_start": args.level_start, "level_max": args.level_max},
    )
    tb = str(run_dir / "tb")
    if args.resume:
        src = Path(args.resume) / (f"{args.resume_from}.zip" if args.resume_from else "model.zip")
        model = MaskablePPO.load(src, env=env, device="cpu", tensorboard_log=tb)
        if args.lr is not None:
            # SB3 measures progress from step 0, so a resumed run starts part-way through it;
            # stretch the schedule over this run only
            p0 = args.timesteps / (args.timesteps + model.num_timesteps)
            sched = linear_schedule(args.lr, args.lr * 0.2)
            model.lr_schedule = lambda p: sched(min(1.0, p / p0))
            model.learning_rate = args.lr
        if args.ent_coef is not None:
            model.ent_coef = args.ent_coef
    else:
        model = build_model(env, args.seed, tb, **({"lr": args.lr} if args.lr else {}), **({"ent_coef": args.ent_coef} if args.ent_coef is not None else {}))
    curriculum = CurriculumCallback(args.level_start, args.level_max)
    curriculum.verbose = 1
    callbacks = [curriculum, EpisodeLoggerCallback(), RunCheckpointCallback(run_dir, args.checkpoint_every)]
    t0 = time.time()
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=not args.resume)
    finally:
        model.save(run_dir / "model.zip")
        write_run_files(
            run_dir,
            encoder,
            level_config(curriculum.level, base_cfg),
            weights,
            meta={
                "seed": args.seed,
                "timesteps": int(model.num_timesteps),
                "level_reached": curriculum.level,
                "curriculum_history": curriculum.history,
                "train_seconds": round(time.time() - t0, 1),
            },
        )
        env.close()
    print(f"saved {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
