"""Evaluate policies on a fixed bank of seeded scenarios.

Example:
    python -m bossai.evaluation.evaluate --model models/kor_bekci_v1 --levels 0 2 4 --episodes 20 --baselines
    python -m bossai.evaluation.evaluate --model models/kor_bekci_v1 --levels 4 --episodes 5 --video worst
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..sim.env import BossFightEnv
from .baselines import BasePolicy, ExpertPolicy, ModelPolicy, NeverDodgePolicy, RandomPolicy

SEED_BASE = 10_000  # evaluation seeds never overlap with training seeds (0..n_envs)


def run_episode(env: BossFightEnv, policy: BasePolicy, seed: int, frames: list | None = None) -> dict:
    obs, _ = env.reset(seed=seed)
    total, done, info = 0.0, False, {}
    hit_skills: list[str] = []
    while not done:
        if isinstance(policy, ModelPolicy):
            policy.set_obs(obs)
        obs, r, term, trunc, info = env.step(policy.act(env))
        hit_skills += info.get("hit_skills", [])
        total += r
        done = term or trunc
        if frames is not None:
            frames.append(env.render())
    st = dict(info["episode_stats"])
    st.update(seed=seed, ret=total, hit_skills=hit_skills)
    return st


def summarize(eps: list[dict]) -> dict:
    minutes = max(sum(e["time"] for e in eps) / 60.0, 1e-9)
    kills = [e for e in eps if e["outcome"] == "kill"]
    return {
        "episodes": len(eps),
        "return": float(np.mean([e["ret"] for e in eps])),
        "kill_rate": len(kills) / len(eps),
        "death_rate": sum(e["outcome"] == "death" for e in eps) / len(eps),
        "timeout_rate": sum(e["outcome"] == "timeout" for e in eps) / len(eps),
        "time_to_kill": float(np.mean([e["time"] for e in kills])) if kills else None,
        "hits_per_min": sum(e["hits"] for e in eps) / minutes,
        "dodge_rate": sum(e["dodges"] for e in eps) / max(1, sum(e["threats"] for e in eps)),
        "dashes_per_min": sum(e["dashes"] for e in eps) / minutes,
        "r_per_min": sum(e["r_presses"] for e in eps) / minutes,
        "dir_changes_per_min": sum(e["direction_changes"] for e in eps) / minutes,
        "safe_action_share": sum(e["safe_moves"] for e in eps) / max(1, sum(e["steps"] for e in eps)),
        "hit_by_skill": dict(Counter(s for e in eps for s in e["hit_skills"])),
    }


def evaluate(policies: list[BasePolicy], levels: list[int], episodes: int) -> dict:
    report: dict = {}
    for level in levels:
        env = BossFightEnv(level=level)
        for pol in policies:
            eps = [run_episode(env, pol, SEED_BASE + level * 1000 + i) for i in range(episodes)]
            report.setdefault(f"level_{level}", {})[pol.name] = {"summary": summarize(eps), "episodes": eps}
    return report


def print_table(report: dict) -> None:
    cols = ["return", "kill_rate", "death_rate", "time_to_kill", "hits_per_min", "dodge_rate", "dashes_per_min", "r_per_min", "dir_changes_per_min"]
    print(f"{'level/policy':24s}" + "".join(f"{c[:11]:>12s}" for c in cols))
    for lvl, pols in report.items():
        for name, data in pols.items():
            s = data["summary"]
            row = "".join(f"{('-' if s[c] is None else f'{s[c]:.2f}'):>12s}" for c in cols)
            print(f"{lvl + '/' + name:24s}{row}")


def save_video(frames: list, path: Path, fps: int = 10) -> None:
    import cv2

    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f[:, :, ::-1])
    vw.release()


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default=None, help="run folder (models/<run>)")
    ap.add_argument("--checkpoint", type=str, default="model.zip")
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--video", choices=["worst", "first"], default=None, help="render one episode of the model to mp4")
    args = ap.parse_args(argv)

    policies: list[BasePolicy] = []
    model = None
    if args.model:
        from ..training.registry import load_model

        model = load_model(args.model, checkpoint=args.checkpoint)
        policies.append(ModelPolicy(model, "ppo"))
    if args.baselines or not policies:
        policies += [RandomPolicy(), NeverDodgePolicy(), ExpertPolicy(attack=False), ExpertPolicy()]
    report = evaluate(policies, args.levels, args.episodes)
    print_table(report)
    if args.model:
        out = Path(args.model) / "eval_report.json"
        slim = {lv: {p: d["summary"] for p, d in pols.items()} for lv, pols in report.items()}
        out.write_text(json.dumps(slim, indent=2))
        failures = [
            dict(e, level=lv)
            for lv, pols in report.items()
            for e in pols.get("ppo", {}).get("episodes", [])
            if e["outcome"] != "kill"
        ]
        (Path(args.model) / "eval_failures.jsonl").write_text("\n".join(json.dumps(f) for f in failures))
        if args.video and model is not None:
            lvl = args.levels[-1]
            eps = report[f"level_{lvl}"]["ppo"]["episodes"]
            pick = min(eps, key=lambda e: e["ret"]) if args.video == "worst" else eps[0]
            frames: list = []
            run_episode(BossFightEnv(level=lvl), ModelPolicy(model, "ppo"), pick["seed"], frames)
            path = Path(args.model) / f"episode_level{lvl}_seed{pick['seed']}.mp4"
            save_video(frames, path)
            print(f"video: {path}")
    return report


if __name__ == "__main__":
    main()
