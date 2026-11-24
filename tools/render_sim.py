"""Render simulator episodes to an mp4 (top-down view).

    python tools/render_sim.py --model models/kor_bekci_v2 --level 3 --seconds 90 --out sim.mp4
    python tools/render_sim.py --policy expert --level 2 --out expert.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bossai.evaluation.baselines import ExpertPolicy, ModelPolicy, NeverDodgePolicy, RandomPolicy  # noqa: E402
from bossai.sim.env import BossFightEnv  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--checkpoint", default="model.zip")
    ap.add_argument("--policy", default="model", choices=["model", "expert", "never_dodge", "random"])
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="sim.mp4")
    args = ap.parse_args(argv)

    if args.policy == "model":
        from bossai.training.registry import load_model

        pol = ModelPolicy(load_model(args.model, checkpoint=args.checkpoint))
    else:
        pol = {"expert": ExpertPolicy(), "never_dodge": NeverDodgePolicy(), "random": RandomPolicy()}[args.policy]
    env = BossFightEnv(level=args.level)
    obs, _ = env.reset(seed=args.seed)
    writer = None
    done = False
    while not done and env.world.t < args.seconds:
        if isinstance(pol, ModelPolicy):
            pol.set_obs(obs)
        obs, _, term, trunc, info = env.step(pol.act(env))
        done = term or trunc
        frame = env.render()[:, :, ::-1].copy()
        if args.title:
            cv2.putText(frame, args.title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 10, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    writer.release()
    st = env.stats
    print(f"{args.out}: t={env.world.t:.1f}s outcome={st.outcome} hits={st.hits} dodges={st.dodges} late={st.late_dodges}")


if __name__ == "__main__":
    main()
