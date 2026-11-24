---
name: train-and-evaluate
description: Train, resume, evaluate and publish the PPO boss agent. Use when starting or resuming training, after the simulator/reward changes, or when the user asks how well the agent plays.
---

# Training workflow

- Fresh run (curriculum 0→4):
  `python -m bossai.training.train --timesteps 3000000 --n-envs 8 --run-name <name> --checkpoint-every 100000`
- After reward/simulator changes that keep the observation schema: fine-tune from
  the last good checkpoint (copy it to `models/<new>/model.zip`, then
  `--resume models/<new> --run-name <new> --level-start 2`).
- After a container restart: background jobs are gone. Resume from the newest
  `models/<run>/checkpoints/step_*.zip` with `--level-start` = level reached.
- Run training with `nohup ... &` and a background watcher that exits when the
  log prints `saved` or the process disappears.
- Progress: `grep -E "total_timesteps|kill_rate|death_rate|early_dodge_rate|  level" models/<run>/train.log | tail`

# Evaluation
- `python -m bossai.evaluation.evaluate --model models/<run> --levels 2 3 4 --episodes 20 --baselines`
  Compare with `expert` (rule-based planner), `pacifist`, `never_dodge`, `random`.
  Report kill/death rate, hits per minute, early-dodge rate, dashes and R presses per minute.
- Failures are written to `models/<run>/eval_failures.jsonl`; `--video worst` renders the worst one.
- Top-down video for the user: `python tools/render_sim.py --model models/<run> --level 3 --seconds 90 --out sim.mp4`
- Shadow mode on recordings: `python -m bossai.runtime.live --model models/<run> --source VIDEO --out shadow.mp4`

# Publishing
Copy `model.zip`, `obs_schema.json`, `action_schema.json`, `sim_config.json`,
`reward.json`, `meta.json`, `eval_report.json` into `pretrained/<run>/`, commit, push.
Never commit checkpoints or tensorboard logs.
