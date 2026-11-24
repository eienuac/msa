"""Run folders: everything needed to reload and trust a trained model.

models/<run>/
  model.zip           SB3 MaskablePPO (weights, optimiser, hyperparameters)
  obs_schema.json     feature list, grid spec, schema hash
  action_schema.json  action heads and meanings
  sim_config.json     simulator parameters used for training
  reward.json         reward weights
  meta.json           git commit, seed, timestamps, training steps
  eval_report.json    written by evaluation
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from pathlib import Path

from ..agent.action_space import MOVE_NAMES, NVEC
from ..agent.observation_encoder import ObservationEncoder


def _jsonable(o):
    if dataclasses.is_dataclass(o):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(o).items()}
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def write_run_files(run_dir: Path, encoder: ObservationEncoder, sim_cfg, reward_weights, meta: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    schema = encoder.schema()
    schema["hash"] = encoder.schema_hash()
    (run_dir / "obs_schema.json").write_text(json.dumps(schema, indent=2))
    (run_dir / "action_schema.json").write_text(
        json.dumps({"nvec": list(NVEC), "move": MOVE_NAMES, "dash": ["no", "yes"], "macro": ["keep", "want_on", "want_off"]}, indent=2)
    )
    (run_dir / "sim_config.json").write_text(json.dumps(_jsonable(sim_cfg), indent=2))
    (run_dir / "reward.json").write_text(json.dumps(_jsonable(reward_weights), indent=2))
    meta = dict(meta, git_commit=git_commit(), saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


class SchemaMismatch(RuntimeError):
    pass


def load_model(run_dir: str | Path, encoder: ObservationEncoder | None = None, checkpoint: str = "model.zip"):
    """Load a MaskablePPO model after verifying the observation schema."""
    from sb3_contrib import MaskablePPO

    run_dir = Path(run_dir)
    encoder = encoder or ObservationEncoder()
    saved = json.loads((run_dir / "obs_schema.json").read_text())
    if saved.get("hash") != encoder.schema_hash():
        raise SchemaMismatch(
            f"model {run_dir} was trained with observation schema {saved.get('hash')}, "
            f"current encoder is {encoder.schema_hash()}"
        )
    return MaskablePPO.load(run_dir / checkpoint, device="cpu")
