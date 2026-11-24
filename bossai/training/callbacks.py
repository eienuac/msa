"""Training callbacks: curriculum, per-component reward logging, run checkpoints."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """Raises the simulator level when the agent reliably wins at the current one."""

    def __init__(self, start: int, max_level: int, window: int = 60, kill_rate: float = 0.8, max_death_rate: float = 0.1):
        super().__init__()
        self.level = start
        self.max_level = max_level
        self.window = window
        self.kill_rate = kill_rate
        self.max_death_rate = max_death_rate
        self.outcomes: deque[str] = deque(maxlen=window)
        self.history: list[tuple[int, int]] = []

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_level", self.level)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            st = info.get("episode_stats")
            if st is None or st.get("level") != self.level:
                continue
            self.outcomes.append(st["outcome"])
        if len(self.outcomes) >= self.window and self.level < self.max_level:
            kills = sum(o == "kill" for o in self.outcomes) / len(self.outcomes)
            deaths = sum(o == "death" for o in self.outcomes) / len(self.outcomes)
            if kills >= self.kill_rate and deaths <= self.max_death_rate:
                self.level += 1
                self.training_env.env_method("set_level", self.level)
                self.outcomes.clear()
                self.history.append((self.num_timesteps, self.level))
                if self.verbose:
                    print(f"[curriculum] step {self.num_timesteps}: level -> {self.level}")
        self.logger.record("curriculum/level", self.level)
        return True


class EpisodeLoggerCallback(BaseCallback):
    """Logs reward components and episode statistics to TensorBoard."""

    def __init__(self, every: int = 2048):
        super().__init__()
        self.every = every
        self.comp = defaultdict(float)
        self.n = 0
        self.eps: deque[dict] = deque(maxlen=100)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for k, v in info.get("reward_components", {}).items():
                self.comp[k] += v
            self.n += 1
            if "episode_stats" in info:
                self.eps.append(info["episode_stats"])
        if self.num_timesteps % self.every < self.training_env.num_envs and self.n:
            for k, v in self.comp.items():
                self.logger.record(f"reward/{k}", v / self.n * 100)  # per 100 steps
            self.comp.clear()
            self.n = 0
            if self.eps:
                e = list(self.eps)
                self.logger.record("episode/kill_rate", np.mean([x["outcome"] == "kill" for x in e]))
                self.logger.record("episode/death_rate", np.mean([x["outcome"] == "death" for x in e]))
                self.logger.record("episode/hits", np.mean([x["hits"] for x in e]))
                threats = max(1, np.sum([x["threats"] for x in e]))
                self.logger.record("episode/dodge_rate", np.sum([x["dodges"] + x.get("late_dodges", 0) for x in e]) / threats)
                self.logger.record("episode/early_dodge_rate", np.sum([x["dodges"] for x in e]) / threats)
                self.logger.record("episode/dashes", np.mean([x["dashes"] for x in e]))
                self.logger.record("episode/r_presses", np.mean([x["r_presses"] for x in e]))
                self.logger.record("episode/time", np.mean([x["time"] for x in e]))
        return True


class RunCheckpointCallback(BaseCallback):
    def __init__(self, run_dir: Path, every: int):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.every = every
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            self.model.save(self.run_dir / "checkpoints" / f"step_{self.num_timesteps}")
            self.model.save(self.run_dir / "model.zip")
        return True
