"""Per-boss perception settings (configs/bosses/<id>.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from .hazard_detection import TelegraphColor
from .rim_detection import RimStyle

BOSS_DIR = Path(__file__).resolve().parents[2] / "configs" / "bosses"


@dataclass(frozen=True)
class BossProfile:
    id: str
    name: str
    ocr_keywords: tuple[str, ...]
    telegraph_style: str = "filled"  # "filled" (zones with a growing fill) or "rim" (bright ring only)
    telegraph_color: TelegraphColor = field(default_factory=TelegraphColor)
    rim_style: RimStyle = field(default_factory=RimStyle)
    default_telegraph: float = 1.25
    min_confirm_age: float = 0.0
    sim_config: str | None = None


def list_profiles() -> list[str]:
    return sorted(p.stem for p in BOSS_DIR.glob("*.yaml"))


def load_profile(boss: str | Path) -> BossProfile:
    path = Path(boss)
    if not path.suffix:
        path = BOSS_DIR / f"{boss}.yaml"
    data = yaml.safe_load(path.read_text()) or {}

    def tup(v):
        return tuple(v) if isinstance(v, list) else v

    color = replace(TelegraphColor(), **{k: tup(v) for k, v in (data.get("telegraph_color") or {}).items()})
    rim = replace(RimStyle(), **{k: tup(v) for k, v in (data.get("rim_style") or {}).items()})
    style = data.get("telegraph_style", "filled")
    if style not in ("filled", "rim"):
        raise ValueError(f"{path}: telegraph_style must be 'filled' or 'rim'")
    return BossProfile(
        id=path.stem,
        name=data.get("name", path.stem),
        ocr_keywords=tuple(data.get("ocr_keywords") or ()),
        telegraph_style=style,
        telegraph_color=color,
        rim_style=rim,
        default_telegraph=float(data.get("default_telegraph", 1.25)),
        min_confirm_age=float(data.get("min_confirm_age", 0.0)),
        sim_config=data.get("sim_config"),
    )
