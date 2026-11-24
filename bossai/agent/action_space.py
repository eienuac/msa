"""Policy action space: MultiDiscrete [move(10), dash(2), macro(3)].

move : 0 idle, 1..8 = N, NE, E, SE, S, SW, W, NW (camera frame), 9 = move to safe zone
dash : 0 no, 1 dash in the move direction
macro: 0 keep, 1 want ON, 2 want OFF

The policy chooses the *wanted* macro state; MacroStateManager decides
whether R is actually pressed.

Masks only remove actions that are impossible or meaningless right now.
Undesirable-but-possible actions are handled with reward costs instead.
"""

from __future__ import annotations

import numpy as np

from ..control.action_controller import AgentIntent
from ..core.game_state import GameState, MacroState
from ..core.safety import SafetyPlanner

NVEC = (10, 2, 3)
MOVE_TO_SAFE = 9
MOVE_NAMES = ["idle", "N", "NE", "E", "SE", "S", "SW", "W", "NW", "to_safe"]


def action_masks(state: GameState, macro: MacroState) -> np.ndarray:
    move = np.ones(NVEC[0], bool)
    dash = np.ones(NVEC[1], bool)
    mac = np.ones(NVEC[2], bool)
    if state.player.dashing:
        move[1:] = False
    if not state.player.can_dash:
        dash[1] = False
    mac[1] = macro == MacroState.OFF and state.boss.targeted
    mac[2] = macro == MacroState.ON
    return np.concatenate([move, dash, mac])


def decode(action: np.ndarray, state: GameState, planner: SafetyPlanner | None, prev_dir: int) -> tuple[AgentIntent, bool]:
    """Convert a raw action to an intent. Returns (intent, invalid_flag)."""
    move, dash, mac = (int(x) for x in action)
    invalid = False
    if move == MOVE_TO_SAFE:
        if planner is None:
            move = 0
        else:
            plan = planner.plan(state, prev_dir)
            move = plan.direction
    if dash and move == 0:
        dash = 0
        invalid = True
    return AgentIntent(move=move, dash=bool(dash), macro=mac), invalid
