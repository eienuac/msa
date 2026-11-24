import math

import numpy as np
import pytest

from bossai.core.game_state import (
    BossState,
    GameState,
    Hazard,
    HazardShape,
    PlayerState,
    direction_angle_diff,
    direction_from_keys,
)
from bossai.core.geometry import CH_BOSS, CH_OCCUPIED, CH_URGENCY, DangerGrid, GridSpec
from bossai.core.safety import EscapeParams, SafetyPlanner, ray_directions, ray_features


def make_state(hazards, player=(0.0, 0.0), boss=(0.0, 6.0), arena=None, charges=3):
    return GameState(
        t=0.0,
        player=PlayerState(pos=np.array(player, float), hp=100, max_hp=100, dash_charges=charges),
        boss=BossState(pos=np.array(boss, float), hp=100, max_hp=100),
        hazards=hazards,
        arena_center=np.zeros(2) if arena else None,
        arena_radius=arena,
    )


def circle(c, r, tti, id=0):
    return Hazard(id=id, shape=HazardShape.CIRCLE, center=np.array(c, float), radius=r, time_to_impact=tti)


class TestShapes:
    def test_circle_contains_and_sdf(self):
        h = circle((1, 1), 2, 1.0)
        pts = np.array([[1, 1], [3.1, 1], [1, 2.9]])
        assert h.contains(pts).tolist() == [True, False, True]
        np.testing.assert_allclose(h.signed_distance(pts), [-2, 0.1, -0.1], atol=1e-9)
        assert h.contains(np.array([[3.4, 1]]), margin=0.5)[0]

    def test_sector_contains(self):
        # 90-degree cone (half angle 45) pointing +y from the origin, radius 5
        h = Hazard(0, HazardShape.SECTOR, np.zeros(2), 5.0, 1.0, direction=math.pi / 2, half_angle=math.pi / 4)
        pts = np.array([[0, 3], [2, 2.5], [3, 1], [0, -1], [0, 5.5]])
        assert h.contains(pts).tolist() == [True, True, False, False, False]
        sd = h.signed_distance(pts)
        assert sd[0] < 0 and sd[2] > 0 and sd[3] > 0

    def test_rect_contains(self):
        h = Hazard(0, HazardShape.RECT, np.zeros(2), 10.0, 1.0, direction=0.0, width=2.0)
        pts = np.array([[5, 0.9], [5, 1.2], [-0.5, 0], [10.5, 0]])
        assert h.contains(pts).tolist() == [True, False, False, False]
        assert h.contains(np.array([[5, 1.2]]), margin=0.3)[0]

    def test_direction_helpers(self):
        assert direction_from_keys({"w"}) == 1
        assert direction_from_keys({"w", "d"}) == 2
        assert direction_from_keys({"s", "a"}) == 6
        assert direction_from_keys({"w", "s"}) == 0
        assert direction_from_keys(set()) == 0
        assert direction_angle_diff(1, 5) == 4
        assert direction_angle_diff(1, 8) == 1
        assert direction_angle_diff(0, 3) == 0


class TestGrid:
    def test_hazard_and_boss_rendered_at_right_place(self):
        spec = GridSpec(size=32, extent=12.0, t_max=2.0)
        st = make_state([circle((0, 6), 2, 0.5)], boss=(6, 0))
        g = DangerGrid(spec).render(st)
        assert g.shape == (5, 32, 32)
        # a hazard 6 m ahead appears in the upper half, centre column
        row = int((12 - 6) / spec.cell)
        col = 16
        assert g[CH_OCCUPIED, row, col] == 1.0
        assert g[CH_URGENCY, row, col] == pytest.approx(0.75)
        assert g[CH_OCCUPIED, 31 - row, col] == 0.0
        # boss 6 m to the right
        assert g[CH_BOSS, 16, int((12 + 6) / spec.cell)] == 1.0


class TestRays:
    p = EscapeParams(walk_speed=5.0, latency=0.0, hitbox=0.0, margin=0.0, step=0.25)

    def test_safe_when_no_hazard(self):
        rf = ray_features(make_state([]), self.p)
        assert not rf.in_danger
        assert np.all(rf.walk_slack == 2.0)

    def test_walk_slack_matches_analytic(self):
        # player at the centre of a 4 m circle impacting in 1.0 s: walking out takes 0.8 s
        rf = ray_features(make_state([circle((0, 0), 4, 1.0)]), self.p)
        assert rf.in_danger
        assert rf.walk_slack.min() == pytest.approx(0.2 - 0.05, abs=0.06)
        # dash covers 5 m in 0.25 s
        assert rf.dash_slack.min() > 0.6

    def test_overlapping_circles_block_some_directions(self):
        hz = [circle((0, 0), 3, 0.8, 0), circle((0, 5), 3, 0.8, 1)]
        rf = ray_features(make_state(hz), self.p, ray_directions(8))
        # going forward runs into the second circle -> worse than going backward
        assert rf.walk_slack[0] < rf.walk_slack[4]

    def test_wall_blocks_escape(self):
        rf = ray_features(make_state([circle((0, 8), 4, 1.0)], player=(0, 8), arena=9.0), self.p, ray_directions(8))
        assert rf.walk_slack[0] == -2.0  # forward hits the wall before leaving the circle


class TestPlanner:
    def test_stays_when_safe(self):
        plan = SafetyPlanner().plan(make_state([circle((10, 10), 2, 1.0)]))
        assert plan.direction == 0

    def test_walks_out_shortest_way(self):
        st = make_state([circle((0, 2.5), 3, 1.5)], boss=(0, 20))
        plan = SafetyPlanner(EscapeParams(latency=0.0)).plan(st)
        assert plan.direction == 5 and not plan.dash  # backward is shortest

    def test_dashes_when_walking_is_too_slow(self):
        st = make_state([circle((0, 0), 5, 0.6)])
        plan = SafetyPlanner(EscapeParams(latency=0.0, walk_speed=5.0)).plan(st)
        assert plan.dash and plan.slack > 0

    def test_no_dash_without_charges(self):
        st = make_state([circle((0, 0), 5, 0.6)], charges=0)
        assert not SafetyPlanner().plan(st).dash
