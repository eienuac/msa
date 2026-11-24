"""Perception tests on frames from the user's Kor Bekçi recordings.

Ground truth comes from the HP numbers printed in the HUD and from manual
inspection of the frames (time stamps refer to the first recording).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from bossai.core.game_state import HazardShape
from bossai.perception.calibration import GroundProjector, fit_circle
from bossai.perception.hazard_detection import HazardDetector, ScreenHazard, fill_from_area
from bossai.perception.tracking import HazardTracker
from bossai.perception.ui_reader import HudReader
from bossai.perception.vision import VisionPipeline

DATA = Path(__file__).parent / "data"


def frame(name: str) -> np.ndarray:
    img = cv2.imread(str(DATA / name))
    assert img is not None, name
    return img


@pytest.fixture(scope="module")
def hud():
    return HudReader()


@pytest.fixture(scope="module")
def det():
    return HazardDetector()


class TestHud:
    def test_macro_on_screenshot(self, hud):
        assert hud.read(frame("macro_on.jpg")).macro_p_on > 0.95

    @pytest.mark.parametrize("name", ["v1_00.30.jpg", "v1_01.45.jpg", "v1_10.40.jpg", "v1_28.80.jpg"])
    def test_macro_off_frames(self, hud, name):
        assert hud.read(frame(name)).macro_p_on < 0.05

    @pytest.mark.parametrize(
        "name,truth",
        [("v1_28.00.jpg", 9346 / 12740), ("v1_28.80.jpg", 9255 / 12740), ("macro_on.jpg", 11668 / 19600), ("v1_00.30.jpg", 1.0)],
    )
    def test_boss_hp_bar_matches_hud_numbers(self, hud, name, truth):
        assert hud.read(frame(name)).boss_hp == pytest.approx(truth, abs=0.02)

    def test_player_hp_bar_drops_after_hit(self, hud):
        full = hud.read(frame("v1_00.30.jpg")).player_hp
        hit = hud.read(frame("v1_02.60.jpg")).player_hp  # HUD shows 1181/1570, bar still animating
        assert full > 0.97
        assert 0.70 < hit < 0.85


class TestHazardDetection:
    @pytest.mark.parametrize("name", ["v1_00.30.jpg", "v1_17.40.jpg", "v1_28.00.jpg", "macro_on.jpg"])
    def test_no_false_positives(self, det, name):
        # includes fire braziers, boss lava glow, a reddish tree trunk and red nameplates
        assert det.detect(frame(name)) == []

    @pytest.mark.parametrize("name", ["v1_01.20.jpg", "v1_01.45.jpg", "v1_01.70.jpg"])
    def test_player_circle_found_and_projected(self, det, name):
        hs = det.detect(frame(name))
        assert len(hs) == 1
        h = hs[0]
        assert h.kind == "ellipse" and h.ellipse_iou > 0.9
        g = GroundProjector().to_ground(h.contour)
        c, r = fit_circle(g[~np.isnan(g).any(1)])
        assert np.linalg.norm(c) < 1.0  # cast on the player's position
        assert 3.0 < r < 5.0  # same skill, same size in every frame

    def test_cone_detected(self, det):
        hs = det.detect(frame("v1_10.40.jpg"))
        assert sum(h.area for h in hs) > 100_000


class TestGeometry:
    def test_projector_round_trip(self):
        p = GroundProjector()
        px = np.array([[958, 668], [400, 900], [1500, 600], [960, 400]], float)
        back = p.to_pixels(p.to_ground(px))
        np.testing.assert_allclose(back, px, atol=1e-6)
        np.testing.assert_allclose(p.to_ground(px[:1]), [[0.0, 0.0]], atol=1e-9)

    def test_forward_is_up_the_screen(self):
        g = GroundProjector().to_ground(np.array([[958, 500.0], [1200, 668.0]]))
        assert g[0, 1] > 1.0 and abs(g[0, 0]) < 0.1  # higher on screen = further forward
        assert g[1, 0] > 1.0  # right on screen = +x


def synthetic(fill_frac: float, n: int = 2000) -> ScreenHazard:
    rng = np.random.default_rng(int(fill_frac * 1000))
    vals = np.where(rng.random(n) < fill_frac, 40.0, 20.0) + rng.normal(0, 1, n)
    return ScreenHazard(
        contour=np.array([[900, 600], [1000, 600], [1000, 700], [900, 700]], np.float32),
        area=10000,
        bbox=(900, 600, 100, 100),
        kind="ellipse",
        mean_strength=float(np.median(vals)),
        strength_sample=vals.astype(np.float32),
    )


class TestTracking:
    def test_fill_from_area(self):
        assert fill_from_area(synthetic(0.0).strength_sample, 20.0) < 0.05
        assert fill_from_area(synthetic(0.25).strength_sample, 20.0) == pytest.approx(0.5, abs=0.05)
        assert fill_from_area(synthetic(1.0).strength_sample, 20.0) > 0.95

    def test_time_to_impact_from_fill_trend(self):
        tr = HazardTracker()
        impact = 1.2
        estimates = []
        for i in range(0, 11):
            t = i * 0.1
            front = t / impact  # radial front grows linearly -> filled area grows with its square
            tracks = tr.update(t, [synthetic(front**2)])
            if tracks and t >= 0.5:
                estimates.append(t + tr.time_to_impact(tracks[0], t))
        assert estimates and abs(np.mean(estimates) - impact) < 0.2

    def test_background_object_is_dropped(self):
        tr = HazardTracker()
        out = []
        for i in range(60):  # 6 s static red blob
            out = tr.update(i * 0.1, [synthetic(0.0)])
        assert out == []


def test_pipeline_outputs_game_state():
    pipe = VisionPipeline()
    for i, name in enumerate(["v1_01.20.jpg", "v1_01.45.jpg", "v1_01.70.jpg"]):
        out = pipe.process(frame(name), 1.2 + 0.25 * i)
    hz = out.state.hazards
    assert len(hz) == 1 and hz[0].shape is HazardShape.CIRCLE
    assert hz[0].contains(np.zeros((1, 2)))[0]  # the player stands in it
    assert out.state.boss.hp > 0.95


def icon_frame(name: str) -> np.ndarray:
    """Place a saved 70x75 Q-icon crop back at its HUD position in a blank 1080p frame."""
    canvas = np.zeros((1080, 1920, 3), np.uint8)
    canvas[950:1025, 594:664] = cv2.imread(str(DATA / f"{name}.png"))
    return canvas


class TestDash:
    @pytest.mark.parametrize(
        "name,ready,charges,empty",
        [
            ("q_ready3", True, 3, False),
            ("q_ready2", True, 2, False),
            ("q_lock1", False, 1, False),
            ("q_lock2", False, 2, False),
            ("q_empty", False, 0, True),
        ],
    )
    def test_icon_states(self, name, ready, charges, empty):
        from bossai.perception.dash_reader import DashIconReader

        r = DashIconReader().read(icon_frame(name))
        assert (r.ready, r.charges, r.empty) == (ready, charges, empty)

    def test_unlock_flash_counts_as_ready(self):
        from bossai.perception.dash_reader import DashIconReader

        assert DashIconReader().read(icon_frame("q_flash")).ready

    def test_filter_needs_two_empty_readings(self):
        from bossai.perception.dash_reader import DashReading, DashStateFilter

        f = DashStateFilter()
        f.update(DashReading(True, 3, False, False, 0.3, {}), 0.0)
        empty = DashReading(False, 0, True, True, 0.0, {})
        assert f.update(empty, 0.1)[0] == 3  # one "empty" frame (press border) is ignored
        charges, can, lock = f.update(empty, 0.2)
        assert charges == 0 and not can and 7.0 < lock <= 8.0


class TestBossText:
    @pytest.mark.parametrize(
        "text,dist,is_boss,behind",
        [
            ("Sv.22Kor Bekcil6m arkanda*", 6, True, True),
            ("Sv 22 Kor Bekci 12ml*", 12, True, False),
            ("(Sv.22 Kor Beki 42m1*>", 2, True, False),
            ("Sv.22 Yosumlu Kaya Orimcegi 1m*", 1, False, False),
            ("", None, False, False),
        ],
    )
    def test_parse(self, text, dist, is_boss, behind):
        from bossai.perception.boss_reader import parse_target_text

        info = parse_target_text(text, ("kor be",))
        assert (info.distance_m, info.is_boss, info.behind) == (dist, is_boss, behind)

    def test_ocr_reads_distance_from_frame(self):
        from bossai.perception.boss_reader import BossInfoReader

        r = BossInfoReader()
        if not r.available:
            pytest.skip("rapidocr_onnxruntime not installed")
        info = r.read(frame("v1_28.00.jpg"), 0.0)  # HUD: "Kor Bekçi | 2 m"
        assert info.is_boss and info.distance_m == 2


def test_background_ocr_does_not_block():
    import time as _time

    from bossai.perception.boss_reader import BossInfoReader

    r = BossInfoReader(background=True)
    if not r.available:
        pytest.skip("rapidocr_onnxruntime not installed")
    img = frame("v1_28.00.jpg")
    t0 = _time.perf_counter()
    first = r.read(img, 0.0)  # returns immediately (nothing read yet)
    assert _time.perf_counter() - t0 < 0.05 and first is None
    for i in range(100):
        _time.sleep(0.05)
        info = r.read(img, 0.05 * (i + 1))
        if info is not None:
            break
    assert info is not None and info.distance_m == 2


class TestCircleSplitting:
    def test_ransac_recovers_overlapping_circles(self):
        from bossai.perception.calibration import fit_circles_ransac

        res = 20
        img = np.zeros((400, 400), np.uint8)
        truth = [((5, 5), 3.0), ((9.5, 5.5), 2.5), ((7, 9), 2.2)]
        for (cx, cy), r in truth:
            cv2.circle(img, (int(cx * res), int(cy * res)), int(r * res), 255, -1)
        cs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        found = fit_circles_ransac(cs[0].reshape(-1, 2) / res, tol=0.25)
        assert len(found) == 3
        for (cx, cy), r in truth:
            assert min(np.hypot(c[0] - cx, c[1] - cy) + abs(rr - r) for c, rr in found) < 0.3

    def test_duplicate_rim_edges_merge(self):
        from bossai.perception.calibration import merge_duplicate_circles

        out = merge_duplicate_circles([(np.array([0.0, 0.0]), 4.0), (np.array([0.2, 0.1]), 3.4), (np.array([8.0, 0.0]), 2.0)])
        assert len(out) == 2

    def test_touching_rain_circles_are_split(self):
        from bossai.perception.vision import VisionConfig, VisionPipeline

        pipe = VisionPipeline(VisionConfig(read_boss_text=False))
        pipe.process(frame("v4_12.40.jpg"), 12.40)
        out = pipe.process(frame("v4_12.43.jpg"), 12.43)
        circles = [h for h in out.state.hazards if h.shape is HazardShape.CIRCLE]
        assert len(circles) >= 3  # several rain circles, not one merged blob
        assert all(h.radius < 5.0 for h in circles)


class TestSpiderRim:
    """Örümcek Ana: pale ring telegraphs on orange sand (1280x720 recordings)."""

    def test_profiles_load(self):
        from bossai.perception.profile import list_profiles, load_profile
        from bossai.perception.rim_detection import RimDetector
        from bossai.perception.vision import VisionConfig, VisionPipeline

        assert {"kor_bekci", "orumcek_ana"} <= set(list_profiles())
        assert load_profile("kor_bekci").telegraph_style == "filled"
        p = VisionPipeline(VisionConfig.for_boss("orumcek_ana", read_boss_text=False))
        assert isinstance(p.detector, RimDetector)
        assert p.cfg.boss_keywords == ("orumcek a",)

    def test_ring_around_player_found(self):
        from bossai.perception.calibration import GroundProjector, fit_circle
        from bossai.perception.rim_detection import RimDetector

        img = frame("spider2_13.00.jpg")
        hs = RimDetector().detect(img)
        assert hs, "the green ring under the player was not found"
        proj = GroundProjector()
        from bossai.perception.calibration import CameraModel

        proj = GroundProjector(CameraModel().scaled(img.shape[1], img.shape[0]))
        best = []
        for h in hs:
            c, r = fit_circle(proj.to_ground(h.contour))
            best.append((np.linalg.norm(c), r))
        dist, r = min(best)
        assert dist < 1.5 and 2.5 < r < 4.5

    @pytest.mark.parametrize("name", ["spider2_05.00.jpg", "spider1_08.00.jpg"])
    def test_no_ring_without_warning(self, name):
        from bossai.perception.rim_detection import RimDetector

        assert RimDetector().detect(frame(name)) == []

    def test_red_zone_rule_misses_the_green_ring(self):
        # documents why the spider needs its own profile: the Kor Bekçi rule does not see its warnings
        from bossai.perception.hazard_detection import HazardDetector

        assert HazardDetector().detect(frame("spider2_13.00.jpg")) == []


def test_pipeline_runs_on_720p_frames():
    from bossai.perception.vision import VisionConfig, VisionPipeline

    pipe = VisionPipeline(VisionConfig.for_boss("orumcek_ana", read_boss_text=False))
    img = frame("spider2_13.00.jpg")
    assert img.shape[:2] == (720, 1280)
    out = pipe.process(img, 0.0)
    assert out.state.player.dash_charges in (0, 1, 2, 3)
    small = cv2.resize(frame("v1_01.45.jpg"), (1280, 720))
    VisionPipeline(VisionConfig(read_boss_text=False)).process(small, 0.0)


@pytest.mark.parametrize("name", ["fire_ring_r22_16.5.jpg", "fire_ring_r33_07.0.jpg"])
def test_player_fire_ring_is_not_a_warning(name):
    # the orange/white ring around the character is the user's own skill (confirmed by the user)
    from bossai.perception.rim_detection import RimDetector

    assert RimDetector().detect(frame(name)) == []
