import random

from bossai.control.action_controller import ActionController, AgentIntent, ControllerConfig
from bossai.control.input_backend import RecordingBackend
from bossai.control.macro_state import MacroConfig, MacroEvidence, MacroStateManager
from bossai.core.game_state import MacroState

DT = 0.1
ON = MacroEvidence(indicator_on=1.0)
OFF = MacroEvidence(indicator_on=0.0)
NONE = MacroEvidence()


def run(mgr, evidences, t0=0.0):
    """Feed a list of evidence objects at DT spacing; return press times."""
    presses = []
    t = t0
    for ev in evidences:
        if mgr.update(t, ev):
            presses.append(round(t, 2))
        t += DT
    return presses, t


class TestMacroStateManager:
    def test_unknown_never_presses_without_probe_conditions(self):
        m = MacroStateManager()
        m.request(True)
        presses, _ = run(m, [NONE] * 100)
        assert presses == []
        assert m.state == MacroState.UNKNOWN

    def test_confirms_off_then_single_press_to_turn_on(self):
        m = MacroStateManager()
        presses, t = run(m, [OFF] * 5)
        assert m.state == MacroState.OFF and presses == []
        m.request(True)
        presses, t = run(m, [OFF] * 3 + [ON] * 5, t)
        assert len(presses) == 1
        assert m.state == MacroState.ON

    def test_no_press_when_already_in_wanted_state(self):
        m = MacroStateManager()
        run(m, [ON] * 5)
        m.request(True)
        presses, _ = run(m, [ON] * 50, 0.5)
        assert presses == []

    def test_lost_press_is_retried_then_locks(self):
        cfg = MacroConfig(max_retries=2, lockout_after_failures=10.0)
        m = MacroStateManager(cfg)
        _, t = run(m, [OFF] * 5)
        m.request(True)
        presses, t = run(m, [OFF] * 60, t)  # the game never reacts
        assert len(presses) == cfg.max_retries + 1  # then locked: no further presses
        assert m.state == MacroState.OFF  # the indicator still clearly says OFF
        # gaps respect the confirmation window
        assert all(b - a >= cfg.confirm_timeout for a, b in zip(presses, presses[1:]))

    def test_probe_resolves_unknown_when_safe(self):
        m = MacroStateManager(MacroConfig(probe_after=1.0))
        m.request(True)
        safe = MacroEvidence(safe_to_probe=True)
        presses, t = run(m, [safe] * 15)
        assert len(presses) == 1  # exactly one probe press
        assert m.state == MacroState.UNKNOWN
        # after the probe the indicator shows ON
        presses2, _ = run(m, [MacroEvidence(indicator_on=1.0, safe_to_probe=True)] * 5, t)
        assert presses2 == [] and m.state == MacroState.ON

    def test_no_probe_while_in_danger(self):
        m = MacroStateManager(MacroConfig(probe_after=0.5))
        m.request(True)
        presses, _ = run(m, [MacroEvidence(safe_to_probe=False)] * 50)
        assert presses == []

    def test_invalidation_goes_unknown(self):
        m = MacroStateManager()
        run(m, [ON] * 5)
        m.update(1.0, MacroEvidence(invalidate=True))
        assert m.state == MacroState.UNKNOWN

    def test_cannot_turn_on_without_target(self):
        m = MacroStateManager()
        _, t = run(m, [OFF] * 5)
        m.request(True)
        presses, _ = run(m, [MacroEvidence(indicator_on=0.0, can_turn_on=False)] * 20, t)
        assert presses == []

    def test_damage_evidence_alone_can_confirm_on(self):
        m = MacroStateManager()
        ev = MacroEvidence(damage_dealt=True, attack_expected=True)
        run(m, [ev] * 10)
        assert m.state == MacroState.ON

    def test_random_sequences_respect_rate_limits(self):
        cfg = MacroConfig()
        for seed in range(30):
            rng = random.Random(seed)
            m = MacroStateManager(cfg)
            presses = []
            t = 0.0
            for _ in range(1500):
                if rng.random() < 0.05:
                    m.request(rng.choice([True, False, None]))
                ev = MacroEvidence(
                    indicator_on=rng.choice([None, 0.0, 1.0, 0.5]),
                    damage_dealt=rng.choice([None, True, False]),
                    attack_expected=rng.random() < 0.5,
                    invalidate=rng.random() < 0.01,
                    safe_to_probe=rng.random() < 0.5,
                )
                if m.update(t, ev):
                    presses.append(t)
                    # after every press the manager waits for confirmation
                    assert m.state in (MacroState.PENDING_ON, MacroState.PENDING_OFF) or m._probing
                t += DT
            gaps = [b - a for a, b in zip(presses, presses[1:])]
            assert all(g >= cfg.min_press_interval - 1e-9 for g in gaps)
            for i, p in enumerate(presses):
                window = [q for q in presses[i:] if q - p < 60.0]
                assert len(window) <= cfg.max_presses_per_min


def make_controller(hold=0.05, lead=0.03):
    be = RecordingBackend()
    mgr = MacroStateManager()
    return be, mgr, ActionController(be, mgr, ControllerConfig(tap_hold=hold, dash_dir_lead=lead))


class TestActionController:
    def test_same_direction_sends_nothing(self):
        be, _, c = make_controller()
        c.apply(0.0, AgentIntent(move=1), NONE)
        be.clear_events()
        for i in range(10):
            c.apply(0.1 * (i + 1), AgentIntent(move=1), NONE)
        assert be.events == []

    def test_direction_change_sends_only_difference(self):
        be, _, c = make_controller()
        c.apply(0.0, AgentIntent(move=1), NONE)  # W
        be.clear_events()
        rep = c.apply(0.1, AgentIntent(move=2), NONE)  # W + D
        assert be.events == [("down", "d")]
        assert rep.direction_changed
        be.clear_events()
        c.apply(0.2, AgentIntent(move=0), NONE)
        assert sorted(be.events) == [("up", "d"), ("up", "w")]

    def test_r_is_tapped_once_and_released(self):
        be, mgr, c = make_controller()
        for i in range(5):
            c.apply(0.1 * i, AgentIntent(), OFF)
        rep = c.apply(0.5, AgentIntent(macro=1), OFF)
        assert rep.r_pressed
        assert "r" in be.pressed
        c.service(0.56)
        assert "r" not in be.pressed
        # further intents while pending do not press again
        for i in range(5):
            c.apply(0.6 + 0.1 * i, AgentIntent(macro=1), OFF)
        assert be.taps("r") == 1

    def test_dash_waits_for_direction_key(self):
        be, _, c = make_controller()
        rep = c.apply(0.0, AgentIntent(move=3, dash=True), NONE)
        assert rep.dash_sent
        assert be.events == [("down", "d")]  # Q is scheduled after the lead time
        c.service(0.031)
        assert ("down", "q") in be.events
        c.service(0.09)
        assert "q" not in be.pressed

    def test_dash_with_held_direction_is_immediate(self):
        be, _, c = make_controller()
        c.apply(0.0, AgentIntent(move=3), NONE)
        c.apply(0.1, AgentIntent(move=3, dash=True), NONE)
        assert ("down", "q") in be.events

    def test_release_all_and_disable(self):
        be, _, c = make_controller()
        c.apply(0.0, AgentIntent(move=2, dash=True), NONE)
        c.service(0.04)
        c.release_all()
        assert be.pressed == set()
        c.enabled = False
        c.apply(0.2, AgentIntent(move=1), NONE)
        assert be.pressed == set()
