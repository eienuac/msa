---
name: new-boss-recording
description: Analyse a newly uploaded gameplay video (a new boss, a new attack, or a calibration clip) and turn it into perception settings and simulator parameters. Use when the user uploads an .mp4 of a fight or asks how a skill behaves.
---

# Analysing a new recording

1. Inspect: `ffprobe` for resolution/fps/duration. The HUD layout assumes 1920x1080.
2. Overview: contact sheet at 1 fps (`ffmpeg -vf "fps=1,scale=400:-1,tile=6x6"`), look at it.
3. Run the vision pipeline and read the timeline:
   `python tools/analyze_video.py VIDEO --out analysis` → annotated mp4 + CSV
   (hazard count, time to impact, macro P(ON), HP, dash).
4. For each attack, cut 0.1-0.2 s frame strips around it and record:
   shape (circle / cone / line / other), where it is cast (player, boss, random),
   size in metres (ground projection), warning duration (appear → impact),
   damage (HP numbers before/after), number of simultaneous zones.
5. Telegraph colour: measure hue/saturation of zone pixels vs. false positives
   (fire, lava, foliage) and set `TelegraphColor.hue_window` / `max_sat` for that
   boss. Keep the hue filter tight; it is what removes false positives.
6. Put the numbers into `configs/<boss>_sim.yaml` (`skill_overrides`, intervals,
   dash/macro timings) and, if the attack type is new, add a `SkillSpec` and its
   spawn code in `bossai/sim/world.py`.
7. Add 2-4 representative frames to `tests/data/` with assertions in
   `tests/test_perception.py` (no false positives, zone found, ground radius).
8. Ask the user only what the video cannot show (damage or not, knockback,
   whether walking cancels something). Write confirmed facts into CLAUDE.md.
