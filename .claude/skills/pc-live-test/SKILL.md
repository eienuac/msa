---
name: pc-live-test
description: Run the agent against the real game on the user's Windows PC (setup check, live shadow mode, later control mode). Use when the user wants to test on their own computer.
---

# Live test on the user's PC

The game only runs on the user's Windows PC. Either a Claude Code session runs
on that PC (Claude Desktop app / Remote Control) or the user runs the commands
and sends back the output and screenshots.

1. Setup (once): Python 3.11+, `git clone https://github.com/eienuac/msa`,
   `pip install -e .[windows,ocr]`.
2. Game: 1920x1080, borderless/windowed fullscreen, default UI scale, camera not
   rotated by mouse during the test, zoom fixed.
3. Check: `python tools/check_setup.py --seconds 20` while standing near the boss.
   It prints what vision reads (macro, HP, dash, boss distance, zones) and saves
   frames to `check_out/`. Compare with what the user sees; fix layout boxes in
   `bossai/perception/ui_reader.py` / `dash_reader.py` if needed.
4. Shadow mode (no keys sent): `python -m bossai.runtime.live --model pretrained/kor_bekci_v3 --source dxcam --show --out shadow.mp4`
   The user plays; the overlay shows what the agent would do. Collect cases where
   it is wrong → turn them into simulator scenarios / perception fixes.
5. Control mode only after shadow mode looks right, only where automation is
   allowed, short sessions, `--window-title` set, F12 = stop + release keys.
   No anti-cheat bypass of any kind.
