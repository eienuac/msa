# Boss AI Agent — project notes for Claude

Talk to the user in Turkish. The user plays the game; you do not have the game.

## What this is
RL (MaskablePPO) agent for MMORPG boss fights. Vision turns the screen into a
`GameState`; a 2D simulator produces the same `GameState`; the policy is
trained in the simulator and runs on the real screen through the same code.
See README.md for the module map and commands.

## Hard rules
- No anti-cheat bypass, input hiding or "humanised" evasion of any kind. The
  input backend is plain SendInput. Default runtime mode is shadow (no keys).
- The policy never presses R directly: it asks for a macro state and
  `MacroStateManager` decides. Never press R while moving, never in UNKNOWN.
- Observation layout changes the schema hash; saved models refuse other
  hashes. Bump `SCHEMA_VERSION` when changing features on purpose.

## Game facts (Kor Bekçi, from the user and the recordings)
- Melee character. R = in-game auto-attack ("Asistan"); ON = yellow animated frame,
  brighter icon and a small digit on the R icon; a white frame is the moment R is
  pressed. Walking does NOT reliably turn the macro off (user correction; recordings
  33/44 show it staying on through many moves) — always read the indicator.
  R next to the boss re-selects it.
- Calibration clip: steady walk ~10 ground units/s (W slightly slower, ~9), dash adds
  ~5-6 units, a player-centred circle has radius ~3.8 units → ~0.4 s to walk out from
  the centre. Camera does not rotate by itself; A/D strafe.
- Q = dash in the movement direction, 3 charges, 1.0 s lock between dashes,
  one charge back every ~3.3 s, ~8 s lockout after the third. No invulnerability.
- Warnings: circles (on the player, around the boss, 3-6 at once), cones from
  the boss, long lines. Outer shape appears at once, inner fill grows to the rim
  in 1.1-1.5 s, then impact. ~7 s between attacks; 12-25% HP per hit.
- "Fire burst": boss raises its arms ~1 s, bursts around it summon adds and push
  the player back. No damage. Adds can take the target.
- The user wants early dodges, not last-second ones (reward penalises exits
  with < 0.5 s left and time spent inside warnings).
- Bosses have profiles in `configs/bosses/<id>.yaml` (telegraph style, colours,
  OCR name fragment, warning duration). Pass `--boss <id>` to the tools.
- Örümcek Ana (spider, orange sand, 3 short 1280x720 clips that look like screen
  captures of a video player): warnings are pale green-yellow translucent discs,
  only the rim stands out → `telegraph_style: rim`. Seen: a ring that follows the
  player for ~2 s, then the player is wrapped in a web cocoon and small spiders
  appear; a huge ring around the boss, ~1.0-1.2 s later a yellow-green poison wave
  over the whole area. Mechanics NOT confirmed by the user yet (dodgeable? damage?
  safe spot?) → no simulator profile or training for it yet. HUD reading on these
  clips is unreliable (different UI scale + player bar).
- Spider, user's own 1080p clips (33/44/11/22): more attack types than rims — a
  filled pale-green disc at the boss (11, ~9.5 s), green light beams/lines in the
  add phase (33), a poison wave (11, ~13 s), banners "YAVRULAR YAĞIYOR", "İPEK
  PORTALI KIRILDI", "DUVARLAR YENİDEN KIRILIYOR", a red-orange variant (44).
  The orange/white fire ring around the character is the user's own skill
  (confirmed by the user), not a warning; the rim rule rejects it (V>235 / S<115). Camera zoom in these clips differs from the Kor Bekçi clips
  (character higher on screen) → the ground projection's foot point is off until
  recalibrated with a clip at the same camera setting.
- HUD at 1920x1080: boss text "Sv. 22 Kor Bekçi | ▲ 5 m" (distance via OCR),
  player/boss HP bars, Q icon digits. Arrow direction is not read yet.

## Environment notes
- Cloud container; it can restart and kill background jobs. Training writes a
  checkpoint every 100k steps; resume with `--resume models/<run>` and set
  `--level-start` to the level already reached.
- GitHub: eienuac/msa (public). Commit as `Claude <noreply@anthropic.com>`.
- Recordings uploaded by the user live under /root/.claude/uploads/ and are
  not part of the repo. Do not commit videos or models/ (see .gitignore);
  released models go to `pretrained/`.

## Checks before committing
`pytest` must pass (about 80 tests, ~10 s).
