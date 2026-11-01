# Psychomanteum demo script (~5 minutes)

Results page for the closing: https://claude.ai/artifact/Hb3nQyqGi9Gr4VhfRYtb7t

Evidence for every claim is in `EXPERIMENTS.md` (entry numbers in brackets).

## Before you start

- Ghostty window, three panes: **top** `uv run splash.py`, **left** Claude Code, **right**
  `uv run psychomanteum.py` (mirror). Click the mirror pane before pressing keys.
- Mirror defaults: `view=prediction`, `fusion_eyes=both`, dither on, features on. Reset with
  `python3 mantis.py set view=prediction fusion_eyes=both dither=true features=true caption='""'`.
- **No camera / bad venue?** `uv run psychomanteum.py --replay recordings/clip_20261030_135948.mp4`
  plays the 16-minute recording in a loop (E12). Everything else works the same.
- Nothing needs the network except Claude in the left pane.

## 1. The question (30 s)

> Can a system that sees the world through two crude, noisy eyes, each lagging by a different,
> randomly changing amount, predict what the camera will see 100 ms from now?

Point at the **top row**: left eye red, right eye blue, 80x45 pixels each, each with its own
wandering delay of 0-4 frames plus noise (the status line shows `eyes L# R#`).

## 2. V1 builds itself (60 s)

Point at the **middle row**: each eye's 64 learned features.

> Nobody designed these. Each eye runs a sparse predictive-coding layer (Olshausen & Field,
> Rao & Ballard): predictions go down, errors come up, and features change by a local
> Hebbian-style rule. With retina-like centre-surround filtering, they become edge and line
> detectors within about a minute of video. [E11]

Press `f` to hide and show the features.

## 3. The fused prediction (60 s)

Point at the **purple panel**: the fused model's prediction of the frame 100 ms ahead.

> Both eyes' V1 activity feeds one predictor that learns online with the delta rule. Where it
> got the future wrong, ghosts appear.

Move your hand: ghosts appear around the movement. Hold still: they fade.
Press `v` to cycle: **prediction** -> **flow** (full-res optical-flow baseline) -> **camera** ->
**error** (heatmap of the model's error), and back.

## 4. Two eyes beat one (60 s)

Press `e` to switch the prediction to the **left eye only**, then **right**, then back to **both**.
(All three models learn all the time; `e` only changes which one you see.)

> On the 16-minute recording, two eyes have 16% lower prediction error than one. [E16]

Live numbers: `python3 mantis.py observe 10` shows `one vs two eyes (err)`. Live 10 s windows
are noisy [E5], so quote the recorded result, not a single live reading.

## 5. Precision weighting and the stabilized map (60 s)

> The raw learned model only beat simply averaging the eyes on 48% of frames - it overreacted
> during slight motion. So each patch weighs three predictions by their recent precision, the
> predictive-coding idea of precision-weighted prediction errors: the learned model, the eyes'
> average pushed along its own motion (the "stabilized map"), and the plain average. Result: 24%
> lower error than averaging the eyes, better on 90% of frames, and better at every level of
> motion. Motion compensation alone is useless when still - the weighting learns when to trust it.
> [E14, E15, E16]

The status line's `skill vs eyes` is this live score (> 0 = better than averaging the eyes).

## 6. Closing the loop with Claude (60 s)

In the Claude pane: `/mantis 3 rounds`, or ask Claude to set a caption.

> The mirror writes its prediction errors every second; Claude reads them, changes a setting,
> and watches what that does - perception, action, changed perception. It writes to the person
> in the mirror through the caption line.

## 7. Honest limits (15 s)

> One 16-minute clip of one person, one seed. The learned readout itself adds little (a linear
> readout tops out at ~5%); most of the gain comes from motion compensation and precision
> weighting. Next: higher stages of the diagram, and more people and scenes. [E13, E16]

## If something breaks

| Problem | Fix |
|---|---|
| Mirror pane black / frozen | `q`, then `uv run psychomanteum.py` again |
| Camera error | Replay mode (see top) |
| Keys do nothing | Click the mirror pane first |
| Features look like noise | They were reset; learned features live in `state/v1_features.npz` |
| `skill vs eyes` negative live | Check `fusion_eyes=both`; precision weighting needs ~1 s to settle after a restart |
