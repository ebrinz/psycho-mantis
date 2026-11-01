---
name: mantis
description: Run Claude's perception-action loop on the psychomanteum mirror - observe its prediction error, adjust its parameters, keep what improves prediction skill. Use when asked to run, drive, tune or "close the loop" on the mirror.
---

# The mantis loop

The psychomanteum mirror (`psychomanteum.py`, running in the Ghostty pane next to you) shows its
prediction of the camera frame a few steps ahead. You are the part of the system that perceives
how well it predicts and acts on how it predicts. Your actions change what it sees next.

## Your senses and hands

- `python3 mantis.py observe 5` - watch the mirror for 5 seconds. Reports motion, prediction error,
  **skill**, error by region (3x3, as the person sees it in the mirror) and current params.
- `python3 mantis.py set key=value ...` - act. Prints the values actually applied (out-of-range values are clamped).
- `python3 mantis.py params` - current parameters.

If observe says the mirror isn't running, ask the user to start it (`uv run psychomanteum.py` in the other pane).

## What you can change

| param | range | effect |
|---|---|---|
| `horizon` | 1-15 | steps ahead to predict. Further ahead is harder; it is the challenge, so don't shrink it just to score better unless told to. |
| `damping` | 0-1.5 | flow baseline only: scales the extrapolated motion. Constant-velocity extrapolation overshoots when people stop or turn, so values below 1 often help, more so at long horizons. |
| `flow_smoothing` | 0-0.95 | flow baseline only: averages the motion estimate over time. Reduces jitter, but lags when motion changes. |
| `flow_window` | 5-41 | flow baseline only: size of the optical-flow window. Larger = smoother, blurrier motion estimate. |
| `eye_max_delay`, `eye_jitter`, `eye_noise`, `eye_width` | | the two eyes' input conditions (wandering delay, noise, resolution). These define the challenge - don't change them to chase skill unless asked. |
| `fusion_rate` | 0-1 | learning rate of the fused two-eye model (the main predictor). 0 freezes it. E13 found higher was better up to 1.0 offline. |
| `pc_rate`, `pc_sparsity` | 0-1, 0-3 | each eye's V1-like predictive-coding layer: learning rate and sparsity. They change the learned features (`v1_err`, `v1_active` in stats) and feed the fused model's inputs. |
| `ghost_gain`, `ghost_floor` | | visual only - how prediction error is drawn as ghosts. Do not use these to chase skill. |
| `dither`, `view` | | display only (`view`: prediction / camera / error). |
| `caption` | text | a line under the mirror - your voice to the person in front of it. |

## Objective

Maximise **model skill** = 1 - fused-model err / err of copying the average of both eyes: how much
better the fused two-eye model predicts the true frame `horizon` steps ahead than just averaging
what the eyes see. Offline it reached +0.02 to +0.05 (EXPERIMENTS.md E13), so expect small numbers.
**Flow skill** is the optical-flow baseline on the undelayed full-res frame - not comparable, and
only its params (`damping`, `flow_*`) affect it. Judge only while the person is moving (observe
reports flow skill "n/a" when motion is too low), and on a replayed clip when comparing settings
(E5: live 5 s windows are too noisy).

## Method

1. Baseline: `observe 5`. If there's too little motion, set a caption asking the person to move
   (e.g. `caption="move slowly side to side"`) and observe again.
2. Pick **one** change with a reason (e.g. "overshoot at the edges of motion -> damping 1.0 to 0.8").
3. `set` it, then `observe 5` (twice if the motion level differs a lot from the baseline -
   skill comparisons are only fair at similar motion).
4. Keep it if skill improved by more than ~0.03, otherwise set it back.
5. Repeat. Stop after the number of rounds the user asked for (default 5), or when asked.

Narrate each round in one short line in the chat: what you saw, what you changed, why, and the
result. Use the caption sparingly to tell the person in the mirror what you're doing (short,
under 60 characters). Finish with a summary of the best settings found and the skill gain over
the baseline, and clear the caption (`caption=""`).
