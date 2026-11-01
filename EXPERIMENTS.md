# Experiment log

Every measurement and empirical claim for the psychomanteum mirror, including negative and
inconclusive results. Newest entries at the bottom. See CLAUDE.md for how to add entries.

Skill = 1 - prediction MSE / copy-last-frame MSE (> 0 means the prediction beats repeating the
last frame). Errors in `mantis.py` / stats output are MSE x1000.

## Open hypotheses (untested)

- Damping < 1 helps at long horizons (claimed in `.claude/skills/mantis/SKILL.md`; E5 found no
  gain at horizon 3, long horizons not tested). For the eyes' motion compensation, see E17: offering
  shorter reaches and letting precision weighting choose helps from horizon 6 up (the full-res
  `flow` baseline's `damping` is still untested).
- Replaying a fixed recorded clip would make parameter comparisons reliable (proposed after E5).
- Motion compensation of the fused eye image ("stabilized map") would beat copy-avg by more than the ~5% linear ceiling (after E13). **Confirmed in E16** (+0.240).
- With V1 frozen, its codes would help the fused readout (E13 confound).
- The crisp NCA mirror's look (E21) resembles psychedelic visual effects (drifting, breathing,
  stretching surfaces) because it is purely local prediction with no object-level constraint,
  as in the REBUS account of psychedelics (relaxed high-level priors). Analogy only; untested.
- **(E29+, testing)** At 1 s, weighting the lineage NCA (E23) against copying the current frame per
  8x8 patch by recent accuracy removes the NCA's still-frame loss without losing its motion gain;
  blending motion fields instead of pixels keeps it crisp. Predictions on held-out frames: P1
  stillest quintile >= 0 (E28: -0.056); P2 overall >= +0.152 (E23); P3 motion-blend sharpness
  within 10% of the NCA's; P4 the 1 s accuracy lag costs < half of the oracle-weight gain. P1 and
  P2 both failing refutes it. Risk: accuracy is only known 30 frames later.
  **-> E30: partly.** P1 holds for soft blends, but a fixed global blend does as well as adaptive
  weights; the 1 s lag defeats the adaptive part (oracle hard pick +0.250 vs lagged +0.153).
  **-> E32/E34: supported in amended form.** Weights from signals known *now* (a gate) rather than
  from past accuracy: +0.174 and still +0.022 on the old recording, **+0.169 and still +0.046 in a
  new room with nothing refitted** (E34). Crisp motion-blend variants pass P1/P3 but not P2.
- **(E35, testing)** The gate is feature-limited (E32). Adding per-patch features for how error
  evolves (motion 6-12 frames back; each source's 133 and 200 ms errors; rigid's 66 ms error; the
  last known 1 s error per source) and local structure (edge energy, divergence of the predicted
  motion, spread of the 10 guesses' motion vectors) will raise it. Predictions, gate fitted on
  old-room half 1 only: >= +0.19 on old-room half 2 and >= +0.185 in the new room (now +0.174 /
  +0.169), still >= 0 in both, sharpness >= 0.85. Groups added one at a time.
  **-> E35: refuted.** 14 extra features add <= +0.005 to a linear gate. A 32-hidden MLP with
  cross-entropy on the *original* 7 features gives +0.186 / +0.179 (nonlinearity, not features);
  shipped as the live gate.

---

## E1 - Kitty graphics throughput in Ghostty (2026-10-29)

**Question:** Can Ghostty display a full camera mirror via the Kitty graphics protocol, and which
transmission mode / encoding should we use?

**Setup:** `kitty_bench.py` (synthetic camera-like frames, no webcam), run in a new Ghostty 1.3.1
window. DA1 sync after every frame, so fps = frames Ghostty accepted, not frames painted.
Raw results: `kitty_bench_results.json`.

**Result** (fps):

| Setup | 320x240 | 640x480 | 1280x960 |
|---|---|---|---|
| Clean, direct base64 | 173 | 43 | 10 |
| Clean, shared memory | 9,158 | 4,184 | 943 |
| Dithered, shared memory | 917 | 242 | 64 |
| Chunky dither, shared memory | 1,959 | 551 | 150 |
| Clean, shared memory + zlib | 238 | 61 | 16 |

- Ghostty supports all three modes: direct (d), temp file (t), shared memory (s).
- Dithered frames compress ~25x with zlib (640x480: 37 KB vs 705 KB clean).
- zlib only helps direct mode; with shared memory it just adds CPU (~64 ms/frame at 1280x960 clean).

**Conclusion:** use shared memory, raw RGB, no zlib. Terminal is not the bottleneck; our CPU is.

**Caveats:** measures ingest, not paint (paint is capped by display refresh). User confirmed by eye
that dithering looked crisp; flicker not reported.

## E2 - Camera and per-frame pipeline cost (2026-10-29)

**Setup:** MacBook webcam via OpenCV from Ghostty. Pipeline from `psychomanteum.py`: 640x360
frames, Farneback flow at half resolution, horizon 3. 90 frames, median times.

**Result:**
- Camera delivers 1920x1080 at 30.2 fps (16:9).
- predict 5.5 ms, compose dithered 3.6 ms, compose clean 4.4 ms -> ~10 ms/frame total.

**Conclusion:** comfortably inside the 33 ms budget at 30 fps.

## E3 - Camera noise floor in the error map (2026-10-29)

**Question:** Why did the ghost overlay fire across a still scene?

**Setup:** still scene (camera facing the ceiling), horizon 3, Gaussian-blurred (9x9) per-pixel
mean abs error, ~50 scored frames.

**Result:** error percentiles 50/90/99/99.9/max = 0.0044 / 0.0075 / 0.0142 / 0.0235 / 0.0893.
With `GHOST_FLOOR = 0.03`, ghost pixels on the still scene: 0.000% (was widespread speckle).

**Conclusion:** camera noise was being drawn as ghosts; floor of 0.03 removes it. Moving-person
error is roughly 0.1-0.3, so the floor shouldn't hide real misses (not yet verified by eye).

## E4 - First live skill readings (2026-10-29)

**Setup:** live mirror, person moving, `mantis.py observe`.

**Result:**
- horizon 3 (~100 ms), damping 1.0: motion 1.96, err 1.04, **skill +0.471** (4 s).
- horizon 15 (~500 ms), damping 0.8: motion 12.91, err 12.82, **skill +0.007** (3 s).

**Conclusion:** flow extrapolation clearly beats copy-last at 100 ms; at 500 ms it's no better.

**Caveats:** single short windows; E5 shows such windows are very noisy.

## E5 - Mantis loop, 5 rounds of live tuning (2026-10-29)

**Question:** Can Claude improve skill by adjusting damping / flow_smoothing / flow_window live?

**Setup:** `/mantis` skill, horizon 3, person moving freely, 5 s `observe` windows.
Defaults: damping 1.0, flow_smoothing 0, flow_window 15.

**Result** (skill, motion in brackets):

| Round | Change | With change | Default at similar motion |
|---|---|---|---|
| Baseline | - | | +0.074 (1.69) |
| 1 | damping 0.8 | +0.323 (5.45), +0.258 (6.26) | +0.344 (9.88) |
| 2 | flow_smoothing 0.5 | +0.036 (6.03) | +0.26 to +0.34 (6-10) |
| 3 | flow_window 25 | +0.034 (2.01) | +0.165 (3.01) |
| 4 | flow_window 9 | +0.406 (5.29), +0.037 (7.46) | +0.352 (9.75) |
| 5 | none, repeated 3x | +0.389 (6.65), +0.531 (2.24), +0.014 (0.88) | |

**Conclusion:**
- No change beat the defaults. flow_smoothing 0.5 was the only consistently bad change (lag).
- **Measurement noise dominates:** identical settings ranged +0.01 to +0.53 across 5 s windows.
  The skill file's "keep if > 0.03" threshold is ~10x below the noise, so single live 5 s
  comparisons can't detect parameter effects.

**Caveats:** free, uncontrolled movement; motion level varied 0.9-9.9 between windows.

**Next:** replay a fixed recorded clip for comparisons; longer, interleaved A/B windows; raise the
keep threshold to match measured noise.

## E6 - Splash screen render cost (2026-10-29)

**Setup:** `splash.py` frame builder (plasma + figlet title + scroller, truecolor half-blocks),
median of 20 frames, built offline (terminal paint not measured).

**Result:** 120x10 cells: 1.4 ms, 37 KB/frame. 200x14: 2.7 ms, 82 KB/frame. Panes under ~9 rows
or ~104 columns fall back from the big `ansi_shadow` title to `small` or plain text.

**Conclusion:** cheap enough to run at 30 fps beside the mirror without competing for CPU.

## E7 - CPU cost of splash and mirror while running together (2026-10-29)

**Question:** Does the splash take meaningful CPU away from ML work?

**Setup:** splash, mirror and Claude Code all running in one Ghostty window (10-core Mac).
`top` sampled 3 x 2 s per condition; then splash paused with SIGSTOP and re-sampled, then resumed.
Percentages are of one core.

**Result:**

| Process | Splash running | Splash paused |
|---|---|---|
| splash.py | 2.8% | 0% |
| Ghostty (drawing all panes) | 11.5% | 10.5% |
| psychomanteum.py (mirror) | 55.4% | 55.7% |

**Conclusion:** the splash costs ~4% of one core in total (~0.4% of the machine); its terminal
drawing adds ~1% to Ghostty. The mirror is the real consumer, at ~55% of one core.

**Caveats:** GPU and battery use not measured.

## E8 - Two-eye input and composite display cost (2026-10-30)

**Setup:** new `psychomanteum.py` (hemifield eyes with wandering delay + noise, red/blue eyes over a
purple prediction, 640x720 composite), live webcam, 60 frames, eye_max_delay 8, eye_jitter 0.5.
Median ms per stage. Replay determinism: the same recorded clip run twice through `Replay` + `Eyes`
for 80 steps (longer than the clip, so it loops).

**Result:**
- eyes 5.0 ms, flow predict 7.1 ms, compose dithered 7.2 ms, compose clean 11.8 ms
  -> ~19 ms/frame dithered, ~24 ms clean (excluding scoring and terminal send).
- Replay: identical eye delays and eye images across both runs; loops correctly; 27 distinct
  (left, right) delay pairs over 80 steps.

**Conclusion:** still inside the 33 ms budget at 30 fps, but with much less headroom than E2
(~10 ms). The learned model will need to fit in roughly 10 ms or the mirror drops below 30 fps.
Replay gives identical input for fair comparisons (fixes the E5 noise problem for future tests).

## E9 - Full-frame binocular eyes replace hemifields (2026-10-30)

**Change:** design switched from left/right hemifields (E8) to two eyes that each see the full
scene at half resolution (320x180), each with its own wandering delay and noise. Composite is now
640x540 (eyes side by side over the full purple prediction). The step-4 ablation becomes
"one eye vs two fused eyes" instead of "cut the callosum at the midline".

**Setup:** live webcam, 60 frames, eye_max_delay 8, eye_jitter 0.5, median ms.

**Result:** eyes 2.5 ms (E8 hemifields: 5.0), compose dithered 5.5 ms (E8: 7.2).

**Conclusion:** binocular version is cheaper; per-frame total ~15 ms dithered, leaving ~15 ms
for the learned model at 30 fps (E8's estimate of ~10 ms no longer applies).

## E10 - Low-resolution eyes (2026-10-30)

**Change:** new `eye_width` param (default 80 -> 80x45 per eye, range 8-320). Eyes are tinted at
their own resolution and enlarged with nearest-neighbour for display (blocky retinal pixels).
The 4-tone dithering is display only; the model gets the eyes' full 0-255 values.

**Setup:** live webcam, 40 frames per width, median ms.

**Result:** eyes 0.3 ms at both 80x45 and 32x18 (E9 at 320x180: 2.5 ms); compose dithered
4.0-4.2 ms. By eye: at 80x45 a person and the doorframe are recognisable; at 32x18 only coarse
light/dark blocks remain.

**Conclusion:** per-frame total drops to ~12 ms, leaving ~20 ms for the learned model.
**Open issue for step 3:** skill compares against copying the last full-resolution frame, which
the model never sees; with low-res eyes the fair baseline is the latest eye image upscaled.

## E11 - Sparse predictive-coding layer per eye (V1-like) (2026-10-30)

**Question:** Does a per-eye sparse predictive-coding layer (Olshausen & Field / Rao & Ballard
style, local Hebbian-style update) learn edge-like features from the low-res eyes, online?

**Setup:** `SparseCoder` in `psychomanteum.py`: 8x8 patches of an 80x45 eye, 64 features
(complete dictionary), 30 iterations of ISTA inference, update phi += rate * residual a^T,
unit-norm features. Patches have their mean removed and are scaled to unit std per frame.
Sweeps ran on the 16 s clip `recordings/clip_20261030_135932.mp4` (405 frames, looped to 1800
steps), left eye only, seeded. "early" = mean of first 100 steps, "late" = mean of last 300.

**Result:**

| Preprocessing | sparsity | rate | rel. error early -> late | active features early -> late |
|---|---|---|---|---|
| none | 0.3 | 0.05 | 0.063 -> 0.042 | 44.9% -> 31.4% |
| none | 1.0 | 0.05 | 0.197 -> 0.095 | 12.0% -> 10.4% |
| none | 2.0 | 0.05 | 0.348 -> 0.207 | 3.6% -> 3.5% |
| none | 1.0 | 0.2 | 0.131 -> 0.089 | 9.7% -> 11.0% |
| centre-surround | 0.5 | 0.2 | 0.123 -> 0.101 | 25.7% -> 17.4% |
| centre-surround | 1.0 | 0.2 | 0.201 -> 0.143 | 9.6% -> 6.2% |

(Relative errors aren't comparable across preprocessing: centre-surround input is a different signal.)

- Sparsity 0.3 (first default): ~40% of features active, so any patch is reconstructable, error
  starts low and features stay noise-like. No real learning pressure.
- Sparsity 1.0, no whitening: oriented light/dark edges and bars appear, mixed with scene
  fragments (curves, blobs) and some still-random features.
- Centre-surround (difference of Gaussians, sigma 1.5, like retina/LGN) + sparsity 1.0:
  features become thin oriented lines, bars and curve segments at many orientations. Not textbook
  Gabors, but clearly edge/line detectors. Judged by eye from the feature grids.
- Cost: ~1 ms per eye. Full mirror frame (eyes + both layers + flow + compose): 14.0 ms median,
  live camera, before scoring and terminal send.

**Conclusion:** step-2 check passed: edge-like features emerge online within ~1 min of video.
Defaults set to centre-surround, sparsity 1.0, rate 0.2.

**Caveats:** one short clip, one seed, visual judgement of "edge-like"; the layer only
reconstructs the current eye image, it does not predict the next frame yet (step 3).

## E12 - Training clip recorded (2026-10-30)

**Data:** `recordings/clip_20261030_135948.mp4`: 29,284 frames, 640x360, ~16.3 min at 30 fps,
212 MB, mirrored (as seen in the mirror). One person moving freely in front of a laptop webcam,
varied motion (starts, stops, turns, speed changes). The 16 s `clip_20261030_135932.mp4` is a
false start, used only for the E11 sweeps.

**First live V1 reading after restart** (fresh camera, features loaded from state): v1_err
0.171 / 0.159 (left / right), 7.0% / 6.8% features active, 30.6 fps.

## E13 - Fused two-eye predictor, offline on the training clip (2026-10-30)

**Question:** Does a predictor fusing both eyes' V1 activity beat copying what the eyes see, and do
two eyes beat one?

**Setup:** `evaluate.py` on `recordings/clip_20261030_135948.mp4` (E12), horizon 3, eyes 80x45,
eye_max_delay 4, eye_jitter 0.2, eye_noise 8, V1 defaults (E11). Everything learns online from
scratch, seeded. `FusionPredictor`: per-patch shared linear readout from each eye's [codes, previous
codes, raw patch] + bias, initialised to the average of the eyes' raw patches, trained with
normalised LMS (delta rule). Target: true frame 3 steps ahead, no delay or noise, luminance at eye
resolution. Prequential: each prediction scored before learning from its target. MSE x1000.
Baselines: copy-avg (mean of both eyes' current images), copy one eye.

**Result 1 - learning rate** (first 5,400 frames, both eyes):

| rate | 0.01 | 0.05 | 0.2 | 0.5 | 1.0 |
|---|---|---|---|---|---|
| skill vs copy-avg | +0.003 | +0.008 | +0.016 | +0.022 | +0.027 |

**Result 2 - ceiling of a linear readout** (scratchpad `diag_ridge.py`: ridge fitted on frames
< 20,000, scored on the remaining 9,284 with fixed weights):

| inputs | skill vs copy-avg |
|---|---|
| raw eye patches only | +0.051 |
| raw + V1 codes | +0.028 |
| raw + V1 codes + previous codes | +0.026 |

**Result 3 - one eye vs two** (full clip, rate 1.0):

| model inputs | MSE | vs copy-avg | vs best single-eye copy |
|---|---|---|---|
| both eyes | 2.556 | +0.017 (last 3 min +0.052) | +0.159 |
| left eye only | 2.937 | -0.129 | +0.033 |
| right eye only | 2.899 | -0.115 | +0.046 |

copy-avg 2.601, best single-eye copy 3.037.

**Conclusions:**
- **Two eyes beat one by ~12%** (2.556 vs 2.899 for the better one-eye model). Most of this comes
  from combining the eyes at all: plain copy-avg already beats copying either eye by ~14%.
- **The learned part adds little:** +1.7% over copy-avg on the full clip, rising to +5.2% in the
  last 3 minutes, so it is still improving. The best possible linear readout reaches only ~+5%,
  so a higher learning rate can't fix this; a linear per-patch readout is the limit.
- **V1 codes did not help the linear readout** on held-out frames (+0.028 vs +0.051 raw only).
  Confound: V1 keeps learning, so its features drift between the fitting and scoring periods.
  Not tested with V1 frozen.

**Caveats:** one clip, one person, one seed; prequential numbers include early learning.

**Likely next gains (hypotheses):** motion compensation of the fused image (the "stabilized map"
stage), and inferring which eye is fresher; neither is possible for a per-patch linear readout.

## E14 - Fused model in the live mirror: ghost calibration and motion dependence (2026-10-30)

**Setup:** fused model integrated into `psychomanteum.py` (bottom panel = its prediction, ghosts =
its error, stats `model_err` / `model_copy_err`), fusion_rate default 1.0 (E13).

**Ghost calibration** (clip frames 3,000-6,000, model error enlarged to display size, percentiles
50/90/99): stillest 20% of frames raw 0.011/0.024/0.037, blurred (sigma 6) 0.012/0.017/0.024; most
motion 20% raw 0.014/0.060/0.280, blurred 0.014/0.068/0.223. -> model error map is blurred
(sigma 6) so stillness stays under ghost_floor 0.03 while motion still shows ghosts.

**Cost:** eyes + V1 + fusion 5.1 ms, flow + compose 12.9 ms -> 18 ms/frame.

**Live result (webcam, person mostly still):** model skill vs copy-avg -0.37 (motion 0.17), then
-0.16 (2.06), -0.21 (0.42) at rate 1.0; after switching to rate 0.1: -0.47 (0.54), -0.77 (0.12).
Saved weights showed the bias drifted to +0.09 everywhere; other weights changed by ~0.001-0.002.
Drifted weights kept as `state/fusion_live_drifted.npz`.

**Replay check** (the real mirror code, `--replay` on the E12 clip, fresh fusion weights, rate 1.0,
10 s windows): skill -0.013 (motion 4.96), -0.000 (6.08), +0.065 (10.97), -0.141 (1.78),
-0.139 (0.98), -0.026 (4.45).

**Conclusions:**
- Not a live-only bug: the same code on the clip reproduces the pattern.
- **The fused model helps during strong motion and hurts in stillness.** E13's positive offline
  skill is a ratio of summed MSE, which high-motion frames dominate, so it hid the losses on
  still frames. Live, the person was mostly still, so skill went negative.
- The rate-0.1 test doesn't separate cause from history (it inherited weights from the rate-1.0
  phase), so "high learning rate causes the stillness loss" is unconfirmed.

**Next (hypothesis):** precision weighting - blend model and copy-avg per patch by their recent
error, so the system trusts the model only where it has recently been better.

## E15 - Precision weighting, and skill by motion level (2026-10-30)

**Question:** Does blending the fused model with the eyes' average per patch, weighted by each
one's recent precision, fix the losses seen in E14? How does skill depend on motion?

**Setup:** `FusionPredictor.track_precision` / `blend`: per patch, running squared error (decay
0.9, ~0.3 s) of the raw model and of copy-avg; trust = copy_err / (model_err + copy_err);
output = trust * model + (1 - trust) * copy-avg. Model still learns from its own raw prediction.
`evaluate.py`, full E12 clip, horizon 3, rate 1.0, same conditions as E13. Motion = MSE between
the true frame and the true frame `horizon` steps earlier (eye resolution); quintiles over frames.
Skill = 1 - summed MSE / summed copy-avg MSE within each group.

**Result** (both eyes; MSE x1000: model 2.532, precision 2.474, copy-avg 2.577, single eye 3.012):

| motion quintile | motion | raw model | precision-weighted |
|---|---|---|---|
| 1 (stillest) | 0.015 | +0.114 | +0.142 |
| 2 | 0.076 | -0.195 | +0.011 |
| 3 | 0.320 | -0.101 | +0.021 |
| 4 | 1.252 | +0.000 | +0.038 |
| 5 (most) | 6.364 | +0.043 | +0.040 |
| whole clip | | +0.018 | **+0.040** |
| frames better than copy-avg | | 47.6% | **79.9%** |

Left eye only (precision-weighted): MSE 2.830, skill vs two-eye copy-avg -0.098. **Two eyes vs
one (both precision-weighted): 2.474 vs 2.830 -> 12.6% lower error.**

**Conclusions:**
- Precision weighting beats the eyes' average in every motion quintile and on 80% of frames,
  doubling the whole-clip gain (+0.018 -> +0.040).
- **Correction to E14:** the raw model does not hurt in stillness - it is best when fully still
  (+0.114, likely learning the brightness bias from noise clipping, cf. the +0.09 bias in E14).
  It hurts during *slight* motion (quintiles 2-3). E14's live negatives were in that range.

**Closing metric for the demo:** precision-weighted fusion of two delayed, noisy 80x45 eyes
predicts the true frame 100 ms ahead better than averaging the eyes on 80% of frames and at every
motion level (+1% to +14%), and two eyes beat one by 12.6%.

**Caveats:** one clip, one person, one seed; gains are small in absolute terms (MSE 2.47 vs 2.58).

**Live confirmation** (webcam, person mostly still, 8 s windows, right after restart with weights
warm-started from the E14 replay): model skill +0.050 (motion 0.10), +0.030 (0.22), +0.048 (0.16) -
positive in the same low-motion regime where E14 measured -0.16 to -0.77 without precision weighting.

## E16 - Motion compensation ("stabilized map") + three-way precision weighting (2026-10-30)

**Question:** Does pushing the eyes' average along its own optical flow (a "stabilized map")
add predictive power, and how should it be combined with the learned model?

**Setup:** `EyeFlow`: Farneback flow between consecutive eye-average images (blurred sigma 1,
upsampled 2x), latest average warped forward by `reach` steps. `PrecisionMixer`: per patch,
weights proportional to 1 / running squared error (decay 0.9) over three sources: learned model,
motion-compensated average, plain average ("mix3"). `evaluate.py`, full E12 clip, horizon 3,
rate 1.0, same conditions as E13/E15. Skill vs copy-avg, summed MSE.

**Result 1 - reach sweep (both eyes):**

| reach | motion-compensated alone | mix3 | mix3 frames better than copy-avg |
|---|---|---|---|
| 3 | +0.135 | +0.210 | 93.7% |
| 5 | -0.121 | **+0.240** | 90.0% |
| 7 | -0.528 | +0.222 | 87.9% |

**Result 2 - by motion quintile (reach 5, both eyes):**

| quintile | motion | model | precision (E15) | motion-comp. alone | mix3 |
|---|---|---|---|---|---|
| 1 (stillest) | 0.015 | +0.114 | +0.142 | -2.195 | +0.132 |
| 2 | 0.076 | -0.195 | +0.011 | -1.943 | +0.096 |
| 3 | 0.320 | -0.101 | +0.021 | -0.720 | +0.208 |
| 4 | 1.252 | +0.000 | +0.038 | -0.062 | +0.284 |
| 5 (most) | 6.364 | +0.043 | +0.040 | +0.110 | +0.242 |

**Result 3 - one eye vs two (reach 5; MSE x1000):** mix3 both 1.959 vs left only 2.320 ->
**two eyes 15.6% lower**. Precision (two sources) in the same runs: both 2.474 vs left 2.821 ->
12.3% lower (E15 reported 2.830 for left; small run-to-run difference, cause not investigated).
Left-only mix3 skill vs the two-eye copy-avg: +0.100, but negative in quintiles 1-3.

**Live** (webcam, after restart, 8 s windows, both eyes): model skill +0.357 (motion 1.66),
+0.314 (1.35), +0.265 (0.35); live one-vs-two errors both 1.51 / left 1.77 / right 1.85, then
0.96 / 1.23 / 1.33, then 0.40 / 0.63 / 0.62. Cost of three eye modes with motion compensation and
mixing: 5.0 ms/frame (~23 ms total per frame).

**Conclusions:**
- **mix3 at reach 5: +0.240 skill (24% lower error than averaging the eyes), better on 90% of
  frames, positive in every motion quintile (+0.10 to +0.28).** reach 5 = horizon 3 + ~2 frames
  of mean eye lag, as expected. reach 3 wins on share of frames and in stillness.
- Motion compensation alone is far worse than averaging when still (it follows noise); precision
  weighting makes it useful by trusting it only where it has recently been right.
- Live skill rose from +0.03-0.05 (E15) to +0.27-0.36.

**Caveats:** one clip, one person, one seed; reach not tuned between 3 and 5; live windows short.

## E17 - Longer horizons: several reaches, smoothed flow and a blurred average (2026-10-31)

**Question:** how does the E16 system hold up further ahead, and can it be made to degrade less?

**Setup:** `evaluate.py` on the full E12 clip, both eyes, rate 1.0, reach = horizon + 2, one
seed. Step 1: E16 system (mix3) at horizons 6, 10, 15 (`results/horizon_{6,10,15}.log`; horizon 3
is E16). Step 2: a second mixer, mixN, scored on the same frames as mix3, with sources = model,
the eyes' average pushed by fractions 0.25/0.5/1.0 of reach along a running average of the flow
(0.7 old + 0.3 new), the average blurred (sigma 2 eye pixels), and the plain average
(`--fractions 0.25 0.5 1 --flow-smoothing 0.7 --blur 2`; `results/h*_{reaches,smooth,blur,all}.log`).
With default options mixN reproduces mix3 exactly (checked on 3000 frames).

**Result 1 - E16 system vs horizon (skill vs copy-avg):**

| horizon | 3 (E16) | 6 | 10 | 15 |
|---|---|---|---|---|
| mix3 | +0.240 | +0.202 | +0.167 | +0.139 |
| motion-comp. alone | -0.121 | -0.355 | -0.648 | -0.966 |
| mix3, quintiles 1 / 2 | +0.132 / +0.096 | +0.039 / +0.045 | -0.024 / -0.011 | -0.093 / -0.045 |

**Result 2 - each change alone, horizon 15 (mix3 +0.139 in every run):**

| mixN variant | skill | better than mix3 | quintile 1 | quintile 5 |
|---|---|---|---|---|
| reaches 0.25/0.5/1 | +0.166 | 53.5% | -0.161 | +0.184 |
| flow smoothing 0.7 | +0.148 | 65.6% | -0.027 | +0.167 |
| blur 2 | +0.163 | 64.1% | -0.184 | +0.186 |
| all three | **+0.182** | **72.7%** | -0.051 | +0.190 |

All three, by quintile at horizon 15: -0.051 / +0.055 / +0.171 / +0.200 / +0.190 (mix3: -0.093 /
-0.045 / +0.090 / +0.144 / +0.163).

**Result 3 - all three vs horizon:**

| horizon | 3 | 6 | 10 | 15 |
|---|---|---|---|---|
| mix3 | +0.240 | +0.202 | +0.167 | +0.139 |
| mixN (all three) | +0.233 | +0.217 | +0.199 | +0.182 |
| mixN better than mix3 | 54.2% | 65.5% | 71.0% | 72.7% |

Cost: 1.32 ms/frame for the new stabilized sources vs 1.20 ms for the single E16 push (80x45,
microbenchmark), per eye mode.

**Conclusions:**
- The E16 system's skill falls from +0.240 to +0.139 between 100 and 500 ms; the weak part is
  motion compensation, whose one-step flow noise is multiplied by the reach.
- The three changes together halve the fall (+0.233 -> +0.182) and the gain grows with horizon
  (-0.007, +0.015, +0.032, +0.043). Each alone helps; smoothing is what protects stillness.
- Adopted in the live mirror for all horizons (`EyeFlow.sources`). The fixed `eyeflow_reach`
  parameter was replaced by `eyeflow_lead` (reach = horizon + lead, default 2), since a fixed
  reach of 5 would under-push at long horizons. Live smoke test on replay at horizon 15: runs
  at 30 fps, no errors (one 1 s window, not a skill measurement).

**Caveats:** one clip, one person, one seed; fractions, smoothing and blur were each tried at one
value (not tuned); horizon 3 is slightly worse (-0.007, within what a seed change might move,
untested); stillest quintile still below copy-avg at horizon 15.

## E18 - Tuning the long-horizon sources at horizon 15, and sharper trust weighting (2026-10-31)

**Question:** which smoothing, blur, reach fractions and lead work best at horizon 15 (500 ms), and
does sharper trust weighting (weights ~ precision ** k) help?

**Setup:** `evaluate.py` mixN, full E12 clip, horizon 15, one seed. **Settings chosen on the first
half of the scored frames only; the second half is the held-out check** (`skill_by_half` added to
`evaluate.py`). Starting point = E17 (fractions 0.25/0.5/1, smoothing 0.7, blur 2, lead 2,
sharpness 1). Round 1 changes one setting at a time; round 2 combines the round-1 winners, tries
sharpness and extends the winning directions. Runner and configs:
`results/tune15/{sweep.sh,configs1.txt,configs2.txt,summary.py}`, logs alongside.

**Result (mixN skill vs copy-avg; mix3 is +0.139 in every run):**

| config | first half | second half | full | quintile 1 |
|---|---|---|---|---|
| E17 start (s0.7 b2) | +0.1769 | +0.1884 | +0.1820 | -0.051 |
| smoothing 0.5 | +0.1841 | +0.1949 | +0.1889 | -0.095 |
| smoothing 0.3 | +0.1839 | +0.1932 | +0.1880 | -0.144 |
| smoothing 0.85 / 0.95 | +0.1583 / +0.1245 | | | |
| blur 1 / 3 / 4 / 6 | +0.1675 / +0.1816 / +0.1838 / +0.1851 | | | |
| lead 0 / 4 (reach 15 / 19) | +0.1740 / +0.1788 | | | |
| fractions 0.5/1 | +0.1782 | | | -0.093 |
| fractions 0.1/0.25/0.5/1 | +0.1712 | | | -0.005 |
| fractions 0.25/0.5/1/1.5 | +0.1784 | | | -0.140 |
| fractions 0.25/0.5/0.75/1 | +0.1740 | | | -0.112 |
| sharpness 1.5 / 2 / 3 (on E17 start) | +0.1755 / +0.1734 / +0.1682 | +0.1875 / +0.1856 / +0.1805 | | -0.022 / -0.013 / -0.013 |
| **s0.5 b6** | **+0.1923** | **+0.1995** | **+0.1955** | -0.112 |
| s0.5 b4 lead 4 | +0.1918 | +0.1996 | +0.1952 | -0.132 |
| s0.5 b4 | +0.1909 | +0.1993 | +0.1946 | -0.110 |
| s0.5 b4 lead 4, fractions 0.5/1 | +0.1891 | +0.1957 | +0.1920 | -0.172 |
| s0.5 b4 lead 4 sharpness 2 / 3 | +0.1881 / +0.1813 | +0.1968 / +0.1898 | | -0.044 / -0.040 |
| s0.5 b4 sharpness 2 | +0.1874 | +0.1964 | +0.1914 | -0.033 |

**Conclusions:**
- Adopted **smoothing 0.5, blur 6** (fractions and lead unchanged) as the `EyeFlow.sources`
  defaults: first half +0.192 vs +0.177; **held-out second half +0.200 vs +0.188 (+0.011)**. Small
  but it holds out of sample. The top three configs are within 0.0015 of each other (noise level).
- Lighter smoothing and more blur help overall but make the stillest quintile worse.
- **Sharper trust weighting is a trade, not a win:** it lowers overall skill (sharpness 2: -0.004)
  but pulls the stillest quintile close to even (-0.110 -> -0.033 at s0.5 b4). Kept at 1, since the
  objective is overall skill; worth revisiting if jitter while standing still looks bad live.
- Trust weights are not stored (running average, decay 0.9), so they re-adapt to a new background
  within about a second. Live at horizon 15 after a background change, with E17 settings: model
  skill +0.144 (10 s window, motion 1.16), two eyes 0.648 vs left 0.732 / right 0.735.
- Default `horizon` changed from 3 to 15 (500 ms ahead).
- Check at horizon 3 with the adopted settings (`results/h3_tuned.log`): mixN **+0.258** vs mix3
  +0.240 (halves +0.256 / +0.261), better on 58.3% of frames - so the tuned version also beats
  E16 at 100 ms, where the untuned E17 version was slightly worse (-0.007).

**Caveats:** one clip, one person, one seed; first and second halves are of the same recording
and session, so "held out" means held-out time, not a new scene.

## E19 - Clean eyes live, and the cost of eye resolution (2026-10-31)

**Question:** what happens live without the eyes' deliberate degradation (noise, wandering delay,
low resolution), and what does more resolution cost?

**Setup:** live mirror, horizon 15, E18 settings, dither off. `mantis.py set eye_noise=0
eye_max_delay=0 eye_jitter=0`, then `observe 10`. Resolution: a copy of the mirror on the E12 clip
(replay) in a pseudo-terminal, `eye_width` changed live 80 -> 160 -> 320, fps from its stats lines
(live mirror stopped, so no competition).

**Result:**
- Clean eyes at 80x45, one 10 s window: motion 17.5 (lots of movement), model skill **+0.314**;
  both / left / right eyes 10.962 / 10.956 / 10.953 - identical, as expected, since without noise
  and delay both eyes see the same image.
- Changing `eye_width` live crashed the mirror: `FusionPredictor.inputs` stacked the previous
  step's codes (50 patches at width 80) with the new ones (220 at width 160). Bug existed since the
  fused predictor was added; fixed by dropping the previous codes when the patch count changes.
- Frame rate by eye width: 80 -> 30 fps, 160 -> ~23 fps, 320 -> ~9 fps (V1 sparse coding runs 30
  iterations per 8x8 patch; patch count grows 4x per doubling).

**Conclusions:** without noise and delay the second eye adds nothing, so the "two eyes beat one"
result (E16) only exists because the eyes are degraded differently. Widths above 80 drop below
30 fps, which also stretches the horizon in time (15 steps at 23 fps is ~650 ms).

**Caveats:** one live window for clean eyes, not comparable to earlier windows (motion 17.5 vs 3.5-6.7);
clean-eye skill not measured offline.

## E20 - Tuning for clean eyes at horizon 15 (2026-10-31)

**Question:** live with clean eyes (E19) the 500 ms prediction went negative in still windows
(+0.021 at motion 1.8, -0.141 at motion 1.5). Why, and what settings suit clean eyes?

**Setup:** `evaluate.py --clean` (eye_noise 0, eye_max_delay 0, eye_jitter 0), full E12 clip,
horizon 15, one seed; chosen on the first half, second half held out. New `--no-model` (mixN
without the learned model). Configs and logs: `results/tune15clean/` (configsA, configsB).

**Result 1 - diagnosis, E18 settings with clean eyes (skill by motion quintile):**

| | q1 (stillest) | q2 | q3 | q4 | q5 | overall |
|---|---|---|---|---|---|---|
| learned model alone | -6.782 | -2.083 | -0.659 | -0.244 | +0.016 | |
| mixN | -0.338 | -0.092 | +0.145 | +0.224 | +0.248 | +0.221 |

Overall clean-eye skill (+0.221) is higher than with noisy eyes (E18 +0.196), but the stillest
fifth is much worse (-0.338 vs -0.112).

**Result 2 - tuning (first half / second half / stillest quintile):**

| config | first | second | q1 |
|---|---|---|---|
| E18 settings (s0.5 b6) | +0.2176 | +0.2247 | -0.338 |
| **smoothing 0** | **+0.2272** | **+0.2332** | -0.378 |
| smoothing 0.3 / 0.7 | +0.2237 / +0.2033 | | |
| blur 0 / 2 / 4 | +0.1963 / +0.2118 / +0.2173 | | -0.203 / -0.306 / -0.335 |
| lead 0 (reach 15) | +0.2159 | | -0.296 |
| sharpness 2 / 3 | +0.2106 / +0.2009 | | -0.282 / -0.301 |
| fractions 0.5/1 / 0.1-1 | +0.2124 / +0.2137 | | -0.483 / -0.224 |
| s0, readout rate 0.3 / 0.1 / 0.03 | +0.2277 / +0.2268 / +0.2252 | +0.2320 / +0.2319 / +0.2326 | -0.374 / -0.361 / -0.326 |
| s0, no learned model | +0.2260 | +0.2301 | -0.444 |
| s0, no model, sharpness 2 | +0.2199 | | -0.328 |
| s0, rate 0.1, lead 0 / fractions 0.1-1 | +0.2249 / +0.2215 | | -0.317 / -0.239 |

**Conclusions:**
- Adopted **flow smoothing 0 for clean eyes** (+0.2332 vs +0.2247 held out). Smoothing was
  protecting against eye noise; without noise it only lags. Made it a live parameter,
  `eyeflow_smoothing` (default 0.5 for the standard noisy eyes, E18; set to 0 in control.json).
- Two of my guesses were wrong: a lower readout learning rate does not tame the learned model when
  still (rate 0.1: model alone -9.409 in q1), and dropping the model makes still frames worse
  (-0.444), so precision weighting is already limiting it. The still-frame loss comes from the
  mix as a whole lagging behind exact copying, which is near-optimal for clean eyes when still.
- **Unsolved:** with clean eyes the prediction loses to copying in the stillest ~40% of frames at
  500 ms. Every setting that helps there (blur 0, smaller fractions, sharpness) costs overall skill.

**Caveats:** one clip, one person, one seed; top configs within 0.001 of each other.

## E21 - Neural cellular automaton, full colour, 1 s ahead: blurry vs crisp (2026-10-31)

**Question:** can a neural cellular automaton trained on the recording predict the full-colour
camera frame 1 s (30 frames) ahead, and can it give one crisp future instead of a blur?

**Setup:** `nca.py` (PyTorch, M1 Pro GPU). Camera frames (not the eyes) at 160x90, RGB. Input: the
current frame and frames 3 and 6 steps earlier (fixed channels) plus 16 hidden channels; one
shared rule (3x3 identity / Sobel x / Sobel y / Laplacian of every channel -> 1x1 conv 96 ->
ReLU -> 1x1 conv 16, last layer zero-initialised = copy) applied 24 times. Trained on the first
80% of the E12 clip, evaluated on the last 20% (split in time); batch 16, Adam 1e-3, cosine decay.
Skill = 1 - MSE / MSE of copying the current frame. Full res = the 160x90 output enlarged and
applied to the 640x360 current frame. `nca.py eval --every 20` (292 held-out frames; full res on
the first 200 of them).
- **delta**: output = colour change added to the current frame; MSE loss; stopped at step 1500 of
  2000 (`results/nca_train.log`).
- **warp**: output = motion field (x10 scale), the current frame resampled along it, so pixels
  move but are never blended; L1 loss; 1500 steps (`results/nca_warp_train.log`). Without the x10
  scale, after 100 steps the largest motion was 0.28 px and skill +0.015.

**Result:**

| held-out | delta (blurry) | warp (crisp) |
|---|---|---|
| skill at 160x90 | +0.226 | +0.113 |
| skill at 640x360 | +0.200 | +0.073 |
| better than copying on | 74.0% of frames | 82.2% of frames |
| motion quintile 1 (stillest) | -0.322 | -0.038 |
| quintile 2 / 3 / 4 / 5 | +0.028 / +0.184 / +0.237 / +0.252 | +0.069 / +0.142 / +0.116 / +0.112 |

Warp training curve (256-frame probe): +0.030 (100), +0.070 (300), +0.094 (500), +0.105 (700),
+0.105 (1000), +0.116 (1200), +0.117 (1500) - flat over the last 400 steps, so not continued.
Live viewer `nca_mirror.py` (replay, during training on the same GPU): 52 ms/frame.

**Conclusions:**
- Both beat copying the current frame 1 s ahead on held-out video. The blurry model scores about
  twice as high, mostly by hedging (averaging possible futures) during motion.
- The crisp model gives up half the skill but commits to one future, is better than copying on
  more frames (82%), and barely loses when still (-0.038 vs -0.322): a motion field of zero is
  an exact copy, while the colour change adds noise everywhere.
- The user judged the blurry output too blurry live; the crisp model is now the `nca_mirror.py` default.

**Caveats:** one clip, one person; held-out frames are later in the same session (same room and
lighting), not a new scene; one training seed; delta stopped early; delta and warp also differ in
loss (MSE vs L1), so the comparison mixes read-out and loss.

## E22 - NCA mirror latency, and learning live (2026-10-31)

**Question:** the crisp NCA mirror (E21) felt slightly delayed; where does the time go, and can
it keep learning from the live camera without slowing the display?

**Setup:** `nca_mirror.py` copy on the E12 clip (replay) in a pseudo-terminal, status line logged;
microbenchmark of the warp NCA (batch 1, 160x90) on the M1 Pro. The live NCA mirror was running at
the same time in all of these, sharing the GPU.

**Result:**
- Frame time 64 ms (~15 fps): the viewer ran the NCA twice per frame (scoring and display). NCA
  forward + copy to CPU: 32.4 ms on the GPU (MPS), 45.6 ms on the CPU; full-res warp 0.7 ms.
  Running it once: **34 ms/frame** (~30 fps).
- `--learn` (CPU thread, 4 threads, batch 8, Adam 1e-4, L1, buffer of the last 60 s of live
  pairs): 32 training steps in ~35 s (~1 step/s, i.e. ~8 samples/s while 30 new pairs/s arrive);
  display 39 ms/frame (vs 33 without learning in the same run); 3 s skill +0.15 -> +0.16 (too
  short to show learning).

**Conclusions:** the felt delay was mostly the doubled NCA call, now fixed. Live learning runs
alongside the display at a small cost, but learns slowly (~1 step/s on the CPU); whether it
improves the prediction in a new room is untested.

**Follow-up, after ~9 min of live learning (~500 steps, new room, 23:03 checkpoint):** a fresh
30 s live clip (902 frames, 30 fps, captured while the mirror kept running) and the held-out 20%
of the E12 recording, both models on the same frames (`scratchpad/compare_live.py`):

| | live clip (new room) | held-out recording (old room) |
|---|---|---|
| recording-trained | +0.064 | +0.115 |
| live-trained | +0.068 | +0.086 |

Live learning so far: **no real gain in the new room (+0.004) and some forgetting of the old one
(-0.029)**. Caveat: one 30 s clip, motion not controlled.

**GPU training cost (M1 Pro, MPS, warp NCA at 160x90, torch 2.15; mirror stopped):** training step
batch 1 / 2 / 4 / 8 = 60 / 117 / 220 / 497 ms (~60 ms per example); inference batch 1 = 19 ms.
With the NCA mirror and its CPU learner running at the same time: 309 / 542 / 997 / 1573 ms and
inference 94 ms. **Implication:** a 30 fps display has ~14 ms spare per frame after its 19 ms
prediction, less than one 60 ms training step; the GPU's spare time is worth ~7 examples/s, about
the same as the CPU learner (~8/s), so moving live learning to the GPU between frames would not
speed it up, and would cost frame rate.

## E23 - NCA lineage: inner steps as a timeline, combined guesses, confidence (2026-11-01)

**Question:** if each of the NCA's inner steps predicts further ahead (a rollout through time), (a)
does that hurt or help the 1 s prediction, (b) does combining the many predictions of the same
moment (made from successive frames) beat the newest one, and (c) does their disagreement show
where the prediction will be wrong?

**Setup:** `nca.py train --out lineage --loss l1 --init-from warp --iters 1500`: the E21 warp
NCA (24 steps), but read out after every step; step k (1-based) predicts 2k frames ahead
(66 ms ... 1.6 s), L1 loss averaged over all 24 steps; warm-started from the E21 warp weights.
Same data split as E21. `nca.py eval --out lineage --every 10` (`results/nca_lineage_eval.log`):
579 held-out target times. Combined guesses: the frame 1 s after t predicted from frames t, t-2,
... t-18 (10 sources, at horizons 1.0 ... 1.6 s); precision weights by source age fitted on the
first half of the targets, scored on the second half. Confidence: per-pixel standard deviation
across the 10 predictions vs the newest prediction's actual error; control = how much each pixel
is moving now (|frame t - frame t-3|).

**Result:**
- Training probe at 1 s: +0.118 (300), +0.129 (600), +0.132 (900), +0.141 (1200), +0.140 (1500).
- Skill by horizon: 0.13 s +0.081, 0.27 s +0.105, 0.40 s +0.116, 0.53 s +0.126, 0.80 s +0.133,
  1.00 s +0.133, 1.33 s +0.131, 1.60 s +0.132.
- **At 1 s: lineage +0.133 vs E21 warp +0.117 on the same frames.**
- Combined guesses (second half, 290 targets): newest only +0.138, plain average +0.148,
  **precision-weighted +0.152**. Weights by age 0.11 (newest) ... 0.09 (oldest): nearly uniform.
- Confidence: ensemble spread correlates 0.425 with the actual error vs 0.303 for the motion
  control; but in the top 10% of pixels both mark 4.0x the error of the rest.
- Mirror (`nca_mirror.py --out lineage`, replay, GPU free): prediction 37 ms/frame, tracers 42,
  confidence 40 (~25 fps; 10 full-res warps per frame for the average, 8 more for tracers).

**Conclusions:**
- A loss at every inner step helps the 1 s prediction (+0.016) as well as giving a timeline.
- Skill relative to copying is flat from ~0.8 s to 1.6 s: copying gets worse about as fast as
  the prediction does.
- Combining the 10 predictions of the same moment adds +0.014. Averaging warped frames blends
  them, so this partly reintroduces the blur the crisp read-out avoided (E21), though much
  less than the delta model, because each source is a sharp warp.
- Disagreement between predictions is a better overall indicator of error than "where it is
  moving" (0.43 vs 0.30 correlation) but no better at singling out the worst 10% of pixels.

**Caveats:** one clip and seed; held-out frames from the same session; warm-started from E21, so
not an independent training run; the weights are fitted and scored on halves of the same
held-out set.

**Next (planned, not built) - an NCA that never stops:** keep the hidden state between frames
instead of resetting it, run a few rule steps per camera frame, and inject each new frame into the
visible channels. Train on sequences (e.g. 32 consecutive frames, truncated backpropagation
through time, loss at every frame for the lineage horizons). Questions: does memory beyond the
current 200 ms of input help (compare with E23 at equal compute)? Does the state stay stable over
minutes live? Does the always-on version cost less per frame (fewer steps, since it does not start
from zero each frame)?

## E25 - How far ahead does the NCA's prediction actually go? (2026-11-01)

**Question:** the lineage NCA mirror (E23) "feels realtime". How much of the actual motion over the
next second does its predicted motion cover, and what does exaggerating it for display cost?

**Setup:** `results/nca_undershoot.py lineage`: 581 held-out recording frames (every 10th), the
lineage model's 1 s motion field (newest source only, no averaging). Actual motion = Farneback
optical flow from the frame 1 s later back to the current frame (the model's convention), 160x90.
Skill vs copying with the predicted motion multiplied by 0.5 ... 3.

**Result:**
- On pixels that actually moved more than 1 cell: actual 2.90 cells, predicted 0.41 ->
  **the prediction covers 14% of the distance.**
- Best single multiplier for the predicted motion against the actual flow (least squares): 0.19.
- Skill at multiplier x0.5 / x1 / x1.5 / x2 / x3: +0.080 / **+0.131** / +0.117 / +0.084 / +0.022.

**Conclusions:**
- The mirror looks like "now" because the prediction moves things about a seventh as far as they
  actually go: under uncertainty at 1 s the error-minimising motion is small.
- Scaling it up does not make it more right (best multiplier < 1: its directions are often off,
  so a larger push adds error). x1 is the most accurate, as trained; x2 still beats copying
  (+0.084) and shows the predicted future more visibly.
- Added `[` `]` to `nca_mirror.py`: display-only multiplier (x0.5 ... x4); the status line's
  score always uses x1.

**Caveats:** Farneback flow over 1 s at 160x90 is itself error-prone (large motions,
occlusions), so the 14% is approximate; one clip.

## E24 - An NCA that never stops (2026-11-01)

**Question:** E23's plan: if the NCA keeps its hidden state between frames (memory) and runs only a
few rule steps per frame, how well does it predict 1 s ahead, does it stay stable over minutes,
and how much cheaper is it?

**Setup:** `nca.py --out persistent`: `PersistentNCA` (24 hidden channels, rule width 96, 6 steps
per camera frame). Each frame, the current frame and the one 3 frames (100 ms) earlier are written
into the visible channels; a zero-initialised 1x1 read-out gives motion fields for 10, 20 and 30
frames ahead. Training (`train_persistent`, batch 8, Adam 1e-3, cosine, L1, 1500 iterations): 8
streams run on through the training 80% carrying their state; each iteration advances every stream
8 frames and backpropagates through those 8 frames (48 rule steps); a stream restarts from zero at a
random place with probability 1/16 per iteration, with no loss for its first 8 frames. Probe: 8 runs
of 96 held-out frames from zero, scored after 32. Evaluation (`eval --out persistent --every 10`,
`results/nca_persistent_eval.log`): one non-stop run from a single zero start through all 5803
held-out frames (3.2 min), scored at 1 s on the E23 target frames, next to the lineage model.

**Getting it to train (three bugs, all mine):**
1. With the current frame only and the rule's last layer **and** the read-out both zero-initialised,
   no gradient reached either: only the read-out bias could learn (a global shift). Probe +0.008,
   +0.004, +0.001 at 200/300/400 iterations (`results/nca_persistent_frame_only_train.log`); the
   two-frame version gave identical numbers at 100 iterations (loss 33.890, +0.006), which exposed
   it. So the "current frame only does not learn" result is void - that variant is untested.
2. With the rule initialised small (std 0.02) and the state bounded by a hard clamp at +/-5 plus a
   0.5%/step leak, the state ran to the bound: 59% of cells at +/-5 within 20 frames, read-out
   weights ~0.0003 after 100 iterations.
3. Fix: state = tanh(state + rule(...)) each step, no leak. Inspection after 100 iterations: state rms
   0.28, no cells above 0.99, steady over 300 frames.
   (Before any of this, without a separate read-out, the raw accumulating state channels gave skill
   -8.5 after 4 iterations.)

**Result:**
- Probe during training: +0.016 (200), +0.027 (400), +0.046 (700), +0.068 (1000), +0.081 (1100),
  +0.078 (1200), +0.077 (1300), +0.078 (1400), +0.079 (1500) - levelled off.
- **Non-stop run: +0.052 at 1 s vs lineage +0.133 on the same frames.** By minute: +0.049 / +0.049 /
  +0.058 / +0.056 (lineage +0.100 / +0.139 / +0.137 / +0.142).
- **Stable:** state rms by minute 0.147 / 0.167 / 0.157 / 0.152 (max 0.227) - no drift over 3.2 min.
- **Cost:** 7.1 ms per frame (median, batch 1, GPU, with the lineage mirror running) vs ~19 ms for
  the 24-step NCAs (E22). Mirror (replay, early checkpoint, training running): 26 ms/frame.

**Conclusions:**
- The always-on automaton works mechanically: it runs indefinitely without drifting and is ~3x
  cheaper per frame.
- Its memory does not make up for having a quarter of the rule steps per frame and being trained
  from scratch: 1 s skill +0.052 vs +0.133. Not a like-for-like test of memory: steps per frame,
  warm start (lineage was warm-started from E21) and read-out all differ.
- The probe (+0.079) overstates the non-stop result (+0.052); the probe runs are short and from
  fixed starting points, so they are not a good proxy.

**Caveats:** one clip, one seed, held-out frames from the same session; settings (6 steps, 24
hidden, tanh, chunk 8) untuned.

## E26 - Graph NCA on superpixels (2026-11-01)

**Question:** if the NCA's cells are superpixels (regions that follow image edges) on a graph
instead of grid cells, and each region moves as a rigid piece, how well does it predict 1 s ahead
compared with the grid NCA - and does the "hallucinatory" stretching (open hypothesis after E21)
go away?

**Setup:** `graph_nca.py` (Grattarola et al. 2021 graph NCA). SLIC superpixels at 160x90
(n_segments 300, compactness 10 -> ~240 regions of ~60 px; 8.2 ms per frame; cached for all
29,284 frames in 2 min with 8 processes, `data/slic_160x90.npy`). Node inputs: mean colour at
`nca.PAST` (now, 100 ms, 200 ms ago) in the region's current pixels, position, size; 16 hidden
channels. Edges: regions that touch. Perception per node: own state, mean neighbour difference,
and that difference weighted by the x and y direction to each neighbour; rule 4x28 -> 96 -> 16,
last layer zero-initialised; 16 steps. Two hidden channels x10 = region motion; every pixel moves
with its region (piecewise-constant flow, warped as E21). Same training as the grid warp NCA
(E21): from scratch, L1, batch 16, Adam 1e-3 cosine, 1500 iterations, first 80% of frames.
`graph_nca.py eval` (`results/nca_graph_eval.log`): every 20th held-out frame, next to the E21 grid
warp NCA on the same frames.

**Result:**
- Training probe: +0.010 (100), +0.023 (300), +0.042 (600), +0.043 (900), +0.045 (1200), +0.045
  (1500). Training took 258 s (vs ~2100 s for the grid NCA).
- **Held-out skill at 1 s: graph +0.046 vs grid +0.113** (same 292 frames); graph better than
  copying on 77.4% of frames.
- By motion quintile (graph / grid): -0.023 / -0.038, +0.051 / +0.069, +0.047 / +0.142,
  +0.039 / +0.116, +0.049 / +0.112.
- Mirror (`nca_mirror.py --out graph`, replay, early checkpoint): 27 ms/frame including SLIC;
  `regions` view draws the superpixel borders.

**Conclusions:**
- Rigid regions cost more than half the skill. Slightly better when still (a region either
  moves or not, so there is less smearing noise), much worse with motion: real motion inside a
  region (fingers, a turning face) cannot be expressed, and regions that straddle a moving edge
  must move all or nothing.
- ~8x faster to train (~240 nodes per frame instead of 14,400 cells).
- Whether the look is less "hallucinatory" is the user's call in the mirror (not yet judged).

**Caveats:** one clip and seed; superpixels are recomputed from each frame alone, so regions
are not tracked between frames; settings (region count, steps) untuned.

## E27 - Graph NCA with region motion smoothed across borders (2026-11-01)

**Question:** E26's rigid regions cost more than half the grid NCA's skill. Does blurring the
per-pixel motion (so neighbouring regions blend at their borders) recover it?

**Setup:** `graph_nca.py train --smooth S` for S = 2 and 4 cells (regions are ~8 cells across): as
E26, but after every pixel takes its region's motion, the motion field is Gaussian-blurred (S
cells, replicate borders) in training and at display; S stored in the checkpoint. Both trained
in parallel from scratch, 1500 iterations; `graph_nca.py eval --smooth S`
(`results/nca_graph_s{2,4}_eval.log`), same 292 held-out frames as E26.

**Result (held-out skill at 1 s; E26 rigid and E21 grid on the same frames):**

| | overall | q1 (stillest) | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| rigid (E26) | +0.046 | -0.023 | +0.051 | +0.047 | +0.039 | +0.049 |
| blur 2 | +0.044 | **+0.014** | +0.047 | +0.045 | +0.039 | +0.046 |
| blur 4 | +0.038 | -0.009 | +0.061 | +0.050 | +0.033 | +0.038 |
| grid (E21) | +0.113 | -0.038 | +0.069 | +0.142 | +0.116 | +0.112 |

Training probe at 1500: blur 2 +0.045, blur 4 +0.039 (E26 rigid +0.045).

**Conclusions:**
- Smoothing does not recover the skill (blur 2 +0.044 vs rigid +0.046; blur 4 worse). The limit is
  one motion per region, not the tearing between regions.
- Blur 2 is the only NCA so far that beats copying in the stillest fifth of frames (+0.014).
- Started in the mirror (`nca_mirror.py --out graph --smooth 2`) for the user to judge the look.

**Caveats:** one clip and seed; differences between rigid and blur 2 overall (0.002) are within
noise; the mirror was started on the step-1400 checkpoint (probe +0.045, same as step 1500).

## E28 - Hybrid: grid NCA motion pulled toward superpixel averages (2026-11-01)

**Question:** E26/E27 mixed two things: rigid region motion and a different predictor (graph NCA).
If the best grid NCA's motion field is pulled toward its own average within each superpixel, how
does skill change with the strength of that pull?

**Setup:** `results/nca_hybrid_eval.py`: lineage NCA (E23), step 15 (1 s, newest source only).
motion = (1 - alpha) * grid motion + alpha * (grid motion averaged over the pixel's SLIC region,
cached labels from E26). No training. Same 292 held-out frames as E26/E27
(`results/nca_hybrid_eval.log`).

**Result:**

| alpha | skill | q1 (stillest) | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| 0 (grid) | +0.130 | -0.056 | +0.072 | +0.150 | +0.134 | +0.131 |
| 0.25 | +0.115 | -0.012 | +0.082 | +0.142 | +0.117 | +0.112 |
| 0.5 | +0.091 | +0.014 | +0.079 | +0.121 | +0.092 | +0.087 |
| 0.75 | +0.068 | +0.025 | +0.067 | +0.094 | +0.067 | +0.063 |
| 1 (rigid regions) | +0.047 | +0.022 | +0.052 | +0.067 | +0.045 | +0.044 |

**Conclusions:**
- **Rigid regions driven by the grid NCA (+0.047) score the same as the graph NCA (+0.046, E26):**
  the graph NCA was not a weaker predictor; one motion per superpixel is what costs the skill.
- A smooth trade-off: each step toward region-shaped motion helps when still and hurts with
  motion; no alpha improves overall skill.
- For the hallucination hypothesis: superpixels are colour patches, not objects; forcing patches
  to move as units costs accuracy at 1 s. An object-level prior would need real object or part
  segmentation to test fairly.

**Caveats:** one clip; alpha fixed per run (not learned or chosen per region); lineage newest
source only (the mirror averages 10 sources).

## E29 - Precision weighting for the NCA, phase 0: non-stop baseline (2026-11-01)

**Question:** before testing the precision-weighting hypothesis (Open hypotheses), what do the
candidate sources score on *every* held-out frame, in order, with the new sharpness measure?

**Setup:** `nca_mix.py baseline` (`results/mix_baseline.log`, per-frame and per-8x8-patch errors in
`results/mix_baseline.npz`): every held-out frame with a full ensemble (5803 targets, 3.2 min),
1 s ahead, 160x90. Sources: copy (frame t); nca (lineage step 15, newest); nca10 (average of the
1 s predictions from frames t, t-2, ... t-18, as E23); rigid (newest motion averaged within each
superpixel of frame t, E28 alpha 1). Sharpness = mean Sobel gradient magnitude of the luma,
prediction / true future frame, median over frames. Halves = first / second half of the targets.

**Result:**

| source | skill | half 1 | half 2 | q1 (still) | q2 | q3 | q4 | q5 | better than copy | sharpness |
|---|---|---|---|---|---|---|---|---|---|---|
| copy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 1.000 |
| nca | +0.131 | +0.127 | +0.135 | -0.057 | +0.061 | +0.148 | +0.143 | +0.130 | 82.2% | 0.943 |
| nca10 | +0.151 | +0.153 | +0.150 | -0.170 | -0.077 | +0.083 | +0.153 | +0.185 | 64.7% | 0.912 |
| rigid | +0.048 | +0.046 | +0.050 | -0.004 | +0.058 | +0.067 | +0.051 | +0.044 | 81.5% | 0.980 |

Flows for 5821 frames: 97 s; whole run 115 s.

**Conclusions:**
- Matches the sampled numbers (E23 nca10 +0.152 on half 2 of every 10th frame; E28 nca +0.130,
  q1 -0.056 on every 20th), so the non-stop evaluator is consistent.
- **nca10's gain is all in motion; it is much worse when still (-0.170) and in slight motion
  (-0.077)** than the newest guess, and softer (0.912 vs 0.943): averaging guesses made up to
  600 ms earlier blends in motion that has since stopped. E23 did not show this split.
- The sources' strengths are complementary by motion level (nca10 in motion, nca in the middle,
  rigid/copy when still) - the situation precision weighting is for.
- Targets for phase 1: P1 q1 >= 0; P2 overall >= +0.151 (nca10 on every frame); P3 sharpness of
  the motion-blend variant >= 0.85 (within 10% of nca's 0.943).

**Caveats:** one clip, held-out frames from the same session; sharpness relative to the true
future frame mixes crispness with how much detail the frame has.

## E30 - Precision weighting for the NCA, phase 1: mixers and controls (2026-11-01)

**Question:** does per-patch precision weighting of the E29 sources (copy, nca, nca10, rigid) at
1 s remove the still-frame loss without losing motion skill (P1, P2), stay crisp (P3), and survive
the 1 s accuracy lag (P4)?

**Setup:** `nca_mix.py mix` (`results/mix_phase1.log`, per-frame MSE in `results/mix_phase1.npz`).
Per 8x8 patch, running mean of each source's squared error (decay d), weights = normalised 1 /
error, **lagged**: a prediction at frame t only sees errors of predictions made at t - 30 or
earlier. A = pixel blend of all four (weights upsampled bilinearly); B = motion blend of copy (zero
motion), nca and rigid - one blended motion field, one warp; C = hard pick of the most precise
source. Controls: **oracle** (same running mean, no lag), **per-frame** (one weight set per frame),
**fixed** (global weights = normalised 1 / mean half-1 patch error). Decay chosen from 0.8 / 0.9 /
0.95 / 0.98 on half 1 of the 5803 targets; half 2 reported; quintiles from half 2. Sharpness as E29.

**Result (half 2):**

| variant | skill | q1 (still) | q2 | q3 | q4 | q5 | better than copy | sharpness |
|---|---|---|---|---|---|---|---|---|
| nca (source) | +0.135 | -0.000 | +0.114 | +0.153 | +0.151 | +0.128 | 88.2% | 0.936 |
| nca10 (source) | +0.150 | -0.275 | -0.048 | +0.129 | +0.181 | +0.174 | 67.3% | 0.903 |
| rigid (source) | +0.050 | +0.041 | +0.074 | +0.060 | +0.052 | +0.043 | 86.8% | 0.979 |
| **A lagged** (d 0.98) | +0.150 | **+0.050** | +0.127 | +0.162 | +0.171 | +0.141 | **95.2%** | 0.938 |
| A oracle | +0.157 | +0.055 | +0.130 | +0.167 | +0.178 | +0.151 | 95.5% | 0.938 |
| A per-frame | +0.150 | +0.049 | +0.125 | +0.162 | +0.171 | +0.142 | 94.9% | 0.938 |
| A fixed | +0.150 | +0.048 | +0.126 | +0.163 | +0.171 | +0.143 | 94.2% | 0.938 |
| **B lagged** (d 0.98) | +0.073 | +0.058 | +0.098 | +0.090 | +0.080 | +0.063 | 95.6% | **0.971** |
| B oracle / per-frame / fixed | +0.075 / +0.074 / +0.074 | | | | | | | 0.971 |
| C lagged (d 0.9) | +0.153 | -0.049 | +0.058 | +0.146 | +0.173 | +0.161 | 85.4% | 0.919 |
| **C oracle** | **+0.250** | +0.119 | +0.191 | +0.260 | +0.278 | +0.244 | 97.6% | 0.921 |
| C per-frame | +0.148 | -0.100 | +0.004 | +0.126 | +0.170 | +0.166 | 79.4% | 0.917 |

Half 1 (used to choose the decay) agrees within 0.01 throughout. Run time 162 s.

**Conclusions, against the predictions:**
- **P1 (still >= 0): holds** for A (+0.050) and B (+0.058); fails for C (-0.049).
- **P2 (overall >= nca10): A ties** (+0.150 vs +0.150) - no loss, no gain. A trades nca10's
  motion skill (q5 +0.174 -> +0.141) for stillness. C +0.153 passes P2 but fails P1. B loses
  half the skill (+0.073).
- **P3 (crisp): holds** - B is *sharper* than the NCA (0.971 vs 0.936); A matches it (0.938).
- **P4 (lag cost): holds for soft blends, fails for the hard pick.** A/B lagged are within 0.007
  of their oracles. C's oracle reaches **+0.250** but lagged only +0.153: the 1 s lag costs
  almost the whole gain.
- **The controls undercut the mechanism:** for A and B, *fixed* global weights score the same as
  lagged per-patch weights (+0.150 vs +0.150; +0.074 vs +0.073), and per-frame = per-patch. With
  the chosen decay (0.98, ~50-frame memory) plus the 30-frame lag, the "adaptive" weights barely
  move. What removes the still-frame loss is blending at all, not tracking recent accuracy.
- A is a more *reliable* predictor than any source (better than copy on 95% of frames vs 67% for
  nca10) at equal skill; B is the crisp, always-safe one at lower skill.
- **The main finding: the sources are strongly complementary per patch (oracle pick +0.250,
  nearly double), but which one wins must be known *now*, not 1 s later.** The next test is a
  fast proxy for "which source will win in this patch": current patch motion, the NCA's
  short-horizon (66-133 ms) errors, ensemble spread - fitted on half 1, scored on half 2.

**Caveats:** one clip; half 2 of one recording; the decay grid stopped at 0.98 (a fixed blend is
the limit of that direction); oracle uses the running mean of current errors, so it is an
optimistic but not perfect upper bound.

## E31 - Precision weighting for the NCA, phase 2: fixed blends in the live mirror (2026-11-01)

**Change:** `nca_mirror.py --out lineage` now shows a fixed blend of four sources by default
(`m` cycles): **blend** = E30 "A fixed" pixel blend (copy 0.229, newest NCA 0.262, 10-guess
average 0.270, rigid-superpixel NCA 0.240); **crisp** = E30 "B fixed" motion blend (newest 0.359 +
rigid 0.328 + copy's 0.313 as zero motion, one warp); **nca10** = the E23 10-guess average. The
status-line score follows the shown mix. Also fixed: the confidence view's ghost opacity reused
the name `alpha`, clobbering the E28 region dial on the next frame (would have raised on
`alpha > 0` with an array) - renamed.

**Setup:** replay of the E12 clip in a pseudo-terminal, status line logged, 50 s, cycling `m`,
then the confidence view with regions 0.25. The user's live lineage mirror shared the GPU.

**Result:** blend 74 ms/frame, crisp 72, nca10 55 (nca10 was 37 ms with the GPU free, E23, so
these are ~1.5x inflated); no errors over 657 frames including the confidence + regions path.
Blend/crisp cost ~20 ms more than nca10 (SLIC 8 ms + four extra full-res warps + region means).

**Conclusions:** the E30 blend is live as the default; live skill readings are the user's to
judge in the pane (3 s windows are noisy, E5). Expect ~20 fps live for blend until SLIC is
computed less often.

**Caveats:** timings with a shared GPU; no live skill measurement recorded.

## E32 - Precision weighting for the NCA, phase 1b: a gate from signals known now (2026-11-01)

**Question:** E30 showed the sources are complementary per patch (oracle pick +0.250) but that
accuracy known 1 s late is useless for choosing. Can a gate choose from signals available *now*?

**Setup:** `nca_mix.py gate` (`results/mix_gate.log`; first attempt `results/mix_gate_v1.log`;
features cached in `results/mix_features.npz`). Per 8x8 patch at time t, seven features: motion
now (|luma t - luma t-3|), motion before (t-3 vs t-6), copy's 66 ms error (frame t-2 vs t), the
NCA's 66 ms error (its step-0 prediction made at t-2 vs t), the NCA's predicted 1 s motion, the gap
between that and its superpixel average, and the spread of the 10 guesses; log, standardised on
half 1. Gate = softmax over the four E29 sources, fitted on half 1 (640k patches), scored on half 2
as a soft blend (upsampled weights) and a hard pick. Losses: **expected** error (mean sum w*err),
**relative** (errors divided by their per-patch mean, so still patches count equally), **xent**
(cross-entropy against the best source). Models: linear, MLP (32 hidden), MLP + 3x3 context of the
features; B = motion-blend gate over copy / nca / rigid. Ceiling: the true per-patch oracle pick.

**Result (half 2, 2902 targets):**

| variant | skill | q1 (still) | q2 | q3 | q4 | q5 | better than copy | sharpness |
|---|---|---|---|---|---|---|---|---|
| nca10 (source) | +0.150 | -0.275 | -0.048 | +0.129 | +0.181 | +0.174 | 67.3% | 0.903 |
| A fixed (E30) | +0.150 | +0.048 | +0.126 | +0.163 | +0.171 | +0.143 | 94.2% | 0.938 |
| true oracle pick | +0.293 | +0.194 | +0.274 | +0.311 | +0.326 | +0.277 | 100% | 0.925 |
| lin expected soft | +0.166 | -0.247 | -0.021 | +0.150 | +0.199 | +0.185 | 70.6% | 0.904 |
| lin relative soft | +0.133 | +0.025 | +0.096 | +0.136 | +0.152 | +0.131 | 91.7% | 0.953 |
| **lin xent soft** | **+0.174** | **+0.022** | +0.120 | +0.178 | +0.200 | +0.170 | 90.4% | 0.927 |
| lin xent pick | +0.157 | -0.162 | +0.044 | +0.157 | +0.186 | +0.164 | 75.1% | 0.911 |
| mlp relative soft | +0.146 | +0.026 | +0.100 | +0.147 | +0.166 | +0.145 | 91.0% | 0.949 |
| mlp+ctx relative soft | +0.150 | +0.026 | +0.102 | +0.150 | +0.168 | +0.150 | 90.9% | 0.948 |
| B mlp relative (motion) | +0.123 | +0.023 | +0.094 | +0.127 | +0.140 | +0.120 | 91.5% | 0.960 |

First attempt (v1, expected loss only): motion-only gate collapsed to "always nca10" (+0.150),
the B gate to "always nca" (+0.135), all-features soft +0.171 with q1 -0.081.

**Conclusions:**
- **Go: `lin xent soft` passes P1 and P2** - +0.174 overall (best so far, vs +0.153) *and*
  positive when still (+0.022), better than copying on 90% of frames, sharpness 0.927.
- The loss decides the regime: expected error chases motion and fails stillness; relative error
  protects stillness but under-exploits motion; cross-entropy with soft weights gets both.
- The soft blend beats the hard pick for the same gate (+0.174 vs +0.157): hedging between
  sources where the gate is unsure pays. The surrogate (sum w*err) underestimates soft blends
  badly (+0.114 vs real +0.174), so soft gates must be scored on real images.
- An MLP and spatial context add almost nothing (+0.004) with the relative loss; the features,
  not the model, are the limit. The gate recovers ~17% of the oracle headroom
  ((0.174 - 0.150) / (0.293 - 0.150)).
- B gate: +0.123, still +0.023, sharpness 0.960 - the crisp always-safe option, up from E30's +0.073.

**Caveats:** one clip; gate fitted on half 1 and scored on half 2 of the same session (phase 3,
a new room, is the real test); features from the model's own outputs (66 ms error, spread) are
cheap live but were not timed; xent + MLP not tried.


## E33 - Precision weighting for the NCA, phase 2b: the gate in the live mirror (2026-11-01)

**Change:** `nca_mirror.py --out lineage` default mix is now **gate** (E32's linear cross-entropy
gate, soft weights, `state/nca_gate.npz`); `m` cycles gate / blend / crisp / nca10. The seven
features are computed live at NCA resolution from what the mirror already holds: recent low-res
frames, the lineage model's step-0 flow from two frames ago (for the 66 ms NCA error), the newest
1 s flow and its superpixel average, and the 10 guesses (for the spread). Weights are upsampled
bilinearly over the 8x8 patches as in the offline evaluation. Also added `r` to record the camera
from the NCA mirror (640x360 mp4, same format as E12) for phase 3.

**Setup:** replay of the E12 clip in a pseudo-terminal, 42 s, `m` once; the user's live mirror
shared the GPU.

**Result:** gate 74 ms/frame, blend 73 (same conditions as E31's 74) - the features cost ~1 ms;
no errors over 539 frames; 3 s live-score snapshots gate +0.10, blend +0.12 (noise, E5).

**Conclusions:** the E32 gate runs live at no extra cost. Whether it holds up in a new room is
phase 3 (`nca_mix.py newroom`), which needs a recording made there.

**Caveats:** timings with a shared GPU.

**Addendum (same day):** the live gate weights (numpy path in `nca_mirror.gate_weights`) were
compared with the offline ones (`nca_mix.Scene.features_at` + `gate_soft`) on E12 frames 26500,
27400 and 28300: max |difference| 0.0043 per patch weight, mean 0.0018 (from cv2.remap vs
grid_sample interpolation). The live gate is the evaluated gate.

**Addendum 2 - phase-3 tool self-check:** `nca_mix.py newroom recordings/clip_20261030_135948.mp4
--start 26328 --seconds 200` (the E12 clip's held-out half 2, decoded and segmented afresh, nothing
refitted; `results/newroom_selfcheck.log`) reproduces E32/E30: gate +0.174 (q1 +0.022, 90.3%),
nca10 +0.150, nca +0.135, blend +0.150, crisp +0.074, rigid +0.050; 115 s for 1.6 min of video.

## E34 - Precision weighting for the NCA, phase 3: a new room, nothing refitted (2026-11-01)

**Question:** every result so far (E21-E33) comes from one recording in one room. Do the mixers'
rankings hold on a recording made in the user's current room, with nothing retrained or refitted?

**Part 1 - the first recording (confounded by frame rate).** `recordings/clip_20261101_104750.mp4`
was recorded from `nca_mirror.py`'s display loop (`r`, E33), which ran at ~10-13 fps while
recording: 3062 frames over ~4.5 min of wall time, written at 30 fps. So consecutive frames are
~100 ms apart, not 33: "30 frames ahead" is ~3 s of real time and the model's input frames
(0/3/6 steps back) span 600 ms instead of 200. `nca_mix.py newroom` on it
(`results/newroom_clip_20261101_104750.log`), 3008 targets:

| variant | skill | q1 (still) | q2 | q3 | q4 | q5 | better than copy | sharpness |
|---|---|---|---|---|---|---|---|---|
| nca | +0.108 | +0.015 | +0.100 | +0.113 | +0.110 | +0.120 | 88.3% | 0.878 |
| nca10 | +0.203 | -0.151 | +0.130 | +0.214 | +0.226 | +0.257 | 81.2% | 0.755 |
| rigid | +0.040 | +0.040 | +0.045 | +0.042 | +0.039 | +0.038 | 87.1% | 0.961 |
| blend (E30) | +0.169 | +0.156 | +0.162 | +0.171 | +0.169 | +0.172 | 96.9% | 0.850 |
| crisp (E30) | +0.066 | +0.083 | +0.071 | +0.067 | +0.062 | +0.063 | 94.6% | 0.938 |
| gate (E32) | +0.197 | +0.128 | +0.176 | +0.201 | +0.198 | +0.212 | 95.0% | 0.817 |

Provisional reading: the E32 pattern survives a new room, a new session and a 3x time-scale
shift - the gate keeps nearly all of nca10's skill (+0.197 vs +0.203) while nca10 again fails when
still (-0.151) and the gate does not (+0.128); the fixed blend is the most reliable (96.9%) at
lower skill. All skills are higher and all sharpness lower than E32 because 3x the motion per
frame makes copying a weaker baseline and rewards hedging. **Not a clean test**: the recorder was
changed to a camera-rate thread (`Recorder` in nca_mirror.py; fake-source test: 55 frames in 2 s
beside a 10 fps display loop) and part 2 below uses a re-recording.

**Part 2 - the 30 fps re-recording (the clean test).** `recordings/clip_20261101_110042.mp4`:
9039 frames in 302 s of wall time (30.0 fps, camera-rate `Recorder`), the user moving and
standing still in the new room. `nca_mix.py newroom` (`results/newroom_clip_20261101_110042.log`),
8985 targets, 232 s:

| variant | skill | q1 (still) | q2 | q3 | q4 | q5 | better than copy | sharpness |
|---|---|---|---|---|---|---|---|---|
| nca | +0.119 | -0.024 | +0.120 | +0.126 | +0.126 | +0.120 | 87.3% | 0.898 |
| nca10 | +0.140 | -0.325 | +0.058 | +0.134 | +0.155 | +0.183 | 71.2% | 0.812 |
| rigid | +0.049 | +0.040 | +0.064 | +0.052 | +0.049 | +0.045 | 86.7% | 0.973 |
| blend (E30) | +0.151 | +0.090 | +0.166 | +0.160 | +0.151 | +0.147 | 94.6% | 0.886 |
| crisp (E30) | +0.074 | +0.086 | +0.097 | +0.080 | +0.072 | +0.065 | 93.6% | 0.955 |
| **gate (E32)** | **+0.169** | **+0.046** | +0.171 | +0.180 | +0.171 | +0.171 | 91.0% | 0.857 |

Old room, held-out half 2 (E32), for comparison: gate +0.174 / q1 +0.022 / 90.4%; blend +0.150 /
+0.048 / 94.2%; nca10 +0.150 / -0.275 / 67.3%; nca +0.135 / -0.000; crisp +0.073 / +0.058.

**Conclusions (phase 3, and the E29-E34 strategy as a whole):**
- **The gate generalises.** In a new room, a new session and five minutes of unseen video, with the
  NCA, the gate and the fixed weights all untouched, every number lands within ~0.02 of the old
  held-out result and the ranking is identical: gate > blend > nca10 > nca > crisp > rigid. The
  gate is again the best overall (+0.169) and positive when still (+0.046); nca10 again collapses
  when still (-0.325).
- **Hypothesis verdict:** the original mechanism (weights from recent accuracy) is refuted by the
  1 s lag (E30); the amended mechanism (weights from signals known now) holds and transfers.
  Predictions: P1 yes (gate, blend, crisp), P2 yes (gate: +0.169 >= nca10's +0.140 here), P3 yes
  (crisp 0.955), P4 moot for the gate (no lag).
- The gate is the right live default; blend is the most reliable (94.6% of frames better than
  copying) when skill matters less than never failing; crisp is the sharp option.
- Headroom remains: the true per-patch oracle was +0.293 on the old clip (E32); the gate takes
  ~15-20% of that gap. Better features (not a bigger model, E32) are the lever.

**Caveats:** still one person; both rooms are the user's; the old-room gate was fitted on half
of one held-out stretch; sharpness is relative to the true frame, which differs between rooms;
the live mirror's real-time behaviour (frame drops, 20 fps in blend/gate mode) is not measured
by these offline runs.


## E35 - Richer gate features (2026-11-01)

**Question:** E32 concluded the gate was feature-limited. Do features describing how error evolves
and local structure raise it (hypothesis in Open hypotheses: >= +0.19 old room, >= +0.185 new
room, still >= 0, sharpness >= 0.85)?

**Setup:** `nca_mix.py gate2 recordings/clip_20261101_110042.mp4` (`results/mix_gate2.log`;
features and errors cached in `results/mix_features2_{old,new}.npz`). FEATURES2 = the E32 seven
plus **trend** (motion 6-9 and 9-12 frames back), **curve** (NCA 133 and 200 ms errors from
lineage steps 1-2 made at t-4 / t-6; copy 133 / 200 ms; rigid 66 ms), **lagged** (each source's
1 s error for the prediction made 30 frames ago, whose target has just arrived), **structure**
(patch edge energy, |divergence| of the predicted motion, spread of the 10 guesses' motion
vectors); **ctx** = 3x3 means of all. Gates fitted on old-room half 1 only (cross-entropy unless
stated), scored as real soft blends on old-room half 2 (2884 targets) and the whole new-room clip
(8955 targets). Also: MLP (32 hidden) with cross-entropy (E32 only tried the MLP with the relative
loss), a soft target (softmax of -err / mean err), and a smaller floor in the log transform (1e-6
instead of 1e-4, because the error features' medians, ~2e-5, sit below 1e-4).

**Result, part 1 (skill / q1 still / better than copy / sharpness):**

| gate | old room | new room |
|---|---|---|
| lin xent base (E32) | +0.174 / +0.022 / 90.4% / 0.927 | +0.169 / +0.046 / 91.0% / 0.857 |
| + trend | +0.175 / +0.021 | +0.170 / +0.046 |
| + curve | +0.175 / +0.021 | +0.169 / +0.045 |
| + lagged | +0.175 / +0.022 | +0.169 / +0.045 |
| + structure (all 21) | +0.176 / +0.019 | +0.170 / +0.041 |
| all + ctx (42) | +0.179 / +0.021 | +0.172 / +0.029 |
| **mlp32 xent, all 21** | **+0.189** / +0.015 / 90.3% / 0.921 | **+0.181** / +0.026 / 91.1% / 0.844 |
| lin soft target, all | +0.151 / +0.051 / 94.7% / 0.942 | +0.150 / +0.092 / 95.1% / 0.889 |
| lin xent base, floor 1e-6 | +0.168 / +0.017 | +0.164 / +0.059 |
| lin xent all, floor 1e-6 | +0.173 / +0.019 | +0.167 / +0.055 |
| mlp32 xent all, floor 1e-6 | +0.188 / +0.016 | +0.181 / +0.029 |

Pass 1: 142 s (old) + 319 s (new); linear fits 3-6 s, MLP 55 s.

**Reading (part 1):**
- **The features hypothesis is refuted:** for a linear gate, 14 extra features add +0.001 to
  +0.005; no group stands out; context +0.005 / +0.003. The log floor does not matter (slightly
  worse, if anything).
- **What helps is the MLP with cross-entropy:** +0.015 (old) and +0.012 (new) over the E32 gate,
  consistent across rooms, at a small cost in stillness (q1 +0.015 / +0.026, still >= 0) and
  sharpness (0.921 / 0.844 - the new-room value is just under the 0.85 target).
- Predictions not met: +0.189 < +0.19 and +0.181 < +0.185, both narrowly; still >= 0 held.
- The soft target trades skill for reliability (95% of frames better than copying; still +0.05 /
  +0.09): a third profile between the gate and the fixed blend.
- **Open:** whether the MLP's gain comes from nonlinearity alone or needs the extra features -
  "mlp32 xent base" was not run. Part 2 below.

**Result, part 2 - attributing the MLP's gain** (`results/mix_gate2_part2.log`, seeded fits;
skill / q1 still / better than copy / sharpness):

| gate | old room | new room |
|---|---|---|
| lin xent base (E32) | +0.174 / +0.022 / 90.4% / 0.927 | +0.169 / +0.048 / 91.1% / 0.857 |
| **mlp32 xent base (7 features)** | **+0.186** / +0.016 / 89.9% / 0.922 | **+0.179** / +0.026 / 90.7% / 0.844 |
| mlp32 xent all (21) | +0.189 / +0.015 / 90.2% / 0.921 | +0.181 / +0.027 / 91.2% / 0.844 |
| mlp64 xent base | +0.185 / +0.021 / 90.5% / 0.923 | +0.179 / +0.035 / 91.3% / 0.847 |
| mlp64 xent all | +0.190 / +0.014 / 90.2% / 0.920 | +0.182 / +0.021 / 91.1% / 0.842 |
| mlp32 xent base + ctx (14) | +0.187 / +0.021 / 90.7% / 0.922 | +0.181 / +0.027 / 91.0% / 0.845 |

**Conclusions:**
- **The gain is nonlinearity, not features.** MLP on the 7 E32 features: +0.012 (old) / +0.010
  (new) over the linear gate; the 14 extra features then add +0.003 / +0.002; width 64 adds
  ~+0.001; context +0.001 / +0.002. E32's "features are the limit" was wrong for the reason that
  E32 only paired the MLP with the relative loss, which caps everything near +0.15.
- **Shipped: mlp32 xent on the 7 base features** (`nca_mix.py save-gate` now saves the MLP;
  `nca_mirror.py` applies it) - same live features as E33, one 7->32->4 network per patch.
  Cost: stillness +0.016 / +0.026 instead of +0.022 / +0.048 (still >= 0), sharpness -0.005 /
  -0.013.
- The E35 predictions were not met (best +0.190 / +0.182 vs >= +0.19 / +0.185); the hypothesis
  as stated is refuted. Per-patch statistics of this kind appear to be near their ceiling for a
  gate (~+0.19 of the oracle's +0.293); what is left is probably not in per-patch statistics at
  all.

**Caveats:** one person, two rooms; the old-room score is in-session held-out (gate fitted on the
first half of the same stretch); sharpness in the new room is below the 0.85 target for every
MLP variant (0.842-0.847).

**Shipping check:** `save-gate` now writes the MLP (W1 32x7, W2 4x32); the E32 linear gate is kept
as `state/nca_gate_linear_E32.npz`. Live vs offline weights on E12 frames 26500 / 27400 / 28300:
max |difference| 0.043 (linear gate: 0.004 - the MLP amplifies the cv2.remap vs grid_sample
interpolation differences), means agree to 0.005. Replay test, GPU free: 58 ms/frame in gate mode,
571 frames, no errors.

