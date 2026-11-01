# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "opencv-python"]
# ///
"""Offline evaluation of the fused two-eye predictor on a recorded clip.

Runs the live pipeline (eyes -> per-eye V1 layers -> fused predictor) over every frame, as fast as
possible, with everything learning online from scratch. Each prediction is scored before the model
learns from its target (prequential), against the true frame `horizon` steps ahead (no delay, no
noise) at eye resolution. Baselines use only what the eyes see: copy the average of both eyes'
current images, or copy one eye. "precision" is the model blended per patch with the eyes'
average by recent precision (FusionPredictor.blend).

Results are also broken down by motion (how much the true frame changed over the horizon), in
quintiles from stillest to most motion.

  uv run evaluate.py recordings/clip.mp4 [--eyes both|left|right] [--horizon 3] [--rate 1.0] [--steps N]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import psychomanteum as pm

SEGMENT = 1800  # report per minute of video at 30 fps
NAMES = ("model", "precision", "copy_avg", "copy_left", "copy_right", "eyeflow", "mix3", "mixN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--eyes", choices=["both", "left", "right"], default="both")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--rate", type=float, default=1.0, help="fusion learning rate")
    parser.add_argument("--reach", type=float, help="steps to push the eye average along its motion (default horizon + 2)")
    parser.add_argument("--fractions", type=float, nargs="+", default=[1.0],
                        help="mixN: one motion-compensated source per fraction of reach")
    parser.add_argument("--flow-smoothing", type=float, default=0.0, help="mixN: running average of the flow (0 = off)")
    parser.add_argument("--blur", type=float, default=0.0, help="mixN: add the eyes' average blurred by this sigma (0 = off)")
    parser.add_argument("--sharpness", type=float, default=1.0, help="mixN: trust weights ~ precision ** sharpness")
    parser.add_argument("--no-model", action="store_true", help="mixN: leave the learned model out")
    parser.add_argument("--clean", action="store_true", help="eyes without noise or delay (E19)")
    parser.add_argument("--steps", type=int, default=0, help="stop after this many frames (0 = whole clip)")
    parser.add_argument("--out", type=Path, help="write results JSON here")
    args = parser.parse_args()

    params = pm.sanitize({"horizon": args.horizon, "fusion_rate": args.rate}
                         | ({"eye_noise": 0, "eye_max_delay": 0, "eye_jitter": 0} if args.clean else {}))
    use = {"both": (0, 1), "left": (0,), "right": (1,)}[args.eyes]
    rng = np.random.default_rng(1)
    eyes, coders, fusion = pm.Eyes(), [pm.SparseCoder.fresh(rng), pm.SparseCoder.fresh(rng)], pm.FusionPredictor(use)
    eyeflow, mixer, mixer_n = pm.EyeFlow(), pm.PrecisionMixer(), pm.PrecisionMixer(sharpness=args.sharpness)
    eyeflow_n = pm.EyeFlow()  # own flow state for mixN
    reach = args.reach if args.reach is not None else args.horizon + 2

    cap = cv2.VideoCapture(str(args.clip))
    pending = {}  # target step -> (inputs, truth at prediction time, model, precision, copy-avg, copy-L, copy-R)
    records = []  # per scored frame: motion, then MSE for each of NAMES
    step, start = 0, time.monotonic()
    while not args.steps or step < args.steps:
        ok, bgr = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        views = eyes.see(frame, params)
        for coder, view in zip(coders, views):
            coder.step(view, params)

        # the truth this step, at eye resolution: score and learn from predictions made `horizon` ago
        eye_size = views[0].shape[1], views[0].shape[0]
        truth = pm.patchify(cv2.resize(frame, eye_size, interpolation=cv2.INTER_AREA).astype(np.float32) @ pm.LUMA / 255)
        if step in pending:
            x, truth_then, guesses, own_copy, sources_n = pending.pop(step)
            motion = float(((truth - truth_then) ** 2).mean())
            records.append([motion] + [float(((g - truth) ** 2).mean()) for g in guesses])
            model, flowed = guesses[0], guesses[5]
            fusion.track_precision(model, own_copy, truth)
            mixer.track([model, flowed, own_copy], truth)
            mixer_n.track(sources_n, truth)
            fusion.learn(x, truth, params["fusion_rate"])

        x = fusion.inputs(coders)
        if x is not None:
            raw = [c.raw for c in coders]
            copy_avg = sum(raw[e] for e in use) / len(use)
            model = fusion.predict(x)
            avg_img = pm.unpatchify(copy_avg, views[0].shape[:2])
            flowed = pm.patchify(eyeflow.predict(avg_img, reach))
            stabilized = eyeflow_n.sources(avg_img, reach, args.fractions, args.flow_smoothing, args.blur)
            sources_n = ([] if args.no_model else [model]) + [pm.patchify(s) for s in stabilized] + [copy_avg]
            guesses = [model, fusion.blend(model, copy_avg), (raw[0] + raw[1]) / 2, raw[0], raw[1], flowed,
                       mixer.blend([model, flowed, copy_avg]), mixer_n.blend(sources_n)]
            pending[step + args.horizon] = (x, truth, guesses, copy_avg, sources_n)

        step += 1
        if step % SEGMENT == 0 and records:
            seg = np.array(records[-SEGMENT:])
            print(f"min {step // SEGMENT:2}: skill vs copy-avg  model {1 - seg[:, 1].sum() / seg[:, 3].sum():+.3f}  "
                  f"precision {1 - seg[:, 2].sum() / seg[:, 3].sum():+.3f}  mix3 {1 - seg[:, 7].sum() / seg[:, 3].sum():+.3f}  "
                  f"mixN {1 - seg[:, 8].sum() / seg[:, 3].sum():+.3f}", flush=True)

    r = np.array(records)
    mse = {name: r[:, i + 1] for i, name in enumerate(NAMES)}

    def skill(name, idx=slice(None)):
        return float(1 - mse[name][idx].sum() / mse["copy_avg"][idx].sum())

    print(f"\n{args.eyes} eyes{' (clean)' if args.clean else ''}, horizon {args.horizon}, rate {args.rate}, {step} frames in {time.monotonic() - start:.0f}s")
    print(f"MSE x1000: " + " | ".join(f"{n} {mse[n].mean() * 1e3:.3f}" for n in NAMES))
    print(f"skill vs copy-avg (summed MSE): model {skill('model'):+.3f}  precision {skill('precision'):+.3f}  "
          f"eyeflow {skill('eyeflow'):+.3f}  mix3 {skill('mix3'):+.3f}  mixN {skill('mixN'):+.3f}  (reach {reach})")
    print(f"mixN: fractions {args.fractions}, flow smoothing {args.flow_smoothing}, blur {args.blur}, sharpness {args.sharpness}{', no model' if args.no_model else ''}; "
          f"mixN better than mix3 on {(mse['mixN'] < mse['mix3']).mean():.1%} of frames")
    halves = (slice(None, len(r) // 2), slice(len(r) // 2, None))  # tune on the first, check on the second
    print("skill by half (first | second): " + "  ".join(
        f"{n} {skill(n, halves[0]):+.3f} | {skill(n, halves[1]):+.3f}" for n in ("mix3", "mixN")))
    print(f"frames where better than copy-avg: model {(mse['model'] < mse['copy_avg']).mean():.1%}  "
          f"precision {(mse['precision'] < mse['copy_avg']).mean():.1%}  mix3 {(mse['mix3'] < mse['copy_avg']).mean():.1%}")

    edges = np.percentile(r[:, 0], [20, 40, 60, 80])
    bins = np.digitize(r[:, 0], edges)
    by_motion = []
    print("\nby motion quintile (1 = stillest):")
    for b in range(5):
        idx = bins == b
        row = {"quintile": b + 1, "motion_mse_x1000": float(r[idx, 0].mean() * 1e3),
               "model": skill("model", idx), "precision": skill("precision", idx),
               "eyeflow": skill("eyeflow", idx), "mix3": skill("mix3", idx), "mixN": skill("mixN", idx)}
        by_motion.append(row)
        print(f"  {b + 1}: motion {row['motion_mse_x1000']:7.3f}  model {row['model']:+.3f}  precision {row['precision']:+.3f}  "
              f"eyeflow {row['eyeflow']:+.3f}  mix3 {row['mix3']:+.3f}  mixN {row['mixN']:+.3f}")

    if args.out:
        args.out.write_text(json.dumps({
            "args": {k: str(v) for k, v in vars(args).items()}, "frames": step,
            "mse_x1000": {n: float(mse[n].mean() * 1e3) for n in NAMES},
            "skill": {n: skill(n) for n in ("model", "precision", "eyeflow", "mix3", "mixN")}, "reach": reach,
            "skill_by_half": {n: [skill(n, h) for h in halves] for n in ("mix3", "mixN")},
            "by_motion": by_motion}, indent=2))


if __name__ == "__main__":
    main()
