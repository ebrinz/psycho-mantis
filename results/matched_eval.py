"""E36: the eye system vs the lineage automaton on the same held-out frames, grey 160x90, same
copy-the-current-frame baseline, matched horizons (133 / 533 / 1000 ms = 4 / 16 / 30 frames).

  uv run results/matched_eval.py eye --horizon 16 --blur 6     # eye system, clean eyes at 160x90, online from frame 0
  uv run results/matched_eval.py nca                           # lineage newest guess, + the eye system's precision
                                                               # mixer with copy, and the live gate at 1 s
  uv run results/matched_eval.py report                        # tables from the saved runs

Scores are MSE of luma over whole 8x8 patches (88 of 90 rows), recorded per prediction time t for
t >= T0 (the held-out stretch) - identical pixels and frames for both systems.
"""
# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "opencv-python", "torch", "scikit-image"]
# ///
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

import psychomanteum as pm

import os
T0 = int(os.environ.get("MATCHED_T0", 23451))  # first held-out prediction time with a full lineage ensemble (as nca_mix)
HORIZONS = {4: 1, 16: 7, 30: 14}  # frames ahead -> lineage read-out index (step k predicts 2k frames)
OUT = ROOT / "results" / "matched"


def luma_patches(rgb01):
    """(H, W, 3) float 0-1 -> patches of luma (n, 64), the scoring unit."""
    return pm.patchify(rgb01 @ pm.LUMA)


def eye(horizon, blur, steps=0):
    params = pm.sanitize({"eye_width": 160, "eye_noise": 0, "eye_max_delay": 0, "eye_jitter": 0, "eyeflow_smoothing": 0.0})
    params["horizon"] = horizon  # 30 is above the live clamp; the pipeline itself has no limit
    rng = np.random.default_rng(1)
    eyes, coders, fusion = pm.Eyes(), [pm.SparseCoder.fresh(rng), pm.SparseCoder.fresh(rng)], pm.FusionPredictor()
    eyeflow, mixer = pm.EyeFlow(), pm.PrecisionMixer()
    reach = horizon + params["eyeflow_lead"]
    cap = cv2.VideoCapture(str(ROOT / "recordings" / "clip_20261030_135948.mp4"))
    pending, rec = {}, []
    step = 0
    while not steps or step < steps:
        ok, bgr = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        views = eyes.see(frame, params)
        for c, v in zip(coders, views):
            c.step(v, params)
        truth = luma_patches(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
        if step in pending:
            x, sources, shown, copy = pending.pop(step)
            mixer.track(sources, truth)
            fusion.learn(x, truth, params["fusion_rate"])
            if step - horizon >= T0:
                rec.append((step - horizon, ((shown - truth) ** 2).mean(), ((copy - truth) ** 2).mean()))
        x = fusion.inputs(coders)
        if x is not None:
            copy = (coders[0].raw + coders[1].raw) / 2  # clean, undelayed: the current frame
            avg_img = pm.unpatchify(copy, (90, 160))
            stabilized = eyeflow.sources(avg_img, reach, smoothing=params["eyeflow_smoothing"], blur=blur)
            sources = [fusion.predict(x)] + [pm.patchify(s) for s in stabilized] + [copy]
            pending[step + horizon] = (x, sources, mixer.blend(sources), copy)
        step += 1
    r = np.array(rec)
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / f"eye_h{horizon}_b{blur:g}.npz", t=r[:, 0], mse=r[:, 1], copy=r[:, 2])
    print(f"eye h{horizon} blur {blur:g}: {len(r)} scored, skill {1 - r[:, 1].sum() / r[:, 2].sum():+.3f}")


def nca_runs():
    import torch
    import graph_nca
    import nca
    import nca_mix
    frames = np.load(nca.FRAMES)
    labels = np.load(graph_nca.LABELS, mmap_mode="r")
    model = nca.load("lineage")
    n = len(frames)
    ts = np.arange(T0, n - 30)
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(nca.DEVICE).permute(0, 3, 1, 2).float() / 255
    lum = lambda img: (img.permute(0, 2, 3, 1).cpu().numpy() @ pm.LUMA)  # (B, H, W)
    truth_cache = {}

    def truth_at(t):
        if t not in truth_cache:
            truth_cache[t] = pm.patchify(frames[t].astype(np.float32) / 255 @ pm.LUMA)
        return truth_cache[t]

    rec = {h: [] for h in HORIZONS}  # (t, nca mse, nca+copy mse, copy mse)
    mixers = {h: pm.PrecisionMixer() for h in HORIZONS}
    pending = {h: {} for h in HORIZONS}
    with torch.no_grad():
        for i in range(0, len(ts), 64):
            b = ts[i:i + 64]
            x, _ = nca.batch(frames, b)
            out = model(x)
            cur = x[:, :3]
            preds = {h: lum(nca.warp(cur, out[:, idx])) for h, idx in HORIZONS.items()}
            cur_l = lum(cur)
            for j, t in enumerate(b):
                copy_p = pm.patchify(cur_l[j])
                for h in HORIZONS:
                    # settle predictions whose targets have arrived (sequential in t, so pending keys <= t are due)
                    if t in pending[h]:
                        sources, shown, tp = pending[h].pop(t)
                        truth = truth_at(t)
                        mixers[h].track(sources, truth)
                        rec[h].append((tp, ((sources[0] - truth) ** 2).mean(), ((shown - truth) ** 2).mean(), ((sources[1] - truth) ** 2).mean()))
                    sources = [pm.patchify(preds[h][j]), copy_p]
                    pending[h][t + h] = (sources, mixers[h].blend(sources), t)
        # flush: targets beyond the last prediction time but inside the clip
        for h in HORIZONS:
            for t in sorted(pending[h]):
                if t < n:
                    sources, shown, tp = pending[h][t]
                    truth = truth_at(t)
                    rec[h].append((tp, ((sources[0] - truth) ** 2).mean(), ((shown - truth) ** 2).mean(), ((sources[1] - truth) ** 2).mean()))
        OUT.mkdir(exist_ok=True)
        for h in HORIZONS:
            r = np.array(rec[h])
            np.savez(OUT / f"nca_h{h}.npz", t=r[:, 0], mse=r[:, 1], mix=r[:, 2], copy=r[:, 3])
            print(f"nca h{h}: {len(r)} scored, newest {1 - r[:, 1].sum() / r[:, 3].sum():+.3f}, +copy mixer {1 - r[:, 2].sum() / r[:, 3].sum():+.3f}")
        # the live gate at 1 s
        scene = nca_mix.Scene(frames, labels, T0 - 18, int(ts[-1]))
        gate = nca_mix.load_gate()
        g = []
        for i in range(0, len(ts), 64):
            b = ts[i:i + 64]
            cur, imgs, _, _, _, _ = scene.sources_at(b)
            w = nca_mix.upsample_weights(nca_mix.gate_soft(scene.features_at(b), gate))[:, :, None]
            pred = lum((w * imgs).sum(1))
            for j, t in enumerate(b):
                truth = truth_at(t + 30)
                g.append((t, ((pm.patchify(pred[j]) - truth) ** 2).mean()))
        r = np.array(g)
        np.savez(OUT / "gate_h30.npz", t=r[:, 0], mse=r[:, 1])
        print(f"gate h30: {len(r)} scored")


def report():
    import glob
    print("E36 matched test: held-out predictions from frame", T0, "- grey luma, 160x90, 88 rows, skill vs copying the current frame")
    for h, ms in ((4, 133), (16, 533), (30, 1000)):
        nc = np.load(OUT / f"nca_h{h}.npz")
        t, copy = nc["t"], nc["copy"]
        rows = {"automaton, newest guess": nc["mse"], "automaton + copy, online precision mixer": nc["mix"]}
        if h == 30:
            gt = np.load(OUT / "gate_h30.npz")
            rows["automaton, live gate (E35)"] = dict(zip(gt["t"], gt["mse"]))
        eyes_runs = {}
        for f in sorted(glob.glob(str(OUT / f"eye_h{h}_b*.npz"))):
            e = np.load(f)
            eyes_runs[Path(f).stem.split("_b")[1]] = dict(zip(e["t"], e["mse"]))
        # align everything on the common prediction times
        common = set(t.tolist())
        for d in list(eyes_runs.values()) + [v for v in rows.values() if isinstance(v, dict)]:
            common &= set(d.keys())
        common = np.array(sorted(common))
        sel = np.isin(t, common)
        t, copy = t[sel], copy[sel]
        al = lambda v: v[sel] if isinstance(v, np.ndarray) else np.array([v[tt] for tt in t])
        rows = {k: al(v) for k, v in rows.items()}
        half = np.arange(len(t)) >= len(t) // 2
        sk = lambda m, mask=slice(None): 1 - m[mask].sum() / copy[mask].sum()
        if eyes_runs:
            best = max(eyes_runs, key=lambda k: sk(al(eyes_runs[k]), ~half))
            for k, v in eyes_runs.items():
                rows[f"eye system, blur {k}{' (chosen on half 1)' if k == best else ''}"] = al(v)
        bins = np.digitize(copy, np.percentile(copy[half], [20, 40, 60, 80]))
        print(f"\n--- {ms} ms ahead ({h} frames): {len(t)} predictions, half 2 = {half.sum()}")
        print(f"{'system':46} {'half 1':>7} {'half 2':>7}   {'q1 still':>8} {'q2':>7} {'q3':>7} {'q4':>7} {'q5':>7}   {'better':>6}")
        for k, m in rows.items():
            q = " ".join(f"{sk(m, half & (bins == i)):+7.3f}" for i in range(5))
            print(f"{k:46} {sk(m, ~half):+7.3f} {sk(m, half):+7.3f}   {q.replace(' ', '', 0):>8}   {(m[half] < copy[half]).mean():6.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eye")
    e.add_argument("--horizon", type=int, required=True)
    e.add_argument("--blur", type=float, default=6)
    e.add_argument("--steps", type=int, default=0)
    sub.add_parser("nca")
    sub.add_parser("report")
    a = ap.parse_args()
    {"eye": lambda: eye(a.horizon, a.blur, a.steps), "nca": nca_runs, "report": report}[a.cmd]()
