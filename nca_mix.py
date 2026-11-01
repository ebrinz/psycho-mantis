# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "opencv-python", "torch", "scikit-image"]
# ///
"""Precision weighting for the lineage NCA at 1 s ahead (E29+), evaluated non-stop over the
held-out video (every frame, in order, as precision tracking needs).

`baseline` (phase 0): for every held-out frame t, four predictions of frame t + HORIZON at 160x90:
  copy   - frame t unchanged (zero motion)
  nca    - the lineage NCA's newest 1 s motion field applied to frame t (E23, step 15)
  nca10  - the average of the 1 s predictions made from frames t, t-2, ... t-18 (E23)
  rigid  - the newest motion averaged within each superpixel of frame t (E28, alpha = 1)
Reports skill vs copying (overall, by motion quintile, by held-out half) and sharpness (mean
Sobel gradient magnitude of the prediction / of the true future frame; 1 = as crisp as reality).
Saves per-frame, per-8x8-patch squared errors for the phase-1 mixers (results/mix_baseline.npz).

`mix` (phase 1): weight the sources per 8x8 patch by precision (1 / running squared error). A
prediction made at frame t may only use errors of predictions whose target has arrived, i.e. made
at t - HORIZON or earlier (the 1 s lag). Variants:
  A  pixel blend of all four sources (weight maps upsampled bilinearly, so no patch seams)
  B  motion blend of copy (zero motion), nca and rigid: one blended motion field, one warp (crisp)
  C  hard pick: each patch takes its most precise source (all four)
Controls: oracle (same running average, no lag), per-frame (one weight set per frame), fixed
(global weights from half 1). The decay is chosen on half 1 of the targets; half 2 is reported.

`gate` (phase 1b, after E30 showed the 1 s accuracy lag defeats adaptive weights but an oracle
pick would nearly double skill): predict which source will win in each patch from signals known
*now* - current and recent patch motion, copy's and the NCA's 66 ms errors (targets already
arrived), the NCA's predicted motion, its disagreement with the rigid version, and the spread of
the 10 guesses. A softmax gate (multinomial logistic regression) is fitted on half 1 to minimise
the expected patch error, and evaluated on half 2 as a soft blend and as a hard pick, next to the
true per-patch oracle pick (the ceiling for any gate).

  uv run nca_mix.py baseline
  uv run nca_mix.py mix
  uv run nca_mix.py gate
  uv run nca_mix.py save-gate      fit the E32 winner on half 1 and save it for the mirror (state/nca_gate.npz)
  uv run nca_mix.py newroom recordings/clip.mp4 [--start F] [--seconds S]
                                   phase 3: score copy / nca / nca10 / rigid / blend / crisp / gate on a new
                                   clip, nothing refitted
  uv run nca_mix.py gate2 recordings/newroom.mp4
                                   E35: richer gate features in groups (trend, error curve, last known 1 s
                                   error, structure, context), fitted on old-room half 1, scored on old-room
                                   half 2 and the new-room clip
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import graph_nca
import nca

SOURCES = ("copy", "nca", "nca10", "rigid")
ENSEMBLE = 10  # source frames in nca10 (every LINEAGE_STRIDE frames back), as E23
PATCH = 8
OUT = nca.ROOT / "results" / "mix_baseline.npz"


def lineage_flows(model, frames, first, last, steps):
    """The lineage model's motion fields at the given inner `steps` for frames first..last:
    float16 (N, len(steps), 2, H, W) on the CPU. With steps = k1 .. k1 + ENSEMBLE - 1, entry j of
    frame t - LINEAGE_STRIDE * j is the prediction of frame t + HORIZON made from that earlier frame."""
    out = np.zeros((last - first + 1, len(steps), 2, nca.H, nca.W), np.float16)
    with torch.no_grad():
        for i in range(first, last + 1, 64):
            b = np.arange(i, min(i + 64, last + 1))
            x, _ = nca.batch(frames, b)
            out[b - first] = model(x)[:, list(steps)].cpu().numpy().astype(np.float16)
    return out


def region_mean(flow, labels):
    """flow (B, 2, H, W) averaged within each region of labels (B, H, W)."""
    b, n = len(labels), int(labels.max()) + 1
    flat = (labels + n * torch.arange(b, device=labels.device)[:, None, None]).reshape(-1)
    count = torch.zeros(b * n, device=flow.device).index_add_(0, flat, torch.ones_like(flat, dtype=flow.dtype))
    means = [torch.zeros(b * n, device=flow.device).index_add_(0, flat, flow[:, c].reshape(-1)) / count.clamp(min=1)
             for c in range(2)]
    return torch.stack(means, 1)[flat].reshape(b, *labels.shape[1:], 2).permute(0, 3, 1, 2)


def sobel_energy(img):
    """Mean gradient magnitude of the luma, per image: (B,)."""
    luma = (img * torch.tensor([0.299, 0.587, 0.114], device=img.device)[None, :, None, None]).sum(1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device)[None, None] / 8
    g = F.pad(luma, (1, 1, 1, 1), mode="replicate")
    return torch.sqrt(F.conv2d(g, kx) ** 2 + F.conv2d(g, kx.transpose(2, 3)) ** 2 + 1e-12).mean((1, 2, 3))


def patch_sq(pred, y):
    """Squared error per 8x8 patch: (B, H // 8, W // 8)."""
    h = (pred.shape[2] // PATCH) * PATCH
    return F.avg_pool2d(((pred - y) ** 2).mean(1, keepdim=True)[:, :, :h], PATCH)[:, 0]


def baseline():
    frames = np.load(nca.FRAMES)
    labels = np.load(graph_nca.LABELS, mmap_mode="r")
    _, test_idx = nca.split(len(frames))
    oldest = nca.LINEAGE_STRIDE * (ENSEMBLE - 1)
    ts = test_idx[test_idx >= test_idx[0] + oldest]  # every held-out frame with a full ensemble
    model, k1 = nca.load("lineage"), nca.lineage_step(nca.HORIZON)
    start = time.monotonic()
    first = int(ts[0]) - oldest
    flows = lineage_flows(model, frames, first, int(ts[-1]), range(k1, k1 + ENSEMBLE))
    print(f"flows for {len(flows)} frames in {time.monotonic() - start:.0f}s")

    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(nca.DEVICE).permute(0, 3, 1, 2).float() / 255
    mse, sharp, patches = [], [], []
    with torch.no_grad():
        for i in range(0, len(ts), 64):
            b = ts[i:i + 64]
            cur, y = to(frames[b]), to(frames[b + nca.HORIZON])
            fl = lambda t, j: torch.from_numpy(flows[t - first, j].astype(np.float32)).to(nca.DEVICE)
            newest = fl(b, 0)
            preds = {"copy": cur, "nca": nca.warp(cur, newest)}
            preds["nca10"] = torch.stack([nca.warp(to(frames[b - nca.LINEAGE_STRIDE * j]), fl(b - nca.LINEAGE_STRIDE * j, j))
                                          for j in range(ENSEMBLE)]).mean(0)
            lab = torch.from_numpy(labels[b].astype(np.int64)).to(nca.DEVICE)
            preds["rigid"] = nca.warp(cur, region_mean(newest, lab))
            truth_sharp = sobel_energy(y)
            mse.append(torch.stack([((preds[s] - y) ** 2).mean((1, 2, 3)) for s in SOURCES], 1).cpu().numpy())
            sharp.append(torch.stack([sobel_energy(preds[s]) / truth_sharp for s in SOURCES], 1).cpu().numpy())
            patches.append(torch.stack([patch_sq(preds[s], y) for s in SOURCES], 1).cpu().numpy().astype(np.float16))
    mse, sharp, patches = np.concatenate(mse), np.concatenate(sharp), np.concatenate(patches)
    np.savez_compressed(OUT, t=ts, sources=np.array(SOURCES), mse=mse, sharp=sharp, patch_sq=patches)

    copy = mse[:, 0]
    sk = lambda col, m=slice(None): 1 - mse[m, col].sum() / copy[m].sum()
    half = np.arange(len(ts)) < len(ts) // 2
    bins = np.digitize(copy, np.percentile(copy, [20, 40, 60, 80]))
    print(f"held-out, every frame: {len(ts)} targets ({len(ts) / 30 / 60:.1f} min), 1 s ahead, 160x90; "
          f"{time.monotonic() - start:.0f}s total")
    print(f"{'source':8} {'skill':>7} {'half 1':>7} {'half 2':>7}   {'q1 (still)':>10} {'q2':>7} {'q3':>7} {'q4':>7} {'q5':>7}   "
          f"{'better':>6}  {'sharpness':>9}")
    for c, s in enumerate(SOURCES):
        quint = f"{sk(c, bins == 0):+10.3f} " + " ".join(f"{sk(c, bins == q):+7.3f}" for q in range(1, 5))
        print(f"{s:8} {sk(c):+7.3f} {sk(c, half):+7.3f} {sk(c, ~half):+7.3f}   {quint}   "
              f"{(mse[:, c] < copy).mean():6.1%}  {np.median(sharp[:, c]):9.3f}")
    print(f"saved {OUT.relative_to(nca.ROOT)}: per-frame MSE, sharpness and per-patch squared errors {patches.shape}")


DECAYS = (0.8, 0.9, 0.95, 0.98)
B_SOURCES = (0, 1, 3)  # copy, nca, rigid: the sources with a single motion field from frame t


def running(err, decay, lag):
    """Running mean of err (N, S, P, Q) over time, where entry i only sees err[:i - lag + 1].
    Before anything is known: ones (equal weights)."""
    out, s = np.ones_like(err), None
    for i in range(len(err)):
        if i - lag >= 0:
            e = err[i - lag]
            s = e.copy() if s is None else decay * s + (1 - decay) * e
            out[i] = s
    return out


def weights(sq, subset=None):
    """Precision weights per source from running squared errors (N, S, P, Q), normalised over the
    sources (or over `subset`, zero elsewhere)."""
    prec = 1 / (sq + 1e-9)
    if subset is not None:
        keep = np.zeros(prec.shape[1], bool)
        keep[list(subset)] = True
        prec = prec * keep[None, :, None, None]
    return prec / prec.sum(1, keepdims=True)


def mix():
    base = np.load(OUT)
    ts, err = base["t"], base["patch_sq"].astype(np.float32)  # (N, S, P, Q)
    n, half = len(ts), np.arange(len(ts)) >= len(ts) // 2  # half: True = half 2 (reported)
    lag = nca.HORIZON
    combos = {}  # name -> weights (N, S, P, Q) (for C: one-hot of the pick)
    for d in DECAYS:
        lagged, oracle = running(err, d, lag), running(err, d, 0)
        per_frame = np.broadcast_to(lagged.mean((2, 3), keepdims=True), lagged.shape)
        for mode, sq in (("lagged", lagged), ("oracle", oracle), ("per-frame", per_frame)):
            combos[f"A {mode} {d}"] = weights(sq)
            combos[f"B {mode} {d}"] = weights(sq, B_SOURCES)
            pick = np.zeros_like(sq)
            np.put_along_axis(pick, sq.argmin(1)[:, None], 1.0, axis=1)
            combos[f"C {mode} {d}"] = pick
    fixed = 1 / err[~half].mean((0, 2, 3))  # global precision of each source on half 1
    combos["A fixed"] = np.broadcast_to((fixed / fixed.sum())[None, :, None, None], err.shape)
    fb = np.zeros_like(fixed)
    fb[list(B_SOURCES)] = fixed[list(B_SOURCES)]
    combos["B fixed"] = np.broadcast_to((fb / fb.sum())[None, :, None, None], err.shape)
    names = list(combos)

    frames = np.load(nca.FRAMES)
    labels = np.load(graph_nca.LABELS, mmap_mode="r")
    model, k1 = nca.load("lineage"), nca.lineage_step(nca.HORIZON)
    oldest = nca.LINEAGE_STRIDE * (ENSEMBLE - 1)
    first = int(ts[0]) - oldest
    start = time.monotonic()
    flows = lineage_flows(model, frames, first, int(ts[-1]), range(k1, k1 + ENSEMBLE))
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(nca.DEVICE).permute(0, 3, 1, 2).float() / 255
    upsample = upsample_weights

    mse = {k: [] for k in names}
    sharp = {k: [] for k in names}
    copy_mse = []
    with torch.no_grad():
        for i in range(0, n, 64):
            sl = slice(i, min(i + 64, n))
            b = ts[sl]
            cur, y = to(frames[b]), to(frames[b + nca.HORIZON])
            fl = lambda t, j: torch.from_numpy(flows[t - first, j].astype(np.float32)).to(nca.DEVICE)
            newest = fl(b, 0)
            rigid_flow = region_mean(newest, torch.from_numpy(labels[b].astype(np.int64)).to(nca.DEVICE))
            imgs = torch.stack([cur, nca.warp(cur, newest),
                                torch.stack([nca.warp(to(frames[b - nca.LINEAGE_STRIDE * j]), fl(b - nca.LINEAGE_STRIDE * j, j))
                                             for j in range(ENSEMBLE)]).mean(0),
                                nca.warp(cur, rigid_flow)], 1)  # (B, S, 3, H, W) in SOURCES order
            motion = torch.stack([torch.zeros_like(newest), newest, torch.zeros_like(newest), rigid_flow], 1)  # (B, S, 2, H, W)
            truth_sharp = sobel_energy(y)
            copy_mse.append(((cur - y) ** 2).mean((1, 2, 3)).cpu().numpy())
            for k in names:
                w = upsample(combos[k][sl])[:, :, None]  # (B, S, 1, H, W)
                if k.startswith("B"):
                    pred = nca.warp(cur, (w * motion).sum(1))
                else:
                    pred = (w * imgs).sum(1)
                mse[k].append(((pred - y) ** 2).mean((1, 2, 3)).cpu().numpy())
                sharp[k].append((sobel_energy(pred) / truth_sharp).cpu().numpy())
    copy_mse = np.concatenate(copy_mse)
    mse = {k: np.concatenate(v) for k, v in mse.items()}
    sharp = {k: np.concatenate(v) for k, v in sharp.items()}
    np.savez_compressed(nca.ROOT / "results" / "mix_phase1.npz", t=ts, copy=copy_mse, **{k.replace(" ", "_"): v for k, v in mse.items()})

    sk = lambda k, m: 1 - mse[k][m].sum() / copy_mse[m].sum()
    bins = np.digitize(copy_mse, np.percentile(copy_mse[half], [20, 40, 60, 80]))  # quintiles of half 2
    print(f"phase 1: {n} targets, decay chosen on half 1 ({(~half).sum()}), reported on half 2 ({half.sum()}); "
          f"{time.monotonic() - start:.0f}s")
    print(f"{'variant':26} {'half 1':>7} {'half 2':>7}   {'q1 (still)':>10} {'q2':>7} {'q3':>7} {'q4':>7} {'q5':>7}   {'better':>6}  {'sharp':>6}")

    def row(k):
        quint = f"{sk(k, half & (bins == 0)):+10.3f} " + " ".join(f"{sk(k, half & (bins == q)):+7.3f}" for q in range(1, 5))
        print(f"{k:26} {sk(k, ~half):+7.3f} {sk(k, half):+7.3f}   {quint}   "
              f"{(mse[k][half] < copy_mse[half]).mean():6.1%}  {np.median(sharp[k][half]):6.3f}")

    base_mse = base["mse"]
    for c, src in enumerate(SOURCES[1:], 1):  # reference rows from phase 0, same halves and quintiles
        mse[f"(source) {src}"] = base_mse[:, c]
        sharp[f"(source) {src}"] = base["sharp"][:, c]
        row(f"(source) {src}")
    for v in "ABC":
        best = max(DECAYS, key=lambda d: sk(f"{v} lagged {d}", ~half))
        print(f"-- {v}: decay {best} (best on half 1)")
        for mode in ("lagged", "oracle", "per-frame"):
            row(f"{v} {mode} {best}")
        if f"{v} fixed" in mse:
            row(f"{v} fixed")


def upsample_weights(w):
    """Patch weights (B, S, P, Q) -> per-pixel (B, S, H, W): bilinear, bottom rows replicated."""
    h8 = (nca.H // PATCH) * PATCH
    w = F.interpolate(torch.from_numpy(np.ascontiguousarray(w)).to(nca.DEVICE), size=(h8, nca.W), mode="bilinear", align_corners=False)
    return F.pad(w, (0, 0, 0, nca.H - h8), mode="replicate")


def pool(x):
    """Per-patch mean of (B, H, W) or (B, 1, H, W): (B, P, Q)."""
    x = x if x.dim() == 4 else x[:, None]
    return F.avg_pool2d(x[:, :, :(x.shape[2] // PATCH) * PATCH], PATCH)[:, 0]


FEATURES = ("motion now", "motion before", "copy 66ms err", "nca 66ms err", "nca motion", "nca-rigid gap", "spread")
LUMA_W = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def fit_gate(X, E, loss="relative", hidden=0, steps=600):
    """Softmax gate over the sources of E, fitted on X (M, F) standardised features, E (M, S) patch errors.
    loss: expected = mean sum_s w_s E_s (dominated by high-error, moving patches); relative = the same
    with each patch's errors divided by their mean (every patch counts equally); xent = cross-entropy
    against the best source. hidden > 0: a one-hidden-layer MLP instead of a linear gate.
    Returns predict(X) -> soft weights (numpy)."""
    torch.manual_seed(0)
    Xt, Et = torch.from_numpy(X).to(nca.DEVICE), torch.from_numpy(E).to(nca.DEVICE)
    f, ns = X.shape[1], E.shape[1]
    net = (torch.nn.Sequential(torch.nn.Linear(f, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, ns)) if hidden
           else torch.nn.Linear(f, ns)).to(nca.DEVICE)
    opt = torch.optim.Adam(net.parameters(), 0.01 if hidden else 0.05)
    target, rel = Et.argmin(1), Et / (Et.mean(1, keepdim=True) + 1e-6)
    for _ in range(steps):
        logits = net(Xt)
        if loss == "xent":
            l = F.cross_entropy(logits, target)
        else:
            l = (torch.softmax(logits, 1) * (rel if loss == "relative" else Et * 1e3)).sum(1).mean()
        opt.zero_grad()
        l.backward()
        opt.step()
    net.eval()

    @torch.no_grad()
    def predict(Xn):
        return torch.cat([torch.softmax(net(torch.from_numpy(Xn[i:i + 262144]).to(nca.DEVICE)), 1).cpu()
                          for i in range(0, len(Xn), 262144)]).numpy()
    predict.net = net
    return predict


GATE_FILE = nca.ROOT / "state" / "nca_gate.npz"
A_FIXED = np.array([0.229, 0.262, 0.270, 0.240], np.float32)  # E30 "A fixed" over SOURCES
B_FIXED = {"nca": 0.359, "rigid": 0.328}  # E30 "B fixed" motion blend (copy's 0.313 is zero motion)


class Scene:
    """A clip at NCA resolution with what the mixers need: frames, superpixel labels and the lineage
    model's motion fields (entry 0 = the 66 ms step, entries 1.. = the 1 s step and its successors;
    with short=3, entries 0-2 are the 66 / 133 / 200 ms steps and the 1 s step starts at entry 3)."""

    def __init__(self, frames, labels, first, last, short=1):
        self.frames, self.labels, self.first, self.short = frames, labels, first, short
        model, k1 = nca.load("lineage"), nca.lineage_step(nca.HORIZON)
        self.flows = lineage_flows(model, frames, first, last, list(range(short)) + list(range(k1, k1 + ENSEMBLE)))

    def to(self, a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(nca.DEVICE).permute(0, 3, 1, 2).float() / 255

    def fl(self, t, j):
        return torch.from_numpy(self.flows[t - self.first, j].astype(np.float32)).to(nca.DEVICE)

    def sources_at(self, b):
        """Images (B, S, 3, H, W), motion fields (B, S, 2, H, W), newest flow, rigid flow, the 10 guesses for frames b."""
        cur = self.to(self.frames[b])
        s = self.short
        newest = self.fl(b, s)
        rigid_flow = region_mean(newest, torch.from_numpy(self.labels[b].astype(np.int64)).to(nca.DEVICE))
        guesses = torch.stack([nca.warp(self.to(self.frames[b - nca.LINEAGE_STRIDE * j]), self.fl(b - nca.LINEAGE_STRIDE * j, s + j))
                               for j in range(ENSEMBLE)])
        imgs = torch.stack([cur, nca.warp(cur, newest), guesses.mean(0), nca.warp(cur, rigid_flow)], 1)
        motion = torch.stack([torch.zeros_like(newest), newest, torch.zeros_like(newest), rigid_flow], 1)
        return cur, imgs, motion, newest, rigid_flow, guesses

    def features_at(self, b):
        """The gate's FEATURES per patch for frames b: (B, F, P, Q) numpy."""
        cur, _, _, newest, rigid_flow, guesses = self.sources_at(b)
        prev3, prev6, prev2 = self.to(self.frames[b - 3]), self.to(self.frames[b - 6]), self.to(self.frames[b - 2])
        pred66 = nca.warp(prev2, self.fl(b - 2, 0))  # the 66 ms prediction made 2 frames ago; its target is cur
        luma = lambda img: (img * LUMA_W.to(img.device)).sum(1, keepdim=True)
        return torch.stack([
            pool((luma(cur) - luma(prev3)).abs()),
            pool((luma(prev3) - luma(prev6)).abs()),
            pool(((prev2 - cur) ** 2).mean(1)),
            pool(((pred66 - cur) ** 2).mean(1)),
            pool(newest.norm(dim=1)),
            pool((newest - rigid_flow).norm(dim=1)),
            pool(guesses.std(0).mean(1)),
        ], 1).cpu().numpy()

    def features2_at(self, b):
        """E35: the E32 features plus the trend, curve, lagged and structure groups: (B, F2, P, Q) numpy,
        in FEATURES2 order. Needs short=3 and frames back to b - HORIZON - 2 * (ENSEMBLE - 1) - 6."""
        base = torch.from_numpy(self.features_at(b)).to(nca.DEVICE)
        cur, _, _, newest, rigid_flow, _ = self.sources_at(b)
        f = lambda k: self.to(self.frames[b - k])
        luma = lambda img: (img * LUMA_W.to(img.device)).sum(1, keepdim=True)
        lab2 = torch.from_numpy(self.labels[b - 2].astype(np.int64)).to(nca.DEVICE)
        rigid66 = nca.warp(f(2), region_mean(self.fl(b - 2, 0), lab2))
        trend = [pool((luma(f(6)) - luma(f(9))).abs()), pool((luma(f(9)) - luma(f(12))).abs())]
        curve = [pool(((nca.warp(f(4), self.fl(b - 4, 1)) - cur) ** 2).mean(1)),  # NCA 133 ms error
                 pool(((nca.warp(f(6), self.fl(b - 6, 2)) - cur) ** 2).mean(1)),  # NCA 200 ms error
                 pool(((f(4) - cur) ** 2).mean(1)), pool(((f(6) - cur) ** 2).mean(1)),  # copy 133 / 200 ms
                 pool(((rigid66 - cur) ** 2).mean(1))]
        _, imgs_then, _, _, _, _ = self.sources_at(b - nca.HORIZON)  # the 1 s predictions whose target is cur
        lagged = [pool(((imgs_then[:, i] - cur) ** 2).mean(1)) for i in range(4)]
        gl = luma(cur)
        gx = gl[..., :, 2:] - gl[..., :, :-2]
        gy = gl[..., 2:, :] - gl[..., :-2, :]
        edge = pool(F.pad(gx.abs(), (1, 1, 0, 0)) + F.pad(gy.abs(), (0, 0, 1, 1)))
        div = pool(F.pad(newest[:, 0:1, :, 2:] - newest[:, 0:1, :, :-2], (1, 1, 0, 0)).abs()
                   + F.pad(newest[:, 1:2, 2:, :] - newest[:, 1:2, :-2, :], (0, 0, 1, 1)).abs())
        flows10 = torch.stack([self.fl(b - nca.LINEAGE_STRIDE * j, self.short + j) for j in range(ENSEMBLE)])  # (10, B, 2, H, W)
        flow_spread = pool(flows10.std(0).norm(dim=1))
        return torch.cat([base, torch.stack(trend + curve + lagged + [edge, div, flow_spread], 1)], 1).cpu().numpy()


FEATURES2 = FEATURES + ("motion 6-9", "motion 9-12",
                        "nca 133ms err", "nca 200ms err", "copy 133ms err", "copy 200ms err", "rigid 66ms err",
                        "copy 1s err (lagged)", "nca 1s err (lagged)", "nca10 1s err (lagged)", "rigid 1s err (lagged)",
                        "edge energy", "flow divergence", "flow spread")
GROUPS = {"base": list(range(7)), "trend": [7, 8], "curve": [9, 10, 11, 12, 13], "lagged": [14, 15, 16, 17],
          "structure": [18, 19, 20]}


def save_gate(hidden=32):
    """Fit the live gate - cross-entropy, soft weights, a 7 -> hidden -> 4 MLP on the E32 features (E35's
    winner; hidden=0 gives E32's linear gate) - on half 1 of the cached features and errors, and save its
    weights and feature standardisation for nca_mirror.py."""
    base = np.load(OUT)
    err = base["patch_sq"].astype(np.float32)
    n, half = len(err), np.arange(len(err)) >= len(err) // 2
    P, Q = err.shape[2:]
    feats = np.load(nca.ROOT / "results" / "mix_features.npz")["features"]
    Z = np.log(feats + 1e-4)
    mu, sd = Z[~half].mean((0, 2, 3)), Z[~half].std((0, 2, 3)) + 1e-6
    X = ((Z - mu[None, :, None, None]) / sd[None, :, None, None]).astype(np.float32).transpose(0, 2, 3, 1).reshape(-1, len(FEATURES))
    E = err.transpose(0, 2, 3, 1).reshape(-1, err.shape[1])
    r2 = np.repeat(half, P * Q)
    net = fit_gate(X[~r2], E[~r2], "xent", hidden).net
    g = lambda t: t.detach().cpu().numpy()
    params = ({"W1": g(net[0].weight), "b1": g(net[0].bias), "W2": g(net[2].weight), "b2": g(net[2].bias)} if hidden
              else {"W": g(net.weight), "b": g(net.bias)})
    np.savez(GATE_FILE, mu=mu.astype(np.float32), sd=sd.astype(np.float32), features=np.array(FEATURES),
             sources=np.array(SOURCES), **params)
    print(f"saved {GATE_FILE.relative_to(nca.ROOT)}: {'MLP ' + str(hidden) if hidden else 'linear'}, fitted on {(~r2).sum()} patches")


def gate():
    base = np.load(OUT)
    ts, err = base["t"], base["patch_sq"].astype(np.float32)  # (N, S, P, Q)
    n, half = len(ts), np.arange(len(ts)) >= len(ts) // 2
    P, Q = err.shape[2:]
    frames = np.load(nca.FRAMES)
    labels = np.load(graph_nca.LABELS, mmap_mode="r")
    oldest = nca.LINEAGE_STRIDE * (ENSEMBLE - 1)
    first = int(ts[0]) - oldest
    start = time.monotonic()
    scene = Scene(frames, labels, first, int(ts[-1]))
    to, sources_at = scene.to, scene.sources_at

    # pass 1: features known at time t, per patch (cached after the first run)
    cache = nca.ROOT / "results" / "mix_features.npz"
    if cache.exists():
        feats = np.load(cache)["features"]
    else:
        feats = np.zeros((n, len(FEATURES), P, Q), np.float32)
        with torch.no_grad():
            for i in range(0, n, 64):
                sl = slice(i, min(i + 64, n))
                feats[sl] = scene.features_at(ts[sl])
        np.savez_compressed(cache, t=ts, features=feats, names=np.array(FEATURES))
        print(f"features for {n} frames in {time.monotonic() - start:.0f}s")

    # standardised log features, plus their 3x3 neighbourhood mean as spatial context
    Z = np.log(feats + 1e-4)
    mu = Z[~half].mean((0, 2, 3), keepdims=True)
    sd = Z[~half].std((0, 2, 3), keepdims=True) + 1e-6
    Z = ((Z - mu) / sd).astype(np.float32)
    ctx = F.avg_pool2d(torch.from_numpy(Z), 3, stride=1, padding=1, count_include_pad=False).numpy()
    rows = lambda a: a.transpose(0, 2, 3, 1).reshape(n * P * Q, -1)  # (N*P*Q, F)
    X, Xc = rows(Z), rows(np.concatenate([Z, ctx], 1))
    E = rows(err)
    r2 = np.repeat(half, P * Q)
    all4, b3 = [0, 1, 2, 3], list(B_SOURCES)
    specs = {  # name: (sources, features, loss, hidden)
        "lin expected": (all4, X, "expected", 0),
        "lin relative": (all4, X, "relative", 0),
        "lin xent": (all4, X, "xent", 0),
        "mlp relative": (all4, X, "relative", 32),
        "mlp+ctx relative": (all4, Xc, "relative", 32),
        "B mlp relative": (b3, X, "relative", 32),
    }
    soft, pick = {}, {}
    for name, (src, Xs, loss, hidden) in specs.items():
        t0 = time.monotonic()
        predict = fit_gate(Xs[~r2], E[~r2][:, src], loss, hidden)
        w = np.zeros((r2.sum(), 4), np.float32)
        w[:, src] = predict(Xs[r2])
        soft[name] = w
        pk = np.zeros_like(w)
        pk[np.arange(len(w)), w.argmax(1)] = 1
        pick[name] = pk
        print(f"fitted {name} in {time.monotonic() - t0:.0f}s", flush=True)

    # surrogate from the cached patch errors (exact for picks, an upper bound on error for soft blends)
    E2 = E[r2]
    per_frame = lambda e: e.reshape(-1, P * Q).mean(1)
    copy_s = per_frame(E2[:, 0])
    bins_s = np.digitize(copy_s, np.percentile(copy_s, [20, 40, 60, 80]))
    print(f"\nsurrogate on half 2 (patch errors): {'gate':20} {'soft':>7} {'q1':>7}   {'pick':>7} {'q1':>7}")
    orc = per_frame(E2.min(1))
    print(f"{'':44} {'true oracle pick':20} {'':>7} {'':>7}   {1 - orc.sum() / copy_s.sum():+7.3f} {1 - orc[bins_s == 0].sum() / copy_s[bins_s == 0].sum():+7.3f}")
    for name in specs:
        es, ep = per_frame((soft[name] * E2).sum(1)), per_frame(E2[np.arange(len(E2)), pick[name].argmax(1)])
        q = bins_s == 0
        print(f"{'':44} {name:20} {1 - es.sum() / copy_s.sum():+7.3f} {1 - es[q].sum() / copy_s[q].sum():+7.3f}   "
              f"{1 - ep.sum() / copy_s.sum():+7.3f} {1 - ep[q].sum() / copy_s[q].sum():+7.3f}")

    # pass 2: real predictions on half 2
    variants = ["true oracle pick"] + [f"{name} {kind}" for name in specs for kind in (("soft",) if name.startswith("B") else ("soft", "pick"))]
    shape = lambda a: a.reshape(-1, P, Q, 4).transpose(0, 3, 1, 2)
    mse = {k: [] for k in variants}
    sharp = {k: [] for k in variants}
    copy_mse = []
    idx2 = np.flatnonzero(half)
    with torch.no_grad():
        for i in range(0, len(idx2), 64):
            sl = idx2[i:i + 64]
            b = ts[sl]
            pos = slice((sl[0] - idx2[0]) * P * Q, (sl[-1] - idx2[0] + 1) * P * Q)  # rows of half 2
            cur, imgs, motion, _, _, _ = sources_at(b)
            y = to(frames[b + nca.HORIZON])
            truth_sharp = sobel_energy(y)
            copy_mse.append(((cur - y) ** 2).mean((1, 2, 3)).cpu().numpy())
            oracle = np.zeros((len(sl), 4, P, Q), np.float32)
            np.put_along_axis(oracle, err[sl].argmin(1)[:, None], 1.0, axis=1)
            w_all = {"true oracle pick": oracle}
            for name in specs:
                w_all[f"{name} soft"] = shape(soft[name][pos])
                w_all[f"{name} pick"] = shape(pick[name][pos])
            for k in variants:
                w = upsample_weights(w_all[k])[:, :, None]
                pred = nca.warp(cur, (w * motion).sum(1)) if k.startswith("B") else (w * imgs).sum(1)
                mse[k].append(((pred - y) ** 2).mean((1, 2, 3)).cpu().numpy())
                sharp[k].append((sobel_energy(pred) / truth_sharp).cpu().numpy())
    copy_mse = np.concatenate(copy_mse)
    mse = {k: np.concatenate(v) for k, v in mse.items()}
    sharp = {k: np.median(np.concatenate(v)) for k, v in sharp.items()}
    np.savez_compressed(nca.ROOT / "results" / "mix_gate.npz", t=ts[half], copy=copy_mse,
                        **{k.replace(" ", "_"): v for k, v in mse.items()})

    phase1 = np.load(nca.ROOT / "results" / "mix_phase1.npz")
    for k, key in (("(E30) A fixed", "A_fixed"), ("(E30) C lagged", "C_lagged_0.9"), ("(E30) C oracle (running)", "C_oracle_0.9")):
        mse[k] = phase1[key][half]
        sharp[k] = float("nan")
    for c, srcname in ((1, "nca"), (2, "nca10")):
        mse[f"(source) {srcname}"] = base["mse"][half, c]
        sharp[f"(source) {srcname}"] = np.median(base["sharp"][half, c])
    bins = np.digitize(copy_mse, np.percentile(copy_mse, [20, 40, 60, 80]))
    sk = lambda k, m=slice(None): 1 - mse[k][m].sum() / copy_mse[m].sum()
    print(f"\nphase 1b: gates fitted on half 1, reported on half 2 ({half.sum()} targets); {time.monotonic() - start:.0f}s")
    print(f"{'variant':26} {'skill':>7}   {'q1 (still)':>10} {'q2':>7} {'q3':>7} {'q4':>7} {'q5':>7}   {'better':>6}  {'sharp':>6}")
    for k in ["(source) nca", "(source) nca10", "(E30) A fixed", "(E30) C lagged", "(E30) C oracle (running)"] + variants:
        quint = f"{sk(k, bins == 0):+10.3f} " + " ".join(f"{sk(k, bins == q):+7.3f}" for q in range(1, 5))
        print(f"{k:26} {sk(k):+7.3f}   {quint}   {(mse[k] < copy_mse).mean():6.1%}  {sharp[k]:6.3f}")

def gate_logits(z, gate):
    """Logits over the sources from standardised features z (..., F): linear (W, b) or MLP (W1, b1, W2, b2) file."""
    if "W1" in gate:
        return np.maximum(z @ gate["W1"].T + gate["b1"], 0) @ gate["W2"].T + gate["b2"]
    return z @ gate["W"].T + gate["b"]


def load_gate():
    g = np.load(GATE_FILE)
    return {k: g[k].astype(np.float32) for k in g.files if k not in ("features", "sources")}


def gate_soft(feats, gate):
    """Soft source weights (B, 4, P, Q) from the saved gate for features (B, F, P, Q)."""
    z = (np.log(feats + 1e-4) - gate["mu"][None, :, None, None]) / gate["sd"][None, :, None, None]
    logits = gate_logits(np.moveaxis(z, 1, -1), gate)  # (B, P, Q, S)
    logits = np.moveaxis(logits, -1, 1)
    e = np.exp(logits - logits.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def newroom(clip, start, seconds):
    """Phase 3: score every mixer on a new clip (nothing refitted). Frames start..start + 30 * seconds."""
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while len(frames) < seconds * 30:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(bgr, (nca.W, nca.H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    frames = np.stack(frames)
    t0 = time.monotonic()
    with Pool(8) as p:
        labels = np.stack(p.map(graph_nca.superpixels, list(frames), chunksize=64))
    oldest = nca.LINEAGE_STRIDE * (ENSEMBLE - 1)
    ts = np.arange(oldest + 6, len(frames) - nca.HORIZON)
    scene = Scene(frames, labels, int(ts[0]) - oldest, int(ts[-1]))
    gate = load_gate()
    names = ["copy", "nca", "nca10", "rigid", "blend (E30)", "crisp (E30)", "gate (saved)"]
    mse, sharp = {k: [] for k in names}, {k: [] for k in names}
    with torch.no_grad():
        for i in range(0, len(ts), 64):
            b = ts[i:i + 64]
            cur, imgs, motion, newest, rigid_flow, _ = scene.sources_at(b)
            y = scene.to(frames[b + nca.HORIZON])
            w = upsample_weights(gate_soft(scene.features_at(b), gate))[:, :, None]
            preds = dict(zip(names[:4], imgs.unbind(1)))
            preds["blend (E30)"] = (torch.from_numpy(A_FIXED).to(nca.DEVICE)[None, :, None, None, None] * imgs).sum(1)
            preds["crisp (E30)"] = nca.warp(cur, B_FIXED["nca"] * newest + B_FIXED["rigid"] * rigid_flow)
            preds["gate (saved)"] = (w * imgs).sum(1)
            truth_sharp = sobel_energy(y)
            for k in names:
                mse[k].append(((preds[k] - y) ** 2).mean((1, 2, 3)).cpu().numpy())
                sharp[k].append((sobel_energy(preds[k]) / truth_sharp).cpu().numpy())
    mse = {k: np.concatenate(v) for k, v in mse.items()}
    sharp = {k: np.median(np.concatenate(v)) for k, v in sharp.items()}
    np.savez_compressed(nca.ROOT / "results" / f"newroom_{Path(clip).stem}.npz", t=ts + start,
                        **{k.split(" ")[0]: v for k, v in mse.items()})
    copy = mse["copy"]
    bins = np.digitize(copy, np.percentile(copy, [20, 40, 60, 80]))
    sk = lambda k, m=slice(None): 1 - mse[k][m].sum() / copy[m].sum()
    print(f"{Path(clip).name} from frame {start}: {len(frames)} frames ({len(frames) / 30 / 60:.1f} min), {len(ts)} targets, "
          f"1 s ahead, 160x90, nothing refitted; {time.monotonic() - t0:.0f}s")
    print(f"{'variant':14} {'skill':>7}   {'q1 (still)':>10} {'q2':>7} {'q3':>7} {'q4':>7} {'q5':>7}   {'better':>6}  {'sharp':>6}")
    for k in names[1:]:
        quint = f"{sk(k, bins == 0):+10.3f} " + " ".join(f"{sk(k, bins == q):+7.3f}" for q in range(1, 5))
        print(f"{k:14} {sk(k):+7.3f}   {quint}   {(mse[k] < copy).mean():6.1%}  {sharp[k]:6.3f}")


def fit_gate2(X, E, loss="xent", hidden=0, steps=600):
    """As fit_gate, plus loss="soft": cross-entropy against softmax(-E / mean E) - a soft best-source target."""
    if loss != "soft":
        return fit_gate(X, E, loss, hidden, steps)
    torch.manual_seed(0)
    Xt, Et = torch.from_numpy(X).to(nca.DEVICE), torch.from_numpy(E).to(nca.DEVICE)
    f, ns = X.shape[1], E.shape[1]
    net = (torch.nn.Sequential(torch.nn.Linear(f, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, ns)) if hidden
           else torch.nn.Linear(f, ns)).to(nca.DEVICE)
    opt = torch.optim.Adam(net.parameters(), 0.01 if hidden else 0.05)
    q = torch.softmax(-Et / (Et.mean(1, keepdim=True) + 1e-6), 1)
    for _ in range(steps):
        l = -(q * torch.log_softmax(net(Xt), 1)).sum(1).mean()
        opt.zero_grad()
        l.backward()
        opt.step()
    net.eval()

    @torch.no_grad()
    def predict(Xn):
        return torch.cat([torch.softmax(net(torch.from_numpy(Xn[i:i + 262144]).to(nca.DEVICE)), 1).cpu()
                          for i in range(0, len(Xn), 262144)]).numpy()
    predict.net = net
    return predict


def load_clip(clip, start=0, seconds=1e9):
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while len(frames) < seconds * 30:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(bgr, (nca.W, nca.H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    frames = np.stack(frames)
    with Pool(8) as p:
        labels = np.stack(p.map(graph_nca.superpixels, list(frames), chunksize=64))
    return frames, labels


def gate2(newclip):
    """E35: richer gate features. Pass 1 on both clips: per-patch errors of the four sources and
    FEATURES2 (cached in results/mix_features2_*.npz). Fit gates on old-room half 1 by cumulative
    feature group, plus an MLP and a soft-target loss on the full set. Pass 2: real soft blends
    scored on old-room half 2 and on the whole new-room clip."""
    back = nca.HORIZON + nca.LINEAGE_STRIDE * (ENSEMBLE - 1) + 6  # frames needed before a target time
    clips = {}
    old_frames = np.load(nca.FRAMES)
    _, test_idx = nca.split(len(old_frames))
    clips["old"] = (old_frames, np.load(graph_nca.LABELS, mmap_mode="r"), test_idx[test_idx >= test_idx[0] + back])
    new_frames, new_labels = load_clip(newclip)
    clips["new"] = (new_frames, new_labels, np.arange(back, len(new_frames) - nca.HORIZON))
    P, Q = nca.H // PATCH, nca.W // PATCH

    def scene_for(name):
        frames, labels, ts = clips[name]
        return Scene(frames, labels, int(ts[0]) - nca.HORIZON - nca.LINEAGE_STRIDE * (ENSEMBLE - 1), int(ts[-1]), short=3), frames, ts

    start = time.monotonic()
    data = {}
    for name in clips:
        cache = nca.ROOT / "results" / f"mix_features2_{name}.npz"
        if cache.exists():
            d = np.load(cache)
            data[name] = (d["features"], d["patch_sq"])
            continue
        scene, frames, ts = scene_for(name)
        feats, errs = np.zeros((len(ts), len(FEATURES2), P, Q), np.float32), np.zeros((len(ts), 4, P, Q), np.float32)
        with torch.no_grad():
            for i in range(0, len(ts), 64):
                sl = slice(i, min(i + 64, len(ts)))
                b = ts[sl]
                feats[sl] = scene.features2_at(b)
                _, imgs, _, _, _, _ = scene.sources_at(b)
                y = scene.to(frames[b + nca.HORIZON])
                errs[sl] = torch.stack([patch_sq(imgs[:, k], y) for k in range(4)], 1).cpu().numpy()
        np.savez_compressed(cache, t=ts, features=feats, patch_sq=errs, names=np.array(FEATURES2))
        data[name] = (feats, errs)
        del scene
        print(f"pass 1 {name}: {len(ts)} targets, {time.monotonic() - start:.0f}s", flush=True)

    # fit on old-room half 1
    feats_old, err_old = data["old"]
    n_old = len(feats_old)
    half = np.arange(n_old) >= n_old // 2
    rows = lambda a: a.transpose(0, 2, 3, 1).reshape(len(a) * P * Q, -1)
    ctx = lambda z: F.avg_pool2d(torch.from_numpy(z), 3, stride=1, padding=1, count_include_pad=False).numpy()
    X = {}  # (floor, with_ctx) -> (old, new) standardised features; floor = the epsilon inside log(f + floor)
    for floor in (1e-4, 1e-6):
        Z = lambda f: np.log(f + floor)
        mu = Z(feats_old[~half]).mean((0, 2, 3), keepdims=True)
        sd = Z(feats_old[~half]).std((0, 2, 3), keepdims=True) + 1e-6
        std = lambda f: ((Z(f) - mu) / sd).astype(np.float32)
        Xo, Xn = std(feats_old), std(data["new"][0])
        X[floor, False] = (Xo, Xn)
        X[floor, True] = (np.concatenate([Xo, ctx(Xo)], 1), np.concatenate([Xn, ctx(Xn)], 1))
    Eo = rows(err_old)
    r2 = np.repeat(half, P * Q)
    full = list(range(len(FEATURES2)))
    specs = {  # name: (feature columns, with_ctx, loss, hidden, floor). E35 part 2: is the MLP's gain nonlinearity or features?
        "lin xent base (E32)": (GROUPS["base"], False, "xent", 0, 1e-4),
        "mlp32 xent base": (GROUPS["base"], False, "xent", 32, 1e-4),
        "mlp32 xent all": (full, False, "xent", 32, 1e-4),
        "mlp64 xent base": (GROUPS["base"], False, "xent", 64, 1e-4),
        "mlp64 xent all": (full, False, "xent", 64, 1e-4),
        "mlp32 xent base + ctx": (GROUPS["base"], True, "xent", 32, 1e-4),
    }
    weights = {}  # name -> {"old": (M2, 4), "new": (M, 4)}
    for name, (cols, with_ctx, loss, hidden, floor) in specs.items():
        t0 = time.monotonic()
        Xa, Xb = X[floor, with_ctx]
        cols_c = cols + [c + len(FEATURES2) for c in cols] if with_ctx else cols
        predict = fit_gate2(rows(Xa)[~r2][:, cols_c], Eo[~r2], loss, hidden)
        weights[name] = {"old": predict(rows(Xa)[r2][:, cols_c]), "new": predict(rows(Xb)[:, cols_c])}
        print(f"fitted {name} ({len(cols_c)} features) in {time.monotonic() - t0:.0f}s", flush=True)

    # pass 2: real soft blends
    for name in ("old", "new"):
        scene, frames, ts = scene_for(name)
        if name == "old":
            ts = ts[half]
        mse, sharp, copy_mse = {k: [] for k in specs}, {k: [] for k in specs}, []
        with torch.no_grad():
            for i in range(0, len(ts), 64):
                sl = slice(i, min(i + 64, len(ts)))
                b = ts[sl]
                cur, imgs, _, _, _, _ = scene.sources_at(b)
                y = scene.to(frames[b + nca.HORIZON])
                truth_sharp = sobel_energy(y)
                copy_mse.append(((cur - y) ** 2).mean((1, 2, 3)).cpu().numpy())
                for k in specs:
                    w = weights[k][name][sl.start * P * Q:sl.stop * P * Q].reshape(-1, P, Q, 4).transpose(0, 3, 1, 2)
                    pred = (upsample_weights(w)[:, :, None] * imgs).sum(1)
                    mse[k].append(((pred - y) ** 2).mean((1, 2, 3)).cpu().numpy())
                    sharp[k].append((sobel_energy(pred) / truth_sharp).cpu().numpy())
        copy_mse = np.concatenate(copy_mse)
        mse = {k: np.concatenate(v) for k, v in mse.items()}
        bins = np.digitize(copy_mse, np.percentile(copy_mse, [20, 40, 60, 80]))
        sk = lambda k, m=slice(None): 1 - mse[k][m].sum() / copy_mse[m].sum()
        print(f"\n{name} room ({'held-out half 2' if name == 'old' else Path(newclip).name}), {len(ts)} targets, "
              f"gates fitted on old-room half 1; {time.monotonic() - start:.0f}s")
        print(f"{'gate':24} {'skill':>7}   {'q1 (still)':>10} {'q2':>7} {'q3':>7} {'q4':>7} {'q5':>7}   {'better':>6}  {'sharp':>6}")
        for k in specs:
            quint = f"{sk(k, bins == 0):+10.3f} " + " ".join(f"{sk(k, bins == q):+7.3f}" for q in range(1, 5))
            print(f"{k:24} {sk(k):+7.3f}   {quint}   {(mse[k] < copy_mse).mean():6.1%}  {np.median(np.concatenate(sharp[k])):6.3f}")
        del scene


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline")
    sub.add_parser("mix")
    sub.add_parser("gate")
    sub.add_parser("save-gate")
    r = sub.add_parser("newroom")
    r.add_argument("clip", type=Path)
    r.add_argument("--start", type=int, default=0, help="first frame to use")
    r.add_argument("--seconds", type=float, default=600, help="how much of the clip to use")
    sub.add_parser("gate2").add_argument("newclip", type=Path)
    args = parser.parse_args()
    if args.cmd == "baseline":
        baseline()
    elif args.cmd == "mix":
        mix()
    elif args.cmd == "save-gate":
        save_gate()
    elif args.cmd == "newroom":
        newroom(args.clip, args.start, args.seconds)
    elif args.cmd == "gate2":
        gate2(args.newclip)
    else:
        gate()


if __name__ == "__main__":
    main()
