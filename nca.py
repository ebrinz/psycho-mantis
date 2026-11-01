# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "opencv-python", "torch"]
# ///
"""Neural cellular automaton (Mordvintsev et al. 2020, "Growing NCA") that predicts the camera
frame `HORIZON` steps (1 s) ahead.

Every cell of a low-resolution grid runs the same small learned rule, seeing only its 3x3
neighbourhood (identity, Sobel x/y and Laplacian of every channel). The grid holds the current
frame and two earlier ones (fixed, so motion is visible) plus hidden channels that the rule
updates for `STEPS` steps; information travels at most one cell per step. Two read-outs:

- `delta`: the first three hidden channels are the colour change to the future frame, enlarged
  and added to the full-res current frame. Minimising error makes it average the possible
  futures, so moving parts blur.
- `warp`: the first two hidden channels are a motion field (where each pixel of the future frame
  comes from in the current one); the full-res current frame is resampled along it. It can only
  move pixels, never blend them, so it commits to one crisp future.
- `lineage`: like `warp`, but read out after every inner step: step k predicts k * LINEAGE_STRIDE
  frames ahead (66 ms ... 1.6 s), so the inner steps are a rollout through time, trained with a
  loss at every step. 1 s ahead is step HORIZON / LINEAGE_STRIDE.
- `persistent` (E24): an NCA that never stops. The hidden state carries over between frames; each
  camera frame and the one PERSISTENT_PAST[1] steps earlier are written into the visible channels
  (with the current frame only it did not learn, E24) and the rule runs a few steps per frame. Dedicated channels hold the motion fields
  for PERSISTENT_HORIZONS. Trained on streams that run on through the recording, carrying their
  state (truncated backpropagation through time), occasionally restarting from zero.

  uv run nca.py prep recordings/clip.mp4                      decode to data/frames_160x90.npy
  uv run nca.py train [--out warp|lineage|persistent] [--loss l1] [--iters N] [--resume | --init-from warp]
  uv run nca.py eval recordings/clip.mp4 [--out warp|lineage|persistent]   skill vs copying, last 20%
"""
import argparse
import functools
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).parent
FRAMES = ROOT / "data" / "frames_160x90.npy"
CHECKPOINTS = {"delta": ROOT / "state" / "nca.pt", "warp": ROOT / "state" / "nca_warp.pt",
               "lineage": ROOT / "state" / "nca_lineage.pt", "persistent": ROOT / "state" / "nca_persistent.pt"}
W, H = 160, 90
HORIZON = 30  # steps ahead to predict: 1 s at 30 fps
PAST = (0, 3, 6)  # input frames: now, 100 ms ago, 200 ms ago
TRAIN_FRACTION = 0.8
LINEAGE_STRIDE = 2  # lineage read-out: inner step k (1-based) predicts k * 2 frames ahead
PERSISTENT_HORIZONS = (10, 20, 30)  # frames ahead read out by the persistent NCA
PERSISTENT_PAST = (0, 3)  # frames written into the persistent NCA each step: now and 100 ms ago
FLOW_SCALE = 10.0  # warp read-out: hidden units -> grid cells, so motion of several cells is in easy reach
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class NCA(nn.Module):
    def __init__(self, out="delta", hidden=16, width=96, steps=24, inputs=3 * len(PAST)):
        super().__init__()
        self.out, self.hidden, self.steps = out, hidden, steps
        channels = inputs + hidden
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sobel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
        lap = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype=torch.float32) / 16
        kernels = torch.stack([ident, sobel, sobel.T, lap])  # (4, 3, 3)
        self.register_buffer("kernels", kernels.repeat(channels, 1, 1)[:, None])  # depthwise, 4 per channel
        self.channels = channels
        self.rule = nn.Sequential(nn.Conv2d(4 * channels, width, 1), nn.ReLU(), nn.Conv2d(width, hidden, 1))
        nn.init.zeros_(self.rule[2].weight)  # start as "no change", i.e. copy the current frame
        nn.init.zeros_(self.rule[2].bias)

    def forward(self, frames):
        """frames: (B, 3 * len(PAST), H, W) in 0-1, current frame first. Returns the colour change
        (B, 3, H, W) for `delta`, the motion field in grid cells (B, 2, H, W: dx, dy) for `warp`, or
        the motion field after every step (B, steps, 2, H, W) for `lineage`."""
        state = torch.zeros(frames.shape[0], self.hidden, *frames.shape[2:], device=frames.device)
        flows = []
        for _ in range(self.steps):
            x = F.pad(torch.cat([frames, state], 1), (1, 1, 1, 1), mode="replicate")
            state = state + self.rule(F.conv2d(x, self.kernels, groups=self.channels))
            if self.out == "lineage":
                flows.append(state[:, :2] * FLOW_SCALE)
        if self.out == "lineage":
            return torch.stack(flows, 1)
        return state[:, :3] if self.out == "delta" else state[:, :2] * FLOW_SCALE


class PersistentNCA(NCA):
    """The NCA that never stops: call once per camera frame with the state it returned last time."""

    def __init__(self, hidden=24, width=96, steps=6):
        super().__init__("persistent", hidden, width, steps, inputs=3 * len(PERSISTENT_PAST))
        self.readout = nn.Conv2d(hidden, 2 * len(PERSISTENT_HORIZONS), 1)
        nn.init.zeros_(self.readout.weight)  # start as "no motion", i.e. copy the current frame
        nn.init.zeros_(self.readout.bias)
        # the rule must not also start at zero: then the state stays zero and no gradient reaches either (E24)
        nn.init.normal_(self.rule[2].weight, std=0.02)

    def forward(self, frame, state=None):
        """frame: (B, 3 * len(PERSISTENT_PAST), H, W) in 0-1, current frame first; state: (B, hidden, H, W)
        from the previous call, or None to start from zero. Returns the new state and the motion fields (B, len(PERSISTENT_HORIZONS), 2, H, W)."""
        if state is None:
            state = torch.zeros(frame.shape[0], self.hidden, *frame.shape[2:], device=frame.device)
        for _ in range(self.steps):
            x = F.pad(torch.cat([frame, state], 1), (1, 1, 1, 1), mode="replicate")
            # tanh keeps the state in (-1, 1) over minutes; a hard clamp let it run to the bound and stick (E24)
            state = torch.tanh(state + self.rule(F.conv2d(x, self.kernels, groups=self.channels)))
        n = len(PERSISTENT_HORIZONS)
        return state, self.readout(state).reshape(frame.shape[0], n, 2, *frame.shape[2:]) * FLOW_SCALE


def lineage_step(horizon):
    """0-based inner step whose read-out predicts `horizon` frames ahead (lineage)."""
    return horizon // LINEAGE_STRIDE - 1


def warp(img, flow):
    """Resample img (B, C, H, W) at each pixel + flow (B, 2, H, W, in pixels)."""
    b, _, h, w = img.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing="ij")
    x = (gx + flow[:, 0]) * 2 / (w - 1) - 1
    y = (gy + flow[:, 1]) * 2 / (h - 1) - 1
    return F.grid_sample(img, torch.stack([x, y], -1), padding_mode="border", align_corners=True)


def predict(model, x, out=None):
    """Predicted future frame at NCA resolution (`out`: the model's output for x, if already computed)."""
    out = model(x) if out is None else out
    if model.out == "lineage":
        out = out[:, lineage_step(HORIZON)]
    return x[:, :3] + out if model.out == "delta" else warp(x[:, :3], out)


def full_res(model, x, frame, out=None):
    """Predicted future frame at the resolution of `frame` (H, W, 3 float 0-1, the current frame)."""
    out = model(x) if out is None else out
    if model.out == "lineage":
        out = out[:, lineage_step(HORIZON)]
    out = out[0].permute(1, 2, 0).cpu().numpy()
    if model.out == "delta":
        return np.clip(frame + cv2.resize(out, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR), 0, 1)
    return warp_full(frame, out)


@functools.cache
def _grid(h, w):
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    return gx, gy


def warp_full(frame, flow):
    """Resample `frame` (H, W, C) along a motion field at NCA resolution (H/4, W/4, 2, in NCA cells)."""
    h, w = frame.shape[:2]
    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    gx, gy = _grid(h, w)
    return cv2.remap(frame, gx + flow[..., 0] * w / W, gy + flow[..., 1] * h / H, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def batch(frames, idx):
    """Inputs and targets for frame indices idx, as float tensors on DEVICE."""
    x = np.concatenate([frames[idx - p] for p in PAST], axis=-1)  # (B, H, W, 9)
    y = frames[idx + HORIZON]
    to = lambda a: torch.from_numpy(a).to(DEVICE).permute(0, 3, 1, 2).float() / 255
    return to(x), to(y)


def prep(clip):
    cap, out = cv2.VideoCapture(str(clip)), []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    FRAMES.parent.mkdir(exist_ok=True)
    np.save(FRAMES, np.stack(out))
    print(f"{len(out)} frames -> {FRAMES}")


def split(n, ahead=HORIZON):
    """Valid prediction times for training and held-out evaluation (split in time, no overlap)."""
    cut = int(n * TRAIN_FRACTION)
    return np.arange(max(PAST), cut - ahead), np.arange(cut + max(PAST), n - ahead)


def lineage_targets(frames, idx, steps):
    """Frames at every lineage horizon: (B, steps, 3, H, W) on DEVICE."""
    y = np.stack([frames[idx + LINEAGE_STRIDE * (k + 1)] for k in range(steps)], 1)
    return torch.from_numpy(y).to(DEVICE).permute(0, 1, 4, 2, 3).float() / 255


def train(iters, batch_size, lr, out, loss_name, resume, init_from):
    frames = np.load(FRAMES)
    model = load(out).train() if resume else NCA(out).to(DEVICE)
    if init_from:  # same rule, different read-out: start from another trained model's weights
        model.load_state_dict(torch.load(CHECKPOINTS[init_from], map_location=DEVICE, weights_only=True))
    ahead = LINEAGE_STRIDE * model.steps if out == "lineage" else HORIZON
    train_idx, test_idx = split(len(frames), ahead)
    loss_fn = F.l1_loss if loss_name == "l1" else F.mse_loss
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    rng = np.random.default_rng(0)
    probe = rng.choice(test_idx, 256, replace=False)
    start, run = time.monotonic(), []
    for it in range(1, iters + 1):
        idx = rng.choice(train_idx, batch_size)
        x, y = batch(frames, idx)
        if out == "lineage":  # a loss at every inner step, against the frame at that step's horizon
            flows, ys = model(x), lineage_targets(frames, idx, model.steps)
            loss = sum(loss_fn(warp(x[:, :3], flows[:, k]), ys[:, k]) for k in range(model.steps)) / model.steps
        else:
            loss = loss_fn(predict(model, x), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if it % 100 == 0 or it == iters:
            print(f"iter {it:5}  train {loss_name} x1000 {np.mean(run) * 1e3:.3f}  held-out skill {skill(model, frames, probe):+.3f}  "
                  f"{time.monotonic() - start:.0f}s", flush=True)
            run = []
            CHECKPOINTS[out].parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINTS[out])


def train_persistent(iters, batch_size, lr, loss_name, chunk=8, warmup=8, restart=1 / 16):
    """Streams run on through the training frames carrying their state; each iteration advances
    every stream `chunk` frames and backpropagates through those frames only. A stream restarts
    from zero at a random place with probability `restart` per iteration (or at the end of the
    training frames); its first `warmup` frames after a restart carry no loss."""
    frames = np.load(FRAMES)
    ahead = max(PERSISTENT_HORIZONS)
    train_idx, test_idx = split(len(frames), ahead + chunk)
    train_idx = train_idx[train_idx >= max(PERSISTENT_PAST)]
    model = PersistentNCA().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    loss_fn = F.l1_loss if loss_name == "l1" else F.mse_loss
    rng = np.random.default_rng(0)
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE).permute(0, 3, 1, 2).float() / 255
    pos, age = rng.choice(train_idx, batch_size), np.zeros(batch_size, int)
    state = torch.zeros(batch_size, model.hidden, H, W, device=DEVICE)
    start, run = time.monotonic(), []
    for it in range(1, iters + 1):
        for b in range(batch_size):
            if pos[b] > train_idx[-1] or rng.random() < restart:
                pos[b], age[b], state[b] = rng.choice(train_idx), 0, 0
        state, terms = state.detach(), []
        for c in range(chunk):
            frame = torch.cat([to(frames[pos + c - p]) for p in PERSISTENT_PAST], 1)
            state, flows = model(frame, state)
            frame = frame[:, :3]
            live = torch.from_numpy(age + c >= warmup).to(DEVICE)
            if live.any():
                for i, h in enumerate(PERSISTENT_HORIZONS):
                    terms.append(loss_fn(warp(frame[live], flows[live, i]), to(frames[pos + c + h])[live]))
        pos, age = pos + chunk, age + chunk
        if not terms:
            continue
        loss = sum(terms) / len(terms)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if it % 100 == 0 or it == iters:
            print(f"iter {it:5}  train {loss_name} x1000 {np.mean(run) * 1e3:.3f}  "
                  f"held-out skill at 1 s {probe_persistent(model, frames, test_idx):+.3f}  {time.monotonic() - start:.0f}s", flush=True)
            run = []
            torch.save(model.state_dict(), CHECKPOINTS["persistent"])


@torch.no_grad()
def probe_persistent(model, frames, test_idx, streams=8, length=96, warmup=32):
    """Quick held-out check: `streams` fixed runs of `length` frames from zero state, scored at 1 s after `warmup`."""
    starts = test_idx[np.linspace(0, len(test_idx) - length - 1, streams).astype(int)]
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE).permute(0, 3, 1, 2).float() / 255
    i1 = PERSISTENT_HORIZONS.index(HORIZON)
    state, err, copy = None, 0.0, 0.0
    for c in range(length):
        frame = torch.cat([to(frames[starts + c - p]) for p in PERSISTENT_PAST], 1)
        state, flows = model(frame, state)
        frame = frame[:, :3]
        if c >= warmup:
            y = to(frames[starts + c + HORIZON])
            err += ((warp(frame, flows[:, i1]) - y) ** 2).sum().item()
            copy += ((frame - y) ** 2).sum().item()
    return 1 - err / copy


@torch.no_grad()
def evaluate_persistent(every):
    """Run the persistent NCA non-stop through the whole held-out stretch from one zero start;
    score 1 s ahead on the E23 target frames, next to the lineage model; report by minute."""
    frames = np.load(FRAMES)
    model, lineage = load("persistent"), load("lineage")
    _, test_idx = split(len(frames), LINEAGE_STRIDE * lineage.steps)
    targets = set(test_idx[test_idx >= test_idx[0] + LINEAGE_STRIDE * 9][::every].tolist())  # as in E23
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE).permute(0, 3, 1, 2).float() / 255
    i1 = PERSISTENT_HORIZONS.index(HORIZON)
    rows, state, times = [], None, []  # (t, persistent sq err, lineage sq err, copy sq err, state rms)
    for t in range(test_idx[0], test_idx[-1] + 1):
        frame = torch.cat([to(frames[t - p:t - p + 1]) for p in PERSISTENT_PAST], 1)
        t0 = time.perf_counter()
        state, flows = model(frame, state)
        frame = frame[:, :3]
        if DEVICE == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        if t in targets:
            y = to(frames[t + HORIZON:t + HORIZON + 1])
            x, _ = batch(frames, np.array([t]))
            rows.append((t, ((warp(frame, flows[:, i1]) - y) ** 2).sum().item(),
                         ((predict(lineage, x) - y) ** 2).sum().item(), ((frame - y) ** 2).sum().item(),
                         state.pow(2).mean().sqrt().item()))
    r = np.array(rows)
    print(f"persistent NCA ({model.steps} steps/frame), one non-stop run through {len(times)} held-out frames "
          f"({len(times) / 30 / 60:.1f} min), scored on {len(r)} targets")
    print(f"at 1 s: persistent {1 - r[:, 1].sum() / r[:, 3].sum():+.3f}  vs lineage step 15 (E23) {1 - r[:, 2].sum() / r[:, 3].sum():+.3f}"
          f"  (same frames)  |  {np.median(times) * 1e3:.1f} ms/frame (median, batch 1)")
    minute = (r[:, 0] - test_idx[0]) // 1800
    for m in np.unique(minute):
        k = minute == m
        print(f"  minute {int(m) + 1}: persistent {1 - r[k, 1].sum() / r[k, 3].sum():+.3f}  lineage {1 - r[k, 2].sum() / r[k, 3].sum():+.3f}"
              f"  state rms {r[k, 4].mean():.3f} (max {r[k, 4].max():.3f})")


@torch.no_grad()
def skill(model, frames, idx, chunk=64):
    """1 - NCA MSE / copy-current-frame MSE, summed over frames idx (160x90)."""
    err = copy = 0.0
    for i in range(0, len(idx), chunk):
        x, y = batch(frames, idx[i:i + chunk])
        err += F.mse_loss(predict(model, x), y, reduction="sum").item()
        copy += F.mse_loss(x[:, :3], y, reduction="sum").item()
    return 1 - err / copy


def load(out="delta"):
    model = (PersistentNCA() if out == "persistent" else NCA(out)).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINTS[out], map_location=DEVICE, weights_only=True))
    return model.eval()


@torch.no_grad()
def evaluate(clip, every, out):
    frames = np.load(FRAMES)
    _, test_idx = split(len(frames))
    model = load(out)
    idx = test_idx[::every]
    # per frame at 160x90: motion (copy MSE), NCA MSE
    rows = []
    for i in range(0, len(idx), 64):
        x, y = batch(frames, idx[i:i + 64])
        pred = predict(model, x)
        rows += zip(((x[:, :3] - y) ** 2).mean((1, 2, 3)).tolist(), ((pred - y) ** 2).mean((1, 2, 3)).tolist())
    r = np.array(rows)
    print(f"{out} model, held-out last {1 - TRAIN_FRACTION:.0%}, {len(idx)} frames, horizon {HORIZON} ({HORIZON / 30:.1f} s)")
    print(f"160x90 MSE x1000: copy {r[:, 0].mean() * 1e3:.3f}  nca {r[:, 1].mean() * 1e3:.3f}  "
          f"skill {1 - r[:, 1].sum() / r[:, 0].sum():+.3f}  better on {(r[:, 1] < r[:, 0]).mean():.1%} of frames")
    bins = np.digitize(r[:, 0], np.percentile(r[:, 0], [20, 40, 60, 80]))
    for b in range(5):
        m = bins == b
        print(f"  motion quintile {b + 1}: copy {r[m, 0].mean() * 1e3:7.3f}  skill {1 - r[m, 1].sum() / r[m, 0].sum():+.3f}")

    # full resolution: the NCA's output enlarged and applied to the full-res current frame
    cap = cv2.VideoCapture(str(clip))
    full = {}
    want = set(idx[:200].tolist()) | {t + HORIZON for t in idx[:200].tolist()}
    for t in range(int(idx[:200].max()) + HORIZON + 1):
        ok, bgr = cap.read()
        if t in want:
            full[t] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    err = copy = 0.0
    for t in idx[:200].tolist():
        x, _ = batch(frames, np.array([t]))
        pred = full_res(model, x, full[t])
        err += ((pred - full[t + HORIZON]) ** 2).sum()
        copy += ((full[t] - full[t + HORIZON]) ** 2).sum()
    print(f"full res ({full[idx[0]].shape[1]}x{full[idx[0]].shape[0]}), first 200 of those frames: skill {1 - err / copy:+.3f}")


@torch.no_grad()
def evaluate_lineage(every, ensemble=10):
    """Lineage model on held-out frames: skill at every inner step; then, for the 1 s target, the
    `ensemble` predictions of the same frame made from frames 0, 2, 4, ... steps older (each at its
    longer horizon): combined, and their disagreement as a confidence map."""
    frames = np.load(FRAMES)
    model, warp_model = load("lineage"), load("warp")
    steps = model.steps
    _, test_idx = split(len(frames), LINEAGE_STRIDE * steps)
    oldest = LINEAGE_STRIDE * (ensemble - 1)  # how much older the oldest source frame is
    idx = test_idx[test_idx >= test_idx[0] + oldest][::every]
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE).permute(0, 3, 1, 2).float() / 255

    # 1. skill at every inner step, and the warp model at 1 s on the same frames
    err, copy = np.zeros(steps), np.zeros(steps)
    warp_err = 0.0
    for i in range(0, len(idx), 64):
        x, _ = batch(frames, idx[i:i + 64])
        flows, ys = model(x), lineage_targets(frames, idx[i:i + 64], steps)
        for k in range(steps):
            err[k] += ((warp(x[:, :3], flows[:, k]) - ys[:, k]) ** 2).sum().item()
            copy[k] += ((x[:, :3] - ys[:, k]) ** 2).sum().item()
        warp_err += ((predict(warp_model, x) - ys[:, lineage_step(HORIZON)]) ** 2).sum().item()
    k1 = lineage_step(HORIZON)
    print(f"lineage model, held-out, {len(idx)} frames")
    print("skill by horizon: " + "  ".join(f"{(k + 1) * LINEAGE_STRIDE / 30:.2f}s {1 - err[k] / copy[k]:+.3f}"
                                          for k in range(steps) if k % 2 == 1 or k == k1))
    print(f"at 1 s: lineage {1 - err[k1] / copy[k1]:+.3f}  vs warp model (E21) {1 - warp_err / copy[k1]:+.3f}  (same frames)")

    # 2. the same 1 s target predicted from `ensemble` source frames, newest first
    preds, truths, motion = [], [], []
    for t in idx:
        srcs = np.array([t - LINEAGE_STRIDE * j for j in range(ensemble)])
        x, _ = batch(frames, srcs)
        flows = model(x)
        preds.append(torch.stack([warp(x[j:j + 1, :3], flows[j:j + 1, k1 + j])[0] for j in range(ensemble)]).cpu().numpy())
        truths.append(frames[t + HORIZON].astype(np.float32).transpose(2, 0, 1) / 255)
        motion.append(np.abs(frames[t].astype(np.float32) - frames[t - 3]).mean(-1) / 255)  # control: moving now
    preds, truths, motion = np.stack(preds), np.stack(truths), np.stack(motion)  # (N, E, 3, H, W), (N, 3, H, W), (N, H, W)
    copy_sq = ((frames[idx].astype(np.float32).transpose(0, 3, 1, 2) / 255 - truths) ** 2)
    half = len(idx) // 2  # weights fitted on the first half, scored on the second
    mse_by_age = ((preds[:half] - truths[:half, None]) ** 2).mean((0, 2, 3, 4))
    w = (1 / mse_by_age) / (1 / mse_by_age).sum()
    combos = {"newest only": preds[:, 0], "plain average": preds.mean(1),
              "precision-weighted": np.tensordot(preds, w, axes=([1], [0]))}
    print(f"1 s target from {ensemble} source frames (newest to {oldest / 30:.2f} s older); scored on the second half "
          f"({len(idx) - half} targets); weights by age: " + " ".join(f"{v:.2f}" for v in w))
    for name, c in combos.items():
        sk = 1 - ((c[half:] - truths[half:]) ** 2).sum() / copy_sq[half:].sum()
        print(f"  {name:18} skill {sk:+.3f}")

    # 3. disagreement between the source predictions as a confidence map
    spread = preds[half:].std(1).mean(1)  # (N, H, W)
    actual = np.abs(preds[half:, 0] - truths[half:]).mean(1)  # error of the newest prediction
    for name, conf in (("ensemble spread", spread), ("motion now (control)", motion[half:])):
        r = np.corrcoef(conf.ravel(), actual.ravel())[0, 1]
        top = conf >= np.percentile(conf, 90)
        print(f"  {name:20} corr with actual error {r:.3f}; error in its top 10% pixels {actual[top].mean() / actual[~top].mean():.1f}x the rest")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep").add_argument("clip", type=Path)
    t = sub.add_parser("train")
    t.add_argument("--iters", type=int, default=2000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--out", choices=["delta", "warp", "lineage", "persistent"], default="delta")
    t.add_argument("--loss", choices=["mse", "l1"], default="mse")
    t.add_argument("--resume", action="store_true", help="continue from the saved checkpoint for --out")
    t.add_argument("--init-from", choices=["delta", "warp"], help="start from this model's weights")
    e = sub.add_parser("eval")
    e.add_argument("clip", type=Path)
    e.add_argument("--out", choices=["delta", "warp", "lineage", "persistent"], default="delta")
    e.add_argument("--every", type=int, default=5, help="evaluate every Nth held-out frame")
    args = parser.parse_args()
    if args.cmd == "prep":
        prep(args.clip)
    elif args.cmd == "train" and args.out == "persistent":
        train_persistent(args.iters, args.batch, args.lr, args.loss)
    elif args.cmd == "train":
        train(args.iters, args.batch, args.lr, args.out, args.loss, args.resume, args.init_from)
    elif args.out == "lineage":
        evaluate_lineage(args.every)
    elif args.out == "persistent":
        evaluate_persistent(args.every)
    else:
        evaluate(args.clip, args.every, args.out)


if __name__ == "__main__":
    main()
