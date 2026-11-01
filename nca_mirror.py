# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "opencv-python", "torch", "scikit-image"]
# ///
"""Full-colour mirror that shows the NCA's prediction of the camera frame 1 s ahead (nca.py).

The NCA works at 160x90; its output (a motion field by default, or a colour change with
`--out delta`) is enlarged and applied to the full-res current frame.

With `--out lineage` (E23) the NCA's inner steps predict 66 ms ... 1.6 s ahead, so each shown frame
averages the predictions of the same moment made from the last ENSEMBLE source frames (newest at
1 s ahead, oldest at 1.6 s). Extra views: tracers (the current frame pushed along its own lineage,
now to 1 s, layered) and confidence (purple ghosts where those predictions disagree). Its shown
prediction mixes four sources - copy / newest NCA / 10-guess average / rigid-superpixel NCA -
(`m` key): gate = per-patch weights from a small gate (a 7 -> 32 -> 4 MLP, E35) on signals known
now (recent motion, the 66 ms errors, the NCA's motion, its gap to the rigid version, the spread
of the guesses; state/nca_gate.npz from `nca_mix.py save-gate`); blend = fixed weights (E30); crisp = one warp
along the fixed-blended motion of copy, NCA and rigid (sharpest, lower skill); nca10 = the plain
10-guess average (E23).

With `--out persistent` (E24) the NCA never stops: its state carries over from frame to frame for
as long as the mirror runs, and tracers layer its 333 ms, 667 ms and 1 s predictions.

With `--out graph` (E26, graph_nca.py) the cells are ~240 superpixels found fresh each frame;
every region moves as a rigid piece, or blended across borders with `--smooth S` (E27, the model
trained with that blur). The regions view draws their borders.
The status line scores each prediction against the frame that actually arrived 1 s later
(skill = 1 - error / error of copying the current frame, over the last 3 s).

With --learn it keeps learning from what it sees: each prediction and the frame that arrived 1 s
later go into a buffer of the last 60 s, which a background thread trains a CPU copy of the NCA
on (so the display keeps the GPU); the display picks up the new weights every 20 steps. It starts
from the last live session if there is one (state/nca_<out>_live.pt), else from the recording-
trained model, and saves to the live file every 100 steps and on quit; the recording-trained
model is never changed.

  uv run nca_mirror.py [--out warp|delta|lineage|persistent|graph [--smooth S]] [--learn] [--replay recordings/x.mp4]

Keys: v  view: prediction / [tracers / confidence /] side by side (now | 1 s ahead) / camera
      [ ]  exaggerate the predicted motion for display (x0.5 ... x4; the score always uses x1, E25)
      , .  lineage: pull the motion toward its average within each superpixel (0 ... 1, E28); this
           changes the prediction itself, so the score follows it
      m    lineage: mix = gate / blend / crisp / nca10 (E30, E32); the score follows it
      r    start / stop recording the camera to recordings/ (640x360 at camera rate, as the E12 clip;
           a thread writes every camera frame, not just the ones the display loop reaches)
      q  quit
"""
import argparse
import copy
import os
import select
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import graph_nca
import nca
import psychomanteum as pm

VIEWS = ["prediction", "side by side", "camera"]
LINEAGE_VIEWS = ["prediction", "tracers", "confidence", "side by side", "camera"]
PERSISTENT_VIEWS = ["prediction", "tracers", "side by side", "camera"]
GRAPH_VIEWS = ["prediction", "regions", "side by side", "camera"]
ENSEMBLE = 10  # lineage: source frames per shown frame (every LINEAGE_STRIDE steps back, E23)
GAINS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)  # display multipliers for the predicted motion ([ ] keys)
GHOST = np.array([0.85, 0.35, 1.0], np.float32)  # confidence view: colour of disagreement
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)  # lineage: pull toward superpixel averages (, . keys)
MIXES = ("gate", "blend", "crisp", "nca10")  # lineage: how the shown prediction is made (m key)
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
GATE_FILE = nca.ROOT / "state" / "nca_gate.npz"
W_BLEND = {"copy": 0.229, "nca": 0.262, "nca10": 0.270, "rigid": 0.240}  # E30 "A fixed": 1 / mean patch error, half 1
W_CRISP = {"nca": 0.359, "rigid": 0.328}  # E30 "B fixed" motion blend; copy's 0.313 is zero motion


def region_pull(flow, labels, alpha):
    """flow (H, W, 2) pulled toward its mean within each superpixel of `labels` by alpha (E28)."""
    if alpha == 0 or labels is None:
        return flow
    n, flat = int(labels.max()) + 1, labels.ravel()
    count = np.bincount(flat, minlength=n)[:, None]
    mean = np.stack([np.bincount(flat, flow[..., c].ravel(), n) for c in range(2)], 1) / np.maximum(count, 1)
    return (1 - alpha) * flow + alpha * mean[labels].astype(np.float32)


def patch_mean(a):
    """(H, W) -> (P, Q): mean over whole 8x8 patches."""
    h, w = (a.shape[0] // 8) * 8, (a.shape[1] // 8) * 8
    return a[:h, :w].reshape(h // 8, 8, w // 8, 8).mean((1, 3))


def upsample(w, h, wd):
    """Patch weights (P, Q, S) -> per-pixel (h, wd, S): bilinear over the whole patches, bottom rows replicated."""
    hh = w.shape[0] * 8 * h // nca.H
    up = cv2.resize(w, (wd, hh), interpolation=cv2.INTER_LINEAR)
    return np.pad(up, ((0, h - hh), (0, 0), (0, 0)), mode="edge")


def gate_weights(history, f0, fr, guesses_low, gate):
    """Per-patch source weights (P, Q, 4) from the E32 gate; features as in nca_mix.FEATURES, at NCA resolution."""
    lo, lo2, lo3, lo6 = history[-1][1], history[-3][1], history[-4][1], history[-7][1]
    pred66 = nca.warp_full(lo2, history[-3][2][0])  # the 66 ms prediction made 2 frames ago; its target is lo
    luma = lambda im: im @ LUMA
    feats = np.stack([
        patch_mean(np.abs(luma(lo) - luma(lo3))),
        patch_mean(np.abs(luma(lo3) - luma(lo6))),
        patch_mean(((lo2 - lo) ** 2).mean(-1)),
        patch_mean(((pred66 - lo) ** 2).mean(-1)),
        patch_mean(np.linalg.norm(f0, axis=-1)),
        patch_mean(np.linalg.norm(f0 - fr, axis=-1)),
        patch_mean(np.std(guesses_low, 0).mean(-1)),
    ], -1)
    z = (np.log(feats + 1e-4) - gate["mu"]) / gate["sd"]
    logits = (np.maximum(z @ gate["W1"].T + gate["b1"], 0) @ gate["W2"].T + gate["b2"]) if "W1" in gate else z @ gate["W"].T + gate["b"]
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)).astype(np.float32)


def lineage_views(history, k1, with_tracers, gain, alpha, mix, gate):
    """history: deque of (full-res frame float 0-1, same at NCA resolution, lineage flows (steps, H, W, 2),
    superpixel labels or None), newest last. Returns the prediction (per `mix`, E30) with motion x gain,
    the tracers image (None unless asked for), the disagreement map (H, W) between the 10 guesses, and
    the same prediction at NCA resolution with motion x1 (for scoring). Motion is pulled toward
    superpixel averages by alpha."""
    sources = list(history)[::-nca.LINEAGE_STRIDE][:ENSEMBLE]
    flows1 = [region_pull(flows[k1 + j], lab, alpha) for j, (_, _, flows, lab) in enumerate(sources)]
    preds = [nca.warp_full(f, gain * fl) for (f, _, _, _), fl in zip(sources, flows1)]
    guesses_low = [nca.warp_full(lo, fl) for (_, lo, _, _), fl in zip(sources, flows1)]
    nca10_low = np.mean(guesses_low, 0)
    nca10 = np.mean(preds, 0)
    spread = np.std(preds, 0).mean(-1) if len(preds) > 1 else np.zeros(nca10.shape[:2], np.float32)
    now, lo, flows, lab = history[-1]
    if mix == "nca10":
        pred, pred_low = nca10, nca10_low
    else:
        f0, fr = flows1[0], region_pull(flows1[0], lab, 1.0)  # newest motion, and it averaged per superpixel
        if mix == "crisp":
            fm = W_CRISP["nca"] * f0 + W_CRISP["rigid"] * fr
            pred, pred_low = nca.warp_full(now, gain * fm), nca.warp_full(lo, fm)
        else:  # blend or gate: the four source images, in nca_mix.SOURCES order
            full = [now, nca.warp_full(now, gain * f0), nca10, nca.warp_full(now, gain * fr)]
            small = [lo, nca.warp_full(lo, f0), nca10_low, nca.warp_full(lo, fr)]
            if mix == "gate" and gate is not None and len(history) >= 7:
                w = gate_weights(history, f0, fr, guesses_low, gate)
                wf, wl = upsample(w, *now.shape[:2]), upsample(w, *lo.shape[:2])
                pred = sum(wf[..., i:i + 1] * full[i] for i in range(4))
                pred_low = sum(wl[..., i:i + 1] * small[i] for i in range(4))
            else:
                wts = [W_BLEND[k] for k in ("copy", "nca", "nca10", "rigid")]
                pred = sum(wt * im for wt, im in zip(wts, full))
                pred_low = sum(wt * im for wt, im in zip(wts, small))
    if not with_tracers:
        return pred, None, spread, pred_low
    tracers = np.mean([now] + [nca.warp_full(now, gain * region_pull(flows[k], lab, alpha)) for k in range(1, k1 + 1, 2)], 0)
    return pred, tracers, spread, pred_low


class Recorder(threading.Thread):
    """Writes every camera frame to an mp4, independently of the display loop's rate (E34: the
    first recording from the display loop came out at ~10 fps)."""

    def __init__(self, source, path):
        super().__init__(daemon=True)
        self.source = source
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (pm.W, pm.H))
        self.stopping, self.frames, self.started = threading.Event(), 0, time.monotonic()
        self.start()

    def run(self):
        seq = 0
        while not self.stopping.is_set():
            new_seq, frame = self.source.next(seq)
            if new_seq != seq and frame is not None:
                self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.frames += 1
            seq = new_seq
        self.writer.release()

    def fps(self):
        return self.frames / max(time.monotonic() - self.started, 1e-6)

    def release(self):
        self.stopping.set()
        self.join(timeout=2)


class LiveLearner(threading.Thread):
    """Keeps training a CPU copy of the NCA on recent live (input, frame 1 s later) pairs."""

    def __init__(self, model, save_to, lr=1e-4, batch=8, seconds=60):
        super().__init__(daemon=True)
        self.save_to = save_to
        self.model = copy.deepcopy(model).to("cpu").train()
        self.opt = torch.optim.Adam(self.model.parameters(), lr)
        self.loss_fn = F.l1_loss if model.out == "warp" else F.mse_loss  # as in training (E21)
        self.pairs = deque(maxlen=seconds * 30)  # (inputs uint8 (H, W, 9), frame 1 s later uint8 (H, W, 3))
        self.batch, self.steps, self.fresh = batch, 0, None
        self.lock = threading.Lock()

    def run(self):
        torch.set_num_threads(4)
        rng = np.random.default_rng()
        while True:
            pairs = list(self.pairs)
            if len(pairs) < 4 * self.batch:
                time.sleep(0.1)
                continue
            pick = rng.choice(len(pairs), self.batch, replace=False)
            to = lambda a: torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).float() / 255
            x, y = to([pairs[i][0] for i in pick]), to([pairs[i][1] for i in pick])
            loss = self.loss_fn(nca.predict(self.model, x), y)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.steps += 1
            if self.steps % 20 == 0:
                with self.lock:
                    self.fresh = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            if self.steps % 100 == 0:
                torch.save(self.fresh or self.model.state_dict(), self.save_to)  # so a crash keeps most of a session

    def take(self):
        """New weights since the last call, or None."""
        with self.lock:
            fresh, self.fresh = self.fresh, None
        return fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", choices=["warp", "delta", "lineage", "persistent", "graph"], default="warp",
                        help="which trained NCA to use")
    parser.add_argument("--learn", action="store_true", help="keep learning from the live input")
    parser.add_argument("--smooth", type=float, default=0.0, help="graph: the model trained with this motion blur (cells)")
    parser.add_argument("--replay", type=Path, help="play this recorded clip in a loop instead of the webcam")
    args = parser.parse_args()
    if args.learn and args.out in ("persistent", "graph"):
        parser.error(f"--learn is not supported for the {args.out} NCA")

    pm.fd = os.open("/dev/tty", os.O_RDWR)
    source = pm.Replay(args.replay) if args.replay else pm.Camera()
    if args.out == "graph":
        from skimage.segmentation import find_boundaries
        model = graph_nca.load(args.smooth)
    else:
        model = nca.load(args.out)
    live_file = nca.CHECKPOINTS[args.out].with_name(f"{nca.CHECKPOINTS[args.out].stem}_live.pt") if args.learn else None
    learner = None
    if args.learn:
        if live_file.exists():
            model.load_state_dict(torch.load(live_file, map_location=nca.DEVICE, weights_only=True))
        learner = LiveLearner(model, live_file)
        learner.start()
    small = deque(maxlen=max(nca.PAST) + 1)  # recent frames at NCA resolution
    pending = {}  # step -> (prediction, current frame, inputs) at NCA resolution, scored HORIZON steps later
    scores = deque(maxlen=90)  # (nca sq err, copy sq err) over the last 3 s
    view, seq, step, layout, times = 0, 0, 0, None, deque(maxlen=30)
    gain = 1.0  # display multiplier for the predicted motion
    alpha = 0.0  # lineage: pull toward superpixel averages
    gate = None
    mixes = MIXES
    if args.out == "lineage" and GATE_FILE.exists():
        g = np.load(GATE_FILE)
        gate = {k: g[k].astype(np.float32) for k in g.files if k not in ("features", "sources")}
    elif args.out == "lineage":
        mixes = MIXES[1:]  # no saved gate: run `nca_mix.py save-gate`
    mix = mixes[0]  # lineage: gate / blend / crisp / nca10
    recorder = None
    views = {"lineage": LINEAGE_VIEWS, "persistent": PERSISTENT_VIEWS, "graph": GRAPH_VIEWS}.get(args.out, VIEWS)
    nca_state = None  # persistent: carried over between frames
    history = deque(maxlen=nca.LINEAGE_STRIDE * (ENSEMBLE - 1) + 1)  # lineage: recent (frame, flows)

    old = termios.tcgetattr(pm.fd)
    tty.setcbreak(pm.fd)
    pm.write(b"\x1b[?25l\x1b[2J")
    try:
        while True:
            if select.select([pm.fd], [], [], 0)[0]:
                for key in os.read(pm.fd, 64).decode(errors="ignore"):
                    if key == "q":
                        return
                    if key == "v":
                        view = (view + 1) % len(views)
                        layout = None
                        pm.write(b"\x1b[2J")
                    if key in "[]":
                        gain = GAINS[min(max(GAINS.index(gain) + (1 if key == "]" else -1), 0), len(GAINS) - 1)]
                    if key == "r" and not args.replay:  # camera only: a second reader would advance a replay
                        if recorder:
                            recorder.release()
                            recorder = None
                        else:
                            pm.RECORDINGS.mkdir(exist_ok=True)
                            recorder = Recorder(source, pm.RECORDINGS / time.strftime("clip_%Y%m%d_%H%M%S.mp4"))
                    if key == "m" and args.out == "lineage":
                        mix = mixes[(mixes.index(mix) + 1) % len(mixes)]
                    if key in ",." and args.out == "lineage":
                        alpha = ALPHAS[min(max(ALPHAS.index(alpha) + (1 if key == "." else -1), 0), len(ALPHAS) - 1)]
            seq, frame = source.next(seq)
            t0 = time.monotonic()
            low = cv2.resize(frame, (nca.W, nca.H), interpolation=cv2.INTER_AREA)
            small.append(low)
            if step in pending:
                pred, then, inputs = pending.pop(step)
                truth = low.astype(np.float32) / 255
                scores.append((((pred - truth) ** 2).mean(), ((then - truth) ** 2).mean()))
                if learner:
                    learner.pairs.append((inputs, low))
            fresh = learner.take() if learner else None
            if fresh:
                model.load_state_dict(fresh)

            shown = frame
            if len(small) == small.maxlen:
                x = np.concatenate([small[-1 - p] for p in nca.PAST], axis=-1)[None]
                with torch.no_grad():
                    xt = torch.from_numpy(x).to(nca.DEVICE).permute(0, 3, 1, 2).float() / 255
                    if args.out == "graph":
                        labels = graph_nca.superpixels(low).astype(np.int64)
                        feats, pos, edges = graph_nca.graph(labels, [small[-1 - p] for p in nca.PAST])
                        to = lambda a: torch.from_numpy(a).to(nca.DEVICE)
                        node_flow = model(to(feats), to(pos), to(edges))
                        # every pixel moves with its region (blurred across borders if the model was trained so)
                        flow = graph_nca.pixel_flow(node_flow, to(labels)[None], model.smooth)[0].permute(1, 2, 0).cpu().numpy()
                        out = None
                    elif args.out == "persistent":
                        now_then = np.concatenate([small[-1 - p] for p in nca.PERSISTENT_PAST], axis=-1)[None]
                        nt = torch.from_numpy(now_then).to(nca.DEVICE).permute(0, 3, 1, 2).float() / 255
                        nca_state, flows = model(nt, nca_state)
                        flows = flows[0].permute(0, 2, 3, 1).cpu().numpy()  # (horizons, H, W, 2)
                        out = None
                    else:
                        out = model(xt)  # once per frame: the NCA is most of the frame time
                    if args.out not in ("lineage", "persistent", "graph"):
                        pred_low = nca.predict(model, xt, out)[0].permute(1, 2, 0).cpu().numpy()
                        pred_full = nca.full_res(model, xt, frame.astype(np.float32) / 255, gain * out)
                if args.out == "graph":
                    pred_full = nca.warp_full(frame.astype(np.float32) / 255, gain * flow)
                    pred_low = nca.warp_full(low.astype(np.float32) / 255, flow)
                    if views[view] == "regions":
                        edge = cv2.resize(find_boundaries(labels).astype(np.uint8), (frame.shape[1], frame.shape[0]),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
                        tracers = pred_full.copy()  # reuse the tracers slot for the regions view
                        tracers[edge] = (1.0, 0.85, 0.2)
                elif args.out == "persistent":
                    now = frame.astype(np.float32) / 255
                    warps = [nca.warp_full(now, gain * f) for f in flows]
                    i1 = nca.PERSISTENT_HORIZONS.index(nca.HORIZON)
                    pred_full = warps[i1]
                    tracers = np.mean([now] + warps, 0)
                    pred_low = nca.warp_full(low.astype(np.float32) / 255, flows[i1])
                elif args.out == "lineage":
                    labels = graph_nca.superpixels(low).astype(np.int64) if (alpha > 0 or mix != "nca10") else None
                    history.append((frame.astype(np.float32) / 255, low.astype(np.float32) / 255,
                                    out[0].permute(0, 2, 3, 1).cpu().numpy(), labels))
                    pred_full, tracers, spread, pred_low = lineage_views(history, nca.lineage_step(nca.HORIZON),
                                                                         views[view] == "tracers", gain, alpha, mix, gate)
                pending[step + nca.HORIZON] = (pred_low, low.astype(np.float32) / 255, x[0])
                to8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
                name = views[view]
                if name in ("tracers", "regions"):
                    shown = to8(tracers)
                elif name == "confidence":
                    ghost = np.clip((spread - 0.01) * 12, 0, 0.8)[..., None]
                    shown = to8(pred_full * (1 - ghost) + GHOST * ghost)
                else:
                    shown = {"prediction": to8(pred_full), "side by side": np.hstack([frame, to8(pred_full)]),
                             "camera": frame}[name]
            step += 1

            if layout is None:
                layout = pm.layout(shown.shape[1], shown.shape[0])
            cols, img_cols, img_rows = layout
            pm.show(np.ascontiguousarray(shown), img_cols, img_rows)
            times.append(time.monotonic() - t0)
            if scores:
                s = np.array(scores)
                score = f"skill vs copying now {1 - s[:, 0].sum() / max(s[:, 1].sum(), 1e-9):+.2f} (last 3 s)"
            else:
                score = "warming up (first score after 1 s)"
            learning = f" | learned {learner.steps} steps" if learner else ""
            shown_gain = f" | motion x{gain:g} (shown)" if gain != 1 else ""
            shown_gain += f" | regions {alpha:g}" if alpha else ""
            shown_gain += f" | mix {mix}" if args.out == "lineage" else ""
            shown_gain += f" | REC {recorder.frames} frames ({recorder.fps():.0f} fps)" if recorder else ""
            status = (f" NCA mirror ({args.out}) | {views[view]} | 1 s ahead{shown_gain} | {score}{learning} | "
                      f"{np.mean(times) * 1e3:.0f} ms/frame | v view  [ ] exaggerate{'  , . regions  m mix' if args.out == 'lineage' else ''}  q quit")
            pm.write(f"\x1b[{img_rows + 1};1H\x1b[2K{status[:cols]}".encode())
    finally:
        if recorder:
            recorder.release()
        if learner and learner.steps:
            torch.save(learner.model.state_dict(), live_file)
        pm.write(b"\x1b_Ga=d,d=A,q=2\x1b\\\x1b[2J\x1b[H\x1b[?25h")
        termios.tcsetattr(pm.fd, termios.TCSADRAIN, old)
        source.cap.release()


if __name__ == "__main__":
    main()
