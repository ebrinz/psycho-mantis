# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "opencv-python"]
# ///
"""Psychomanteum: a predictive webcam mirror drawn in Ghostty with Kitty graphics.

Top: what each "eye" sees (left red, right blue). Both eyes see the full scene at low resolution
(`eye_width`, shown as enlarged blocky pixels), each through its own noisy, randomly wandering
delay (LGN-like routing).
Middle: each eye's V1-like predictive-coding layer, shown as its 64 learned features.
Bottom (purple): the prediction of the frame `horizon` steps ahead, at eye resolution, smoothly
enlarged: per patch, a precision-weighted blend of the fused two-eye model, the eyes' average
pushed along its own motion (a "stabilized map"), and the plain eyes' average. Where earlier predictions missed what the camera actually saw, a
ghost appears. The `flow` view shows the optical-flow baseline on the undelayed full-res frame.

Closing the loop: every second the mirror appends its prediction stats to
state/stats.jsonl, and it hot-reloads its parameters from state/control.json.
mantis.py is the other side (observe / set), driven by Claude Code.

Usage:
  uv run psychomanteum.py                          live webcam
  uv run psychomanteum.py --replay recordings/x.mp4  replay a clip in a loop (same input every run)

Keys:
  d      dither the bottom panel on / off (the eyes are never dithered)
  f      learned features on / off
  e      eyes used by the displayed prediction: both / left / right
  v      bottom view: prediction (fused model) / flow (baseline) / camera / error
  + / -  prediction horizon (steps ahead)
  r      start / stop recording the camera to recordings/
  q      quit
"""
import argparse
import base64
import fcntl
import json
import os
import select
import struct
import termios
import threading
import time
import tty
from collections import deque
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 360
FLOW_SCALE = 0.5  # optical flow runs at half resolution
VIEWS = ["prediction", "flow", "camera", "error"]  # prediction = fused two-eye model; flow = full-res baseline
EYE_MODES = {"both": (0, 1), "left": (0,), "right": (1,)}
CHOICES = {"view": VIEWS, "fusion_eyes": list(EYE_MODES)}
LUMA = np.array([0.299, 0.587, 0.114], np.float32)

ROOT = Path(__file__).parent
STATE, RECORDINGS = ROOT / "state", ROOT / "recordings"
CONTROL, STATS = STATE / "control.json", STATE / "stats.jsonl"
FEATURES_FILE, FUSION_FILE = STATE / "v1_features.npz", STATE / "fusion.npz"
PATCH, N_FEATURES = 8, 64
FEATURES_PX = 180  # display size of each eye's feature grid

# name: (default, min, max) for numbers, (default,) otherwise
PARAMS = {
    "horizon": (15, 1, 15),  # steps ahead to predict (15 = 500 ms at 30 fps)
    "damping": (1.0, 0.0, 1.5),  # scales the extrapolated velocity; < 1 hedges against overshoot
    "flow_smoothing": (0.0, 0.0, 0.95),  # exponential average of the flow field across steps
    "flow_window": (15, 5, 41),  # optical-flow averaging window (pixels at half resolution)
    "eye_max_delay": (4, 0, 15),  # each eye's delay wanders between 0 and this many steps
    "eye_jitter": (0.2, 0.0, 1.0),  # chance per step that an eye's delay moves by one step
    "eye_noise": (8.0, 0.0, 50.0),  # pixel noise (std) added to each eye's input
    "eye_width": (80, 8, 320),  # each eye's resolution in pixels across (16:9), like a coarse retina
    "pc_rate": (0.2, 0.0, 1.0),  # predictive-coding learning rate (0 freezes the learned features)
    "pc_sparsity": (1.0, 0.0, 3.0),  # how few features may explain each patch (higher = sparser)
    "fusion_rate": (1.0, 0.0, 1.0),  # learning rate of the two-eye fused predictor (0 freezes it; E13)
    "fusion_eyes": ("both",),  # which eyes the displayed prediction uses: both / left / right (all three learn)
    "eyeflow_lead": (2.0, 0.0, 10.0),  # push the eyes' average along its motion by horizon + this (eye lag, E16)
    "eyeflow_smoothing": (0.5, 0.0, 0.95),  # running average of the eyes' flow: 0.5 for noisy eyes (E18), 0 for clean (E20)
    "features": (True,),  # show each eye's learned features
    "ghost_gain": (4.0, 0.0, 20.0),  # visual only: how strongly error shows as ghosts
    "ghost_floor": (0.03, 0.0, 0.2),  # visual only: error below this is camera noise, not a ghost
    "dither": (True,),
    "view": ("prediction",),
    "caption": ("",),  # a line of text under the mirror
}
MAX_EYE_DELAY = PARAMS["eye_max_delay"][2]

# 4-tone palettes, dark to light: left eye red, right eye blue, fused model purple
RED = np.array([[40, 5, 10], [110, 20, 30], [200, 50, 60], [255, 130, 130]], np.uint8)
BLUE = np.array([[5, 10, 40], [20, 40, 120], [50, 110, 220], [140, 190, 255]], np.uint8)
PURPLE = np.array([[25, 5, 40], [80, 25, 120], [160, 70, 210], [230, 160, 255]], np.uint8)
GHOST = np.array([200, 255, 255], np.uint8)

fd = None  # /dev/tty, opened in main()


def write(data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def sanitize(raw):
    """Fill in defaults, coerce types and clamp ranges so a bad control file can't break the mirror."""
    params = {}
    for name, spec in PARAMS.items():
        default, value = spec[0], raw.get(name, spec[0])
        try:
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, (int, float)):
                value = min(spec[2], max(spec[1], type(default)(value)))
            elif name in CHOICES:
                value = value if value in CHOICES[name] else default
            else:
                value = str(value)[:120]
        except (TypeError, ValueError):
            value = default
        params[name] = value
    return params


def save_control(params):
    STATE.mkdir(exist_ok=True)
    tmp = CONTROL.with_suffix(".tmp")
    tmp.write_text(json.dumps(params, indent=2))
    tmp.replace(CONTROL)


def load_control():
    try:
        return sanitize(json.loads(CONTROL.read_text())), None
    except FileNotFoundError:
        return sanitize({}), None
    except (json.JSONDecodeError, OSError) as e:
        return None, f"control.json ignored: {e}"


def bayer(n):
    m = np.array([[0, 2], [3, 1]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size


THRESHOLD = np.tile(bayer(8), (H // 8 + 1, W // 8 + 1))[:H, :W]


def ramp(palette):
    """256-step colour ramp through a palette, for clean (undithered) tinting."""
    stops = np.linspace(0, 255, len(palette))
    return np.stack([np.interp(np.arange(256), stops, palette[:, c]) for c in range(3)], axis=1).astype(np.uint8)


RAMPS = {id(p): ramp(p) for p in (RED, BLUE, PURPLE)}


def tint(rgb, palette, dithered):
    """Recolour by luminance through a palette: ordered-dithered to its 4 tones, or as a smooth ramp."""
    lum = rgb @ LUMA / 255
    if dithered:
        h, w = lum.shape
        idx = np.clip((lum * (len(palette) - 1) + THRESHOLD[:h, :w]).astype(int), 0, len(palette) - 1)
        return palette[idx]
    return RAMPS[id(palette)][(lum * 255).astype(np.uint8)]


class Camera:
    """Grabs frames on a background thread so the render loop always gets the latest one."""

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise SystemExit("Could not open webcam (check camera permission for Ghostty).")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame, self.seq = None, 0
        self.cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ok, f = self.cap.read()
            if not ok:
                continue
            f = cv2.flip(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA), 1)
            with self.cond:
                self.frame, self.seq = cv2.cvtColor(f, cv2.COLOR_BGR2RGB), self.seq + 1
                self.cond.notify_all()

    def next(self, last_seq):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout=1)
            return self.seq, self.frame


class Replay:
    """Plays a recorded clip in a loop: every frame in order, at most 30 fps, so every run sees identical input."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open {path}")
        self.seq, self.due = 0, time.monotonic()

    def next(self, last_seq):
        ok, f = self.cap.read()
        if not ok:  # end of clip: loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        time.sleep(max(0.0, self.due - time.monotonic()))
        self.due = max(self.due + 1 / 30, time.monotonic())
        self.seq += 1
        return self.seq, cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


class Eyes:
    """Two eyes, each seeing the full scene at low resolution through its own noisy, wandering delay."""

    def __init__(self):
        self.rng = np.random.default_rng(0)  # seeded so replays get the same delays and noise
        self.history = deque(maxlen=MAX_EYE_DELAY + 1)
        self.delays = [0, 0]

    def see(self, frame, p):
        self.history.append(cv2.resize(frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
        size = (p["eye_width"], max(1, round(p["eye_width"] * 9 / 16)))
        views = []
        for e in range(2):
            if self.rng.random() < p["eye_jitter"]:
                self.delays[e] += int(self.rng.choice((-1, 1)))
            self.delays[e] = int(np.clip(self.delays[e], 0, min(p["eye_max_delay"], len(self.history) - 1)))
            seen = cv2.resize(self.history[-1 - self.delays[e]], size, interpolation=cv2.INTER_AREA)
            if p["eye_noise"] > 0:
                noise = self.rng.standard_normal(seen.shape, dtype=np.float32) * p["eye_noise"]
                seen = np.clip(seen + noise, 0, 255).astype(np.uint8)
            views.append(seen)
        return views


def patchify(gray):
    """Non-overlapping 8x8 patches of a grayscale image as rows (n_patches, 64), in reading order."""
    h, w = (gray.shape[0] // PATCH) * PATCH, (gray.shape[1] // PATCH) * PATCH
    return gray[:h, :w].reshape(h // PATCH, PATCH, w // PATCH, PATCH).swapaxes(1, 2).reshape(-1, PATCH * PATCH)


def unpatchify(patches, shape):
    """Inverse of patchify for an image of the given (h, w), cropped to whole patches."""
    rows, cols = shape[0] // PATCH, shape[1] // PATCH
    return patches.reshape(rows, cols, PATCH, PATCH).swapaxes(1, 2).reshape(rows * PATCH, cols * PATCH)


class SparseCoder:
    """One eye's predictive-coding layer (Olshausen & Field 1996, Rao & Ballard 1999).

    Each 8x8 patch x is explained by a sparse code a over learned features phi. The prediction
    phi a goes top-down; the residual x - phi a comes back up and drives both inference (which
    features are active) and learning (phi += rate * residual a^T, a local Hebbian-style rule).
    """

    def __init__(self, phi):
        self.phi = phi
        self.rel_err, self.active = 1.0, 0.0
        self.codes = self.raw = None  # last step's sparse codes and raw (unfiltered) patches

    @staticmethod
    def fresh(rng):
        phi = rng.standard_normal((PATCH * PATCH, N_FEATURES)).astype(np.float32)
        return SparseCoder(phi / np.linalg.norm(phi, axis=0))

    def step(self, eye, p, iters=30):
        gray = eye.astype(np.float32) @ LUMA / 255
        self.raw = patchify(gray)
        x = patchify(gray - cv2.GaussianBlur(gray, (0, 0), 1.5))  # centre-surround, like retina/LGN: keeps edges
        if len(x) == 0:
            self.codes = None
            return
        x = x - x.mean(axis=1, keepdims=True)  # contrast only, not brightness
        x /= x.std() + 1e-6

        # inference: iterative shrinkage, i.e. predict top-down, correct by the bottom-up residual
        step = 1 / np.linalg.norm(self.phi, 2) ** 2
        a = np.zeros((len(x), N_FEATURES), np.float32)
        for _ in range(iters):
            residual = x - a @ self.phi.T
            a += step * residual @ self.phi
            a = np.sign(a) * np.maximum(np.abs(a) - step * p["pc_sparsity"], 0)
        residual = x - a @ self.phi.T

        # learning: each feature moves along the residual it was active for, then is renormalised
        self.phi += p["pc_rate"] * residual.T @ a / len(x)
        self.phi /= np.linalg.norm(self.phi, axis=0) + 1e-6
        self.rel_err = float((residual ** 2).sum() / ((x ** 2).sum() + 1e-6))
        self.active = float((a != 0).mean())
        self.codes = a

    def image(self, palette):
        """The 64 learned features as an 8x8 grid of 8x8 tiles, mid-tone = 0."""
        tiles = self.phi.T.reshape(N_FEATURES, PATCH, PATCH)
        tiles = tiles / (np.abs(tiles).max(axis=(1, 2), keepdims=True) + 1e-6) * 0.5 + 0.5
        side = int(np.sqrt(N_FEATURES))
        grid = np.zeros((side * (PATCH + 1) - 1,) * 2, np.float32)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, side)
            grid[r * (PATCH + 1):r * (PATCH + 1) + PATCH, c * (PATCH + 1):c * (PATCH + 1) + PATCH] = tile
        img = RAMPS[id(palette)][(grid * 255).astype(np.uint8)]
        return cv2.resize(img, (FEATURES_PX, FEATURES_PX), interpolation=cv2.INTER_NEAREST)


def load_learned():
    """Each eye's V1 layer and the fused predictor, as learned in earlier runs (or fresh)."""
    try:
        saved = np.load(FEATURES_FILE)
        coders = [SparseCoder(saved["left"]), SparseCoder(saved["right"])]
    except (FileNotFoundError, KeyError, ValueError):
        rng = np.random.default_rng(1)
        coders = [SparseCoder.fresh(rng), SparseCoder.fresh(rng)]
    fusions = {mode: FusionPredictor(use) for mode, use in EYE_MODES.items()}
    try:
        saved = np.load(FUSION_FILE)
        for mode, fusion in fusions.items():
            key = f"w_{mode}" if f"w_{mode}" in saved else ("w" if mode == "both" and "w" in saved else None)
            if key and saved[key].shape == fusion.w.shape:
                fusion.w = saved[key]
    except (FileNotFoundError, ValueError):
        pass
    return coders, fusions


def save_learned(coders, fusions):
    STATE.mkdir(exist_ok=True)
    np.savez(FEATURES_FILE, left=coders[0].phi, right=coders[1].phi)
    np.savez(FUSION_FILE, **{f"w_{mode}": f.w for mode, f in fusions.items()})


class FusionPredictor:
    """Fuses both eyes' V1 activity into a prediction of the frame `horizon` steps ahead, at eye resolution.

    For every patch, one shared linear readout maps [codes now, codes one step ago, raw patch] of
    each eye (plus a bias) to the future patch's luminance. It starts as the average of the eyes'
    raw patches (the copy baseline) and learns online with the delta rule (normalised LMS): when the
    future frame arrives, weights move along prediction error x input - a local, error-driven update.
    `use` picks which eyes feed it, for the one-eye vs two-eye comparison.
    """

    def __init__(self, use=(0, 1)):
        self.use = use
        n_in = 3 * N_FEATURES * 2 + 1  # per eye: codes, previous codes, raw patch (64 each); + bias
        self.w = np.zeros((n_in, PATCH * PATCH), np.float32)
        for e in use:  # start as the average of the eyes in use
            raw = e * 3 * N_FEATURES + 2 * N_FEATURES
            self.w[raw:raw + PATCH * PATCH] = np.eye(PATCH * PATCH) / len(use)
        self.prev_codes = None
        self.model_sq = self.copy_sq = None  # per-patch running squared errors, for precision weighting

    def inputs(self, coders):
        """Per-patch input rows, or None until both eyes have codes."""
        codes = [c.codes for c in coders]
        if any(c is None for c in codes):
            return None
        same = self.prev_codes is not None and self.prev_codes[0].shape == codes[0].shape  # eye_width can change live
        prev = self.prev_codes if same else codes
        self.prev_codes = codes
        parts = []
        for e in range(2):
            if e in self.use:
                parts += [codes[e], prev[e], coders[e].raw]
            else:
                parts += [np.zeros_like(codes[e])] * 2 + [np.zeros_like(coders[e].raw)]
        return np.hstack(parts + [np.ones((len(codes[0]), 1), np.float32)])

    def predict(self, x):
        return np.clip(x @ self.w, 0, 1)

    def learn(self, x, target, rate):
        """Delta rule toward the target patches that actually arrived for input x."""
        error = target - x @ self.w
        self.w += rate * x.T @ error / (len(x) * float((x ** 2).sum(axis=1).mean()) + 1e-6)

    def track_precision(self, guess, copy_avg, target, decay=0.9):
        """Running squared error, per patch, of the model and of copying the eyes' average."""
        model_sq, copy_sq = ((guess - target) ** 2).mean(axis=1), ((copy_avg - target) ** 2).mean(axis=1)
        if self.model_sq is None or self.model_sq.shape != model_sq.shape:
            self.model_sq, self.copy_sq = model_sq, copy_sq
        self.model_sq = decay * self.model_sq + (1 - decay) * model_sq
        self.copy_sq = decay * self.copy_sq + (1 - decay) * copy_sq

    def blend(self, guess, copy_avg):
        """Precision weighting: per patch, trust the model in proportion to its recent precision
        (1 / squared error) relative to the eyes' average. Falls back to the average when untracked."""
        if self.model_sq is None or self.model_sq.shape[0] != len(guess):
            return copy_avg
        trust = self.copy_sq / (self.model_sq + self.copy_sq + 1e-9)
        return trust[:, None] * guess + (1 - trust[:, None]) * copy_avg


class EyeFlow:
    """Motion compensation on the eyes' average image (a "stabilized map"): estimate optical flow
    between consecutive eye averages and push the latest one forward along it by `reach` steps.
    Works at eye resolution, on what the eyes actually see (noisy, delayed)."""

    def __init__(self):
        self.prev = self.smooth = None

    def predict(self, avg, reach):
        """avg: eye-average grayscale (0-1, eye resolution). Returns the extrapolated image."""
        flow = self.flow(avg)
        return avg if flow is None else self.warp(avg, flow, reach)

    def flow(self, avg):
        """Per-pixel motion (eye pixels per step) since the previous call, or None on the first."""
        big = cv2.resize(cv2.GaussianBlur(avg, (0, 0), 1.0), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        big = (np.clip(big, 0, 1) * 255).astype(np.uint8)
        if self.prev is None or self.prev.shape != big.shape:
            self.prev = big
            return None
        flow = cv2.calcOpticalFlowFarneback(self.prev, big, None, 0.5, 2, 9, 3, 5, 1.1, 0)
        self.prev = big
        return cv2.resize(flow, (avg.shape[1], avg.shape[0])) / 2

    def sources(self, avg, reach, fractions=(0.25, 0.5, 1.0), smoothing=0.5, blur=6.0):
        """Candidates for long horizons (E17; defaults tuned at horizon 15 in E18): the average pushed
        by several fractions of `reach` along a running average of the flow (precision weighting then
        picks the damping per patch), plus the average blurred, which is the safer guess where the
        future is uncertain."""
        flow = self.flow(avg)
        if flow is None:
            flow = np.zeros(avg.shape + (2,), np.float32)
        if self.smooth is None or self.smooth.shape != flow.shape:
            self.smooth = flow
        self.smooth = smoothing * self.smooth + (1 - smoothing) * flow
        out = [self.warp(avg, self.smooth, f * reach) for f in fractions]
        return out + [cv2.GaussianBlur(avg, (0, 0), blur)] if blur > 0 else out

    @staticmethod
    def warp(avg, flow, reach):
        gy, gx = np.mgrid[0:avg.shape[0], 0:avg.shape[1]].astype(np.float32)
        return cv2.remap(avg, gx - reach * flow[..., 0], gy - reach * flow[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)


class PrecisionMixer:
    """Precision weighting over any number of predictions: per patch, each source's weight is
    proportional to its precision (1 / running squared error)."""

    def __init__(self, decay=0.9, sharpness=1.0):
        self.decay, self.sq, self.sharpness = decay, None, sharpness  # weights ~ precision ** sharpness

    def track(self, guesses, target):
        sq = np.stack([((g - target) ** 2).mean(axis=1) for g in guesses])
        if self.sq is None or self.sq.shape != sq.shape:
            self.sq = sq
        self.sq = self.decay * self.sq + (1 - self.decay) * sq

    def blend(self, guesses):
        if self.sq is None or self.sq.shape != (len(guesses), len(guesses[0])):
            return guesses[-1]  # untracked: fall back to the last source (by convention the eyes' average)
        precision = (self.sq.min(axis=0) / (self.sq + 1e-12)) ** self.sharpness  # scaled to <= 1, so powers can't overflow
        weights = precision / precision.sum(axis=0)
        return sum(w[:, None] * g for w, g in zip(weights, guesses))


class FlowPredictor:
    """Predicts `horizon` steps ahead by assuming every pixel keeps its (damped) optical-flow velocity."""

    def __init__(self):
        self.prev_gray, self.flow = None, None
        self.grid_y, self.grid_x = np.mgrid[0:H, 0:W].astype(np.float32)

    def predict(self, frame, p):
        small = cv2.resize(frame, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, p["flow_window"], 3, 5, 1.2, 0)
        self.prev_gray = gray
        s = p["flow_smoothing"]
        self.flow = flow if self.flow is None else s * self.flow + (1 - s) * flow
        flow = cv2.resize(self.flow, (W, H)) / FLOW_SCALE
        # the pixel at x in `horizon` steps is the one now at x - horizon * velocity
        reach = p["horizon"] * p["damping"]
        map_x = self.grid_x - reach * flow[..., 0]
        map_y = self.grid_y - reach * flow[..., 1]
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def compose(eyes, coders, bottom, ghost_err, p):
    """Eyes side by side on top (red left, blue right), then each eye's learned features, then the
    bottom panel: `bottom` (RGB, W x H) tinted purple with `ghost_err` (W x H, or None) drawn as
    ghosts - or, in the error view, `ghost_err` as a heatmap."""
    dithered = p["dither"]
    # tint at the eye's own resolution, then enlarge with hard edges so each retinal pixel stays visible;
    # never dithered, so each pixel shows the full 0-255 value the model gets
    rows = [np.hstack([cv2.resize(tint(eye, palette, False), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
                       for eye, palette in zip(eyes, (RED, BLUE))])]
    if p["features"]:
        row = np.zeros((FEATURES_PX, W, 3), np.uint8)
        for i, (coder, palette) in enumerate(zip(coders, (RED, BLUE))):
            x0 = i * W // 2 + (W // 2 - FEATURES_PX) // 2
            row[:, x0:x0 + FEATURES_PX] = coder.image(palette)
        rows.append(row)

    if p["view"] == "error":
        heat = np.clip(ghost_err * p["ghost_gain"] * 255, 0, 255).astype(np.uint8)
        return np.vstack(rows + [cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)])
    bottom = tint(bottom, PURPLE, dithered)
    if ghost_err is not None:
        ghost = np.clip((ghost_err - p["ghost_floor"]) * p["ghost_gain"], 0, 1)
        if dithered:
            bottom[ghost * 0.8 > THRESHOLD] = GHOST  # ordered-dithered ghost: dissolves at low error
        else:
            alpha = ghost[..., None] * 0.7
            bottom = (bottom * (1 - alpha) + GHOST * alpha).astype(np.uint8)
    return np.vstack(rows + [bottom])


def enlarge(gray):
    """A model-resolution grayscale image (0-1) as a W x H display image, smoothly upscaled.
    Rows the model doesn't cover (eye height not a multiple of the patch size) stay dark."""
    out = np.zeros((H, W), np.float32)
    h = round(gray.shape[0] * W / gray.shape[1])
    out[:h] = cv2.resize(gray, (W, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(out, 0, 1)


class StatsWindow:
    """Accumulates prediction scores for one second, then appends a summary line to stats.jsonl."""

    def __init__(self):
        self.start, self.steps, self.scored = time.monotonic(), 0, 0
        self.sq_pred = self.sq_copy = 0.0
        self.grid = np.zeros((3, 3), np.float64)
        self.model = {mode: [0, 0.0, 0.0] for mode in EYE_MODES}  # mode -> [scored, sq model, sq copy-avg]

    def add(self, sq_pred, sq_copy, err):
        self.scored += 1
        self.sq_pred += sq_pred
        self.sq_copy += sq_copy
        self.grid += cv2.resize(err, (3, 3), interpolation=cv2.INTER_AREA)

    def add_model(self, mode, sq_model, sq_copy_avg):
        m = self.model[mode]
        m[0] += 1
        m[1] += sq_model
        m[2] += sq_copy_avg

    def flush(self, params, extra):
        elapsed = time.monotonic() - self.start
        fps = self.steps / elapsed
        n = max(self.scored, 1)
        row = {
            "time": round(time.time(), 1),
            "fps": round(fps, 1),
            "ahead_ms": round(params["horizon"] * 1000 / max(fps, 1)),
            "scored": self.scored,
            "err": round(self.sq_pred / n * 1e3, 3),  # prediction MSE x1000
            "copy_err": round(self.sq_copy / n * 1e3, 3),  # copy-last-frame MSE x1000 (also a motion measure)
            "err_grid": np.round(self.grid / n * 100, 2).tolist(),  # mean abs error x100 per region, mirror view
            # displayed fused model vs copying the average of both eyes, at eye resolution (MSE x1000)
            "model_scored": self.model[params["fusion_eyes"]][0],
            "model_err": round(self.model[params["fusion_eyes"]][1] / max(self.model[params["fusion_eyes"]][0], 1) * 1e3, 3),
            "model_copy_err": round(self.model[params["fusion_eyes"]][2] / max(self.model[params["fusion_eyes"]][0], 1) * 1e3, 3),
            # every eye mode's model error, for the one-eye vs two-eye comparison
            "model_err_by_eyes": {mode: round(m[1] / max(m[0], 1) * 1e3, 3) for mode, m in self.model.items()},
            **extra,
            "params": params,
        }
        with open(STATS, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row


def layout(img_w, img_h):
    """Largest image (in cells) with this aspect that fits the pane, leaving two rows for status and caption."""
    rows, cols, xpix, ypix = struct.unpack("HHHH", fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8))
    cell_w, cell_h = xpix / cols, ypix / rows
    img_cols = cols
    img_rows = int(cols * cell_w * img_h / img_w / cell_h)
    if img_rows > rows - 2:
        img_rows = rows - 2
        img_cols = int(img_rows * cell_h * img_w / img_h / cell_w)
    return cols, img_cols, img_rows


def show(rgb, img_cols, img_rows):
    """Send a frame via Kitty graphics shared memory (raw RGB, no compression), replacing image 1."""
    data = rgb.tobytes()
    shm = shared_memory.SharedMemory(create=True, size=len(data), track=False)
    shm.buf[:len(data)] = data
    name = "/" + shm.name.lstrip("/")
    shm.close()  # the terminal unlinks it after reading
    h, w, _ = rgb.shape
    keys = f"a=T,i=1,p=1,f=24,s={w},v={h},c={img_cols},r={img_rows},C=1,q=2,t=s,S={len(data)}"
    write(b"\x1b[H\x1b_G" + keys.encode() + b";" + base64.standard_b64encode(name.encode()) + b"\x1b\\")


def main():
    global fd
    parser = argparse.ArgumentParser(description="Predictive webcam mirror for Ghostty.")
    parser.add_argument("--replay", type=Path, help="play this recorded clip in a loop instead of the webcam")
    args = parser.parse_args()

    fd = os.open("/dev/tty", os.O_RDWR)
    source = Replay(args.replay) if args.replay else Camera()
    source_name = f"replay:{args.replay.name}" if args.replay else "camera"
    eyes, predictor = Eyes(), FlowPredictor()
    coders, fusions = load_learned()
    eye_flows = {mode: EyeFlow() for mode in EYE_MODES}  # motion compensation per eye mode ("stabilized map")
    mixers = {mode: PrecisionMixer() for mode in EYE_MODES}
    params, warning = load_control()
    params = params or sanitize({})
    save_control(params)
    control_mtime = CONTROL.stat().st_mtime
    STATS.write_text("")

    pending = {}  # step -> (flow prediction made for that step, frame at prediction time)
    fusion_pending = {}  # step -> (copy-average of both eyes, {eye mode: (inputs, [model, motion-compensated, own copy], shown)})
    err = np.zeros((H, W), np.float32)  # flow baseline's error, full resolution
    model_err = np.zeros((H, W), np.float32)  # fused model's error, enlarged from eye resolution
    model_gray = None  # fused model's latest prediction, enlarged
    window = StatsWindow()
    recorder = None
    last = None
    step, seq = 0, 0
    current_layout = None

    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    write(b"\x1b[?25l\x1b[2J")
    try:
        while True:
            if select.select([fd], [], [], 0)[0]:
                changed = dict(params)
                for key in os.read(fd, 64).decode(errors="ignore"):
                    if key == "q":
                        return
                    elif key == "d":
                        changed["dither"] = not changed["dither"]
                    elif key == "f":
                        changed["features"] = not changed["features"]
                    elif key == "e":
                        modes = list(EYE_MODES)
                        changed["fusion_eyes"] = modes[(modes.index(changed["fusion_eyes"]) + 1) % len(modes)]
                    elif key == "v":
                        changed["view"] = VIEWS[(VIEWS.index(changed["view"]) + 1) % len(VIEWS)]
                    elif key in "+=":
                        changed["horizon"] += 1
                    elif key == "-":
                        changed["horizon"] -= 1
                    elif key == "r":
                        if recorder:
                            recorder.release()
                            recorder = None
                        else:
                            RECORDINGS.mkdir(exist_ok=True)
                            path = RECORDINGS / time.strftime("clip_%Y%m%d_%H%M%S.mp4")
                            recorder = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
                if changed != params:
                    save_control(sanitize(changed))  # picked up by the reload below

            mtime = CONTROL.stat().st_mtime if CONTROL.exists() else control_mtime
            if mtime != control_mtime:
                loaded, warning = load_control()
                if loaded:
                    params = loaded
                    save_control(params)  # write back the clamped values so mantis.py sees what applied
                control_mtime = CONTROL.stat().st_mtime

            seq, frame = source.next(seq)
            if frame is None:
                continue
            step += 1
            window.steps += 1
            if recorder:
                recorder.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # score the prediction made `horizon` steps ago against what the camera sees now
            if step in pending:
                predicted, then = pending.pop(step)
                diff = np.abs(predicted.astype(np.int16) - frame).mean(axis=2) / 255
                err = cv2.GaussianBlur(diff.astype(np.float32), (9, 9), 0)
                copy_diff = np.abs(then.astype(np.int16) - frame).mean(axis=2) / 255
                window.add(float((diff ** 2).mean()), float((copy_diff ** 2).mean()), err)
            pending = {s: p for s, p in pending.items() if s > step}

            views = eyes.see(frame, params)
            for coder, view in zip(coders, views):
                coder.step(view, params)

            # fused two-eye model: score and learn from the prediction made `horizon` steps ago,
            # against the true frame (no delay, no noise) at eye resolution
            eye_h, eye_w = views[0].shape[:2]
            truth_img = cv2.resize(frame, (eye_w, eye_h), interpolation=cv2.INTER_AREA).astype(np.float32) @ LUMA / 255
            truth = patchify(truth_img)
            # all three eye modes (both / left / right) predict and learn every step; one is displayed
            if step in fusion_pending:
                copy_avg, by_mode = fusion_pending.pop(step)
                if copy_avg.shape == truth.shape:  # skip if eye_width changed in between
                    copy_sq = float(((copy_avg - truth) ** 2).mean())
                    for mode, (x, sources, shown) in by_mode.items():
                        # the mirror shows (and is scored on) the precision-weighted blend, see E15/E16
                        window.add_model(mode, float(((shown - truth) ** 2).mean()), copy_sq)
                        if mode == params["fusion_eyes"]:
                            # blurred so stillness stays under ghost_floor (99th pct 0.024 when still, see E14)
                            model_err = cv2.GaussianBlur(enlarge(unpatchify(np.abs(shown - truth), truth_img.shape)), (0, 0), 6)
                        mixers[mode].track(sources, truth)
                        fusions[mode].learn(x, truth, params["fusion_rate"])
            fusion_pending = {s: v for s, v in fusion_pending.items() if s > step}
            raw = [c.raw for c in coders]
            by_mode = {}
            for mode, fusion in fusions.items():
                x = fusion.inputs(coders)
                if x is not None:
                    own_copy = sum(raw[e] for e in fusion.use) / len(fusion.use)  # the eyes this model sees
                    reach = params["horizon"] + params["eyeflow_lead"]
                    stabilized = eye_flows[mode].sources(unpatchify(own_copy, truth_img.shape), reach,
                                                         smoothing=params["eyeflow_smoothing"])
                    # learned model, motion-compensated at 3 reaches, blurred average, plain average (E17)
                    sources = [fusion.predict(x)] + [patchify(s) for s in stabilized] + [own_copy]
                    by_mode[mode] = (x, sources, mixers[mode].blend(sources))
            if by_mode:
                fusion_pending[step + params["horizon"]] = ((raw[0] + raw[1]) / 2, by_mode)
                model_gray = enlarge(unpatchify(by_mode[params["fusion_eyes"]][2], truth_img.shape))

            prediction = predictor.predict(frame, params)
            pending[step + params["horizon"]] = (prediction, frame)

            if time.monotonic() - window.start >= 1.0:
                last = window.flush(params, {
                    "source": source_name, "eye_delays": list(eyes.delays),
                    "v1_err": [round(c.rel_err, 3) for c in coders],  # residual / input energy per eye
                    "v1_active": [round(c.active, 3) for c in coders],  # fraction of features active
                })
                window = StatsWindow()
                if step % 900 < 30:  # roughly every 30 s
                    save_learned(coders, fusions)

            view = params["view"]
            if view == "prediction" and model_gray is not None:
                bottom, ghost_err = np.repeat((model_gray * 255).astype(np.uint8)[..., None], 3, axis=2), model_err
            elif view == "flow":
                bottom, ghost_err = prediction, err
            elif view == "error":
                bottom, ghost_err = None, model_err
            else:
                bottom, ghost_err = frame, None
            image = compose(views, coders, bottom, ghost_err, params)
            if layout(image.shape[1], image.shape[0]) != current_layout:
                current_layout = layout(image.shape[1], image.shape[0])
                write(b"\x1b_Ga=d,d=A,q=2\x1b\\\x1b[2J")
            cols, img_cols, img_rows = current_layout
            show(image, img_cols, img_rows)

            if last:
                flow_skill = 1 - last["err"] / last["copy_err"] if last["copy_err"] > 0 else 0.0
                model_skill = 1 - last["model_err"] / last["model_copy_err"] if last["model_copy_err"] > 0 else 0.0
                hud = (f"{'REC ' if recorder else ''}{params['view']} ({params['fusion_eyes']} eyes)"
                       f"{' dither' if params['dither'] else ''} | "
                       f"ahead {params['horizon']} (~{last['ahead_ms']}ms) | eyes L{eyes.delays[0]} R{eyes.delays[1]} | "
                       f"V1 err {last['v1_err'][0]:.2f}/{last['v1_err'][1]:.2f} | "
                       f"skill {model_skill:+.2f} vs eyes (flow {flow_skill:+.2f}) | {last['fps']:.0f}fps | d f e v +/- r q")
                write(f"\x1b[{img_rows + 1};1H\x1b[2K{hud[:cols]}".encode())
            caption = warning or params["caption"]
            write(f"\x1b[{img_rows + 2};1H\x1b[2K{caption[:cols]}".encode())
    finally:
        save_learned(coders, fusions)
        if recorder:
            recorder.release()
        write(b"\x1b_Ga=d,d=A,q=2\x1b\\\x1b[2J\x1b[H\x1b[?25h")
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        source.cap.release()


if __name__ == "__main__":
    main()
