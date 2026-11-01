# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "opencv-python", "torch", "scikit-image"]
# ///
"""Graph neural cellular automaton (Grattarola et al. 2021, "Learning Graph Cellular Automata") on
superpixels, predicting the camera frame 1 s ahead (E26).

Each frame (160x90) is split into ~240 SLIC superpixels: regions of similar colour that follow
edges. Each region is a cell; regions that touch are neighbours. A cell sees its mean colour now,
100 ms and 200 ms ago (as the grid NCA, nca.PAST), its position and size, plus hidden channels.
The rule is the grid NCA's, with graph versions of its filters: own state, mean difference to the
neighbours, and that difference weighted by the direction to each neighbour (x and y, standing in
for the Sobel filters). After `steps` steps, two hidden channels are the region's motion; every
pixel moves with its region, so regions shift as rigid pieces and cannot stretch. With --smooth S
the per-pixel motion is then blurred (Gaussian, S cells), so neighbouring regions blend at their
borders instead of tearing (E27); S is stored in the checkpoint.

  uv run graph_nca.py prep                  superpixels for every frame -> data/slic_160x90.npy
  uv run graph_nca.py train [--smooth S] [--iters N]   train on the first 80% -> state/nca_graph[_sS].pt
  uv run graph_nca.py eval [--smooth S]                held-out skill at 1 s, next to the grid warp NCA (E21)
"""
import argparse
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import slic
from torch import nn

import nca

LABELS = nca.ROOT / "data" / "slic_160x90.npy"
def checkpoint(smooth):
    return nca.ROOT / "state" / (f"nca_graph_s{smooth:g}.pt" if smooth else "nca_graph.pt")
N_SEGMENTS = 300  # SLIC target; ~240 regions of ~60 px at 160x90
INPUTS = 3 * len(nca.PAST) + 3  # mean colour at each PAST frame, x, y, size


def superpixels(img):
    return slic(img, n_segments=N_SEGMENTS, compactness=10, start_label=0).astype(np.uint16)


def prep():
    frames = np.load(nca.FRAMES, mmap_mode="r")
    with Pool(8) as pool:
        labels = pool.map(superpixels, (np.ascontiguousarray(f) for f in frames), chunksize=64)
    np.save(LABELS, np.stack(labels))
    print(f"{len(labels)} frames -> {LABELS}")


def graph(labels, past):
    """labels: (H, W) region ids; past: frames at nca.PAST offsets, each (H, W, 3) uint8.
    Returns node features (n, INPUTS), node positions (n, 2) in cells, edges (2, E) both ways."""
    n = int(labels.max()) + 1
    flat = labels.ravel()
    size = np.bincount(flat, minlength=n).astype(np.float32)
    colour = [np.stack([np.bincount(flat, f[..., c].ravel(), n) for c in range(3)], 1) / size[:, None] / 255 for f in past]
    gy, gx = np.mgrid[0:labels.shape[0], 0:labels.shape[1]]
    pos = np.stack([np.bincount(flat, gx.ravel(), n), np.bincount(flat, gy.ravel(), n)], 1) / size[:, None]
    feats = np.concatenate(colour + [pos / [nca.W, nca.H], (size / size.mean())[:, None]], 1)
    pairs = np.concatenate([np.stack([labels[:, :-1].ravel(), labels[:, 1:].ravel()]),
                            np.stack([labels[:-1].ravel(), labels[1:].ravel()])], 1)
    pairs = np.unique(pairs[:, pairs[0] != pairs[1]], axis=1)
    edges = np.unique(np.concatenate([pairs, pairs[::-1]], 1), axis=1)
    return feats.astype(np.float32), pos.astype(np.float32), edges


def batch_graphs(frames, labels, idx):
    """Graphs for frame indices idx, joined into one disconnected graph. Returns tensors on DEVICE:
    feats, pos, edges, and per-frame pixel -> global node index maps (B, H, W)."""
    feats, pos, edges, maps, offset = [], [], [], [], 0
    for t in idx:
        lab = labels[t].astype(np.int64)
        f, p, e = graph(lab, [frames[t - k] for k in nca.PAST])
        feats.append(f), pos.append(p), edges.append(e + offset), maps.append(lab + offset)
        offset += len(f)
    to = lambda a: torch.from_numpy(a).to(nca.DEVICE)
    return to(np.concatenate(feats)), to(np.concatenate(pos)), to(np.concatenate(edges, 1)), to(np.stack(maps))


class GraphNCA(nn.Module):
    out = "graph"

    def __init__(self, hidden=16, width=96, steps=16, smooth=0.0):
        super().__init__()
        self.hidden, self.steps = hidden, steps
        self.register_buffer("smooth", torch.tensor(float(smooth)))  # blur of the per-pixel motion, in cells
        channels = INPUTS + hidden
        self.rule = nn.Sequential(nn.Linear(4 * channels, width), nn.ReLU(), nn.Linear(width, hidden))
        nn.init.zeros_(self.rule[2].weight)  # start as "no motion", i.e. copy the current frame (as E21)
        nn.init.zeros_(self.rule[2].bias)

    def forward(self, feats, pos, edges):
        """Returns each node's motion (n, 2: dx, dy in cells)."""
        src, dst = edges
        d = pos[src] - pos[dst]
        unit = d / d.norm(dim=1, keepdim=True).clamp(min=1e-3)  # direction from each node to its neighbour
        deg = torch.zeros(len(feats), 1, device=feats.device).index_add_(0, dst, torch.ones(len(dst), 1, device=feats.device))
        state = torch.zeros(len(feats), self.hidden, device=feats.device)
        for _ in range(self.steps):
            h = torch.cat([feats, state], 1)
            diff = h[src] - h[dst]
            agg = lambda w: torch.zeros_like(h).index_add_(0, dst, diff * w) / deg
            state = state + self.rule(torch.cat([h, agg(1.0), agg(unit[:, :1]), agg(unit[:, 1:])], 1))
        return state[:, :2] * nca.FLOW_SCALE


def pixel_flow(node_flow, maps, smooth=0.0):
    """Every pixel takes its region's motion, then (smooth > 0) a Gaussian blur of `smooth` cells: (B, 2, H, W)."""
    flow = node_flow[maps].permute(0, 3, 1, 2)
    smooth = float(smooth)
    if smooth <= 0:
        return flow
    r = int(3 * smooth)
    k = torch.exp(-torch.arange(-r, r + 1, device=flow.device, dtype=flow.dtype) ** 2 / (2 * smooth ** 2))
    k = (k / k.sum()).repeat(2, 1)[:, None]  # (2, 1, 2r + 1), depthwise
    flow = F.conv2d(F.pad(flow, (r, r, 0, 0), mode="replicate"), k[..., None, :], groups=2)
    return F.conv2d(F.pad(flow, (0, 0, r, r), mode="replicate"), k[..., :, None], groups=2)


def train(iters, batch_size, lr, smooth):
    frames, labels = np.load(nca.FRAMES), np.load(LABELS, mmap_mode="r")
    train_idx, test_idx = nca.split(len(frames))
    model = GraphNCA(smooth=smooth).to(nca.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    rng = np.random.default_rng(0)
    probe = rng.choice(test_idx, 256, replace=False)
    start, run = time.monotonic(), []
    for it in range(1, iters + 1):
        idx = rng.choice(train_idx, batch_size)
        x, y = nca.batch(frames, idx)
        feats, pos, edges, maps = batch_graphs(frames, labels, idx)
        loss = F.l1_loss(nca.warp(x[:, :3], pixel_flow(model(feats, pos, edges), maps, model.smooth)), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if it % 100 == 0 or it == iters:
            print(f"iter {it:5}  train l1 x1000 {np.mean(run) * 1e3:.3f}  held-out skill {skill(model, frames, labels, probe):+.3f}  "
                  f"{time.monotonic() - start:.0f}s", flush=True)
            run = []
            torch.save(model.state_dict(), checkpoint(smooth))


@torch.no_grad()
def skill(model, frames, labels, idx, chunk=64):
    err = copy = 0.0
    for i in range(0, len(idx), chunk):
        b = idx[i:i + chunk]
        x, y = nca.batch(frames, b)
        feats, pos, edges, maps = batch_graphs(frames, labels, b)
        err += ((nca.warp(x[:, :3], pixel_flow(model(feats, pos, edges), maps, model.smooth)) - y) ** 2).sum().item()
        copy += ((x[:, :3] - y) ** 2).sum().item()
    return 1 - err / copy


def load(smooth=0.0):
    model = GraphNCA().to(nca.DEVICE)
    state = torch.load(checkpoint(smooth), map_location=nca.DEVICE, weights_only=True)
    state.setdefault("smooth", torch.tensor(0.0))  # checkpoints from before smoothing (E26)
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def evaluate(every, smooth):
    frames, labels = np.load(nca.FRAMES), np.load(LABELS, mmap_mode="r")
    _, test_idx = nca.split(len(frames))
    idx = test_idx[::every]
    model, grid = load(smooth), nca.load("warp")
    rows = []
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]
        x, y = nca.batch(frames, b)
        feats, pos, edges, maps = batch_graphs(frames, labels, b)
        pg = nca.warp(x[:, :3], pixel_flow(model(feats, pos, edges), maps, model.smooth))
        rows += zip(((x[:, :3] - y) ** 2).mean((1, 2, 3)).tolist(), ((pg - y) ** 2).mean((1, 2, 3)).tolist(),
                    ((nca.predict(grid, x) - y) ** 2).mean((1, 2, 3)).tolist())
    r = np.array(rows)
    sk = lambda col, m=slice(None): 1 - r[m, col].sum() / r[m, 0].sum()
    print(f"graph NCA on superpixels (motion blur {float(model.smooth):g} cells), held-out, {len(idx)} frames, 1 s ahead (160x90)")
    print(f"skill: graph {sk(1):+.3f}  vs grid warp NCA (E21) {sk(2):+.3f}  (same frames); "
          f"graph better than copying on {(r[:, 1] < r[:, 0]).mean():.1%} of frames")
    bins = np.digitize(r[:, 0], np.percentile(r[:, 0], [20, 40, 60, 80]))
    for q in range(5):
        m = bins == q
        print(f"  motion quintile {q + 1}: graph {sk(1, m):+.3f}  grid {sk(2, m):+.3f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    t = sub.add_parser("train")
    t.add_argument("--iters", type=int, default=1500)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--smooth", type=float, default=0.0, help="blur the per-pixel motion by this many cells")
    e = sub.add_parser("eval")
    e.add_argument("--every", type=int, default=20)
    e.add_argument("--smooth", type=float, default=0.0, help="which trained model (its --smooth)")
    args = parser.parse_args()
    if args.cmd == "prep":
        prep()
    elif args.cmd == "train":
        train(args.iters, args.batch, args.lr, args.smooth)
    else:
        evaluate(args.every, args.smooth)


if __name__ == "__main__":
    main()
