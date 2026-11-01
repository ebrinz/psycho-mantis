"""Hybrid (E28): the grid lineage NCA's 1 s motion field, pulled toward its average within each
superpixel by alpha (0 = grid unchanged, 1 = every region moves rigidly with the grid's mean motion).
Same held-out frames as E26/E27 (every 20th)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, torch
import nca, graph_nca

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
frames, labels = np.load(nca.FRAMES), np.load(graph_nca.LABELS, mmap_mode="r")
_, test_idx = nca.split(len(frames))
idx = test_idx[::20]
model = nca.load("lineage")
k1 = nca.lineage_step(nca.HORIZON)
rows = []
with torch.no_grad():
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]
        x, y = nca.batch(frames, b)
        flow = model(x)[:, k1]  # (B, 2, H, W)
        lab = torch.from_numpy(labels[b].astype(np.int64)).to(nca.DEVICE)  # (B, H, W)
        n = int(lab.max()) + 1
        flat = (lab + n * torch.arange(len(b), device=nca.DEVICE)[:, None, None]).reshape(-1)  # global region id
        count = torch.zeros(len(b) * n, device=nca.DEVICE).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
        region = torch.stack([torch.zeros(len(b) * n, device=nca.DEVICE).index_add_(0, flat, flow[:, c].reshape(-1))
                              / count.clamp(min=1) for c in range(2)], 1)[flat].reshape(len(b), *lab.shape[1:], 2).permute(0, 3, 1, 2)
        row = [((x[:, :3] - y) ** 2).mean((1, 2, 3))]
        for a in ALPHAS:
            row.append(((nca.warp(x[:, :3], (1 - a) * flow + a * region) - y) ** 2).mean((1, 2, 3)))
        rows += torch.stack(row, 1).tolist()
r = np.array(rows)
bins = np.digitize(r[:, 0], np.percentile(r[:, 0], [20, 40, 60, 80]))
print(f"hybrid: lineage NCA motion pulled toward its superpixel average, {len(idx)} held-out frames, 1 s ahead")
for j, a in enumerate(ALPHAS):
    sk = lambda m=slice(None): 1 - r[m, j + 1].sum() / r[m, 0].sum()
    print(f"  alpha {a:4.2f}: skill {sk():+.3f}   by motion quintile " + " ".join(f"{sk(bins == q):+.3f}" for q in range(5)))
