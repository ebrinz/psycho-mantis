"""How far does the NCA's 1 s motion field fall short of the actual motion? Held-out recording frames."""
import sys
sys.path.insert(0, "/Users/crashy/Development/psycho-mantis")
import cv2, numpy as np, torch
import nca

out = sys.argv[1] if len(sys.argv) > 1 else "lineage"
frames = np.load(nca.FRAMES, mmap_mode="r")
model = nca.load(out)
_, test_idx = nca.split(len(frames), 48)
idx = test_idx[::10]
gray = lambda f: cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY)
gains = (0.5, 1.0, 1.5, 2.0, 3.0)
num = den = 0.0
mag_true, mag_model = [], []
err = {g: 0.0 for g in gains}
copy = 0.0
with torch.no_grad():
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]
        x, y = nca.batch(frames, b)
        o = model(x)
        flow = (o[:, nca.lineage_step(nca.HORIZON)] if out == "lineage" else o)  # (B, 2, H, W)
        for g in gains:
            err[g] += ((nca.warp(x[:, :3], g * flow) - y) ** 2).sum().item()
        copy += ((x[:, :3] - y) ** 2).sum().item()
        fm = flow.permute(0, 2, 3, 1).cpu().numpy()
        for j, t in enumerate(b):
            # actual motion in the model's convention: future(p) = now(p + f(p))
            ft = cv2.calcOpticalFlowFarneback(gray(frames[t + nca.HORIZON]), gray(frames[t]), None, 0.5, 3, 9, 3, 5, 1.1, 0)
            moving = np.linalg.norm(ft, axis=-1) > 1.0  # pixels that actually moved > 1 cell
            num += (fm[j] * ft).sum()
            den += (fm[j] * fm[j]).sum()
            if moving.any():
                mag_true.append(np.linalg.norm(ft[moving], axis=-1).mean())
                mag_model.append(np.linalg.norm(fm[j][moving], axis=-1).mean())
print(f"{out} model, {len(idx)} held-out frames, 1 s ahead")
print(f"best single multiplier for the model's motion (least squares vs actual flow): {num / den:.2f}")
print(f"on pixels that actually moved > 1 cell: actual {np.mean(mag_true):.2f} cells, predicted {np.mean(mag_model):.2f} "
      f"-> predicts {np.mean(mag_model) / np.mean(mag_true):.0%} of the distance")
print("skill vs copying at display multiplier: " + "  ".join(f"x{g:g} {1 - err[g] / copy:+.3f}" for g in gains))
