# /// script
# requires-python = ">=3.10"
# dependencies = ["opencv-python", "numpy"]
# ///
"""Delayed webcam mirror: live feed next to a time-delayed copy of yourself.

Keys:
  [ / ]  decrease / increase delay by 50ms
  v      toggle view: side-by-side / delayed only
  r      start / stop recording the raw camera feed to data/
  q      quit
"""
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

MAX_DELAY_S = 3.0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam (check camera permission for your terminal).")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
buffer = deque(maxlen=int(MAX_DELAY_S * fps * 1.5))  # (timestamp, frame)

delay_s = 0.5
side_by_side = True
writer = None
ts_file = None


def delayed_frame(now):
    target = now - delay_s
    for t, f in reversed(buffer):
        if t <= target:
            return f
    return buffer[0][1]  # not enough history yet


def start_recording(shape):
    out_dir = Path("data") / time.strftime("session_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    h, w = shape[:2]
    vw = cv2.VideoWriter(str(out_dir / "camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    f = open(out_dir / "timestamps.csv", "w", newline="")
    csv.writer(f).writerow(["frame", "time_s", "delay_s"])
    print(f"Recording to {out_dir}")
    return vw, f


frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    now = time.monotonic()
    buffer.append((now, frame))

    if writer is not None:
        writer.write(frame)
        csv.writer(ts_file).writerow([frame_idx, f"{now:.4f}", delay_s])
        frame_idx += 1

    delayed = cv2.flip(delayed_frame(now), 1)
    view = np.hstack([cv2.flip(frame, 1), delayed]) if side_by_side else delayed.copy()

    label = f"delay {delay_s * 1000:.0f}ms" + ("  REC" if writer is not None else "")
    cv2.putText(view, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("delayed mirror", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("["):
        delay_s = max(0.0, delay_s - 0.05)
    elif key == ord("]"):
        delay_s = min(MAX_DELAY_S, delay_s + 0.05)
    elif key == ord("v"):
        side_by_side = not side_by_side
    elif key == ord("r"):
        if writer is None:
            writer, ts_file = start_recording(frame.shape)
            frame_idx = 0
        else:
            writer.release()
            ts_file.close()
            writer = None
            print("Recording stopped")

if writer is not None:
    writer.release()
    ts_file.close()
cap.release()
cv2.destroyAllWindows()
