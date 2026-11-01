# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy"]
# ///
"""Benchmark Kitty graphics throughput in the current terminal, with and without dithering.

Uses synthetic camera-like frames (no webcam needed). For each config it sends frames,
syncs with the terminal after every frame (so we measure frames actually processed, not
just buffered), and records fps. Results go to kitty_bench_results.json next to this file.

Configs: resolution x style (clean / dither / chunky) x zlib on/off x transmission
(d = direct base64 over the tty, t = temp file, s = shared memory).
Run it inside Ghostty. Ctrl-C to abort.
"""
import base64
import fcntl
import json
import os
import select
import struct
import sys
import tempfile
import termios
import time
import tty
import zlib
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "kitty_bench_results.json"
RESOLUTIONS = [(160, 120), (320, 240), (640, 480), (1280, 960)]
STYLES = ["clean", "dither", "chunky"]  # chunky = dither at 1/4 res, upscaled nearest-neighbour by us
MAX_FRAMES, MAX_SECONDS = 40, 2.0

# Game Boy-ish 4-tone palette
PALETTE = np.array([[15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15]], np.uint8)

fd = os.open("/dev/tty", os.O_RDWR)


def write(data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def read_until(predicate, timeout=5.0):
    buf, deadline = b"", time.monotonic() + timeout
    while not predicate(buf):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            raise TimeoutError(f"terminal did not respond (got {buf!r})")
        buf += os.read(fd, 4096)
    return buf


def sync():
    """Primary device attributes query: the reply arrives only after everything before it is processed."""
    write(b"\x1b[c")
    return read_until(lambda b: b"\x1b[?" in b and b.rstrip().endswith(b"c"))


def bayer(n):
    m = np.array([[0, 2], [3, 1]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size


BAYER8 = bayer(8)


def dither(rgb):
    h, w, _ = rgb.shape
    lum = rgb @ np.array([0.299, 0.587, 0.114]) / 255
    threshold = np.tile(BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w]
    idx = np.clip((lum * (len(PALETTE) - 1) + threshold).astype(int), 0, len(PALETTE) - 1)
    return PALETTE[idx]


def synthetic_frames(w, h, n=16):
    """Smooth moving colour field + a bright blob + sensor noise, roughly camera-like."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:h, 0:w] / max(w, h)
    frames = []
    for i in range(n):
        p = i / n * 2 * np.pi
        base = 0.5 + 0.25 * np.sin(x * 9 + p)[..., None] * np.cos(y * 7 - p)[..., None] * np.array([1, 0.6, 0.3])
        bx, by = 0.5 + 0.25 * np.cos(p), 0.4 + 0.1 * np.sin(p)
        base += 0.45 * np.exp(-((x - bx) ** 2 + (y - by) ** 2) / 0.01)[..., None]
        img = base * 255 + rng.normal(0, 6, (h, w, 3))
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


def style_frame(rgb, style):
    if style == "clean":
        return rgb
    if style == "dither":
        return dither(rgb)
    small = dither(rgb[::4, ::4])
    return np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)


def graphics_cmd(keys, payload=b""):
    return b"\x1b_G" + keys.encode() + b";" + payload + b"\x1b\\"


def transmit(data, keys, mode):
    """Send pixel data with the given control keys using transmission mode d/t/s."""
    if mode == "d":
        b64 = base64.standard_b64encode(data)
        chunks = [b64[i:i + 4096] for i in range(0, len(b64), 4096)] or [b""]
        out = [graphics_cmd(f"{keys},m={int(len(chunks) > 1)}", chunks[0])]
        out += [graphics_cmd(f"m={int(i < len(chunks) - 1)}", c) for i, c in enumerate(chunks[1:], 1)]
        write(b"".join(out))
    elif mode == "t":
        f = tempfile.NamedTemporaryFile(prefix="tty-graphics-protocol-", delete=False)
        f.write(data)
        f.close()
        write(graphics_cmd(f"{keys},t=t", base64.standard_b64encode(f.name.encode())))
    else:
        shm = shared_memory.SharedMemory(create=True, size=len(data), track=False)
        shm.buf[:len(data)] = data
        name = "/" + shm.name.lstrip("/")
        shm.close()
        write(graphics_cmd(f"{keys},t=s,S={len(data)}", base64.standard_b64encode(name.encode())))


def supported(mode):
    """Ask the terminal whether it accepts a tiny image via this transmission mode."""
    transmit(b"\x00\x00\x00", "a=q,i=4242,s=1,v=1,f=24", mode)
    reply = sync()
    return b"_Gi=4242;OK" in reply


def layout():
    rows, cols, xpix, ypix = struct.unpack("HHHH", fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8))
    cell_w, cell_h = xpix / cols, ypix / rows
    avail_rows = rows - 3
    img_rows = min(avail_rows, int(cols * cell_w * 3 / 4 / cell_h))
    img_cols = int(img_rows * cell_h * 4 / 3 / cell_w)
    return rows, img_cols, img_rows


def status(rows, text):
    write(f"\x1b[{rows - 1};1H\x1b[2K{text}".encode())


def run_config(frames, style, compress, mode, rows, img_cols, img_rows):
    prep_t = term_t = 0.0
    n, start = 0, time.monotonic()
    while n < MAX_FRAMES and time.monotonic() - start < MAX_SECONDS:
        t0 = time.monotonic()
        img = style_frame(frames[n % len(frames)], style)
        h, w, _ = img.shape
        data = img.tobytes()
        keys = f"a=T,i=1,p=1,f=24,s={w},v={h},c={img_cols},r={img_rows},C=1,q=2"
        if compress:
            data = zlib.compress(data, 1)
            keys += ",o=z"
        t1 = time.monotonic()
        write(b"\x1b[H")
        transmit(data, keys, mode)
        sync()
        t2 = time.monotonic()
        prep_t += t1 - t0
        term_t += t2 - t1
        n += 1
    elapsed = time.monotonic() - start
    return {"fps": n / elapsed, "prep_ms": 1000 * prep_t / n, "terminal_ms": 1000 * term_t / n,
            "bytes_per_frame": len(data)}


def main():
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    write(b"\x1b[?25l\x1b[2J")
    results = {"term": os.environ.get("TERM_PROGRAM"), "modes": {}, "runs": []}
    try:
        for mode in "dts":
            try:
                results["modes"][mode] = supported(mode)
            except TimeoutError:
                results["modes"][mode] = False
        modes = [m for m, ok in results["modes"].items() if ok]
        rows, img_cols, img_rows = layout()
        results["layout"] = {"img_cols": img_cols, "img_rows": img_rows}
        for w, h in RESOLUTIONS:
            frames = synthetic_frames(w, h)
            for style in STYLES:
                for compress in (False, True):
                    for mode in modes:
                        label = f"{w}x{h} {style:6} {'zlib' if compress else 'raw '} mode={mode}"
                        status(rows, f"running {label} ...")
                        r = run_config(frames, style, compress, mode, rows, img_cols, img_rows)
                        r.update(res=f"{w}x{h}", style=style, zlib=compress, mode=mode)
                        results["runs"].append(r)
                        status(rows, f"{label}: {r['fps']:.1f} fps")
    except KeyboardInterrupt:
        results["aborted"] = True
    finally:
        RESULTS.write_text(json.dumps(results, indent=2))
        write(graphics_cmd("a=d,d=A,q=2") + b"\x1b[2J\x1b[H\x1b[?25h")
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print(f"transmission modes supported: {results['modes']}")
    print(f"{'config':40} {'fps':>6} {'prep ms':>8} {'term ms':>8} {'KB/frame':>9}")
    for r in results["runs"]:
        label = f"{r['res']} {r['style']} {'zlib' if r['zlib'] else 'raw'} mode={r['mode']}"
        print(f"{label:40} {r['fps']:6.1f} {r['prep_ms']:8.1f} {r['terminal_ms']:8.1f} {r['bytes_per_frame'] / 1024:9.0f}")
    print(f"\nsaved to {RESULTS}")


if __name__ == "__main__":
    main()
