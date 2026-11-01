# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "pyfiglet"]
# ///
"""Demoscene splash for a Ghostty pane: plasma, sine-wobbling title, scroller. q or Ctrl-C quits."""
import math
import os
import select
import shutil
import sys
import termios
import time
import tty

import numpy as np
import pyfiglet

TITLE = "PSYCHO MANTIS"
SCROLLER = ("   *** PSYCHOMANTEUM ***   a mirror that sees a moment ahead   ***   "
            "prediction error manifests as ghosts   ***   CIMC hackathon 2026   ***   "
            "greetz to the visual cortex   ***")
FPS = 30


def title_lines(cols, rows):
    """Biggest rendering of the title that fits the pane."""
    for font in ("ansi_shadow", "small"):
        lines = pyfiglet.figlet_format(TITLE, font=font, width=1000).rstrip("\n").split("\n")
        lines = [line.rstrip() for line in lines]
        if max(map(len, lines)) <= cols and len(lines) + 1 <= rows:
            return lines
    return [TITLE]


def plasma(cols, rows, t):
    """Two pixels per cell (top/bottom half-block) of dim ghost-green plasma, shape (rows*2, cols, 3)."""
    y, x = np.mgrid[0:rows * 2, 0:cols].astype(np.float32)
    x, y = x * 0.5, y  # cells are ~2x taller than wide
    cx, cy = cols * 0.25 + 10 * math.sin(t * 0.4), rows + rows * 0.6 * math.cos(t * 0.3)
    v = (np.sin(x * 0.12 + t) + np.sin(y * 0.15 + t * 0.7) + np.sin((x + y) * 0.08 + t * 0.5)
         + np.sin(np.hypot(x - cx, y - cy) * 0.18 - t * 1.3))
    phase = v * 0.25 + t * 0.05
    rgb = 0.5 + 0.5 * np.cos(2 * np.pi * (phase[..., None] + np.array([0.55, 0.33, 0.45])))
    return (rgb * np.array([40, 110, 100])).astype(np.uint8)


def frame(cols, rows, t, title):
    px = plasma(cols, rows, t)
    top, bottom = px[0::2].tolist(), px[1::2].tolist()
    chars = [["▀"] * cols for _ in range(rows)]
    fg = [row[:] for row in top]
    bg = bottom

    def put(r, c, ch, color):
        if 0 <= r < rows and 0 <= c < cols and ch != " ":
            chars[r][c], fg[r][c] = ch, color
            bg[r][c] = [(a + b) // 2 for a, b in zip(top[r][c], bottom[r][c])]

    # title: each column bobs on a sine wave, with a white glint sweeping across
    width, height = max(map(len, title)), len(title)
    x0, y0 = (cols - width) // 2, (rows - 1 - height) // 2
    amp = max(0, (rows - 1 - height) // 2)
    glint = (t * 25) % (width + 40) - 20
    for c in range(width):
        dy = round(amp * math.sin(c * 0.18 + t * 3))
        shine = max(0.0, 1 - abs(c - glint) / 6)
        base = (90 + 60 * math.sin(c * 0.1 + t), 255, 200 + 55 * math.sin(c * 0.07 - t))
        color = [int(b + (255 - b) * shine) for b in base]
        for r, line in enumerate(title):
            if c < len(line):
                put(y0 + r + dy, x0 + c, line[c], color)

    # scroller on the bottom row in blues; the text scrolls left while shimmer bands drift right, slower
    offset = int(t * 18)
    deep, icy = (40, 70, 160), (190, 225, 255)
    for c in range(cols):
        ch = SCROLLER[(c + offset) % len(SCROLLER)]
        shimmer = (0.5 + 0.5 * math.cos(2 * math.pi * (c - t * 6) / 28)) ** 3
        put(rows - 1, c, ch, [int(d + (i - d) * shimmer) for d, i in zip(deep, icy)])

    out = []
    for r in range(rows):
        out.append(f"\x1b[{r + 1};1H")
        for c in range(cols):
            (fr, fgr, fb), (br, bgr, bb) = fg[r][c], bg[r][c]
            out.append(f"\x1b[38;2;{fr};{fgr};{fb};48;2;{br};{bgr};{bb}m{chars[r][c]}")
    out.append("\x1b[0m")
    return "".join(out)


def main():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    sys.stdout.write("\x1b[?1049h\x1b[?25l")  # alternate screen, hide cursor
    start, size, title = time.monotonic(), None, None
    try:
        while not (select.select([fd], [], [], 0)[0] and "q" in os.read(fd, 64).decode(errors="ignore")):
            t0 = time.monotonic()
            if shutil.get_terminal_size() != size:
                size = shutil.get_terminal_size()
                title = title_lines(size.columns, size.lines)
            sys.stdout.write(frame(size.columns, size.lines, t0 - start, title))
            sys.stdout.flush()
            time.sleep(max(0.0, 1 / FPS - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
