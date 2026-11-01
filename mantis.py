#!/usr/bin/env python3
"""Claude's side of the psychomanteum loop: perceive the mirror's prediction error, act on its parameters.

  python3 mantis.py observe [seconds]   summarise the mirror's stats over the next N seconds (default 5)
  python3 mantis.py set key=value ...   change mirror parameters (values parsed as JSON, else text)
  python3 mantis.py params              print the current parameters
"""
import json
import sys
import time
from pathlib import Path

STATE = Path(__file__).parent / "state"
CONTROL, STATS = STATE / "control.json", STATE / "stats.jsonl"
LOW_MOTION = 0.3  # copy_err below this: too little movement for skill to mean anything


def read_params():
    if not CONTROL.exists():
        sys.exit("No state/control.json - start the mirror first (uv run psychomanteum.py).")
    return json.loads(CONTROL.read_text())


def observe(seconds):
    start = STATS.stat().st_size if STATS.exists() else 0
    time.sleep(seconds)
    if not STATS.exists() or STATS.stat().st_size < start:  # mirror restarted and truncated the log
        start = 0
    with open(STATS) as f:
        f.seek(start)
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if r["scored"]]
    if not rows:
        sys.exit("No stats arrived - is the mirror running?")

    scored = sum(r["scored"] for r in rows)
    err = sum(r["err"] * r["scored"] for r in rows) / scored
    copy_err = sum(r["copy_err"] * r["scored"] for r in rows) / scored
    grid = [[sum(r["err_grid"][i][j] * r["scored"] for r in rows) / scored for j in range(3)] for i in range(3)]
    p = rows[-1]["params"]
    if any(r["params"] != p for r in rows):
        print("note: parameters changed during this window")

    print(f"observed {len(rows)}s, {scored} scored frames, {rows[-1]['fps']:.0f} fps, "
          f"predicting {p['horizon']} steps (~{rows[-1]['ahead_ms']} ms) ahead")
    print(f"  motion / copy-last err  {copy_err:7.3f}")
    print(f"  flow prediction err     {err:7.3f}")
    if copy_err < LOW_MOTION:
        print("  flow skill              n/a (too little motion to judge)")
    else:
        print(f"  flow skill              {1 - err / copy_err:+7.3f}  (full-res flow baseline vs copying the last frame)")
    model_rows = [r for r in rows if r.get("model_scored")]
    if model_rows:
        n = sum(r["model_scored"] for r in model_rows)
        model_err = sum(r["model_err"] * r["model_scored"] for r in model_rows) / n
        model_copy = sum(r["model_copy_err"] * r["model_scored"] for r in model_rows) / n
        print(f"  fused-model err         {model_err:7.3f}   vs copying the eyes' average {model_copy:7.3f}")
        print(f"  model skill             {1 - model_err / model_copy:+7.3f}  (> 0 means fusing beats just averaging the eyes)")
    by_eyes = [r["model_err_by_eyes"] for r in model_rows if "model_err_by_eyes" in r]
    if by_eyes:
        means = {k: sum(b[k] for b in by_eyes) / len(by_eyes) for k in by_eyes[0]}
        print("  one vs two eyes (err)   " + "  ".join(f"{k} {v:.3f}" for k, v in means.items()))
    print("  error by region (x100, as seen in the mirror):")
    for row in grid:
        print("    " + "  ".join(f"{v:5.2f}" for v in row))
    print("  params: " + " ".join(f"{k}={json.dumps(v)}" for k, v in p.items()))


def set_params(assignments):
    params = read_params()
    for a in assignments:
        key, _, raw = a.partition("=")
        if key not in params:
            sys.exit(f"Unknown parameter {key!r}. Known: {', '.join(params)}")
        try:
            params[key] = json.loads(raw)
        except json.JSONDecodeError:
            params[key] = raw
    tmp = CONTROL.with_suffix(".tmp")
    tmp.write_text(json.dumps(params, indent=2))
    tmp.replace(CONTROL)
    time.sleep(0.3)  # let the mirror reload and clamp
    print("now: " + " ".join(f"{k}={json.dumps(v)}" for k, v in read_params().items()))


if __name__ == "__main__":
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "observe":
        observe(float(args[0]) if args else 5.0)
    elif cmd == "set" and args:
        set_params(args)
    elif cmd == "params":
        print(json.dumps(read_params(), indent=2))
    else:
        sys.exit(__doc__)
