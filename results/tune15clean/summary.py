import json, sys
from pathlib import Path
rows = []
for name in sys.argv[2:] or [l.split("|")[0] for l in open(sys.argv[1]) if l.strip()]:
    f = Path(f"/Users/crashy/Development/psycho-mantis/results/tune15clean/{name}.json")
    if not f.exists():
        print(f"{name:12} MISSING"); continue
    d = json.loads(f.read_text())
    a, b = d["skill_by_half"]["mixN"]
    rows.append((a, name, b, d["skill"]["mixN"], d["by_motion"][0]["mixN"]))
print(f"{'config':12} {'first half':>10} {'second':>8} {'full':>7} {'q1 still':>9}")
for a, name, b, full, q1 in sorted(rows, reverse=True):
    print(f"{name:12} {a:+10.4f} {b:+8.4f} {full:+7.4f} {q1:+9.3f}")
