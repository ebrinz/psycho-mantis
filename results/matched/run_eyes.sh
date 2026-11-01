#!/bin/bash
# E36: smoke-test the eye-system run, then launch the six full runs (3 horizons x 2 blurs) in parallel
cd "$(dirname "$0")/../.."
PY=(uv run --python 3.13 --with numpy --with opencv-python --with torch --with scikit-image python)
MATCHED_T0=200 OMP_NUM_THREADS=2 "${PY[@]}" results/matched_eval.py eye --horizon 16 --blur 6 --steps 420 2>&1 | grep -v "^Installed\|Download\|UserWarning\|warnings.warn" || exit 1
rm -f results/matched/eye_h16_b6.npz
for h in 4 16 30; do for b in 6 12; do
  OMP_NUM_THREADS=2 nohup "${PY[@]}" results/matched_eval.py eye --horizon $h --blur $b > results/matched/eye_h${h}_b${b}.log 2>&1 &
done; done
sleep 2; echo "launched: $(pgrep -f 'matched_eval.py eye' | wc -l | tr -d ' ') eye runs"
