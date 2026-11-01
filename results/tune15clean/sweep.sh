#!/bin/sh
# usage: sweep.sh configs.txt  (lines "name|evaluate.py args"), 6 in parallel, horizon 15, clean eyes
cd /Users/crashy/Development/psycho-mantis
while IFS='|' read -r name rest; do
  echo "uv run evaluate.py recordings/clip_20261030_135948.mp4 --horizon 15 --clean $rest --out results/tune15clean/$name.json > results/tune15clean/$name.log 2>&1"
done < "$1" | xargs -P 6 -I{} sh -c '{}'
