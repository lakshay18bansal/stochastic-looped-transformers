#!/usr/bin/env bash
# Reproduce every experiment, table and figure of the report (CPU only).
# Each queue is resumable: finished runs (with eval.json) are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${WORKERS:-6}

python3 scripts/run_queue.py scripts/jobs_color.txt --workers "$W"   # ~150 colouring runs
python3 scripts/run_queue.py scripts/jobs_seq.txt   --workers "$W"   # 87 length-generalisation runs
python3 scripts/run_queue.py scripts/jobs_dual.txt  --workers 4      # dual-ascent vs augmented Lagrangian

cd analysis
python3 diagram.py
python3 color_analysis.py
python3 seq_analysis.py
python3 extra_analysis.py
cd ../report && tectonic main.tex
