"""Run a list of experiment commands with a fixed number of parallel workers.

Each line of the job file is a shell command containing ``--out <dir>``; jobs whose
``<dir>/eval.json`` already exists are skipped, so the queue is resumable.

    python scripts/run_queue.py scripts/jobs_seq.txt --workers 6
"""
import argparse
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run(cmd: str) -> tuple[str, int, float]:
    out = re.search(r"--out\s+(\S+)", cmd).group(1)
    if (Path(out) / "eval.json").exists():
        return out, 0, 0.0
    Path(out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(Path(out) / "stdout.log", "w") as f:
        rc = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    return out, rc, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobfile")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    jobs = [l.strip() for l in open(a.jobfile) if l.strip() and not l.startswith("#")]
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs)):
            out, rc, dt = fut.result()
            print(f"[{i + 1}/{len(jobs)}] rc={rc} {dt / 60:.1f} min  {out}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
