"""Run a list of experiment commands with a fixed number of parallel workers.

Each line of the job file is one command containing ``--out <dir>``. Jobs whose
``<dir>/eval.json`` already exists are skipped, so the queue can be stopped and
started again at any time. Commands that start with ``python3`` are run with the
same Python interpreter that runs this script (so a virtual environment is used
automatically, on any operating system).

    python scripts/run_queue.py scripts/jobs_seq.txt --workers 6
"""
import argparse
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ACTIVE, _LOCK, _STOP = set(), threading.Lock(), threading.Event()


def out_dir(cmd: str) -> Path:
    out = Path(re.search(r"--out\s+(\S+)", cmd).group(1))
    return out if out.is_absolute() else ROOT / out


def run(cmd: str) -> tuple[str, int, float]:
    """Run one job. Returns (output folder, return code, seconds); 0 seconds means it was already done."""
    d = out_dir(cmd)
    if (d / "eval.json").exists():
        return str(d.relative_to(ROOT) if d.is_relative_to(ROOT) else d), 0, 0.0
    if _STOP.is_set():
        return str(d), -1, 0.0
    d.mkdir(parents=True, exist_ok=True)
    args = shlex.split(cmd)
    if args[0] in ("python3", "python"):
        args[0] = sys.executable
    t0 = time.time()
    with open(d / "stdout.log", "w") as f:
        proc = subprocess.Popen(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        with _LOCK:
            _ACTIVE.add(proc)
        if _STOP.is_set():
            proc.terminate()
        try:
            rc = proc.wait()
        finally:
            with _LOCK:
                _ACTIVE.discard(proc)
    return str(d.relative_to(ROOT) if d.is_relative_to(ROOT) else d), rc, time.time() - t0


def stop_all():
    """Start no new jobs and stop the running ones."""
    _STOP.set()
    with _LOCK:
        procs = list(_ACTIVE)
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()


def install_stop_handlers():
    """Ctrl+C or `kill` stop all jobs. A run started with nohup keeps running when the terminal closes."""
    def handler(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    if hasattr(signal, "SIGHUP") and signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
        signal.signal(signal.SIGHUP, handler)


def run_all(jobs, workers, report):
    """Run jobs in parallel, calling report(i, n, out, rc, seconds) after each one. Returns failed folders."""
    install_stop_handlers()
    failed = []
    ex = ThreadPoolExecutor(workers)
    try:
        futs = [ex.submit(run, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            out, rc, dt = fut.result()
            if rc:
                failed.append(out)
            report(i, len(jobs), out, rc, dt)
    except KeyboardInterrupt:
        print("\nstopping all running jobs ...", flush=True)
        stop_all()
        ex.shutdown(wait=True, cancel_futures=True)
        sys.exit("stopped. Run the same command again to continue; finished runs are kept.")
    ex.shutdown()
    return failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobfile")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    jobs = [l.strip() for l in open(a.jobfile) if l.strip() and not l.startswith("#")]
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)

    def report(i, n, out, rc, dt):
        status = "already done" if dt == 0 and rc == 0 else f"rc={rc} {dt / 60:.1f} min"
        print(f"[{i}/{n}] {status}  {out}", flush=True)

    failed = run_all(jobs, a.workers, report)
    if failed:
        print(f"{len(failed)} job(s) failed; see <out>/stdout.log", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
