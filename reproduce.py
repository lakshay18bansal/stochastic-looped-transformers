"""One entry point to check, rerun and rebuild everything in this project.

    python reproduce.py check                      # environment check + smoke test (under 1 min)
    python reproduce.py analysis                   # tables and figures from results/   (under 1 min)
    python reproduce.py experiments --workers 8    # all 238 training runs (resumable)
    python reproduce.py all --fresh --workers 8    # everything from scratch, then compare

``--fresh`` moves the shipped results to results_shipped/ (once) and retrains every run
into results/. Interrupted? Run the same command again: finished runs are skipped.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_queue import out_dir, run_all as run_jobs  # noqa: E402

JOB_FILES = ["jobs_seq.txt", "jobs_seq_extra.txt", "jobs_color.txt", "jobs_dual.txt"]   # longest runs first
ANALYSIS = ["diagram.py", "color_analysis.py", "seq_analysis.py", "extra_analysis.py", "report_figures.py",
            "report_numbers.py"]
REQUIRED = {"torch": "2.15.0", "numpy": "2.5.4", "matplotlib": "3.11.2"}


def banner(msg):
    print(f"\n=== {msg} ===", flush=True)


def py(args, cwd=ROOT):
    rc = subprocess.call([sys.executable, *args], cwd=cwd)
    if rc:
        sys.exit(f"command failed ({rc}): python {' '.join(args)}")


# --------------------------------------------------------------------------- check
def check(_):
    banner("environment")
    print("python", sys.version.split()[0], "|", sys.platform, "| CPU cores:", os.cpu_count())
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is needed (tested with 3.12).")
    for mod, ver in REQUIRED.items():
        try:
            m = __import__(mod)
        except ImportError:
            sys.exit(f"missing package '{mod}'. Run: pip install -r requirements.txt")
        have = m.__version__.split("+")[0]
        note = "" if have == ver else f"   (tested with {ver}; other versions may give slightly different numbers)"
        print(f"{mod:11s} {have}{note}")

    banner("smoke test: one tiny run of each experiment type")
    tmp = Path(tempfile.mkdtemp(prefix="stochloop_smoke_"))
    t0 = time.time()
    py(["-m", "stochloop.train_seq", "--threads", "1", "--task", "dyck1", "--method", "rlhalt", "--budget", "3",
        "--seed_mode", "loop", "--m", "2", "--q", "4", "--steps", "20", "--eval_lengths", "5,12",
        "--eval_examples", "32", "--eval_seeds", "2", "--out", str(tmp / "seq")])
    py(["-m", "stochloop.coloring", "--threads", "1", "--steps", "20", "--seed_mode", "nodeloop", "--m", "2",
        "--q", "8", "--method", "rlhalt", "--budget", "3", "--eval_sizes", "8,16", "--eval_graphs", "16",
        "--eval_seeds", "2", "--out", str(tmp / "color")])
    ok = (tmp / "seq" / "eval.json").exists() and (tmp / "color" / "eval.json").exists()
    shutil.rmtree(tmp, ignore_errors=True)
    if not ok:
        sys.exit("smoke test FAILED")
    print(f"smoke test passed in {time.time() - t0:.0f} s. The setup works.")


# --------------------------------------------------------------------------- experiments
def experiments(a):
    res, shipped = ROOT / "results", ROOT / "results_shipped"
    if a.fresh and not shipped.exists():
        if res.exists():
            res.rename(shipped)
            print(f"moved the shipped results to {shipped.name}/ (kept for comparison)")
        res.mkdir()
    jobs = [l.strip() for f in JOB_FILES for l in open(ROOT / "scripts" / f) if l.strip() and not l.startswith("#")]
    todo = [j for j in jobs if not (out_dir(j) / "eval.json").exists()]
    banner(f"experiments: {len(jobs)} runs in total, {len(todo)} still to do, {a.workers} in parallel")
    if not todo:
        print("all runs are finished (results/ already has every eval.json). "
              "Use `all --fresh` to retrain everything from scratch.")
        return
    t0, times = time.time(), []

    def report(i, n, out, rc, dt):
        if rc == 0:
            times.append(dt)
        eta = statistics.mean(times) * (n - i) / a.workers / 3600 if times else float("nan")
        print(f"[{i}/{n}] {'FAILED' if rc else 'done'} in {dt / 60:5.1f} min  {out}"
              f"   | elapsed {(time.time() - t0) / 3600:.1f} h, about {eta:.1f} h left", flush=True)

    failed = run_jobs(todo, a.workers, report)
    if failed:
        sys.exit(f"{len(failed)} run(s) failed, e.g. {failed[0]} (see its stdout.log). Run the command again to retry.")
    print(f"all runs finished in {(time.time() - t0) / 3600:.1f} h")


# --------------------------------------------------------------------------- analysis
def analysis(_):
    banner("tables and figures (analysis/ -> results/tables/, figures/)")
    for script in ANALYSIS:
        print("running", script, flush=True)
        rc = subprocess.call([sys.executable, script], cwd=ROOT / "analysis", stdout=subprocess.DEVNULL)
        if rc:
            sys.exit(f"{script} failed")
    print("done: results/tables/ and figures/ are up to date")


# --------------------------------------------------------------------------- compare
def _key_numbers(ev):
    out = {}
    for r in ev["rows"]:
        if "length" in r:                                    # sequence task
            for k in ("oracle", "policy"):
                out[f"len{r['length']}_{k}"] = r[k]
        else:                                                # colouring
            for k in ("success", "success_sampled_depth", "p05_graph_success", "verify_stop_success"):
                out[f"N{r['n']}_{k}"] = r[k]
    return out


def compare(_):
    res, shipped = ROOT / "results", ROOT / "results_shipped"
    banner("compare results/ with results_shipped/")
    if not shipped.exists():
        print("results_shipped/ does not exist: nothing to compare (it is created by `all --fresh`).")
        return
    exact, close, diff, missing, worst = 0, 0, 0, 0, (0.0, "")
    for f in sorted(shipped.rglob("eval.json")):
        new = res / f.relative_to(shipped)
        if not new.exists():
            missing += 1
            continue
        a, b = _key_numbers(json.load(open(f))), _key_numbers(json.load(open(new)))
        d = max((abs(a[k] - b.get(k, float("nan"))) for k in a), default=0.0)
        if d == 0:
            exact += 1
        elif d <= 0.02:
            close += 1
        else:
            diff += 1
        if d > worst[0]:
            worst = (d, str(new.parent.relative_to(ROOT)))
    n = exact + close + diff
    lines = [f"runs compared: {n}   (missing in results/: {missing})",
             f"identical: {exact}   within 2 points: {close}   differ by more: {diff}",
             f"largest difference: {100 * worst[0]:.1f} points ({worst[1] or '-'})"]
    print("\n".join(lines))
    (res / "compare_report.txt").write_text("\n".join(lines) + "\n")
    print("saved results/compare_report.txt")


def run_all(a):
    experiments(a)
    analysis(a)
    compare(a)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["check", "experiments", "analysis", "compare", "all"])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="runs in parallel (default: half the CPU threads)")
    p.add_argument("--fresh", action="store_true",
                   help="retrain everything: move the shipped results to results_shipped/ first")
    a = p.parse_args()
    {"check": check, "experiments": experiments, "analysis": analysis, "compare": compare,
     "all": run_all}[a.stage](a)


if __name__ == "__main__":
    main()
