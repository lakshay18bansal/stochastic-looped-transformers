"""Recompute every number quoted in the report (report/main.tex) from results/.

Writes results/tables/report_numbers.md. Run from the analysis/ folder (or via reproduce.py).
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "tables" / "report_numbers.md"


def runs(path):
    return [json.load(open(f)) for f in sorted(glob.glob(str(RES / path / "s*" / "eval.json")))]


def logs(path):
    return [[json.loads(l) for l in open(f)] for f in sorted(glob.glob(str(RES / path / "s*" / "train_log.jsonl")))]


def row(r, n, key="n"):
    return next(x for x in r["rows"] if x[key] == n)


def ood(r, key):
    return float(np.mean([x[key] for x in r["rows"] if x["length"] > 10]))


def pct(v):
    return f"{100 * v:.1f}"


L = ["# Numbers used in the report", "", "Recomputed from results/ by analysis/report_numbers.py.", ""]

# ------------------------------------------------------------------ Table 1 (Dyck-1)
L += ["## Table 1: Dyck-1, accuracy (%) on lengths 12-32 (best number of loops)", "",
      "| method | runs | avg. loops (train) | loops (test) | mean +- std | worst run | policy acc. |",
      "|---|---|---|---|---|---|---|"]
T1 = [("K=12", "fixed12"), ("K=6", "fixed6"), ("K=n", "length"), ("K=n+-3", "window"),
      ("RL-Halting, no budget", "rlhalt"), ("budgeted RL-Halting, B=5.5", "rlhaltB"),
      ("budgeted RL-Halting, B=3", "rlhaltB3"), ("budgeted RL-Halting, B=8", "rlhaltB8"),
      ("budgeted RL-Halting + seed, q=4", "seed_rlhaltB_q4"), ("budgeted RL-Halting + seed, q=1", "seed_rlhaltB_q1")]
dyck = {}
for name, key in T1:
    R = runs(f"seq/dyck1/{key}")
    tr = [np.mean([x["depth"] for x in lg if x["step"] >= lg[-1]["step"] / 2]) for lg in logs(f"seq/dyck1/{key}")]
    orc = [ood(r, "oracle") for r in R]
    dyck[key] = orc
    L.append(f"| {name} | {len(R)} | {np.mean(tr):.1f} | {np.mean([ood(r, 'policy_depth') for r in R]):.1f} | "
             f"{pct(np.mean(orc))} +- {pct(np.std(orc))} | {pct(min(orc))} | {pct(np.mean([ood(r, 'policy') for r in R]))} |")
L.append("")
L.append("(loops at test for K=n and K=n+-3 equal the test length n, 20.9 on average over lengths 12-32)")
L.append("")

# ------------------------------------------------------------------ Dyck-1 text numbers
flips = {}
for key in ("rlhalt", "rlhaltB", "rlhaltB3", "rlhaltB8", "seed_rlhaltB_q4", "seed_rlhaltB_q1"):
    per_run = [np.mean([np.mean(x["flip_by_depth"][x["length"]:]) for x in r["rows"] if x["length"] > 10])
               for r in runs(f"seq/dyck1/{key}")]
    flips[key] = float(np.mean(per_run))          # same definition as the flip column of seq_analysis.py
ratios = []
for r in runs("seq/dyck1/rlhaltB3"):
    pi = np.asarray(row(r, 10, "length")["pi_mean"])
    ratios.append(float(np.mean(pi[1:8] / pi[:7])))
L += ["## Dyck-1 numbers in the text", "",
      "- share of predictions that change from one loop to the next (after n loops, lengths 12-32, mean per method): "
      + ", ".join(f"{k} {100 * v:.2f}%" for k, v in flips.items()),
      f"- B=3: stopping probability falls by a factor {np.mean(ratios):.3f} per loop "
      f"(constant hazard with mean 3, cut at 12 loops: 0.669)",
      f"- spread between runs: budgeted RL-Halting {pct(np.std(dyck['rlhaltB']))} -> with seed (q=4) "
      f"{pct(np.std(dyck['seed_rlhaltB_q4']))} points; worst run {pct(min(dyck['rlhaltB']))}% -> "
      f"{pct(min(dyck['seed_rlhaltB_q4']))}%",
      f"- seed with q=1: mean {pct(np.mean(dyck['seed_rlhaltB_q1']))}%",
      f"- K=6 runs: {', '.join(pct(v) for v in dyck['fixed6'])}",
      f"- B=3 runs: {', '.join(pct(v) for v in dyck['rlhaltB3'])}; RL-Halting (no budget) runs: "
      f"{', '.join(pct(v) for v in dyck['rlhalt'])}"]
copy_best = max(ood(r, "oracle") for k in ("fixed12", "fixed6", "length", "window", "rlhalt", "rlhaltB",
                                           "seed_rlhaltB_q1", "seed_rlhaltB_q4", "seed_window_q4")
                for r in runs(f"seq/copy/{k}"))
seeded_id_means = [np.mean([float(np.mean([x["policy"] for x in r["rows"] if x["length"] <= 10]))
                            for r in runs(f"seq/copy/{k}")]) for k in ("seed_rlhaltB_q1", "seed_rlhaltB_q4", "seed_window_q4")]
copy_method_best = max(np.mean([ood(r, "oracle") for r in runs(f"seq/copy/{k}")])
                       for k in ("fixed12", "fixed6", "length", "window", "rlhalt", "rlhaltB",
                                 "seed_rlhaltB_q1", "seed_rlhaltB_q4", "seed_window_q4"))
L += [f"- Copy: best method average on lengths 12-32 {pct(copy_method_best)}% (best single run {pct(copy_best)}%); "
      f"seeded models on training lengths (policy accuracy, mean per method): "
      f"{', '.join(pct(v) for v in seeded_id_means)}%", ""]

# ------------------------------------------------------------------ Table 2 (colouring)
def cstat(group, key, n, metric):
    R = runs(f"color/{group}/{key}")
    v = [row(r, n)[metric] if metric != "best8" else row(r, n)["best_of_k"].get("8") for r in R]
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))


L += ["## Table 2: colouring, 64 vertices", "",
      "| model | IDs B=3 success | never solved | best of 8 | no IDs B=4 success | best of 8 |", "|---|---|---|---|---|---|"]
for name, key, pk in (("no seed, fixed rounds", "det_fixed", "success"),
                      ("no seed, learned stopping", "det_rlhalt", "success_sampled_depth"),
                      ("seed, q=1, fixed rounds", "seed_q1_fixed", "success"),
                      ("seed, q=8, fixed rounds", "seed_q8_fixed", "success"),
                      ("seed, q=8, learned stopping", "seed_q8_rlhalt", "success_sampled_depth")):
    s, sd = cstat("ids", f"{key}_B3", 64, pk)
    nev = cstat("ids", f"{key}_B3", 64, "frac_graphs_never")[0]
    b8 = cstat("ids", f"{key}_B3", 64, "best8")[0]
    a_s = cstat("anon", f"{key}_B4", 64, pk)[0] if runs(f"color/anon/{key}_B4") else float("nan")
    a_b = cstat("anon", f"{key}_B4", 64, "best8")[0] if runs(f"color/anon/{key}_B4") else float("nan")
    L.append(f"| {name} | {s:.2f} +- {sd:.2f} | {pct(nev)}% | {b8:.2f} | {a_s:.2f} | {a_b:.2f} |")
L.append("")

# ------------------------------------------------------------------ colouring text numbers
ex = json.load(open(RES / "tables" / "extra.json"))
jn = ex["jensen"]
d_mix = max(abs(v["measured"] - v["uninformed"]) for m in jn.values() for v in m.values())
gaps = [v["at_mean_depth"] - v["measured"] for m in jn.values() for v in m.values()]
vs = [row(r, 64) for r in runs("color/ids/seed_q8_fixed_B3")]
ok = [r for r in vs if r["frac_graphs_never"] < 0.5]
bd = ex["breadth_depth"]["per_t"]
err = max(abs(v["measured"][k] - v["heterogeneous"][k]) for v in bd.values() for k in v["measured"])
L += ["## Colouring numbers in the text", "",
      f"- learned stopping vs. input-independent random stop: {len(gaps)} settings, max difference {pct(d_mix)} points",
      f"- stopping at round(mean) is better by up to {pct(max(gaps))} points ({sum(g > 0 for g in gaps)} of {len(gaps)} settings)",
      f"- stop when valid, B=3 model: success {pct(np.mean([r['verify_stop_success'] for r in ok]))}% with "
      f"{np.mean([r['verify_stop_depth'] for r in ok]):.2f} rounds on average; fixed 3 rounds: "
      f"{pct(np.mean([r['success'] for r in vs]))}%",
      f"- best rounds per seed t* = {ex['breadth_depth']['t_star']}; 2 seeds x 3 rounds: "
      f"{pct(bd['3']['measured']['2'])}%, 1 seed x 6 rounds: {pct(bd['6']['measured']['1'])}%, "
      f"4 seeds x 3 rounds: {pct(bd['3']['measured']['4'])}%; max prediction error {pct(err)} points"]
for d, key in ((16, "d16"), (64, "d64")):
    s = cstat("size", f"seed_q8_fixed_{key}", 64, "success")[0]
    t = cstat("size", f"det_fixed_{key}", 64, "success")[0]
    p = cstat("size", f"seed_q8_fixed_{key}", 64, "p05_graph_success")[0]
    L.append(f"- width {d}: seeded {pct(s)}% vs. no seed {pct(t)}%; seeded 5th-percentile graph {p:.2f}")
L.append(f"- width 32 (main runs): seeded 5th-percentile graph "
         f"{cstat('ids', 'seed_q8_fixed_B3', 64, 'p05_graph_success')[0]:.2f}")
qs = {q: cstat("qsweep", f"seed_q{q}_fixed_B3", 64, "success")[0] for q in (2, 4, 16, 32)}
qs[8] = cstat("ids", "seed_q8_fixed_B3", 64, "success")[0]
qs[1] = cstat("ids", "seed_q1_fixed_B3", 64, "success")[0]
L.append("- q sweep (B=3, 64 vertices): " + ", ".join(f"q={q}: {pct(qs[q])}%" for q in sorted(qs)))
for key in ("plain_dual", "aug_lagrangian"):
    d = ex["dual"][key]
    L.append(f"- {key}: training depth {min(x['train_depth_min'] for x in d):.1f}-{max(x['train_depth_max'] for x in d):.1f} "
             f"(2nd half of training), test depth {', '.join(f'{x['test_depth']:.2f}' for x in d)}")
L.append("")

# ------------------------------------------------------------------ compute
tot = 0.0
for jf in ("jobs_seq", "jobs_seq_extra", "jobs_color", "jobs_dual"):
    ts = []
    for line in open(ROOT / "scripts" / f"{jf}.txt"):
        if "--out" in line:
            out = line.split("--out ")[1].split()[0]
            ts.append(json.load(open(ROOT / out / "eval.json"))["train_time_s"])
    med = statistics.median(ts)
    tot += sum(min(t, 2 * med) for t in ts)      # runs paused by laptop sleep are capped at 2x the median
L += ["## Compute", "", f"- {sum(1 for _ in RES.rglob('eval.json'))} runs, about {tot / 3600:.0f} CPU-hours", ""]

OUT.write_text("\n".join(L) + "\n")
print("\n".join(L))
