"""The two compact figures used in the short report (figures/report_dyck, figures/report_color)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import style
from style import BLUE, GREEN, INK2, MAGENTA, ORANGE, VIOLET, YELLOW, plt

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "results" / "seq" / "dyck1"
COL = ROOT / "results" / "color"
FIG = ROOT / "figures"


def seq_runs(key):
    return [json.load(open(f)) for f in sorted((SEQ / key).glob("s*/eval.json"))]


def col_runs(group, key):
    return [json.load(open(f)) for f in sorted((COL / group / key).glob("s*/eval.json"))]


def ood(run, key="oracle"):
    return float(np.mean([r[key] for r in run["rows"] if r["length"] > 10]))


def row(run, n, key="n"):
    return next(r for r in run["rows"] if r[key] == n)


# --------------------------------------------------------------------------- #
def fig_dyck():
    methods = [("fixed12", "K=12", INK2), ("fixed6", "K=6", INK2), ("window", "K=n±3", INK2),
               ("rlhalt", "RL-Halt\nno budget", VIOLET), ("rlhaltB", "RL-Halt\nB=5.5", BLUE),
               ("rlhaltB3", "RL-Halt\nB=3", BLUE), ("seed_rlhaltB_q4", "RL-Halt\n+ seed", MAGENTA)]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.05), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    for i, (k, lab, c) in enumerate(methods):
        v = [ood(r) for r in seq_runs(k)]
        jit = np.linspace(-0.12, 0.12, len(v))
        ax.scatter(i + jit, v, s=16, color=c, alpha=0.85, lw=0, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=c, lw=2.2)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m[1] for m in methods], fontsize=5.8)
    ax.set_ylabel("accuracy on lengths 12-32")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("(a) all training runs (dots), mean (bar)")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    for k, lab, c, ls in [("fixed12", "K=12", INK2, "-"), ("fixed6", "K=6", YELLOW, "-"),
                          ("rlhaltB3", "RL-Halt B=3", BLUE, "-")]:
        runs = seq_runs(k)
        acc = np.mean([row(r, 24, "length")["acc_by_depth"] for r in runs], 0)
        ax.plot(np.arange(1, len(acc) + 1), acc, color=c, ls=ls, label=lab)
    ax.set_xlabel("loops at test time")
    ax.set_ylabel("accuracy (length 24)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("(b) accuracy vs. test loops")
    ax.text(20, 0.91, "RL-Halt B=3", color=BLUE, fontsize=6.5)
    ax.text(2, 0.68, "K=6", color=YELLOW, fontsize=6.5)
    ax.text(24, 0.56, "K=12", color=INK2, fontsize=6.5)
    fig.tight_layout(w_pad=1.5)
    style.save(fig, FIG / "report_dyck")


# --------------------------------------------------------------------------- #
def fig_color():
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.95))

    # (a) per-graph success over 16 seeds, B=2, N=32
    ax = axes[0]
    bins = np.linspace(0.05, 0.95, 10)
    for key, lab, c, off in [("det_fixed_B2", "no seed", INK2, -0.022), ("seed_q8_fixed_B2", "with seed (q=8)", BLUE, 0.022)]:
        h = np.sum([row(r, 32)["graph_success_hist"] for r in col_runs("ids", key)], 0).astype(float)
        ax.bar(bins + off, h / h.sum(), width=0.044, color=c, label=lab)
    ax.set_xlabel("share of seeds that work")
    ax.set_ylabel("share of graphs")
    ax.set_title("(a) per-graph success")
    ax.legend(fontsize=5.8, loc="upper center")

    # (b) anonymous rings, B=4
    ax = axes[1]
    ns = [8, 16, 32, 64]
    det = [np.mean([row(r, n)["success"] for r in col_runs("anon", "det_fixed_B4")]) for n in ns]
    sd = col_runs("anon", "seed_q8_fixed_B4")
    one = [np.mean([row(r, n)["success"] for r in sd]) for n in ns]
    b8 = [np.mean([row(r, n)["best_of_k"]["8"] for r in sd]) for n in ns]
    ax.plot(ns, b8, color=GREEN, marker="o", label="seed, best of 8")
    ax.plot(ns, one, color=BLUE, marker="o", label="seed, 1 try")
    ax.plot(ns, det, color=INK2, marker="s", label="no seed")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("cycle size N")
    ax.set_ylabel("success")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("(b) no IDs, 4 rounds")
    ax.legend(fontsize=5.8, loc="center left")

    # (c) seeded q=8 at N=64: success vs rounds actually used
    ax = axes[2]
    Bs = [2, 3, 4, 6]
    for kind, lab, c, mk in [("fixed", "fixed", BLUE, "o"), ("rlhalt", "learned stop", ORANGE, "^"),
                             ("verify", "stop when valid", GREEN, "s")]:
        xs, ys = [], []
        for B in Bs:
            runs = col_runs("ids", f"seed_q8_{'rlhalt' if kind == 'rlhalt' else 'fixed'}_B{B}")
            r64 = [row(r, 64) for r in runs]
            if kind == "fixed":
                xs.append(B); ys.append(np.mean([r["success"] for r in r64]))
            elif kind == "rlhalt":
                xs.append(np.mean([r["depth_sampled"] for r in r64])); ys.append(np.mean([r["success_sampled_depth"] for r in r64]))
            else:
                ok = [r for r in r64 if r["frac_graphs_never"] < 0.5]       # leave out the one collapsed run
                xs.append(np.mean([r["verify_stop_depth"] for r in ok])); ys.append(np.mean([r["verify_stop_success"] for r in ok]))
        ax.plot(xs, ys, color=c, marker=mk, label=lab)
    ax.set_xlabel("avg. rounds used")
    ax.set_ylabel("success (N=64)")
    ax.set_ylim(0.3, 1.03)
    ax.set_title("(c) same avg. compute")
    ax.legend(fontsize=5.8, loc="lower right")
    fig.tight_layout(w_pad=1.0)
    style.save(fig, FIG / "report_color")


if __name__ == "__main__":
    style.apply()
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
                         "lines.linewidth": 1.6, "lines.markersize": 3.5})
    fig_dyck()
    fig_color()
