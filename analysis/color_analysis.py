"""Aggregate the cycle-colouring experiments into tables (JSON/Markdown/LaTeX) and figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import style
from style import AQUA, BLUE, INK2, MAGENTA, ORANGE, VIOLET, YELLOW, GREEN, RED, plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "color"
FIG = ROOT / "figures"
TAB = ROOT / "results" / "tables"

MODELS = {   # key -> (label, colour)
    "det_fixed": ("Deterministic, fixed rounds", INK2),
    "det_rlhalt": ("Deterministic, learned halting", ORANGE),
    "seed_q1_fixed": ("Seeded ERM (q=1), fixed", AQUA),
    "seed_q8_fixed": ("Seeded q-norm (q=8), fixed", BLUE),
    "seed_q8_rlhalt": ("Seeded q-norm, learned halting", MAGENTA),
}


def load(group: str, name: str) -> list[dict]:
    runs = []
    for f in sorted((RES / group / name).glob("s*/eval.json")):
        runs.append(json.load(open(f)))
    return runs


def stat(runs, n, key, sub=None):
    vals = []
    for r in runs:
        row = next((x for x in r["rows"] if x["n"] == n), None)
        if row is None:
            continue
        v = row[key] if sub is None else row[key][str(sub)] if str(sub) in row[key] else row[key].get(sub)
        vals.append(v)
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, np.nan)


def policy_key(name):
    # learned-halting models are evaluated with the Las-Vegas rule (sample tau ~ pi),
    # which is the rule whose expected cost equals the training budget.
    return "success_sampled_depth" if "rlhalt" in name else "success"


# --------------------------------------------------------------------------- #
def fig_budget(n=32):
    budgets = {"fixed": [1, 2, 3, 4, 6], "rlhalt": [2, 3, 4, 6]}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    for key, (lab, col) in MODELS.items():
        bs = budgets["rlhalt" if "rlhalt" in key else "fixed"]
        rows = [(B, load("ids", f"{key}_B{B}")) for B in bs]
        rows = [(B, r) for B, r in rows if r]
        if not rows:
            continue
        x = [B for B, _ in rows]
        for ax, metric in zip(axes, ("avg", "p05", "best8")):
            if metric == "avg":
                m = [stat(r, n, policy_key(key)) for _, r in rows]
            elif metric == "p05":
                m = [stat(r, n, "p05_graph_success") for _, r in rows]
            else:
                if "seed" not in key:
                    m = [stat(r, n, policy_key(key)) for _, r in rows]
                else:
                    m = [stat(r, n, "best_of_k", 8) for _, r in rows]
            mu = np.array([v[0] for v in m]); sd = np.array([v[1] for v in m])
            ax.plot(x, mu, marker="o", color=col, label=lab)
            ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
    titles = ("Average success", "Worst-case: 5th-pct per-graph success", "Best-of-8 seeds (verified)")
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.set_xlabel("average rounds B (loop iterations)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel(f"success, N={n}")
    axes[0].legend(loc="lower right")
    style.save(fig, FIG / "color_budget")


def fig_hist(B=2, n=32):
    keys = ["det_fixed", "seed_q1_fixed", "seed_q8_fixed"]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.5), sharey=True)
    for ax, key in zip(axes, keys):
        runs = load("ids", f"{key}_B{B}")
        if not runs:
            continue
        h = np.mean([next(x for x in r["rows"] if x["n"] == n)["graph_success_hist"] for r in runs], 0)
        h = h / h.sum()
        ax.bar(np.arange(10) * 0.1 + 0.05, h, width=0.092, color=MODELS[key][1])
        ax.set_title(MODELS[key][0])
        ax.set_xlabel("per-graph success rate over 16 seeds")
    axes[0].set_ylabel("fraction of graphs")
    style.save(fig, FIG / "color_hist")


def fig_breadth_depth(n=64):
    """Fixed total compute C = k x B: k seeded runs of B rounds + local verification,
    versus one deterministic run of C rounds."""
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    cols = {1: RED, 2: ORANGE, 3: YELLOW, 4: AQUA, 6: BLUE}
    frontier = {}
    for B in (1, 2, 3, 4, 6):
        runs = load("ids", f"seed_q8_fixed_B{B}")
        if not runs:
            continue
        ks = [1, 2, 4, 8, 16]
        mu = [stat(runs, n, "best_of_k", k)[0] for k in ks]
        C = [k * B for k in ks]
        ax.plot(C, mu, marker="o", color=cols[B], label=f"seeded, {B} rounds x k seeds", lw=1.6)
        for c, v in zip(C, mu):
            frontier[c] = max(frontier.get(c, 0), v)
    det = [(B, stat(load("ids", f"det_fixed_B{B}"), n, "success")[0]) for B in (1, 2, 3, 4, 6)]
    det = [(b, v) for b, v in det if not np.isnan(v)]
    ax.plot([d[0] for d in det], [d[1] for d in det], marker="s", color=INK2, ls="--",
            label="deterministic, C rounds")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("total compute C = seeds x rounds")
    ax.set_ylabel(f"success (N={n})")
    ax.set_title("Breadth vs depth at fixed total compute")
    ax.legend(fontsize=6.5, loc="lower right")
    style.save(fig, FIG / "color_breadth_depth")
    return frontier


def fig_anon():
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7), sharey=True)
    keys = [("det_fixed", "Deterministic"), ("seed_q1_fixed", "Seeded ERM"),
            ("seed_q8_fixed", "Seeded q=8"), ("seed_q8_rlhalt", "Seeded q=8, learned halting")]
    for ax, B in zip(axes, (2, 4, 6)):
        for key, lab in keys:
            runs = load("anon", f"{key}_B{B}")
            if not runs:
                continue
            ns = [x["n"] for x in runs[0]["rows"]]
            mu = [stat(runs, n, policy_key(key))[0] for n in ns]
            ax.plot(ns, mu, marker="o", color=MODELS[key][1], label=lab)
            if "seed" in key and "rlhalt" not in key:
                b8 = [stat(runs, n, "best_of_k", 8)[0] for n in ns]
                ax.plot(ns, b8, ls=":", color=MODELS[key][1], lw=1.4)
        ax.set_xscale("log", base=2)
        ax.set_title(f"anonymous ring, B={B} rounds")
        ax.set_xlabel("cycle size N")
    axes[0].set_ylabel("success (dotted: best-of-8)")
    axes[0].legend(fontsize=6.5)
    style.save(fig, FIG / "color_anon")


def fig_size_q():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    n = 64
    ds = [16, 32, 64]
    for key, col, lab in (("det_fixed", INK2, "Deterministic"), ("seed_q8_fixed", BLUE, "Seeded q=8")):
        avg, p05, b8 = [], [], []
        for d in ds:
            runs = load("ids", f"{key}_B3") if d == 32 else load("size", f"{key}_d{d}")
            avg.append(stat(runs, n, "success")[0])
            p05.append(stat(runs, n, "p05_graph_success")[0])
            b8.append(stat(runs, n, "best_of_k", 8)[0] if "seed" in key else avg[-1])
        axes[0].plot(ds, avg, marker="o", color=col, label=f"{lab}: average")
        axes[0].plot(ds, p05, marker="v", ls="--", color=col, label=f"{lab}: 5th pct graph")
        if "seed" in key:
            axes[0].plot(ds, b8, marker="^", ls=":", color=col, label=f"{lab}: best-of-8")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(ds, [str(d) for d in ds])
    axes[0].set_xlabel("width d (parameters ~ d^2)")
    axes[0].set_ylabel(f"success, N={n}, B=3")
    axes[0].set_title("Fixed network size")
    axes[0].legend(fontsize=6)
    qs = [1, 2, 4, 8, 16, 32]
    avg, p05, div, b8 = [], [], [], []
    for q in qs:
        runs = load("ids", f"seed_q{q}_fixed_B3") if q in (1, 8) else load("qsweep", f"seed_q{q}_fixed_B3")
        avg.append(stat(runs, n, "success")[0]); p05.append(stat(runs, n, "p05_graph_success")[0])
        div.append(stat(runs, n, "seed_diverse")[0]); b8.append(stat(runs, n, "best_of_k", 8)[0])
    ax = axes[1]
    ax.plot(qs, avg, marker="o", color=BLUE, label="average")
    ax.plot(qs, p05, marker="v", color=VIOLET, label="5th pct graph")
    ax.plot(qs, b8, marker="^", color=GREEN, label="best-of-8")
    ax.plot(qs, div, marker="s", color=ORANGE, label="seed-dependent graphs")
    ax.set_xscale("log", base=2)
    ax.set_xticks(qs, [str(q) for q in qs])
    ax.set_xlabel("q (1 = ERM, large = min-max)")
    ax.set_title(f"Adversarial strength, N={n}, B=3")
    ax.legend(fontsize=6.5)
    style.save(fig, FIG / "color_size_q")


def fig_halting(n=32):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.8))
    for B, col in zip((2, 3, 4, 6), (ORANGE, AQUA, BLUE, VIOLET)):
        runs = load("ids", f"seed_q8_rlhalt_B{B}")
        if not runs:
            continue
        pi = np.mean([next(x for x in r["rows"] if x["n"] == n)["pi_mean"] for r in runs], 0)
        axes[0].plot(np.arange(1, len(pi) + 1), pi, color=col, label=f"B={B}")
    axes[0].set_xlabel("rounds t")
    axes[0].set_ylabel("pi(t)")
    axes[0].set_title(f"Learned stopping distribution (N={n})")
    axes[0].legend()
    # Jensen gap: fixed depth B vs stochastic depth with mean B
    Bs = [2, 3, 4, 6]
    for key, mk, col, lab in (("seed_q8_fixed", "success", BLUE, "fixed depth B"),
                              ("seed_q8_rlhalt", "success_sampled_depth", MAGENTA, "learned halting, tau~pi, E[tau]=B"),
                              ("seed_q8_rlhalt", "success_online", VIOLET, "learned halting, median stop")):
        mu = [stat(load("ids", f"{key}_B{B}"), n, mk)[0] for B in Bs]
        axes[1].plot(Bs, mu, marker="o", color=col, label=lab)
    vs = [stat(load("ids", f"seed_q8_rlhalt_B{B}"), n, "verify_stop_success")[0] for B in Bs]
    axes[1].plot(Bs, vs, marker="^", ls=":", color=GREEN, label="stop at first valid round (verifier)")
    axes[1].set_xlabel("average rounds B")
    axes[1].set_ylabel(f"success, N={n}")
    axes[1].set_title("Stochastic stopping at fixed mean compute")
    axes[1].legend(fontsize=6)
    style.save(fig, FIG / "color_halting")


def tables():
    TAB.mkdir(parents=True, exist_ok=True)
    lines = []
    out = {}
    hdr = "| setting | model | B | N | success | 5th-pct graph | never-solved graphs | best-of-8 | mean rounds |"
    lines += [hdr, "|" + "---|" * 9]
    for group in ("ids", "anon"):
        for B in (2, 3, 4, 6):
            for key in MODELS:
                runs = load(group, f"{key}_B{B}")
                if not runs:
                    continue
                for n in (16, 64):
                    pk = policy_key(key)
                    s, sd = stat(runs, n, pk)
                    p05 = stat(runs, n, "p05_graph_success")[0]
                    nev = stat(runs, n, "frac_graphs_never")[0]
                    b8 = stat(runs, n, "best_of_k", 8)[0] if "seed" in key else s
                    dk = "depth_sampled" if "rlhalt" in key else "depth_policy"
                    dep = stat(runs, n, dk)[0]
                    out[f"{group}/{key}/B{B}/N{n}"] = dict(success=s, success_std=sd, p05=p05, never=nev,
                                                          best8=b8, rounds=dep, runs=len(runs))
                    lines.append(f"| {group} | {MODELS[key][0]} | {B} | {n} | {s:.3f} ± {sd:.3f} | {p05:.2f} | "
                                 f"{nev:.3f} | {b8:.3f} | {dep:.2f} |")
    (TAB / "color_main.md").write_text("\n".join(lines) + "\n")
    json.dump(out, open(TAB / "color_main.json", "w"), indent=1)
    # ablations
    abl = {}
    for name in ("seed_q8_fixed_B3", "seed_static_q8_B3", "seed_q8_m1_B3", "seed_q8_m8_B3", "det_q8_B3",
                 "seed_q8_rlhalt_ent0.001_B3", "seed_q8_rlhalt_ent0.0_B3", "seed_q8_rlhalt_B3"):
        grp = "ids" if name in ("seed_q8_fixed_B3", "seed_q8_rlhalt_B3") else "ablate"
        runs = load(grp, name)
        if not runs:
            continue
        pk = "success_sampled_depth" if "rlhalt" in name else "success"
        abl[name] = {n: dict(success=stat(runs, n, pk)[0], p05=stat(runs, n, "p05_graph_success")[0],
                             best8=stat(runs, n, "best_of_k", 8)[0], diverse=stat(runs, n, "seed_diverse")[0],
                             online=stat(runs, n, "success_online")[0])
                     for n in (16, 64)}
    json.dump(abl, open(TAB / "color_ablations.json", "w"), indent=1)
    return out, abl


if __name__ == "__main__":
    style.apply()
    FIG.mkdir(exist_ok=True)
    fig_budget()
    fig_hist()
    fr = fig_breadth_depth()
    fig_anon()
    fig_size_q()
    fig_halting()
    out, abl = tables()
    print(json.dumps(abl, indent=1)[:3000])
    print("breadth-depth frontier", fr)
