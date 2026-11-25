"""Aggregate the length-generalisation experiments (Dyck-1, Copy) into tables and figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import style
from style import AQUA, BLUE, GREEN, INK2, MAGENTA, ORANGE, RED, VIOLET, YELLOW, plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "seq"
FIG = ROOT / "figures"
TAB = ROOT / "results" / "tables"
TRAIN_MAX = 10

METHODS = {  # key -> (label, colour, policy description)
    "fixed12": ("K=12 (fixed)", YELLOW),
    "fixed6": ("K=6 (fixed, budget-matched)", RED),
    "length": ("K=n", ORANGE),
    "window": ("K=n±3 (uniform)", AQUA),
    "rlhalt": ("RL-Halt (unconstrained)", VIOLET),
    "rlhaltB": ("RL-Halt, E[τ]=5.5 (ours)", BLUE),
    "seed_rlhaltB_q1": ("Seeded RL-Halt, q=1 (ours)", GREEN),
    "seed_rlhaltB_q4": ("Seeded RL-Halt, q=4 (ours)", MAGENTA),
    "seed_window_q4": ("Seeded K=n±3, q=4", INK2),
}


def load(task, key):
    runs = []
    for d in sorted((RES / task / key).glob("s*")):
        f = d / "eval.json"
        if f.exists():
            r = json.load(open(f))
            log = [json.loads(l) for l in open(d / "train_log.jsonl")]
            half = [x for x in log if x["step"] >= log[-1]["step"] / 2]
            r["train_depth"] = float(np.mean([x["depth"] for x in half]))
            r["train_unroll"] = float(np.mean([x["unroll"] for x in half]))
            r["train_acc"] = float(np.mean([x["acc"] for x in log[-5:]]))
            runs.append(r)
    return runs


def curve(runs, key):
    lens = [x["length"] for x in runs[0]["rows"]]
    arr = np.array([[x[key] for x in r["rows"]] for r in runs])
    return np.array(lens), arr


def ood_mean(run, key):
    return float(np.mean([x[key] for x in run["rows"] if x["length"] > TRAIN_MAX]))


# --------------------------------------------------------------------------- #
def table(task):
    rows, out = [], {}
    for key, (lab, _) in METHODS.items():
        runs = load(task, key)
        if not runs:
            continue
        lens, orc = curve(runs, "oracle")
        ood_mask = lens > TRAIN_MAX
        mean_curve = orc.mean(0)
        front = [l for l, v in zip(lens[ood_mask], mean_curve[ood_mask]) if v >= 0.9]
        per = {k: [ood_mean(r, k) for r in runs] for k in
               ("oracle", "policy", "online", "depth_vote", "seed_vote", "policy_depth", "online_depth", "seed_diverse")}
        flip = []
        for r in runs:
            fl = [np.mean(x["flip_by_depth"][x["length"]:]) for x in r["rows"] if x["length"] > TRAIN_MAX]
            flip.append(np.mean(fl))
        ent = [np.mean([x["entropy"] for x in r["rows"]]) for r in runs]
        d = dict(
            label=lab, runs=len(runs), params=runs[0]["n_params"],
            train_depth=float(np.mean([r["train_depth"] for r in runs])),
            train_unroll=float(np.mean([r["train_unroll"] for r in runs])),
            id_acc=float(np.mean([np.mean([x["policy"] for x in r["rows"] if x["length"] <= TRAIN_MAX]) for r in runs])),
            frontier90=int(max(front)) if front else TRAIN_MAX,
            flip_after_n=float(np.mean(flip)), entropy_bits=float(np.mean(ent)),
        )
        for k, v in per.items():
            d[f"ood_{k}"] = float(np.mean(v))
            d[f"ood_{k}_std"] = float(np.std(v))
        d["gap"] = d["ood_oracle"] - d["ood_policy"]
        out[key] = d
        rows.append(d)
    TAB.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(TAB / f"seq_{task}.json", "w"), indent=1)
    md = ["| method | runs | train E[depth] | inference depth (OOD) | ID acc | OOD oracle | OOD policy | depth-vote | seed-vote | std (oracle) | gap | Front@90 | flip |",
          "|" + "---|" * 13]
    for d in rows:
        md.append(f"| {d['label']} | {d['runs']} | {d['train_depth']:.2f} | {d['ood_policy_depth']:.1f} | {100*d['id_acc']:.1f} | "
                  f"{100*d['ood_oracle']:.1f} | {100*d['ood_policy']:.1f} | {100*d['ood_depth_vote']:.1f} | {100*d['ood_seed_vote']:.1f} | "
                  f"{100*d['ood_oracle_std']:.1f} | {100*d['gap']:.1f} | {d['frontier90']} | {d['flip_after_n']:.3f} |")
    (TAB / f"seq_{task}.md").write_text("\n".join(md) + "\n")
    return out


def fig_curves(task, keys):
    fig, axes = plt.subplots(1, len(keys), figsize=(2.15 * len(keys), 2.4), sharey=True)
    for ax, key in zip(axes, keys):
        runs = load(task, key)
        if not runs:
            continue
        lab, col = METHODS[key]
        lens, orc = curve(runs, "oracle")
        for c in orc:
            ax.plot(lens, c, color=col, lw=0.8, alpha=0.45)
        ax.plot(lens, orc.mean(0), color=col, lw=2.2)
        _, pol = curve(runs, "policy")
        ax.plot(lens, pol.mean(0), color=col, lw=1.3, ls="--")
        ax.axvspan(0, TRAIN_MAX + 0.5, color="#f0efec", zorder=0)
        ax.set_title(lab, fontsize=8)
        ax.set_xlabel("length n")
        ax.set_xlim(4, 33)
    axes[0].set_ylabel("accuracy (solid: oracle, dashed: policy)")
    style.save(fig, FIG / f"seq_{task}_curves")


def fig_spread(tasks, keys):
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.8 * len(tasks), 2.9))
    for ax, task in zip(np.atleast_1d(axes), tasks):
        for i, key in enumerate(keys):
            runs = load(task, key)
            if not runs:
                continue
            lab, col = METHODS[key]
            o = [ood_mean(r, "oracle") for r in runs]
            p = [ood_mean(r, "policy") for r in runs]
            jit = np.linspace(-0.12, 0.12, len(o))
            ax.scatter(i + jit - 0.17, o, s=16, color=col, edgecolor="white", linewidth=0.6, zorder=3)
            ax.scatter(i + jit + 0.17, p, s=16, color=col, marker="D", edgecolor="white", linewidth=0.6, zorder=3, alpha=0.7)
            ax.plot([i - 0.32, i - 0.02], [np.mean(o)] * 2, color=col, lw=2.2)
            ax.plot([i + 0.02, i + 0.32], [np.mean(p)] * 2, color=col, lw=2.2, ls=":")
        ax.set_xticks(range(len(keys)), [METHODS[k][0] for k in keys], rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("mean OOD accuracy (n=12..32)")
        ax.set_title(f"{task}: per-run OOD accuracy (●oracle  ◆policy)")
        ax.set_ylim(-0.03, 1.03)
    style.save(fig, FIG / "seq_spread")


def fig_depth(task, keys, lengths=(10, 16, 24)):
    fig, axes = plt.subplots(2, len(keys), figsize=(2.2 * len(keys), 4.2), sharex=True, sharey="row")
    cols = [ORANGE, BLUE, VIOLET]
    for j, key in enumerate(keys):
        runs = load(task, key)
        if not runs:
            continue
        for L, c in zip(lengths, cols):
            acc = np.mean([next(x for x in r["rows"] if x["length"] == L)["acc_by_depth"] for r in runs], 0)
            fl = np.mean([next(x for x in r["rows"] if x["length"] == L)["flip_by_depth"] for r in runs], 0)
            axes[0, j].plot(np.arange(1, len(acc) + 1), acc, color=c, lw=1.6, label=f"n={L}")
            axes[1, j].plot(np.arange(1, len(fl) + 1), fl, color=c, lw=1.6)
        axes[0, j].set_title(METHODS[key][0], fontsize=8)
        axes[1, j].set_xlabel("inference loops K")
    axes[0, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("flip rate")
    axes[0, 0].legend(fontsize=7)
    style.save(fig, FIG / f"seq_{task}_depth")


def fig_stop(task, keys):
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(2.6 * (len(keys) + 1), 2.5))
    for ax, key in zip(axes, keys):
        runs = load(task, key)
        if not runs:
            continue
        for L, c in zip((5, 10, 16, 24, 32), (YELLOW, ORANGE, AQUA, BLUE, VIOLET)):
            pi = np.mean([next(x for x in r["rows"] if x["length"] == L)["pi_mean"] for r in runs], 0)
            ax.plot(np.arange(1, len(pi) + 1), pi, color=c, lw=1.5, label=f"n={L}")
        ax.set_title(METHODS[key][0], fontsize=8)
        ax.set_xlabel("loop t")
    axes[0].set_ylabel("π(t | x)")
    axes[0].legend(fontsize=6.5)
    ax = axes[-1]
    for key in keys:
        runs = load(task, key)
        if not runs:
            continue
        lens, ent = curve(runs, "entropy")
        ax.plot(lens, ent.mean(0), color=METHODS[key][1], marker="o", label=METHODS[key][0])
    w = np.log2(7)
    ax.axhline(w, color=INK2, ls=":", lw=1)
    ax.text(31, w + 0.05, "uniform ±3", ha="right", fontsize=7, color=INK2)
    ax.set_xlabel("length n")
    ax.set_ylabel("stopping entropy (bits)")
    ax.legend(fontsize=6)
    style.save(fig, FIG / f"seq_{task}_stopping")


def fig_compute(tasks):
    """Mean OOD policy accuracy (large marker) and every run (small markers) against the
    number of loops actually executed per OOD input."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks), 3.0))
    handles = {}
    for ax, task in zip(np.atleast_1d(axes), tasks):
        for key, (lab, col) in METHODS.items():
            runs = load(task, key)
            if not runs:
                continue
            xs = [ood_mean(r, "policy_depth") for r in runs]
            ys = [ood_mean(r, "policy") for r in runs]
            ax.scatter(xs, ys, s=10, color=col, alpha=0.45, lw=0)
            handles[lab] = ax.scatter(np.mean(xs), np.mean(ys), s=55, color=col, edgecolor="white", zorder=3)
            if key.startswith("rlhalt") or key.startswith("seed_rlhalt"):
                xo = np.mean([ood_mean(r, "online_depth") for r in runs])
                yo = np.mean([ood_mean(r, "online") for r in runs])
                ax.scatter(xo, yo, marker="^", s=40, color=col, edgecolor="white", zorder=3)
        ax.set_xlabel("loops per OOD input")
        ax.set_ylabel("OOD accuracy (policy)")
        ax.set_title(f"{task}")
    handles["on-line median stop"] = plt.Line2D([], [], marker="^", ls="", color=INK2)
    fig.legend(list(handles.values()), list(handles), fontsize=6.5, loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    style.save(fig, FIG / "seq_compute")


if __name__ == "__main__":
    style.apply()
    FIG.mkdir(exist_ok=True)
    core = ["fixed12", "fixed6", "length", "window", "rlhalt", "rlhaltB", "seed_rlhaltB_q4"]
    tasks = [t for t in ("dyck1", "copy") if (RES / t).exists()]
    for t in tasks:
        res = table(t)
        print(t)
        for k, d in res.items():
            print(f"  {k:18s} runs={d['runs']} trainD={d['train_depth']:.2f} infD={d['ood_policy_depth']:.1f} "
                  f"ID={d['id_acc']:.3f} orc={d['ood_oracle']:.3f}±{d['ood_oracle_std']:.3f} pol={d['ood_policy']:.3f}±{d['ood_policy_std']:.3f} "
                  f"onl={d['ood_online']:.3f} dv={d['ood_depth_vote']:.3f} sv={d['ood_seed_vote']:.3f} div={d['ood_seed_diverse']:.3f} F90={d['frontier90']} flip={d['flip_after_n']:.3f}")
        fig_curves(t, core)
        fig_depth(t, ["fixed12", "length", "window", "rlhalt", "rlhaltB", "seed_rlhaltB_q4"])
        fig_stop(t, ["rlhalt", "rlhaltB", "seed_rlhaltB_q4"])
    fig_spread(tasks, list(METHODS))
    fig_compute(tasks)
