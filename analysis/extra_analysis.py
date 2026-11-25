"""Supporting analyses for the theory section.

1. Dual ascent vs augmented Lagrangian for the budget constraint E[tau] = B (Appendix).
2. Breadth vs depth (Prop. 4): predict best-of-k success from single-seed success s(t),
   under seed independence and with the measured per-graph heterogeneity, and compare to
   the measured Las Vegas amplification; report the predicted optimal depth t*.
3. Jensen penalty of learned stopping (Prop. 3): measured success with tau ~ pi versus
   (a) the uninformed mixture sum_t pi(t) s(t) and (b) a deterministic stop at round(E[tau]).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import style
from style import AQUA, BLUE, INK2, MAGENTA, ORANGE, RED, YELLOW, plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "color"
FIG = ROOT / "figures"
TAB = ROOT / "results" / "tables"


def _rows(path: Path) -> dict:
    return {r["n"]: r for r in json.load(open(path))["rows"]}


def _runs(group: str, name: str) -> list[dict]:
    return [_rows(f) for f in sorted((RES / group / name).glob("s*/eval.json"))]


# --------------------------------------------------------------------------- #
def fig_dual(budget: float = 4.0) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.8))
    variants = [("plain_dual", "dual ascent ($\\rho=0$)", ORANGE),
                ("aug_lagrangian", "augmented Lagrangian ($\\rho=1$)", BLUE)]
    summary = {}
    for key, lab, col in variants:
        for s, f in enumerate(sorted((RES / "dual" / key).glob("s*/train_log.jsonl"))):
            log = [json.loads(line) for line in open(f)]
            step = [x["step"] for x in log]
            kw = dict(color=col, lw=1.3, alpha=1.0 if s == 0 else 0.55, ls="-" if s == 0 else "--",
                      label=lab if s == 0 else None)
            axes[0].plot(step, [x["depth"] for x in log], **kw)
            axes[1].plot(step, [x["lam"] for x in log], **kw)
            half = [x["depth"] for x in log[len(log) // 2:]]
            ev = _rows(f.parent / "eval.json")
            summary.setdefault(key, []).append(dict(
                train_depth_mean=float(np.mean(half)), train_depth_std=float(np.std(half)),
                train_depth_min=float(min(half)), train_depth_max=float(max(half)),
                test_depth=float(np.mean([r["depth_sampled"] for r in ev.values()])),
                success_n64=ev[64]["success_sampled_depth"], success_n16=ev[16]["success_sampled_depth"]))
    axes[0].axhline(budget, color=INK2, lw=0.8, ls=":")
    axes[0].set_ylabel("$\\mathbb{E}_\\pi[\\tau]$ on training batch")
    axes[0].set_title(f"Mean depth (budget $B={budget:g}$)")
    axes[1].axhline(0, color=INK2, lw=0.8, ls=":")
    axes[1].set_ylabel("multiplier $\\lambda$")
    axes[1].set_title("Dual variable")
    for ax in axes:
        ax.set_xlabel("training step")
    axes[0].legend(fontsize=7, loc="upper right")
    style.save(fig, FIG / "dual_trajectory")
    return summary


# --------------------------------------------------------------------------- #
def _hetero_pred(hist: list[int], s: float, k: int) -> float:
    """E_g[1 - (1 - p_g)^k] with p_g at the centres of the 10-bin per-graph histogram.
    The top bin is placed at 1 (graphs solved by every seed); the bottom bin's value is
    calibrated so that the mixture mean equals the measured single-seed success s."""
    h = np.asarray(hist, float)
    h = h / h.sum()
    c = np.linspace(0.05, 0.95, 10)
    if h[-1]:
        c[-1] = 1.0
    if h[0]:
        c[0] = float(np.clip((s - (h[1:] * c[1:]).sum()) / h[0], 0.0, 0.1))
    return float((h * (1 - (1 - c) ** k)).sum())


def breadth_depth(n: int = 64) -> dict:
    ks = [1, 2, 4, 8, 16]
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.9))
    cols = {1: RED, 2: ORANGE, 3: YELLOW, 4: AQUA, 6: BLUE}
    for t in (1, 2, 3, 4, 6):
        runs = _runs("ids", f"seed_q8_fixed_B{t}")
        if not runs:
            continue
        s = float(np.mean([r[n]["success"] for r in runs]))
        meas = [float(np.mean([r[n]["best_of_k"][str(k)] for r in runs])) for k in ks]
        indep = [1 - (1 - s) ** k for k in ks]
        het = [float(np.mean([_hetero_pred(r[n]["graph_success_hist"], r[n]["success"], k) for r in runs]))
               for k in ks]
        g = -np.log(max(1 - s, 1e-6))
        out[t] = dict(s=s, rate=g / t, measured=dict(zip(ks, meas)), independent=dict(zip(ks, indep)),
                      heterogeneous=dict(zip(ks, het)))
        C = [k * t for k in ks]
        axes[0].plot(C, meas, color=cols[t], marker="o", lw=1.5, label=f"t={t}")
        axes[0].plot(C, indep, color=cols[t], ls=":", lw=1.0)
        axes[0].plot(C, het, color=cols[t], ls="--", lw=1.0)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("total compute C = k seeds x t rounds")
    axes[0].set_ylabel(f"best-of-k success (N={n})")
    axes[0].set_title("Measured (solid) vs predicted")
    axes[0].plot([], [], color=INK2, ls=":", label="independent seeds")
    axes[0].plot([], [], color=INK2, ls="--", label="+ graph heterogeneity")
    axes[0].legend(fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.33), ncol=2)
    ts = sorted(out)
    axes[1].bar([str(t) for t in ts], [out[t]["rate"] for t in ts], color=[cols[t] for t in ts])
    axes[1].set_xlabel("rounds per seed t")
    axes[1].set_ylabel("$g(t)/t = -\\log(1-s(t))/t$")
    axes[1].set_title("Amplification rate per round (Prop. 4)")
    style.save(fig, FIG / "breadth_depth_theory")
    t_star = max(ts, key=lambda t: out[t]["rate"])
    return dict(per_t=out, t_star=t_star)


# --------------------------------------------------------------------------- #
def jensen() -> dict:
    """For each learned-halting colouring model: measured success under tau ~ pi versus the
    uninformed mixture sum_t pibar(t) s(t) (stopping independent of the input) and versus
    the same model stopped deterministically at round(E[tau])."""
    res = {}
    for group, names in (("ids", [f"{p}_B{b}" for p in ("det_rlhalt", "seed_q8_rlhalt") for b in (2, 3, 4, 6)]),
                         ("anon", [f"seed_q8_rlhalt_B{b}" for b in (2, 4, 6)])):
        for name in names:
            runs = _runs(group, name)
            if not runs:
                continue
            rec = {}
            for n in (16, 64):
                meas, mix, at_mean, tau = [], [], [], []
                for r in runs:
                    row = r[n]
                    pi, s = np.asarray(row["pi_mean"]), np.asarray(row["success_by_depth"])
                    et = float((pi * np.arange(1, len(pi) + 1)).sum())
                    meas.append(row["success_sampled_depth"])
                    mix.append(float((pi * s[:len(pi)]).sum()))
                    at_mean.append(float(s[int(round(et)) - 1]))
                    tau.append(et)
                rec[n] = dict(measured=float(np.mean(meas)), uninformed=float(np.mean(mix)),
                              at_mean_depth=float(np.mean(at_mean)), mean_tau=float(np.mean(tau)))
            res[f"{group}/{name}"] = rec
    return res


def jensen_table(res: dict) -> str:
    lines = ["| model | N | E[tau] | stop at round(E[tau]) | uninformed mixture | learned tau ~ pi |",
             "|---|---|---|---|---|---|"]
    for k, rec in res.items():
        for n, v in rec.items():
            lines.append(f"| {k} | {n} | {v['mean_tau']:.2f} | {v['at_mean_depth']:.3f} | "
                         f"{v['uninformed']:.3f} | {v['measured']:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    style.apply()
    TAB.mkdir(parents=True, exist_ok=True)
    dual = fig_dual()
    bd = breadth_depth()
    jn = jensen()
    json.dump(dict(dual=dual, breadth_depth=bd, jensen=jn), open(TAB / "extra.json", "w"), indent=1)
    (TAB / "jensen.md").write_text(jensen_table(jn))
    print(json.dumps(dual, indent=1))
    print("t* =", bd["t_star"], {t: round(v["rate"], 3) for t, v in bd["per_t"].items()})
    for t, v in bd["per_t"].items():
        print(t, "meas", {k: round(x, 3) for k, x in v["measured"].items()},
              "indep", {k: round(x, 3) for k, x in v["independent"].items()},
              "het", {k: round(x, 3) for k, x in v["heterogeneous"].items()})
    print(jensen_table(jn))
