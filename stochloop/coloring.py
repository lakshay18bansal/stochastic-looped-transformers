"""Distributed 3-colouring of cycles with a *looped*, locally-masked transformer.

Each vertex is a token; the attention mask only lets a vertex see itself and its
two cycle neighbours, so one transformer layer = one synchronous communication
round, and the number of loop iterations is the number of rounds (= compute).

Vertices carry unique identifiers drawn at random from a pool of P ids (the
cycle is therefore a uniformly random Hamiltonian cycle over the chosen ids),
or are anonymous (``--anonymous``), in which case no deterministic algorithm can
break symmetry and randomness is strictly necessary.

Training loss (per graph, per seed): the partial colouring loss
        sum_{(u,v) in E} sum_c P(C_u = c) P(C_v = c),
an upper bound on the probability of an invalid colouring, aggregated over
graphs x seeds with the q-norm objective.  Depth (number of rounds) is fixed,
randomised, or learned with (budget-constrained) RL-Halting.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .model import LoopedTransformer, exit_distribution, init_weights, n_params
from .objectives import BudgetDual, MovingBaseline, qnorm, rlhalt_loss, scheduled_loss

N_COLORS = 3


def ring_mask(n: int) -> torch.Tensor:
    i = torch.arange(n)
    d = (i.unsqueeze(0) - i.unsqueeze(1)) % n
    return (d == 0) | (d == 1) | (d == n - 1)


def sample_graphs(b: int, n: int, pool: int, anonymous: bool, rng: np.random.Generator) -> torch.Tensor:
    if anonymous:
        return torch.zeros(b, n, dtype=torch.long)
    ids = np.stack([rng.permutation(pool)[:n] for _ in range(b)]) + 1
    return torch.from_numpy(ids).long()


def coloring_loss(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(-1)                                   # [B, N, C]
    return (p * p.roll(-1, dims=1)).sum(-1).sum(-1)          # sum over edges (i, i+1)


def valid_coloring(logits: torch.Tensor) -> torch.Tensor:
    c = logits.argmax(-1)
    return (c != c.roll(-1, dims=1)).all(-1)


def valid_sampled(logits: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    p = logits.softmax(-1)
    c = torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=gen).view(p.shape[:-1])
    return (c != c.roll(-1, dims=1)).all(-1)


# --------------------------------------------------------------------------- #
def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="fixed", choices=["fixed", "window", "rlhalt"])
    p.add_argument("--budget", type=float, default=4.0, help="average number of rounds")
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--t_max", type=int, default=12)
    p.add_argument("--dual_lr", type=float, default=0.005)
    p.add_argument("--rho", type=float, default=1.0, help="augmented-Lagrangian weight (0 = plain dual ascent)")
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--seed_mode", default="nodeloop")
    p.add_argument("--m", type=int, default=4)
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--anonymous", action="store_true")
    p.add_argument("--pool", type=int, default=64)
    p.add_argument("--n_min", type=int, default=6)
    p.add_argument("--n_max", type=int, default=16)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--run_seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--eval_sizes", default="8,16,32,64")
    p.add_argument("--eval_graphs", type=int, default=256)
    p.add_argument("--eval_seeds", type=int, default=16)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    prog = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, a, eval_seed: int = 7) -> dict:
    """Per size: success over (graph, seed), per-graph success distribution, Las Vegas
    best-of-k amplification, per-round success curves and the halting distribution."""
    model.eval()
    halting = a.method == "rlhalt"
    seeded = a.seed_mode not in ("none", "fixed")
    S = a.eval_seeds if seeded else 1
    H = a.t_max + 8                                       # test horizon beyond training
    rows = []
    for n in [int(v) for v in a.eval_sizes.split(",")]:
        if not a.anonymous and n > a.pool:
            continue
        rng = np.random.default_rng(eval_seed * 1000 + n)
        gen = torch.Generator().manual_seed(eval_seed + n)
        x = sample_graphs(a.eval_graphs, n, a.pool, a.anonymous, rng)
        mask = ring_mask(n)
        ok_depth = torch.zeros(S, H, a.eval_graphs, dtype=torch.bool)
        ok_samp = torch.zeros(S, a.eval_graphs, dtype=torch.bool)
        pis = torch.zeros(S, H, a.eval_graphs)
        for s in range(S):
            seeds = model.draw_seeds(a.seed_mode, a.eval_graphs, n, H, generator=gen)
            logits, hz = model(x, H, seeds=seeds, attn_mask=mask, need_halt=halting)
            ok_depth[s] = torch.stack([valid_coloring(l) for l in logits])
            if halting:
                pis[s] = exit_distribution(hz)
            k = _policy_depth(a, n) - 1
            ok_samp[s] = valid_sampled(logits[min(k, H - 1)], gen)
        idx = torch.arange(a.eval_graphs)
        if halting:
            pol = pis.argmax(1)                                             # [S, G]
            onl = (pis.cumsum(1) >= 0.5).float().argmax(1)
            # Las Vegas rule: sample a stopping time from pi (run-time is random)
            smp = torch.distributions.Categorical(probs=pis.permute(0, 2, 1)).sample()
        else:
            k = min(_policy_depth(a, n), H) - 1
            pol = onl = smp = torch.full((S, a.eval_graphs), k, dtype=torch.long)
        ok = lambda d: torch.stack([ok_depth[s, d[s], idx] for s in range(S)])   # [S, G]
        ok_pol, ok_onl, ok_smp = ok(pol), ok(onl), ok(smp)
        per_graph = ok_pol.float().mean(0)
        row = {
            "n": n,
            "success": float(ok_pol.float().mean()),
            "success_online": float(ok_onl.float().mean()),
            "success_sampled_depth": float(ok_smp.float().mean()),
            "success_output_sampling": float(ok_samp.float().mean()),
            "depth_policy": float(pol.float().mean() + 1),
            "depth_online": float(onl.float().mean() + 1),
            "depth_sampled": float(smp.float().mean() + 1),
            "p05_graph_success": float(per_graph.quantile(0.05)),
            "frac_graphs_never": float((per_graph == 0).float().mean()),
            "graph_success_hist": np.histogram(per_graph.numpy(), bins=np.linspace(0, 1, 11))[0].tolist(),
            "success_by_depth": ok_depth.float().mean((0, 2)).round(decimals=4).tolist(),
            "seed_diverse": float((ok_pol != ok_pol[:1]).any(0).float().mean()) if S > 1 else 0.0,
        }
        # Las Vegas amplification: k independent seeds + local verification, success if any is valid
        lv = {}
        for kk in (1, 2, 4, 8, 16):
            if kk <= S:
                lv[kk] = float(ok_pol[:kk].any(0).float().mean())
        row["best_of_k"] = lv
        # anytime / on-line verification: stop at the first round whose colouring is valid
        first_ok = torch.where(ok_depth.any(1), ok_depth.float().argmax(1) + 1, torch.full((S, a.eval_graphs), H))
        row["verify_stop_success"] = float(ok_depth.any(1).float().mean())
        row["verify_stop_depth"] = float(first_ok.float().mean())
        if halting:
            row["pi_mean"] = pis.mean((0, 2)).round(decimals=5).tolist()
        rows.append(row)
    return {"rows": rows}


def _policy_depth(a, n):
    return int(round(a.budget))


def main(argv=None):
    a = get_args(argv)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    torch.manual_seed(a.run_seed)
    model = LoopedTransformer(a.pool + 1, a.d_model, a.n_layers, a.n_heads, n_out=N_COLORS)
    init_weights(model, a.n_layers)
    rng = np.random.default_rng(10_000 + a.run_seed)
    py_rng = random.Random(20_000 + a.run_seed)
    gen = torch.Generator().manual_seed(30_000 + a.run_seed)
    torch.manual_seed(40_000 + a.run_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    baseline, dual = MovingBaseline(0.9), BudgetDual(a.budget if a.method == "rlhalt" else None, a.dual_lr, a.rho)
    m = a.m if a.seed_mode not in ("none", "fixed") else 1
    n_graphs = a.batch // m
    log = (out / "train_log.jsonl").open("w")
    t0 = time.time()
    model.train()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        n = int(rng.integers(a.n_min, a.n_max + 1))
        x = sample_graphs(n_graphs, n, a.pool, a.anonymous, rng).repeat_interleave(m, 0)
        mask = ring_mask(n)
        seeds = model.draw_seeds(a.seed_mode, x.shape[0], n, a.t_max, generator=gen)
        if a.method == "rlhalt":
            loss, info = rlhalt_loss(model, x, seeds, a.t_max, coloring_loss, valid_coloring, a.q, m,
                                     baseline, dual, 4, a.ent_coef, attn_mask=mask)
        else:
            k = int(round(a.budget))
            if a.method == "fixed":
                depths = torch.full((x.shape[0],), k, dtype=torch.long)
            else:
                depths = torch.tensor([py_rng.randint(k - a.window, k + a.window) for _ in range(x.shape[0])]).clamp(1, a.t_max)
            loss, info = scheduled_loss(model, x, depths, seeds, coloring_loss, valid_coloring, a.q, m,
                                        attn_mask=mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == a.steps - 1:
            log.write(json.dumps({"step": step, "n": n, "loss": float(loss.detach()), "time": round(time.time() - t0, 1), **info}) + "\n")
            log.flush()
    log.close()
    torch.save(model.state_dict(), out / "model.pt")
    res = evaluate(model, a)
    res["n_params"] = n_params(model)
    res["train_time_s"] = round(time.time() - t0, 1)
    (out / "eval.json").write_text(json.dumps(res))
    brief = {r["n"]: (round(r["success"], 3), round(r["p05_graph_success"], 3), round(r["depth_policy"], 2),
                      r["best_of_k"].get(16, r["best_of_k"].get(1))) for r in res["rows"]}
    print(json.dumps({"out": str(out), "eval": brief, "time": res["train_time_s"]}))


if __name__ == "__main__":
    main()
