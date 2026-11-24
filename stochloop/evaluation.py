"""Evaluation of looped models on length generalisation.

For every evaluated length n and every random seed draw we unroll the model to a
common test horizon T_test and record, per loop depth t, the predicted sequence,
its exact-match correctness and (for halting models) the stopping distribution.

Metrics
-------
oracle         any depth t <= max(T_train, n+2) is correct (diagnostic, uses labels)
policy         the method's own stopping rule, no labels
                 fixed  : t = K            length / window : t = n
                 halting: t = argmax_t pi(t | x)  over T_test
online         halting models only: stop at the first t with CDF_pi(t) >= 1/2
               (computable on-line, so its cost is the realised depth)
depth_vote     mode of the predicted sequences across depths, weighted by pi(t|x)
               (halting) or uniformly over [policy-w, policy+w] (others).  This
               turns the randomness of the stopping time into a majority vote,
               at no extra cost beyond the unrolled horizon.
seed_vote      majority vote over S random seeds at the policy depth (seeded models)
flip_rate      Pr[ y_hat_{t+1} != y_hat_t ]
"""
from __future__ import annotations

import numpy as np
import torch

from .model import exit_distribution
from .objectives import exact_match
from .tasks import fixed_length_batch


def _masked_pred(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = logits.argmax(-1)
    return torch.where(mask, p, torch.full_like(p, -1))


def _weighted_mode(preds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """preds [K, N, T], weights [K, N] -> index k* [N] of the heaviest prediction class."""
    eq = (preds.unsqueeze(0) == preds.unsqueeze(1)).all(-1).float()   # [K, K, N]
    score = (eq * weights.unsqueeze(0)).sum(1)                        # [K, N]
    return score.argmax(0)


@torch.no_grad()
def evaluate(model, task: str, method: str, lengths: list[int], t_train: int, t_test: int,
             seed_mode: str = "none", n_seeds: int = 1, n_examples: int = 512,
             fixed_k: int = 12, window: int = 3, eval_seed: int = 42, batch_size: int = 256) -> dict:
    model.eval()
    halting = method == "rlhalt"
    seeded = seed_mode not in ("none", "fixed")
    S = n_seeds if seeded else 1
    rows = []
    for n in lengths:
        rng = np.random.default_rng(eval_seed + 7919 * n)
        gen = torch.Generator().manual_seed(eval_seed + n)
        H = t_test
        h_or = max(t_train, n + 2)
        agg = {k: 0.0 for k in ("oracle", "policy", "policy_depth", "online", "online_depth",
                                "depth_vote", "seed_vote", "seed_depth_vote", "seed_diverse",
                                "entropy")}
        acc_by_depth = np.zeros(H)
        flip_by_depth = np.zeros(H - 1)
        pi_mean = np.zeros(H)
        per_input_success = []            # success rate over seeds, per input (policy depth)
        done = 0
        while done < n_examples:
            b = min(batch_size, n_examples - done)
            batch = fixed_length_batch(task, b, n, rng)
            seed_preds, seed_correct, seed_pis = [], [], []
            for s in range(S):
                seeds = model.draw_seeds(seed_mode, b, batch.x.shape[1], H, generator=gen)
                logits, hazards = model(batch.x, H, seeds=seeds, valid=batch.valid, need_halt=halting)
                preds = torch.stack([_masked_pred(l, batch.mask) for l in logits])         # [H, b, T]
                corr = torch.stack([exact_match(l, batch.y, batch.mask) for l in logits])  # [H, b]
                seed_preds.append(preds)
                seed_correct.append(corr)
                seed_pis.append(exit_distribution(hazards) if halting else None)
            preds = torch.stack(seed_preds)           # [S, H, b, T]
            corr = torch.stack(seed_correct)          # [S, H, b]
            idx = torch.arange(b)

            # ---------------- per-depth curves and flips (averaged over seeds)
            acc_by_depth += corr.float().mean(0).sum(1).numpy()
            flips = (preds[:, 1:] != preds[:, :-1]).any(-1).float().mean(0).sum(1).numpy()
            flip_by_depth += flips
            agg["oracle"] += corr[:, :h_or].any(1).float().mean(0).sum().item()

            # ---------------- policy depth per (seed, example)
            if halting:
                pis = torch.stack(seed_pis)                          # [S, H, b]
                pi_mean += pis.mean(0).sum(1).numpy()
                agg["entropy"] += float(-(pis * pis.clamp_min(1e-12).log2()).sum(1).mean(0).sum())
                pol = pis.argmax(1)                                  # [S, b], 0-based
                cdf = pis.cumsum(1)
                onl = (cdf >= 0.5).float().argmax(1)
                w_depth = pis
            else:
                k = fixed_k if method == "fixed" else n
                k = min(k, H)
                pol = torch.full((S, b), k - 1, dtype=torch.long)
                onl = pol
                w_depth = torch.zeros(S, H, b)
                lo, hi = max(1, k - window), min(H, k + window)
                w_depth[:, lo - 1:hi] = 1.0
            pol_corr = torch.stack([corr[s, pol[s], idx] for s in range(S)])          # [S, b]
            onl_corr = torch.stack([corr[s, onl[s], idx] for s in range(S)])
            agg["policy"] += pol_corr.float().mean(0).sum().item()
            agg["policy_depth"] += (pol.float() + 1).mean(0).sum().item()
            agg["online"] += onl_corr.float().mean(0).sum().item()
            agg["online_depth"] += (onl.float() + 1).mean(0).sum().item()
            per_input_success.append(pol_corr.float().mean(0).numpy())

            # ---------------- depth vote (per seed, then averaged)
            dv = []
            for s in range(S):
                ks = _weighted_mode(preds[s], w_depth[s])
                dv.append(corr[s, ks, idx])
            agg["depth_vote"] += torch.stack(dv).float().mean(0).sum().item()

            # ---------------- seed vote at the policy depth, and seed x depth vote
            if S > 1:
                pol_pred = torch.stack([preds[s, pol[s], idx] for s in range(S)])      # [S, b, T]
                ks = _weighted_mode(pol_pred, torch.ones(S, b))
                agg["seed_vote"] += pol_corr[ks, idx].float().sum().item()
                agg["seed_diverse"] += (pol_pred != pol_pred[:1]).any(-1).any(0).float().sum().item()
                flat_pred = preds.reshape(S * H, b, -1)
                flat_w = w_depth.reshape(S * H, b)
                ks = _weighted_mode(flat_pred, flat_w)
                agg["seed_depth_vote"] += corr.reshape(S * H, b)[ks, idx].float().sum().item()
            else:
                agg["seed_vote"] += pol_corr[0].float().sum().item()
                agg["seed_depth_vote"] += torch.stack(dv)[0].float().sum().item()
            done += b
        row = {"length": n, "n_examples": done, "oracle_horizon": h_or}
        row.update({k: v / done for k, v in agg.items()})
        row["acc_by_depth"] = (acc_by_depth / done).round(4).tolist()
        row["flip_by_depth"] = (flip_by_depth / done).round(4).tolist()
        if halting:
            row["pi_mean"] = (pi_mean / done).round(5).tolist()
        succ = np.concatenate(per_input_success)
        row["success_hist"] = np.histogram(succ, bins=np.linspace(0, 1, 11))[0].tolist()
        row["p05_success"] = float(np.quantile(succ, 0.05))
        rows.append(row)
    return {"rows": rows}


def summarise(result: dict, ood_min: int) -> dict:
    ood = [r for r in result["rows"] if r["length"] >= ood_min]
    keys = ("oracle", "policy", "online", "depth_vote", "seed_vote", "seed_depth_vote",
            "policy_depth", "online_depth", "seed_diverse")
    out = {f"ood_{k}": float(np.mean([r[k] for r in ood])) for k in keys}
    idr = [r for r in result["rows"] if r["length"] < ood_min]
    out.update({f"id_{k}": float(np.mean([r[k] for r in idr])) for k in ("oracle", "policy")})
    front = [r["length"] for r in ood if r["oracle"] >= 0.9]
    out["frontier90"] = max(front) if front else None
    return out
