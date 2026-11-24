"""Training objectives.

* Hand-designed depth schedules (fixed, length-matched, uniform window).
* RL-Halting: learned stochastic stopping trained with REINFORCE on the negative
  sampled-depth loss (moving-average baseline + entropy bonus).
* Budget-constrained RL-Halting (new): the same learner with an *equality*
  constraint on the expected number of loop iterations,
        min_theta,phi  E[ l(theta; x, y, tau) ]   s.t.   E_x E_{tau~pi_phi(.|x)}[tau] = B,
  solved with a Lagrange multiplier updated by dual ascent.  This makes learned
  stochastic stopping directly comparable to hand-designed schedules at a fixed
  average compute budget B.
* q-norm (relaxed min-max) aggregation over random seeds:
        L_q = ( 1/n sum_i ( 1/m sum_j l(x_i, r_j) )^q )^{1/q}
  q = 1 recovers ERM, q -> inf the oblivious-adversary (worst-input) loss.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from .model import exit_distribution


def per_example_ce(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean token CE over supervised positions.  logits [B,T,V] -> [B]."""
    ce = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
    m = mask.float()
    return (ce * m).sum(1) / m.sum(1).clamp_min(1.0)


def exact_match(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((logits.argmax(-1) == y) | ~mask).all(-1)


def qnorm(losses: torch.Tensor, q: float) -> torch.Tensor:
    """losses [n, m] (examples x seeds) -> scalar relaxed min-max loss."""
    per_input = losses.mean(1)
    if q == 1:
        return per_input.mean()
    # (mean x^q)^(1/q) computed in log space for stability
    lx = torch.log(per_input.clamp_min(1e-12))
    return torch.exp((torch.logsumexp(q * lx, 0) - math.log(per_input.numel())) / q)


def scheduled_depths(method: str, n: torch.Tensor, t_max: int, fixed_k: int = 12,
                     window: int = 3, rng: random.Random | None = None) -> torch.Tensor:
    if method == "fixed":
        return torch.full_like(n, fixed_k)
    if method == "length":
        return n.clamp(1, t_max)
    if method == "window":
        rng = rng or random
        d = torch.tensor([rng.randint(int(k) - window, int(k) + window) for k in n.tolist()], dtype=n.dtype)
        return d.clamp(1, t_max)
    raise ValueError(method)


class MovingBaseline:
    def __init__(self, decay: float = 0.9):
        self.decay, self.value = decay, None

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        m = r.mean().detach()
        self.value = m if self.value is None else self.decay * self.value + (1 - self.decay) * m
        return self.value


class BudgetDual:
    """Augmented-Lagrangian handling of the equality constraint  E[tau] = B.

    penalty(phi) = lam * (D - B) + rho/2 * (D - B)^2,   D = batch mean of E_pi[tau | x]
    lam <- lam + eta * (D - B)            (dual ascent)

    The quadratic term damps the oscillations that plain dual ascent exhibits.
    """

    def __init__(self, budget: float | None, lr: float = 0.01, rho: float = 1.0):
        self.budget, self.lr, self.rho, self.lam = budget, lr, rho, 0.0

    def penalty(self, mean_depth: torch.Tensor) -> torch.Tensor:
        g = mean_depth - self.budget
        return self.lam * g + 0.5 * self.rho * g * g

    def step(self, mean_depth: float) -> None:
        if self.budget is not None:
            self.lam += self.lr * (mean_depth - self.budget)


def scheduled_loss(model, x, depths, seeds, loss_fn, acc_fn, q, m, valid=None, attn_mask=None):
    """Supervise every example at its own (scheduled) depth; unroll to the batch max."""
    K = int(depths.max())
    logits, _ = model(x, K, seeds=seeds, valid=valid, attn_mask=attn_mask)
    stacked = torch.stack(logits)                                   # [K, B, ..., V]
    sel = stacked[depths - 1, torch.arange(len(depths))]
    per_ex = loss_fn(sel)
    loss = qnorm(per_ex.view(-1, m), q)
    with torch.no_grad():
        acc = acc_fn(sel).float().mean()
    return loss, {"task_loss": float(per_ex.detach().mean()), "depth": float(depths.float().mean()),
                  "unroll": K, "acc": float(acc)}


def rlhalt_loss(model, x, seeds, t_max, loss_fn, acc_fn, q, m, baseline: MovingBaseline,
                dual: BudgetDual, n_samples: int = 4, ent_coef: float = 0.01,
                valid=None, attn_mask=None):
    """Learned stochastic stopping (REINFORCE) with optional average-compute constraint."""
    logits, hazards = model(x, t_max, seeds=seeds, valid=valid, attn_mask=attn_mask, need_halt=True)
    pi = exit_distribution(hazards)                                  # [T, B]
    step_loss = torch.stack([loss_fn(l) for l in logits])           # [T, B]
    B = pi.shape[1]
    dist = torch.distributions.Categorical(probs=pi.T)
    tau = dist.sample((n_samples,))                                  # [S, B]
    logp = dist.log_prob(tau)
    chosen = step_loss.T.gather(1, tau.T).T                          # [S, B]
    reward = -chosen.detach()
    adv = reward - baseline(reward)
    policy_loss = -(adv * logp).mean()
    entropy = -(pi * pi.clamp_min(1e-12).log()).sum(0).mean()
    depth_grid = torch.arange(1, t_max + 1, dtype=pi.dtype).unsqueeze(1)
    exp_depth = (pi * depth_grid).sum(0)                             # [B], differentiable in phi
    task = qnorm(chosen.mean(0).view(-1, m), q)
    loss = task + policy_loss - ent_coef * entropy
    if dual.budget is not None:
        loss = loss + dual.penalty(exp_depth.mean())
    dual.step(float(exp_depth.detach().mean()))
    with torch.no_grad():
        stacked = torch.stack(logits)
        acc = acc_fn(stacked[tau[0], torch.arange(B)]).float().mean()
    return loss, {"task_loss": float(chosen.detach().mean()), "depth": float(exp_depth.detach().mean()),
                  "entropy": float(entropy.detach()), "lam": dual.lam, "unroll": t_max, "acc": float(acc)}
