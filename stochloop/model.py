"""Looped transformer with random-seed encoding and a hazard-style halting head.

    H(0) = 0
    H(t) = LN_f( Block_theta( H(t-1) + E(x) + W_r r_t ) ),     t = 1..K
    logits(t) = Readout(H(t))
    lambda_t  = sigmoid( q_phi( Pool(H(t)) ) )                   (stopping hazard)

* E(x) is re-injected at every iteration (input injection).
* r_t is a random seed encoding (RSE).  ``seed_mode`` decides how it is drawn:
    - "none"    : r_t = 0                       (deterministic model)
    - "fixed"   : r_t = r_0, one constant vector (deterministic, A_{r0} baseline)
    - "example" : one r per example, shared by all tokens and all loops
    - "loop"    : fresh r per example *and* per loop iteration
    - "node"    : one r per token (used for graph tasks: per-vertex randomness)
    - "nodeloop": fresh r per token and per loop iteration
* Every model owns W_r and the halting head, whether or not they are used, so
  all variants have exactly the same parameter count (fixed network size).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED_MODES = ("none", "fixed", "example", "loop", "node", "nodeloop")


class Block(nn.Module):
    """Pre-LN transformer layer.  ``attn_mask`` (bool, True = may attend) overrides causal."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        if attn_mask is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class LoopedTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                 seed_dim: int = 16, mlp_ratio: int = 4, halt_bias: float = 0.0,
                 n_out: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.seed_dim = seed_dim
        self.embed = nn.Embedding(vocab_size, d_model)
        self.seed_proj = nn.Linear(seed_dim, d_model, bias=False)
        self.layers = nn.ModuleList([Block(d_model, n_heads, mlp_ratio) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, n_out or vocab_size)
        self.halt_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, halt_bias)
        self.register_buffer("r0", torch.randn(seed_dim, generator=torch.Generator().manual_seed(1234)))

    # ------------------------------------------------------------------ seeds
    def draw_seeds(self, mode: str, batch: int, length: int, depth: int,
                   generator: torch.Generator | None = None) -> torch.Tensor | None:
        """Return seeds of shape [depth, B, L or 1, seed_dim] (or None)."""
        dev = self.r0.device
        if mode == "none":
            return None
        if mode == "fixed":
            return self.r0.view(1, 1, 1, -1).expand(depth, batch, 1, -1)
        shape = {
            "example": (1, batch, 1, self.seed_dim),
            "loop": (depth, batch, 1, self.seed_dim),
            "node": (1, batch, length, self.seed_dim),
            "nodeloop": (depth, batch, length, self.seed_dim),
        }[mode]
        r = torch.randn(shape, generator=generator, device=dev)
        return r.expand(depth, -1, -1, -1) if r.shape[0] == 1 else r

    # ---------------------------------------------------------------- forward
    @staticmethod
    def _pool(h: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if valid is None:
            return h.mean(1)
        w = valid.unsqueeze(-1).to(h.dtype)
        return (h * w).sum(1) / w.sum(1).clamp_min(1.0)

    def forward(self, x: torch.Tensor, depth: int, seeds: torch.Tensor | None = None,
                valid: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None,
                need_halt: bool = False, detach_halt: bool = True,
                node_halt: bool = False):
        """Unroll ``depth`` iterations.

        Returns (logits_list, hazard_list) where logits_list[t] is [B, T, V]
        and hazard_list[t] is [B] (or [B, T] if node_halt) or empty.
        """
        e = self.embed(x)
        h = torch.zeros_like(e)
        logits, hazards = [], []
        for t in range(depth):
            inj = e
            if seeds is not None:
                inj = e + self.seed_proj(seeds[t])
            z = h + inj
            for layer in self.layers:
                z = layer(z, attn_mask)
            h = self.ln_f(z)
            logits.append(self.readout(h))
            if need_halt:
                hh = h.detach() if detach_halt else h
                if node_halt:
                    hazards.append(torch.sigmoid(self.halt_head(hh).squeeze(-1)))
                else:
                    hazards.append(torch.sigmoid(self.halt_head(self._pool(hh, valid)).squeeze(-1)))
        return logits, hazards


def exit_distribution(hazards: list[torch.Tensor]) -> torch.Tensor:
    """Truncated hazard -> stopping distribution.  hazards: list of [B]; returns [T, B].

    pi(t) = lambda_t * prod_{i<t}(1 - lambda_i)  for t < T,  pi(T) = prod_{i<T}(1 - lambda_i).
    """
    lam = torch.stack(hazards, 0).clamp(1e-6, 1 - 1e-6)
    lam = torch.cat([lam[:-1], torch.ones_like(lam[-1:])], 0)
    log_surv = torch.cumsum(torch.log1p(-lam[:-1]), 0)
    log_surv = torch.cat([torch.zeros_like(lam[:1]), log_surv], 0)
    return torch.exp(log_surv) * lam


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def init_weights(model: nn.Module, n_layers: int) -> None:
    """GPT-2 style init; residual projections scaled by 1/sqrt(2 n_layers)."""
    for name, p in model.named_parameters():
        if "halt_head" in name:
            continue
        if p.dim() >= 2:
            std = 0.02
            if name.endswith(".proj.weight") or name.endswith("mlp.2.weight"):
                std = 0.02 / math.sqrt(2 * n_layers)
            nn.init.normal_(p, 0.0, std)
        elif name.endswith("bias") and "ln" not in name:
            nn.init.zeros_(p)
