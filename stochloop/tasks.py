"""Synthetic length-generalisation tasks (binary addition, Dyck-1, copy).

All tasks use a *slot-filling* (non-autoregressive) format: the answer slots are
part of the input sequence (filled with a PAD token) and the looped model predicts
every answer token in parallel from its final hidden state.  An example is
counted as correct only if every supervised position is correct, including the
trailing end/pad position.

Formats follow the public reference implementation of the looped-stopping study
(Apache-2.0) so that numbers are comparable in spirit; generators are vectorised
and trimmed to the longest example in each batch to save CPU time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    x: torch.Tensor        # [B, T] input tokens
    y: torch.Tensor        # [B, T] target tokens
    mask: torch.Tensor     # [B, T] bool, supervised positions
    valid: torch.Tensor    # [B, T] bool, positions that belong to the example
    n: torch.Tensor        # [B] problem length


# --------------------------------------------------------------------------- #
# Binary addition:  a + b = s   (MSB first)
#   tokens: 0/1 digits, 2 '+', 3 PAD, 4 IGNORE(target only), 5 '='
# --------------------------------------------------------------------------- #
ADD_VOCAB = 6


def _addition(lengths: np.ndarray, rng: np.random.Generator) -> Batch:
    B = len(lengths)
    nmax = int(lengths.max())
    T = 3 * nmax + 3                       # 2n+2 input tokens, n+1 answer slots, 1 end slot
    x = np.full((B, T), 3, dtype=np.int64)
    y = np.full((B, T), 3, dtype=np.int64)
    mask = np.zeros((B, T), dtype=bool)
    valid = np.zeros((B, T), dtype=bool)
    for i, n in enumerate(lengths):
        a = rng.integers(0, 2, n)
        b = rng.integers(0, 2, n)
        x[i, :n] = a
        x[i, n] = 2
        x[i, n + 1:2 * n + 1] = b
        x[i, 2 * n + 1] = 5
        av = int("".join(map(str, a)), 2)
        bv = int("".join(map(str, b)), 2)
        s = np.array(list(np.binary_repr(av + bv, width=n + 1)), dtype=np.int64)
        y[i, :2 * n + 1] = 4
        y[i, 2 * n + 1:3 * n + 2] = s
        mask[i, 2 * n + 1:] = True         # answer + trailing PAD (acts as end token)
        valid[i, :3 * n + 3] = True
    return Batch(*(torch.from_numpy(v) for v in (x, y, mask, valid)), torch.from_numpy(lengths.astype(np.int64)))


# --------------------------------------------------------------------------- #
# Dyck-1 completion: given an unbalanced prefix, emit the closing suffix + EOS
#   tokens: 0 '(' 1 ')' 2 SEP 3 PAD 4 EOS 5 IGNORE
# --------------------------------------------------------------------------- #
DYCK_VOCAB = 6


def _dyck_prefix(n: int, rng: np.random.Generator):
    """Uniform random walk prefix of length n that stays >=0 and ends >0."""
    for _ in range(1000):
        depth, toks = 0, []
        ok = True
        for pos in range(n):
            left = n - pos - 1
            moves = []
            moves.append(0)                      # open is always feasible
            if depth > 0 and not (left == 0 and depth - 1 == 0):
                moves.append(1)
            m = moves[rng.integers(0, len(moves))]
            depth += 1 if m == 0 else -1
            toks.append(m)
        if ok and depth > 0:
            return toks, depth
    raise RuntimeError("dyck prefix generation failed")


def _dyck(lengths: np.ndarray, rng: np.random.Generator) -> Batch:
    B = len(lengths)
    nmax = int(lengths.max())
    T = 2 * nmax + 2
    x = np.full((B, T), 3, dtype=np.int64)
    y = np.full((B, T), 3, dtype=np.int64)
    mask = np.zeros((B, T), dtype=bool)
    valid = np.zeros((B, T), dtype=bool)
    for i, n in enumerate(lengths):
        toks, depth = _dyck_prefix(int(n), rng)
        x[i, :n] = toks
        x[i, n] = 2
        y[i, :n + 1] = 5
        ans = [1] * depth + [4]
        y[i, n + 1:n + 1 + len(ans)] = ans
        mask[i, n + 1:n + 1 + len(ans)] = True
        valid[i, :2 * n + 2] = True
    return Batch(*(torch.from_numpy(v) for v in (x, y, mask, valid)), torch.from_numpy(lengths.astype(np.int64)))


# --------------------------------------------------------------------------- #
# Copy: x_1..x_n SEP -> x_1..x_n PAD
#   tokens: 0/1 digits, 2 SEP, 3 PAD, 4 IGNORE
# --------------------------------------------------------------------------- #
COPY_VOCAB = 5


def _copy(lengths: np.ndarray, rng: np.random.Generator) -> Batch:
    B = len(lengths)
    nmax = int(lengths.max())
    T = 2 * nmax + 2
    x = np.full((B, T), 3, dtype=np.int64)
    y = np.full((B, T), 3, dtype=np.int64)
    mask = np.zeros((B, T), dtype=bool)
    valid = np.zeros((B, T), dtype=bool)
    for i, n in enumerate(lengths):
        d = rng.integers(0, 2, n)
        x[i, :n] = d
        x[i, n] = 2
        y[i, :n] = 4
        y[i, n:2 * n] = d
        mask[i, n:] = True
        valid[i, :2 * n + 2] = True
    return Batch(*(torch.from_numpy(v) for v in (x, y, mask, valid)), torch.from_numpy(lengths.astype(np.int64)))


TASKS = {
    "addition": (_addition, ADD_VOCAB),
    "dyck1": (_dyck, DYCK_VOCAB),
    "copy": (_copy, COPY_VOCAB),
}


def vocab_size(task: str) -> int:
    return TASKS[task][1]


def sample_batch(task: str, batch_size: int, min_len: int, max_len: int,
                 rng: np.random.Generator) -> Batch:
    lengths = rng.integers(min_len, max_len + 1, size=batch_size)
    return TASKS[task][0](lengths, rng)


def fixed_length_batch(task: str, batch_size: int, n: int, rng: np.random.Generator) -> Batch:
    return TASKS[task][0](np.full(batch_size, n, dtype=np.int64), rng)
