"""Train / evaluate a looped transformer on a length-generalisation task.

Example
-------
python -m stochloop.train_seq --task addition --method rlhalt --budget 5.5 \
       --seed_mode loop --m 4 --q 4 --run_seed 0 --out results/seq/addition/srh_q4/s0
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

from .evaluation import evaluate, summarise
from .model import LoopedTransformer, init_weights, n_params
from .objectives import (BudgetDual, MovingBaseline, exact_match, per_example_ce, rlhalt_loss,
                         scheduled_depths, scheduled_loss)
from .tasks import sample_batch, vocab_size


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="addition", choices=["addition", "dyck1", "copy"])
    p.add_argument("--method", default="window", choices=["fixed", "length", "window", "rlhalt"])
    p.add_argument("--fixed_k", type=int, default=12)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--budget", type=float, default=None, help="E[tau] constraint for rlhalt")
    p.add_argument("--dual_lr", type=float, default=0.005)
    p.add_argument("--rho", type=float, default=1.0, help="augmented-Lagrangian weight (0 = plain dual ascent)")
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--seed_mode", default="none")
    p.add_argument("--m", type=int, default=1, help="random seeds per training example")
    p.add_argument("--q", type=float, default=1.0, help="q-norm over inputs (1 = ERM)")
    p.add_argument("--t_train", type=int, default=12)
    p.add_argument("--min_len", type=int, default=1)
    p.add_argument("--max_len", type=int, default=10)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=64, help="total sequences per step (examples x m)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--run_seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--eval_lengths", default="5,10,12,14,16,20,24,28,32")
    p.add_argument("--eval_examples", type=int, default=512)
    p.add_argument("--eval_seeds", type=int, default=8)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    prog = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * 0.5 * (1 + math.cos(math.pi * prog))


def main(argv=None):
    a = get_args(argv)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))

    torch.manual_seed(a.run_seed)
    model = LoopedTransformer(vocab_size(a.task), a.d_model, a.n_layers, a.n_heads)
    init_weights(model, a.n_layers)
    data_rng = np.random.default_rng(10_000 + a.run_seed)
    py_rng = random.Random(20_000 + a.run_seed)
    seed_gen = torch.Generator().manual_seed(30_000 + a.run_seed)
    torch.manual_seed(40_000 + a.run_seed)         # sampling of halting depths
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
    baseline, dual = MovingBaseline(0.9), BudgetDual(a.budget, a.dual_lr, a.rho)
    n_ex = a.batch // a.m

    log = (out / "train_log.jsonl").open("w")
    t0 = time.time()
    model.train()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        batch = sample_batch(a.task, n_ex, a.min_len, a.max_len, data_rng)
        if a.m > 1:  # replicate each example m times, each copy gets its own seed
            rep = lambda t: t.repeat_interleave(a.m, 0)
            batch.x, batch.y, batch.mask, batch.valid, batch.n = map(rep, (batch.x, batch.y, batch.mask, batch.valid, batch.n))
        seeds = model.draw_seeds(a.seed_mode, batch.x.shape[0], batch.x.shape[1], a.t_train, generator=seed_gen)
        loss_fn = lambda lg: per_example_ce(lg, batch.y, batch.mask)
        acc_fn = lambda lg: exact_match(lg, batch.y, batch.mask)
        if a.method == "rlhalt":
            loss, info = rlhalt_loss(model, batch.x, seeds, a.t_train, loss_fn, acc_fn, a.q, a.m,
                                     baseline, dual, a.n_samples, a.ent_coef, valid=batch.valid)
        else:
            depths = scheduled_depths(a.method, batch.n, a.t_train, a.fixed_k, a.window, py_rng)
            loss, info = scheduled_loss(model, batch.x, depths, seeds, loss_fn, acc_fn, a.q, a.m,
                                        valid=batch.valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == a.steps - 1:
            rec = {"step": step, "loss": float(loss.detach()), "gn": float(gn), "time": round(time.time() - t0, 1), **info}
            log.write(json.dumps(rec) + "\n")
            log.flush()
    log.close()
    torch.save(model.state_dict(), out / "model.pt")

    lengths = [int(v) for v in a.eval_lengths.split(",")]
    t_test = max(lengths) + 2
    res = evaluate(model, a.task, a.method, lengths, a.t_train, t_test, a.seed_mode, a.eval_seeds,
                   a.eval_examples, a.fixed_k, a.window)
    res["summary"] = summarise(res, a.max_len + 1)
    res["n_params"] = n_params(model)
    res["train_time_s"] = round(time.time() - t0, 1)
    (out / "eval.json").write_text(json.dumps(res))
    print(json.dumps({"out": str(out), **res["summary"], "time": res["train_time_s"]}))


if __name__ == "__main__":
    main()
