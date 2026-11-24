# Learned Stochastic Computation in Looped Transformers

*When to stop, what to randomise, and how to spend a fixed compute budget.*

This project merges two lines of work into one model and studies it at a **fixed network size** and a
**fixed average compute** budget:

| | Paper 1 — *Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping* (NeurIPS 2026) | Paper 2 — *Learning Randomized Algorithms with Transformers* (ICLR 2025) |
|---|---|---|
| randomness | **when to stop**: the number of loop iterations τ | **what to compute**: a random seed r fed to the network |
| learned by | RL-Halting (REINFORCE on the sampled-depth loss) | the q-norm (relaxed min–max) objective over seeds |
| benefit | stable length extrapolation across runs | robustness on worst-case inputs, amplification by voting |
| compute | unconstrained | fixed depth |

The merged **stochastic looped transformer** re-injects a random seed encoding at every loop iteration and has a
hazard-based stopping head. It is trained with the q-norm objective over seeds, sampled-depth supervision, and an
*equality* constraint `E[τ] = B` on average compute. The constraint is enforced with an augmented-Lagrangian dual
(**budgeted RL-Halting**). Every variant has exactly the same parameter count.

The report is in [`report/main.pdf`](report/main.pdf).

## Repository layout

```
stochloop/            library
  model.py            looped transformer, seed encodings, stopping head, exit distribution
  objectives.py       depth schedules, RL-Halting, budgeted RL-Halting (augmented Lagrangian), q-norm
  tasks.py            Dyck-1, Copy, binary addition (slot-filling format)
  evaluation.py       oracle / policy / on-line / depth-vote / seed-vote metrics, flip rates
  train_seq.py        length-generalisation experiments
  coloring.py         looped distributed 3-colouring of cycles (train + evaluate)
scripts/
  jobs_*.txt          the exact command for every run reported
  run_queue.py        resumable parallel job runner
  run_all.sh          reproduce everything (experiments -> tables -> figures -> PDF)
analysis/             aggregation, tables and figures
results/              raw per-run outputs (eval.json, train_log.jsonl) and aggregated tables
figures/              all figures (PDF + PNG)
report/               LaTeX source and compiled report
```

## Quick start

```bash
pip install -r requirements.txt
# one run: budgeted, seeded RL-Halting on Dyck-1 (about 30 min on one CPU core)
python3 -m stochloop.train_seq --task dyck1 --method rlhalt --budget 5.5 \
        --seed_mode loop --m 4 --q 4 --steps 2000 --out runs/demo
# one colouring run: seeded q-norm model with 3 communication rounds (about 4 min)
python3 -m stochloop.coloring --seed_mode nodeloop --m 4 --q 8 --method fixed --budget 3 --out runs/color_demo
# everything
WORKERS=6 bash scripts/run_all.sh
```

All experiments run on a CPU, with no GPU needed. The full grid took about 8 hours on a 12-core laptop CPU.

## Main methods (all with the same ~100k / ~13k parameters)

| key | depth schedule | seed | objective |
|---|---|---|---|
| `fixed12`, `fixed6` | K fixed | – | ERM |
| `length` | K = n | – | ERM |
| `window` | K = n + U{-3..3} | – | ERM |
| `rlhalt` | learned τ ~ π_φ, no compute constraint (paper 1) | – | ERM |
| `rlhaltB` | learned τ with **E[τ] = 5.5** (= E[n], matched to `length`) | – | ERM |
| `seed_rlhaltB_q{1,4}` | learned τ with E[τ] = 5.5 | fresh per loop | q-norm over m = 4 seeds |
| colouring `det_*` / `seed_q*` | fixed / learned rounds, average B | none / per-vertex, per-round | q-norm |

## Acknowledgement of reused material

The task formats and the RL-Halting estimator follow the public reference implementation of paper 1
(Apache-2.0). The cycle-colouring task and the q-norm objective follow paper 2. All code here is a fresh
implementation.
