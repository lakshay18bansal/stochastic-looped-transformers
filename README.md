# Learned Stochastic Computation in Looped Transformers

Code and results for our DA4022 course project at IIT Guwahati
(Lakshay Bansal, Mayukh Kundu, Prayansh Kumar, Krish Randar, Sakshi Gupta, Ritvij Gopal).

## About the project

We started from two papers:

1. Kuo et al., "Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping", NeurIPS 2026.
   A looped transformer runs the same block again and again, and it learns when to stop (RL-Halting).
2. von Oswald et al., "Learning Randomized Algorithms with Transformers", ICLR 2025.
   The transformer gets a random seed as input and learns to act like a randomized algorithm.

We put both ideas into one looped model. All versions of the model have the same number of parameters, and the
average number of loops is kept fixed at a budget B. We trained 238 models on Dyck-1, Copy and 3-colouring of
cycle graphs to see which kind of randomness actually helps when compute is fixed.

In short, the random seed helped at test time (on cycles where all vertices look the same, it took colouring from
0% to 84% in one try), while the learned stopping time only helped during training. Copy did not train well for
any method at our scale.

## Folders

```
stochloop/      model, training and evaluation code
  model.py        the looped transformer (seed input + halting head)
  objectives.py   losses, RL-Halting, the budget constraint
  tasks.py        Dyck-1 and Copy data
  evaluation.py   evaluation for Dyck-1 and Copy
  train_seq.py    one Dyck-1 or Copy run
  coloring.py     one graph colouring run
scripts/
  jobs_*.txt      the exact command for every run we did (238 lines in total)
  run_queue.py    runs a job file in parallel and skips runs that are already done
analysis/       makes the tables and figures from results/
results/        output of all 238 runs, and the tables in results/tables/
reproduce.py    one script for checking, rerunning and analysing everything
```

Every run has its own folder in `results/`, for example `results/seq/dyck1/rlhaltB3/s0/` or
`results/color/ids/seed_q8_fixed_B3/s0/`. Inside are `args.json` (settings), `train_log.jsonl` (training curve)
and `eval.json` (test results). We did not include the trained weights since nothing needs them.

## Setup

We used Ubuntu 24.04 with Python 3.12. Only a CPU is needed.

```bash
unzip stochastic-looped-transformers.zip
cd stochastic-looped-transformers
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If `python3 -m venv` complains about `ensurepip`, install it with `sudo apt install python3-venv` and try again.
The install takes a few minutes.

## Quick check

```bash
python reproduce.py check
```

This trains two tiny models and should print `smoke test passed` in under a minute.

## Tables and figures

```bash
python reproduce.py analysis
```

This takes less than a minute. It rewrites the tables in `results/tables/` and draws all figures into `figures/`.
All the numbers we used in the report are in `results/tables/report_numbers.md`.

## Running all experiments again

```bash
python reproduce.py all --fresh --workers 8
```

This moves our results to `results_shipped/`, trains all 238 models again into `results/`, makes the tables and
figures, and finally compares the new results with ours (`results/compare_report.txt`).

It takes around 45 to 60 CPU-hours in total. Each run uses one CPU core and about 0.5 GB of RAM, so set `--workers`
to at most the number of cores you have. On our 12-core laptop this works out to roughly:

| --workers | time |
|---|---|
| 4 | 11 hours |
| 8 | 6 hours |
| 12 | 5 hours |

If it stops for any reason, run the same command again. Finished runs are skipped and Dyck-1 / Copy runs continue
from their last checkpoint. To keep it running after closing the terminal:

```bash
nohup python reproduce.py all --fresh --workers 8 > reproduce.log 2>&1 &
tail -f reproduce.log
```

You can also run one group at a time (after the `--fresh` step has moved our results away):

```bash
python scripts/run_queue.py scripts/jobs_color.txt --workers 8       # 147 colouring runs, ~17 CPU-hours
python scripts/run_queue.py scripts/jobs_dual.txt --workers 4        # 4 runs, ~1 CPU-hour
python scripts/run_queue.py scripts/jobs_seq.txt --workers 8         # 60 Dyck-1 / Copy runs, ~20-30 CPU-hours
python scripts/run_queue.py scripts/jobs_seq_extra.txt --workers 8   # 27 extra seeds, ~8-10 CPU-hours
python reproduce.py analysis
```

Or a single run, by copying any line from a job file:

```bash
python -m stochloop.train_seq --threads 1 --task dyck1 --method rlhalt --budget 3 --steps 2000 --run_seed 0 --out runs/dyck1_B3
python -m stochloop.coloring --threads 1 --steps 3000 --seed_mode nodeloop --m 4 --q 8 --method fixed --budget 3 --run_seed 0 --out runs/color_B3
```

A Dyck-1 or Copy run takes 15 to 25 minutes and a colouring run 2 to 11 minutes.


## Which script makes what

- `analysis/report_numbers.py`: `results/tables/report_numbers.md`, all numbers used in the report
- `analysis/seq_analysis.py`: Dyck-1 and Copy tables (`seq_dyck1.md`, `seq_copy.md`) and `figures/seq_*`
- `analysis/color_analysis.py`: colouring tables (`color_main.md`, `color_ablations.json`) and `figures/color_*`
- `analysis/extra_analysis.py`: `jensen.md`, `extra.json`, `figures/dual_trajectory`, `figures/breadth_depth_theory`
- `analysis/report_figures.py`: the two figures used in the report
- `analysis/diagram.py`: the model diagram

