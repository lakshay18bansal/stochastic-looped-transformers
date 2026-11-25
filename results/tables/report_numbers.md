# Numbers used in the report

Recomputed from results/ by analysis/report_numbers.py.

## Table 1: Dyck-1, accuracy (%) on lengths 12-32 (best number of loops)

| method | runs | avg. loops (train) | loops (test) | mean +- std | worst run | policy acc. |
|---|---|---|---|---|---|---|
| K=12 | 5 | 12.0 | 12.0 | 77.4 +- 26.6 | 31.9 | 67.3 |
| K=6 | 5 | 6.0 | 6.0 | 87.0 +- 22.3 | 42.9 | 83.7 |
| K=n | 5 | 5.5 | 20.9 | 100.0 +- 0.0 | 100.0 | 94.8 |
| K=n+-3 | 5 | 5.7 | 20.9 | 65.9 +- 31.6 | 16.2 | 65.4 |
| RL-Halting, no budget | 5 | 6.0 | 2.6 | 79.7 +- 24.8 | 48.7 | 78.3 |
| budgeted RL-Halting, B=5.5 | 5 | 5.5 | 1.0 | 76.2 +- 21.6 | 47.4 | 76.1 |
| budgeted RL-Halting, B=3 | 3 | 3.0 | 1.0 | 99.8 +- 0.2 | 99.6 | 99.8 |
| budgeted RL-Halting, B=8 | 3 | 8.0 | 34.0 | 82.3 +- 25.0 | 47.0 | 51.3 |
| budgeted RL-Halting + seed, q=4 | 5 | 5.5 | 1.0 | 87.9 +- 9.8 | 76.1 | 75.1 |
| budgeted RL-Halting + seed, q=1 | 5 | 5.5 | 1.0 | 69.7 +- 9.1 | 60.2 | 60.4 |

(loops at test for K=n and K=n+-3 equal the test length n, 20.9 on average over lengths 12-32)

## Dyck-1 numbers in the text

- share of predictions that change from one loop to the next (after n loops, lengths 12-32, mean per method): rlhalt 0.16%, rlhaltB 0.07%, rlhaltB3 0.05%, rlhaltB8 1.73%, seed_rlhaltB_q4 0.03%, seed_rlhaltB_q1 0.00%
- B=3: stopping probability falls by a factor 0.663 per loop (constant hazard with mean 3, cut at 12 loops: 0.669)
- spread between runs: budgeted RL-Halting 21.6 -> with seed (q=4) 9.8 points; worst run 47.4% -> 76.1%
- seed with q=1: mean 69.7%
- K=6 runs: 100.0, 42.9, 99.8, 100.0, 92.6
- B=3 runs: 100.0, 99.9, 99.6; RL-Halting (no budget) runs: 100.0, 100.0, 100.0, 50.0, 48.7
- Copy: best method average on lengths 12-32 14.7% (best single run 28.3%); seeded models on training lengths (policy accuracy, mean per method): 31.9, 3.6, 8.5%

## Table 2: colouring, 64 vertices

| model | IDs B=3 success | never solved | best of 8 | no IDs B=4 success | best of 8 |
|---|---|---|---|---|---|
| no seed, fixed rounds | 0.61 +- 0.43 | 39.3% | nan | 0.00 | nan |
| no seed, learned stopping | 0.80 +- 0.08 | 20.2% | nan | nan | nan |
| seed, q=1, fixed rounds | 0.51 +- 0.36 | 30.9% | 0.68 | 0.27 | 0.38 |
| seed, q=8, fixed rounds | 0.74 +- 0.05 | 0.0% | 1.00 | 0.84 | 1.00 |
| seed, q=8, learned stopping | 0.54 +- 0.03 | 0.9% | 0.93 | 0.63 | 0.83 |

## Colouring numbers in the text

- learned stopping vs. input-independent random stop: 22 settings, max difference 1.7 points
- stopping at round(mean) is better by up to 23.3 points (21 of 22 settings)
- stop when valid, B=3 model: success 100.0% with 3.21 rounds on average; fixed 3 rounds: 73.9%
- best rounds per seed t* = 3; 2 seeds x 3 rounds: 93.8%, 1 seed x 6 rounds: 81.2%, 4 seeds x 3 rounds: 99.1%; max prediction error 3.6 points
- width 16: seeded 42.3% vs. no seed 84.6%; seeded 5th-percentile graph 0.29
- width 64: seeded 71.9% vs. no seed 87.0%; seeded 5th-percentile graph 0.54
- width 32 (main runs): seeded 5th-percentile graph 0.55
- q sweep (B=3, 64 vertices): q=1: 50.8%, q=2: 78.7%, q=4: 76.7%, q=8: 73.9%, q=16: 72.7%, q=32: 66.5%
- plain_dual: training depth 1.4-9.3 (2nd half of training), test depth 5.66, 4.15
- aug_lagrangian: training depth 3.8-4.3 (2nd half of training), test depth 4.06, 4.13

## Compute

- 238 runs, about 61 CPU-hours

