# First MLP versus MCB:16 — 2026-11-21

The validation-selected MLP improved from **0.080% to 0.485% wins** against
MCB:16. It remains much weaker than its teacher. Twenty minutes of repeated
imitation training substantially overfit this corpus; the best validation
checkpoint came from epoch 3, about 4.13 seconds into training.

## Primary comparison

Each checkpoint played the same 20,000 games: 10,000 randomly selected unique
nonnegative 63-bit deck seeds, each played in both seats. MCB RNG seeds were
fixed before evaluation. No evaluation deck appeared in the teacher corpus.

| MLP checkpoint | Wins | Losses | Draws | Win rate |
|---|---:|---:|---:|---:|
| Fresh random initialization (`before.pt`) | 16 | 19,984 | 0 | 0.080% |
| Minimum validation loss (`after.pt`) | 97 | 19,903 | 0 | 0.485% |

The paired gain is **0.405 percentage points**, with a **95% deck-grouped
percentile bootstrap interval of 0.305–0.510 points** (10,000 resamples).
The win-rate intervals are 0.045–0.120% before and 0.390–0.585% after.
There were 97 loss-to-win changes and 16 win-to-loss changes. The relative
win-rate increase is about sixfold, but MCB still wins 99.515% of games.

| MLP seat | Fresh wins / 10,000 | Selected wins / 10,000 |
|---|---:|---:|
| First player | 9 | 44 |
| Second player | 7 | 53 |

Intervals resample whole deck pairs, preserving dependence between seats and
before/after games. They describe matchup sampling uncertainty for these fixed
policies, not variability across initialization or training runs. Only one
initialization and one training seed were tested.

## Training

The architecture remained **1082 → 256 → 256 → 487**, with ReLU hidden layers,
468,199 parameters, legal-action masking, and deterministic argmax play.
Claims used the existing deterministic proof routine.

MCB:16 played 1,500 recorded games against each of Builder, Tactician, Planner,
and another MCB:16: 6,000 games on deck seeds 10000–15999. Authoritative replay
yielded 155,197 teacher decisions: 124,400 training and 30,797 validation
examples. Validation held out complete deck groups. Hidden opponent hands and
future draws never entered the network.

Training used AdamW, learning rate 0.001, batch size 512, seed 1, and Apple MPS
in the project's `spor` Python environment. Two resumable 600-second segments
provided **1,200.24 seconds** of optimization, validation, and checkpointing,
excluding dataset loading and fingerprinting. The final checkpoint completed
947 epochs plus a partial epoch, for 230,280 gradient updates.

| Checkpoint | Validation loss | Teacher-action validation accuracy |
|---|---:|---:|
| Selected: epoch 3, update 729 | 3.6731 | 5.55% |
| Final: update 230,280 | 21.5209 | 4.59% |

The final partial epoch's training accuracy was 94.97% and loss 0.1778, measured
during updating. The large training/validation gap indicates memorization.
`after.pt` is a byte-identical copy of `best.pt`, selected solely by validation
loss. `last.pt` retains the actual final weights and optimizer state. Evaluation
outcomes did not influence checkpoint selection.

A separate 1,000-game diagnostic of `last.pt` is recorded in
`final-diagnostic.json`; it uses the first 1,000 scheduled games and does not
replace the primary comparison. On this same subset, the fresh model won 1 game,
the validation-selected model won 6, and the final model won 4. There were no
draws. These few wins do not establish a reliable difference between selected
and final weights; the overfitting conclusion comes from held-out teacher loss.

Verification included 70 passing focused Python tests (one skipped), native
worker integration checks, and a final audit of frozen source, worker,
checkpoint and schedule hashes. The complete command history, including broader
test runs, remains under the artifact root's `logs/` directory.

## Runtime and artifacts

Summed evaluation slice wall times were 2,990.35 seconds before and 2,889.06
seconds after. Some slices overlapped each other or MPS training, so these are
neither total experiment elapsed time nor controlled performance benchmarks.
CPU policy inference totaled 35.09 seconds for 369,382 decisions before and
34.92 seconds for 378,400 after. These are batched, amortized timings, not
interactive turn latency.

The accompanying `results.json` contains the complete primary aggregate;
`manifest.json` records initialization, schedule, worker, source and checkpoint
hashes. Raw weights, schedule, recordings, per-game results, training metrics,
and individual command logs remain in the ignored local directory
`models/mlp-experiment-2026-11-21/`. Each shell command has a separate log with
argv, timestamps, output, status and wall time. Expected failing test/setup
attempts are retained alongside successful commands.

The manifest records the random schedule master seed
`318576952742040875856913753623906778349` and initialization seed `20261121`.
Use a fresh artifact directory when reproducing the experiment; existing logs
and evaluation slices cannot be overwritten. The implementation and invocation
examples are documented in [NEURAL.md](../../docs/NEURAL.md).

To recompute this run's primary summary from its saved slices:

```sh
python benchmarks/log_command.py --log RUN/logs/recompute-summary.log -- \
  python -m benchmarks.mlp_experiment report --root RUN
```

Replace `RUN` with the artifact directory. To play the selected checkpoint
against a running REST server:

```sh
python -m schotten.neural create \
  --checkpoint models/mlp-experiment-2026-11-21/after.pt --device cpu
```

The next experiment should test generalization improvements, such as suit
permutation augmentation and more diverse teacher positions, rather than simply
repeating more epochs on this fixed corpus. These results do not establish
which intervention will work best.
