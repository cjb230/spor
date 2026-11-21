# MLP versus MCB:16 experiment

Approved before/after protocol: freeze a fresh 1082–256–256–487 MLP, evaluate
20,000 games from 10,000 randomly chosen nonnegative 63-bit deck seeds in both
seat orientations, imitate MCB:16 for 1,200 seconds, and reevaluate the identical
schedule. Evaluation seeds and MCB RNG seeds are frozen before seeing results.

Training corpus: 1,500 MCB:16 games against each of Builder, Tactician, Planner,
and another MCB:16. Deck ranges 10000–15999 are disjoint across conditions and
excluded from evaluation. Native recordings are authoritatively replayed; only
MCB:16 decisions enter supervised training. Validation holds out deck groups.

Use batched JSON-lines Rust game/MCB workers and CPU PyTorch inference for
evaluation. Workers expose only player observations. Keep the REST player intact.
Train with MPS in two resumable 600-second segments, excluding data preparation;
checkpoint model/optimizer/shuffle position and select best validation loss without
consulting evaluation results. Record the last checkpoint too.

- [x] Implement and test native batched worker and Python evaluator.
- [x] Implement and test timed training with mid-epoch resume and best checkpoint.
- [x] Freeze manifest, schedule and initialized checkpoint hashes.
- [x] Generate and validate the separate teacher corpus.
- [x] Run a 100-game pilot, then the full baseline in bounded logged commands.
- [x] Train for two 600-second budgets and freeze the selected checkpoint.
- [x] Repeat evaluation with identical schedule and worker binary.
- [x] Report wins/draws/losses, seat breakdown, timings and paired deck-bootstrap CI.

Results: [dated report](../../../benchmarks/mlp-comparison-2026-11-21/RESULTS.md).
The primary comparison completed all 40,000 games. Selected-model wins rose
from 16 to 97 per 20,000 games; final training weights are retained separately.

Every shell command receives a separate log under
`models/mlp-experiment-2026-11-21/logs/`, including command argv, working directory,
timestamps, combined stdout/stderr, exit status and wall time. Files use exclusive
creation to avoid overwriting command history. Each command has a timeout.
Models, raw games and evaluation results remain under the ignored experiment root;
publish a concise result report and the reproduction commands in documentation.
