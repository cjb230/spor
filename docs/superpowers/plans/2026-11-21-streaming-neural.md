# Streaming neural experiments

Approved scope: implement, verify with small runs, and provide commands for
40 million training examples × 3 passes and 125 million × 1 pass. Do not launch
the large experiments. Existing REST play and checkpoint format stay compatible.

## Design and choices

Use the existing Rust engine and bots to generate pre-action teacher decisions.
Save compressed NumPy shards with packed observation/mask bits, two raw card
counts and uint16 action labels. This avoids loading full corpora into RAM and
avoids retaining millions of verbose game recordings. Validation consists of
separate games and does not count toward the requested training examples.
Generation tracks deterministic game cursors and commits shards atomically.

Training shuffles shard order and rows within each shard, decodes only batches,
and updates the existing MLP with masked cross entropy and AdamW. Each epoch
visits every training row exactly once. The epoch option is the total target,
including on resume. Persist optimizer, ordering identity, minibatch cursor,
and validation-selected best weights; always retain final weights too. No
augmentation or automatic early stopping in this controlled comparison.

Alternatives considered: the existing all-in-memory loader cannot fit the
requested scale; generating labels online prevents reuse for three exact
passes and complicates interruption recovery. Compact sharded data separates
the expensive teacher work from repeatable training and supports both runs.

Progress uses a background heartbeat, including during native generation,
shard I/O and validation. Report counts, throughput, elapsed time and ETA.
Concurrent writers are rejected; config and data hashes guard resume.

## Tasks

- [x] Packed shard storage, integrity checks and progress heartbeat; red/green tests.
- [x] Native teacher generation and Python orchestration; exact counts, split
  isolation, deterministic resume and encoding parity tests.
- [x] Streaming trainer; bounded memory, exact passes, resume equivalence tests.
- [x] CLI integration, documentation and executable experiment commands.
- [x] Native/Python small end-to-end runs, regression tests and independent review.

## Execution record

Implement in the shared checkout to keep the user's command paths and existing
cached toolchain/artifacts usable. Independent workers own generator and trainer
files; coordinator owns storage, CLI, integration and docs. Every substantive
shell command is separately logged under `/private/tmp/spor-stream-*.log`.
Large datasets and output weights will be ignored by Git. No commit requested.

## Verification, completed 2026-11-22

- Full Python suite: 385 passed, 124 optional tests skipped before the final
  resume-heartbeat addition. After that fix, all 81 streaming tests passed.
- Native generator: 2 tests passed; release build and Clippy with warnings
  denied passed, run from `rust/` to apply native CPU flags.
- Real MCB:16 CLI generation: exactly 400 training + 64 validation rows;
  CPU training consumed three passes (1,200 presentations, 21 updates) and MPS
  consumed one (400 presentations, 7 updates). Portable final checkpoints were
  inspected for expected epoch counts, dataset identity and finite weights.
- Resuming the completed three-pass run performed no extra optimization.
- Native optimized pilot: 2,048 training + 128 validation rows in 20.31 seconds.
- Independent review confirmed all important findings resolved: best/last
  checkpoint crash transactions, epoch targets behind partial progress, and
  heartbeat coverage while checking resume shard integrity.
- Exact 40/125-million commands and timing extrapolation are documented in
  `docs/STREAMING_EXPERIMENTS.md`. No large experiment was launched.

Command logs are retained under `models/streaming-dev-2026-11-21/logs/` after
copying the per-command `/private/tmp/spor-stream-*.log` history.
