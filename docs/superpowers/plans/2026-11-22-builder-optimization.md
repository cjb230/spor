# Faster Builder for example generation

User requested optimizing Builder as far as practical for generation speed.
Preserve exact policy behavior, including immediate wins, claim priority,
formation scores, incomplete formations, first-completion ties and low
stone/card tie breaks. This applies to standalone Builder and MCB:16 rollouts.

## Approach

Add a specialized Builder selector instead of allocating/sorting Candidate
objects and allocating an Evaluator HashMap every turn. Reuse evaluations of
hand triples and stone-plus-hand pairs within one decision. Use compact claim
masks and compare only the best rank. Keep the old generic candidate evaluator
as an oracle in tests; Tactician and Planner retain their current behavior.
Consider skipping partial-formation scoring after finding a provable new claim,
but only when ranking equivalence is established. Prefer safe, bounded stack
storage; benchmark any further lookup-table or direct-rollout integration before
retaining it. No teacher sampling or game-rule changes.

## Tasks

- [x] Preserve current native worker and establish a repeatable Builder microbenchmark.
- [x] Write oracle/equivalence and allocation regression tests, observe failures.
- [x] Implement specialized Builder and shared completion work; measure alternatives.
- [x] Validate native game traces and sampled MCB:16 examples byte-for-byte.
- [x] Sequential before/after generation benchmarks at the same thread count.
- [x] Relevant release tests, full Python tests, Clippy and independent review.
- [x] Publish measurements and update generation-time guidance.

All shell commands get individual logs under
`models/streaming-dev-2026-11-21/logs/builder-*.log`. Long tasks use the bounded
logger timeout. Do not run the full Rust Monte Carlo suite in debug mode; use
release builds and run heavy tests separately from performance measurements.
Keep existing uncommitted streaming/queue work intact. No commit requested.

## Execution record

The specialized path passed 2,500 random-view and 80 full-game oracle comparisons,
including an allocation check. Coordinator added exact RNG/action comparisons
for Builder's direct bitmask rollout sampling and an independently checked proof
helper accepting precomputed strength. A 157,464-byte dense triple lookup was
checked over all distinct-card permutations and retained after alternating
generation measurements and a longer comparison showed improvement.

Final 32,768-row native comparison (14 threads, seed 2001, MCB:16, default
opponents): original worker 157.058 s / 208.64 examples/s; optimized worker
24.087 s / 1,360.42 examples/s. Both row hashes are
`0384d94e7135b35b11f05016f057b9f8fb9435ae4fe252cbdca52b02ab24bc1b`;
all 32 cursors and 750,059 compressed bytes match. Raw artifacts and preserved
workers/sources are under `models/builder-optimization-2026-11-22/`.
Independent read-only review found no important correctness issues.

The final optimized repeat after the old-worker run measured 24.946 s /
1,313.54 examples/s with the same bytes and cursors. Final projected generation
times are 8.2–8.5 hours for 40M and 25.5–26.4 hours for 125M. Full release Rust
suite: 125 passed. Full Python suite: 390 passed, 124 optional skips. Clippy
all-targets with warnings denied passed. Main CLI and worker rebuilt; no commit
or full-sized experiment requested or performed.
