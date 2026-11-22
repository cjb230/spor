# Faster deterministic example generation

User approved implementing the first two generation optimizations: replace the
fixed eight-game barrier with a continuously replenished bounded queue, and reuse
games already computed ahead across successive native requests. Preserve the
existing game seeds, teacher decisions, example order, count truncation and
resume cursor semantics. Record remaining optimizations for future work.

## Design

Keep a persistent native scheduler. Game jobs use independently seeded bots and
may complete in any order; a bounded reorder buffer emits games in counter
order. Keep enough jobs ahead to occupy all worker threads, replenishing as
ordered games are consumed. Cache unused completed games across requests with
the same generation configuration and expected cursor. Reset/cancel queued work
when seed, split, strategies, trace mode or cursor changes. Cancellation must
not hide errors in games whose examples are actually consumed.

Use automatic core count by default, retain explicit --threads for comparisons.
Benchmark 8, 10 and 14 threads on this M4 Pro (10 P + 4 E cores), with identical
request sequences. Do not launch the large training experiments. Native binary
hash changes intentionally make pre-upgrade generation resume incompatible.

## Tasks

- [x] Native queue/cache: bounded memory, deterministic ordering, cancellation,
  error propagation and tests with uneven mock jobs.
- [x] Cross-request regression against the previous worker and thread counts.
- [x] Automatic thread default, CLI/docs and regression tests.
- [x] Fair native throughput measurements at 8/10/14 threads.
- [x] Compression measurement and storage estimate; future optimization notes.
- [x] Full relevant checks and independent review.

## Measurements

Sequential isolated 4,096-row comparisons: old eight-thread worker 102.7/s;
new worker at 8/10/14 threads 154.9/174.6/178.5 per second. Row hashes,
request cursors and compressed size match across every setting. Initial
measurements overlapped a background debug test suite and are excluded; that
suite was explicitly terminated before all four accepted runs.

Compression: 94,997 bytes for 4,096 examples, projecting to 2.90 GB for 125M.
The guide recommends 5–10 GB free space for the corpus and working files.

## Verification

- Full Python suite: 390 passed, 124 optional tests skipped.
- Native generator: all 10 focused tests passed in release mode; Clippy with
  warnings denied passed.
- Real CLI default-thread smoke: 14 threads selected automatically, exactly
  256 training and 64 validation rows committed successfully.
- Golden pre-queue request sequences matched at 1, 3 and 14 threads. Sampled
  MCB:16 benchmarks matched the old worker's row hash, cursors and compressed
  size at 8, 10 and 14 threads. Accepted runs are recorded under
  `benchmarks/generation-queue-2026-11-22/results.json`.
- Independent read-only review found no important bugs. Its runtime and sampled
  MCB uncertainties were closed by the real benchmarks and shutdown checks.
- Changes remain uncommitted. No large generation or training job was started.

## Future optimizations (not part of this change)

1. Specialized Builder rollout move choice: avoid constructing/sorting full
   candidate lists, copying unused candidate boards and allocating fresh scratch
   caches for each simulated turn; preserve scoring and deterministic ties.
2. A Monte Carlo move-selection-only API: avoid generating sorted per-move
   estimates when callers only need the selected move and claims.
3. Exact score bounds to stop evaluating candidates that cannot win even under
   their best possible remaining sample outcomes; preserve all tie-break rules.

Measure hot spots before implementing these. Reducing sample count, changing
rollout policies or opponent mix changes the teacher/data distribution and is
a separate experiment.
