# Builder optimization — 2026-11-22

The specialized Builder preserves the original policy while removing repeated
work from standalone decisions and the Builder rollouts used by MCB.

## Measurements

On the M4 Pro, both generation versions used 14 threads and the identical
32,768-example schedule (1,280 contributing games):

| Version | Seconds | Examples/s | Speedup |
|---|---:|---:|---:|
| Before Builder optimization | 157.06 | 208.6 | 1.00× |
| Optimized | 24.09 | 1,360.4 | 6.52× |
| Optimized, repeat after baseline | 24.95 | 1,313.5 | 6.30× |

All three produced row SHA-256
`0384d94e7135b35b11f05016f057b9f8fb9435ae4fe252cbdca52b02ab24bc1b`,
identical per-request cursors and 750,059 compressed bytes. The
[raw results and hashes](results.json) include the compact-table intermediate
variant too. Three alternating short comparisons favored the dense table by
about 6% in median throughput; the longer comparison showed about 3% wall-time
and 5% CPU-work improvement over that intermediate variant. These small
differences are descriptive measurements, not statistical significance claims.

The Builder-only benchmark repeated 1,153 legal observations 1,280 times
(1,475,840 decisions). Time fell from 3.765 seconds to 0.345 seconds:
**2,551 ns to 233 ns per decision**, approximately **10.9× faster**. Input and
chosen-move fingerprints match. These timings exclude corpus setup and rules
warmup.

At the final generation rates, **40M examples project to 8.2–8.5 hours**, and
**125M to 25.5–26.4 hours**, excluding training. These are extrapolations from
short runs; sustained throughput, opponent mix and system load can change them.
The subsequently completed full-size generation run is documented below.

## Completed 40M generation run

The subsequent full run completed **40,000,000 training examples plus 100,000
validation examples** using seed 1001, 14 threads and the worker hash from the
optimized benchmark. The user reported **34,500 seconds (9 hours 35 minutes)**;
the final generator progress timer records 34,472 seconds. Using the reported
elapsed time, throughput was **1,159 training examples/s**, or **1,162 total
examples/s** including validation. This sustained measurement supersedes the
earlier 8.2–8.5-hour extrapolation.

The manifest marks generation complete. Its 613 compressed shards occupy
**916,368,180 bytes**, about **0.916 GB / 0.853 GiB**, or **22.85 bytes per
example**. The training game counter reached 1,567,599; validation reached 3,939.
At the same rate and compression, 125M training examples plus 100K validation
examples would require approximately **30 hours** and **2.86 GB** of shards.

The [compact run summary](completed-40m.json) records configuration, manifest
hash, counts and sizes. The completion flag, file sizes and final progress log
were checked; this reporting step did not replay or rehash all 40.1M examples.
Dataset shards, weights and console logs remain outside Git.

## Implementation

- Select the best rank directly; avoid allocating and sorting Candidate objects
  with copied boards and avoid per-turn completion HashMaps.
- Share each hand-triple evaluation among its three possible opening cards;
  share each anchored pair evaluation among its two candidate cards.
- Process complete formations first. If any new claim is provable, partial
  formations cannot outrank it under the original ranking and need no scoring.
- Use compact ownership masks for win checks, preserving the original score,
  claim priority, completion ties and low-stone/low-card tie breaks.
- Reuse computed formation strengths in proofs. Cache completed opponents'
  strengths instead of repeatedly evaluating the same formation.
- In Builder rollouts, select epsilon-random legal moves directly in the old
  card-major/stone-minor order. Preserve every RNG draw, including `below(1)`
  when a random branch has a single legal move. Passes consume no RNG.
- Use a 157,464-byte direct card-triple table for Builder, initialized with
  the existing rules tables. All distinct-card permutations match the compact
  lookup; the compact table remains in use elsewhere.

The original generic Builder evaluator remains as a test oracle. Tactician,
Planner, game rules, teacher sample counts and observation encoding retain
their existing semantics. No unsafe production code or global mutable search
cache was introduced.

## Verification approach

Oracle comparisons cover 2,500 random disjoint views, 80 full game traces,
short hands, existing wins, exact ties, passes and claims. Allocation checks
confirm zero heap allocations after rules warmup. Separate rollout tests check
the chosen move and future RNG outputs at epsilon 0, 0.1 and 1, including passes
and forced moves. Proof helpers are checked against the independent Monte Carlo
claim engine. The dense table is checked against every distinct triple in all
six permutations.

The fixed generation benchmark uses seed 2001, MCB:16, 14 worker threads and
the default Builder/Tactician/Planner/MCB:16 opponent mix. Runs are sequential
with no concurrent test jobs. The previous executable is preserved locally in
`models/builder-optimization-2026-11-22/` alongside raw measurements and the
intermediate compact-table variant. It remains available for old-binary resume.

## Reproduction

From `rust/`, build the native executable and examples with the project's native
CPU flags:

```sh
cargo build --release --locked --bin schotten-native \
  --example neural_generate --example builder_cost
```

From the repository root, repeat the 32,768-example native generation benchmark:

```sh
python -m benchmarks.generation_queue \
  --worker rust/target/release/examples/neural_generate \
  --threads 14 --requests 32 --output models/builder-benchmark.json
```

The Builder-only benchmark repeats 1,153 seeded legal observations 1,280 times:

```sh
./rust/target/release/examples/builder_cost 1280
```

Use a fresh log for every invocation and run timing comparisons sequentially.
Checkpoints and datasets generated before this change remain valid; generation
resume still requires the original worker bytes because manifests guard the
worker hash.

## Final checks

The full release Rust suite passed **125 tests** across all targets. The full
Python suite passed **390 tests**, with 124 optional tests skipped. Clippy passed
for every target with warnings denied. Independent review found no important
correctness issues. The native CLI and generation worker were rebuilt with
the existing native CPU flags and release LTO.
