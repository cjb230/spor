# Rust engine, REST server, and bots

> Historical design/development record. Later changes may supersede the original
> scope, limits and verification counts. See the [current documentation](../../README.md)
> for supported commands and behavior.

## Intent

The user requested maximum practical performance on an M4 Pro MacBook Pro,
authorized porting greedy/random bots, and requires REST for future Python deep
learning clients. Implement in Rust alongside the Python reference, preserving
the existing terminal and REST clients and game semantics.

## Approach

A shared native engine powers an Axum REST server and a native parallel series
runner. A server-only port leaves Python and per-turn HTTP overhead in self-play;
a native-only simulator would not support Python training clients. C was considered;
Rust provides suitable native data structures and HTTP/JSON/concurrency libraries
while reducing manual memory-management work. Retain Python as the reference.

## Requirements

- Preserve all documented REST operations, response models, bearer authentication,
  status/error codes, atomic rejection, idempotent turn UUIDs, hidden-information
  views, lazy timeouts, long-polling, SSE, legal moves and post-game reveal.
- Preserve rules, greedy scoring and random uniform selection. Native bots reason
  only from player-visible state. Alternate starting strategies in series, report
  win/draw totals, wall time, throughput; support quiet output and explicit seeds.
- Preserve Python integer-seeded deck shuffle for reproducible differential tests.
- Keep game state bounded and compact, use 54-card bitmasks, precompute three-card
  formation strengths and avoid repeated allocation in the rules hot path.
- Share lookup tables across games; parallelize independent native games with a
  configurable worker count. REST serializes each game, permitting different games
  to progress independently. Do not hold a game lock across network waits.
- Build optimized ARM64 release binaries with native CPU targeting, LTO, and one
  codegen unit. Benchmark, without claiming an unmeasured best configuration.
- No GPU/Metal work: this workload is small integer operations. Neural inference
  stays in Python and outside this port's scope.
- State remains in memory; restart loses games. Native series does not retain full
  server history by default. Preserve the Python implementation and its tests.

## Verification

Rust unit/integration tests, exhaustive formation comparison, seeded differential
full-game replay against Python, existing REST behavioral tests against the Rust
server, Python bot compatibility, native series determinism across worker counts,
negative request validation, long-poll/SSE wakeups, timeout and replay behavior.
Compare native engine-only Python/Rust workloads separately from real HTTP series.
Record hardware, build flags, workload, seeds and elapsed times with results.

## Scope decisions

Use existing OpenAPI YAML as the contract, include it with the server; persistence,
training algorithms, new batch APIs, GPU kernels and neural-network bots are not
part of this task. Keep files in the current workspace: all existing source files
are untracked, so a separate worktree would omit the reference implementation.
