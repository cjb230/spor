# Tournament Implementation Plan

> Historical design/development record. Later changes may supersede the original
> scope, limits and verification counts. See the [current documentation](../../README.md)
> for supported commands and behavior.

**Goal:** Configurable native round robin with replay recording, live progress and standings.
**Architecture:** Shared native match helper; tournament config/scheduler/runner;
CLI/config merge and one-thread progress renderer; replay JSON and results.
**Spec:** docs/superpowers/specs/2026-11-17-tournament.md

- [x] Add failing configuration/schedule/progress/replay/CLI tests.
- [x] Extract shared match helper without changing old series seeded outcomes.
- [x] Implement round-robin execution, bounded aggregation and selected recordings.
- [x] Integrate JSON config/CLI overrides and single-line progress, final table.
- [x] Verify worker-independent results, replay actual recorded games, validate
  invalid config/paths and final progress counts, run full Rust/Python checks.
- [x] Build release, smoke-test CLI/config and terminal progress, independently
  review, document exact seed/count/recording behavior and example commands.

## Verification ledger

- Shared recorded/unrecorded matches have identical results and replay their
  recorded turn intents; old greedy/random 100-game result remains 95/5.
- CLI/config tests cover overrides, named entrants, negative individual seeds,
  invalid/unknown/duplicate fields, positional-array rejection, table fields and
  one progress line with a final newline.
- Tournament tests cover pair/seat/seed schedule, standings at different worker
  counts, draws, first-N/all/disabled recording, deterministic replay files,
  no overwrites, I/O failure, callback errors and huge-schedule cancellation.
- Python suite: 137 passed, 123 opt-in live-server cases skipped normally.
- Native CLI smoke: 300 games, five selected recordings, table sums 300 victories.
  JSON config smoke: 30 games, all 30 recorded and verified complete.
- Real terminal run: 50,000 games with five agents, monotonic progress updated
  on one line through completion, final standings sum to 50,000 victories.
- Independent review found duplicate JSON fields being silently overwritten by
  Value parsing. A failing regression reproduced recording/seed overwrites;
  typed deserialization now reads original text and rejects duplicates.
- No new dependencies, commits or branches. Existing REST and Python strategies
  are unchanged. Replay samples from manual checks are under /private/tmp only.
