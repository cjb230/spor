# Deterministic Bots Implementation Plan

> Historical design/development record. Later changes may supersede the original
> scope, limits and verification counts. See the [current documentation](../../README.md)
> for supported commands and behavior.

> **For agentic workers:** Execute inline with a bounded parallel strategy-module task.

**Goal:** Native builder, tactician and planner strategies with repeatable decisions.
**Architecture:** PlayerView-only evaluation module, existing Bot facade and series.
**Tech Stack:** Existing Rust crate, precomputed rules and rayon runner.
**Spec:** docs/superpowers/specs/2026-11-16-deterministic-bots.md

## Constraints and review focus

- Preserve old seeded random/greedy traces and REST contract.
- Never consult hidden engine state or infer an opponent's private hand.
- Fixed search limits and deterministic stone/card tie-breaks; all claims provable.
- Winning moves outrank heuristic scores, and planner's search cannot omit them.
- Bound planner work and validate actual series speed rather than assuming it.

## Tasks

- [x] Write failing rule fixtures and integration tests for deterministic decisions,
  winning claims, passing, complete games and worker-count independence.
- [x] Implement builder/tactician/planner in deterministic.rs, integrate Bot and CLI.
- [x] Run focused and full Rust tests, preserve existing Python and RNG fixtures.
- [x] Build native release, measure strategy matchups and search throughput.
- [x] Review independently, resolve important findings, update documentation.

Work in the current workspace; no commits or new branches.

## Verification ledger

- Watched the missing module/placeholder decisions fail before implementation.
- 46 Rust tests pass: 33 library, one CLI, five deterministic integration and seven
  server tests. 137 Python tests pass; 123 opt-in live-server tests skip normally.
- Rustfmt and Clippy with warnings denied pass.
- Old 100-game seeded greedy/random outcome remains 95/5; full oracle traces pass.
- All five strategies complete legal self-play with card conservation, both
  auto-claim modes. New deterministic policies ignore bot seeds/prior calls and
  produce identical series results at one and four workers.
- Fixed a reproduced planner terminal-loss priority bug: its known-losing secure
  claim now loses to an available shortlisted defense. Red-to-green regression
  planner_defends_against_a_fifth_stone_instead_of_taking_a_losing_claim.
- Tightened planner to strict worst-case evaluation before claim count, matching
  the approved design; a separate nonterminal red-to-green regression covers the
  distinction. Independent review found no further actionable issues.
- Six seeded 1,000-game series completed; recorded results and performance limits
  in rust/DETERMINISTIC_BOTS.md. No claim that more complexity always wins more.
