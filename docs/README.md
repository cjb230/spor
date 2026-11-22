# Documentation

Current usage and behavior are documented in:

- [Project README](../README.md): Python setup, terminal controls, bots and API overview.
- [Browser guide](WEB_UI.md): choose among eight opponents, highlight recent cards,
  browse recordings and control replays.
- [Neural player guide](NEURAL.md): MLP structure, training from teacher games,
  saved weights, resume, and Python REST play.
- [Large MLP experiments](STREAMING_EXPERIMENTS.md): compact dataset generation,
  disk-backed training, progress, and resumable 40/125-million-example commands.
- [Generation performance](GENERATION_PERFORMANCE.md): CPU scheduling, cached
  future games, compressed storage estimates and deferred optimization ideas.
- [OpenAPI contract](../openapi.yaml): game endpoints, schemas and errors shared by both servers.
- [Native README](../rust/README.md): Rust builds, server, strategies, seeded series,
  tournaments, recordings and tests.
- [Native benchmarks](../rust/BENCHMARKS.md): saved M4 Pro measurements and workload limits.
- [Monte Carlo experiments](../benchmarks/mc-comparison-2026-11-19/RESULTS.md):
  MCB/MCP sampling comparisons and their [protocol](../benchmarks/mc-comparison-2026-11-19/PROTOCOL.md).
- [Deterministic bot checks](../rust/DETERMINISTIC_BOTS.md): historical matchups and regression coverage.

## Historical design and implementation records

The dated files under `superpowers/` record the original requirements, plans and
verification notes. They describe development at that time; use the guides above
for current options and behavior. Later additions include Monte Carlo strategies,
random tournament seeds and head-to-head standings.

| Feature | Design | Implementation plan |
|---|---|---|
| Terminal client | [Design](superpowers/specs/2026-11-16-terminal-design.md) | [Plan](superpowers/plans/2026-11-16-terminal.md) |
| Native engine and server | [Design](superpowers/specs/2026-11-16-rust-design.md) | [Plan](superpowers/plans/2026-11-16-rust.md) |
| Deterministic bots | [Design](superpowers/specs/2026-11-16-deterministic-bots.md) | [Plan](superpowers/plans/2026-11-16-deterministic-bots.md) |
| Tournaments | [Design](superpowers/specs/2026-11-17-tournament.md) | [Plan](superpowers/plans/2026-11-17-tournament.md) |
| Browser game and recordings | [Design](superpowers/specs/2026-11-19-web-game-design.md) | [Plan](superpowers/plans/2026-11-19-web-game.md) |
| Neural MLP player | [Design](superpowers/specs/2026-11-19-mlp-design.md) | [Plan](superpowers/plans/2026-11-19-mlp.md) |
