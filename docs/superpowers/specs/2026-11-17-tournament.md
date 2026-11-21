# Native tournaments

> Historical design/development record. Later changes may supersede the original
> scope, limits and verification counts. See the [current documentation](../../README.md)
> for supported commands and behavior.

User requirements: choose competitors, number of games, deck and agent seeds,
optional recorded games, a single updating x / y games played line and a final
table of total victories. Native Rust only; preserve existing series and REST.

Use a round robin over every unordered pair, `games_per_pair` explicit rather
than an ambiguous total. Alternate the first player within each pairing. Support
JSON config with CLI overrides, multiple named entrants using the same policy,
base deck/agent seeds plus individual agent seeds. Missing base seeds default to
1 for reproducibility; print resolved seeds. Validate at least two unique entrant
names, known strategies, positive game count, bounded checked total and recording
count before playing. Work in the current workspace, no commits or branches.

Schedule games deterministically, independent of worker count. Pair ordering is
entrant input order, game ordering within each pair is round index ascending.
For each round, deck seed = base deck seed + round index, shared across pairings.
An entrant's RNG seed = its configured seed + round index; if not individually
configured, configured seed = base agent seed + entrant index. Seeds apply before
seat swapping. Record exact effective seeds as decimal strings to preserve i128.

Recording: zero means no files; positive N records the first N scheduled games;
`all` is also accepted. Write replay JSON per selected game, including initial
hands, draw order, turn intents, agent/seat metadata, exact seeds and outcome.
Create a unique run directory below record_dir; do not overwrite previous runs.
Write resolved run manifest and results there. Validate recording settings before
games, fail with nonzero exit on I/O errors and never silently drop a recording.

Use rayon independent matches; do not retain every game's history or all outcomes
in memory when not recording. Only one thread writes progress at most ten times
per second, using carriage return and flush; always emit initial and final counts
and a newline before exiting. Worker completion drives counts, not scheduling.
Aggregate standings wins/losses/draws/played, sort by wins descending then entrant
input order. Include elapsed wall time and total draws/games. Print no game-by-game
logs. On errors, report failure and clear progress; never count a failed game as
a draw. Existing seeded series traces must remain identical.
