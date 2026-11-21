# MCB versus MCP experiments

Each matchup command has a 1,100-second subprocess timeout, below the user's
20-minute per-command limit. A 20-game pilot took 29.10 seconds; the first
100-game block took 136.28 seconds. No game recordings are requested.

Run sequentially on 14 workers. At eight samples, use these independent blocks:

| Block | Deck seeds | MCP starting RNG seed | MCB starting RNG seed |
|---|---|---:|---:|
| 1 | 1001–1100 | 42 | 43 |
| 2 | 2001–2100 | 123 | 456 |
| 3 | 3001–3100 | 10007 | 20011 |

Run each block in both entrant orders. Keep RNG seeds attached to the same bot
when reversing. The tournament already alternates starting seats; reversing on
the same deck sequence additionally gives each policy the opposite seat/deal for
each deck. Within a block, mirrored games are paired, not independent samples.

Then compare both policies at 16 samples on the same decks and agent seeds as
block 1 (1001–1100, MCP seed 42, MCB seed 43), again in both entrant orders.
This choice was amended before running the 16-sample condition to control deck
and RNG differences when comparing sample budgets.

The identical-position timing probe measured a median planner/builder time ratio
of 27.46 and an aggregate ratio of 31.70. As an approximate compute-matched check,
also compare MCB at 256 samples against MCP at 8 (32 times as many MCB samples),
using decks 1001–1050, MCP starting seed 42 and MCB starting seed 43, in both
entrant orders. These are approximate compute budgets, not enforced time limits;
the ratio varies by position. Fifty games per orientation bounds command duration.

The 20-game pilot overlaps the first eight-sample block and must not be pooled
as additional evidence. The user's previously reported 1000-game run is external
reference data and is not included in locally measured totals.

Measure computation separately with identical legal observations, world seeds,
eight samples, one decision worker and both rollout policies. Use six early/mid/
late positions from two native games, three repetitions per position, alternating
which rollout policy is measured first. No matches run during that timing probe.

Save command arguments, binary SHA-256, full stdout/stderr, elapsed wall time and
standings per matchup. Do not change the engine or bot policies during experiments.

Reproduction note: the saved timing probe measured Builder and Planner only,
producing 36 records (18 paired decisions). The current `mc_rollout_cost` example
also measures Greedy and produces 54 records; compare Builder/Planner rows when
reproducing the historical ratios.
