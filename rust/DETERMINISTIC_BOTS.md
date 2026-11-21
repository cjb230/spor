# Deterministic strategy checks

The native runner supports `builder`, `tactician` and `planner`, alongside
`random`, `greedy`, and the Monte Carlo strategies. The three deterministic
policies use fixed stone/card tie-breaking and no RNG-dependent decisions.
`schotten.bot` offers greedy/random, the browser offers eight native opponents,
and `schotten.neural` provides the Python [MLP player](../docs/NEURAL.md) over REST.
See [README.md](README.md#native-strategies) for native rules and commands.

## Initial matchups

These are historical measurements, not a fresh benchmark of the current checkout.
They exclude the Monte Carlo strategies; rerun the commands below to evaluate
the current code.

One 1,000-game series per matchup, deck and bot seed 1, alternating starting seats,
14 workers, native release build on the M4 Pro. No draws occurred.

| Matchup | First policy wins | Second policy wins |
|---|---:|---:|
| builder vs random | 1,000 | 0 |
| tactician vs random | 997 | 3 |
| planner vs random | 1,000 | 0 |
| builder vs greedy | 908 | 92 |
| tactician vs builder | 437 | 563 |
| planner vs tactician | 498 | 502 |

These samples do not establish a general ordering. More computation did not
consistently produce more wins: the measured planner lost this series against
tactician, and builder beat tactician. They provide distinct, inspectable training
baselines rather than a promised progression of playing strength.

Measured internal series throughput against random was about 55,000 games/s for
builder, 36,000 for tactician and 4,400 for planner. These are single short runs,
not repeated benchmark medians; process startup is excluded from the runner's
reported time. Future rule/weight changes should rerun these matchups.

Reproduce any row, replacing the two policy names:

```sh
./rust/target/release/schotten-native series --games 1000 --strategy builder --opponent-strategy greedy --seed 1 --bot-seed 1 --threads 14 --quiet
```

## Correctness checks

Tests cover known-hand combination building, exact formation ties, match-winning
claims, board-route priorities, card opportunity cost, excluding known cards from
opponent replies, conservative opponent claim proofs, passing, legal complete
games with/without automatic claiming, repeatability across seeds and worker
counts, and unchanged random/greedy reference traces.

Planner was also tested on a position where taking a secure claim exposes an
opponent's fifth stone. Avoiding that provable immediate defeat now outranks
taking the extra claim, provided the defensive move is in its fixed shortlist.
More generally, planner maximizes worst-case board value before claim count;
immediate match wins remain the first priority. A separate nonterminal regression
checks that an extra claim cannot override a better worst-case evaluation.
