# Native Schotten Totten

See [BENCHMARKS.md](BENCHMARKS.md) for measured M4 Pro results and workload limits.

Rust implementation of the game engine, REST server, five fixed bot policies,
three Monte Carlo strategies, and parallel series/tournament runners.
The Python server remains the reference implementation. The Python bot CLI supports
`greedy` and `random`; a separate [neural player](../docs/NEURAL.md) supports `mlp`
over REST. The eight built-in Rust strategies are available in native tournaments;
those tournaments do not load Python neural checkpoints.

## Build for this Mac

With Rust 1.88 or newer installed (edition 2024), from the repository root:

```sh
cd rust
cargo build --release --locked
cd ..
```

Run Cargo from `rust/` so `.cargo/config.toml` selects
`-C target-cpu=native`. On the measured M4 Pro Mac the compiler reports `apple-m4`.
Release builds use optimization level 3, fat LTO and one codegen unit.
The resulting binary is specific to this machine's CPU features.

Install Rust normally for ongoing development. The built executable runs without
Cargo, Rust or Python installed. All commands below use paths relative to the
repository root unless a code block explicitly changes directories.

## REST server

From the repository root:

```sh
./rust/target/release/schotten-native serve --port 8000
```

Open **http://127.0.0.1:8000/** for the browser game: choose any of the eight native
opponents (default `mcb`)
or watch recorded tournaments with stepping, seeking and autoplay. Use
`--recordings-dir PATH` to choose the recording library (default
`tournament-recordings`, relative to the working directory). See the
[browser guide](../docs/WEB_UI.md) for setup and controls.
Cards can be dragged onto legal stones to play immediately. Completed browser
games save automatically under `browser-games/` in the recording library.

Stop the Python server first if it is already using that port. The server defaults
to `127.0.0.1`; `--host` and `--port` select the listener. `--workers N` controls
Tokio worker threads, with zero (default) using available logical CPUs.
Individual games are serialized while different games can progress independently.

Existing Python clients work with the same API:

```sh
python3 -m schotten.terminal create --url http://127.0.0.1:8000
python3 -m schotten.bot join GAME_ID --strategy greedy --url http://127.0.0.1:8000
python3 -m schotten.bot series --games 100 --seed 1 --bot-seed 1 --quiet --url http://127.0.0.1:8000
```

`GET /openapi.yaml` serves the shared API contract. No Python runtime is required
by the Rust server itself. The Rust server does not provide Python FastAPI's
`/docs` or `/openapi.json` routes. Authentication, legal-move enumeration, per-player
views, turn versions/UUIDs, long-polling, SSE and post-game logs retain their
existing semantics. Game state and history are in memory and are lost on restart;
the server does not automatically evict completed games.
Server deadlines use millisecond clock resolution.

## Native bot series

```sh
./rust/target/release/schotten-native series --games 100 --seed 1 --bot-seed 1 --quiet
./rust/target/release/schotten-native series --games 10000 --seed 1 --bot-seed 1 --threads 1 --quiet
```

Default strategies are greedy against random. Use `--strategy` and
`--opponent-strategy` to select a policy. Starting seats alternate. `--threads N`
controls independent game workers; zero uses the available CPU count. Explicit
seeds make results independent of thread count. Quiet mode omits individual games.
Both seeds are optional; omitted seeds use randomness. For game index `i` (starting
at zero), the deck seed is `seed + i` and entrant `e` uses
`bot_seed + 2*i + e`, assigned before swapping starting seats.
Output includes wall-clock elapsed seconds and games per second. With parallel
execution, individual game results are printed in game order after the run.

Native series calls the same engine directly, avoiding HTTP, JSON and Python.
Bots receive only their player-visible hand, table and unseen-card pool.
Simulation does not retain server event histories. Use the REST server when
connecting a Python learning agent; native series measures a different workload
from a Python client repeatedly sending HTTP requests.

## Tournaments

Run a native round robin with live progress and a final victory table:

```sh
./rust/target/release/schotten-native tournament --agents random greedy builder --games-per-pair 100 --deck-seed 1 --agent-seed 42
```

Every unordered pair plays `games_per_pair` games, alternating who starts. Three
entrants with 100 games per pair means 300 total games and 200 games per entrant.
Use an even per-pair count for exactly balanced starting seats. Omitting `--agents`
selects the five fixed native policies (no Monte Carlo agents). No server is needed.
Defaults are 100 games per pair (1,000 games for the five default entrants),
deck and agent seed bases of 1, and `threads=0` to use available CPUs.

Progress is written to stderr on one carriage-return-updated line, such as
`123 / 300 games played`. It counts completed games, not queued work. No per-game
move logs are printed. The final stdout table is sorted by total victories,
with played/lost/drawn counts and win percentages; elapsed wall time is included.
Below it, a head-to-head matrix shows each row entrant's win percentage against each column
entrant (with games played), in standings order, so you can compare any two entrants directly
instead of reading only the overall mix. When recording is enabled, `results.json` carries the same data as
`head_to_head` (win/loss/draw counts per pair).
Redirect stdout to save the final table while keeping progress visible.

Use a JSON config file, optionally overriding fields with flags:

```sh
./rust/target/release/schotten-native tournament --config rust/tournament.example.json
./rust/target/release/schotten-native tournament --config rust/tournament.example.json --games-per-pair 1000 --record-games 0
```

See [tournament.example.json](tournament.example.json). Config fields are `agents`,
`games_per_pair`, `deck_seed`, `agent_seed`, `record_games`, `record_dir` and `threads`.
Agent entries may be policy names or objects with `name`, `strategy` and optional
`seed`. A strategy may also be `mcg`, `mcb` or `mcp`, optionally with `:N` samples
(see Monte Carlo strategies below). Names must be unique. Unknown fields,
duplicate JSON keys and invalid values are rejected. `--agents` replaces the
configured entrant list; `--agent-seeds` must provide exactly one seed per entrant.
Paths in config files and flags are relative to the current working directory.
The example config enables 10 recordings and sets 14 worker threads explicitly.

Multiple named entrants can use the same policy. Assign individual starting seeds
in entrant order from the CLI:

```sh
./rust/target/release/schotten-native tournament --agents first=greedy second=greedy baseline=random --games-per-pair 100 --deck-seed 7 --agent-seeds 10 20 30
```

Both base seeds default to 1. At round index `r` within each pairing, the deck
seed is `deck_seed + r`, shared across pairings. An entrant's RNG seed is its
individual starting seed plus `r`; without an individual seed, its starting seed
is `agent_seed + entrant_index`. Indices start at zero. Seats are swapped after
assigning entrant seeds. Explicit settings reproduce results across worker counts.
Deterministic policies ignore their RNG seeds. This tournament schedule differs
from the two-player `series` command's seed increment formula.

Use `random` instead of an integer to choose a starting seed from the operating
system's randomness source:

```sh
./rust/target/release/schotten-native tournament --agents random greedy builder --games-per-pair 100 --deck-seed random --agent-seed random
./rust/target/release/schotten-native tournament --agents random greedy --deck-seed random --agent-seeds random random
```

The first command randomizes the deck seed and one common agent seed base;
entrants without individual seeds inherit that base plus their index. The second
chooses an independent starting seed for each entrant. Integers and `random`
can be mixed in `--agent-seeds`. Explicit individual seeds still override the base.

JSON accepts `"random"` for `deck_seed`, `agent_seed`, or an entrant's `seed`.
Random starting seeds are resolved once per tournament run, before games start;
the same deterministic per-round increments and shared deck seeds across pairings
still apply. This does not draw fresh OS randomness for every individual game.
The resolved numbers are printed and, when recording is enabled, saved in the run
manifest and game files. The manifest's resolved config can reproduce the run.
Omitted seeds still default to 1, and deterministic policies still ignore agent
RNG seeds. Randomness-source errors stop the run before any games are played.

Recording is disabled by default. `--record-games N` records the first N scheduled
games, independent of parallel completion order. `--record-games all` (or the
bare `--record-games` flag) records every game; `--record-games 0` disables it,
including when overriding an enabled config setting. Choose the parent directory
with `--record-dir PATH` (default `tournament-recordings`).
An explicit recording count must not exceed the total number of scheduled games.

Each recorded run creates a unique subdirectory, so previous runs are not
overwritten. Selected games contain JSON replay data: agent/seat metadata, exact
effective seeds, initial hands, deck draw order, turn intents and outcome. A run
manifest and results are saved alongside them. Failed recording writes abort with
a nonzero exit status rather than silently dropping data. Large recording counts
can consume substantial disk space; unrecorded games retain no histories.

Run directories contain `manifest.json`, `results.json`, and selected games named
`game-000001.json`, etc. Game files use schema version 1 and store effective seeds
as decimal strings. The manifest's `config` object can be saved as a standalone
JSON config to repeat the run. Replay files contain turn intents and revealed
cards; the browser's **Recordings** mode can play them. There is no replay CLI
or terminal replay viewer.

### Native strategies

| Name | Decision rule | Tie-breaking |
|---|---|---|
| `random` | Uniform choice over legal card/stone pairs | Seeded RNG |
| `greedy` | Existing partial-formation heuristic and provable claims | Seeded RNG |
| `builder` | Strong completions supported by the remaining hand | Lowest stone, then card ID |
| `tactician` | Hand-building plus board routes, threats and card opportunity cost | Lowest stone, then card ID |
| `planner` | Fixed shortlist of tactician moves, then worst-case unseen-card replies | Lowest stone, then card ID |

The last three are Rust-only deterministic policies. They use only `PlayerView`,
never private engine hands or the deck order, and include all provable claims.
Identical observations produce identical moves, independent of bot seed, prior
calls or worker count. Use `--seed` for repeatable deals; `--bot-seed` still affects
a stochastic opponent. `random` and `greedy` retain their existing behavior.

```sh
./rust/target/release/schotten-native series --games 100 --strategy builder --opponent-strategy greedy --seed 1 --bot-seed 1 --quiet
./rust/target/release/schotten-native series --games 100 --strategy tactician --opponent-strategy builder --seed 1 --quiet
./rust/target/release/schotten-native series --games 100 --strategy planner --opponent-strategy tactician --seed 1 --quiet
```

Planner searches a fixed number of candidates and one opponent placement, with
no wall-time cutoff and no guessed draw. An unseen card is a possible opponent
card, not a known one, so its worst-case evaluation is intentionally conservative.
Extra complexity does not guarantee a higher win rate; compare policies over
seeded series with alternating starting seats.
Initial measured matchups are recorded in [DETERMINISTIC_BOTS.md](DETERMINISTIC_BOTS.md).

### Monte Carlo strategies (`mcg`, `mcb`, `mcp`)

Three Monte Carlo players differ only in the policy that plays the simulated games:

| Name | Plays simulated games with | Default samples |
|---|---|---|
| `mcg` | the simulator's greedy heuristic | 100 |
| `mcb` | the native `builder` policy | 20 |
| `mcp` | the native `planner` policy | 5 |

Add `:N` to set samples per decision: `mcg:200`, `mcb:30`, `mcp:10`. They work anywhere a
strategy name does: `tournament --agents`, JSON `strategy`, and `series --strategy`.
Sample counts must be nonnegative integers fitting native `usize`, with no arbitrary
cap (unsigned 64-bit on this Mac). Zero skips Monte Carlo sampling.

```sh
./rust/target/release/schotten-native tournament --agents builder planner strong=mcg:200 b=mcb:20 --games-per-pair 20
./rust/target/release/schotten-native series --games 50 --strategy mcb:20 --opponent-strategy planner --seed 1 --quiet
```

In JSON, give the same string as the strategy (`{"name":"strong","strategy":"mcg:200","seed":7}`).
Entrant names must be unique, so use `name=mcb:N` (or a JSON object) to enter several sample counts.
The old name `mc` was renamed `mcg` and is no longer accepted.

At each turn the agent imagines N possible worlds (the unseen cards shuffled into the opponent's
hand and the deck), plays every legal move to the end in each world with the rollout policy
controlling both seats (10% random moves for variety), and picks the move with the
highest estimated score (win = 1, draw = 0.5, loss = 0). Ties use the greedy
heuristic, then stable legal-move order.
`mcg:0` (or `mcb:0`, `mcp:0`) skips the search and plays the `greedy` heuristic's best move.
Every Monte Carlo agent claims every provable stone and sees only its `PlayerView`.
In the simulations, each seat gets the view the engine would give it, built from the
simulated game, so the builder and planner rollouts run the real native policies.

Cost is the main limit: a decision runs approximately `samples × legal moves`
complete simulated games, with additional cost for builder/planner rollouts.
Start with small sample counts and measure your workload before a large tournament.
The original [benchmarks](BENCHMARKS.md) cover greedy/random; the later
[MCB/MCP experiments](../benchmarks/mc-comparison-2026-11-19/RESULTS.md) include
sampling comparisons and identical-position rollout timings.
Each decision uses one thread (the tournament already runs a game per core), and
the search seed comes from the entrant's seeded RNG, so results are repeatable
and independent of `--threads`. These agents depend on their agent seed.

The module lives in `src/montecarlo/` (own bitmask rules and simulator for speed).
`tests/montecarlo_rules.rs` checks its rules against brute-force enumeration and the native
`rules` module, and `bots.rs` tests play full games through the native engine.

## Tests

```sh
cd rust
cargo test --release --locked
cargo fmt --check
cargo clippy --release --all-targets -- -D warnings
```

To run the external differential API tests, start the native server on port 8001
and run this from the repository root:

```sh
SCHOTTEN_RUST_URL=http://127.0.0.1:8001 python3 -m pytest tests/test_rust_server.py -q
```

These replay complete games against both implementations, including shuffled
decks, legal moves, claims, hidden information, logs, errors and stream behavior.
Without that environment variable they skip, so the normal Python test suite
does not require a running native server.

## Reproducing measurements

Start Python on port 18080 and native Rust on port 18081, in separate terminals:

```sh
python3 -m uvicorn schotten.app:app --host 127.0.0.1 --port 18080 --no-access-log
./rust/target/release/schotten-native serve --port 18081
```

Then, from the repository root:

```sh
python3 -m benchmarks.compare
```

This runs three samples sequentially for each workload, includes process startup,
and reports medians as JSON. It compares the same Python bots and 100 seeded games
against both REST servers, then runs a direct Python baseline and 100,000 native
games at 1, 4, 8 and 14 workers. All use deck and bot seed 1 and alternate starting
seats. Use `--games`, `--native-games`, `--threads` and `--repeats` to vary the run.
The direct Python baseline retains Pydantic player views and event histories;
native series uses compact views and omits histories. These are complete simulation
paths, not a microbenchmark of the rules function alone.
