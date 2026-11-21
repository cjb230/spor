# Schotten Totten game server

The first neural Python player, **`mlp`**, supports optional PyTorch/MPS training,
versioned weight checkpoints and REST play. See the [neural player guide](docs/NEURAL.md).

An authoritative two-player server with a [REST API](openapi.yaml), a full-screen
terminal client, and reference Python bots. The game uses 54 cards (six colors,
values 1–9), six-card hands, and nine stones. Win by claiming five stones or three
adjacent stones. There are no tactics cards.

An optimized Rust server and native self-play runner support five fixed policies
(`random`, `greedy`, `builder`, `tactician`, `planner`) and three Monte Carlo
strategies (`mcg`, `mcb`, `mcp`). See [rust/README.md](rust/README.md) for build,
launch, strategy and reproducibility details.
The native `tournament` command supports round-robin standings, live progress,
JSON configuration and optional replay recording.

## Quick start

Run from the repository root. This checkout selects the local pyenv environment
`spor` through `.python-version` (currently Python 3.15.2). Use that environment
for development, tests and training:

```sh
pyenv version
python -m pip install -r requirements.txt
python -m uvicorn schotten.app:app --reload
```

On a fresh machine, create the `spor` environment using a supported installed
Python version, for example `pyenv virtualenv 3.15.2 spor` with pyenv-virtualenv.
Alternatively, without pyenv, use Python 3.10 or newer in a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn schotten.app:app --reload
```

The Python server listens on `http://127.0.0.1:8000`, with interactive API docs
at `/docs` and generated OpenAPI JSON at `/openapi.json`. Keep the server running
while using the terminal client or Python bots below. Run all client commands
from the repository root in the same Python environment.

## Layout

| Path | What it is |
|---|---|
| `schotten/rules.py` | Pure rules: formation strength, claim proofs, win check |
| `schotten/game.py` | Game state machine: validation, turns, views, event log |
| `schotten/app.py` | FastAPI transport (auth, long-poll, SSE) |
| `schotten/client.py` | Typed REST client; works against a URL or a TestClient |
| `schotten/bot.py` | Bot (`greedy` / `random`) and CLI |
| `schotten/neural/` | Optional PyTorch MLP, recording loader, trainer, checkpoints and REST player |
| `schotten/terminal.py` | Full-screen human client with keyboard controls |
| `rust/` | Native engine, compatible REST server, bots, series and tournaments |
| `benchmarks/` | REST/direct simulation comparison and saved timing samples |
| `tests/` | Python rules, engine, API, clients and Rust differential tests |
| `openapi.yaml` | The contract; `tests/test_spec_parity.py` keeps the app in line with it |
| `docs/` | Documentation index and historical design/implementation records |

## Browser play and replays

The Rust server includes a visual web interface for playing against native bots and
watching recorded tournament games. Build and launch it from the repository root:

```sh
cd rust
cargo build --release --locked
cd ..
./rust/target/release/schotten-native serve --port 8000
```

Open **http://127.0.0.1:8000/**. Choose **New game** to play or **Recordings** to
watch saved games with a turn slider, stepping and autoplay. Choose Random,
Greedy, Builder, Tactician, Planner, MCG, MCB or MCP. Monte Carlo sample counts
are configurable; MCB with 20 samples is the default. The recordings root defaults to
`tournament-recordings/` and can be changed with `--recordings-dir PATH`.
Drag a card onto a legal stone to play immediately, or select and confirm it.
Completed browser games automatically join the recording library.
See the [browser guide](docs/WEB_UI.md) for controls, recording commands and the
browser-specific API. This interface requires the Rust server.

## API and custom clients

Both servers implement the same game API:

| Method and path | Purpose |
|---|---|
| `POST /games` | Create a game and receive seat 0's token |
| `POST /games/{game_id}/join` | Join as seat 1 and start the game |
| `GET /games/{game_id}` | Read your player view; optionally long-poll |
| `GET /games/{game_id}/legal-moves` | Enumerate plays and current provable claims on your turn |
| `POST /games/{game_id}/turn` | Submit a play and optional claims atomically |
| `GET /games/{game_id}/events` | Stream public events over SSE |
| `GET /games/{game_id}/log` | Read turn history and, after game over, revealed hands/deck |

Creating and joining are unauthenticated. Other operations require
`Authorization: Bearer <token>` using the token returned for that game and seat.
Views expose your hand, the public table, the opponent's hand count and
`unseen_cards`, which combines the hidden opponent hand and remaining deck.
Cards are objects such as `{"color":"red","value":9}`; API stone indices are 0–8.

Create options are `seed` (deck shuffle), `auto_claim` (default false), and
`turn_timeout_seconds` (default null; at least 5 when set). Automatic claiming
checks both players after a legal turn. Seat 0 moves first.

Send the current `version` with each turn. A turn includes `play` (card and stone)
and `claims`; claims are checked after placement, before drawing a replacement.
Passing is allowed only when no legal play exists. An invalid play or claim
rejects the whole turn; an expired clock may still cause a forfeit. An optional
UUID `turn_id` makes an accepted turn safe to retry for that seat: the server
returns the current view without applying it twice.

For long-polling, send `since=<version>&timeout=30` (timeout range 1–60 seconds).
An unchanged active game returns HTTP 204 when the wait expires. SSE supports
`Last-Event-ID` resumption; multiple events can share one transition version.
See [openapi.yaml](openapi.yaml) for schemas and error codes.

`SchottenClient` in [schotten/client.py](schotten/client.py) wraps create, join,
view, legal moves, turn and log operations with typed models. `view(since=...)`
returns `None` on HTTP 204; failed API calls raise `ApiError` with `status`, `code`
and `detail`. Close `client.http` when finished. For an example agent loop, see
`Bot.run` in [schotten/bot.py](schotten/bot.py).

## Terminal client

Start the server, then open a full-screen human client in another terminal:

```
python -m uvicorn schotten.app:app --reload
python -m schotten.terminal create
```

The board displays the game ID. A second player can join from another terminal:

```
python -m schotten.terminal join g_YOUR_GAME_ID
```

To play against a bot instead, join with `python -m schotten.bot join g_YOUR_GAME_ID`.
Both clients default to `http://localhost:8000`; put `--url http://host:8000`
after `create` or `join` to use another server. Creating supports `--seed`,
`--auto-claim`, and `--turn-timeout SECONDS` too.

- Up/down selects a card; left/right selects a stone. Enter submits the move.
- Stones marked `+` are legal targets for the selected card, as reported by the server.
- Enter includes all provable claims, including those enabled by the selected play.
  If no card can be played, Enter passes and claims any provable stones.
- `R` refreshes; `Q` or Escape exits. Quitting does not pause the server's turn clock.
- Stones are displayed as 1–9 (the API uses 0–8). Card labels use `R`, `B`, `G`,
  `Y`, `P`, and `Br` for brown, followed by the value.
- `you 1st` / `opp 1st` shows who completed a stone first and wins an exact tie.

Requires an interactive terminal of at least 80 columns by 24 rows and Python's
`curses` module (included with typical macOS/Linux Python installations).
The board refreshes automatically and stays responsive during network requests.

## Bots

`python -m schotten.bot` supports `greedy` and `random`. The separate
`python -m schotten.neural` command runs the checkpoint-based `mlp` player;
see the [neural guide](docs/NEURAL.md) for optional dependencies and training.
The eight native strategies run through Rust's `series` and `tournament` commands.

```sh
python -m schotten.bot selfplay                       # two bots, one game, verbose
python -m schotten.bot selfplay --auto-claim --seed 9 # reproducible deck
python -m schotten.bot create                         # prints a game id, waits for a joiner
python -m schotten.bot join g_81f3 --strategy random  # take the other seat
```

Options such as `--url`, `--strategy` and `--bot-seed` go after the subcommand.

Compare strategies over 100 games with per-game results and a final win/draw tally:

```
python -m schotten.bot series --games 100 --strategy greedy --opponent-strategy random --seed 1 --bot-seed 1
```

The server must be running. The strategies alternate the first-player seat.
Deck seeds increase by one per game; bot seeds vary by game and strategy entrant,
so the same starting seeds reproduce the series. Win percentages use all completed
games, including draws. Failed games stop the series with a nonzero exit code and
are excluded from the tally. Series output omits individual card plays.

Add `--quiet` (or `-q`) to print only final wins, draws, and elapsed wall-clock
seconds. Errors still appear on stderr. Elapsed time includes creating games,
network waits, playing, and closing connections, and is also shown without quiet mode:

```
python -m schotten.bot series --games 100 --seed 1 --bot-seed 1 --quiet
```

The bot works only from its own player view. It works out claims locally from
`unseen_cards`, and the server remains the judge: a rejected claim just means it
plays on without it. Write your own player by using `SchottenClient` and replacing
`Bot.choose`.

## Known limits

- Active state is in memory and lost on restart. Run the Python reference server
  with one process; the Rust server supports multiple async worker threads.
  Completed recordings persist, but do not restore active sessions.
- Turn clocks are enforced lazily, on the next request or long-poll wake-up.
- The `greedy` bot is a simple heuristic: it favours promising partial formations and
  takes stones it can prove. It does not reason about what the opponent is building.

## Verification and benchmarks

```sh
python -m pytest -q
```

The Python suite covers rules, complete games, API behavior, bots, terminal
controls, series and contract parity. External Rust server tests skip unless
`SCHOTTEN_RUST_URL` is set. Native test commands and instructions for those
differential tests are in [rust/README.md](rust/README.md#tests).
Neural tests also require the optional `requirements-ml.txt` dependencies.

For saved measurements and their workload limits, see
[rust/BENCHMARKS.md](rust/BENCHMARKS.md) and
[rust/DETERMINISTIC_BOTS.md](rust/DETERMINISTIC_BOTS.md). To rerun the comparison,
follow [Reproducing measurements](rust/README.md#reproducing-measurements).
The [documentation index](docs/README.md) also links the historical design records.
