# Browser game and recordings

The native Rust server serves a browser interface for playing against native
bot opponents and watching recorded games.
Everything runs locally; no frontend build, account or external service is needed.

## Start the table

From the repository root, build the binary:

```sh
cd rust
cargo build --release --locked
cd ..
./rust/target/release/schotten-native serve --port 8000
```

Open **http://127.0.0.1:8000/** in a browser. Stop another server using port 8000,
including the Python reference server, or choose another port. The Python server
continues to provide the original REST API and terminal clients; the browser UI
is served by the Rust binary.

HTML, JavaScript and CSS are embedded in the executable. After changing them,
rebuild the release binary, restart the server, and reload the browser page.
Safari's reload-from-origin shortcut is **Option–Command–R**; Command–Shift–R
opens Reader and does not force a reload. Current browser responses disable caching.

Choose **New game**, pick an opponent, then deal. MCB remains the default.

| Opponent | Policy | Default samples |
|---|---|---:|
| Random | Random legal placements | — |
| Greedy | Partial-formation heuristic | — |
| Builder | Builds strong formations using its remaining hand | — |
| Tactician | Formation building plus board routes and threats | — |
| Planner | Evaluates shortlisted moves against possible replies | — |
| MCG | Monte Carlo with greedy rollouts | 100 |
| MCB | Monte Carlo with builder rollouts | 20 |
| MCP | Monte Carlo with planner rollouts | 5 |

The sample-count control appears for Monte Carlo opponents, with no arbitrary
sampling cap. Use a nonnegative whole number; 0 skips search and chooses the greedy
heuristic's best move. Counts use the native unsigned 64-bit integer range on this Mac.
A live warning appears when the chosen policy and count are estimated to exceed
2 seconds per turn. This does not prevent starting the game.
The estimate uses opening-position measurements on the M4 Pro, rounded upward:
0.00075 seconds per sample for MCG, 0.005 for MCB and 0.18 for MCP.
Later turns are usually faster; position and server load affect actual time.
Planner rollouts can be substantially slower.
The policies do not imply a guaranteed playing-strength ranking. Optional deal and bot seeds
reproduce a game. Both are signed 64-bit integers. With no seeds, deals and bot
choices use randomness.

Select a card from your hand, select a highlighted stone, then choose **Play card**.
You can also drag a hand card onto a highlighted stone or its column to play it
immediately. Dropping outside a legal column cancels the drag. On touchscreens,
hold a card briefly before dragging; a normal swipe still scrolls the page.
All provable claims are included automatically, including claims enabled by the
selected placement. Pass is available only when no card can be played. The opponent's
cards stay hidden. The page shows thinking progress and the final result.

The opponent's latest played card has a turquoise outline and **PLAYED** badge.
Your newly drawn card has a gold outline and **NEW** badge. A summary above the
board names both cards and the opponent's stone. These markers persist while
you choose a move and through polling; they update when the next card is played
or drawn. Playing your marked draw clears its marker if no replacement is drawn.
The server retains these markers, so reloading or reconnecting restores them.
The initial hand is not marked as new.

Browser storage keeps the current game's ID and human seat token so reloading
can restore it, including its selected opponent. The opponent continues thinking when the page disconnects. The server
keeps games in memory: restarting it loses active games. Browser play has no turn
clock. Switching to recordings preserves your active game.

Completed browser games are saved automatically and appear in **Recordings** as
**You vs your chosen opponent**. The result panel confirms when the replay is saved. Files live
under `browser-games/` in the configured recordings root and remain available
after restarting the server. Deals, bot seeds, turns, claims and outcomes are
recorded; seat tokens are never saved. Unfinished games remain in memory only.
If saving fails, the page reports it; fix the directory or disk problem and use
**Retry saving replay** to retry without changing the game result.

## Watch saved games

Choose **Recordings** to browse saved browser games and recorded tournaments. Select a game, then use
play/pause, previous/next, start/end, the turn slider and playback speed. Replay
mode reveals both original hands and shows the last turn. It reconstructs the
recorded game through the engine; it does not rerun the agents.

By default the server reads `tournament-recordings/` under its working directory.
Choose a different root when launching:

```sh
./rust/target/release/schotten-native serve --recordings-dir /path/to/tournament-recordings
```

The library expects run subdirectories containing schema-version-1 files named
`game-000001.json`, etc., as written by the native tournament command. Refresh
the library after recording more games. Missing directories show an empty library;
corrupt or incomplete games are skipped with diagnostics. Watching recordings is read-only.
Files larger than 1 MiB and games exceeding 128 turns are rejected.
The library scans at most 10,000 game files across run directories and reports
“Recording library limit reached” if more are encountered.

To produce a small recorded tournament:

```sh
./rust/target/release/schotten-native tournament --agents greedy mcb:1 --games-per-pair 2 --record-games all
```

The browser supports mouse, touch and keyboard focus/activation. On narrow
screens the nine-stone board scrolls horizontally; the hand and controls fit
within the page. Card labels and symbols supplement colors.

## Browser-specific HTTP endpoints

These routes extend the native server and are separate from the shared Python
[OpenAPI contract](../openapi.yaml).

| Method and path | Behavior |
|---|---|
| `POST /web/games` | Create a human seat-0 game against a selected bot; return seat grant |
| `GET /web/games/{id}` | Read human view and bot state; requires human bearer token |
| `POST /web/games/{id}/turn` | Submit a versioned turn and schedule one opponent reply |
| `GET /web/recordings` | List validated recorded games and warnings |
| `GET /web/recordings/{run}/{file}` | Return metadata and board snapshots for playback |

Creation accepts `opponent` (default `mcb`), optional `samples`, optional `seed`
and optional `bot_seed`. Use one of the eight lowercase policy names above.
Monte Carlo sample counts default to 100/20/5 for `mcg`/`mcb`/`mcp`; explicit
counts must be nonnegative integers fitting native `usize`, with no additional cap.
Fixed policies reject `samples`; colon overrides such as
`mcb:1000` are rejected. Pass the count separately.
Views include `opponent: {name, strategy}`, with the effective strategy such as
`random` or `mcb:20`. Saved recordings retain the same name and effective policy.
Views also include `recent_cards: {opponentPlay, drawn}`. `opponentPlay` is the
latest opponent placement (`card` and zero-based `stone`); `drawn` is the human's
latest draw if still held. Each is null before its first event. Opponent draws
remain private. Browser responses use `Cache-Control: no-store`.
Game views add `bot_thinking`, `bot_error`, and `legal_moves` (null outside the
human's turn), plus `recording_saved`, `recording_error` and `recording_id`.
`recording_id` is null until the completed game's replay is saved. An authenticated
read retries a failed save. Turn requests use the normal `version`, optional UUID `turn_id`,
and optional `play`; the server computes provable claims for the human.

The recording list returns `games` and `warnings`. Each game's `id` consists of
its run directory and JSON filename. Replay responses contain `metadata` and
`frames`: the initial position followed by one frame per accepted turn. Each
frame includes the seat-0 game view, `opponent_hand`, and `last_turn` (null at the
start). Private hands are exposed only for finished recorded games.

The browser interface shares the server's listener. Anyone who can reach that
listener can browse its configured recordings; keep the default loopback binding
for local use.

## Development checks

From `rust/`, run `cargo test --release --locked` and `cargo fmt --check`.
Release tests keep the existing Monte Carlo full-game tests practical. If Node.js
is installed, run `node --test rust/web/app.test.cjs` from the repository root for
the browser polling regressions. Node.js is not required to serve or play the game.

To repeat the timing calibration, run `cargo run --release --example mc_rollout_cost`
from `rust/`. The probe measures all three rollout policies in six fixed positions,
three times each, with eight samples and one decision worker. On this M4 Pro the
2026-11-19 median opening times per sample were approximately 0.000629 seconds
(MCG), 0.00435 seconds (MCB), and 0.165 seconds (MCP). These estimates assume an
optimized release build on the server's CPU, rather than the browser's device.
