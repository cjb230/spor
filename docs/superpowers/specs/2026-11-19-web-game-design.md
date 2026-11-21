# Browser play and recorded games

> Historical design/development record. Later changes may supersede the original
> scope, limits and verification counts. See the [current documentation](../../README.md)
> for supported commands and behavior.

Status: approved by the user on 2026-11-19.

## Intent

Provide an attractive browser interface for playing Schotten Totten against the
existing native `mcb` agent and watching saved tournament games. The user wants
both flows on a web page rather than in the terminal.

## Approaches

1. Extend the native Rust server with browser assets and web-specific endpoints.
   This directly reuses the authoritative engine, MCB and local recordings. This
   is the proposed choice; it runs locally and needs no new hosted service.
2. Use a separate frontend and backend. This allows independent deployments but
   adds setup, origin configuration and a second process for the same experience.
3. Host on Sites. Native MCB and recordings would need a reachable external
   backend or a substantial engine port. This is deferred for this repo-based UI.

The user approved approach 1.

## Experience and appearance

Use the whole page as a game screen. Dark forest-green tabletop, high-contrast
cream text, richly colored playing cards, softly shaded stone markers and clear
selection highlights. Use CSS and SVG shapes rather than external image assets.
Cards include color names/symbols and values so color is not the only cue.

Desktop layout:

```text
SCHOTTEN TOTTEN                  Play / Recordings
MCB · thinking/ready             stones won / deck count
                   opponent formations
                   nine boundary stones
                   your formations
Your hand                       selected card / Play card
                               turn feedback / new game
```

On narrow screens keep all nine columns readable via horizontal board scrolling,
with the hand and controls fitting the viewport. Support mouse, touch and keyboard
activation, visible focus and reduced motion. Show loading, empty and error states.

## Play against MCB

Serve the interface at `/` in the native server. A new game seats the human at
seat 0 and MCB at seat 1. Offer MCB samples per decision, default 20, with a small
bounded UI range (1–100). Optional deal/agent seeds support reproducible games.
No turn timer by default. Human cards remain visible; MCB's hand remains hidden.

Select a hand card, highlight legal stones, choose a stone and explicitly submit.
Include all provable human claims after placement, matching terminal behavior.
Expose a pass action only when passing is legal. Show whose turn it is, thinking
feedback, both stone totals, remaining deck, and final outcome. Allow starting
another game and restoring an active game after reload using its seat token.

MCB uses `Bot` and `Game::player_view` exactly as native series does. Run expensive
decisions in a blocking worker, outside async request threads. Serialize bot work
per game, prevent duplicate moves, and validate the version before applying a
computed move. Polling observes progress and can resume after page reload.
Do not rely on a human-turn HTTP request remaining connected for the bot to move.

Use the existing REST game operations where possible. Add a bot-game creation
operation and trigger native bot turns after accepted human turns. Existing
two-human games and clients retain their behavior. The UI does not receive the
bot's token or private state. Keep unfinished games in memory as today.

## Watching recordings

Read schema-version-1 tournament JSON under a configurable recordings root,
default `tournament-recordings` relative to the server's working directory.
List existing run/game files with entrant names, strategies and outcome; provide
an empty-state explanation when no recordings exist. Refresh finds newly written
files. Skip malformed/incomplete files with a useful diagnostic, without breaking
the whole library. Resolve recordings through validated IDs under that root.

Load a recording and reconstruct initial hands, draw order, plays, explicit
claims, automatic endgame settlement and outcome through native engine rules.
Serve validated board snapshots for each turn. Replays must agree with recorded
outcomes and never run MCB again. Preserve exact wide seeds; use revealed cards
to restore the original deal where appropriate rather than narrowing seeds.

Show both hands in replay mode, clearly labeled. Provide previous/next turn,
start/end, play/pause, speed selection and a turn slider. Describe the last move
and claims; retain entrant metadata and final result. Opening a replay does not
discard an active human game. Playback timers stop when leaving replay mode.

## Boundaries and builder decisions

No public hosting, accounts, matchmaking, terminal replay viewer or tournament
management UI in this change. No changes to MCB scoring or game rules. No new
JavaScript build tool is needed: serve HTML/CSS/ES modules with the Rust binary.
Existing historical benchmark results remain historical.

The builder chooses exact endpoint names, module boundaries, typography, responsive
breakpoints and implementation details. Add any required API contract entries,
document startup and recordings options, and retain existing API parity checks.

## Verification

Test bot-game creation, a legal human/MCB exchange, hidden-information boundaries,
duplicate submissions, recording enumeration/path validation, replay reconstruction
and final outcomes against real tournament fixtures. Verify existing Rust and
Python tests, formatting and applicable checks. In a browser, exercise card/stone
selection, an MCB reply, replay seeking/playback, reload and a narrow viewport.
Report any browser QA limitation. Provide the local URL and exact launch command.
