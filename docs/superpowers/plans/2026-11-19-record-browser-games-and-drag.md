# Browser recording and dragging

> Historical design/development record. Later changes may supersede the original
> scope, limits and verification counts. See the [current documentation](../../README.md)
> for supported commands and behavior.

The user requested automatic recording of played games and dragging cards onto
stones. Extend the existing local UI and native sessions in place.

## Behavior

- Every completed browser game is saved automatically into the configured
  recordings root, using the existing schema and replay validator.
- Resolve random deck/bot seeds once at creation so the saved deal can be
  reconstructed exactly. Never include seat tokens in saved files.
- Save on either human or MCB game-ending turns. Publish files atomically,
  avoid duplicates, report disk errors, and retry failed saves when reconnecting.
- A card dropped on a legal stone submits immediately. Invalid, canceled or stale
  drags submit nothing. Preserve click/select/confirm and keyboard play.
- Support pointer/touch input without replacing a live drag's source element.

## Tasks

- [x] Primary: failing full-game/restart/replay test, then native recording writer
  and session integration. Add failure and seed boundary regressions.
- [x] UI agent: failing drag behavior tests, pointer drag/drop interaction,
  target feedback and recording status; own browser assets only.
- [x] Primary: integration tests, full suites, release build, browser drag QA,
  docs, independent review, and refresh the local service.

## Validation

Recorded games must survive creating a new server router and replay to the same
board/outcome. Saving must not modify accepted turn semantics, expose private
hands while a game runs, duplicate records on retry, or silently lose disk errors.
Invalid drops, replay mode and MCB thinking must never cause a submission.


## Results

- Full-game recording test failed on missing recording_saved before implementation,
  then passed including discovery/replay through a fresh router with no session.
- Random deals and signed-64-bit seed boundaries replay correctly. Disk write
  failure is visible and authenticated GET retries it without changing outcome.
- Atomic publication tests confirm idempotence, no overwrite of differing content,
  no temporary files in the library and no escaped-directory symlink writes.
- Drag handlers failed missing-function tests before implementation; browser tests
  cover legal/invalid/canceled/stale drops, single submission and touch hold/scroll.
- Independent review found a missing visible save-retry action; the failing-first
  regression now passes with the Retry saving replay button.
- Final Rust release suite: 97 passed. Node suite: 13 passed. Python: 137 passed,
  123 external-server tests skipped. Rust fmt passes. Clippy passes with the same
  pre-existing needless_range_loop warning explicitly allowed.
- Chrome desktop QA: card dragged onto stone 1 plays immediately, MCB replies,
  and a subsequent outside-board drop leaves the board unchanged. Touch behavior
  has automated coverage; no physical touchscreen was available for browser QA.
- Updated service runs on 18083. The older 18082 service was left intact so an
  existing in-memory game is not discarded. The new page saves newly played games.
