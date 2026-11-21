# Browser opponent selection

> Historical design/development record. Later changes may supersede the original
> scope, limits and verification counts. See the [current documentation](../../README.md)
> for supported commands and behavior.

User request: choose different opponents in browser games.

- Keep MCB as default; offer random, greedy, builder, tactician, planner, mcg,
  mcb and mcp in New game.
- Use native Monte Carlo defaults 100/20/5 respectively; permit 1–100 samples.
  Fixed policies do not accept samples. Reject unknown names and colon overrides
  so browser search bounds cannot be bypassed.
- Include immutable opponent name/strategy in player views. Dialog changes must
  not relabel a running game; reload restores labels from server metadata.
- Save actual opponent names and effective strategies in recording metadata.
- Preserve dragging, automatic saves, seat privacy and existing REST clients.

## Tasks

- [x] Primary: failing API/native-reply/default/validation/recording tests, then
  session policy resolution and metadata integration.
- [x] UI agent: selector, policy descriptions, conditional sample controls,
  generic live labels and behavioral tests.
- [x] Primary: docs, full checks/build, browser QA, independent review, launch.


## Verification

- API opponent tests first failed with unknown-field and missing-metadata errors.
- All eight browser opponents complete a real reply identical to the same seeded
  native policy. Defaults, invalid names/counts and replay opponent metadata pass.
- Full release Rust suite: 100 passed. JavaScript behavior suite: 17 passed.
- Python suite: 137 passed, 123 external-server tests skipped.
- Rust fmt passes; clippy passes with the existing needless_range_loop allowed.
- Independent review found no important issues.
- Chrome QA: selector shows all eight opponents; Builder hides sample controls,
  starts a Builder game and retains its correct name after reload.
- Updated local service launched at http://127.0.0.1:18084/; prior service retained
  to preserve any active in-memory games. Documentation updated.
