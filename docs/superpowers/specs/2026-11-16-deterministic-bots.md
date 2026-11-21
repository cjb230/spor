# Deterministic native bots

> Historical design/development record. Later changes may supersede the original
> scope, limits and verification counts. See the [current documentation](../../README.md)
> for supported commands and behavior.

The user approved builder, tactician and planner, restricted to native Rust use.
No REST or Python strategy changes are required.

All use PlayerView only; choose a legal play, then include all provable claims.
Rank immediate winning claims first. Resolve equal evaluations by lowest stone
index then lowest card ID; decision-making does not consult RNG, time or threads.
Existing random/greedy semantics and seeded results remain unchanged.

- Builder: prefer secure claims, otherwise favor formations that can be completed
  strongly with cards currently in hand. Penalize completed losing formations.
- Tactician: extend builder with five-stone/adjacent-three board priorities,
  connecting friendly holdings, contesting opposing winning routes, avoiding
  provably unwinnable stones and conserving cards useful on other stones.
- Planner: search a fixed shortlist of eight tactician-ranked moves, enumerate
  plausible single-card opponent replies using unseen cards, and choose the best
  worst-case board evaluation. Do not model unknown draws or infer private hands.
  Opponent claims may only be awarded when provable even against the conservative
  superset of all cards they could think we might use; otherwise use public
  formation evaluation instead of assuming a claim is legal.

Implement strategy evaluation in a separate deterministic module, keeping the
existing bots module as the public interface and native series runner. Add CLI
names, focused rule fixtures, seed/worker independence tests and full-game legality
checks. Document scoring/search limits and report measured performance and strength
without assuming that increasing complexity improves win rate.
