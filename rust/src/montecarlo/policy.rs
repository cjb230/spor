//! The fast heuristic used to play out simulated games (and as the "greedy" baseline).
//! It is the Python bot's `partial_score`, ported, plus epsilon-greedy move choice.

use super::cards::{self, Card};
use super::rng::Rng;
use super::rules;
use super::sim::{Move, Sim};
use crate::deterministic::{self, Mode};
use crate::engine::{Line, PlayerView, ViewStone};

/// How promising a stone is with one or two cards; completed stones score by strength.
pub fn partial_score(c: &[Card]) -> f32 {
    match c.len() {
        3 => {
            let s = rules::evaluate([c[0], c[1], c[2]]);
            rules::formation_of(s) as f32 * 100.0 + (s & 31) as f32
        }
        1 => cards::value(c[0]) as f32 / 2.0,
        2 => {
            let (va, vb) = (cards::value(c[0]) as f32, cards::value(c[1]) as f32);
            let gap = (va - vb).abs() as i32;
            let same = cards::color(c[0]) == cards::color(c[1]);
            if same && gap == 1 {
                60.0 + (va + vb) / 4.0
            } else if gap == 0 {
                50.0 + va
            } else if same && gap == 2 {
                45.0
            } else if same {
                35.0 + (va + vb) / 4.0
            } else if gap == 1 {
                30.0 + (va + vb) / 4.0
            } else {
                (va + vb) / 2.0
            }
        }
        _ => 0.0,
    }
}

/// Score of playing `card` on `stone` for seat `p`, using only what `p` can see on the board.
#[inline]
fn move_score(sim: &Sim, p: usize, card: Card, stone: usize) -> f32 {
    let s = &sim.stones[stone];
    let n = s.n[p] as usize;
    let mut mine = [0 as Card; 3];
    mine[..n].copy_from_slice(&s.cards[p][..n]);
    mine[n] = card;
    let mut score = partial_score(&mine[..n + 1]);
    if n + 1 == 3 && s.n[1 - p] == 3 {
        score += if rules::evaluate(mine) > rules::evaluate(s.cards[1 - p]) {
            5.0
        } else {
            -20.0
        };
    }
    score
}

/// Choose a play for seat `p`: the best-scoring move (ties broken at random), or with
/// probability `epsilon` a uniformly random legal move. None means no legal play (pass).
pub fn rollout_move(sim: &Sim, p: usize, rng: &mut Rng, epsilon: f32) -> Option<Move> {
    let list = sim.legal_moves(p);
    let moves = list.as_slice();
    if moves.is_empty() {
        return None;
    }
    if epsilon > 0.0 && rng.f32() < epsilon {
        return Some(moves[rng.below(moves.len())]);
    }
    let mut best = f32::NEG_INFINITY;
    let mut chosen = moves[0];
    let mut ties = 0usize;
    for &(card, stone) in moves {
        let s = move_score(sim, p, card, stone as usize);
        if s > best {
            best = s;
            chosen = (card, stone);
            ties = 1;
        } else if s == best {
            ties += 1;
            if rng.below(ties) == 0 {
                chosen = (card, stone);
            }
        }
    }
    Some(chosen)
}

/// Which policy plays both seats inside simulated games.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rollout {
    /// The fast partial-formation heuristic (`greedy`).
    Greedy,
    /// The native `builder` policy.
    Builder,
    /// The native `planner` policy.
    Planner,
}

/// What seat `p` can see in the simulated game, in the native engine's view type.
fn player_view(sim: &Sim, p: usize) -> PlayerView {
    let line = |cards: &[u8; 3], n: u8| Line::from_slice(&cards[..n as usize]);
    let side = |seat: i8| (seat >= 0).then_some(seat as usize == p);
    let mut stones = [ViewStone {
        yours: Line::default(),
        theirs: Line::default(),
        claimed_by: None,
        first_to_complete: None,
    }; 9];
    for (view, st) in stones.iter_mut().zip(sim.stones.iter()) {
        *view = ViewStone {
            yours: line(&st.cards[p], st.n[p]),
            theirs: line(&st.cards[1 - p], st.n[1 - p]),
            claimed_by: side(st.claimed),
            first_to_complete: side(st.first),
        };
    }
    PlayerView {
        version: 0,
        hand: sim.hands[p],
        unseen: sim.pool(p),
        stones,
    }
}

/// Like [`rollout_move`], but the non-random choice comes from the chosen policy.
pub fn rollout_move_with(
    kind: Rollout,
    sim: &Sim,
    p: usize,
    rng: &mut Rng,
    epsilon: f32,
) -> Option<Move> {
    let mode = match kind {
        Rollout::Greedy => return rollout_move(sim, p, rng, epsilon),
        Rollout::Builder => return builder_rollout_move(sim, p, rng, epsilon),
        Rollout::Planner => Mode::Planner,
    };
    let list = sim.legal_moves(p);
    let moves = list.as_slice();
    if moves.is_empty() {
        return None;
    }
    if epsilon > 0.0 && rng.f32() < epsilon {
        return Some(moves[rng.below(moves.len())]);
    }
    if moves.len() == 1 {
        return Some(moves[0]);
    }
    let play = deterministic::choose(&player_view(sim, p), mode)?;
    Some((play.card.id(), play.stone as u8))
}

/// Builder enumerates candidates itself; only the epsilon branch needs a
/// uniformly sampled legal move. Select that move directly in the original
/// card-major/stone-minor order, without constructing a 54-entry MoveList.
fn builder_rollout_move(sim: &Sim, p: usize, rng: &mut Rng, epsilon: f32) -> Option<Move> {
    let hand = sim.hands[p];
    if hand == 0 {
        return None;
    }
    let mut open = 0u16;
    for (index, stone) in sim.stones.iter().enumerate() {
        if stone.claimed < 0 && stone.n[p] < 3 {
            open |= 1 << index;
        }
    }
    if open == 0 {
        return None;
    }
    let stones = open.count_ones() as usize;
    let count = hand.count_ones() as usize * stones;
    // Preserve both RNG draws even when the random branch has only one move.
    if epsilon > 0.0 && rng.f32() < epsilon {
        let index = rng.below(count);
        let mut cards = hand;
        for _ in 0..index / stones {
            cards &= cards - 1;
        }
        for _ in 0..index % stones {
            open &= open - 1;
        }
        return Some((cards.trailing_zeros() as u8, open.trailing_zeros() as u8));
    }
    if count == 1 {
        return Some((hand.trailing_zeros() as u8, open.trailing_zeros() as u8));
    }
    deterministic::choose(&player_view(sim, p), Mode::Builder)
        .map(|play| (play.card.id(), play.stone as u8))
}

#[cfg(test)]
mod builder_tests {
    use super::*;
    use crate::montecarlo::sim::Claims;

    // The previous Builder wrapper, including exact random-number consumption.
    fn original(sim: &Sim, p: usize, rng: &mut Rng, epsilon: f32) -> Option<Move> {
        let list = sim.legal_moves(p);
        let moves = list.as_slice();
        if moves.is_empty() {
            return None;
        }
        if epsilon > 0.0 && rng.f32() < epsilon {
            return Some(moves[rng.below(moves.len())]);
        }
        if moves.len() == 1 {
            return Some(moves[0]);
        }
        deterministic::choose(&player_view(sim, p), Mode::Builder)
            .map(|play| (play.card.id(), play.stone as u8))
    }

    fn compare(sim: &Sim, p: usize, seed: u64) {
        for epsilon in [0.0, 0.1, 1.0] {
            let mut before = Rng::new(seed);
            let mut after = Rng::new(seed);
            assert_eq!(
                rollout_move_with(Rollout::Builder, sim, p, &mut after, epsilon),
                original(sim, p, &mut before, epsilon),
                "epsilon={epsilon}"
            );
            for _ in 0..8 {
                assert_eq!(after.next_u64(), before.next_u64());
            }
        }
    }

    #[test]
    fn builder_rollouts_preserve_moves_and_random_streams() {
        for seed in 0..64 {
            let mut rng = Rng::new(seed);
            let mut sim = Sim::deal(&mut rng);
            for turn in 0..128 {
                let p = sim.turn;
                compare(&sim, p, seed * 128 + turn);
                let play = rollout_move(&sim, p, &mut rng, 0.3);
                if sim.apply_turn(p, play, Claims::Auto).unwrap().is_some() {
                    break;
                }
            }
        }
    }

    #[test]
    fn builder_rollouts_preserve_pass_and_forced_move_rng() {
        let mut sim = Sim::from_deal([0, 1], &[]);
        compare(&sim, 0, 21);
        for stone in &mut sim.stones {
            stone.claimed = 1;
        }
        sim.stones[4].claimed = -1;
        compare(&sim, 1, 42);
        sim.stones[4].claimed = 0;
        compare(&sim, 1, 73);
    }
}
