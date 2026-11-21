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
        Rollout::Builder => Mode::Builder,
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
