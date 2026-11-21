//! The Monte Carlo player.
//!
//! For one decision it draws `samples` possible worlds, each a random way the unseen cards
//! could be split between the opponent's hand and the deck. Every legal move is then played
//! out to the end in every one of those worlds (the same worlds for all moves, which removes
//! most of the noise when comparing them), and the move that wins most often is chosen.
//! Claims are not part of the decision: after the chosen play it claims every stone it can
//! prove, since that never hurts.

use std::time::{Duration, Instant};

use super::obs::Observation;
use super::policy::{self, Rollout};
use super::rng::Rng;
use super::sim::{Claims, Move, Outcome, Sim};

#[derive(Clone, Debug)]
pub struct McConfig {
    /// Possible worlds sampled per decision. Each legal move is played out once in each, so
    /// a decision costs about `samples x (number of legal moves)` simulated games. 0 = no
    /// simulation, just the greedy heuristic.
    pub samples: usize,
    /// Worker threads. Results do not depend on this.
    pub threads: usize,
    /// Seed: the same observation and seed always give the same decision.
    pub seed: u64,
    /// Chance that a simulated player picks a random move instead of the heuristic's.
    pub epsilon: f32,
    /// Policy that plays both seats in the simulated games.
    pub rollout: Rollout,
}

impl Default for McConfig {
    fn default() -> Self {
        McConfig {
            samples: 200,
            threads: 1,
            seed: 1,
            epsilon: 0.1,
            rollout: Rollout::Greedy,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MoveEstimate {
    pub play: Move,
    /// Estimated win probability (draw counts half).
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct Decision {
    /// None = pass (no legal play).
    pub play: Option<Move>,
    /// Stones to claim this turn, 0-based.
    pub claims: Vec<u8>,
    /// Estimates for every legal move, best first. Empty when no simulation was needed.
    pub estimates: Vec<MoveEstimate>,
    pub elapsed: Duration,
}

/// Play a simulated game to the end with the rollout policy for both seats.
pub fn rollout(mut sim: Sim, rng: &mut Rng, kind: Rollout, epsilon: f32) -> Outcome {
    loop {
        let p = sim.turn;
        let mv = policy::rollout_move_with(kind, &sim, p, rng, epsilon);
        if let Some(outcome) = sim
            .apply_turn(p, mv, Claims::Auto)
            .expect("rollout policy made an illegal move")
        {
            return outcome;
        }
    }
}

/// Total score of each move over the given sample indices.
fn score_samples(
    obs: &Observation,
    moves: &[Move],
    cfg: &McConfig,
    indices: impl Iterator<Item = usize>,
) -> Vec<f64> {
    let me = obs.you;
    let mut sums = vec![0.0f64; moves.len()];
    for i in indices {
        let world = Sim::from_observation(obs, &mut Rng::stream(cfg.seed, i as u64));
        for (k, &mv) in moves.iter().enumerate() {
            let mut sim = world.clone();
            // Same random stream for every move in this world: differences come from the move.
            let mut rng = Rng::stream(cfg.seed ^ 0x5DEE_CE66_D1CE_F00D, i as u64);
            let outcome = match sim.apply_turn(me, Some(mv), Claims::Auto) {
                Ok(Some(o)) => o,
                Ok(None) => rollout(sim, &mut rng, cfg.rollout, cfg.epsilon),
                Err(e) => panic!("simulated an illegal move: {e}"),
            };
            sums[k] += outcome.value_for(me);
        }
    }
    sums
}

pub fn decide(obs: &Observation, cfg: &McConfig) -> Result<Decision, String> {
    let start = Instant::now();
    obs.validate()?;
    let moves = obs.legal_moves();
    let finish = |play: Option<Move>, estimates: Vec<MoveEstimate>| Decision {
        claims: obs.provable_claims(play),
        play,
        estimates,
        elapsed: start.elapsed(),
    };

    if moves.is_empty() {
        return Ok(finish(None, vec![]));
    }
    let heuristic: Vec<f32> = moves.iter().map(|&m| obs.heuristic(m)).collect();
    let best_by_heuristic = (0..moves.len())
        .max_by(|&a, &b| {
            heuristic[a]
                .partial_cmp(&heuristic[b])
                .unwrap()
                .then(b.cmp(&a))
        })
        .unwrap();
    if moves.len() == 1 || cfg.samples == 0 {
        return Ok(finish(Some(moves[best_by_heuristic]), vec![]));
    }

    let threads = cfg.threads.clamp(1, cfg.samples);
    let mut totals = vec![0.0f64; moves.len()];
    if threads == 1 {
        totals = score_samples(obs, &moves, cfg, 0..cfg.samples);
    } else {
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|t| {
                    let moves = &moves;
                    scope.spawn(move || {
                        score_samples(obs, moves, cfg, (t..cfg.samples).step_by(threads))
                    })
                })
                .collect();
            for w in workers {
                for (total, part) in totals
                    .iter_mut()
                    .zip(w.join().expect("worker thread panicked"))
                {
                    *total += part;
                }
            }
        });
    }

    let n = cfg.samples as f64;
    // Scores are multiples of 0.5, so the sums are exact and do not depend on thread count.
    // Equal win rates fall back to the heuristic, then to the earlier move.
    let mut order: Vec<usize> = (0..moves.len()).collect();
    order.sort_by(|&a, &b| {
        totals[b]
            .partial_cmp(&totals[a])
            .unwrap()
            .then(heuristic[b].partial_cmp(&heuristic[a]).unwrap())
            .then(a.cmp(&b))
    });
    let estimates = order
        .iter()
        .map(|&k| MoveEstimate {
            play: moves[k],
            value: totals[k] / n,
        })
        .collect();
    Ok(finish(Some(moves[order[0]]), estimates))
}
