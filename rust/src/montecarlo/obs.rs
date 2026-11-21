//! What one player is allowed to know: the server's "player view", in Rust form.

use super::cards::{self, Card, CardSet, bit};
use super::policy;
use super::rules::{self, Proof};
use super::sim::Move;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    You,
    Them,
}

#[derive(Clone, Debug)]
pub struct StoneObs {
    pub yours: Vec<Card>,
    pub theirs: Vec<Card>,
    pub claimed: Option<Side>,
    pub first: Option<Side>,
}

#[derive(Clone, Debug)]
pub struct Observation {
    /// Seat number (0 or 1).
    pub you: usize,
    pub hand: Vec<Card>,
    pub stones: Vec<StoneObs>,
    pub deck_count: usize,
    pub opp_hand_count: usize,
    /// Every card that is neither on the table nor in your hand: the opponent's hand plus the deck.
    pub unseen: CardSet,
}

impl Observation {
    /// Cheap sanity checks, so a malformed view is reported instead of producing nonsense.
    pub fn validate(&self) -> Result<(), String> {
        if self.stones.len() != 9 {
            return Err(format!("expected 9 stones, got {}", self.stones.len()));
        }
        let unseen = self.unseen.count_ones() as usize;
        if unseen != self.deck_count + self.opp_hand_count {
            return Err(format!(
                "unseen cards ({unseen}) != deck ({}) + opponent hand ({})",
                self.deck_count, self.opp_hand_count
            ));
        }
        let mut seen: CardSet = self.unseen;
        for &c in self.hand.iter().chain(
            self.stones
                .iter()
                .flat_map(|s| s.yours.iter().chain(s.theirs.iter())),
        ) {
            if seen & bit(c) != 0 {
                return Err(format!(
                    "card {} appears twice in the view",
                    cards::short(c)
                ));
            }
            seen |= bit(c);
        }
        if seen != cards::ALL {
            return Err("the view does not account for all 54 cards".to_string());
        }
        Ok(())
    }

    /// Every (card, stone) play available to you.
    pub fn legal_moves(&self) -> Vec<Move> {
        let mut out = Vec::new();
        for &card in &self.hand {
            for (i, s) in self.stones.iter().enumerate() {
                if s.claimed.is_none() && s.yours.len() < 3 {
                    out.push((card, i as u8));
                }
            }
        }
        out
    }

    /// Can stone `idx` be claimed once `play` has been made? (The pool is exactly the unseen
    /// cards: placing a card from your own hand does not change it.)
    pub fn claimable_after(&self, play: Option<Move>, idx: usize) -> bool {
        let s = &self.stones[idx];
        if s.claimed.is_some() {
            return false;
        }
        let mut mine = s.yours.clone();
        let mut first = s.first;
        if let Some((card, stone)) = play {
            if stone as usize == idx {
                mine.push(card);
                if mine.len() == 3 && first.is_none() {
                    first = Some(Side::You);
                }
            }
        }
        if mine.len() != 3 {
            return false;
        }
        matches!(
            rules::check_claim(
                [mine[0], mine[1], mine[2]],
                &s.theirs,
                self.unseen,
                first == Some(Side::You)
            ),
            Proof::Provable
        )
    }

    /// All stones you could claim after `play`, as 0-based indices.
    pub fn provable_claims(&self, play: Option<Move>) -> Vec<u8> {
        (0..self.stones.len())
            .filter(|&i| self.claimable_after(play, i))
            .map(|i| i as u8)
            .collect()
    }

    /// The greedy heuristic of the Python bot, including the bonus for a stone you can claim now.
    pub fn heuristic(&self, mv: Move) -> f32 {
        let (card, stone) = mv;
        let s = &self.stones[stone as usize];
        let mut mine = s.yours.clone();
        mine.push(card);
        let mut score = policy::partial_score(&mine);
        if mine.len() == 3 {
            if self.claimable_after(Some(mv), stone as usize) {
                score += 1000.0;
            } else if s.theirs.len() == 3 {
                let (a, b) = (
                    rules::evaluate([mine[0], mine[1], mine[2]]),
                    rules::evaluate([s.theirs[0], s.theirs[1], s.theirs[2]]),
                );
                score += if a > b { 5.0 } else { -20.0 };
            }
        }
        score
    }
}
