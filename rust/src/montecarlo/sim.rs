//! A full-information game state for simulation. Turn order, claim timing, passing and the
//! "stuck" settlement follow `schotten/game.py`, so a rollout plays by the server's rules.

use super::cards::{self, ALL, Card, CardSet, bit};
use super::obs::{Observation, Side, StoneObs};
use super::rng::Rng;
use super::rules::{self, Proof};

pub const STONES: usize = 9;
/// A play: (card, stone index 0-8).
pub type Move = (Card, u8);

#[derive(Clone, Copy, Debug)]
pub struct Stone {
    pub cards: [[Card; 3]; 2],
    pub n: [u8; 2],
    /// -1 = undecided, else the seat that claimed it.
    pub claimed: i8,
    /// -1 = nobody has completed it, else the seat that completed first (wins exact ties).
    pub first: i8,
    // Cache for claim proofs: a set of pool cards that would let the opponent beat this
    // seat's formation. While all of them are still unseen, there is no need to re-check.
    has_w: [bool; 2],
    w: [CardSet; 2],
}

impl Stone {
    const EMPTY: Stone = Stone {
        cards: [[0; 3]; 2],
        n: [0; 2],
        claimed: -1,
        first: -1,
        has_w: [false; 2],
        w: [0; 2],
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Win(usize),
    Draw,
}

impl Outcome {
    /// Result from `seat`'s point of view: 1 win, 0.5 draw, 0 loss.
    pub fn value_for(self, seat: usize) -> f64 {
        match self {
            Outcome::Win(w) if w == seat => 1.0,
            Outcome::Win(_) => 0.0,
            Outcome::Draw => 0.5,
        }
    }
}

pub enum Claims<'a> {
    /// Claim everything provable (what a sensible player always does).
    Auto,
    /// Claim exactly these stones; any that is not provable is an error.
    Explicit(&'a [u8]),
}

#[derive(Clone)]
pub struct MoveList {
    moves: [Move; 54],
    len: usize,
}

impl MoveList {
    #[inline]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Clone)]
pub struct Sim {
    pub stones: [Stone; STONES],
    pub hands: [CardSet; 2],
    pub table: CardSet,
    deck: [Card; 54],
    deck_len: usize,
    pub turn: usize,
}

impl Sim {
    /// A fresh game: shuffle, deal six cards each, seat 0 to move.
    pub fn deal(rng: &mut Rng) -> Sim {
        let mut all: [Card; 54] = [0; 54];
        for (i, c) in all.iter_mut().enumerate() {
            *c = i as Card;
        }
        rng.shuffle(&mut all);
        let mut sim = Sim {
            stones: [Stone::EMPTY; STONES],
            hands: [0; 2],
            table: 0,
            deck: [0; 54],
            deck_len: 0,
            turn: 0,
        };
        for &c in &all[..6] {
            sim.hands[0] |= bit(c);
        }
        for &c in &all[6..12] {
            sim.hands[1] |= bit(c);
        }
        sim.deck[..42].copy_from_slice(&all[12..]);
        sim.deck_len = 42;
        sim
    }

    /// A game from an explicit deal: each seat's hand and the deck (cards are drawn from the end).
    pub fn from_deal(hands: [CardSet; 2], deck: &[Card]) -> Sim {
        let mut sim = Sim {
            stones: [Stone::EMPTY; STONES],
            hands,
            table: 0,
            deck: [0; 54],
            deck_len: deck.len(),
            turn: 0,
        };
        sim.deck[..deck.len()].copy_from_slice(deck);
        sim
    }

    /// One possible world consistent with what `obs` can see: the unseen cards are shuffled and
    /// split into the opponent's hand and the deck. `obs` must pass `validate()`.
    pub fn from_observation(obs: &Observation, rng: &mut Rng) -> Sim {
        let me = obs.you;
        let them = 1 - me;
        let mut sim = Sim {
            stones: [Stone::EMPTY; STONES],
            hands: [0; 2],
            table: 0,
            deck: [0; 54],
            deck_len: 0,
            turn: me,
        };
        let seat_of = |side: Option<Side>| match side {
            Some(Side::You) => me as i8,
            Some(Side::Them) => them as i8,
            None => -1,
        };
        let mut table = 0;
        for (i, so) in obs.stones.iter().enumerate() {
            let st = &mut sim.stones[i];
            for (k, &c) in so.yours.iter().enumerate() {
                st.cards[me][k] = c;
                table |= bit(c);
            }
            st.n[me] = so.yours.len() as u8;
            for (k, &c) in so.theirs.iter().enumerate() {
                st.cards[them][k] = c;
                table |= bit(c);
            }
            st.n[them] = so.theirs.len() as u8;
            st.claimed = seat_of(so.claimed);
            st.first = seat_of(so.first);
        }
        sim.table = table;
        sim.hands[me] = obs.hand.iter().fold(0, |m, &c| m | bit(c));

        let mut buf = [0 as Card; 54];
        let mut n = 0;
        for c in cards::iter(obs.unseen) {
            buf[n] = c;
            n += 1;
        }
        rng.shuffle(&mut buf[..n]);
        let k = obs.opp_hand_count;
        for &c in &buf[..k] {
            sim.hands[them] |= bit(c);
        }
        sim.deck[..n - k].copy_from_slice(&buf[k..n]);
        sim.deck_len = n - k;
        sim
    }

    /// What seat `p` can see right now.
    pub fn observe(&self, p: usize) -> Observation {
        let side = |seat: i8| {
            if seat < 0 {
                None
            } else if seat as usize == p {
                Some(Side::You)
            } else {
                Some(Side::Them)
            }
        };
        let stones = self
            .stones
            .iter()
            .map(|s| StoneObs {
                yours: s.cards[p][..s.n[p] as usize].to_vec(),
                theirs: s.cards[1 - p][..s.n[1 - p] as usize].to_vec(),
                claimed: side(s.claimed),
                first: side(s.first),
            })
            .collect();
        Observation {
            you: p,
            hand: cards::iter(self.hands[p]).collect(),
            stones,
            deck_count: self.deck_len,
            opp_hand_count: self.hands[1 - p].count_ones() as usize,
            unseen: self.pool(p),
        }
    }

    /// Cards seat `p` cannot see: everything not on the table and not in their hand.
    #[inline]
    pub fn pool(&self, p: usize) -> CardSet {
        ALL & !self.table & !self.hands[p]
    }

    pub fn deck_len(&self) -> usize {
        self.deck_len
    }

    pub fn legal_moves(&self, p: usize) -> MoveList {
        let mut open = 0u16;
        for (i, s) in self.stones.iter().enumerate() {
            if s.claimed < 0 && s.n[p] < 3 {
                open |= 1 << i;
            }
        }
        let mut list = MoveList {
            moves: [(0, 0); 54],
            len: 0,
        };
        for card in cards::iter(self.hands[p]) {
            let mut m = open;
            while m != 0 {
                let s = m.trailing_zeros() as u8;
                m &= m - 1;
                list.moves[list.len] = (card, s);
                list.len += 1;
            }
        }
        list
    }

    pub fn has_legal(&self, p: usize) -> bool {
        self.hands[p] != 0 && self.stones.iter().any(|s| s.claimed < 0 && s.n[p] < 3)
    }

    fn claim_check(&mut self, p: usize, idx: usize, pool: CardSet) -> bool {
        let s = &mut self.stones[idx];
        if s.has_w[p] && s.w[p] & !pool == 0 {
            return false; // the cards that blocked it earlier are all still out there
        }
        let mine = s.cards[p];
        let their = s.cards[1 - p];
        let k = s.n[1 - p] as usize;
        let tiebreak = s.first == p as i8;
        match rules::check_claim(mine, &their[..k], pool, tiebreak) {
            Proof::Provable => true,
            Proof::Blocked(w) => {
                s.has_w[p] = true;
                s.w[p] = w;
                false
            }
        }
    }

    fn winner_of(&self, order: [usize; 2]) -> Option<usize> {
        for &p in &order {
            let (mut count, mut run, mut adjacent) = (0, 0, false);
            for s in &self.stones {
                if s.claimed == p as i8 {
                    count += 1;
                    run += 1;
                    adjacent |= run >= 3;
                } else {
                    run = 0;
                }
            }
            if count >= 5 || adjacent {
                return Some(p);
            }
        }
        None
    }

    /// Play one turn for seat `p` (who must be on turn): place the card, claim, check for a
    /// win, draw, pass the turn. Returns the outcome if the game just ended.
    ///
    /// Like the server, a rejected turn changes nothing (`Auto` claims cannot be rejected, so
    /// the state is only snapshotted for `Explicit`).
    pub fn apply_turn(
        &mut self,
        p: usize,
        play: Option<Move>,
        claims: Claims,
    ) -> Result<Option<Outcome>, String> {
        if matches!(claims, Claims::Explicit(_)) {
            let backup = self.clone();
            let result = self.apply_unchecked(p, play, claims);
            if result.is_err() {
                *self = backup;
            }
            result
        } else {
            self.apply_unchecked(p, play, claims)
        }
    }

    fn apply_unchecked(
        &mut self,
        p: usize,
        play: Option<Move>,
        claims: Claims,
    ) -> Result<Option<Outcome>, String> {
        if self.turn != p {
            return Err(format!("not seat {p}'s turn"));
        }
        match play {
            None => {
                if self.has_legal(p) {
                    return Err("passed with a legal play available".into());
                }
            }
            Some((card, stone)) => {
                let st = self
                    .stones
                    .get(stone as usize)
                    .ok_or("stone out of range")?;
                if self.hands[p] & bit(card) == 0 {
                    return Err(format!("{} is not in seat {p}'s hand", cards::short(card)));
                }
                if st.claimed >= 0 || st.n[p] >= 3 {
                    return Err(format!("stone {stone} is not open to seat {p}"));
                }
                let st = &mut self.stones[stone as usize];
                st.cards[p][st.n[p] as usize] = card;
                st.n[p] += 1;
                if st.n[p] == 3 && st.first < 0 {
                    st.first = p as i8;
                }
                st.has_w = [false; 2];
                self.hands[p] &= !bit(card);
                self.table |= bit(card);
            }
        }

        // Claims are judged after placing the card but before drawing, as the server does.
        let pool = self.pool(p);
        match claims {
            Claims::Auto => {
                for idx in 0..STONES {
                    let s = &self.stones[idx];
                    if s.claimed < 0 && s.n[p] == 3 && self.claim_check(p, idx, pool) {
                        self.stones[idx].claimed = p as i8;
                    }
                }
            }
            Claims::Explicit(list) => {
                for &idx in list {
                    let idx = idx as usize;
                    let s = self.stones.get(idx).ok_or("claimed stone out of range")?;
                    if s.claimed >= 0 {
                        return Err(format!("stone {idx} is already claimed"));
                    }
                    if s.n[p] < 3 {
                        return Err(format!("stone {idx}: seat {p} does not have three cards"));
                    }
                    if !self.claim_check(p, idx, pool) {
                        return Err(format!("stone {idx} could still be lost"));
                    }
                }
                for &idx in list {
                    self.stones[idx as usize].claimed = p as i8;
                }
            }
        }

        if let Some(w) = self.winner_of([p, 1 - p]) {
            return Ok(Some(Outcome::Win(w)));
        }
        if play.is_some() && self.deck_len > 0 {
            self.deck_len -= 1;
            self.hands[p] |= bit(self.deck[self.deck_len]);
        }
        self.turn = 1 - p;
        if !self.has_legal(0) && !self.has_legal(1) {
            return Ok(Some(self.resolve_stuck(p)));
        }
        Ok(None)
    }

    /// Neither player can play any more: award every provable stone, then count.
    fn resolve_stuck(&mut self, mover: usize) -> Outcome {
        for who in [mover, 1 - mover] {
            let pool = self.pool(who);
            for idx in 0..STONES {
                let s = &self.stones[idx];
                if s.claimed < 0 && s.n[who] == 3 && self.claim_check(who, idx, pool) {
                    self.stones[idx].claimed = who as i8;
                }
            }
        }
        if let Some(w) = self.winner_of([mover, 1 - mover]) {
            return Outcome::Win(w);
        }
        let count = |seat: i8| self.stones.iter().filter(|s| s.claimed == seat).count();
        match count(0).cmp(&count(1)) {
            std::cmp::Ordering::Greater => Outcome::Win(0),
            std::cmp::Ordering::Less => Outcome::Win(1),
            std::cmp::Ordering::Equal => Outcome::Draw,
        }
    }
}
