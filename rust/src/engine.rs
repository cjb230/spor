use crate::rng::PythonRandom;
use crate::rules::{self, FULL_POOL};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::fmt;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct GameError {
    pub code: &'static str,
    pub detail: String,
}

impl GameError {
    pub fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
    pub fn status(&self) -> u16 {
        match self.code {
            "UNAUTHORIZED" => 401,
            "NOT_FOUND" => 404,
            "GAME_NOT_STARTED" | "GAME_OVER" | "NOT_YOUR_TURN" => 403,
            "STALE_VERSION" | "GAME_FULL" => 409,
            _ => 422,
        }
    }
}
impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.detail)
    }
}
impl std::error::Error for GameError {}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default)]
pub struct Options {
    pub auto_claim: bool,
    pub turn_timeout_seconds: Option<i64>,
    pub seed: Option<i64>,
}
impl Options {
    pub fn validate(&self) -> Result<(), GameError> {
        if self.turn_timeout_seconds.is_some_and(|s| s < 5) {
            return Err(GameError::new(
                "INVALID_REQUEST",
                "turn_timeout_seconds must be at least 5",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Color {
    Red,
    Blue,
    Green,
    Yellow,
    Purple,
    Brown,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
pub struct Card {
    pub color: Color,
    pub value: u8,
}
impl Card {
    pub fn from_id(id: u8) -> Self {
        Self {
            color: [
                Color::Red,
                Color::Blue,
                Color::Green,
                Color::Yellow,
                Color::Purple,
                Color::Brown,
            ][(id / 9) as usize],
            value: id % 9 + 1,
        }
    }
    pub fn id(self) -> u8 {
        self.color as u8 * 9 + self.value - 1
    }
    fn validate(self) -> Result<(), GameError> {
        if !(1..=9).contains(&self.value) {
            return Err(GameError::new(
                "INVALID_REQUEST",
                "card value must be between 1 and 9",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct Play {
    pub card: Card,
    pub stone: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TurnRequest {
    pub version: i64,
    pub turn_id: Option<Uuid>,
    pub play: Option<Play>,
    #[serde(default)]
    pub claims: Vec<usize>,
}
impl TurnRequest {
    pub fn validate(&self) -> Result<(), GameError> {
        if let Some(p) = &self.play {
            p.card.validate()?;
            if p.stone >= 9 {
                return Err(GameError::new(
                    "INVALID_REQUEST",
                    "stone index must be between 0 and 8",
                ));
            }
        }
        if self.claims.iter().any(|&c| c >= 9) {
            return Err(GameError::new(
                "INVALID_REQUEST",
                "claim index must be between 0 and 8",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Line {
    cards: [u8; 3],
    len: u8,
}
impl Line {
    pub fn from_slice(cards: &[u8]) -> Self {
        assert!(cards.len() <= 3);
        let mut out = Self::default();
        out.cards[..cards.len()].copy_from_slice(cards);
        out.len = cards.len() as u8;
        out
    }
    pub fn len(&self) -> usize {
        self.len as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_slice(&self) -> &[u8] {
        &self.cards[..self.len()]
    }
    pub fn push(&mut self, c: u8) {
        assert!(self.len < 3);
        self.cards[self.len()] = c;
        self.len += 1;
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Stone {
    cards: [Line; 2],
    claimed_by: Option<usize>,
    first_to_complete: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ViewStone {
    pub yours: Line,
    pub theirs: Line,
    pub claimed_by: Option<bool>,
    pub first_to_complete: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerView {
    pub version: u64,
    pub hand: u64,
    pub unseen: u64,
    pub stones: [ViewStone; 9],
}

pub struct Game {
    pub id: String,
    pub tokens: [Option<String>; 2],
    pub version: u64,
    pub turn: Option<usize>,
    pub deadline: Option<i64>,
    pub status: &'static str,
    pub events: Vec<Value>,
    options: Options,
    hands: [u64; 2],
    deck: Vec<u8>,
    stones: [Stone; 9],
    winner: Option<usize>,
    win_reason: Option<&'static str>,
    log: Vec<Value>,
    initial_hands: [Vec<u8>; 2],
    draw_order: Vec<u8>,
    seen_turns: HashSet<(usize, Uuid)>,
    record: bool,
}

fn token() -> String {
    let mut bytes = [0u8; 24];
    getrandom::fill(&mut bytes).expect("operating system random source unavailable");
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(32);
    for b in bytes.as_chunks::<3>().0 {
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for shift in [18, 12, 6, 0] {
            out.push(ALPHABET[((n >> shift) & 63) as usize] as char);
        }
    }
    out
}

fn card_json(cs: impl IntoIterator<Item = u8>) -> Value {
    serde_json::to_value(cs.into_iter().map(Card::from_id).collect::<Vec<_>>()).unwrap()
}

fn table_mask(stones: &[Stone; 9]) -> u64 {
    rules::mask(
        stones
            .iter()
            .flat_map(|s| s.cards.iter())
            .flat_map(|line| line.as_slice().iter().copied()),
    )
}

fn has_play(stones: &[Stone; 9], hand: u64, seat: usize) -> bool {
    hand != 0
        && stones
            .iter()
            .any(|s| s.claimed_by.is_none() && s.cards[seat].len() < 3)
}

fn provable(stones: &[Stone; 9], idx: usize, seat: usize, pool: u64) -> bool {
    let s = &stones[idx];
    s.claimed_by.is_none()
        && s.cards[seat].len() == 3
        && rules::claim_is_provable(
            s.cards[seat].as_slice(),
            s.cards[1 - seat].as_slice(),
            pool,
            s.first_to_complete == Some(seat),
        )
}

impl Game {
    pub fn new(id: String, options: Options, _now_ms: i64, record: bool) -> Self {
        let mut rng = PythonRandom::new(options.seed);
        let mut game = Self {
            id,
            options,
            tokens: [
                Some(if record {
                    token()
                } else {
                    "native-seat-0".into()
                }),
                None,
            ],
            version: 0,
            turn: None,
            deadline: None,
            status: "waiting_for_opponent",
            events: Vec::new(),
            hands: [0; 2],
            deck: Vec::new(),
            stones: [Stone::default(); 9],
            winner: None,
            win_reason: None,
            log: Vec::new(),
            initial_hands: Default::default(),
            draw_order: Vec::new(),
            seen_turns: HashSet::new(),
            record,
        };
        game.deal(&mut rng);
        game
    }

    fn deal(&mut self, rng: &mut PythonRandom) {
        let mut deck = std::array::from_fn::<_, 54, _>(|i| i as u8);
        rng.shuffle(&mut deck);
        self.hands = [
            rules::mask(deck[..6].iter().copied()),
            rules::mask(deck[6..12].iter().copied()),
        ];
        if self.record {
            self.initial_hands = [deck[..6].to_vec(), deck[6..12].to_vec()];
            self.draw_order = deck[12..].iter().rev().copied().collect();
        }
        self.deck = deck[12..].to_vec();
    }

    pub fn reseed_words(&mut self, words: &[u32]) -> Result<(), GameError> {
        if self.status != "waiting_for_opponent" {
            return Err(GameError::new(
                "INVALID_REQUEST",
                "seed can only be changed before join",
            ));
        }
        if words.is_empty() {
            return Err(GameError::new(
                "INVALID_REQUEST",
                "seed words must not be empty",
            ));
        }
        self.deal(&mut PythonRandom::from_words(words));
        Ok(())
    }

    pub fn over(&self) -> bool {
        matches!(self.status, "finished" | "forfeit")
    }

    pub fn outcome(&self) -> Option<(Option<usize>, &'static str)> {
        self.win_reason.map(|reason| (self.winner, reason))
    }

    fn restart_clock(&mut self, now: i64) {
        self.deadline = self
            .options
            .turn_timeout_seconds
            .map(|s| now.saturating_add(s.saturating_mul(1000)));
    }

    fn commit(&mut self, pending: Vec<Value>) {
        self.version += 1;
        if self.record {
            for mut event in pending {
                event["version"] = json!(self.version);
                self.events.push(event);
            }
        }
    }

    fn finish(
        &mut self,
        winner: Option<usize>,
        reason: &'static str,
        status: &'static str,
    ) -> Value {
        self.status = status;
        self.winner = winner;
        self.win_reason = Some(reason);
        self.turn = None;
        self.deadline = None;
        if !self.record {
            return Value::Null;
        }
        let mut event = json!({"type":"game_finished","win_reason":reason});
        if let Some(w) = winner {
            event["winner_seat"] = json!(w);
        }
        event
    }

    fn claim_event(&self, seat: usize, idx: usize) -> Value {
        if !self.record {
            return Value::Null;
        }
        json!({"type":"stone_claimed","seat":seat,"stone":idx,
            "formation":rules::formation_label(rules::evaluate(self.stones[idx].cards[seat].as_slice()))})
    }

    fn require_turn(&self, seat: usize) -> Result<(), GameError> {
        if seat >= 2 {
            return Err(GameError::new("INVALID_REQUEST", "seat must be 0 or 1"));
        }
        if self.status == "waiting_for_opponent" {
            return Err(GameError::new(
                "GAME_NOT_STARTED",
                "waiting for a second player",
            ));
        }
        if self.over() {
            return Err(GameError::new("GAME_OVER", "the game has finished"));
        }
        if self.turn != Some(seat) {
            return Err(GameError::new(
                "NOT_YOUR_TURN",
                "it is the other player's turn",
            ));
        }
        Ok(())
    }

    pub fn join(&mut self, now_ms: i64) -> Result<String, GameError> {
        if self.tokens[1].is_some() {
            return Err(GameError::new("GAME_FULL", "both seats are taken"));
        }
        let token = if self.record {
            token()
        } else {
            "native-seat-1".into()
        };
        self.tokens[1] = Some(token.clone());
        self.status = "in_progress";
        self.turn = Some(0);
        self.restart_clock(now_ms);
        let pending = if self.record {
            vec![
                json!({"type":"player_joined","seat":1}),
                json!({"type":"game_started"}),
            ]
        } else {
            Vec::new()
        };
        self.commit(pending);
        Ok(token)
    }

    pub fn check_timeout(&mut self, now_ms: i64) -> bool {
        if self.status != "in_progress" || !self.deadline.is_some_and(|d| now_ms >= d) {
            return false;
        }
        let event = self.finish(Some(1 - self.turn.unwrap()), "forfeit", "forfeit");
        self.commit(vec![event]);
        true
    }

    pub fn play_turn(
        &mut self,
        seat: usize,
        req: TurnRequest,
        now_ms: i64,
    ) -> Result<(), GameError> {
        req.validate()?;
        if seat >= 2 {
            return Err(GameError::new("INVALID_REQUEST", "seat must be 0 or 1"));
        }
        self.check_timeout(now_ms);
        if req
            .turn_id
            .is_some_and(|id| self.seen_turns.contains(&(seat, id)))
        {
            return Ok(());
        }
        self.require_turn(seat)?;
        if req.version < 0 || req.version as u64 != self.version {
            return Err(GameError::new(
                "STALE_VERSION",
                format!(
                    "game is at version {}, request used {}",
                    self.version, req.version
                ),
            ));
        }
        let mut hand = self.hands[seat];
        let mut stones = self.stones;
        if let Some(play) = &req.play {
            let c = play.card.id();
            if hand & (1u64 << c) == 0 {
                return Err(GameError::new(
                    "CARD_NOT_IN_HAND",
                    "that card is not in your hand",
                ));
            }
            let s = &mut stones[play.stone];
            if s.claimed_by.is_some() {
                return Err(GameError::new(
                    "STONE_ALREADY_CLAIMED",
                    format!("stone {} is already claimed", play.stone),
                ));
            }
            if s.cards[seat].len() >= 3 {
                return Err(GameError::new(
                    "STONE_FULL",
                    format!("you already have three cards on stone {}", play.stone),
                ));
            }
            hand &= !(1u64 << c);
            s.cards[seat].push(c);
            if s.cards[seat].len() == 3 && s.first_to_complete.is_none() {
                s.first_to_complete = Some(seat);
            }
        } else if has_play(&stones, hand, seat) {
            return Err(GameError::new(
                "PLAY_REQUIRED",
                "you have a legal play, so you must make one",
            ));
        }
        let table = table_mask(&stones);
        let pool = FULL_POOL & !(table | hand);
        let mut claimed_now = Vec::new();
        let mut explicit = Vec::new();
        let mut seen_claims = 0u16;
        for &idx in &req.claims {
            if seen_claims & (1 << idx) != 0 {
                continue;
            }
            seen_claims |= 1 << idx;
            let stone = &stones[idx];
            if stone.claimed_by.is_some() {
                return Err(GameError::new(
                    "STONE_ALREADY_CLAIMED",
                    format!("stone {idx} is already claimed"),
                ));
            }
            if stone.cards[seat].len() < 3 {
                return Err(GameError::new(
                    "CLAIM_INCOMPLETE",
                    format!("you do not have three cards on stone {idx}"),
                ));
            }
            if !provable(&stones, idx, seat, pool) {
                return Err(GameError::new(
                    "CLAIM_NOT_PROVABLE",
                    format!("stone {idx} could still be lost"),
                ));
            }
            stones[idx].claimed_by = Some(seat);
            claimed_now.push((seat, idx));
            explicit.push(idx);
        }
        if self.options.auto_claim {
            for who in [seat, 1 - seat] {
                let who_pool = if who == seat {
                    pool
                } else {
                    FULL_POOL & !(table | self.hands[who])
                };
                for idx in 0..9 {
                    if provable(&stones, idx, who, who_pool) {
                        stones[idx].claimed_by = Some(who);
                        claimed_now.push((who, idx));
                    }
                }
            }
        }
        self.hands[seat] = hand;
        self.stones = stones;
        let mut pending = Vec::new();
        if self.record {
            if let Some(play) = &req.play {
                pending.push(
                    json!({"type":"card_played","seat":seat,"card":play.card,"stone":play.stone}),
                );
            }
            for (who, idx) in claimed_now {
                pending.push(self.claim_event(who, idx));
            }
        }
        if let Some((winner, reason)) = self.check_win(seat) {
            pending.push(self.finish(Some(winner), reason, "finished"));
        } else {
            if req.play.is_some()
                && let Some(card) = self.deck.pop()
            {
                self.hands[seat] |= 1u64 << card;
            }
            self.turn = Some(1 - seat);
            self.restart_clock(now_ms);
            if self.record {
                pending.push(json!({"type":"turn_passed","seat":seat}));
            }
            if !(has_play(&self.stones, self.hands[0], 0)
                || has_play(&self.stones, self.hands[1], 1))
            {
                self.resolve_stuck(&mut pending, seat);
            }
        }
        self.commit(pending);
        if self.record {
            self.log.push(
                json!({"version":self.version,"seat":seat,"play":req.play,"claims":explicit}),
            );
        }
        if let Some(id) = req.turn_id {
            self.seen_turns.insert((seat, id));
        }
        Ok(())
    }

    fn check_win(&self, mover: usize) -> Option<(usize, &'static str)> {
        rules::check_win(&self.stones.map(|s| s.claimed_by), [mover, 1 - mover])
    }

    fn resolve_stuck(&mut self, pending: &mut Vec<Value>, mover: usize) {
        let table = table_mask(&self.stones);
        for who in [mover, 1 - mover] {
            let pool = FULL_POOL & !(table | self.hands[who]);
            for idx in 0..9 {
                if provable(&self.stones, idx, who, pool) {
                    self.stones[idx].claimed_by = Some(who);
                    if self.record {
                        pending.push(self.claim_event(who, idx));
                    }
                }
            }
        }
        let (winner, reason) = if let Some((w, r)) = self.check_win(mover) {
            (Some(w), r)
        } else {
            let count = |seat| {
                self.stones
                    .iter()
                    .filter(|s| s.claimed_by == Some(seat))
                    .count()
            };
            let (a, b) = (count(0), count(1));
            (
                if a > b {
                    Some(0)
                } else if b > a {
                    Some(1)
                } else {
                    None
                },
                "stalemate",
            )
        };
        pending.push(self.finish(winner, reason, "finished"));
    }

    pub fn player_view(&self, seat: usize) -> PlayerView {
        assert!(seat < 2);
        PlayerView {
            version: self.version,
            hand: self.hands[seat],
            unseen: FULL_POOL & !(table_mask(&self.stones) | self.hands[seat]),
            stones: self.stones.map(|s| ViewStone {
                yours: s.cards[seat],
                theirs: s.cards[1 - seat],
                claimed_by: s.claimed_by.map(|w| w == seat),
                first_to_complete: s.first_to_complete.map(|w| w == seat),
            }),
        }
    }

    pub fn view(&self, seat: usize) -> Value {
        let v = self.player_view(seat);
        let side = |s: Option<usize>| s.map(|who| if who == seat { "you" } else { "them" });
        let phase = if self.status == "waiting_for_opponent" {
            "waiting"
        } else if self.over() {
            "game_over"
        } else if self.turn == Some(seat) {
            "your_turn"
        } else {
            "opponent_turn"
        };
        let deadline = self
            .deadline
            .and_then(DateTime::<Utc>::from_timestamp_millis)
            .map(|dt| {
                dt.to_rfc3339_opts(
                    if dt.timestamp_subsec_millis() == 0 {
                        SecondsFormat::Secs
                    } else {
                        SecondsFormat::Micros
                    },
                    true,
                )
            });
        json!({"game_id":self.id,"version":self.version,"status":self.status,"you":seat,"turn":self.turn,"phase":phase,
            "hand":card_json(rules::cards(v.hand)),"opponent_hand_count":self.hands[1-seat].count_ones(),"deck_count":self.deck.len(),
            "stones":self.stones.iter().enumerate().map(|(i,s)| json!({"index":i,"yours":card_json(s.cards[seat].as_slice().iter().copied()),
                "theirs":card_json(s.cards[1-seat].as_slice().iter().copied()),"claimed_by":side(s.claimed_by),"first_to_complete":side(s.first_to_complete)})).collect::<Vec<_>>(),
            "unseen_cards":card_json(rules::cards(v.unseen)),"turn_deadline":deadline,"winner":side(self.winner),"win_reason":self.win_reason})
    }

    pub fn legal_moves(&self, seat: usize) -> Result<Value, GameError> {
        self.require_turn(seat)?;
        let targets: Vec<usize> = (0..9)
            .filter(|&i| {
                self.stones[i].claimed_by.is_none() && self.stones[i].cards[seat].len() < 3
            })
            .collect();
        let plays = if targets.is_empty() {
            Vec::new()
        } else {
            rules::cards(self.hands[seat])
                .map(|c| json!({"card":Card::from_id(c),"stones":targets}))
                .collect()
        };
        let pool = FULL_POOL & !(table_mask(&self.stones) | self.hands[seat]);
        let claims: Vec<_> = (0..9)
            .filter(|&i| provable(&self.stones, i, seat, pool))
            .collect();
        Ok(json!({"version":self.version,"plays":plays,"claimable":claims}))
    }

    pub fn log_view(&self) -> Value {
        let revealed = if self.over() {
            json!({"initial_hands":self.initial_hands.iter().map(|h| card_json(h.iter().copied())).collect::<Vec<_>>(),"draw_order":card_json(self.draw_order.iter().copied())})
        } else {
            Value::Null
        };
        json!({"game_id":self.id,"status":self.status,"entries":self.log,"revealed":revealed})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn game(auto_claim: bool) -> Game {
        let mut g = Game::new(
            "g_test".into(),
            Options {
                seed: Some(1),
                auto_claim,
                ..Default::default()
            },
            0,
            true,
        );
        g.join(0).unwrap();
        g
    }
    fn req(g: &Game, card: u8, stone: usize, claims: Vec<usize>) -> TurnRequest {
        TurnRequest {
            version: g.version as i64,
            turn_id: None,
            play: Some(Play {
                card: Card::from_id(card),
                stone,
            }),
            claims,
        }
    }
    fn rig(g: &mut Game, cards: &[u8], stone: usize, seat: usize) {
        for &c in cards {
            for h in &mut g.hands {
                *h &= !(1u64 << c);
            }
            g.deck.retain(|&d| d != c);
        }
        g.stones[stone].cards[seat] = Line::from_slice(&cards[..cards.len() - 1]);
        g.hands[seat] |= 1 << cards[cards.len() - 1];
    }
    #[test]
    fn seed_one_views_and_first_draw_match_python() {
        let mut g = game(false);
        assert_eq!(
            g.view(0)["hand"],
            json!([
                {"color":"red","value":3},{"color":"blue","value":1},{"color":"blue","value":3},
                {"color":"green","value":2},{"color":"green","value":8},{"color":"purple","value":3}
            ])
        );
        assert_eq!(g.view(0)["unseen_cards"].as_array().unwrap().len(), 48);
        assert!(g.view(0)["turn_deadline"].is_null());
        assert!(g.log_view()["revealed"].is_null());
        let r = req(&g, 2, 3, vec![]);
        g.play_turn(0, r, 0).unwrap();
        assert_eq!(g.hands[0] & (1 << 8), 1 << 8);
        assert_eq!((g.version, g.turn, g.deck.len()), (2, Some(1), 41));
        assert_eq!(
            g.events[2],
            json!({"version":2,"type":"card_played","seat":0,"card":{"color":"red","value":3},"stone":3})
        );
        assert_eq!(
            g.events[3],
            json!({"version":2,"type":"turn_passed","seat":0})
        );
    }
    #[test]
    fn rejected_claim_preserves_entire_state_then_duplicate_claims_succeed() {
        let mut g = game(false);
        rig(&mut g, &[9, 10, 11], 0, 0);
        let before = (
            g.view(0),
            g.view(1),
            g.log_view(),
            g.events.clone(),
            g.deck.clone(),
        );
        let r = req(&g, 11, 0, vec![0]);
        assert_eq!(g.play_turn(0, r, 0).unwrap_err().code, "CLAIM_NOT_PROVABLE");
        assert_eq!(
            before,
            (
                g.view(0),
                g.view(1),
                g.log_view(),
                g.events.clone(),
                g.deck.clone()
            )
        );
        let mut g = game(false);
        rig(&mut g, &[15, 16, 17], 0, 0);
        let r = req(&g, 17, 0, vec![0, 0]);
        g.play_turn(0, r, 0).unwrap();
        assert_eq!(g.stones[0].claimed_by, Some(0));
        assert_eq!(g.log_view()["entries"][0]["claims"], json!([0]));
    }
    #[test]
    fn auto_claim_wins_before_draw_and_reveals_original_deal() {
        let mut g = game(true);
        g.stones[0].claimed_by = Some(0);
        g.stones[1].claimed_by = Some(0);
        rig(&mut g, &[15, 16, 17], 2, 0);
        let n = g.deck.len();
        let r = req(&g, 17, 2, vec![]);
        g.play_turn(0, r, 0).unwrap();
        assert_eq!(
            (g.status, g.winner, g.win_reason),
            ("finished", Some(0), Some("three_adjacent"))
        );
        assert_eq!(g.deck.len(), n);
        assert_eq!(g.turn, None);
        assert_eq!(g.view(1)["winner"], "them");
        assert_eq!(
            g.log_view()["revealed"]["draw_order"][0],
            json!({"color":"red","value":9})
        );
    }
    #[test]
    fn retries_are_seat_scoped_and_timeout_runs_before_retry() {
        let mut g = game(false);
        g.options.turn_timeout_seconds = Some(5);
        g.restart_clock(0);
        let mut r = req(&g, 2, 0, vec![]);
        r.turn_id = Some(uuid::Uuid::nil());
        g.play_turn(0, r.clone(), 0).unwrap();
        g.play_turn(0, r.clone(), 1).unwrap();
        assert_eq!(g.version, 2);
        assert_eq!(
            g.play_turn(1, r.clone(), 1).unwrap_err().code,
            "STALE_VERSION"
        );
        g.play_turn(0, r, 5000).unwrap();
        assert_eq!((g.version, g.status, g.winner), (3, "forfeit", Some(0)));
        assert!(!g.check_timeout(6000));
    }
    #[test]
    fn stuck_exact_tie_settles_and_empty_game_draws() {
        let mut g = game(false);
        g.hands = [0, 0];
        g.deck.clear();
        g.stones[0].cards = [
            Line::from_slice(&[0, 10, 20]),
            Line::from_slice(&[27, 37, 47]),
        ];
        g.stones[0].first_to_complete = Some(0);
        g.play_turn(
            0,
            TurnRequest {
                version: 1,
                turn_id: None,
                play: None,
                claims: vec![],
            },
            0,
        )
        .unwrap();
        assert_eq!(
            (g.winner, g.win_reason, g.stones[0].claimed_by),
            (Some(0), Some("stalemate"), Some(0))
        );
        let mut g = game(false);
        g.hands = [0, 0];
        g.play_turn(
            0,
            TurnRequest {
                version: 1,
                turn_id: None,
                play: None,
                claims: vec![],
            },
            0,
        )
        .unwrap();
        assert_eq!((g.winner, g.win_reason), (None, Some("stalemate")));
    }
    #[test]
    fn schema_and_direct_call_validation_and_stale_signed_version() {
        assert!(
            Options {
                turn_timeout_seconds: Some(4),
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            serde_json::from_value::<TurnRequest>(
                json!({"version":1,"play":{"card":{"color":"pink","value":1},"stone":0}})
            )
            .is_err()
        );
        for bad in [
            json!({"version":1,"claims":[9]}),
            json!({"version":1,"play":{"card":{"color":"red","value":0},"stone":0}}),
        ] {
            let r: TurnRequest = serde_json::from_value(bad).unwrap();
            assert_eq!(r.validate().unwrap_err().code, "INVALID_REQUEST");
        }
        let mut g = game(false);
        let mut r = req(&g, 2, 0, vec![]);
        r.version = -1;
        assert_eq!(g.play_turn(0, r, 0).unwrap_err().code, "STALE_VERSION");
        assert_eq!(
            g.play_turn(2, req(&g, 2, 0, vec![]), 0).unwrap_err().code,
            "INVALID_REQUEST"
        );
    }
    #[test]
    fn arbitrary_precision_seed_reinitializes_deal_before_join() {
        let mut g = Game::new("large".into(), Options::default(), 0, true);
        g.reseed_words(&[7, 0, 65536]).unwrap();
        assert_eq!(
            g.initial_hands,
            [vec![20, 52, 26, 27, 15, 24], vec![50, 17, 30, 39, 7, 5]]
        );
        assert_eq!(
            g.draw_order,
            vec![
                0, 29, 32, 18, 21, 46, 38, 45, 8, 16, 41, 3, 14, 2, 19, 4, 28, 25, 6, 53, 40, 47,
                36, 10, 37, 43, 12, 34, 51, 44, 11, 48, 13, 22, 35, 49, 23, 9, 31, 42, 33, 1
            ]
        );
        g.join(0).unwrap();
        let before = g.view(0);
        assert_eq!(g.reseed_words(&[1]).unwrap_err().code, "INVALID_REQUEST");
        assert_eq!(g.view(0), before);
    }
    #[test]
    fn valid_claim_followed_by_bad_claim_is_atomic() {
        let mut g = game(false);
        rig(&mut g, &[15, 16, 17], 0, 0);
        let before = (g.view(0), g.view(1), g.log_view(), g.events.clone());
        let r = req(&g, 17, 0, vec![0, 1]);
        assert_eq!(g.play_turn(0, r, 0).unwrap_err().code, "CLAIM_INCOMPLETE");
        assert_eq!(
            before,
            (g.view(0), g.view(1), g.log_view(), g.events.clone())
        );
    }
    #[test]
    fn opponents_hidden_partition_does_not_affect_player_information() {
        let mut g = game(false);
        let before = g.player_view(0);
        let json_before = g.view(0);
        let out = rules::cards(g.hands[1]).next().unwrap();
        let into = g.deck.pop().unwrap();
        g.hands[1] = (g.hands[1] & !(1u64 << out)) | (1u64 << into);
        g.deck.push(out);
        assert_eq!(g.player_view(0), before);
        assert_eq!(g.view(0), json_before);
    }
    #[test]
    fn deadlines_use_python_utc_precision_and_inclusive_boundary() {
        let mut g = Game::new(
            "clock".into(),
            Options {
                seed: Some(1),
                turn_timeout_seconds: Some(5),
                ..Default::default()
            },
            0,
            true,
        );
        g.join(1).unwrap();
        assert_eq!(g.view(0)["turn_deadline"], "1970-01-01T00:00:05.001000Z");
        assert!(!g.check_timeout(5000));
        assert!(g.check_timeout(5001));
        assert_eq!(
            g.events.last().unwrap(),
            &json!({"version":2,"type":"game_finished","winner_seat":1,"win_reason":"forfeit"})
        );
        assert!(g.view(0)["turn_deadline"].is_null());
    }
}
