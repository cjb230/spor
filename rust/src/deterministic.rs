use crate::bots::Bot;
use crate::engine::{Card, Line, Play, PlayerView, ViewStone};
use crate::rules::{self, Strength};
use std::collections::HashMap;

const TERMINAL_VALUE: i64 = 1_000_000;

#[derive(Clone, Copy)]
pub enum Mode {
    Builder,
    Tactician,
    Planner,
}

#[derive(Clone, Copy, Default)]
struct Profile {
    value: i64,
    control: i64,
    owner: Option<bool>,
}

struct Candidate {
    play: Play,
    stones: [ViewStone; 9],
    hand: u64,
    claims: u32,
    winning: bool,
    score: i64,
}

#[derive(Default)]
struct Evaluator {
    completions: HashMap<(u64, u64), Option<Strength>>,
}

impl Evaluator {
    fn worst_reply(&mut self, view: &PlayerView, candidate: &Candidate) -> i64 {
        if candidate.winning {
            return TERMINAL_VALUE;
        }
        let base = candidate
            .stones
            .map(|s| self.profile(s, candidate.hand, view.unseen));
        let mut worst = i64::MAX;
        for card in rules::cards(reply_cards(view, candidate)) {
            let unseen = view.unseen & !(1u64 << card);
            let mut profiles = base;
            for (i, stone) in candidate.stones.iter().enumerate() {
                if stone.theirs.len() == 3 && base[i].owner.is_none() {
                    profiles[i] = self.profile(*stone, candidate.hand, unseen);
                }
            }
            for (i, &stone) in candidate.stones.iter().enumerate() {
                if stone.claimed_by.is_some() || stone.theirs.len() == 3 {
                    continue;
                }
                let mut reply = stone;
                reply.theirs.push(card);
                if reply.theirs.len() == 3 && reply.first_to_complete.is_none() {
                    reply.first_to_complete = Some(false);
                }
                let mut result = profiles;
                result[i] = self.profile(reply, candidate.hand, unseen);
                worst = worst.min(board_value(&result));
            }
        }
        if worst == i64::MAX {
            candidate.score
        } else {
            worst
        }
    }

    fn completion(&mut self, line: Line, hand: u64) -> Option<Strength> {
        let key = (rules::mask(line.as_slice().iter().copied()), hand);
        if let Some(&strength) = self.completions.get(&key) {
            return strength;
        }
        let strength = match line.len() {
            3 => Some(rules::evaluate(line.as_slice())),
            2 => rules::cards(hand)
                .map(|c| rules::evaluate(&[line.as_slice()[0], line.as_slice()[1], c]))
                .max(),
            1 => {
                let mut best = None;
                let mut left = hand;
                while left != 0 {
                    let a = left.trailing_zeros() as u8;
                    left &= left - 1;
                    for b in rules::cards(left) {
                        best = best.max(Some(rules::evaluate(&[line.as_slice()[0], a, b])));
                    }
                }
                best
            }
            _ => None,
        };
        self.completions.insert(key, strength);
        strength
    }

    fn own_value(&mut self, stone: ViewStone, hand: u64) -> i64 {
        if stone.yours.is_empty() {
            return 0;
        }
        let strength = self.completion(stone.yours, hand);
        let mut value = strength.map_or_else(
            || public_value(stone.yours),
            |s| 8 * i64::from(s) + 100 * stone.yours.len() as i64,
        );
        if stone.yours.len() == 3 && stone.theirs.len() == 3 {
            let mine = rules::evaluate(stone.yours.as_slice());
            let theirs = rules::evaluate(stone.theirs.as_slice());
            if mine < theirs || (mine == theirs && stone.first_to_complete != Some(true)) {
                value -= 4_000;
            }
        }
        if stone.theirs.len() == 3 && stone.yours.len() < 3 {
            let theirs = rules::evaluate(stone.theirs.as_slice());
            if strength.is_some_and(|s| s <= theirs) {
                value -= 1_200;
            }
        }
        value
    }

    fn profile(&mut self, stone: ViewStone, hand: u64, unseen: u64) -> Profile {
        let owner = stone.claimed_by.or_else(|| {
            (stone.theirs.len() == 3
                && rules::claim_is_provable(
                    stone.theirs.as_slice(),
                    stone.yours.as_slice(),
                    hand | unseen,
                    stone.first_to_complete == Some(false),
                ))
            .then_some(false)
        });
        if let Some(ours) = owner {
            return Profile {
                value: if ours { 6_000 } else { -6_000 },
                control: if ours { 1_000 } else { -1_000 },
                owner,
            };
        }
        let own = self.own_value(stone, hand);
        let enemy = public_value(stone.theirs);
        let control = ((own - enemy) / 3).clamp(-850, 850);
        Profile {
            value: own - enemy,
            control,
            owner,
        }
    }

    fn candidates(&mut self, view: &PlayerView, mode: Mode) -> Vec<Candidate> {
        let base_claims = Bot::provable_stones(view, None);
        let mut candidates = Vec::new();
        for stone in 0..9 {
            if view.stones[stone].claimed_by.is_some() || view.stones[stone].yours.len() == 3 {
                continue;
            }
            for card in rules::cards(view.hand) {
                let mut stones = view.stones;
                let hand = view.hand & !(1u64 << card);
                stones[stone].yours.push(card);
                if stones[stone].yours.len() == 3 && stones[stone].first_to_complete.is_none() {
                    stones[stone].first_to_complete = Some(true);
                }
                for &idx in &base_claims {
                    stones[idx].claimed_by = Some(true);
                }
                let target = stones[stone];
                let new_claim = target.yours.len() == 3
                    && rules::claim_is_provable(
                        target.yours.as_slice(),
                        target.theirs.as_slice(),
                        view.unseen,
                        target.first_to_complete == Some(true),
                    );
                if new_claim {
                    stones[stone].claimed_by = Some(true);
                }
                let claims = base_claims.len() as u32 + u32::from(new_claim);
                let winning = won(&stones, true);
                let score = if matches!(mode, Mode::Builder) {
                    self.own_value(stones[stone], hand)
                } else {
                    let profiles = stones.map(|s| self.profile(s, hand, view.unseen));
                    board_value(&profiles)
                };
                candidates.push(Candidate {
                    play: Play {
                        card: Card::from_id(card),
                        stone,
                    },
                    stones,
                    hand,
                    claims,
                    winning,
                    score,
                });
            }
        }
        candidates.sort_by(|a, b| {
            (b.winning, b.claims, b.score)
                .cmp(&(a.winning, a.claims, a.score))
                .then_with(|| {
                    (a.play.stone, a.play.card.id()).cmp(&(b.play.stone, b.play.card.id()))
                })
        });
        candidates
    }
}

fn public_value(line: Line) -> i64 {
    match line.as_slice() {
        [] => 0,
        [a] => 100 + i64::from(a % 9 + 1) * 8,
        [a, b] => {
            let gap = (i16::from(a % 9) - i16::from(b % 9)).abs();
            let same = a / 9 == b / 9;
            let shape = match (same, gap) {
                (true, 1 | 2) => 700,
                (_, 0) => 600,
                (true, _) => 420,
                (_, 1 | 2) => 260,
                _ => 0,
            };
            200 + shape + i64::from(a % 9 + b % 9 + 2) * 8
        }
        _ => 300 + 8 * i64::from(rules::evaluate(line.as_slice())),
    }
}

fn won(stones: &[ViewStone; 9], ours: bool) -> bool {
    let claimed = stones.map(|s| s.claimed_by.map(|who| usize::from(!who)));
    rules::check_win(&claimed, [0, 1]).is_some_and(|(seat, _)| seat == usize::from(!ours))
}

fn board_value(profiles: &[Profile; 9]) -> i64 {
    let owners = profiles.map(|s| s.owner.map(|who| usize::from(!who)));
    if let Some((seat, _)) = rules::check_win(&owners, [0, 1]) {
        return if seat == 0 {
            TERMINAL_VALUE
        } else {
            -TERMINAL_VALUE
        };
    }
    let mut value = profiles.iter().map(|p| p.value).sum::<i64>();
    let own_claims = profiles.iter().filter(|p| p.owner == Some(true)).count() as i64;
    let enemy_claims = profiles.iter().filter(|p| p.owner == Some(false)).count() as i64;
    for p in profiles {
        if p.owner.is_none() {
            value += p.control * (own_claims * own_claims + enemy_claims * enemy_claims) / 2;
        }
    }
    for route in profiles.windows(3) {
        for ours in [true, false] {
            if route.iter().any(|p| p.owner == Some(!ours)) {
                continue;
            }
            let held = route.iter().filter(|p| p.owner == Some(ours)).count();
            let weight = [1, 3, 12][held.min(2)];
            let control = route
                .iter()
                .filter(|p| p.owner.is_none())
                .map(|p| p.control)
                .sum::<i64>();
            value += control * weight;
        }
    }
    value
}

fn reply_cards(view: &PlayerView, candidate: &Candidate) -> u64 {
    let public = rules::mask(candidate.stones.iter().flat_map(|s| {
        s.yours
            .as_slice()
            .iter()
            .chain(s.theirs.as_slice())
            .copied()
    }));
    view.unseen & !view.hand & !public
}

pub fn choose(view: &PlayerView, mode: Mode) -> Option<Play> {
    let mut evaluator = Evaluator::default();
    let mut candidates = evaluator.candidates(view, mode);
    if !matches!(mode, Mode::Planner) || candidates.first().is_none_or(|c| c.winning) {
        return candidates.into_iter().next().map(|c| c.play);
    }
    candidates.truncate(8);
    let mut best = None;
    for candidate in candidates {
        let value = evaluator.worst_reply(view, &candidate);
        let rank = (candidate.winning, value, candidate.claims);
        let tie = (candidate.play.stone, candidate.play.card.id());
        if best.as_ref().is_none_or(|(old_rank, old_tie, _)| {
            rank > *old_rank || (rank == *old_rank && tie < *old_tie)
        }) {
            best = Some((rank, tie, candidate.play));
        }
    }
    best.map(|(_, _, play)| play)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::Bot;
    use crate::engine::{Card, Line, ViewStone};
    use crate::rules::{self, FULL_POOL};

    fn position(hand: &[u8]) -> PlayerView {
        PlayerView {
            version: 0,
            hand: rules::mask(hand.iter().copied()),
            unseen: FULL_POOL & !rules::mask(hand.iter().copied()),
            stones: [ViewStone::default(); 9],
        }
    }

    fn refresh(view: &mut PlayerView) {
        let public = rules::mask(view.stones.iter().flat_map(|s| {
            s.yours
                .as_slice()
                .iter()
                .chain(s.theirs.as_slice())
                .copied()
        }));
        assert_eq!(view.hand & public, 0);
        view.unseen = FULL_POOL & !(view.hand | public);
    }

    fn play(stone: usize, card: u8) -> Play {
        Play {
            stone,
            card: Card::from_id(card),
        }
    }

    #[test]
    fn builder_uses_a_combination_supported_by_the_remaining_hand() {
        let view = position(&[0, 1, 2, 8, 17, 35]);
        assert_eq!(choose(&view, Mode::Builder), Some(play(0, 0)));
    }

    #[test]
    fn exact_tie_loses_to_the_opponents_first_completion() {
        let mut view = position(&[20]);
        view.stones[0].yours = Line::from_slice(&[0, 10]);
        view.stones[0].theirs = Line::from_slice(&[27, 37, 47]);
        view.stones[0].first_to_complete = Some(false);
        refresh(&mut view);
        assert!(Bot::provable_stones(&view, Some(&play(0, 20))).is_empty());
        for mode in [Mode::Builder, Mode::Tactician, Mode::Planner] {
            assert_ne!(choose(&view, mode).unwrap().stone, 0);
        }
    }

    #[test]
    fn immediate_adjacent_match_win_precedes_an_unrelated_secure_claim() {
        let mut view = position(&[8, 26]);
        view.stones[0].claimed_by = Some(true);
        view.stones[1].claimed_by = Some(true);
        view.stones[2].yours = Line::from_slice(&[6, 7]);
        view.stones[8].yours = Line::from_slice(&[24, 25]);
        refresh(&mut view);
        for mode in [Mode::Builder, Mode::Tactician, Mode::Planner] {
            assert_eq!(choose(&view, mode), Some(play(2, 8)));
        }
    }

    #[test]
    fn tactician_contests_an_opposing_adjacent_route() {
        let mut view = position(&[2, 11]);
        view.stones[7].claimed_by = Some(false);
        view.stones[8].claimed_by = Some(false);
        view.stones[2].yours = Line::from_slice(&[0]);
        view.stones[6].theirs = Line::from_slice(&[27, 28]);
        view.stones[6].yours = Line::from_slice(&[9]);
        refresh(&mut view);
        assert_eq!(choose(&view, Mode::Builder).unwrap().stone, 2);
        assert_eq!(choose(&view, Mode::Tactician).unwrap().stone, 6);
    }

    #[test]
    fn tactician_preserves_the_only_card_that_completes_another_stone() {
        let mut view = position(&[2, 11, 20]);
        view.stones[0].yours = Line::from_slice(&[0, 1]);
        view.stones[8].yours = Line::from_slice(&[19]);
        view.stones[6].claimed_by = Some(false);
        view.stones[7].claimed_by = Some(false);
        refresh(&mut view);
        let candidates = Evaluator::default().candidates(&view, Mode::Tactician);
        let score = |p| candidates.iter().find(|c| c.play == p).unwrap().score;
        assert!(score(play(8, 11)) > score(play(8, 2)));
        assert_eq!(choose(&view, Mode::Tactician).unwrap().card.id(), 11);
    }

    #[test]
    fn policies_have_fixed_low_stone_then_card_ties() {
        let mut view = position(&[0, 9]);
        for (index, stone) in view.stones.iter_mut().enumerate().take(8).skip(1) {
            stone.claimed_by = Some(index.is_multiple_of(2));
        }
        for mode in [Mode::Builder, Mode::Tactician, Mode::Planner] {
            let chosen = choose(&view, mode);
            assert_eq!(chosen, Some(play(0, 0)));
            assert_eq!(chosen, choose(&view, mode));
        }
    }

    #[test]
    fn no_legal_placement_returns_none() {
        for mode in [Mode::Builder, Mode::Tactician, Mode::Planner] {
            assert_eq!(choose(&position(&[]), mode), None);
            let mut view = position(&[0]);
            for stone in &mut view.stones {
                stone.claimed_by = Some(true);
            }
            assert_eq!(choose(&view, mode), None);
        }
    }

    #[test]
    fn planner_replies_exclude_known_hand_and_public_cards() {
        let mut view = position(&[0, 9]);
        view.stones[3].theirs = Line::from_slice(&[1, 2]);
        refresh(&mut view);
        let candidate = Evaluator::default()
            .candidates(&view, Mode::Tactician)
            .remove(0);
        let replies = reply_cards(&view, &candidate);
        assert_ne!(replies, 0);
        assert_eq!(replies & view.hand, 0);
        assert_eq!(replies & rules::mask([1, 2]), 0);
        assert_eq!(replies, view.unseen);
    }

    #[test]
    fn planner_matches_a_bounded_exhaustive_single_reply_oracle() {
        let mut view = position(&[1, 2, 11]);
        for (index, stone) in view.stones.iter_mut().enumerate().take(8).skip(1) {
            stone.claimed_by = Some(!index.is_multiple_of(2));
        }
        view.stones[4].claimed_by = None;
        view.stones[5].claimed_by = Some(false);
        view.stones[6].claimed_by = Some(true);
        view.stones[7].claimed_by = Some(false);
        view.stones[0].yours = Line::from_slice(&[0]);
        view.stones[4].yours = Line::from_slice(&[9]);
        view.stones[8].yours = Line::from_slice(&[18]);
        view.stones[0].theirs = Line::from_slice(&[27, 28]);
        view.stones[8].theirs = Line::from_slice(&[39, 40]);
        refresh(&mut view);
        view.unseen = rules::mask([29, 38, 47]);
        let mut evaluator = Evaluator::default();
        let candidates = evaluator.candidates(&view, Mode::Tactician);
        assert_eq!(candidates.len(), 9);
        let mut oracle = Vec::new();
        for candidate in candidates.into_iter().take(8) {
            let mut worst = i64::MAX;
            for card in rules::cards(view.unseen) {
                for stone in [0, 4, 8] {
                    if candidate.stones[stone].claimed_by.is_some() {
                        continue;
                    }
                    let mut stones = candidate.stones;
                    stones[stone].theirs.push(card);
                    if stones[stone].first_to_complete.is_none() {
                        stones[stone].first_to_complete = Some(false);
                    }
                    let unseen = view.unseen & !(1u64 << card);
                    let profiles = stones.map(|s| evaluator.profile(s, candidate.hand, unseen));
                    worst = worst.min(board_value(&profiles));
                }
            }
            assert_eq!(evaluator.worst_reply(&view, &candidate), worst);
            oracle.push((candidate, worst));
        }
        assert!(oracle.iter().any(|(_, score)| *score != oracle[0].1));
        assert!(oracle.iter().all(|(_, score)| score.abs() < 1_000_000));
        oracle.sort_by(|(a, sa), (b, sb)| {
            (b.winning, sb, b.claims)
                .cmp(&(a.winning, sa, a.claims))
                .then_with(|| {
                    (a.play.stone, a.play.card.id()).cmp(&(b.play.stone, b.play.card.id()))
                })
        });
        assert_eq!(choose(&view, Mode::Planner), Some(oracle[0].0.play.clone()));
        assert_eq!(choose(&view, Mode::Planner), choose(&view, Mode::Planner));
    }

    #[test]
    fn enemy_claim_proof_keeps_our_known_completion_card_in_its_pool() {
        let mut evaluator = Evaluator::default();
        let stone = ViewStone {
            yours: Line::from_slice(&[6, 7]),
            theirs: Line::from_slice(&[14, 15, 16]),
            first_to_complete: Some(false),
            ..Default::default()
        };
        let hand = rules::mask([8]);
        let unseen = rules::mask([0, 1]);
        assert!(rules::claim_is_provable(
            stone.theirs.as_slice(),
            stone.yours.as_slice(),
            unseen,
            true,
        ));
        assert_eq!(evaluator.profile(stone, hand, unseen).owner, None);
        assert_eq!(evaluator.profile(stone, 0, unseen).owner, Some(false));
    }

    #[test]
    fn planner_maximizes_nonterminal_reply_value_before_claim_count() {
        let mut view = position(&[6, 7, 8]);
        view.unseen = rules::mask([0, 9, 18]);
        for index in [0, 2, 4] {
            view.stones[index].claimed_by = Some(false);
        }
        view.stones[5].theirs = Line::from_slice(&[14, 15, 16]);
        view.stones[5].first_to_complete = Some(false);
        view.stones[8].yours = Line::from_slice(&[17, 26]);
        let mut evaluator = Evaluator::default();
        let risky = evaluator.candidates(&view, Mode::Tactician).remove(0);
        assert_eq!(risky.play.stone, 8);
        let risky_value = evaluator.worst_reply(&view, &risky);
        assert!(risky_value > -TERMINAL_VALUE);
        let chosen = choose(&view, Mode::Planner).unwrap();
        let defended = evaluator
            .candidates(&view, Mode::Tactician)
            .into_iter()
            .find(|candidate| candidate.play == chosen)
            .unwrap();
        assert!(evaluator.worst_reply(&view, &defended) > risky_value);
    }

    #[test]
    fn first_completion_wins_an_exact_tie_in_public_board_evaluation() {
        let mut evaluator = Evaluator::default();
        let mut stone = ViewStone {
            yours: Line::from_slice(&[0, 10, 20]),
            theirs: Line::from_slice(&[27, 37, 47]),
            first_to_complete: Some(true),
            ..Default::default()
        };
        assert_eq!(evaluator.profile(stone, 0, 0).owner, None);
        stone.first_to_complete = Some(false);
        assert_eq!(evaluator.profile(stone, 0, 0).owner, Some(false));
    }
}
