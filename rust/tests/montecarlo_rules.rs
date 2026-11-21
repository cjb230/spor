//! Check the closed-form "best completion" and the claim proofs against plain enumeration,
//! on random inputs, independently of the Python fixtures.

use schotten_native::montecarlo::cards::{self, ALL, CardSet, bit};
use schotten_native::montecarlo::rng::Rng;
use schotten_native::montecarlo::rules::{self, Proof};

fn random_subset(rng: &mut Rng, universe: CardSet, size: usize) -> CardSet {
    let mut cards: Vec<u8> = cards::iter(universe).collect();
    rng.shuffle(&mut cards);
    cards.iter().take(size).fold(0, |m, &c| m | bit(c))
}

fn brute_best(partial: &[u8], pool: CardSet) -> Option<u8> {
    let cards: Vec<u8> = cards::iter(pool).collect();
    let mut best: Option<u8> = None;
    let mut take = |s: u8| best = Some(best.map_or(s, |b| b.max(s)));
    match partial.len() {
        3 => take(rules::evaluate([partial[0], partial[1], partial[2]])),
        2 => cards
            .iter()
            .for_each(|&x| take(rules::evaluate([partial[0], partial[1], x]))),
        1 => {
            for i in 0..cards.len() {
                for j in i + 1..cards.len() {
                    take(rules::evaluate([partial[0], cards[i], cards[j]]));
                }
            }
        }
        _ => {
            for i in 0..cards.len() {
                for j in i + 1..cards.len() {
                    for k in j + 1..cards.len() {
                        take(rules::evaluate([cards[i], cards[j], cards[k]]));
                    }
                }
            }
        }
    }
    best
}

#[test]
fn closed_form_best_triple_matches_enumeration() {
    let mut rng = Rng::new(11);
    for round in 0..4000 {
        // mostly small pools (where the weaker formations decide), some large ones
        let size = if round % 10 == 0 {
            3 + rng.below(40)
        } else {
            3 + rng.below(14)
        };
        let pool = random_subset(&mut rng, ALL, size);
        let (got, witness) = rules::best_completion(&[], pool, u8::MAX).unwrap();
        assert_eq!(
            Some(got),
            brute_best(&[], pool),
            "pool {:?}",
            cards::iter(pool).collect::<Vec<_>>()
        );
        // the witness really is three pool cards that make that formation
        let w: Vec<u8> = cards::iter(witness).collect();
        assert_eq!(w.len(), 3);
        assert_eq!(witness & !pool, 0);
        assert_eq!(rules::evaluate([w[0], w[1], w[2]]), got);
    }
}

#[test]
fn best_completion_with_cards_already_down_matches_enumeration() {
    let mut rng = Rng::new(12);
    for _ in 0..3000 {
        let k = rng.below(4);
        let have = random_subset(&mut rng, ALL, k);
        let size = rng.below(20);
        let pool = random_subset(&mut rng, ALL & !have, size);
        let partial: Vec<u8> = cards::iter(have).collect();
        let got = rules::best_completion(&partial, pool, u8::MAX).map(|(s, _)| s);
        assert_eq!(
            got,
            brute_best(&partial, pool),
            "partial {partial:?} pool size {size}"
        );
    }
}

fn reference_provable(mine: [u8; 3], theirs: &[u8], pool: CardSet, tiebreak: bool) -> bool {
    let mine_s = rules::evaluate(mine);
    match brute_best(theirs, pool) {
        None => true,
        Some(best) => best < mine_s || (best == mine_s && tiebreak),
    }
}

#[test]
fn claim_proofs_match_enumeration() {
    let mut rng = Rng::new(13);
    let (mut yes, mut no) = (0, 0);
    for _ in 0..6000 {
        let mine_set = random_subset(&mut rng, ALL, 3);
        let mine: Vec<u8> = cards::iter(mine_set).collect();
        let k = rng.below(4);
        let theirs_set = random_subset(&mut rng, ALL & !mine_set, k);
        let theirs: Vec<u8> = cards::iter(theirs_set).collect();
        let size = rng.below(25);
        let pool = random_subset(&mut rng, ALL & !mine_set & !theirs_set, size);
        let tiebreak = rng.below(2) == 0;
        let want = reference_provable([mine[0], mine[1], mine[2]], &theirs, pool, tiebreak);
        let got = rules::check_claim([mine[0], mine[1], mine[2]], &theirs, pool, tiebreak);
        assert_eq!(
            matches!(got, Proof::Provable),
            want,
            "mine {mine:?} theirs {theirs:?} tiebreak {tiebreak}"
        );
        if let Proof::Blocked(w) = got {
            // a blocked claim names cards that are really in the pool
            assert_eq!(w & !pool, 0);
        }
        if want { yes += 1 } else { no += 1 }
    }
    assert!(yes > 300 && no > 300, "{yes} provable, {no} blocked");
}

/// The Monte Carlo rules are a separate implementation for speed; they must agree with the
/// native engine's rules on formation strength and on claim proofs.
#[test]
fn montecarlo_rules_agree_with_native_engine_rules() {
    use schotten_native::rules as native;
    let mut rng = Rng::new(21);
    let (mut yes, mut no) = (0, 0);
    for _ in 0..20000 {
        let mine_set = random_subset(&mut rng, ALL, 3);
        let mine: Vec<u8> = cards::iter(mine_set).collect();
        let k = rng.below(4);
        let theirs_set = random_subset(&mut rng, ALL & !mine_set, k);
        let theirs: Vec<u8> = cards::iter(theirs_set).collect();
        let size = rng.below(25);
        let pool = random_subset(&mut rng, ALL & !mine_set & !theirs_set, size);
        let tiebreak = rng.below(2) == 0;
        let ours = matches!(
            rules::check_claim([mine[0], mine[1], mine[2]], &theirs, pool, tiebreak),
            Proof::Provable
        );
        assert_eq!(
            ours,
            native::claim_is_provable(&mine, &theirs, pool, tiebreak),
            "mine {mine:?} theirs {theirs:?} tiebreak {tiebreak}"
        );
        if ours {
            yes += 1
        } else {
            no += 1
        }
        // strength ordering must match too (compare through both orderings)
        let other = random_subset(&mut rng, ALL & !mine_set, 3);
        let o: Vec<u8> = cards::iter(other).collect();
        let a =
            rules::evaluate([mine[0], mine[1], mine[2]]).cmp(&rules::evaluate([o[0], o[1], o[2]]));
        let b = native::evaluate(&mine).cmp(&native::evaluate(&o));
        assert_eq!(a, b, "{mine:?} vs {o:?}");
    }
    assert!(yes > 1000 && no > 1000, "{yes}/{no}");
}
