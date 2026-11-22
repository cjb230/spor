use std::sync::OnceLock;

pub const FULL_POOL: u64 = (1u64 << 54) - 1;
pub type Strength = u16;

pub fn mask(cards: impl IntoIterator<Item = u8>) -> u64 {
    cards.into_iter().fold(0, |m, c| m | (1u64 << c))
}

pub fn cards(pool: u64) -> impl Iterator<Item = u8> {
    let mut left = pool & FULL_POOL;
    std::iter::from_fn(move || {
        if left == 0 {
            None
        } else {
            let c = left.trailing_zeros() as u8;
            left &= left - 1;
            Some(c)
        }
    })
}

struct Tables {
    strengths: Box<[u8]>,
    builder_strengths: Box<[u8]>,
    ranked: Vec<(Strength, u64)>,
}
static TABLES: OnceLock<Tables> = OnceLock::new();

fn combination_index(mut a: usize, mut b: usize, mut c: usize) -> usize {
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    if b > c {
        std::mem::swap(&mut b, &mut c);
    }
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    c * (c - 1) * (c - 2) / 6 + b * (b - 1) / 2 + a
}

fn tables() -> &'static Tables {
    TABLES.get_or_init(|| {
        let mut strengths = vec![0u8; 24804];
        let mut builder_strengths = vec![0u8; 54 * 54 * 54];
        let mut ranked = Vec::with_capacity(24804);
        for a in 0u8..54 {
            for b in a + 1..54 {
                for c in b + 1..54 {
                    let mut vs = [a % 9 + 1, b % 9 + 1, c % 9 + 1];
                    vs.sort_unstable();
                    let color = a / 9 == b / 9 && b / 9 == c / 9;
                    let run = vs[1] == vs[0] + 1 && vs[2] == vs[1] + 1;
                    let formation = if color && run {
                        4
                    } else if vs[0] == vs[2] {
                        3
                    } else if color {
                        2
                    } else if run {
                        1
                    } else {
                        0
                    };
                    let s = formation * 32 + vs.iter().sum::<u8>();
                    strengths[combination_index(a as usize, b as usize, c as usize)] = s;
                    for [x, y, z] in [
                        [a, b, c],
                        [a, c, b],
                        [b, a, c],
                        [b, c, a],
                        [c, a, b],
                        [c, b, a],
                    ] {
                        builder_strengths[(x as usize * 54 + y as usize) * 54 + z as usize] = s;
                    }
                    ranked.push((s as Strength, mask([a, b, c])));
                }
            }
        }
        ranked.sort_unstable_by_key(|a| std::cmp::Reverse(a.0));
        Tables {
            strengths: strengths.into_boxed_slice(),
            builder_strengths: builder_strengths.into_boxed_slice(),
            ranked,
        }
    })
}

pub fn warmup() {
    let _ = tables();
}

pub(crate) fn builder_strengths() -> &'static [u8] {
    &tables().builder_strengths
}

pub fn evaluate(cs: &[u8]) -> Strength {
    assert_eq!(cs.len(), 3, "a formation has exactly three cards");
    debug_assert!(cs.iter().all(|&c| c < 54));
    debug_assert!(cs[0] != cs[1] && cs[1] != cs[2] && cs[0] != cs[2]);
    tables().strengths[combination_index(cs[0] as usize, cs[1] as usize, cs[2] as usize)]
        as Strength
}

pub fn formation_label(s: Strength) -> &'static str {
    ["sum", "run", "color", "three_of_a_kind", "color_run"][(s / 32) as usize]
}

pub fn claim_is_provable(mine: &[u8], theirs: &[u8], pool: u64, tiebreak_mine: bool) -> bool {
    let strength = evaluate(mine);
    claim_is_provable_strength(strength, theirs, pool, tiebreak_mine)
}

/// Internal callers that already evaluated a complete formation can reuse its
/// strength. The proof and exact-tie rules are identical to the public helper.
pub(crate) fn claim_is_provable_strength(
    strength: Strength,
    theirs: &[u8],
    pool: u64,
    tiebreak_mine: bool,
) -> bool {
    let blocks = |s| s > strength || (s == strength && !tiebreak_mine);
    let needed = 3 - theirs.len();
    if needed == 0 {
        return !blocks(evaluate(theirs));
    }
    if (pool & FULL_POOL).count_ones() < needed as u32 {
        return true;
    }
    match needed {
        1 => !cards(pool).any(|c| blocks(evaluate(&[theirs[0], theirs[1], c]))),
        2 => {
            let mut left = pool & FULL_POOL;
            while left != 0 {
                let a = left.trailing_zeros() as u8;
                left &= left - 1;
                if cards(left).any(|b| blocks(evaluate(&[theirs[0], a, b]))) {
                    return false;
                }
            }
            true
        }
        3 => {
            for &(s, m) in &tables().ranked {
                if !blocks(s) {
                    break;
                }
                if m & pool == m {
                    return false;
                }
            }
            true
        }
        _ => unreachable!(),
    }
}

pub fn check_win(claimed: &[Option<usize>], order: [usize; 2]) -> Option<(usize, &'static str)> {
    for seat in order {
        if claimed.iter().filter(|&&c| c == Some(seat)).count() >= 5 {
            return Some((seat, "five_stones"));
        }
        if claimed
            .windows(3)
            .any(|w| w.iter().all(|&c| c == Some(seat)))
        {
            return Some((seat, "three_adjacent"));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_formation_matches_python_fixture_and_is_permutation_invariant() {
        let mut hash = 14695981039346656037u64;
        for a in 0..54 {
            for b in a + 1..54 {
                for c in b + 1..54 {
                    let s = evaluate(&[a, b, c]);
                    assert_eq!(s, evaluate(&[c, a, b]));
                    hash = (hash ^ s as u64).wrapping_mul(1099511628211);
                }
            }
        }
        assert_eq!(hash, 8258880572027140577);
    }
    #[test]
    fn exhaustive_python_formation_and_total_byte_oracle() {
        let mut hash = 14695981039346656037u64;
        for a in 0..54 {
            for b in a + 1..54 {
                for c in b + 1..54 {
                    let strength = evaluate(&[a, b, c]);
                    for byte in [(strength / 32) as u8, (strength % 32) as u8] {
                        hash = (hash ^ byte as u64).wrapping_mul(1099511628211);
                    }
                }
            }
        }
        assert_eq!(hash, 0xebf803366e80f211);
    }
    #[test]
    fn exact_ties_and_own_hand_change_claim_proof() {
        assert!(claim_is_provable(&[0, 10, 20], &[27, 37, 47], 0, true));
        assert!(!claim_is_provable(&[0, 10, 20], &[27, 37, 47], 0, false));
        let mine = [14, 15, 16];
        let theirs = [6, 7];
        let pool = FULL_POOL & !mask(mine.into_iter().chain(theirs));
        assert!(!claim_is_provable(&mine, &theirs, pool, true));
        assert!(claim_is_provable(&mine, &theirs, pool & !(1 << 8), true));
        assert!(claim_is_provable(&[0, 11, 22], &[8], 1 << 53, true));
        assert!(claim_is_provable(
            &[6, 7, 8],
            &[],
            FULL_POOL & !mask([6, 7, 8]),
            true
        ));
    }

    #[test]
    fn precomputed_strength_claims_match_standard_proofs() {
        use crate::montecarlo::rng::Rng;
        let mut rng = Rng::new(719);
        for _ in 0..1000 {
            let mut cards: Vec<u8> = (0..54).collect();
            rng.shuffle(&mut cards);
            let mine = &cards[..3];
            for count in 0..=3 {
                let theirs = &cards[3..3 + count];
                let available = &cards[3 + count..];
                let n = rng.below(available.len() + 1);
                let pool = mask(available[..n].iter().copied());
                for first in [false, true] {
                    let expected = matches!(
                        crate::montecarlo::rules::check_claim(
                            [mine[0], mine[1], mine[2]],
                            theirs,
                            pool,
                            first,
                        ),
                        crate::montecarlo::rules::Proof::Provable
                    );
                    assert_eq!(
                        claim_is_provable_strength(evaluate(mine), theirs, pool, first),
                        expected
                    );
                    assert_eq!(claim_is_provable(mine, theirs, pool, first), expected);
                }
            }
        }
    }

    #[test]
    fn builder_dense_lookup_matches_every_distinct_triple_permutation() {
        let dense = builder_strengths();
        assert_eq!(dense.len(), 54 * 54 * 54);
        for a in 0u8..54 {
            for b in a + 1..54 {
                for c in b + 1..54 {
                    for [x, y, z] in [
                        [a, b, c],
                        [a, c, b],
                        [b, a, c],
                        [b, c, a],
                        [c, a, b],
                        [c, b, a],
                    ] {
                        assert_eq!(
                            dense[(x as usize * 54 + y as usize) * 54 + z as usize] as Strength,
                            evaluate(&[a, b, c])
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn winning_order_and_five_stones_precedence() {
        let claimed = [
            Some(0),
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(1),
            None,
            None,
            None,
        ];
        assert_eq!(check_win(&claimed, [1, 0]), Some((1, "three_adjacent")));
        assert_eq!(check_win(&[Some(0); 9], [0, 1]), Some((0, "five_stones")));
    }
}
