//! Cards as small integers, sets of cards as 54-bit masks.
//!
//! Card index = colour * 9 + (value - 1). Colour order matches the Python
//! server's `Color` enum: red, blue, green, yellow, purple, brown.

pub type Card = u8;
pub type CardSet = u64;

pub const NUM_CARDS: usize = 54;
pub const ALL: CardSet = (1u64 << NUM_CARDS) - 1;
pub const COLORS: [&str; 6] = ["red", "blue", "green", "yellow", "purple", "brown"];

#[inline]
pub fn make(color: u8, value: u8) -> Card {
    debug_assert!(color < 6 && (1..=9).contains(&value));
    color * 9 + (value - 1)
}

#[inline]
pub fn color(c: Card) -> u8 {
    c / 9
}

#[inline]
pub fn value(c: Card) -> u8 {
    c % 9 + 1
}

#[inline]
pub fn bit(c: Card) -> CardSet {
    1u64 << c
}

/// Iterate the cards in a set, lowest index first.
pub fn iter(mut set: CardSet) -> impl Iterator<Item = Card> {
    std::iter::from_fn(move || {
        if set == 0 {
            None
        } else {
            let c = set.trailing_zeros() as Card;
            set &= set - 1;
            Some(c)
        }
    })
}

pub fn from_name(color: &str, value: u8) -> Option<Card> {
    let c = COLORS.iter().position(|n| *n == color)? as u8;
    (1..=9).contains(&value).then(|| make(c, value))
}

pub fn color_name(c: Card) -> &'static str {
    COLORS[color(c) as usize]
}

/// Short form such as "R7" (brown is N), matching the terminal client.
pub fn short(c: Card) -> String {
    format!(
        "{}{}",
        ["R", "B", "G", "Y", "P", "N"][color(c) as usize],
        value(c)
    )
}
