//! Monte Carlo player: samples possible hidden worlds and plays each legal move out.
//!
//! Self-contained (own card bitmasks, rules, RNG and simulator) so rollouts stay fast. Card ids
//! and rules match the native engine; `tests/montecarlo.rs` checks claim proofs against
//! `crate::rules` and replays games through the native engine.

#![allow(
    clippy::needless_range_loop,
    clippy::unnecessary_map_or,
    clippy::collapsible_if
)]

pub mod cards;
pub mod mc;
pub mod obs;
pub mod policy;
pub mod rng;
pub mod rules;
pub mod sim;

pub use mc::{McConfig, decide};
pub use policy::Rollout;
