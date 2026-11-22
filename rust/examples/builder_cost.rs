//! Repeatable Builder-only benchmark on seeded legal game observations.
use schotten_native::{
    bots::Bot,
    deterministic::{self, Mode},
    engine::{Game, Options, PlayerView, TurnRequest},
    rules,
};
use std::{hint::black_box, time::Instant};

fn byte(hash: &mut u64, value: u8) {
    *hash = (*hash ^ u64::from(value)).wrapping_mul(1099511628211);
}

fn fingerprint(views: &[PlayerView]) -> u64 {
    let mut hash = 14695981039346656037;
    for view in views {
        for value in view
            .hand
            .to_le_bytes()
            .into_iter()
            .chain(view.unseen.to_le_bytes())
        {
            byte(&mut hash, value);
        }
        for stone in &view.stones {
            for line in [stone.yours, stone.theirs] {
                byte(&mut hash, line.len() as u8);
                for &value in line.as_slice() {
                    byte(&mut hash, value);
                }
            }
            for flag in [stone.claimed_by, stone.first_to_complete] {
                byte(&mut hash, flag.map_or(0, |ours| if ours { 1 } else { 2 }));
            }
        }
    }
    hash
}

fn main() {
    let repeats: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("positive repeats"))
        .unwrap_or(128);
    assert!(repeats > 0);
    rules::warmup();
    let mut views = Vec::new();
    for seed in 1..=32 {
        let mut game = Game::new(
            format!("builder-cost-{seed}"),
            Options {
                seed: Some(seed),
                ..Default::default()
            },
            0,
            false,
        );
        game.join(0).unwrap();
        let mut bots = [
            Bot::new("builder", Some(seed)).unwrap(),
            Bot::new(
                if seed % 2 == 0 { "random" } else { "greedy" },
                Some(seed + 1000),
            )
            .unwrap(),
        ];
        while !game.over() {
            let seat = game.turn.unwrap();
            let view = game.player_view(seat);
            views.push(view.clone());
            let (play, claims) = bots[seat].choose(&view);
            game.play_turn(
                seat,
                TurnRequest {
                    version: game.version as i64,
                    turn_id: None,
                    play,
                    claims,
                },
                0,
            )
            .unwrap();
        }
    }
    let positions_hash = fingerprint(&views);
    let mut choices_hash = 14695981039346656037u64;
    let start = Instant::now();
    for _ in 0..repeats {
        for view in &views {
            let play = black_box(deterministic::choose(black_box(view), Mode::Builder));
            let label = play.map_or(486, |p| p.card.id() as usize * 9 + p.stone);
            for value in (label as u16).to_le_bytes() {
                byte(&mut choices_hash, value);
            }
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let decisions = views.len() * repeats;
    println!(
        "{}",
        serde_json::json!({"positions":views.len(),"repeats":repeats,
        "decisions":decisions,"seconds":seconds,"ns_per_decision":seconds*1e9/decisions as f64,
        "positions_fnv64":positions_hash.to_string(),"choices_fnv64":choices_hash.to_string()})
    );
}
