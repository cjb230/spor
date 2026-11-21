use std::time::Instant;

use schotten_native::{
    bots::Bot,
    engine::{Game, Options, TurnRequest},
    montecarlo::{
        self, McConfig, Rollout,
        obs::{Observation, Side, StoneObs},
    },
    rules,
};

fn observation(game: &Game, seat: usize) -> Observation {
    let view = game.player_view(seat);
    let counts = game.view(seat);
    let side = |value: Option<bool>| value.map(|ours| if ours { Side::You } else { Side::Them });
    Observation {
        you: 0,
        hand: rules::cards(view.hand).collect(),
        unseen: view.unseen,
        deck_count: counts["deck_count"].as_u64().unwrap() as usize,
        opp_hand_count: counts["opponent_hand_count"].as_u64().unwrap() as usize,
        stones: view
            .stones
            .iter()
            .map(|stone| StoneObs {
                yours: stone.yours.as_slice().to_vec(),
                theirs: stone.theirs.as_slice().to_vec(),
                claimed: side(stone.claimed_by),
                first: side(stone.first_to_complete),
            })
            .collect(),
    }
}

fn main() {
    let mut records = Vec::new();
    for seed in [7, 42] {
        let mut game = Game::new(
            "cost".into(),
            Options {
                seed: Some(seed),
                ..Default::default()
            },
            0,
            false,
        );
        game.join(0).unwrap();
        let mut bots = [
            Bot::new("builder", Some(1)).unwrap(),
            Bot::new("greedy", Some(2)).unwrap(),
        ];
        for turn in 0..=28 {
            if game.over() {
                break;
            }
            let seat = game.turn.unwrap();
            if [4, 16, 28].contains(&turn) {
                let observation = observation(&game, seat);
                for repeat in 0..3 {
                    let kinds = if repeat % 2 == 0 {
                        [Rollout::Greedy, Rollout::Builder, Rollout::Planner]
                    } else {
                        [Rollout::Planner, Rollout::Builder, Rollout::Greedy]
                    };
                    for kind in kinds {
                        let started = Instant::now();
                        let decision = montecarlo::decide(
                            &observation,
                            &McConfig {
                                samples: 8,
                                threads: 1,
                                seed: 100 + repeat,
                                rollout: kind,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                        records.push(serde_json::json!({"deck_seed":seed,"turn":turn,"repeat":repeat,"policy":format!("{kind:?}"),"samples":8,"legal_moves":observation.legal_moves().len(),"seconds":started.elapsed().as_secs_f64(),"chosen_move":decision.play}));
                    }
                }
            }
            let view = game.player_view(seat);
            let (play, claims) = bots[seat].choose(&view);
            game.play_turn(
                seat,
                TurnRequest {
                    version: view.version as i64,
                    turn_id: None,
                    play,
                    claims,
                },
                0,
            )
            .unwrap();
        }
    }
    println!("{}", serde_json::to_string_pretty(&records).unwrap());
}
