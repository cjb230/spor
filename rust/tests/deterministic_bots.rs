use schotten_native::bots::Bot;
use schotten_native::engine::{Game, Line, Options, PlayerView, TurnRequest};
use schotten_native::rules;

const NAMES: [&str; 3] = ["builder", "tactician", "planner"];

#[test]
fn deterministic_strategies_ignore_seed_and_prior_rng_state() {
    let mut game = Game::new(
        "test".into(),
        Options {
            seed: Some(7),
            ..Default::default()
        },
        0,
        false,
    );
    game.join(0).unwrap();
    for name in NAMES {
        let view = game.player_view(0);
        let expected = Bot::new(name, Some(0)).unwrap().choose(&view);
        for seed in [None, Some(1), Some(-1), Some(i64::MAX)] {
            let mut bot = Bot::new(name, seed).unwrap();
            for _ in 0..3 {
                assert_eq!(bot.choose(&view), expected, "{name} seed={seed:?}");
            }
        }
    }
}

#[test]
fn all_new_strategies_take_an_immediate_winning_claim() {
    let mut view = PlayerView {
        version: 1,
        hand: rules::mask([17, 53]),
        unseen: 0,
        stones: Default::default(),
    };
    view.stones[0].claimed_by = Some(true);
    view.stones[1].claimed_by = Some(true);
    view.stones[2].yours = Line::from_slice(&[15, 16]);
    view.unseen = rules::FULL_POOL & !rules::mask([15, 16, 17, 53]);
    for name in NAMES {
        let (play, claims) = Bot::new(name, None).unwrap().choose(&view);
        let play = play.unwrap();
        assert_eq!((play.stone, play.card.id()), (2, 17), "{name}");
        assert!(claims.contains(&2));
    }
}

#[test]
fn all_new_strategies_pass_and_still_claim_when_no_card_can_be_played() {
    let mut view = PlayerView {
        version: 1,
        hand: 0,
        unseen: 0,
        stones: Default::default(),
    };
    view.stones[4].yours = Line::from_slice(&[6, 7, 8]);
    view.stones[4].first_to_complete = Some(true);
    for name in NAMES {
        let (play, claims) = Bot::new(name, None).unwrap().choose(&view);
        assert!(play.is_none());
        assert_eq!(claims, vec![4]);
    }
}

#[test]
fn deterministic_strategies_finish_legal_games_against_greedy() {
    for name in NAMES {
        for seed in [0, 7] {
            let mut game = Game::new(
                "test".into(),
                Options {
                    seed: Some(seed),
                    ..Default::default()
                },
                0,
                false,
            );
            game.join(0).unwrap();
            let mut bots = [
                Bot::new(name, None).unwrap(),
                Bot::new("greedy", Some(seed)).unwrap(),
            ];
            for _ in 0..128 {
                if game.over() {
                    break;
                }
                let seat = game.turn.unwrap();
                let view = game.player_view(seat);
                let (play, claims) = bots[seat].choose(&view);
                let legal = game.legal_moves(seat).unwrap();
                if let Some(play) = &play {
                    assert_ne!(view.hand & (1 << play.card.id()), 0);
                    assert!(view.stones[play.stone].claimed_by.is_none());
                    assert!(view.stones[play.stone].yours.len() < 3);
                } else {
                    assert!(legal["plays"].as_array().unwrap().is_empty());
                }
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
            assert!(game.over(), "{name} seed={seed}");
        }
    }
}

#[test]
fn planner_defends_against_a_fifth_stone_instead_of_taking_a_losing_claim() {
    let mut view = PlayerView {
        version: 1,
        hand: rules::mask([6, 7, 8]),
        unseen: rules::mask([0, 9, 18]),
        stones: Default::default(),
    };
    for index in [0, 2, 4, 6] {
        view.stones[index].claimed_by = Some(false);
    }
    view.stones[5].theirs = Line::from_slice(&[14, 15, 16]);
    view.stones[5].first_to_complete = Some(false);
    view.stones[8].yours = Line::from_slice(&[17, 26]);
    assert!(rules::claim_is_provable(
        &[14, 15, 16],
        &[],
        (view.hand & !(1 << 8)) | view.unseen,
        true
    ));
    let risky = Bot::new("tactician", None).unwrap().choose(&view);
    assert_eq!(risky.0.unwrap().stone, 8);
    assert_eq!(risky.1, vec![8]);
    let (defense, _) = Bot::new("planner", None).unwrap().choose(&view);
    assert_eq!(defense.unwrap().stone, 5);
}
