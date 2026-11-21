import copy
import json
import math
import random
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from schotten.bot import Bot
from schotten.cards import FULL_DECK
from schotten.game import Game
from schotten.neural.data import Example, load_examples, split_examples
from schotten.neural.encoding import decode_action
from schotten.schemas import CreateGameRequest, TurnRequest


NOW = datetime(2026, 11, 19, tzinfo=timezone.utc)


def recording(seed=42, strategies=("mcb:8", "greedy")):
    game = Game("training-fixture", CreateGameRequest(seed=seed), NOW)
    game.join(NOW)
    chooser = random.Random(17)
    views = []
    while not game.over:
        seat = game.turn
        view = game.view(seat)
        options = Bot.options(view)
        play = chooser.choice(options) if options else None
        views.append(view)
        game.play_turn(
            seat,
            TurnRequest(
                version=game.version,
                play=play,
                claims=Bot.provable_stones(view, play),
            ),
            NOW,
        )
    seats = [
        {
            "seat": seat,
            "entrant_index": 10 + seat,
            "name": f"player-{seat}",
            "strategy": strategy,
            "seed": str(17 + seat),
        }
        for seat, strategy in enumerate(strategies)
    ]
    winner = None if game.winner is None else seats[game.winner]
    return {
        "schema_version": 1,
        "game_id": game.id,
        "schedule_index": 0,
        "round": 0,
        "deck_seed": str(seed),
        "seats": seats,
        "winner_seat": game.winner,
        "winner_entrant_index": None if winner is None else winner["entrant_index"],
        "winner": None if winner is None else winner["name"],
        "reason": game.win_reason,
        "turns": len(game.log),
        "log": game.log_view().model_dump(mode="json"),
    }, views


@pytest.fixture(scope="module")
def completed_game():
    return recording()


def write_recording(directory, document, name="game-000001.json"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_teacher_examples_are_before_actions_and_use_only_the_selected_seat(
    tmp_path, completed_game
):
    document, views = completed_game
    path = write_recording(tmp_path, document)
    examples = load_examples([path], ["mcb"])
    selected = [
        (entry, view)
        for entry, view in zip(document["log"]["entries"], views)
        if entry["seat"] == 0
    ]
    assert len(examples) == len(selected) > 0
    for example, (entry, view) in zip(examples, selected):
        assert len(example.features) == 1082
        assert len(example.mask) == 487
        assert all(type(value) is float for value in example.features)
        assert all(type(value) is bool for value in example.mask)
        assert example.mask[example.action]
        play = decode_action(example.action)
        assert (None if play is None else play.model_dump(mode="json")) == entry["play"]
        assert example.features[:54] == [
            float(card in view.hand) for card in FULL_DECK
        ]
        assert example.features[-2:] == [view.deck_count / 42, view.opponent_hand_count / 6]
        assert sum(example.mask) == max(1, len(Bot.options(view)))
        assert example.group == "42"
        assert example.source == str(path.resolve())
    assert examples[0].features[54:1026] == [0.0] * 972
    assert examples[0].features[-2:] == [1.0, 1.0]
    with pytest.raises(FrozenInstanceError):
        examples[0].action = 0


@pytest.mark.parametrize(
    ("teachers", "selected_seats"),
    [(["mcb:8"], {0}), (["greedy"], {1}), (["mcb", "greedy"], {0, 1})],
)
def test_teacher_matching_uses_strategy_not_display_name(
    tmp_path, completed_game, teachers, selected_seats
):
    document, _ = completed_game
    path = write_recording(tmp_path, document)
    examples = load_examples([path], teachers)
    assert len(examples) == sum(
        entry["seat"] in selected_seats for entry in document["log"]["entries"]
    )


def test_bare_teacher_matches_bare_strategy(tmp_path, completed_game):
    document = copy.deepcopy(completed_game[0])
    document["seats"][0]["strategy"] = "mcb"
    path = write_recording(tmp_path, document)
    assert load_examples([path], ["mcb"])


@pytest.mark.parametrize("teachers", [[], [""], ["mcb:80"], ["player-0"]])
def test_missing_or_unmatched_teachers_are_rejected(tmp_path, completed_game, teachers):
    path = write_recording(tmp_path, completed_game[0])
    with pytest.raises(ValueError):
        load_examples([path], teachers)


def test_discovery_is_recursive_sorted_and_deduplicated(tmp_path, completed_game):
    document, _ = completed_game
    later = write_recording(tmp_path / "z" / "run", document)
    earlier = write_recording(tmp_path / "a", document)
    (tmp_path / "manifest.json").write_text("not a game", encoding="utf-8")
    examples = load_examples([tmp_path, earlier, tmp_path / "a"], ["mcb"])
    sources = [example.source for example in examples]
    assert sources == sorted(sources)
    assert set(sources) == {str(earlier.resolve()), str(later.resolve())}
    assert sources.count(str(earlier.resolve())) == sources.count(str(later.resolve()))


@pytest.mark.parametrize("seed", [-42, 2**80 + 5, -(2**127)])
def test_signed_and_wide_deck_seeds_replay_without_precision_loss(tmp_path, seed):
    document, _ = recording(seed)
    path = write_recording(tmp_path, document)
    examples = load_examples([path], ["mcb"])
    assert {example.group for example in examples} == {str(abs(seed))}


@pytest.mark.parametrize(
    "tamper",
    [
        "schema",
        "game_id",
        "seed",
        "turn_count",
        "turn_version",
        "turn_seat",
        "illegal_card",
        "illegal_stone",
        "claims",
        "initial_hands",
        "draw_order",
        "winner_seat",
        "winner_name",
        "winner_entrant",
        "reason",
        "status",
        "seat_index",
        "entrant_index",
        "truncated",
        "extra_turn",
        "boolean_seat",
        "float_version",
    ],
)
def test_corrupt_recordings_are_rejected_with_source_path(
    tmp_path, completed_game, tamper
):
    document = copy.deepcopy(completed_game[0])
    first = document["log"]["entries"][0]
    if tamper == "schema":
        document["schema_version"] = 2
    elif tamper == "game_id":
        document["game_id"] = "different"
    elif tamper == "seed":
        document["deck_seed"] = "43"
    elif tamper == "turn_count":
        document["turns"] += 1
    elif tamper == "turn_version":
        first["version"] += 1
    elif tamper == "turn_seat":
        first["seat"] = 1
    elif tamper == "illegal_card":
        first["play"]["card"] = document["log"]["revealed"]["initial_hands"][1][0]
    elif tamper == "illegal_stone":
        first["play"]["stone"] = 9
    elif tamper == "claims":
        first["claims"] = [0]
    elif tamper == "initial_hands":
        hands = document["log"]["revealed"]["initial_hands"]
        hands[0][0], hands[1][0] = hands[1][0], hands[0][0]
    elif tamper == "draw_order":
        draws = document["log"]["revealed"]["draw_order"]
        draws[0], draws[1] = draws[1], draws[0]
    elif tamper == "winner_seat":
        document["winner_seat"] = 1 - document["winner_seat"]
    elif tamper == "winner_name":
        document["winner"] = "someone else"
    elif tamper == "winner_entrant":
        document["winner_entrant_index"] = 99
    elif tamper == "reason":
        document["reason"] = "forfeit"
    elif tamper == "status":
        document["log"]["status"] = "in_progress"
    elif tamper == "seat_index":
        document["seats"][0]["seat"] = 1
    elif tamper == "entrant_index":
        document["seats"][0]["entrant_index"] = 11
    elif tamper == "truncated":
        document["log"]["entries"].pop()
        document["turns"] -= 1
    elif tamper == "extra_turn":
        document["log"]["entries"].append(copy.deepcopy(first))
        document["turns"] += 1
    elif tamper == "boolean_seat":
        first["seat"] = False
    elif tamper == "float_version":
        first["version"] = float(first["version"])
    path = write_recording(tmp_path, document)
    with pytest.raises(ValueError, match="game-000001.json"):
        load_examples([path], ["mcb"])


def test_corruption_in_unselected_player_turn_is_still_rejected(tmp_path, completed_game):
    document = copy.deepcopy(completed_game[0])
    document["log"]["entries"][1]["claims"] = [0]
    path = write_recording(tmp_path, document)
    with pytest.raises(ValueError):
        load_examples([path], ["mcb"])


@pytest.mark.parametrize("contents", ["{", "[]", '{"schema_version":1,"schema_version":2}'])
def test_malformed_json_is_rejected(tmp_path, contents):
    path = tmp_path / "game-bad.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="game-bad.json"):
        load_examples([path], ["mcb"])


def test_duplicate_fields_in_otherwise_valid_json_are_rejected(tmp_path, completed_game):
    path = tmp_path / "game-duplicate.json"
    contents = json.dumps(completed_game[0]).replace(
        '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
    )
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_examples([path], ["mcb"])


def test_nonstandard_json_numbers_are_rejected(tmp_path, completed_game):
    document = copy.deepcopy(completed_game[0])
    document["schedule_index"] = math.nan
    path = write_recording(tmp_path, document)
    with pytest.raises(ValueError, match="game-000001.json"):
        load_examples([path], ["mcb"])


def test_empty_or_missing_input_paths_are_rejected(tmp_path):
    for paths in ([], [tmp_path], [tmp_path / "missing.json"]):
        with pytest.raises(ValueError):
            load_examples(paths, ["mcb"])


def test_same_deal_and_observations_do_not_depend_on_hidden_metadata(
    tmp_path, completed_game
):
    positive = copy.deepcopy(completed_game[0])
    negative = copy.deepcopy(positive)
    negative["deck_seed"] = "-42"
    negative["seats"][0]["seed"] = "9999999999999999999999"
    negative["seats"][1]["seed"] = "-73"
    positive_path = write_recording(tmp_path / "positive", positive)
    negative_path = write_recording(tmp_path / "negative", negative)
    first = load_examples([positive_path], ["mcb"])
    second = load_examples([negative_path], ["mcb"])
    assert [example.features for example in first] == [example.features for example in second]
    assert [example.mask for example in first] == [example.mask for example in second]
    assert {example.group for example in first + second} == {"42"}


def grouped_examples():
    return [
        Example([float(group)], [True], 0, str(group), f"source-{group}-{index}")
        for group in range(8)
        for index in range(group + 1)
    ]


def test_split_keeps_all_examples_of_each_deck_together():
    examples = grouped_examples()
    training, validation = split_examples(examples, 0.25, 17)
    assert training and validation
    assert not {example.group for example in training} & {
        example.group for example in validation
    }
    assert len({example.group for example in validation}) == 2
    assert sorted(example.source for example in training + validation) == sorted(
        example.source for example in examples
    )
    assert split_examples(examples, 0.25, 17) == (training, validation)
    assert split_examples(examples, 0.25, 18) != (training, validation)
    reversed_training, reversed_validation = split_examples(list(reversed(examples)), 0.25, 17)
    assert {example.group for example in reversed_validation} == {
        example.group for example in validation
    }
    assert {example.group for example in reversed_training} == {
        example.group for example in training
    }


def test_zero_validation_fraction_allows_single_deck():
    examples = [Example([], [], 0, "42", "one")]
    assert split_examples(examples, 0, 3) == (examples, [])


@pytest.mark.parametrize("fraction", [0.000001, 0.999999])
def test_positive_validation_fraction_always_produces_two_nonempty_parts(fraction):
    examples = [Example([], [], 0, str(group), "test") for group in range(2)]
    training, validation = split_examples(examples, fraction, 3)
    assert len(training) == len(validation) == 1


@pytest.mark.parametrize("fraction", [-0.1, 1, 1.1, math.nan, math.inf])
def test_invalid_split_fractions_are_rejected(fraction):
    with pytest.raises(ValueError):
        split_examples(grouped_examples(), fraction, 3)


def test_validation_requires_at_least_two_distinct_deals():
    with pytest.raises(ValueError):
        split_examples([Example([], [], 0, "42", "test")] * 3, 0.2, 3)
    with pytest.raises(ValueError):
        split_examples([], 0, 3)
