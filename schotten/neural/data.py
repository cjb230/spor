"""Validated native tournament recordings for supervised policy learning."""

import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from schotten.errors import GameError
from schotten.game import Game
from schotten.schemas import CreateGameRequest, GameLog, TurnRequest

from .encoding import action_id, encode_view, legal_mask


@dataclass(frozen=True)
class Example:
    features: list[float]
    mask: list[bool]
    action: int
    group: str
    source: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _seed(value: object, label: str) -> int:
    _require(
        isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value) is not None,
        f"invalid {label}",
    )
    return int(value)


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"invalid JSON number {value}")


def _selected(strategy: str, teachers: Sequence[str]) -> bool:
    return any(
        strategy == teacher or (":" not in teacher and strategy.startswith(teacher + ":"))
        for teacher in teachers
    )


def _replay(path: Path, teachers: Sequence[str]) -> list[Example]:
    document = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_json_object,
        parse_constant=_invalid_constant,
    )
    _require(isinstance(document, dict), "recording must be a JSON object")
    _require(
        type(document["schema_version"]) is int and document["schema_version"] == 1,
        "unsupported recording schema",
    )
    game_id = document["game_id"]
    _require(isinstance(game_id, str) and bool(game_id), "invalid game identity")
    deck_seed = _seed(document["deck_seed"], "deck seed")
    seats = document["seats"]
    _require(isinstance(seats, list) and len(seats) == 2, "expected two seats")
    for index, seat in enumerate(seats):
        _require(isinstance(seat, dict), "invalid seat metadata")
        _require(type(seat["seat"]) is int and seat["seat"] == index, "invalid seat index")
        _require(
            type(seat["entrant_index"]) is int and seat["entrant_index"] >= 0,
            "invalid entrant index",
        )
        for field in ("name", "strategy"):
            _require(
                isinstance(seat[field], str) and bool(seat[field].strip()),
                f"invalid seat {field}",
            )
        _seed(seat["seed"], "agent seed")
    _require(
        seats[0]["entrant_index"] != seats[1]["entrant_index"],
        "seats must represent different entrants",
    )
    winner = document["winner_seat"]
    _require(
        winner is None or (type(winner) is int and winner in (0, 1)),
        "invalid winner seat",
    )
    winning_entrant = None if winner is None else seats[winner]["entrant_index"]
    _require(
        document["winner"] == (None if winner is None else seats[winner]["name"])
        and type(document["winner_entrant_index"]) is type(winning_entrant)
        and document["winner_entrant_index"] == winning_entrant,
        "winner metadata disagrees with seats",
    )
    raw_log = document["log"]
    _require(isinstance(raw_log, dict), "missing turn log")
    raw_entries = raw_log["entries"]
    _require(isinstance(raw_entries, list), "invalid turn log")
    _require(
        type(document["turns"]) is int and document["turns"] == len(raw_entries),
        "turn count disagrees with log",
    )
    recorded_log = GameLog.model_validate(raw_log)
    now = datetime(2000, 1, 1, tzinfo=timezone.utc)
    game = Game(game_id, CreateGameRequest(seed=deck_seed), now)
    game.join(now)
    examples = []
    for raw_entry, entry in zip(raw_entries, recorded_log.entries):
        _require(
            type(raw_entry["version"]) is int and entry.version == game.version + 1,
            "turn versions are not sequential integers",
        )
        _require(type(raw_entry["seat"]) is int, "invalid turn seat")
        _require(not game.over and entry.seat == game.turn, "turn seat disagrees with replay")
        if _selected(seats[entry.seat]["strategy"], teachers):
            view = game.view(entry.seat)
            mask = legal_mask(view)
            action = action_id(entry.play)
            _require(mask[action], "teacher action is not legal")
            examples.append(
                Example(encode_view(view), mask, action, str(abs(deck_seed)), str(path))
            )
        game.play_turn(
            entry.seat,
            TurnRequest(version=game.version, play=entry.play, claims=entry.claims),
            now,
        )
    _require(
        game.over and game.winner == winner and game.win_reason == document["reason"],
        "recorded outcome disagrees with replay",
    )
    _require(
        json.dumps(game.log_view().model_dump(mode="json"), sort_keys=True, allow_nan=False)
        == json.dumps(raw_log, sort_keys=True, allow_nan=False),
        "recorded deal or log disagrees with replay",
    )
    return examples


def load_examples(paths: Sequence[Path], teachers: Sequence[str]) -> list[Example]:
    """Replay recordings and extract pre-action observations for selected strategies.

    Bare teacher names also match sampling suffixes, while names containing a
    colon match exactly. Sources are canonical absolute paths, sorted and deduplicated.
    No examples are returned unless every discovered recording validates.
    """
    _require(
        bool(teachers)
        and not isinstance(teachers, str)
        and all(isinstance(teacher, str) and bool(teacher.strip()) for teacher in teachers),
        "at least one nonempty teacher strategy is required",
    )
    files = set()
    for supplied_path in paths:
        path = Path(supplied_path).resolve()
        if path.is_dir():
            files.update(
                candidate.resolve()
                for candidate in path.rglob("game-*.json")
                if candidate.is_file()
            )
        elif path.is_file():
            files.add(path)
        else:
            raise ValueError(f"recording path does not exist: {path}")
    _require(bool(files), "no native game recordings found")
    examples = []
    for path in sorted(files):
        try:
            examples.extend(_replay(path, teachers))
        except (OSError, ValueError, KeyError, TypeError, IndexError, GameError) as error:
            raise ValueError(f"{path}: invalid recording: {error}") from error
    _require(bool(examples), f"no moves found for teacher strategies: {', '.join(teachers)}")
    return examples


def split_examples(
    examples: Sequence[Example], validation_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """Partition whole deck groups reproducibly, preserving order within each part."""
    _require(
        math.isfinite(validation_fraction) and 0 <= validation_fraction < 1,
        "validation fraction must be finite and in [0, 1)",
    )
    _require(bool(examples), "cannot split an empty dataset")
    if validation_fraction == 0:
        return list(examples), []
    groups = sorted({example.group for example in examples})
    _require(len(groups) >= 2, "validation requires at least two distinct deck groups")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    validation_groups = set(groups[:count])
    training = [example for example in examples if example.group not in validation_groups]
    validation = [example for example in examples if example.group in validation_groups]
    return training, validation
