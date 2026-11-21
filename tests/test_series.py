import random
import re

from fastapi.testclient import TestClient
import pytest

from schotten.app import create_app
from schotten.bot import Bot, main
from schotten.client import SchottenClient


def test_series_tallies_actual_games_and_alternates_seats(monkeypatch, capsys):
    app = create_app()
    strategies = []
    clients = []

    def make_bot(url, strategy, seed, name):
        strategies.append(strategy)
        client = SchottenClient(http=TestClient(app))
        clients.append(client)
        return Bot(client, strategy, random.Random(seed), name=name)

    monkeypatch.setattr("schotten.bot._make_bot", make_bot)
    assert main(["series", "--games", "4", "--seed", "1", "--bot-seed", "3"]) == 0
    output = capsys.readouterr().out
    assert strategies == ["greedy", "random", "random", "greedy"] * 2
    assert len(re.findall(r"Game \d/4:", output)) == 4
    greedy = int(re.search(r"greedy: (\d+) wins", output).group(1))
    random_wins = int(re.search(r"random: (\d+) wins", output).group(1))
    draws = int(re.search(r"Draws: (\d+)", output).group(1))
    assert greedy + random_wins + draws == 4
    assert greedy == output.count("winner=greedy")
    assert random_wins == output.count("winner=random")
    assert draws == output.count("winner=draw")
    assert all(client.http.is_closed for client in clients)


def test_series_rejects_nonpositive_game_count():
    with pytest.raises(SystemExit) as exc:
        main(["series", "--games", "0"])
    assert exc.value.code == 2


@pytest.mark.parametrize("quiet", [False, True])
def test_series_failure_is_not_counted_as_a_draw(monkeypatch, capsys, quiet):
    app = create_app()
    clients = []

    def make_bot(url, strategy, seed, name):
        client = SchottenClient(http=TestClient(app))
        clients.append(client)
        return Bot(client, strategy, random.Random(seed), name=name)

    def fail(self, view):
        raise RuntimeError("test failure")

    monkeypatch.setattr("schotten.bot._make_bot", make_bot)
    monkeypatch.setattr(Bot, "take_turn", fail)
    assert main(["series", "--games", "2", *(["--quiet"] if quiet else [])]) == 1
    captured = capsys.readouterr()
    assert "test failure" in captured.err
    if not quiet:
        assert "Completed: 0/2" in captured.out
    assert "Draws: 0" in captured.out
    assert "Elapsed:" in captured.out
    assert all(client.http.is_closed for client in clients)


def test_quiet_series_prints_only_tallies_and_elapsed_time(monkeypatch, capsys):
    app = create_app()

    def make_bot(url, strategy, seed, name):
        return Bot(SchottenClient(http=TestClient(app)), strategy, random.Random(seed), name=name)

    clock = iter([10.0, 12.5])
    monkeypatch.setattr("schotten.bot._make_bot", make_bot)
    monkeypatch.setattr("schotten.bot.perf_counter", lambda: next(clock), raising=False)
    assert main(["series", "--games", "2", "--seed", "1", "--bot-seed", "1", "--quiet"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert re.fullmatch(r"greedy: \d+ wins \(\d+\.\d%\)", lines[0])
    assert re.fullmatch(r"random: \d+ wins \(\d+\.\d%\)", lines[1])
    assert lines[2] == "Draws: 0"
    assert lines[3] == "Elapsed: 2.50s"
