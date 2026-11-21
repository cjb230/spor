from pathlib import Path

import pytest

from benchmarks.mlp_experiment import prepare, summarize


def test_schedule_is_mirrored_unique_and_excludes_teacher_decks(tmp_path):
    schedule = prepare(tmp_path, decks=100, master_seed=7)
    games = schedule["games"]
    assert len(games) == 200
    assert len({game["seed"] for game in games}) == 100
    for first, second in zip(games[::2], games[1::2]):
        assert first["seed"] == second["seed"]
        assert first["bot_seed"] == second["bot_seed"]
        assert [first["mlp_seat"], second["mlp_seat"]] == [0, 1]
        assert 0 <= first["seed"] < 2**63
        assert not 10000 <= first["seed"] < 16000
    assert prepare(tmp_path / "repeat", decks=100, master_seed=7)["games"] == games
    with pytest.raises(FileExistsError):
        prepare(tmp_path, decks=100, master_seed=7)


def test_paired_summary_accounts_for_draws_and_mirrors():
    schedule = [{"seed":11,"bot_seed":1,"mlp_seat":0},{"seed":11,"bot_seed":1,"mlp_seat":1},
                {"seed":22,"bot_seed":2,"mlp_seat":0},{"seed":22,"bot_seed":2,"mlp_seat":1}]
    def rows(winners):
        return [{"schedule_index":index,**game,"winner":winner} for index,(game,winner) in enumerate(zip(schedule,winners))]
    result = summarize(schedule, rows(["mcb:16"]*4), rows(["mlp",None,"mcb:16","mlp"]), resamples=100, seed=1)
    assert result["before"]["losses"] == 4
    assert result["after"]["wins"] == 2
    assert result["after"]["draws"] == 1
    assert result["after"]["score"] == 0.625
    assert result["score_improvement"] == 0.625
    assert result["decks"] == 2
    with pytest.raises(ValueError):
        summarize(schedule, rows(["mlp"]*4)[:-1], rows(["mlp"]*4))
