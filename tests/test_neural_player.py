import random

import pytest

torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient

from schotten.app import create_app
from schotten.bot import Bot
from schotten.client import SchottenClient
from schotten.neural.encoding import action_id, legal_mask
from schotten.neural.model import Policy, save_checkpoint
from schotten.neural.player import NeuralBot


@pytest.mark.parametrize("neural_seat", [0, 1])
def test_checkpoint_player_finishes_a_rest_game_on_either_seat(tmp_path, neural_seat):
    checkpoint = tmp_path / "untrained.pt"
    torch.manual_seed(7)
    save_checkpoint(checkpoint, Policy(hidden=(16, 16)))
    with TestClient(create_app()) as http:
        clients = [SchottenClient(http=http), SchottenClient(http=http)]
        grant = clients[0].create_game(seed=7)
        clients[1].join(grant.game_id)
        players = [Bot(clients[seat], "random", random.Random(seat)) for seat in range(2)]
        players[neural_seat] = NeuralBot(clients[neural_seat], checkpoint, device="cpu")
        for _ in range(120):
            view = clients[0].view()
            if view.phase == "game_over":
                break
            player = players[view.turn]
            current = player.client.view()
            if view.turn == neural_seat:
                play, _ = player.choose(current)
                assert legal_mask(current)[action_id(play)]
            player.take_turn(current)
        assert clients[0].view().phase == "game_over"


def test_cli_initializes_and_inspects_explicitly_untrained_weights(tmp_path, capsys):
    from schotten.neural.__main__ import main
    path = tmp_path / "mlp.pt"
    assert main(["init", "--output", str(path), "--seed", "7"]) == 0
    assert path.is_file()
    assert "untrained" in capsys.readouterr().out.lower()
    assert main(["inspect", str(path)]) == 0
    output = capsys.readouterr().out
    assert "468199" in output and '"epoch": 0' in output
