"""Play whole games with random legal moves and check invariants hold throughout."""
import random
from datetime import datetime, timezone

import pytest

from schotten.cards import FULL_DECK
from schotten.game import Game, table_cards
from schotten.schemas import CreateGameRequest, Play, TurnRequest

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def assert_cards_conserved(g):
    cards = list(g.deck) + [c for h in g.hands for c in h] + table_cards(g.stones)
    assert len(cards) == 54 and set(cards) == set(FULL_DECK)


def play_random_game(seed, claim_prob, auto_claim):
    rng = random.Random(seed)
    g = Game("g", CreateGameRequest(seed=seed, auto_claim=auto_claim), NOW)
    g.join(NOW)
    for _ in range(300):
        if g.over:
            break
        seat = g.turn
        moves = g.legal_moves(seat)
        play = None
        if moves.plays:
            option = rng.choice(moves.plays)
            play = Play(card=option.card, stone=rng.choice(option.stones))
        claims = [i for i in moves.claimable if rng.random() < claim_prob]
        g.play_turn(seat, TurnRequest(version=g.version, play=play, claims=claims), NOW)
        assert_cards_conserved(g)
    assert g.over, "game did not terminate"
    return g


@pytest.mark.parametrize("auto_claim", [False, True])
@pytest.mark.parametrize("claim_prob", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("seed", range(8))
def test_random_games_terminate_with_a_result(seed, claim_prob, auto_claim):
    g = play_random_game(seed, claim_prob, auto_claim)
    assert g.win_reason in {"five_stones", "three_adjacent", "stalemate"}
    claimed = [s.claimed_by for s in g.stones]
    if g.win_reason == "five_stones":
        assert claimed.count(g.winner) >= 5
    assert g.log_view().revealed is not None
