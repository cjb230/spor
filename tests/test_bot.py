"""Bots talk to the app only through the REST client, exactly as an outside player would."""
import random

import pytest
from fastapi.testclient import TestClient

from schotten.app import create_app
from schotten.bot import Bot
from schotten.client import ApiError, SchottenClient
from schotten.schemas import Play


def play_game(seed, strategies=("greedy", "random"), **options):
    """Alternate two bots over one in-process app until the game ends. Returns both final views."""
    with TestClient(create_app()) as tc:
        a, b = SchottenClient(http=tc), SchottenClient(http=tc)
        a.create_game(seed=seed, **options)
        b.join(a.game_id)
        bots = [
            Bot(a, strategies[0], random.Random(seed)),
            Bot(b, strategies[1], random.Random(seed + 1)),
        ]
        for _ in range(300):
            views = [bot.client.view() for bot in bots]
            if views[0].phase == "game_over":
                return views
            for bot, view in zip(bots, views):
                if view.phase == "your_turn":
                    bot.take_turn(view)
        raise AssertionError("game did not finish")


@pytest.mark.parametrize("strategies", [("greedy", "random"), ("random", "random"), ("greedy", "greedy")])
@pytest.mark.parametrize("auto_claim", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_bots_finish_games_with_consistent_views(seed, auto_claim, strategies):
    mine, theirs = play_game(seed, strategies, auto_claim=auto_claim)
    assert mine.phase == theirs.phase == "game_over"
    assert mine.win_reason == theirs.win_reason
    flip = {"you": "them", "them": "you", None: None}
    assert theirs.winner == flip[mine.winner]


def test_greedy_takes_a_provable_stone_and_claims_it():
    with TestClient(create_app()) as tc:
        a, b = SchottenClient(http=tc), SchottenClient(http=tc)
        a.create_game(seed=3)
        b.join(a.game_id)
        bot = Bot(a, "greedy", random.Random(0))
        view = a.view()
        # Fake a view where one card completes an unbeatable run 7-8-9 of one colour.
        c = view.hand[0]
        from schotten.cards import Card

        top = [Card(c.color, 8), Card(c.color, 9)]
        stones = [s.model_copy() for s in view.stones]
        stones[0] = stones[0].model_copy(update={"yours": top})
        seven = Card(c.color, 7)
        fake = view.model_copy(
            update={
                "stones": stones,
                "hand": [seven, *[h for h in view.hand if h != seven]][:6],
                "unseen_cards": [u for u in view.unseen_cards if u not in (seven, *top)],
            }
        )
        play, claims = bot.choose(fake)
        assert play == Play(card=seven, stone=0) and claims == [0]


def test_bot_passes_when_it_has_no_legal_play():
    with TestClient(create_app()) as tc:
        a, b = SchottenClient(http=tc), SchottenClient(http=tc)
        a.create_game(seed=1)
        b.join(a.game_id)
        view = a.view().model_copy(update={"hand": []})
        assert Bot(a, "greedy").choose(view) == (None, [])


def test_client_surfaces_server_errors():
    with TestClient(create_app()) as tc:
        a, b = SchottenClient(http=tc), SchottenClient(http=tc)
        a.create_game(seed=1)
        b.join(a.game_id)
        with pytest.raises(ApiError) as e:
            b.turn(b.view().version, Play(card=b.view().hand[0], stone=0))
        assert (e.value.status, e.value.code) == (403, "NOT_YOUR_TURN")
        stranger = Bot(SchottenClient(http=tc, game_id=a.game_id, token="bad"))
        with pytest.raises(ApiError) as e:
            stranger.client.view()
        assert e.value.code == "UNAUTHORIZED"


def test_greedy_beats_random_more_often_than_not():
    wins = sum(play_game(seed)[0].winner == "you" for seed in range(30))
    assert wins >= 20, f"greedy only won {wins}/30 against random"
