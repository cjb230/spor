from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from schotten.cards import FULL_DECK, Card, Color
from schotten.errors import GameError
from schotten.game import Game
from schotten.schemas import CreateGameRequest, Play, TurnRequest

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def card(color, value):
    return Card(Color(color), value)


def new_game(**options):
    g = Game("g_test", CreateGameRequest(seed=1, **options), NOW)
    g.join(NOW)
    return g


def take(g, c):
    """Remove a card from wherever it is so rigged states stay consistent."""
    for hand in g.hands:
        if c in hand:
            hand.remove(c)
            return
    g.deck.remove(c)


def turn(g, seat, play=None, claims=(), **kw):
    req = TurnRequest(version=g.version, play=play, claims=list(claims), **kw)
    g.play_turn(seat, req, NOW)


def code_of(fn):
    with pytest.raises(GameError) as exc:
        fn()
    return exc.value.code


def rig_color_run(g, seat, stone=0, colour="blue", values=(7, 8, 9)):
    """Two cards already on the stone, the third in hand. Returns the card to play."""
    a, b, last = (card(colour, v) for v in values)
    for c in (a, b, last):
        take(g, c)
    g.stones[stone].cards[seat] = [a, b]
    g.hands[seat].append(last)
    return last


def test_deal_and_start():
    g = new_game()
    assert [len(h) for h in g.hands] == [6, 6]
    assert len(g.deck) == 42
    assert g.status == "in_progress" and g.turn == 0 and g.version == 1
    assert [e.type for e in g.events] == ["player_joined", "game_started"]


def test_play_draws_and_passes_turn():
    g = new_game()
    c = g.hands[0][0]
    turn(g, 0, Play(card=c, stone=3))
    assert g.stones[3].cards[0] == [c]
    assert len(g.hands[0]) == 6 and len(g.deck) == 41
    assert g.turn == 1 and g.version == 2
    assert [e.type for e in g.events_after(1)] == ["card_played", "turn_passed"]


def test_views_hide_the_opponents_hand():
    g = new_game()
    v = g.view(0)
    assert v.opponent_hand_count == 6
    assert len(v.unseen_cards) == 48  # everything except my own six
    assert not set(v.hand) & set(v.unseen_cards)
    assert set(v.hand) | set(v.unseen_cards) == set(FULL_DECK)
    assert v.phase == "your_turn" and g.view(1).phase == "opponent_turn"


def test_rejections():
    g = new_game()
    mine = g.hands[0][0]
    stranger = g.view(0).unseen_cards[0]
    assert code_of(lambda: turn(g, 1, Play(card=g.hands[1][0], stone=0))) == "NOT_YOUR_TURN"
    assert code_of(lambda: g.play_turn(0, TurnRequest(version=0, play=Play(card=mine, stone=0)), NOW)) == "STALE_VERSION"
    assert code_of(lambda: turn(g, 0, Play(card=stranger, stone=0))) == "CARD_NOT_IN_HAND"
    assert code_of(lambda: turn(g, 0)) == "PLAY_REQUIRED"

    g.stones[0].claimed_by = 1
    assert code_of(lambda: turn(g, 0, Play(card=mine, stone=0))) == "STONE_ALREADY_CLAIMED"
    g.stones[0].claimed_by = None

    full = [card("red", 1), card("red", 2), card("red", 3)]
    for c in full:
        take(g, c)
    g.stones[0].cards[0] = full
    spare = g.hands[0][0]  # chosen after rigging, so it is certainly still in hand
    assert code_of(lambda: turn(g, 0, Play(card=spare, stone=0))) == "STONE_FULL"


def test_claim_incomplete():
    g = new_game()
    last = rig_color_run(g, 0)
    other = next(c for c in g.hands[0] if c != last)
    assert code_of(lambda: turn(g, 0, Play(card=other, stone=1), claims=[0])) == "CLAIM_INCOMPLETE"


def test_provable_claim_succeeds():
    g = new_game()
    last = rig_color_run(g, 0)
    turn(g, 0, Play(card=last, stone=0), claims=[0])
    assert g.stones[0].claimed_by == 0 and g.stones[0].first_to_complete == 0
    assert any(e.type == "stone_claimed" and e.formation == "color_run" for e in g.events)


def test_unprovable_claim_rejects_the_whole_turn_atomically():
    g = new_game()
    last = rig_color_run(g, 0, colour="blue", values=(1, 2, 3))
    hand_before, version_before = list(g.hands[0]), g.version
    assert code_of(lambda: turn(g, 0, Play(card=last, stone=0), claims=[0])) == "CLAIM_NOT_PROVABLE"
    assert g.hands[0] == hand_before and g.version == version_before
    assert len(g.stones[0].cards[0]) == 2 and g.turn == 0


def test_three_adjacent_wins_and_ends_the_game():
    g = new_game()
    assert g.log_view().revealed is None  # hidden while playing
    g.stones[0].claimed_by = 0
    g.stones[1].claimed_by = 0
    last = rig_color_run(g, 0, stone=2)
    turn(g, 0, Play(card=last, stone=2), claims=[2])
    assert (g.status, g.winner, g.win_reason) == ("finished", 0, "three_adjacent")
    assert g.view(0).winner == "you" and g.view(1).winner == "them"
    assert g.view(0).phase == "game_over"
    assert code_of(lambda: turn(g, 1, Play(card=g.hands[1][0], stone=3))) == "GAME_OVER"
    assert g.log_view().revealed is not None


def test_auto_claim_awards_provable_stones():
    g = new_game(auto_claim=True)
    last = rig_color_run(g, 0)
    turn(g, 0, Play(card=last, stone=0))
    assert g.stones[0].claimed_by == 0


def test_turn_id_makes_retries_idempotent():
    g = new_game()
    tid = uuid4()
    req = TurnRequest(version=g.version, turn_id=tid, play=Play(card=g.hands[0][0], stone=0))
    g.play_turn(0, req, NOW)
    version = g.version
    g.play_turn(0, req, NOW)  # same request again: no error, no second move
    assert g.version == version and len(g.stones[0].cards[0]) == 1


def test_pass_only_allowed_without_a_legal_play():
    g = new_game()
    g.hands[0] = []
    turn(g, 0)  # empty hand: passing is fine
    assert g.turn == 1


def test_clock_forfeit():
    g = new_game(turn_timeout_seconds=10)
    assert g.view(0).turn_deadline == NOW + timedelta(seconds=10)
    assert not g.check_timeout(NOW + timedelta(seconds=9))
    assert g.check_timeout(NOW + timedelta(seconds=11))
    assert (g.status, g.winner, g.win_reason) == ("forfeit", 1, "forfeit")


def test_second_join_is_refused():
    g = new_game()
    assert code_of(lambda: g.join(NOW)) == "GAME_FULL"


def test_stuck_game_is_settled_by_the_server():
    g = new_game()
    g.hands = [[], []]
    g.deck = []
    g.stones[0].cards = [
        [card("red", 1), card("blue", 2), card("green", 3)],
        [card("yellow", 1), card("purple", 2), card("brown", 3)],
    ]
    g.stones[0].first_to_complete = 0  # exact tie goes to the first to complete
    turn(g, 0)  # pass: nobody can play any more
    assert g.stones[0].claimed_by == 0
    assert (g.status, g.winner, g.win_reason) == ("finished", 0, "stalemate")
