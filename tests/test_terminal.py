import queue
import random
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from schotten.app import create_app
from schotten.bot import Bot
from schotten.client import ApiError, SchottenClient
from schotten.cards import Card, Color
from schotten.terminal import TerminalSession, card_label, draw_screen, network_worker, parser, run_screen


@pytest.fixture
def players():
    with TestClient(create_app()) as http:
        host = SchottenClient(http=http)
        guest = SchottenClient(http=http)
        host.create_game(seed=7)
        yield host, guest


def test_waiting_and_opponent_views_have_no_legal_moves(players):
    host, guest = players
    session = TerminalSession(host)
    session.refresh()
    assert session.view.phase == "waiting" and session.legal is None
    guest.join(host.game_id)
    other = TerminalSession(guest)
    other.refresh()
    assert other.view.phase == "opponent_turn" and other.legal is None


def test_submit_uses_server_legal_moves(players):
    host, guest = players
    guest.join(host.game_id)
    session = TerminalSession(host)
    session.refresh()
    card = session.view.hand[0]
    assert session.targets(0) == list(range(9))
    session.submit(0, 4)
    assert session.view.stones[4].yours == [card]
    assert session.view.phase == "opponent_turn"
    assert session.legal is None


def test_reject_illegal_target_without_changing_game(players):
    host, guest = players
    guest.join(host.game_id)
    session = TerminalSession(host)
    session.refresh()
    version = session.view.version
    with pytest.raises(ValueError, match="legal"):
        session.submit(0, 9)
    assert host.view().version == version


def test_session_finishes_real_game_with_claims_and_forced_passes(players):
    host, guest = players
    guest.join(host.game_id)
    session = TerminalSession(host)
    opponent = Bot(guest, "random", rng=random.Random(0))
    for _ in range(150):
        session.refresh()
        if session.view.phase == "game_over":
            break
        if session.view.phase == "your_turn":
            if session.legal.plays:
                option = session.legal.plays[0]
                session.submit(session.view.hand.index(option.card), option.stones[0])
            else:
                session.submit(0, 0)
        else:
            opponent.take_turn(guest.view())
    assert session.view.phase == "game_over"
    assert any(event.claims for event in host.log().entries)
    assert any(event.play and event.play.stone in event.claims for event in host.log().entries)


def test_stale_turn_can_be_refreshed(players):
    host, guest = players
    guest.join(host.game_id)
    session = TerminalSession(host)
    session.refresh()
    Bot(host).take_turn(host.view())
    with pytest.raises(ApiError):
        session.submit(0, 0)
    session.refresh()
    assert session.view.phase == "opponent_turn"


class Screen:
    def __init__(self, height=24, width=80):
        self.height, self.width = height, width
        self.output = []

    def getmaxyx(self):
        return self.height, self.width

    def erase(self):
        self.output.clear()

    def addnstr(self, row, column, value, limit, *attrs):
        assert 0 <= row < self.height and 0 <= column < self.width
        self.output.append(value[:limit])

    def refresh(self):
        pass


def test_render_board_hand_and_small_terminal(players):
    host, guest = players
    guest.join(host.game_id)
    session = TerminalSession(host)
    session.refresh()
    screen = Screen()
    draw_screen(screen, session, 0, 0, "Ready")
    output = " ".join(screen.output)
    assert host.game_id in output
    assert "Your hand" in output and "Opponent" in output
    assert "Ready" in output and "Enter" in output
    tiny = Screen(8, 30)
    draw_screen(tiny, session, 0, 0, "")
    assert "80" in " ".join(tiny.output)


def test_cli_create_and_join():
    create = parser().parse_args(["create", "--seed", "9", "--auto-claim"])
    assert create.seed == 9 and create.auto_claim
    join = parser().parse_args(["join", "g_test", "--url", "http://host:8000"])
    assert join.game_id == "g_test" and join.url == "http://host:8000"


def test_card_labels_distinguish_brown_from_blue():
    assert card_label(Card(Color.BLUE, 3)) == "B3"
    assert card_label(Card(Color.BROWN, 3)) == "Br3"


def test_keyboard_submits_a_move_without_blocking_rendering(players):
    host, guest = players
    guest.join(host.game_id)

    class InteractiveScreen(Screen):
        def keypad(self, value):
            pass

        def timeout(self, value):
            pass

        def getch(self):
            time.sleep(0.01)
            output = " ".join(self.output)
            if "Opponent's turn" in output:
                return ord("q")
            if "Claimable:" in output and "[loading]" not in output:
                assert "Loading game..." not in output
                return 10
            assert time.monotonic() < deadline, "UI did not submit a turn"
            return -1

    deadline = time.monotonic() + 5
    run_screen(InteractiveScreen(), host)
    assert guest.view().stones[0].theirs


def test_uncertain_submission_is_not_retried(players, monkeypatch):
    host, guest = players
    guest.join(host.game_id)
    session = TerminalSession(host)
    session.refresh()
    calls = []

    def disconnected(*args, **kwargs):
        calls.append(args)
        raise httpx.ReadTimeout("connection lost")

    monkeypatch.setattr(host, "turn", disconnected)
    commands, results = queue.Queue(), queue.Queue()
    commands.put(("submit", session, 0, 0))
    commands.put(None)
    worker = threading.Thread(target=network_worker, args=(host, commands, results))
    worker.start()
    updated, message, failed = results.get(timeout=5)
    worker.join(timeout=5)
    assert failed and "may have succeeded" in message
    assert updated.legal is None and len(calls) == 1


def test_enter_during_refresh_is_preserved(players, monkeypatch):
    host, guest = players
    guest.join(host.game_id)
    original = TerminalSession.refresh
    refreshing, release = threading.Event(), threading.Event()
    calls = []

    def refresh(session):
        calls.append(None)
        if len(calls) == 2:
            refreshing.set()
            assert release.wait(2)
        original(session)

    monkeypatch.setattr(TerminalSession, "refresh", refresh)

    class InteractiveScreen(Screen):
        stage = 0

        def keypad(self, value):
            pass

        def timeout(self, value):
            pass

        def getch(self):
            time.sleep(0.01)
            output = " ".join(self.output)
            if "Opponent's turn" in output:
                return ord("q")
            if self.stage == 0 and "Claimable:" in output and "[loading]" not in output:
                self.stage = 1
                return ord("r")
            if self.stage == 1 and refreshing.is_set():
                self.stage = 2
                release.set()
                return 10
            assert time.monotonic() < deadline, "Enter was lost during refresh"
            return -1

    deadline = time.monotonic() + 3
    try:
        run_screen(InteractiveScreen(), host)
    finally:
        release.set()
    assert guest.view().stones[0].theirs


def test_board_displays_completion_precedence(players):
    host, guest = players
    guest.join(host.game_id)
    session = TerminalSession(host)
    session.refresh()
    session.view.stones[0].first_to_complete = "you"
    session.view.stones[1].first_to_complete = "them"
    screen = Screen()
    draw_screen(screen, session, 0, 0, "")
    output = " ".join(screen.output)
    assert "you 1st" in output and "opp 1st" in output


def test_waiting_instructions_include_server_url(players):
    host, _ = players
    session = TerminalSession(host)
    session.refresh()
    screen = Screen()
    draw_screen(screen, session, 0, 0, "")
    assert "--url http://testserver" in " ".join(screen.output)
