"""Full-screen human client: python -m schotten.terminal create or join GAME_ID."""
import argparse
import curses
import queue
import shlex
import sys
import threading
import time
from datetime import datetime, timezone
from uuid import uuid4

import httpx

from .client import ApiError, SchottenClient
from .rules import claim_is_provable
from .schemas import Play


class TerminalSession:
    def __init__(self, client):
        self.client = client
        self.view = None
        self.legal = None

    def refresh(self):
        view = self.client.view()
        legal = self.client.legal_moves() if view.phase == "your_turn" else None
        if legal is not None and legal.version != view.version:
            self.view, self.legal = self.client.view(), None
            return
        self.view, self.legal = view, legal

    def targets(self, card_index):
        if self.legal is None or not 0 <= card_index < len(self.view.hand):
            return []
        card = self.view.hand[card_index]
        return next((option.stones for option in self.legal.plays if option.card == card), [])

    def submit(self, card_index, stone_index):
        if self.legal is None or self.view.phase != "your_turn":
            raise ValueError("Wait for your turn and the legal moves to load.")
        play = None
        if self.legal.plays:
            if stone_index not in self.targets(card_index):
                raise ValueError("Choose a legal stone for the selected card.")
            play = Play(card=self.view.hand[card_index], stone=stone_index)
        claims = []
        for stone in self.view.stones:
            mine = list(stone.yours)
            first = stone.first_to_complete
            if play is not None and play.stone == stone.index:
                mine.append(play.card)
                if len(mine) == 3 and first is None:
                    first = "you"
            if stone.claimed_by is None and len(mine) == 3:
                if claim_is_provable(mine, stone.theirs, self.view.unseen_cards, first == "you"):
                    claims.append(stone.index)
        self.view = self.client.turn(self.view.version, play, claims, turn_id=uuid4())
        self.legal = None


def card_label(card):
    color = "Br" if card.color.value == "brown" else card.color.value[0].upper()
    return f"{color}{card.value}"


def draw_screen(screen, session, card_index, stone_index, message):
    screen.erase()
    height, width = screen.getmaxyx()

    def write(row, column, value, attrs=0):
        if row < height and column < width - 1:
            try:
                screen.addnstr(row, column, str(value), width - column - 1, attrs)
            except curses.error:
                pass

    if height < 24 or width < 80:
        write(0, 0, "Resize to at least 80 x 24. Q quits.")
        screen.refresh()
        return

    view = session.view
    write(0, 1, f"SCHOTTEN TOTTEN   Game {session.client.game_id}", curses.A_BOLD)
    if view is None:
        write(3, 1, "Connecting...")
    else:
        phase = {
            "waiting": "Waiting for an opponent",
            "your_turn": "Your turn",
            "opponent_turn": "Opponent's turn",
            "game_over": "Game over",
        }[view.phase]
        write(1, 1, f"{phase} | Deck: {view.deck_count} | Opponent hand: {view.opponent_hand_count}")
        if view.phase == "waiting":
            write(2, 1, f"Join: python -m schotten.terminal join {view.game_id} \\")
            write(3, 3, f"--url {shlex.quote(str(session.client.http.base_url))}")
        elif view.phase == "game_over":
            result = "Draw" if view.winner is None else "You won!" if view.winner == "you" else "Opponent won"
            write(2, 1, f"{result} ({view.win_reason})", curses.A_BOLD)
        elif view.turn_deadline is not None:
            seconds = max(0, int((view.turn_deadline - datetime.now(timezone.utc)).total_seconds()))
            write(2, 1, f"Turn clock: {seconds}s")

        write(4, 1, "Opponent", curses.A_BOLD)
        targets = session.targets(card_index)
        for stone in view.stones:
            column = 2 + stone.index * 8
            for offset, card in enumerate(stone.theirs):
                write(5 + offset, column + 2, card_label(card))
            attrs = curses.A_REVERSE if stone.index == stone_index else 0
            if stone.index in targets:
                attrs |= curses.A_BOLD
            marker = "+" if stone.index in targets else " "
            write(9, column, f"{marker}[{stone.index + 1}]{marker}", attrs)
            owner = {None: "open", "you": "YOURS", "them": "THEIRS"}[stone.claimed_by]
            write(10, column, owner)
            first = {None: "", "you": "you 1st", "them": "opp 1st"}[stone.first_to_complete]
            write(11, column, first)
            for offset, card in enumerate(stone.yours):
                write(12 + offset, column + 2, card_label(card))
        write(15, 1, "You", curses.A_BOLD)
        write(17, 1, "Your hand:")
        for index, card in enumerate(view.hand):
            attrs = curses.A_REVERSE if index == card_index else 0
            write(18, 2 + index * 10, f"{index + 1}:{card_label(card)}", attrs)
        write(20, 1, "R red | B blue | G green | Y yellow | P purple | BROWN=Br")
        if session.legal is not None:
            claims = ", ".join(str(index + 1) for index in session.legal.claimable) or "none"
            write(16, 1, f"Claimable: {claims} | + legal target | Claims included on Enter")
    write(height - 2, 1, message)
    write(height - 1, 1, "Up/down: card | Left/right: stone | Enter: play/pass | R: refresh | Q: quit")
    screen.refresh()


def network_worker(client, commands, results):
    try:
        while True:
            command = commands.get()
            if command is None:
                return
            action, previous, card_index, stone_index = command
            session = TerminalSession(client)
            session.view, session.legal = previous.view, previous.legal
            try:
                if action == "submit":
                    session.submit(card_index, stone_index)
                else:
                    session.refresh()
                results.put((session, "Move submitted." if action == "submit" else "", False))
            except (ApiError, httpx.HTTPError, ValueError) as exc:
                detail = str(exc)
                if action == "submit" and isinstance(exc, httpx.HTTPError):
                    detail = "Connection lost; move may have succeeded. Refreshing before another move."
                session.legal = None
                results.put((session, detail, True))
    finally:
        client.http.close()


def run_screen(screen, client):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.keypad(True)
    screen.timeout(100)
    commands, results = queue.Queue(), queue.Queue()
    worker = threading.Thread(target=network_worker, args=(client, commands, results), daemon=True)
    worker.start()
    session = TerminalSession(client)
    card_index = stone_index = 0
    busy = False
    active_action = None
    pending_submission = None
    message = "Loading game..."
    refresh_at = 0
    try:
        while True:
            try:
                session, notice, failed = results.get_nowait()
                busy = False
                if notice:
                    message = notice
                elif message == "Loading game...":
                    message = "Ready. Select a card and a stone, then press Enter."
                card_index = min(card_index, max(0, len(session.view.hand) - 1)) if session.view else 0
                refresh_at = time.monotonic() + (3 if failed else 1)
                if not failed and session.view and session.view.phase == "your_turn" and session.legal is None:
                    refresh_at = 0
                if pending_submission is not None:
                    version, selected_card, selected_stone = pending_submission
                    pending_submission = None
                    if not failed and session.legal is not None and session.view.version == version:
                        commands.put(("submit", session, selected_card, selected_stone))
                        busy = True
                        active_action = "submit"
                        message = "Submitting..."
                    elif not failed:
                        message = "The position changed. Review it and press Enter again."
            except queue.Empty:
                pass
            if not busy and time.monotonic() >= refresh_at:
                if session.view is None or session.view.phase != "game_over":
                    commands.put(("refresh", session, 0, 0))
                    busy = True
                    active_action = "refresh"
            draw_screen(screen, session, card_index, stone_index, message + (" [loading]" if busy else ""))
            key = screen.getch()
            if key in (ord("q"), ord("Q"), 27):
                return
            if key == curses.KEY_LEFT:
                stone_index = (stone_index - 1) % 9
            elif key == curses.KEY_RIGHT:
                stone_index = (stone_index + 1) % 9
            elif key in (curses.KEY_UP, curses.KEY_DOWN) and session.view and session.view.hand:
                card_index = (card_index + (1 if key == curses.KEY_DOWN else -1)) % len(session.view.hand)
            elif key in (ord("r"), ord("R")):
                refresh_at = 0
            elif key in (10, 13, curses.KEY_ENTER) and not (busy and active_action == "submit"):
                height, width = screen.getmaxyx()
                if height < 24 or width < 80:
                    continue
                if session.legal is None:
                    message = "Wait for your turn and the legal moves to load."
                elif session.legal.plays and stone_index not in session.targets(card_index):
                    message = "Choose a stone marked + for this card."
                elif busy:
                    pending_submission = (session.view.version, card_index, stone_index)
                    message = "Move queued until refresh completes."
                else:
                    commands.put(("submit", session, card_index, stone_index))
                    busy = True
                    active_action = "submit"
                    message = "Submitting..."
    finally:
        commands.put(None)


def parser():
    arguments = argparse.ArgumentParser(description=__doc__)
    subcommands = arguments.add_subparsers(dest="command", required=True)
    create = subcommands.add_parser("create", help="create a game and wait for an opponent")
    create.add_argument("--seed", type=int)
    create.add_argument("--auto-claim", action="store_true")
    create.add_argument("--turn-timeout", type=int, metavar="SECONDS")
    join = subcommands.add_parser("join", help="join an existing game")
    join.add_argument("game_id")
    for command in (create, join):
        command.add_argument("--url", default="http://localhost:8000", help="server base URL")
    return arguments


def main(argv=None):
    args = parser().parse_args(argv)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("This client needs an interactive terminal (at least 80 x 24).", file=sys.stderr)
        return 1
    client = SchottenClient(args.url)
    try:
        if args.command == "create":
            client.create_game(args.auto_claim, args.turn_timeout, args.seed)
        else:
            client.join(args.game_id)
    except (ApiError, httpx.HTTPError) as exc:
        client.http.close()
        print(str(exc), file=sys.stderr)
        return 1
    try:
        curses.wrapper(run_screen, client)
    except KeyboardInterrupt:
        pass
    except curses.error as exc:
        print(f"Cannot initialize terminal: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
