"""A bot that plays Schotten Totten through the REST API only.

    python -m schotten.bot selfplay                 # two bots, one game, against a local server
    python -m schotten.bot create --strategy greedy # open a game, print its id, wait for a joiner
    python -m schotten.bot join g_81f3              # take the second seat

It uses `legal-moves`-style reasoning but works from the player view alone, so it is
also a worked example of what a client needs: claims are worked out locally from
`unseen_cards`, and the server stays the judge (a rejected claim just means "play on").
"""
import argparse
import random
import sys
import threading
import uuid
from time import perf_counter
from typing import Optional

from .cards import Card
from .client import ApiError, SchottenClient
from .rules import claim_is_provable, evaluate
from .schemas import GameView, Play

STRATEGIES = ("greedy", "random")


def card_str(c: Card) -> str:
    return f"{c.color.value}{c.value}"


def partial_score(cards: list) -> float:
    """How promising a stone with one or two cards is; completed stones score by strength."""
    if len(cards) == 3:
        formation, total = evaluate(cards)
        return formation * 100 + total
    if len(cards) == 1:
        return cards[0].value / 2
    a, b = cards
    gap = abs(a.value - b.value)
    same_color = a.color == b.color
    if same_color and gap == 1:
        return 60 + (a.value + b.value) / 4
    if gap == 0:
        return 50 + a.value
    if same_color and gap == 2:
        return 45
    if same_color:
        return 35 + (a.value + b.value) / 4
    if gap == 1:
        return 30 + (a.value + b.value) / 4
    return (a.value + b.value) / 2


class Bot:
    def __init__(
        self,
        client: SchottenClient,
        strategy: str = "greedy",
        rng: Optional[random.Random] = None,
        verbose: bool = False,
        name: Optional[str] = None,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
        self.client = client
        self.strategy = strategy
        self.rng = rng or random.Random()
        self.verbose = verbose
        self.name = name or strategy

    # ---- decision making -----------------------------------------------

    @staticmethod
    def options(view: GameView) -> list:
        return [
            Play(card=card, stone=s.index)
            for card in view.hand
            for s in view.stones
            if s.claimed_by is None and len(s.yours) < 3
        ]

    @staticmethod
    def provable_stones(view: GameView, play: Optional[Play]) -> list:
        """Stones I could claim after making `play`, proven against every unseen card."""
        out = []
        for s in view.stones:
            mine, first = list(s.yours), s.first_to_complete
            if play is not None and play.stone == s.index:
                mine.append(play.card)
                if len(mine) == 3 and first is None:
                    first = "you"
            if (
                s.claimed_by is None
                and len(mine) == 3
                and claim_is_provable(mine, s.theirs, view.unseen_cards, first == "you")
            ):
                out.append(s.index)
        return out

    def score(self, view: GameView, play: Play) -> float:
        stone = view.stones[play.stone]
        mine = [*stone.yours, play.card]
        score = partial_score(mine)
        if len(mine) == 3 and play.stone in self.provable_stones(view, play):
            score += 1000  # a stone I can take right now
        elif len(stone.theirs) == 3 and len(mine) == 3:
            score += 5 if evaluate(mine) > evaluate(stone.theirs) else -20
        return score

    def choose(self, view: GameView) -> tuple:
        """Return (play or None, claims)."""
        options = self.options(view)
        if not options:
            return None, self.provable_stones(view, None)
        if self.strategy == "random":
            play = self.rng.choice(options)
        else:
            scored = [(self.score(view, p), p) for p in options]
            best = max(s for s, _ in scored)
            play = self.rng.choice([p for s, p in scored if s == best])
        return play, self.provable_stones(view, play)

    # ---- acting --------------------------------------------------------

    def take_turn(self, view: GameView) -> GameView:
        """Play one turn from `view` and return the resulting view."""
        play, claims = self.choose(view)
        try:
            new = self.client.turn(view.version, play, claims, turn_id=uuid.uuid4())
        except ApiError as e:
            if e.code != "CLAIM_NOT_PROVABLE" or not claims:
                raise
            self._say(f"claim rejected ({e.detail}); playing without claims")
            claims = []
            new = self.client.turn(view.version, play, claims, turn_id=uuid.uuid4())
        what = f"plays {card_str(play.card)} on stone {play.stone}" if play else "passes"
        self._say(f"{what}" + (f", claims {claims}" if claims else ""))
        return new

    def run(self, poll_seconds: int = 30) -> GameView:
        """Play until the game ends, long-polling while it is the opponent's turn."""
        view = self.client.view()
        while view.phase != "game_over":
            if view.phase == "your_turn":
                view = self.take_turn(view)
            else:
                view = self.client.view(since=view.version, timeout=poll_seconds) or self.client.view()
        self._say(self.result(view))
        return view

    @staticmethod
    def result(view: GameView) -> str:
        if view.winner is None:
            return f"game over: draw ({view.win_reason})"
        return f"game over: {'I won' if view.winner == 'you' else 'I lost'} ({view.win_reason})"

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.name}] {msg}", flush=True)


# ---- command line -------------------------------------------------------


def _make_bot(url: str, strategy: str, seed: Optional[int], name: str) -> Bot:
    return Bot(SchottenClient(url), strategy, random.Random(seed), verbose=True, name=name)


def run_series(args) -> int:
    started = perf_counter()
    strategies = [args.strategy, args.opponent_strategy]
    labels = strategies if strategies[0] != strategies[1] else [f"{strategies[0]} (A)", f"{strategies[1]} (B)"]
    wins = [0, 0]
    draws = completed = 0
    failed = False
    for index in range(args.games):
        order = [0, 1] if index % 2 == 0 else [1, 0]
        bots = []
        try:
            for seat, entrant in enumerate(order):
                seed = None if args.bot_seed is None else args.bot_seed + index * 2 + entrant
                bot = _make_bot(args.url, strategies[entrant], seed, f"seat{seat}")
                bot.verbose = False
                bots.append(bot)
            deck_seed = None if args.seed is None else args.seed + index
            grant = bots[0].client.create_game(args.auto_claim, args.turn_timeout, deck_seed)
            bots[1].client.join(grant.game_id)
            view = bots[0].client.view()
            while view.phase != "game_over":
                bot = bots[view.turn]
                view = bot.take_turn(bot.client.view())
            if view.winner is None:
                draws += 1
                winner = "draw"
            else:
                seat = view.you if view.winner == "you" else 1 - view.you
                entrant = order[seat]
                wins[entrant] += 1
                winner = labels[entrant]
            completed += 1
            if not args.quiet:
                print(f"Game {completed}/{args.games}: winner={winner} ({view.win_reason})", flush=True)
        except Exception as exc:
            print(f"Game {index + 1} failed: {exc}", file=sys.stderr)
            failed = True
            break
        finally:
            for bot in bots:
                bot.client.http.close()
    elapsed = perf_counter() - started
    if not args.quiet:
        print(f"\nCompleted: {completed}/{args.games}")
    for label, count in zip(labels, wins):
        percent = count / completed * 100 if completed else 0
        print(f"{label}: {count} wins ({percent:.1f}%)")
    print(f"Draws: {draws}")
    if completed and not args.quiet:
        leader = labels[0] if wins[0] > wins[1] else labels[1]
        print("Overall: tied" if wins[0] == wins[1] else f"Overall winner: {leader}")
    print(f"Elapsed: {elapsed:.2f}s")
    return 1 if failed else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m schotten.bot", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared options live on each subcommand, so they go after it: `selfplay --url ...`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default="http://localhost:8000", help="server base URL")
    common.add_argument("--strategy", choices=STRATEGIES, default="greedy")
    common.add_argument("--bot-seed", type=int, help="seed for the bot's own choices")

    def game_options(sp):
        sp.add_argument("--auto-claim", action="store_true")
        sp.add_argument("--turn-timeout", type=int, metavar="SECONDS")
        sp.add_argument("--seed", type=int, help="deck shuffle seed")

    game_options(sub.add_parser("create", parents=[common], help="create a game and wait for an opponent"))
    sub.add_parser("join", parents=[common], help="join a game").add_argument("game_id")
    sp = sub.add_parser("selfplay", parents=[common], help="run two bots against each other")
    game_options(sp)
    sp.add_argument("--opponent-strategy", choices=STRATEGIES, default="random")
    series = sub.add_parser("series", parents=[common], help="compare strategies over multiple games")
    game_options(series)
    series.add_argument("--games", type=int, default=100)
    series.add_argument("--quiet", "-q", action="store_true", help="print only final tallies and elapsed time")
    series.add_argument("--opponent-strategy", choices=STRATEGIES, default="random")
    args = p.parse_args(argv)

    if args.cmd == "series":
        if args.games < 1:
            p.error("--games must be at least 1")
        return run_series(args)

    if args.cmd == "join":
        bot = _make_bot(args.url, args.strategy, args.bot_seed, "bot")
        bot.client.join(args.game_id)
        bot.run()
        return 0

    host = _make_bot(args.url, args.strategy, args.bot_seed, "seat0")
    grant = host.client.create_game(args.auto_claim, args.turn_timeout, args.seed)
    print(f"game {grant.game_id} created; waiting for an opponent", flush=True)
    if args.cmd == "create":
        host.run()
        return 0

    guest = _make_bot(args.url, args.opponent_strategy, args.bot_seed, "seat1")
    guest.client.join(grant.game_id)
    results: dict = {}
    errors: list = []

    def go(label, bot):
        try:
            results[label] = bot.run()
        except Exception as e:  # surface thread failures instead of hanging the join
            errors.append(f"{label}: {e!r}")

    threads = [threading.Thread(target=go, args=(b.name, b)) for b in (host, guest)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    log = host.client.log()
    print(f"finished after {len(log.entries)} turns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
