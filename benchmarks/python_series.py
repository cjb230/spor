"""Direct Python engine/bot baseline, without HTTP; retains reference views and logs."""
import argparse
import random
from datetime import datetime, timezone
from time import perf_counter

from schotten.bot import Bot
from schotten.game import Game
from schotten.schemas import CreateGameRequest, TurnRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bot-seed", type=int, default=1)
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be at least 1")
    counts = [0, 0, 0]
    started = perf_counter()
    now = datetime.now(timezone.utc)
    for index in range(args.games):
        order = [0, 1] if index % 2 == 0 else [1, 0]
        game = Game("benchmark", CreateGameRequest(seed=args.seed + index), now)
        game.join(now)
        bots = [
            Bot(None, ["greedy", "random"][entrant], random.Random(args.bot_seed + index * 2 + entrant))
            for entrant in order
        ]
        while not game.over:
            seat = game.turn
            play, claims = bots[seat].choose(game.view(seat))
            game.play_turn(seat, TurnRequest(version=game.version, play=play, claims=claims), now)
        counts[2 if game.winner is None else order[game.winner]] += 1
    elapsed = perf_counter() - started
    for strategy, count in zip(["greedy", "random"], counts):
        print(f"{strategy}: {count} wins ({count / args.games * 100:.1f}%)")
    print(f"Draws: {counts[2]}")
    print(f"Elapsed: {elapsed:.6f}s")
    print(f"Throughput: {args.games / elapsed:.2f} games/s")


if __name__ == "__main__":
    main()
